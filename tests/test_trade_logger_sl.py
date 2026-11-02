"""
Тесты логирования SL/TP в trades и sl_updates.

Проверяют:
  - log_trade пишет новые поля (stop_loss, take_profit,
    initial_stop_loss, initial_take_profit);
  - log_sl_update пишет отдельный файл {SYMBOL}_sl_updates_YYYY_MM.csv
    с position_id и дельтой изменения.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

import trading.trade_logger as trade_logger_module
from trading.trade_logger import TradeLogger


# ---------------------------------------------------------------- #
# Фикстуры
# ---------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def isolated_logs(tmp_path, monkeypatch):
    """Изолирует папку логов для каждого теста."""
    monkeypatch.setattr(trade_logger_module, "LOGS_DIR", tmp_path / "logs")
    yield


# ---------------------------------------------------------------- #
# log_trade — расширенная схема
# ---------------------------------------------------------------- #

class TestLogTradeExtended:
    def test_writes_sl_tp_columns(self):
        logger = TradeLogger("XRPUSDT")
        logger.log_trade({
            "position_id": "XRPUSDT_test_1",
            "entry_time": datetime(2026, 9, 26, 18, 15, tzinfo=timezone.utc),
            "exit_time": datetime(2026, 9, 26, 18, 38, tzinfo=timezone.utc),
            "direction": "short",
            "entry_price": 1.5176,
            "exit_price": 1.5241,
            "qty": 5.1,
            "position_value_usdt": 7.74,
            "result_pct": -0.43,
            "pnl_usdt": -0.03,
            "exit_reason": "stop_loss",
            "stop_loss": 1.5241,
            "take_profit": 1.5059,
            "initial_stop_loss": 1.5338,
            "initial_take_profit": 1.5059,
            "entry_reasons": ["close < sma_50", "macd_hist < 0"],
        })

        path = logger._get_file_path("trades")
        assert path.exists()

        lines = path.read_text(encoding="utf-8-sig").splitlines()
        header = lines[0]
        row = lines[1]

        # Заголовок расширен.
        for col in (
            "stop_loss", "take_profit",
            "initial_stop_loss", "initial_take_profit",
        ):
            assert col in header

        # Значения на месте.
        assert "1.5241" in row
        assert "1.5059" in row
        assert "1.5338" in row

    def test_missing_sl_tp_columns_are_empty(self):
        """Старые trades без SL/TP пишутся пустыми значениями."""
        logger = TradeLogger("XRPUSDT")
        logger.log_trade({
            "position_id": "p_old",
            "direction": "long",
            "entry_price": 1.0,
            "exit_price": 1.1,
            "qty": 10,
            "result_pct": 10.0,
            "exit_reason": "take_profit",
        })

        path = logger._get_file_path("trades")
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        # Колонки в заголовке есть, но значения пустые.
        header = lines[0].split(",")
        row = lines[1].split(",")

        sl_idx = header.index("stop_loss")
        tp_idx = header.index("take_profit")
        init_sl_idx = header.index("initial_stop_loss")
        init_tp_idx = header.index("initial_take_profit")

        assert row[sl_idx] == ""
        assert row[tp_idx] == ""
        assert row[init_sl_idx] == ""
        assert row[init_tp_idx] == ""


# ---------------------------------------------------------------- #
# log_sl_update — отдельный файл
# ---------------------------------------------------------------- #

class TestLogSlUpdate:
    def test_creates_separate_file(self):
        logger = TradeLogger("XRPUSDT")
        logger.log_sl_update(
            position_id="XRPUSDT_test_1",
            direction="short",
            old_sl=1.5338,
            new_sl=1.5328,
            reason="trailing",
            extra={"atr_14": 0.005, "close": 1.5176},
        )

        path = logger._get_file_path("sl_updates")
        assert path.exists()
        assert path.name.startswith("XRPUSDT_sl_updates_")
        assert path.name.endswith(".csv")

    def test_row_contains_position_id_and_delta(self):
        logger = TradeLogger("XRPUSDT")
        logger.log_sl_update(
            position_id="XRPUSDT_test_1",
            direction="short",
            old_sl=1.5338,
            new_sl=1.5328,
            reason="trailing",
            extra={"atr_14": 0.005},
        )

        path = logger._get_file_path("sl_updates")
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        header = lines[0].split(",")
        row = lines[1].split(",")

        # position_id на месте.
        pid_idx = header.index("position_id")
        assert row[pid_idx] == "XRPUSDT_test_1"

        # delta посчитана правильно.
        delta_idx = header.index("delta")
        assert abs(float(row[delta_idx]) - (-0.001)) < 1e-8

        # reason на месте.
        reason_idx = header.index("reason")
        assert row[reason_idx] == "trailing"

        # extra — валидный JSON.
        extra_idx = header.index("extra")
        extra_raw = row[extra_idx]
        # Убираем обрамляющие кавычки CSV, если есть.
        if extra_raw.startswith('"') and extra_raw.endswith('"'):
            extra_raw = extra_raw[1:-1].replace('""', '"')
        parsed = json.loads(extra_raw)
        assert parsed["atr_14"] == 0.005

    def test_multiple_updates_append(self):
        """Несколько обновлений одного position_id пишутся построчно."""
        logger = TradeLogger("XRPUSDT")
        for i, (old, new) in enumerate([
            (1.5338, 1.5328),
            (1.5328, 1.5324),
            (1.5324, 1.5321),
        ]):
            logger.log_sl_update(
                position_id="XRPUSDT_test_1",
                direction="short",
                old_sl=old,
                new_sl=new,
                reason="trailing",
            )

        path = logger._get_file_path("sl_updates")
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        # Заголовок + 3 строки.
        assert len(lines) == 4
        # Все три строки содержат один и тот же position_id.
        for line in lines[1:]:
            assert "XRPUSDT_test_1" in line

    def test_breakeven_and_trailing_reasons(self):
        """reason корректно передаётся (breakeven, trailing, breakeven+trailing)."""
        logger = TradeLogger("XRPUSDT")
        logger.log_sl_update(
            position_id="p1",
            direction="long",
            old_sl=99.0,
            new_sl=100.1,
            reason="breakeven+trailing",
        )
        logger.log_sl_update(
            position_id="p2",
            direction="long",
            old_sl=99.0,
            new_sl=100.0,
            reason="breakeven",
        )

        path = logger._get_file_path("sl_updates")
        content = path.read_text(encoding="utf-8-sig")
        assert "breakeven+trailing" in content
        assert "breakeven" in content

    def test_empty_extra_is_valid_json(self):
        """Без extra пишется пустой JSON-объект {}."""
        logger = TradeLogger("XRPUSDT")
        logger.log_sl_update(
            position_id="p1",
            direction="long",
            old_sl=100.0,
            new_sl=101.0,
            reason="trailing",
        )

        path = logger._get_file_path("sl_updates")
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        header = lines[0].split(",")
        row = lines[1].split(",")
        extra_idx = header.index("extra")
        extra_raw = row[extra_idx]
        if extra_raw.startswith('"') and extra_raw.endswith('"'):
            extra_raw = extra_raw[1:-1].replace('""', '"')
        assert json.loads(extra_raw) == {}

# ---------------------------------------------------------------- #
# log_event — HTML-лог
# ---------------------------------------------------------------- #

class TestLogEventHtml:
    def test_creates_html_file_on_first_event(self):
        logger = TradeLogger("XRPUSDT")
        logger.log_event("START", "Автоторговля запущена", "")

        path = logger._get_html_path()
        assert path.exists()
        assert path.name.startswith("XRPUSDT_events_")
        assert path.name.endswith(".html")
        # Недельный суффикс в имени файла.
        import re
        assert re.search(r"_events_\d{4}_W\d{2}\.html$", path.name), (
            f"Ожидался недельный суффикс, получено: {path.name}"
        )

    def test_event_has_data_date_attribute(self):
        """
        Каждое событие получает data-date для клиентской фильтрации.
        Формат — YYYY-MM-DD (первые 10 символов timestamp).
        """
        logger = TradeLogger("XRPUSDT")
        logger._append_html_event(
            "2026-09-27 18:00:00", "START", "запуск", "",
        )
        logger._append_html_event(
            "2026-09-28 10:00:00", "MGMT", "SL", "",
        )

        content = logger._get_html_path().read_text(encoding="utf-8")
        assert 'data-date="2026-09-27"' in content
        assert 'data-date="2026-09-28"' in content

    def test_html_has_doctype_and_styles(self):
        logger = TradeLogger("XRPUSDT")
        logger.log_event("START", "запуск", "")

        content = logger._get_html_path().read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "<style>" in content
        assert "ev-info" in content
        # CSS использует одинарные скобки (шаблон через .replace,
        # а не .format). Двойные — признак старой сломанной версии.
        assert "{{" not in content
        assert "}}" not in content
        # Разделитель времени присутствует.
        assert "time-separator" in content
        # Панель поиска и скрипт присутствуют.
        assert "search-bar" in content
        assert 'id="date-filter"' in content
        assert 'id="event-filter"' in content
        assert "applyFilters" in content
        assert "collectEventTypes" in content
        # Каждое событие имеет атрибут data-date.
        assert 'data-date="' in content

    def test_html_has_event_filter(self):
        """
        HTML-шаблон содержит выпадающий список для фильтрации
        по типу события и скрипт его наполнения.
        """
        logger = TradeLogger("XRPUSDT")
        logger.log_event("START", "запуск", "")

        content = logger._get_html_path().read_text(encoding="utf-8")

        # Есть select с id="event-filter" и placeholder.
        assert 'id="event-filter"' in content
        assert "— все —" in content
        # Функция сбора типов из DOM.
        assert "collectEventTypes" in content
        # Обработчик на DOMContentLoaded.
        assert "DOMContentLoaded" in content
        # Общая функция фильтрации, учитывающая оба фильтра.
        assert "applyFilters" in content

    def test_html_groups_by_group_key_not_timestamp(self):
        """
        Если передан group, разделитель ставится при смене группы,
        а не при смене секунды. Это склеивает FILTERS + LONG + SHORT
        одной свечи в один визуальный блок, даже если записаны
        в разные секунды.
        """
        logger = TradeLogger("XRPUSDT")

        # Одна свеча, три события в разные секунды — один разделитель.
        logger._append_html_event(
            "2026-09-27 18:00:00", "SIGNAL_CHECK", "FILTERS", "",
            group="candle-1800",
        )
        logger._append_html_event(
            "2026-09-27 18:00:01", "SIGNAL_CHECK", "LONG", "",
            group="candle-1800",
        )
        logger._append_html_event(
            "2026-09-27 18:00:02", "SIGNAL_CHECK", "SHORT", "",
            group="candle-1800",
        )
        # Новая свеча — новый разделитель.
        logger._append_html_event(
            "2026-09-27 18:15:00", "SIGNAL_CHECK", "FILTERS", "",
            group="candle-1815",
        )

        content = logger._get_html_path().read_text(encoding="utf-8")
        assert content.count('<div class="time-separator"') == 2

    def test_html_inserts_separator_on_timestamp_change(self):
        """Разные timestamps дают разные разделители, одинаковые — один."""
        logger = TradeLogger("XRPUSDT")

        # Один и тот же timestamp → один разделитель на все события.
        logger._append_html_event(
            "2026-09-27 18:00:00", "MGMT", "SL #1", ""
        )
        logger._append_html_event(
            "2026-09-27 18:00:00", "MGMT", "SL #2", ""
        )
        # Другой timestamp → ещё один разделитель.
        logger._append_html_event(
            "2026-09-27 18:15:00", "MGMT", "SL #3", ""
        )

        content = logger._get_html_path().read_text(encoding="utf-8")
        # Ищем открывающий тег разделителя без закрывающей ">":
        # теперь он содержит data-date="...", поэтому точное
        # совпадение с '<div class="time-separator">' не работает.
        assert content.count('<div class="time-separator"') == 2
        assert "2026-09-27 18:00:00" in content
        assert "2026-09-27 18:15:00" in content

    def test_classifies_events(self):
        logger = TradeLogger("XRPUSDT")
        logger.log_event("OPEN", "позиция открыта", "")
        logger.log_event("CLOSE_WIN", "PnL=+0.62%", "")
        logger.log_event("CLOSE_LOSS", "PnL=-0.43%", "")
        logger.log_event("ERROR", "что-то упало", "")
        logger.log_event("SIGNAL_CHECK", "фильтры", "")
        logger.log_event("CONFIG_REJECT", "нельзя", "")

        content = logger._get_html_path().read_text(encoding="utf-8")
        assert "ev-success" in content
        assert "ev-danger" in content
        assert "ev-signal" in content
        assert "ev-warning" in content

    def test_html_escapes_special_chars(self):
        logger = TradeLogger("XRPUSDT")
        logger.log_event(
            "SIGNAL_CHECK",
            "close < sma_50 & rsi > 60",
            "details: <html> & 'test'",
        )
        content = logger._get_html_path().read_text(encoding="utf-8")
        # Эскейпинг.
        assert "&lt;" in content or "&gt;" in content or "&amp;" in content

    def test_html_and_csv_written_together(self):
        """log_event пишет и в CSV, и в HTML."""
        logger = TradeLogger("XRPUSDT")
        logger.log_event("START", "запуск", "")

        assert logger._get_file_path("events").exists()
        assert logger._get_html_path().exists()

    def test_html_appends_multiple_events(self):
        logger = TradeLogger("XRPUSDT")
        for i in range(3):
            logger.log_event("MGMT", f"SL обновлён #{i}", "")

        content = logger._get_html_path().read_text(encoding="utf-8")
        # Три блока MGMT. Ищем именно открывающий тег события,
        # а не подстроку "ev-management" — она также встречается
        # в CSS-селекторах (.ev-management и .ev-management .type).
        assert content.count('class="event ev-management"') == 3
        # Шапка только одна.
        assert content.count("<!DOCTYPE html>") == 1