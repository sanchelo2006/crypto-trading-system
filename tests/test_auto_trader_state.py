"""
Unit-тесты для персистентности позиций и устойчивости цикла автоторговли.

Покрывают правки:
  - сохранение/загрузка состояния позиций (positions/*.json)
  - восстановление позиций при рестарте (не помечаются как orphan)
  - orphan-позиции (add-only sync с биржей)
  - детект закрытий после сетевого сбоя
  - устойчивость _run_loop к ConnectionError
  - _compute_tp_sl для всех веток (ATR / проценты / оба)
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


import config
from trading.auto_trader import (
    AutoTraderConfig,
    AutoTraderSymbol,
    NETWORK_RETRY_LIMIT,
)
from trading.position_manager import PositionManager




# ---------------------------------------------------------------- #
# Хелперы
# ---------------------------------------------------------------- #

class FakeExecutor:
    """Минимальная замена BybitExecutor для unit-тестов."""

    def __init__(self):
        self.positions = []          # список dict: {"side", "size", "avgPrice"}
        self.orders = {}             # order_id -> order_info
        self.closed_pnl = []         # список записей
        self.instrument = {"tickSize": "0.0001", "qtyStep": "0.1"}
        self.ticker_price = 1.6

    def get_position_info(self, symbol):
        return True, list(self.positions)

    def get_order_info(self, symbol, order_id):
        if order_id in self.orders:
            return True, self.orders[order_id]
        return False, "not_found"

    def get_closed_pnl(self, symbol, limit=20):
        return True, list(self.closed_pnl)

    def get_instrument_info(self, symbol):
        return True, dict(self.instrument)

    def round_price(self, price, tick):
        t = float(tick)
        return round(round(price / t) * t, 10)

    def get_ticker_price(self, symbol):
        return True, self.ticker_price

    def place_market_order(self, symbol, side, qty, reduce_only=False):
        return True, {"orderId": "fake-order-1"}

    def set_trading_stop(self, symbol, stop_loss, take_profit):
        return True, None

    def set_leverage(self, symbol, leverage):
        return True, None

    def calculate_position_size(self, symbol, capital_usdt, leverage, entry_price):
        return True, 5.4


class FakeMemory:
    """Заглушка memory.get(symbol).get_state()."""

    def __init__(self):
        self._state = {"strategy_spec": {"risk_controls": {"enabled": False}}}

    def get(self, symbol):
        return self

    def get_state(self):
        return self._state


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    """Подменяет config.DATA_DIR на временную папку."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    return tmp_path / "auto_trading" / "positions"


@pytest.fixture
def trader(state_dir):
    """Готовый AutoTraderSymbol с фейковым executor."""
    executor = FakeExecutor()
    cfg = AutoTraderConfig(symbol="XRPUSDT")
    t = AutoTraderSymbol(
        cfg=cfg,
        executor=executor,
        fetcher=MagicMock(),
        memory=FakeMemory(),
        backtest_engine=MagicMock(),
        backtest_optimizer=MagicMock(),
        ai_client=MagicMock(),
        notify_fn=MagicMock(),
        total_capital_usdt=1000.0,
    )
    # Для _save_state / _load_state нужно, чтобы Path был валидный.
    assert t._state_path.parent == state_dir
    return t


def _make_position(direction="long", entry=1.6, qty=5.4,
                   sl=1.56, tp=1.72, entry_time=None):
    return {
        "position_id": "XRPUSDT_1700000000",
        "direction": direction,
        "entry_price": entry,
        "qty": qty,
        "stop_loss": sl,
        "take_profit": tp,
        "entry_time": entry_time or datetime.now(timezone.utc) - timedelta(minutes=10),
        "entry_reasons": [],
        "entry_snapshot": {},
        "management": {},
        "bybit_order_id": "order-abc",
        "tpsl_error": None,
    }


# ---------------------------------------------------------------- #
# _compute_tp_sl — все ветки
# ---------------------------------------------------------------- #

class TestComputeTpSl:
    def test_both_atr_mults(self, trader):
        pm = trader.position_manager
        spec = {"stop_loss_atr_mult": 1.5, "take_profit_atr_mult": 3.0}
        indicators = {"atr_14": 0.01}
        sl, tp = pm._compute_tp_sl("long", 1.6, spec, indicators)
        assert sl < 1.6 < tp
        assert isinstance(sl, float)
        assert isinstance(tp, float)

    def test_atr_sl_only_tp_from_percent(self, trader):
        """Раньше это был источник TypeError: TP не из ATR, но всё равно падало."""
        pm = trader.position_manager
        spec = {
            "stop_loss_atr_mult": 1.5,
            "take_profit_pct": [5.0],
        }
        indicators = {"atr_14": 0.01}
        sl, tp = pm._compute_tp_sl("long", 1.6, spec, indicators)
        assert sl < 1.6 < tp

    def test_atr_tp_only_sl_from_percent(self, trader):
        pm = trader.position_manager
        spec = {
            "stop_loss_pct": 2.0,
            "take_profit_atr_mult": 3.0,
        }
        indicators = {"atr_14": 0.01}
        sl, tp = pm._compute_tp_sl("long", 1.6, spec, indicators)
        assert sl < 1.6 < tp

    def test_no_atr_percent_fallback(self, trader):
        pm = trader.position_manager
        spec = {"stop_loss_pct": 2.0, "take_profit_pct": [5.0]}
        indicators = {}
        sl, tp = pm._compute_tp_sl("long", 1.6, spec, indicators)
        assert sl < 1.6 < tp

    def test_atr_nan_falls_back_to_percent(self, trader):
        import math
        pm = trader.position_manager
        spec = {
            "stop_loss_atr_mult": 1.5,
            "take_profit_atr_mult": 3.0,
            "stop_loss_pct": 2.0,
            "take_profit_pct": [5.0],
        }
        indicators = {"atr_14": math.nan}
        sl, tp = pm._compute_tp_sl("long", 1.6, spec, indicators)
        assert sl < 1.6 < tp

    def test_huge_atr_is_clamped(self, trader):
        """ATR выброс 1.0 → SL/TP не должны улетать за клампы 2.5%/10%."""
        pm = trader.position_manager
        spec = {"stop_loss_atr_mult": 50, "take_profit_atr_mult": 50}
        indicators = {"atr_14": 1.0}   # абсурдно большой ATR
        sl, tp = pm._compute_tp_sl("long", 1.6, spec, indicators)
        # SL не ниже -2.5%
        assert sl >= 1.6 * (1 - 0.025) - 1e-6
        # TP не выше +10%
        assert tp <= 1.6 * (1 + 0.10) + 1e-6

    def test_short_direction(self, trader):
        pm = trader.position_manager
        spec = {"stop_loss_pct": 2.0, "take_profit_pct": [5.0]}
        sl, tp = pm._compute_tp_sl("short", 1.6, spec, {})
        assert sl > 1.6 > tp


# ---------------------------------------------------------------- #
# Персистентность
# ---------------------------------------------------------------- #

class TestPersistence:
    def test_save_writes_json(self, trader, state_dir):
        trader.position_manager.positions = [_make_position()]
        trader._save_state()

        assert trader._state_path.exists()
        with open(trader._state_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["symbol"] == "XRPUSDT"
        assert len(data["positions"]) == 1
        assert data["positions"][0]["direction"] == "long"

    def test_save_is_atomic_no_tmp_left(self, trader):
        trader.position_manager.positions = [_make_position()]
        trader._save_state()
        tmp = trader._state_path.with_name(trader._state_path.name + ".tmp")
        assert not tmp.exists()

    def test_load_restores_positions(self, trader):
        pos = _make_position()
        trader.position_manager.positions = [pos]
        trader._consumed_closed_pnl_ids = {"order-x", "order-y"}
        trader._save_state()

        # Симулируем рестарт: новый трейдер с теми же путями.
        trader2 = AutoTraderSymbol(
            cfg=AutoTraderConfig(symbol="XRPUSDT"),
            executor=FakeExecutor(),
            fetcher=MagicMock(),
            memory=FakeMemory(),
            backtest_engine=MagicMock(),
            backtest_optimizer=MagicMock(),
            ai_client=MagicMock(),
            notify_fn=MagicMock(),
            total_capital_usdt=1000.0,
        )

        assert len(trader2.position_manager.positions) == 1
        assert trader2.position_manager.positions[0]["direction"] == "long"
        assert trader2._consumed_closed_pnl_ids == {"order-x", "order-y"}

    def test_load_skips_corrupt_entries(self, trader, state_dir):
        # Кривой JSON: одна запись валидная, вторая — без обязательных полей.
        state_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "symbol": "XRPUSDT",
            "positions": [
                trader._serialize_position(_make_position()),
                {"direction": "short"},  # мусор
            ],
            "consumed_closed_pnl_ids": [],
        }
        with open(trader._state_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        trader._load_state()
        assert len(trader.position_manager.positions) == 1

    def test_load_no_file_is_noop(self, trader):
        assert not trader._state_path.exists()
        # не должно бросить исключение
        trader._load_state()
        assert trader.position_manager.positions == []

    def test_save_load_roundtrip_datetime(self, trader):
        dt = datetime(2026, 9, 25, 13, 18, 47, tzinfo=timezone.utc)
        pos = _make_position(entry_time=dt)
        trader.position_manager.positions = [pos]
        trader._save_state()

        trader.position_manager.positions = []
        trader._load_state()
        restored = trader.position_manager.positions[0]
        assert isinstance(restored["entry_time"], datetime)
        assert restored["entry_time"].year == 2026


# ---------------------------------------------------------------- #
# sync_with_exchange
# ---------------------------------------------------------------- #

class TestSyncWithExchange:
    def test_adds_orphan_when_exchange_has_position(self, trader):
        trader.executor.positions = [
            {"side": "Buy", "size": "5.4", "avgPrice": "1.6"},
        ]
        ok, err = trader.position_manager.sync_with_exchange()
        assert ok
        assert len(trader.position_manager.positions) == 1
        orphan = trader.position_manager.positions[0]
        assert orphan["is_orphan"] is True
        assert orphan["direction"] == "long"

    def test_saved_position_is_not_orphan(self, trader):
        # Локальная позиция уже есть.
        trader.position_manager.positions = [_make_position(direction="long")]
        trader.executor.positions = [
            {"side": "Buy", "size": "5.4", "avgPrice": "1.6"},
        ]
        trader.position_manager.sync_with_exchange()
        # Должно остаться 1 (наша), без orphan.
        assert len(trader.position_manager.positions) == 1
        assert not trader.position_manager.positions[0].get("is_orphan")

    def test_empty_exchange_does_not_remove_local(self, trader):
        """Раньше sync чистил локальные — теперь НЕ должен (это делает detect)."""
        trader.position_manager.positions = [_make_position()]
        trader.executor.positions = []
        trader.position_manager.sync_with_exchange()
        assert len(trader.position_manager.positions) == 1

    def test_no_duplicate_orphan_on_repeat_calls(self, trader):
        trader.executor.positions = [
            {"side": "Buy", "size": "5.4", "avgPrice": "1.6"},
        ]
        trader.position_manager.sync_with_exchange()
        trader.position_manager.sync_with_exchange()
        trader.position_manager.sync_with_exchange()
        assert len(trader.position_manager.positions) == 1


# ---------------------------------------------------------------- #
# _detect_closed_positions
# ---------------------------------------------------------------- #

class TestDetectClosed:
    def test_orphan_closed_is_registered(self, trader):
        orphan = {
            "position_id": "XRPUSDT_orphan_1",
            "direction": "long",
            "entry_price": 1.6,
            "qty": 5.4,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "entry_time": datetime.now(timezone.utc) - timedelta(minutes=10),
            "entry_reasons": [],
            "entry_snapshot": {},
            "management": {},
            "is_orphan": True,
        }
        trader.position_manager.positions = [orphan]
        # Биржа пуста → закрытие.
        trader.executor.positions = []

        trader._detect_closed_positions()
        # Позиция удалена.
        assert trader.position_manager.positions == []

    def test_recent_position_skipped_by_grace(self, trader):
        fresh = _make_position(
            entry_time=datetime.now(timezone.utc) - timedelta(seconds=30),
        )
        trader.position_manager.positions = [fresh]
        trader.executor.positions = []   # якобы нет
        trader._detect_closed_positions()
        # Грейс-период 120с — свежая позиция не должна закрываться.
        assert len(trader.position_manager.positions) == 1


# ---------------------------------------------------------------- #
# _run_loop — устойчивость к сетевым сбоям
# ---------------------------------------------------------------- #

class TestRunLoop:
    def test_connection_error_does_not_stop_trader(self, trader):
        """ConnectionError не должен останавливать торговлю."""
        call_count = {"n": 0}
        # Запоминаем максимум счётчика во время работы цикла: после
        # успешной итерации он честно сбрасывается в 0, это правильное
        # поведение, но проверять надо то, что цикл НЕ ОСТАНОВИЛСЯ,
        # а не финальное значение счётчика.
        max_network_errors = {"v": 0}

        def fake_iteration():
            call_count["n"] += 1
            # Снимок счётчика после предыдущей итерации.
            if trader.consecutive_network_errors > max_network_errors["v"]:
                max_network_errors["v"] = trader.consecutive_network_errors

            if call_count["n"] <= 3:
                raise ConnectionError("network down")
            # На 4-й итерации — успех.
            trader.is_running = False   # выходим из цикла

        with patch.object(trader, "_loop_iteration", side_effect=fake_iteration), \
             patch("time.sleep"):
            trader.is_running = True
            trader._run_loop()

        # До успешной итерации счётчик успел вырасти до 3.
        assert max_network_errors["v"] >= 3
        # Цикл НЕ вызывал stop() — проверяем через отсутствие исключений
        # и через то, что мы дошли до 4-й итерации.
        assert call_count["n"] >= 4

    def test_network_counter_resets_on_success(self, trader):
        """
        После успешной итерации consecutive_network_errors обнуляется —
        это защищает от накопления счётчика через редкие сбои сети.
        """
        call_count = {"n": 0}

        def fake_iteration():
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ConnectionError("network down")
            if call_count["n"] == 2:
                # здесь счётчик должен быть 1
                assert trader.consecutive_network_errors == 1
            trader.is_running = False

        with patch.object(trader, "_loop_iteration", side_effect=fake_iteration), \
             patch("time.sleep"):
            trader.is_running = True
            trader._run_loop()

        # После успешной итерации — сброшено.
        assert trader.consecutive_network_errors == 0

    def test_non_network_errors_stop_after_limit(self, trader):
        """Обычные исключения останавливают трейдер после NETWORK_RETRY_LIMIT."""
        def fake_iteration():
            raise RuntimeError("bug in code")

        with patch.object(trader, "_loop_iteration", side_effect=fake_iteration), \
             patch("time.sleep"), \
             patch.object(trader, "stop") as mock_stop:
            trader.is_running = True
            trader._run_loop()
            mock_stop.assert_called_once()
            assert mock_stop.call_args.kwargs["reason"] == "repeated_errors"

    def test_connection_error_resets_non_network_counter(self, trader):
        """После сетевого сбоя _non_network_errors должен обнулиться."""
        calls = {"n": 0}

        def fake_iteration():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("code bug")
            if calls["n"] == 2:
                raise ConnectionError("net down")
            trader.is_running = False

        with patch.object(trader, "_loop_iteration", side_effect=fake_iteration), \
             patch("time.sleep"):
            trader.is_running = True
            trader._run_loop()

        assert trader._non_network_errors == 0


# ---------------------------------------------------------------- #
# Интеграция: открытие → сохранение → закрытие → сохранение
# ---------------------------------------------------------------- #

class TestIntegration:
    def test_open_persists_state(self, trader):
        trader.executor.orders["fake-order-1"] = {
            "orderId": "fake-order-1",
            "cumExecQty": "5.4",
            "avgPrice": "1.6",
        }
        spec = {"stop_loss_pct": 2.0, "take_profit_pct": [5.0]}
        ok, err = trader.position_manager.open_position(
            direction="long",
            qty=5.4,
            current_price=1.6,
            side_spec=spec,
            entry_reasons=[],
            indicators={},
        )
        assert ok
        # Эмулируем вызов _save_state, который делает auto_trader после успеха.
        trader._save_state()
        assert trader._state_path.exists()

    def test_close_saves_empty_state(self, trader):
        trader.position_manager.positions = [_make_position()]
        trader._save_state()
        trader.position_manager.positions = []
        trader._save_state()

        with open(trader._state_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["positions"] == []