"""
Тесты авто-санитизации strategy_spec перед запуском автоторговли.

Проверяют метод AutoTraderSymbol._sanitize_strategy_spec:
  - max_hold_candles добавляется ВСЕГДА, если не задан;
  - trailing_atr_mult добавляется ТОЛЬКО если в management нет
    ни одного другого механизма сопровождения (breakeven / exit_on_*);
  - существующие значения не перезаписываются;
  - метод идемпотентен (повторный вызов ничего не меняет).
"""

from unittest.mock import MagicMock

import pytest

from trading.auto_trader import (
    AutoTraderConfig,
    AutoTraderSymbol,
    DEFAULT_TRAILING_ATR_MULT,
    DEFAULT_MAX_HOLD_CANDLES,
)


# ---------------------------------------------------------------- #
# Хелпер
# ---------------------------------------------------------------- #

def _make_trader_with_spec(spec: dict):
    """Создаёт трейдер с замоканным memory, отдающим переданный spec."""
    memory = MagicMock()
    cm = MagicMock()
    cm.get_state.return_value = {"strategy_spec": spec}
    memory.get.return_value = cm

    trader = AutoTraderSymbol(
        AutoTraderConfig("XRPUSDT"),
        executor=MagicMock(),
        fetcher=MagicMock(),
        memory=memory,
        backtest_engine=MagicMock(),
        backtest_optimizer=MagicMock(),
        ai_client=MagicMock(),
        notify_fn=MagicMock(),
        total_capital_usdt=1000.0,
    )
    return trader, cm


# ---------------------------------------------------------------- #
# Тесты
# ---------------------------------------------------------------- #

def test_sanitize_adds_trailing_when_no_management():
    """Пустой management → добавляет и trailing, и max_hold."""
    spec = {
        "long":  {"entry": [], "management": {}},
        "short": {"entry": [], "management": {}},
    }
    trader, cm = _make_trader_with_spec(spec)

    changed = trader._sanitize_strategy_spec()

    assert changed is True
    for side in ("long", "short"):
        mgmt = spec[side]["management"]
        assert mgmt["trailing_atr_mult"] == DEFAULT_TRAILING_ATR_MULT
        assert mgmt["max_hold_candles"] == DEFAULT_MAX_HOLD_CANDLES

    cm.update_state.assert_called_once()
    cm.save.assert_called_once()


def test_sanitize_does_not_overwrite_existing_trailing():
    """Существующий trailing_atr_mult не перезаписывается."""
    spec = {
        "long": {
            "entry": [],
            "management": {
                "trailing_atr_mult": 3.5,
                "max_hold_candles": 48,
            },
        },
        "short": {"entry": [], "management": {}},
    }
    trader, cm = _make_trader_with_spec(spec)

    trader._sanitize_strategy_spec()

    # Long не тронут — оба поля уже заданы.
    assert spec["long"]["management"]["trailing_atr_mult"] == 3.5
    assert spec["long"]["management"]["max_hold_candles"] == 48
    # Short получил оба дефолта.
    assert (
        spec["short"]["management"]["trailing_atr_mult"]
        == DEFAULT_TRAILING_ATR_MULT
    )
    assert (
        spec["short"]["management"]["max_hold_candles"]
        == DEFAULT_MAX_HOLD_CANDLES
    )


def test_sanitize_does_not_add_trailing_if_breakeven_present():
    """
    Если модель осознанно выбрала move_to_breakeven_at_tp_pct —
    trailing НЕ добавляется. Это случай HYPEUSDT.
    """
    spec = {
        "long": {
            "entry": [],
            "management": {
                "move_to_breakeven_at_tp_pct": 0.5,
                "breakeven_offset_pct": 0.1,
                "trailing_atr_mult": None,
                "max_hold_candles": None,
            },
        },
        "short": {"entry": [], "management": {}},
    }
    trader, cm = _make_trader_with_spec(spec)

    changed = trader._sanitize_strategy_spec()

    assert changed is True
    # Long: trailing НЕ добавлен (есть breakeven), max_hold добавлен.
    assert "trailing_atr_mult" in spec["long"]["management"]
    assert spec["long"]["management"]["trailing_atr_mult"] is None
    assert (
        spec["long"]["management"]["max_hold_candles"]
        == DEFAULT_MAX_HOLD_CANDLES
    )
    # Short: management пустой → оба поля добавлены.
    assert (
        spec["short"]["management"]["trailing_atr_mult"]
        == DEFAULT_TRAILING_ATR_MULT
    )
    assert (
        spec["short"]["management"]["max_hold_candles"]
        == DEFAULT_MAX_HOLD_CANDLES
    )


def test_sanitize_does_not_add_trailing_if_exit_on_level_present():
    """Любой exit_on_* — тоже считается механизмом сопровождения."""
    spec = {
        "long": {
            "entry": [],
            "management": {
                "exit_on_level_failure": True,
                "trailing_atr_mult": None,
                "max_hold_candles": None,
            },
        },
    }
    trader, cm = _make_trader_with_spec(spec)

    trader._sanitize_strategy_spec()

    # Trailing не добавлен, потому что есть exit_on_level_failure.
    assert spec["long"]["management"]["trailing_atr_mult"] is None
    # max_hold всё равно добавлен.
    assert (
        spec["long"]["management"]["max_hold_candles"]
        == DEFAULT_MAX_HOLD_CANDLES
    )


def test_sanitize_creates_management_if_missing():
    """Отсутствующая секция management создаётся."""
    spec = {"long": {"entry": []}, "short": {"entry": []}}
    trader, cm = _make_trader_with_spec(spec)

    trader._sanitize_strategy_spec()

    for side in ("long", "short"):
        assert isinstance(spec[side].get("management"), dict)
        assert (
            spec[side]["management"]["max_hold_candles"]
            == DEFAULT_MAX_HOLD_CANDLES
        )


def test_sanitize_idempotent():
    """
    Повторный вызов не даёт изменений.
    Это гарантирует, что в spec не накапливаются фантомные дефолты.
    """
    spec = {
        "long": {"entry": [], "management": {}},
        "short": {"entry": [], "management": {}},
    }
    trader, cm = _make_trader_with_spec(spec)

    first = trader._sanitize_strategy_spec()
    second = trader._sanitize_strategy_spec()

    assert first is True
    assert second is False
    # update_state вызывается только один раз — при первом проходе.
    assert cm.update_state.call_count == 1


def test_sanitize_noop_when_spec_invalid():
    """Если strategy_spec не dict — ничего не делаем и не падаем."""
    spec = None
    trader, cm = _make_trader_with_spec(spec)

    result = trader._sanitize_strategy_spec()

    assert result is False
    cm.update_state.assert_not_called()