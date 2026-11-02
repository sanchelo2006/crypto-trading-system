"""
Тесты проверки минимального размера ордера (minNotionalValue),
расчёта размера позиции с защитой от ErrCode 110094
и подхвата новых настроек в risk_state.json.
"""

import pytest

import config
from trading.executor import BybitExecutor
from trading.risk_manager import RiskManager


# ---------------------------------------------------------------- #
# Хелпер: фейковая HTTP-сессия для BybitExecutor
# ---------------------------------------------------------------- #

class FakeSession:
    def __init__(self, instrument_info=None):
        self.instrument_info = instrument_info or {
            "lotSizeFilter": {
                "qtyStep": "0.1",
                "minOrderQty": "0.1",
                "maxOrderQty": "1000000",
                "minNotionalValue": "5",
            },
            "priceFilter": {"tickSize": "0.0001"},
        }

    def get_instruments_info(self, category, symbol):
        return {
            "retCode": 0,
            "result": {"list": [self.instrument_info]},
        }


@pytest.fixture
def executor(monkeypatch):
    """BybitExecutor с фейковой сессией и валидными тестовыми ключами."""
    monkeypatch.setattr(config, "BYBIT_API_KEY", "k")
    monkeypatch.setattr(config, "BYBIT_API_SECRET", "s")

    ex = BybitExecutor.__new__(BybitExecutor)
    ex.testnet = False
    ex.api_key = "k"
    ex.api_secret = "s"
    ex.default_leverage = 10
    ex.margin_mode = "ISOLATED_MARGIN"
    ex.session = FakeSession()
    ex._instruments_cache = {}
    return ex


# ---------------------------------------------------------------- #
# validate_symbol_settings — pre-flight проверка настроек
# ---------------------------------------------------------------- #

class TestValidateSymbolSettings:
    def test_ok_when_notional_above_min(self, executor):
        # position_value = 100 * 0.5 = 50 USDT; * leverage 10 = 500 USDT > 5
        ok, warn = executor.validate_symbol_settings(
            symbol="XRPUSDT",
            capital_usdt=100.0,
            leverage=10,
            position_size_pct=50.0,
            entry_price=1.6,
        )
        assert ok
        assert warn is None

    def test_fails_when_notional_below_min(self, executor):
        # position_value = 1 * 0.5 = 0.5 USDT; * 1 = 0.5 USDT < 5
        ok, warn = executor.validate_symbol_settings(
            symbol="XRPUSDT",
            capital_usdt=1.0,
            leverage=1,
            position_size_pct=50.0,
            entry_price=1.6,
        )
        assert not ok
        assert "5.00 USDT" in warn
        assert "XRPUSDT" in warn

    def test_fails_when_qty_below_min_order_qty(self, executor):
        # notional большой, но entry огромный → qty крошечный
        ok, warn = executor.validate_symbol_settings(
            symbol="XRPUSDT",
            capital_usdt=100.0,
            leverage=1,
            position_size_pct=50.0,
            entry_price=10000.0,   # qty = 50/10000 = 0.005 → round down = 0
        )
        assert not ok


# ---------------------------------------------------------------- #
# calculate_position_size — защита от ErrCode 110094
# ---------------------------------------------------------------- #

class TestCalculatePositionSize:
    def test_returns_error_when_notional_too_small(self, executor):
        # capital=1, lev=1, entry=1.6 → qty = 0.625 → round down = 0.6
        # notional = 0.6 * 1.6 = 0.96 USDT < 5
        ok, result = executor.calculate_position_size(
            symbol="XRPUSDT",
            capital_usdt=1.0,
            leverage=1,
            entry_price=1.6,
        )
        assert not ok
        assert isinstance(result, str)
        assert "минимальной" in result.lower() or "minimum" in result.lower()

    def test_returns_qty_when_notional_ok(self, executor):
        ok, qty = executor.calculate_position_size(
            symbol="XRPUSDT",
            capital_usdt=100.0,
            leverage=10,
            entry_price=1.6,
        )
        assert ok
        assert isinstance(qty, float)
        assert qty > 0


# ---------------------------------------------------------------- #
# RiskManager: подхват новых настроек при перезапуске
# ---------------------------------------------------------------- #

@pytest.fixture
def tmp_state_file(tmp_path, monkeypatch):
    """Подменяет STATE_FILE на временный путь."""
    new_path = tmp_path / "risk_state.json"
    import trading.risk_manager as rm_mod
    monkeypatch.setattr(rm_mod, "STATE_FILE", new_path)
    return new_path


class TestRiskManagerSettingsChange:
    def test_initial_capital_saved_and_loaded(self, tmp_state_file):
        rm = RiskManager(
            symbol="XRPUSDT",
            daily_loss_limit_pct=5.0,
            monthly_loss_limit_pct=15.0,
            allocated_capital_usdt=100.0,
        )
        rm.record_closed_trade(pnl_pct=5.0, position_value_usdt=10.0)
        # current = 100 + 0.5 = 100.5

        # Новый экземпляр с теми же настройками → ничего не меняется.
        rm2 = RiskManager(
            symbol="XRPUSDT",
            daily_loss_limit_pct=5.0,
            monthly_loss_limit_pct=15.0,
            allocated_capital_usdt=100.0,
        )
        assert abs(rm2.current_capital - 100.5) < 1e-6
        assert rm2.initial_capital == 100.0

    def test_new_settings_trigger_recalculation(self, tmp_state_file):
        rm = RiskManager(
            symbol="XRPUSDT",
            daily_loss_limit_pct=5.0,
            monthly_loss_limit_pct=15.0,
            allocated_capital_usdt=100.0,
        )
        rm.record_closed_trade(pnl_pct=5.0, position_value_usdt=10.0)
        # current = 100.5

        # Пользователь поменял долю: база теперь 200 вместо 100.
        rm2 = RiskManager(
            symbol="XRPUSDT",
            daily_loss_limit_pct=5.0,
            monthly_loss_limit_pct=15.0,
            allocated_capital_usdt=200.0,
        )
        # Дельта = +100. Накопленные 0.5 переносятся → 200.5
        assert abs(rm2.current_capital - 200.5) < 1e-6
        assert rm2.initial_capital == 200.0

    def test_settings_shrink_reduces_capital(self, tmp_state_file):
        rm = RiskManager(
            symbol="XRPUSDT",
            daily_loss_limit_pct=5.0,
            monthly_loss_limit_pct=15.0,
            allocated_capital_usdt=200.0,
        )
        rm.record_closed_trade(pnl_pct=5.0, position_value_usdt=10.0)
        # current = 200.5

        # База уменьшена до 100. Дельта = -100. current = 100.5
        rm2 = RiskManager(
            symbol="XRPUSDT",
            daily_loss_limit_pct=5.0,
            monthly_loss_limit_pct=15.0,
            allocated_capital_usdt=100.0,
        )
        assert abs(rm2.current_capital - 100.5) < 1e-6

    def test_daily_pnl_preserved_on_settings_change(self, tmp_state_file):
        rm = RiskManager(
            symbol="XRPUSDT",
            daily_loss_limit_pct=5.0,
            monthly_loss_limit_pct=15.0,
            allocated_capital_usdt=100.0,
        )
        rm.record_closed_trade(pnl_pct=5.0, position_value_usdt=10.0)
        # daily = +0.5, monthly = +0.5

        rm2 = RiskManager(
            symbol="XRPUSDT",
            daily_loss_limit_pct=5.0,
            monthly_loss_limit_pct=15.0,
            allocated_capital_usdt=200.0,
        )
        # PnL периода сохраняется, меняется только база.
        assert abs(rm2.daily_realized_pnl - 0.5) < 1e-6
        assert abs(rm2.monthly_realized_pnl - 0.5) < 1e-6