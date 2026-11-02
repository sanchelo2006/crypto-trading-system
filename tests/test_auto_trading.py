"""
Unit-тесты модуля автоматической торговли (trading/*).

Запуск:
    pytest tests/test_auto_trading.py -v -s
    pytest tests/test_auto_trading.py -v -k "not Real"      # без сети

Реальные тесты (TestBybitExecutorReal) требуют в .env:
    BYBIT_API_KEY, BYBIT_API_SECRET, BYBIT_TESTNET=true
"""

import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, create_autospec, patch

import pytest
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import trading.risk_manager as risk_manager_module
import trading.trade_logger as trade_logger_module
from trading.executor import BybitExecutor, BybitError  # noqa: F401
from trading.signal_detector import SignalDetector
from trading.position_manager import PositionManager
from trading.risk_manager import RiskManager
from trading.trade_logger import TradeLogger
from trading.auto_trader import (
    AutoTraderConfig,
    AutoTraderSymbol,
    MAX_POSITIONS_PER_SYMBOL,
    is_auto_trading_running,
    get_all_states,
)
from strategy_spec import default_strategy_spec


# ============================================================================
# Общие фикстуры
# ============================================================================

@pytest.fixture(autouse=True)
def _isolated_storage(tmp_path, monkeypatch):
    """
    Изолирует состояние на диске для КАЖДОГО теста:
    - risk_state.json риск-менеджера
    - CSV-логи TradeLogger
    Без этого PnL накапливался между запусками (-302, -606, -906 ...).
    """
    monkeypatch.setattr(risk_manager_module, "STATE_FILE", tmp_path / "risk_state.json")
    monkeypatch.setattr(trade_logger_module, "LOGS_DIR", tmp_path / "logs")
    yield


def _make_executor_mock() -> MagicMock:
    """
    Мок BybitExecutor с проверкой сигнатур методов.
    Вызов с несуществующим kwargs (например category=...) упадёт TypeError.
    """
    executor = create_autospec(BybitExecutor, instance=True)
    executor.get_position_info.return_value = (True, [])
    executor.get_ticker_price.return_value = (True, 0.50)
    executor.place_market_order.return_value = (True, {"orderId": "order-1"})
    executor.set_trading_stop.return_value = (True, "ok")
    executor.set_leverage.return_value = (True, "ok")
    executor.get_usdt_balance.return_value = (True, 1000.0)
    executor.calculate_position_size.return_value = (True, 100.0)
    executor.get_instrument_info.return_value = (True, {
        "symbol": "XRPUSDT", "tickSize": 0.0001, "qtyStep": 1.0,
        "minOrderQty": 1.0, "maxOrderQty": 1_000_000.0, "minNotionalValue": 5.0,
    })

    # По умолчанию настройки считаются валидными.
    # Тесты на отказ старта переопределяют это значение.
    executor.validate_symbol_settings.return_value = (True, None)

    # Факт исполнения ордера: (успех, данные или None).
    # None означает: используем переданные qty/current_price как есть.
    executor.get_order_info.return_value = (True, None)

    # round_price — статический метод. Подключаем реальную реализацию,
    # иначе мок вернёт MagicMock и сломает проверки TP/SL.
    executor.round_price.side_effect = BybitExecutor.round_price

    return executor


def _simple_spec() -> dict:
    """Стратегия с предсказуемыми условиями для проверки детектора."""
    return {
        "long": {
            "entry": [
                {"type": "compare", "left": "close", "op": ">", "right": "sma_50"},
                {"type": "range", "value": "rsi_14", "min": 40, "max": 70},
            ],
            "stop_loss_pct": 2.0,
            "take_profit_pct": [4.0],
        },
        "short": {
            "entry": [
                {"type": "compare", "left": "close", "op": "<", "right": "sma_50"},
                {"type": "range", "value": "rsi_14", "min": 30, "max": 60},
            ],
            "stop_loss_pct": 2.0,
            "take_profit_pct": [4.0],
        },
        "filters": [
            {"type": "compare", "left": "adx_14", "op": ">", "right": 20},
        ],
    }


def _full_indicators(**overrides) -> dict:
    base = {
        "timestamp": datetime.now(timezone.utc),
        "open": 0.50, "high": 0.51, "low": 0.49, "close": 0.50, "volume": 1000.0,
        "sma_20": 0.495, "sma_50": 0.48, "rsi_14": 55.0, "atr_14": 0.02,
        "adx_14": 25.0, "macd": 0.001, "macd_signal": 0.0005, "macd_hist": 0.0005,
        "vwap": 0.5, "pivot": 0.5, "resistance_1": 0.52, "support_1": 0.48,
        "avg_volume_20": 900.0, "avg_volume_96": 800.0,
    }
    base.update(overrides)
    return base


# ============================================================================
# Окружение
# ============================================================================

def test_env_variables_loaded():
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    testnet = os.getenv("BYBIT_TESTNET")

    print(f"\n🔍 BYBIT_API_KEY: {'✅ установлен' if api_key else '❌ не найден'}")
    print(f"🔍 BYBIT_API_SECRET: {'✅ установлен' if api_secret else '❌ не найден'}")
    print(f"🔍 BYBIT_TESTNET: {testnet or '❌ не найден'}")


# ============================================================================
# BybitExecutor — без сети (HTTP-сессия замокана)
# ============================================================================

class TestBybitExecutorUnit:

    @pytest.fixture(autouse=True)
    def setup(self):
        with patch("trading.executor.HTTP") as http_cls:
            self.session = http_cls.return_value
            self.executor = BybitExecutor(
                testnet=True, api_key="test_key", api_secret="test_secret",
                default_leverage=10,
            )
            yield

    def test_usdt_balance_uses_wallet_balance_when_withdraw_empty(self):
        """UTA 2.0: availableToWithdraw == '' → fallback на walletBalance."""
        self.session.get_wallet_balance.return_value = {
            "retCode": 0,
            "result": {"list": [{"coin": [
                {"coin": "BTC", "walletBalance": "0.1"},
                {"coin": "USDT", "availableToWithdraw": "", "walletBalance": "1234.56", "equity": "1240.0"},
            ]}]},
        }
        success, balance = self.executor.get_usdt_balance()
        assert success is True
        assert balance == pytest.approx(1234.56)

    def test_usdt_balance_prefers_available_to_withdraw(self):
        self.session.get_wallet_balance.return_value = {
            "retCode": 0,
            "result": {"list": [{"coin": [
                {"coin": "USDT", "availableToWithdraw": "500.5", "walletBalance": "1000"},
            ]}]},
        }
        success, balance = self.executor.get_usdt_balance()
        assert success and balance == pytest.approx(500.5)

    def test_usdt_balance_no_usdt_returns_zero(self):
        self.session.get_wallet_balance.return_value = {
            "retCode": 0, "result": {"list": [{"coin": [{"coin": "BTC", "walletBalance": "1"}]}]},
        }
        success, balance = self.executor.get_usdt_balance()
        assert success and balance == 0.0

    def test_usdt_balance_api_error(self):
        self.session.get_wallet_balance.return_value = {"retCode": 10001, "retMsg": "bad request"}
        success, balance = self.executor.get_usdt_balance()
        assert success is False and balance is None

    def test_set_leverage_ok(self):
        self.session.set_leverage.return_value = {"retCode": 0, "retMsg": "OK"}
        success, _ = self.executor.set_leverage("XRPUSDT", 10)
        assert success
        self.session.set_leverage.assert_called_once_with(
            category="linear", symbol="XRPUSDT", buyLeverage="10", sellLeverage="10",
        )

    def test_set_leverage_not_modified_is_success(self):
        """ErrCode 110043 (leverage not modified) — плечо уже такое, это успех."""
        self.session.set_leverage.side_effect = Exception(
            "leverage not modified (ErrCode: 110043) (ErrTime: 20:27:23)."
        )
        success, _ = self.executor.set_leverage("XRPUSDT", 10)
        assert success is True

    def test_set_leverage_real_error(self):
        self.session.set_leverage.side_effect = Exception("invalid symbol (ErrCode: 10001)")
        success, err = self.executor.set_leverage("BADUSDT", 10)
        assert success is False and "10001" in err

    def test_recent_executions_uses_v5_execution_list(self):
        """GET /v5/execution/list → pybit session.get_executions."""
        self.session.get_executions.return_value = {
            "retCode": 0, "result": {"list": [{"execId": "1", "execPrice": "0.5"}]},
        }
        success, executions = self.executor.get_recent_executions("XRPUSDT", limit=10)
        assert success and len(executions) == 1
        self.session.get_executions.assert_called_once_with(
            category="linear", symbol="XRPUSDT", limit=10,
        )

    def test_instrument_info_parsing_and_cache(self):
        self.session.get_instruments_info.return_value = {
            "retCode": 0,
            "result": {"list": [{
                "symbol": "XRPUSDT",
                "priceFilter": {"tickSize": "0.0001"},
                "lotSizeFilter": {"qtyStep": "1", "minOrderQty": "1", "maxOrderQty": "1000000", "minNotionalValue": "5"},
            }]},
        }
        success, info = self.executor.get_instrument_info("XRPUSDT")
        assert success
        assert info["tickSize"] == pytest.approx(0.0001)
        assert info["qtyStep"] == 1.0 and info["minOrderQty"] == 1.0

        self.executor.get_instrument_info("XRPUSDT")
        assert self.session.get_instruments_info.call_count == 1, "второй вызов должен идти из кэша"

    def test_ticker_price(self):
        self.session.get_tickers.return_value = {
            "retCode": 0, "result": {"list": [{"lastPrice": "1.2345"}]},
        }
        success, price = self.executor.get_ticker_price("XRPUSDT")
        assert success and price == pytest.approx(1.2345)

    def test_place_market_order_params(self):
        self.session.place_order.return_value = {"retCode": 0, "result": {"orderId": "abc"}}
        success, result = self.executor.place_market_order(
            "XRPUSDT", "Buy", 100, reduce_only=False, stop_loss=0.49, take_profit=0.52,
        )
        assert success and result["orderId"] == "abc"
        kwargs = self.session.place_order.call_args.kwargs
        assert kwargs["category"] == "linear"
        assert kwargs["orderType"] == "Market"
        assert kwargs["qty"] == "100"
        assert kwargs["reduceOnly"] is False
        assert kwargs["positionIdx"] == 0
        assert kwargs["stopLoss"] == "0.49" and kwargs["takeProfit"] == "0.52"

    def test_calculate_position_size_by_capital(self):
        self.executor._instruments_cache["XRPUSDT"] = {
            "tickSize": 0.0001, "qtyStep": 1.0, "minOrderQty": 1.0,
            "maxOrderQty": 1e6, "minNotionalValue": 5.0,
        }
        # (100 USDT × 10x) / 0.5 = 2000 контрактов
        success, qty = self.executor.calculate_position_size(
            "XRPUSDT", capital_usdt=100.0, leverage=10, entry_price=0.5,
        )
        assert success and qty == 2000.0

    def test_calculate_position_size_by_risk(self):
        self.executor._instruments_cache["XRPUSDT"] = {
            "tickSize": 0.0001, "qtyStep": 1.0, "minOrderQty": 1.0,
            "maxOrderQty": 1e6, "minNotionalValue": 5.0,
        }
        # риск 1% от 1000 = 10 USDT; SL 2% от 0.5 = 0.01/контракт → 1000 контрактов
        success, qty = self.executor.calculate_position_size(
            "XRPUSDT", capital_usdt=1000.0, leverage=10, entry_price=0.5,
            risk_pct=1.0, stop_loss_pct=2.0,
        )
        assert success and qty == 1000.0

    def test_calculate_position_size_respects_min_qty(self):
        self.executor._instruments_cache["XRPUSDT"] = {
            "tickSize": 0.0001, "qtyStep": 1.0, "minOrderQty": 10.0,
            "maxOrderQty": 1e6, "minNotionalValue": 5.0,
        }
        success, qty = self.executor.calculate_position_size(
            "XRPUSDT", capital_usdt=1.0, leverage=1, entry_price=0.5,
        )
        assert success and qty == 10.0

    def test_round_quantity_and_price(self):
        assert BybitExecutor.round_quantity(123.456, 0.01) == pytest.approx(123.45)
        assert BybitExecutor.round_quantity(123.999, 1.0) == 123.0
        assert BybitExecutor.round_quantity(5.0, 0) == 5.0
        assert BybitExecutor.round_price(1.23456, 0.0001) == pytest.approx(1.2345)
        assert BybitExecutor.round_price(1.23456, 0) == 1.23456

    def test_missing_keys_raise(self):
        with patch("trading.executor.config") as cfg:
            cfg.BYBIT_API_KEY = ""
            cfg.BYBIT_API_SECRET = ""
            with pytest.raises(ValueError):
                BybitExecutor(testnet=True)


# ============================================================================
# BybitExecutor — реальное подключение к testnet
# ============================================================================

class TestBybitExecutorReal:

    @pytest.fixture(autouse=True)
    def setup(self):
        api_key = os.getenv("BYBIT_API_KEY")
        api_secret = os.getenv("BYBIT_API_SECRET")
        testnet = os.getenv("BYBIT_TESTNET", "false").lower() == "true"

        if not api_key or not api_secret:
            pytest.skip("BYBIT_API_KEY и BYBIT_API_SECRET не установлены")
        if not testnet:
            pytest.skip("Тесты выполняются только на testnet (BYBIT_TESTNET=true)")

        self.executor = BybitExecutor(testnet=True, default_leverage=10)

    def test_get_wallet_balance_raw(self):
        success, result = self.executor.get_wallet_balance()
        assert success, f"Ошибка: {result}"
        assert "list" in result
        print(f"\n✅ Кошелёк получен, аккаунтов: {len(result['list'])}")

    def test_get_usdt_balance(self):
        success, balance = self.executor.get_usdt_balance()
        assert success, "Не удалось получить баланс USDT"
        assert isinstance(balance, float)
        assert balance >= 0
        print(f"\n✅ Баланс USDT: {balance:.2f}")

    def test_get_ticker_price(self):
        success, price = self.executor.get_ticker_price("XRPUSDT")
        assert success and isinstance(price, float) and price > 0
        print(f"\n✅ Текущая цена XRPUSDT: {price:.4f}")

    def test_get_instrument_info(self):
        success, info = self.executor.get_instrument_info("XRPUSDT")
        assert success, f"Ошибка: {info}"
        assert info["qtyStep"] > 0 and info["tickSize"] > 0
        print(f"\n✅ XRPUSDT: qtyStep={info['qtyStep']}, tickSize={info['tickSize']}, "
              f"minQty={info['minOrderQty']}")

    def test_get_position(self):
        success, positions = self.executor.get_position_info("XRPUSDT")
        assert success and isinstance(positions, list)
        print(f"\n✅ Записей о позициях XRPUSDT: {len(positions)}")

    def test_get_recent_executions(self):
        success, executions = self.executor.get_recent_executions("XRPUSDT", limit=10)
        assert success, f"Ошибка: {executions}"
        assert isinstance(executions, list)
        print(f"\n✅ Исполнений XRPUSDT: {len(executions)}")

    def test_set_leverage_idempotent(self):
        """Повторная установка того же плеча (110043) должна быть успехом."""
        success, result = self.executor.set_leverage("XRPUSDT", 10)
        assert success, f"Ошибка: {result}"
        success, result = self.executor.set_leverage("XRPUSDT", 10)
        assert success, f"Повторная установка: {result}"
        print("\n✅ Плечо 10x установлено (повтор — ok)")

    def test_calculate_position_size_real(self):
        success, price = self.executor.get_ticker_price("XRPUSDT")
        assert success
        success, qty = self.executor.calculate_position_size(
            "XRPUSDT", capital_usdt=50.0, leverage=5, entry_price=price,
        )
        assert success and qty > 0
        print(f"\n✅ Размер позиции на 50 USDT × 5x @ {price:.4f}: {qty}")


# ============================================================================
# SignalDetector
# ============================================================================

class TestSignalDetector:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.detector = SignalDetector()

    def test_empty_spec(self):
        assert self.detector.detect_signals({}, _full_indicators(), {}) == []
        assert self.detector.detect_signals(None, _full_indicators(), {}) == []

    def test_default_spec_returns_list(self):
        signals = self.detector.detect_signals(default_strategy_spec(), _full_indicators(), {})
        assert isinstance(signals, list)

    def test_long_signal(self):
        ind = _full_indicators(close=0.50, sma_50=0.48, rsi_14=55.0, adx_14=25.0)
        signals = self.detector.detect_signals(_simple_spec(), ind, {})
        assert len(signals) == 1
        assert signals[0]["direction"] == "long"
        assert signals[0]["side_spec"]["stop_loss_pct"] == 2.0
        assert isinstance(signals[0]["reasons"], list) and signals[0]["reasons"]
        print(f"\n✅ LONG: {signals[0]['reasons']}")

    def test_short_signal(self):
        ind = _full_indicators(close=0.46, sma_50=0.48, rsi_14=45.0, adx_14=25.0)
        signals = self.detector.detect_signals(_simple_spec(), ind, {})
        assert len(signals) == 1 and signals[0]["direction"] == "short"

    def test_global_filter_blocks(self):
        ind = _full_indicators(close=0.50, sma_50=0.48, rsi_14=55.0, adx_14=15.0)
        assert self.detector.detect_signals(_simple_spec(), ind, {}) == []

    def test_rsi_out_of_range_blocks(self):
        ind = _full_indicators(close=0.50, sma_50=0.48, rsi_14=80.0, adx_14=25.0)
        assert self.detector.detect_signals(_simple_spec(), ind, {}) == []

    def test_entry_any_or_groups(self):
        spec = _simple_spec()
        spec["long"] = {
            "entry_any": [
                [{"type": "compare", "left": "rsi_14", "op": ">", "right": 99}],   # не сработает
                [{"type": "compare", "left": "close", "op": ">", "right": "sma_50"}],  # сработает
            ],
            "stop_loss_pct": 2.0, "take_profit_pct": [4.0],
        }
        ind = _full_indicators(close=0.50, sma_50=0.48, rsi_14=80.0, adx_14=25.0)
        signals = self.detector.detect_signals(spec, ind, {})
        assert [s["direction"] for s in signals] == ["long"]

    def test_insufficient_data(self):
        signals = self.detector.detect_signals(_simple_spec(), {"close": 0.5}, {})
        assert signals == []


# ============================================================================
# PositionManager
# ============================================================================

class TestPositionManager:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.executor = _make_executor_mock()
        self.manager = PositionManager(self.executor, "XRPUSDT", "linear")

    def _open_long(self, qty=100.0, price=0.50):
        ok, err = self.manager.open_position(
            direction="long", qty=qty, current_price=price,
            side_spec={"stop_loss_pct": 2.0, "take_profit_pct": [4.0]},
            entry_reasons=["test"], indicators=_full_indicators(atr_14=None),
        )
        assert ok, err
        return self.manager.positions[-1]

    def test_init(self):
        assert self.manager.symbol == "XRPUSDT"
        assert self.manager.positions == []

    def test_sync_with_exchange_empty(self):
        ok, err = self.manager.sync_with_exchange()
        assert ok and err is None
        self.executor.get_position_info.assert_called_once_with("XRPUSDT")

    def test_sync_does_not_clear_local_when_exchange_empty(self):
        """
        sync_with_exchange НЕ удаляет локальные позиции, если биржа пуста.

        Удаление и логирование CLOSE делает _detect_closed_positions.
        """
        self._open_long()
        self.executor.get_position_info.return_value = (True, [{"side": "Buy", "size": "0"}])
        self.manager.sync_with_exchange()
        # Позиция должна остаться — её удалит detect_closed_positions.
        assert len(self.manager.positions) == 1

    def test_sync_adds_orphan_when_exchange_has_position(self):
        """Если биржа вернула позицию, а локально её нет — добавляем orphan."""
        self.executor.get_position_info.return_value = (
            True, [{"side": "Buy", "size": "100", "avgPrice": "0.5"}]
        )
        self.manager.sync_with_exchange()
        assert len(self.manager.positions) == 1
        assert self.manager.positions[0]["is_orphan"] is True

    def test_sync_keeps_local_when_exchange_has_position(self):
        self._open_long()
        self.executor.get_position_info.return_value = (True, [{"side": "Buy", "size": "100"}])
        self.manager.sync_with_exchange()
        assert len(self.manager.positions) == 1

    def test_open_long_position(self):
        pos = self._open_long()
        assert pos["direction"] == "long"
        assert pos["stop_loss"] == pytest.approx(0.49)
        assert pos["take_profit"] == pytest.approx(0.52)
        assert pos["bybit_order_id"] == "order-1"

        kwargs = self.executor.place_market_order.call_args.kwargs
        assert kwargs["side"] == "Buy" and kwargs["reduce_only"] is False
        self.executor.set_trading_stop.assert_called_once()

    def test_open_short_position_side(self):
        ok, _ = self.manager.open_position(
            direction="short", qty=50, current_price=0.50,
            side_spec={"stop_loss_pct": 2.0, "take_profit_pct": [4.0]},
            entry_reasons=[], indicators=_full_indicators(atr_14=None),
        )
        assert ok
        assert self.executor.place_market_order.call_args.kwargs["side"] == "Sell"
        pos = self.manager.positions[0]
        assert pos["stop_loss"] > 0.50 > pos["take_profit"]

    def test_open_position_order_failure(self):
        self.executor.place_market_order.return_value = (False, "insufficient balance")
        ok, err = self.manager.open_position(
            direction="long", qty=100, current_price=0.5,
            side_spec={}, entry_reasons=[], indicators={},
        )
        assert not ok and "insufficient" in err
        assert self.manager.positions == []

    def test_open_position_survives_tp_sl_failure(self):
        self.executor.set_trading_stop.return_value = (False, "err")
        pos = self._open_long()
        assert pos is not None and len(self.manager.positions) == 1

    def test_close_position_pnl(self):
        pos = self._open_long(price=0.50)
        ok, trade = self.manager.close_position(pos, "take_profit", current_price=0.52)
        assert ok
        # result_pct нетто: брутто 4.0% минус round-trip комиссия 2×0.055%.
        assert trade["result_pct"] == pytest.approx(4.0 - 0.11)
        assert trade["exit_reason"] == "take_profit"
        assert self.manager.positions == []
        kwargs = self.executor.place_market_order.call_args.kwargs
        assert kwargs["side"] == "Sell" and kwargs["reduce_only"] is True

    def test_close_all_positions(self):
        self._open_long()
        self._open_long()
        trades = self.manager.close_all_positions(reason="manual_stop")
        assert len(trades) == 2 and self.manager.positions == []

    def test_check_exits_stop_loss_and_take_profit(self):
        pos = self._open_long(price=0.50)          # SL 0.49, TP 0.52
        exits = self.manager.check_exits(_full_indicators(low=0.485, high=0.50))
        assert exits and exits[0][1] == "stop_loss"
        assert exits[0][2] == pytest.approx(pos["stop_loss"])

        exits = self.manager.check_exits(_full_indicators(low=0.50, high=0.525))
        assert exits and exits[0][1] == "take_profit"

        assert self.manager.check_exits(_full_indicators(low=0.495, high=0.505)) == []

    def test_check_exit_management_trend_loss(self):
        pos = self._open_long(price=0.50)
        pos["management"] = {"exit_on_trend_loss": True}
        exits = self.manager.check_exits(_full_indicators(low=0.495, high=0.505, close=0.495, sma_50=0.50))
        assert exits and exits[0][1] == "trend_loss"

    def test_compute_tp_sl_with_atr(self):
        sl, tp = self.manager._compute_tp_sl(
            "long", 1.0, {"stop_loss_atr_mult": 1.5, "take_profit_atr_mult": 3.0},
            {"atr_14": 0.02},
        )
        # ATR дал бы SL = 0.97 (-3%), но MAX_STOP_PCT=2.5% клампит до 0.975.
        assert sl == pytest.approx(0.975)
        # TP: 1.0 + 3*0.02 = 1.06. Кламп 10% не срабатывает.
        assert tp == pytest.approx(1.06)

    def test_compute_tp_sl_clamps_extreme_atr(self):
        """Если ATR даёт SL дальше 2.5% — обрезаем до 2.5%."""
        sl, tp = self.manager._compute_tp_sl(
            "long", 1.0, {"stop_loss_atr_mult": 50.0, "take_profit_atr_mult": 50.0},
            {"atr_14": 1.0},   # абсурдный ATR
        )
        assert sl == pytest.approx(0.975)   # -2.5% от entry
        assert tp == pytest.approx(1.10)    # +10% от entry

    def test_compute_tp_sl_pct_fallback(self):
        sl, tp = self.manager._compute_tp_sl(
            "short", 1.0, {"stop_loss_pct": 2.0, "take_profit_pct": [4.0]}, {"atr_14": float("nan")},
        )
        assert sl == pytest.approx(1.02) and tp == pytest.approx(0.96)


# ============================================================================
# RiskManager
# ============================================================================

class TestRiskManager:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.rm = RiskManager("XRPUSDT", daily_loss_limit_pct=5.0,
                              monthly_loss_limit_pct=15.0, allocated_capital_usdt=1000.0)

    def test_fresh_state(self):
        assert self.rm.daily_realized_pnl == 0.0
        assert self.rm.monthly_realized_pnl == 0.0
        assert not self.rm.daily_stop_triggered
        assert not self.rm.monthly_optimization_triggered

    def test_record_closed_trade(self):
        self.rm.record_closed_trade(pnl_pct=-2.0, position_value_usdt=100.0)
        assert self.rm.daily_realized_pnl == pytest.approx(-2.0)
        assert self.rm.monthly_realized_pnl == pytest.approx(-2.0)
        self.rm.record_closed_trade(pnl_pct=5.0, position_value_usdt=100.0)
        assert self.rm.daily_realized_pnl == pytest.approx(3.0)

    def test_loss_pct_helpers(self):
        self.rm.record_closed_trade(-3.0, 1000.0)   # -30 USDT → капитал 970

        # Лимиты считаются от ТЕКУЩЕГО капитала (970), а не от начального (1000):
        # дневной:  (-30 + -20) / 970 * 100 = -5.1546...
        # месячный: (-30) / 970 * 100       = -3.0928...
        expected_daily = (-50.0 / 970.0) * 100.0
        expected_monthly = (-30.0 / 970.0) * 100.0

        assert self.rm.get_total_daily_loss_pct(-20.0) == pytest.approx(expected_daily)
        assert self.rm.get_total_monthly_loss_pct() == pytest.approx(expected_monthly)

    def test_daily_stop_not_triggered_below_limit(self):
        self.rm.record_closed_trade(-4.0, 1000.0)   # -40 USDT = -4%
        assert self.rm.should_stop_daily(0.0) is False

    def test_daily_stop_triggered_once(self):
        self.rm.record_closed_trade(-6.0, 1000.0)   # -60 USDT = -6% < -5%
        assert self.rm.should_stop_daily(0.0) is True
        assert self.rm.should_stop_daily(0.0) is False, "повторно не должен срабатывать"

    def test_daily_stop_includes_unrealized(self):
        self.rm.record_closed_trade(-3.0, 1000.0)   # -30
        assert self.rm.should_stop_daily(unrealized_pnl_usdt=-10.0) is False   # -4%
        assert self.rm.should_stop_daily(unrealized_pnl_usdt=-25.0) is True    # -5.5%

    def test_monthly_optimize(self):
        self.rm.record_closed_trade(-10.0, 1000.0)  # -10%
        assert self.rm.should_optimize_monthly() is False
        self.rm.record_closed_trade(-6.0, 1000.0)   # -16%
        assert self.rm.should_optimize_monthly() is True
        assert self.rm.should_optimize_monthly() is False
        self.rm.reset_monthly_optimization_flag()
        assert self.rm.should_optimize_monthly() is True

    def test_state_persists_between_instances(self):
        self.rm.record_closed_trade(-2.5, 1000.0)
        assert risk_manager_module.STATE_FILE.exists()
        restored = RiskManager("XRPUSDT", 5.0, 15.0, 1000.0)
        assert restored.daily_realized_pnl == pytest.approx(-25.0)
        assert restored.monthly_realized_pnl == pytest.approx(-25.0)

    def test_state_isolated_per_symbol(self):
        self.rm.record_closed_trade(-2.0, 1000.0)
        other = RiskManager("SOLUSDT", 5.0, 15.0, 1000.0)
        assert other.daily_realized_pnl == 0.0

    def test_rollover_resets_daily(self):
        self.rm.record_closed_trade(-6.0, 1000.0)
        assert self.rm.should_stop_daily(0.0)
        self.rm.current_date = self.rm.current_date - timedelta(days=1)
        self.rm._check_rollover()
        assert self.rm.daily_realized_pnl == 0.0
        assert self.rm.daily_stop_triggered is False
        assert self.rm.monthly_realized_pnl == pytest.approx(-60.0)

    def test_rollover_resets_monthly(self):
        self.rm.record_closed_trade(-6.0, 1000.0)
        self.rm.current_month = "2000-01"
        self.rm._check_rollover()
        assert self.rm.monthly_realized_pnl == 0.0
        assert self.rm.monthly_optimization_triggered is False

    def test_zero_capital_safe(self):
        rm = RiskManager("ZERO", 5.0, 15.0, 0.0)
        assert rm.get_total_daily_loss_pct(-100) == 0.0
        assert rm.should_stop_daily(-100) is False


# ============================================================================
# TradeLogger
# ============================================================================

class TestTradeLogger:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.logger = TradeLogger("XRPUSDT")

    def test_file_naming(self):
        suffix = self.logger._week_suffix()
        path = self.logger._get_file_path("trades")
        assert path.name == f"XRPUSDT_trades_{suffix}.csv"
        assert path.parent == trade_logger_module.LOGS_DIR

    def test_week_suffix_format(self):
        """Суффикс недели соответствует формату YYYY_Www."""
        import re
        suffix = self.logger._week_suffix()
        assert re.fullmatch(r"\d{4}_W\d{2}", suffix), (
            f"Ожидался формат YYYY_Www, получено: {suffix}"
        )

    def test_log_trade_creates_csv(self):
        self.logger.log_trade({
            "position_id": "p-1",
            "entry_time": datetime.now(timezone.utc),
            "exit_time": datetime.now(timezone.utc),
            "direction": "long", "entry_price": 0.50, "exit_price": 0.52,
            "qty": 100, "position_value_usdt": 50.0, "result_pct": 4.0,
            "pnl_usdt": 2.0, "exit_reason": "take_profit",
            "entry_reasons": ["RSI ok", "ADX ok"],
        })
        path = self.logger._get_file_path("trades")
        assert path.exists()
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        assert lines[0].startswith("timestamp,position_id,")
        assert "p-1" in lines[1] and "RSI ok; ADX ok" in lines[1]

        self.logger.log_trade({"position_id": "p-2", "direction": "short"})
        assert len(path.read_text(encoding="utf-8-sig").splitlines()) == 3, "заголовок пишется один раз"

    def test_log_event(self):
        self.logger.log_event("START", "started", "details")
        path = self.logger._get_file_path("events")
        content = path.read_text(encoding="utf-8-sig")
        assert "START" in content and "started" in content

    def test_log_optimization(self):
        self.logger.log_optimization({
            "trigger_reason": "monthly_loss_limit",
            "history": [1, 2, 3],
            "best": {
                "aggregated_test_stats": {"return_pct": 3.2, "profit_factor": 1.4, "total_trades": 12},
                "holdout_summary": {"status": "passed", "return_pct": 1.1},
            },
            "new_strategy_saved": True,
        })
        content = self.logger._get_file_path("optimization").read_text(encoding="utf-8-sig")
        assert "monthly_loss_limit" in content and "passed" in content and "True" in content


# ============================================================================
# AutoTrader
# ============================================================================

class TestAutoTrader:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.executor = _make_executor_mock()
        self.notifications = []
        self.memory = MagicMock()
        self.cm = MagicMock()
        self.cm.get_state.return_value = {"strategy_spec": _simple_spec()}
        self.memory.get.return_value = self.cm

        self.cfg = AutoTraderConfig("XRPUSDT", capital_share_pct=30.0,
                                    position_size_pct=50.0, leverage=10)
        self.trader = AutoTraderSymbol(
            self.cfg, executor=self.executor, fetcher=MagicMock(), memory=self.memory,
            backtest_engine=MagicMock(), backtest_optimizer=MagicMock(), ai_client=MagicMock(),
            notify_fn=lambda m, t: self.notifications.append((m, t)),
            total_capital_usdt=1000.0,
        )

    def test_config_defaults(self):
        cfg = AutoTraderConfig("SOLUSDT")
        assert cfg.category == "linear" and cfg.leverage == 10
        # Дневной лимит намеренно жёсткий (1%): автоторговля в стадии валидации,
        # день останавливается после ~1 среднего убытка. Месячный — 15%.
        assert cfg.daily_loss_limit_pct == 1.0 and cfg.monthly_loss_limit_pct == 15.0

    def test_registry_empty(self):
        assert is_auto_trading_running() is False
        assert is_auto_trading_running("XRPUSDT") is False
        assert get_all_states() == {}

    def test_allocated_capital(self):
        assert self.trader.allocated_capital == pytest.approx(300.0)
        assert self.trader.risk_manager.initial_capital == pytest.approx(300.0)
        assert self.trader.risk_manager.current_capital == pytest.approx(300.0)

    def test_get_state(self):
        state = self.trader.get_state()
        assert state["symbol"] == "XRPUSDT"
        assert state["is_running"] is False
        assert state["open_positions"] == 0
        assert state["allocated_capital"] == 300.0

    def test_start_blocked_by_auto_mode(self):
        with patch("trading.auto_trader.AutoModeState.get_state", return_value={"is_running": True}):
            self.trader.start()
        assert self.trader.is_running is False
        assert self.notifications and self.notifications[-1][1] == "warning"

    def test_setup_leverage(self):
        assert self.trader._setup_leverage() is True
        self.executor.set_leverage.assert_called_once_with("XRPUSDT", 10)
        self.executor.set_leverage.return_value = (False, "err")
        assert self.trader._setup_leverage() is False

    def test_start_refused_when_settings_invalid(self):
        """
        Если pre-flight проверка настроек вернула False — трейдер
        не стартует, is_running остаётся False, поток не создаётся.
        """
        self.executor.get_ticker_price.return_value = (True, 1000.0)
        self.executor.validate_symbol_settings.return_value = (
            False,
            "стоимость позиции меньше минимальной",
        )

        result = self.trader.start()

        assert result is False
        assert self.trader.is_running is False
        assert self.trader._thread is None

    def test_start_succeeds_when_settings_valid(self):
        """Когда настройки валидны — start() возвращает True и поднимает поток."""
        self.executor.get_ticker_price.return_value = (True, 0.5)
        self.executor.validate_symbol_settings.return_value = (True, None)

        result = self.trader.start()

        assert result is True
        # Останавливаем поток, чтобы тест не оставлял живые фоновые задачи.
        self.trader.is_running = False
        if self.trader._thread:
            self.trader._thread.join(timeout=2)

    def test_unrealized_pnl(self):
        self.executor.get_position_info.return_value = (True, [
            {"side": "Buy", "size": "100", "unrealisedPnl": "-3.5"},
            {"side": "Sell", "size": "0", "unrealisedPnl": ""},
        ])
        assert self.trader._get_unrealized_pnl() == pytest.approx(-3.5)

    def test_open_position_from_signal(self):
        signal = {"direction": "long", "side_spec": _simple_spec()["long"], "reasons": ["r"]}
        self.trader._open_position_from_signal(signal, _full_indicators(atr_14=None))

        # размер позиции = 300 × 50% = 150 USDT
        kwargs = self.executor.calculate_position_size.call_args.kwargs
        assert kwargs["capital_usdt"] == pytest.approx(150.0) and kwargs["leverage"] == 10
        assert len(self.trader.position_manager.positions) == 1
        assert self.notifications[-1][1] == "positive"

    def test_process_signals_respects_limit(self):
        for _ in range(MAX_POSITIONS_PER_SYMBOL):
            self.trader.position_manager.positions.append({"position_id": "x", "direction": "long",
                                                           "entry_price": 0.5, "qty": 1})
        self.trader._process_signals(_full_indicators(close=0.50, sma_50=0.48, rsi_14=55, adx_14=25), {})
        self.executor.place_market_order.assert_not_called()

    def test_process_signals_stops_without_spec(self):
        # С Шага 4.3c спека читается через режим-зависимый слот памяти,
        # поэтому мок должен отвечать на get_strategy_spec_for_regime.
        self.cm.get_state.return_value = {}
        self.cm.get_strategy_spec_for_regime.return_value = None
        self.trader.is_running = True
        self.trader._process_signals(_full_indicators(), {})
        assert self.trader.is_running is False

    def test_detect_closed_positions(self):
        # Чистим позиции от setUp, чтобы тест не зависел от его состояния.
        self.trader.position_manager.positions = []

        # entry_time ставим на 5 минут назад, чтобы пройти grace period
        # ...
        self.trader.position_manager.positions.append({
            "position_id": "p-1", "direction": "long", "entry_price": 0.50,
            "qty": 100.0,
            "entry_time": datetime.now(timezone.utc) - timedelta(minutes=5),
            "entry_reasons": [],
            "stop_loss": 0.45,
            "take_profit": 0.60,
            "management": {},
            "entry_snapshot": {},
        })
        self.executor.get_position_info.return_value = (True, [{"side": "Buy", "size": "0"}])
        self.executor.get_ticker_price.return_value = (True, 0.52)

        # Closed-PnL вернёт данные с биржи — заглушка "нет данных",
        # чтобы сработал fallback через тикер.
        self.executor.get_closed_pnl.return_value = (True, [])

        self.trader._detect_closed_positions()

        assert self.trader.position_manager.positions == []
        # 100 × 0.5 = 50 USDT × 4% = +2 USDT
        # 100 × 0.5 = 50 USDT × 4% = +2 USDT брутто;
        # нетто: минус round-trip комиссия от входного notionа 50 × 0.11% = 0.055 USDT.
        assert self.trader.risk_manager.daily_realized_pnl == pytest.approx(2.0 - 0.055)
        assert self.trader.trade_logger._get_file_path("trades").exists()

    def test_daily_stop_closes_and_stops(self):
        self.trader.position_manager.positions.append({
            "position_id": "p-1", "direction": "long", "entry_price": 0.50,
            "qty": 100.0, "entry_time": datetime.now(timezone.utc), "entry_reasons": [],
        })
        self.trader.is_running = True
        self.trader._handle_daily_stop()
        assert self.trader.is_running is False
        assert self.trader.position_manager.positions == []
        assert self.executor.place_market_order.call_args.kwargs["reduce_only"] is True

    def test_insufficient_balance_stops_trading(self):
        self.trader.is_running = True
        self.executor.place_market_order.return_value = (False, "Insufficient balance")
        signal = {"direction": "long", "side_spec": _simple_spec()["long"], "reasons": []}
        self.trader._open_position_from_signal(signal, _full_indicators(atr_14=None))
        assert self.trader.is_running is False


# ============================================================================
# Интеграция (всё на моках, без сети)
# ============================================================================

class TestIntegration:

    def test_signal_to_close_workflow(self):
        executor = _make_executor_mock()
        detector = SignalDetector()
        pm = PositionManager(executor, "XRPUSDT")
        rm = RiskManager("XRPUSDT", 5.0, 15.0, 1000.0)
        tl = TradeLogger("XRPUSDT")

        ind = _full_indicators(close=0.50, sma_50=0.48, rsi_14=55.0, adx_14=25.0, atr_14=None)
        signals = detector.detect_signals(_simple_spec(), ind, {})
        assert len(signals) == 1

        sig = signals[0]
        ok, _ = pm.open_position(sig["direction"], 100.0, 0.50, sig["side_spec"], sig["reasons"], ind)
        assert ok and len(pm.positions) == 1

        exits = pm.check_exits(_full_indicators(low=0.48, high=0.50))
        assert exits and exits[0][1] == "stop_loss"

        position, reason, price = exits[0]
        ok, trade = pm.close_position(position, reason, current_price=price)
        # стоп -2.0% брутто; нетто минус round-trip комиссия 0.11%.
        assert ok and trade["result_pct"] == pytest.approx(-2.0 - 0.11)

        value = trade["qty"] * trade["entry_price"]
        rm.record_closed_trade(trade["result_pct"], value)
        tl.log_trade({**trade, "position_value_usdt": value, "pnl_usdt": value * trade["result_pct"] / 100})

        # PnL в USDT считается от нетто result_pct (-2.11%): 50 × -2.11% = -1.055.
        assert rm.daily_realized_pnl == pytest.approx(value * (-2.0 - 0.11) / 100)
        assert tl._get_file_path("trades").exists()
        print("\n✅ Цепочка сигнал → открытие → SL → учёт риска → лог отработала")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])