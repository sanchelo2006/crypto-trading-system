#!/usr/bin/env python3
"""Unit-тесты для risk_controls в backtest_engine."""
import unittest
import sys
import os
import numpy as np
import pandas as pd

from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest_engine import BacktestEngine


class _MockFetcher:
    def __init__(self, df):
        self._df = df

    def fetch_ohlcv(self, symbol, interval, start, end, category, progress_callback=None):
        return self._df.copy()


class TestRiskControlsUnit(unittest.TestCase):
    """Юнит-тесты приватных методов без тяжёлых вычислений."""

    def setUp(self):
        self.engine = BacktestEngine(fetcher=_MockFetcher(pd.DataFrame()))

    def test_disabled_when_missing(self):
        controls = self.engine._get_risk_controls({})
        self.assertFalse(controls["enabled"])
        self.assertEqual(controls["max_consecutive_losses"], 0)

    def test_disabled_when_flag_false(self):
        controls = self.engine._get_risk_controls({"risk_controls": {"enabled": False}})
        self.assertFalse(controls["enabled"])

    def test_normalized_defaults(self):
        controls = self.engine._get_risk_controls({"risk_controls": {"enabled": True}})
        self.assertTrue(controls["enabled"])
        self.assertEqual(controls["max_consecutive_losses"], 3)
        self.assertEqual(controls["pause_candles_after_losses"], 96)
        self.assertEqual(controls["max_losses_in_window"], 5)
        self.assertEqual(controls["loss_window_trades"], 10)

    def test_clamps_to_minimums(self):
        controls = self.engine._get_risk_controls({"risk_controls": {
            "enabled": True,
            "max_consecutive_losses": 0,
            "pause_candles_after_losses": 0,
            "max_losses_in_window": 0,
            "loss_window_trades": 0,
        }})
        self.assertEqual(controls["max_consecutive_losses"], 1)
        self.assertEqual(controls["pause_candles_after_losses"], 1)

    def test_should_pause_disabled(self):
        controls = {"enabled": False}
        trade = {"result_pct": -1.0}
        self.assertFalse(
            self.engine._should_pause_after_loss(trade, 5, [1, 1, 1, 1, 1], controls)
        )

    def test_should_pause_profit_never(self):
        controls = {"enabled": True, "max_consecutive_losses": 3,
                    "loss_window_trades": 10, "max_losses_in_window": 5}
        trade = {"result_pct": 1.0}
        self.assertFalse(
            self.engine._should_pause_after_loss(trade, 3, [1, 1, 1, 0, 0], controls)
        )

    def test_should_pause_consecutive(self):
        controls = {"enabled": True, "max_consecutive_losses": 3,
                    "loss_window_trades": 10, "max_losses_in_window": 99}
        trade = {"result_pct": -1.0}
        self.assertTrue(
            self.engine._should_pause_after_loss(trade, 3, [1, 1, 1], controls)
        )

    def test_should_pause_window_rule(self):
        controls = {"enabled": True, "max_consecutive_losses": 99,
                    "loss_window_trades": 5, "max_losses_in_window": 3}
        trade = {"result_pct": -1.0}
        self.assertTrue(
            self.engine._should_pause_after_loss(trade, 2, [1, 0, 1, 1, 0], controls)
        )


class TestRiskControlsSimulation(unittest.TestCase):
    """Интеграционный тест: пауза активируется и пропускает свечи."""

    def _run_stats(self, spec):
        n = 300
        ts = pd.date_range("2026-07-01", periods=n, freq="15min", tz="UTC")
        closes = np.linspace(1.0, 0.7, n)  # нисходящий тренд: long убыточен
        opens = closes
        highs = closes + 0.001
        lows = closes - 0.001
        df = pd.DataFrame({
            "timestamp": ts, "open": opens, "high": highs,
            "low": lows, "close": closes, "volume": np.full(n, 1000.0),
        })
        engine = BacktestEngine(fetcher=_MockFetcher(df))
        result = engine.run(
            symbol="TESTUSDT", category="linear", strategy_spec=spec,
            start_date=ts[0], end_date=ts[-1],
            initial_capital=1000.0, commission_pct=0.0, rules_text="",
        )
        return result["stats"]

    def test_pause_activates_after_losses(self):
        spec = {
            "long": {
                "entry": [
                    {"type": "compare", "left": "close", "op": ">", "right": 0},
                    # Трендовый фильтр намеренно не ограничивающий
                    # (close > 0.5 × sma_50): данные теста по замыслу
                    # нисходящие, чтобы генерировать серии убытков для
                    # проверки паузы; честное close > sma_50 заблокировало
                    # бы все входы. Барьеру валидатора достаточно наличия
                    # трендового имени (sma_50) в условиях входа.
                    {"type": "compare", "left": "close", "op": ">", "right": {"name": "sma_50", "mult": 0.5}},
                    {"type": "volume", "op": ">=", "ref": "avg_volume_96", "mult": 1.0},
                ],
                "stop_loss_pct": 1.0,
                "take_profit_pct": [10.0],
            },
            "risk_controls": {
                "enabled": True,
                "max_consecutive_losses": 3,
                "pause_candles_after_losses": 96,
                "max_losses_in_window": 5,
                "loss_window_trades": 10,
            },
        }
        stats = self._run_stats(spec)
        self.assertTrue(stats.get("risk_controls_enabled"))
        self.assertGreater(stats.get("cooldown_activations", 0), 0,
                           "пауза должна активироваться после серии убытков")
        self.assertGreater(stats.get("skipped_candles_by_cooldown", 0), 0,
                           "во время паузы свечи должны пропускаться")


if __name__ == "__main__":
    unittest.main(verbosity=2)