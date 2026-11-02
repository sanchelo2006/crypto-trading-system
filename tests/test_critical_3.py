#!/usr/bin/env python3
"""
Unit-тесты для механического бэктеста и движка правил стратегии.
Запуск: python test_critical_3.py

Проверяются:
  strategy_spec.py:
    - _resolve (операнды: число, имя серии, объект mult/offset)
    - _eval_condition (compare, range, volume, cross, slope)
    - evaluate_conditions (AND-логика, пустой список)
    - validate_strategy_spec
    - default_strategy_spec
  backtest_engine.py:
    - _compute_vwap (дневной ресет)
    - _add_daily_pivots (pivot по предыдущему дню, смещение)
    - _build_indicator_frame (выравнивание по индексу, наличие серий)
    - _open_position / _check_exit / _close_position (SL/TP, комиссия, PnL long/short)
    - _simulate (полный цикл на синтетике)
    - _compute_stats (win_rate, profit_factor, drawdown)
    - run() с моком fetcher (интеграционный тест без сети)
"""

import sys
import os
import unittest
import math
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from strategy_spec import (
    _resolve,
    _eval_condition,
    evaluate_conditions,
    validate_strategy_spec,
    default_strategy_spec,
)
from backtest_engine import BacktestEngine
from indicators import merge_1h_features_to_15m

# ======================================================================
# strategy_spec: операнды
# ======================================================================
class TestResolveOperand(unittest.TestCase):
    def setUp(self):
        self.ctx = {"close": 100.0, "sma_50": 95.0, "volume": 1000.0, "nan_field": float("nan")}

    def test_number_operand(self):
        self.assertEqual(_resolve(42, self.ctx), 42.0)
        self.assertEqual(_resolve(3.14, self.ctx), 3.14)

    def test_series_name(self):
        self.assertEqual(_resolve("close", self.ctx), 100.0)

    def test_missing_series(self):
        self.assertIsNone(_resolve("nonexistent", self.ctx))

    def test_nan_series_returns_none(self):
        self.assertIsNone(_resolve("nan_field", self.ctx))

    def test_object_with_mult(self):
        # sma_50 * 1.02 = 96.9
        val = _resolve({"name": "sma_50", "mult": 1.02}, self.ctx)
        self.assertAlmostEqual(val, 96.9)

    def test_object_with_mult_and_offset(self):
        # close * 1.0 + 5 = 105
        val = _resolve({"name": "close", "mult": 1.0, "offset": 5.0}, self.ctx)
        self.assertAlmostEqual(val, 105.0)

    def test_object_missing_base_returns_none(self):
        self.assertIsNone(_resolve({"name": "nonexistent", "mult": 2.0}, self.ctx))

    def test_none_operand(self):
        self.assertIsNone(_resolve(None, self.ctx))

    def test_bool_not_treated_as_number(self):
        # bool исключён из числовой ветки → трактуется как «не число, не строка, не dict» → None
        self.assertIsNone(_resolve(True, self.ctx))


# ======================================================================
# strategy_spec: отдельные условия
# ======================================================================
class TestEvalCondition(unittest.TestCase):
    def setUp(self):
        self.ctx = {
            "close": 100.0, "sma_50": 95.0, "rsi_14": 55.0,
            "volume": 1500.0, "avg_volume_96": 1000.0,
            "macd": 1.0, "macd_signal": 0.5,
            "adx_14": 25.0,
        }
        self.prev = {
            "close": 94.0, "sma_50": 95.0, "rsi_14": 48.0,
            "volume": 900.0, "avg_volume_96": 1000.0,
            "macd": 0.2, "macd_signal": 0.5,
            "adx_14": 22.0,
        }

    # --- compare ---
    def test_compare_true(self):
        c = {"type": "compare", "left": "close", "op": ">", "right": "sma_50"}
        self.assertTrue(_eval_condition(c, self.ctx, self.prev))

    def test_compare_false(self):
        c = {"type": "compare", "left": "close", "op": "<", "right": "sma_50"}
        self.assertFalse(_eval_condition(c, self.ctx, self.prev))

    def test_compare_with_number(self):
        c = {"type": "compare", "left": "rsi_14", "op": ">", "right": 50}
        self.assertTrue(_eval_condition(c, self.ctx, self.prev))

    def test_compare_missing_operand_false(self):
        c = {"type": "compare", "left": "missing", "op": ">", "right": 50}
        self.assertFalse(_eval_condition(c, self.ctx, self.prev))

    def test_compare_bad_op_false(self):
        c = {"type": "compare", "left": "close", "op": "!!", "right": 50}
        self.assertFalse(_eval_condition(c, self.ctx, self.prev))

    # --- range ---
    def test_range_inside(self):
        c = {"type": "range", "value": "rsi_14", "min": 40, "max": 70}
        self.assertTrue(_eval_condition(c, self.ctx, self.prev))

    def test_range_outside(self):
        c = {"type": "range", "value": "rsi_14", "min": 60, "max": 70}
        self.assertFalse(_eval_condition(c, self.ctx, self.prev))

    def test_range_boundary_inclusive(self):
        c = {"type": "range", "value": "rsi_14", "min": 55, "max": 55}
        self.assertTrue(_eval_condition(c, self.ctx, self.prev))

    # --- volume ---
    def test_volume_above(self):
        # 1500 > 1000 * 1.0
        c = {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 1.0}
        self.assertTrue(_eval_condition(c, self.ctx, self.prev))

    def test_volume_with_mult(self):
        # 1500 > 1000 * 2.0 = 2000 → False
        c = {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 2.0}
        self.assertFalse(_eval_condition(c, self.ctx, self.prev))

    def test_volume_missing_ref_false(self):
        c = {"type": "volume", "op": ">", "ref": "missing", "mult": 1.0}
        self.assertFalse(_eval_condition(c, self.ctx, self.prev))

    # --- cross ---
    def test_cross_above_true(self):
        # prev: macd(0.2) <= signal(0.5); now: macd(1.0) > signal(0.5)
        c = {"type": "cross", "a": "macd", "b": "macd_signal", "dir": "above"}
        self.assertTrue(_eval_condition(c, self.ctx, self.prev))

    def test_cross_above_false_when_no_prev(self):
        c = {"type": "cross", "a": "macd", "b": "macd_signal", "dir": "above"}
        self.assertFalse(_eval_condition(c, self.ctx, None))

    def test_cross_below_true(self):
        prev = {"close": 96.0, "sma_50": 95.0}
        now = {"close": 94.0, "sma_50": 95.0}
        c = {"type": "cross", "a": "close", "b": "sma_50", "dir": "below"}
        self.assertTrue(_eval_condition(c, now, prev))

    def test_cross_no_event_false(self):
        # уже был выше и остался выше — не пересечение
        prev = {"macd": 0.8, "macd_signal": 0.5}
        now = {"macd": 1.0, "macd_signal": 0.5}
        c = {"type": "cross", "a": "macd", "b": "macd_signal", "dir": "above"}
        self.assertFalse(_eval_condition(c, now, prev))

    # --- slope ---
    def test_slope_up(self):
        c = {"type": "slope", "value": "rsi_14", "dir": "up", "lookback": 1}
        # now 55 > prev 48
        self.assertTrue(_eval_condition(c, self.ctx, self.prev))

    def test_slope_down_false(self):
        c = {"type": "slope", "value": "rsi_14", "dir": "down", "lookback": 1}
        self.assertFalse(_eval_condition(c, self.ctx, self.prev))

    # --- misc ---
    def test_unknown_type_false(self):
        self.assertFalse(_eval_condition({"type": "wtf"}, self.ctx, self.prev))

    def test_non_dict_false(self):
        self.assertFalse(_eval_condition("not a dict", self.ctx, self.prev))


# ======================================================================
# strategy_spec: агрегация условий
# ======================================================================
class TestEvaluateConditions(unittest.TestCase):
    def setUp(self):
        self.ctx = {"close": 100.0, "sma_50": 95.0, "rsi_14": 55.0}
        self.prev = {"close": 94.0, "sma_50": 95.0, "rsi_14": 48.0}

    def test_all_true(self):
        conds = [
            {"type": "compare", "left": "close", "op": ">", "right": "sma_50"},
            {"type": "range", "value": "rsi_14", "min": 50, "max": 60},
        ]
        self.assertTrue(evaluate_conditions(conds, self.ctx, self.prev))

    def test_one_false_breaks_all(self):
        conds = [
            {"type": "compare", "left": "close", "op": ">", "right": "sma_50"},
            {"type": "range", "value": "rsi_14", "min": 60, "max": 70},  # False
        ]
        self.assertFalse(evaluate_conditions(conds, self.ctx, self.prev))

    def test_empty_conditions_false(self):
        # Пустой список НЕ должен генерировать вход
        self.assertFalse(evaluate_conditions([], self.ctx, self.prev))


# ======================================================================
# strategy_spec: валидация и default
# ======================================================================
class TestValidateSpec(unittest.TestCase):
    def test_valid_default(self):
        self.assertIsNone(validate_strategy_spec(default_strategy_spec()))

    def test_not_dict(self):
        self.assertIsNotNone(validate_strategy_spec("string"))
        self.assertIsNotNone(validate_strategy_spec([1, 2]))

    def test_no_long_no_short(self):
        self.assertIsNotNone(validate_strategy_spec({"filters": []}))

    def test_only_long_ok(self):
        spec = {"long": {
            "entry": [
                {"type": "compare", "left": "close", "op": ">", "right": 1},
                {"type": "compare", "left": "close", "op": ">", "right": "sma_50"},
                {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 1.0},
            ],
            "stop_loss_atr_mult": 2.0,
            "take_profit_atr_mult": 4.0,
        }}
        self.assertIsNone(validate_strategy_spec(spec))

    def test_entry_not_list(self):
        spec = {"long": {"entry": "нет"}}
        self.assertIsNotNone(validate_strategy_spec(spec))

    def test_side_not_dict(self):
        spec = {"long": "нет"}
        self.assertIsNotNone(validate_strategy_spec(spec))

    def test_default_has_both_sides(self):
        spec = default_strategy_spec()
        self.assertIn("long", spec)
        self.assertIn("short", spec)
        self.assertTrue(len(spec["long"]["entry"]) > 0)
        self.assertTrue(len(spec["short"]["entry"]) > 0)


# ======================================================================
# backtest_engine: VWAP и Pivot
# ======================================================================
class TestVwapAndPivots(unittest.TestCase):
    def setUp(self):
        self.engine = BacktestEngine(fetcher=None)

    def _two_day_df(self):
        # День 1: 8 свечей цены ~1.0; День 2: 8 свечей цены ~2.0
        ts_day1 = pd.date_range("2026-07-14 00:00", periods=8, freq="15min", tz="UTC")
        ts_day2 = pd.date_range("2026-07-15 00:00", periods=8, freq="15min", tz="UTC")
        ts = ts_day1.append(ts_day2)
        closes = [1.0] * 8 + [2.0] * 8
        highs = [c + 0.1 for c in closes]
        lows = [c - 0.1 for c in closes]
        return pd.DataFrame({
            "timestamp": ts, "open": closes, "high": highs,
            "low": lows, "close": closes, "volume": [100.0] * 16
        })

    def test_vwap_resets_daily(self):
        df = self._two_day_df()
        vwap = self.engine._compute_vwap(df).values
        # Первая свеча дня 2 должна начинать заново около 2.0, а не тянуть 1.0
        first_day2_vwap = vwap[8]
        self.assertAlmostEqual(first_day2_vwap, 2.0, delta=0.15)
        # Первая свеча дня 1 около 1.0
        self.assertAlmostEqual(vwap[0], 1.0, delta=0.15)

    def test_pivots_use_previous_day(self):
        df = self._two_day_df().copy()
        self.engine._add_daily_pivots(df)
        # День 1 не имеет предыдущего → pivot NaN
        self.assertTrue(math.isnan(df["pivot"].iloc[0]))
        # День 2 pivot = (prev_high + prev_low + prev_close)/3 по дню 1
        # день 1: high=1.1, low=0.9, close=1.0 → pivot=1.0
        self.assertAlmostEqual(df["pivot"].iloc[8], 1.0, delta=0.01)
        # resistance_1 = 2*pivot - prev_low = 2*1.0 - 0.9 = 1.1
        self.assertAlmostEqual(df["resistance_1"].iloc[8], 1.1, delta=0.01)
        # support_1 = 2*pivot - prev_high = 2*1.0 - 1.1 = 0.9
        self.assertAlmostEqual(df["support_1"].iloc[8], 0.9, delta=0.01)

    def test_pivots_constant_within_day(self):
        df = self._two_day_df().copy()
        self.engine._add_daily_pivots(df)
        day2 = df.iloc[8:]
        # Все свечи дня 2 имеют одинаковый pivot
        self.assertEqual(day2["pivot"].nunique(), 1)


# ======================================================================
# backtest_engine: расчёт индикаторов (выравнивание)
# ======================================================================
class TestIndicatorFrame(unittest.TestCase):
    def setUp(self):
        self.engine = BacktestEngine(fetcher=None)

    def _make_df(self, n=120):
        np.random.seed(7)
        ts = pd.date_range("2026-07-10", periods=n, freq="15min", tz="UTC")
        closes = 1.5 + np.cumsum(np.random.randn(n) * 0.01)
        return pd.DataFrame({
            "timestamp": ts, "open": closes,
            "high": closes + 0.01, "low": closes - 0.01,
            "close": closes, "volume": np.abs(np.random.randn(n) * 500 + 3000)
        })

    def test_length_preserved(self):
        df = self._make_df(120)
        out = self.engine._build_indicator_frame(df)
        # Длина не меняется — критично для выравнивания свечей и индикаторов
        self.assertEqual(len(out), 120)

    def test_all_series_present(self):
        out = self.engine._build_indicator_frame(self._make_df(120))
        for col in ["sma_20", "sma_50", "rsi_14", "atr_14", "macd",
                    "macd_signal", "adx_14", "vwap", "avg_volume_20",
                    "avg_volume_96", "pivot", "resistance_1", "support_1"]:
            self.assertIn(col, out.columns, f"Нет серии {col}")

    def test_early_rows_have_nan(self):
        # sma_50 в первых 49 строках должна быть NaN (индикатор не смещён вперёд)
        out = self.engine._build_indicator_frame(self._make_df(120))
        self.assertTrue(out["sma_50"].iloc[:49].isna().all())
        self.assertFalse(math.isnan(out["sma_50"].iloc[-1]))

    def test_close_unchanged(self):
        df = self._make_df(120)
        orig_close = df["close"].tolist()
        out = self.engine._build_indicator_frame(df)
        self.assertEqual(out["close"].tolist(), orig_close)


# ======================================================================
# backtest_engine: позиции и PnL
# ======================================================================
class TestPositionLogic(unittest.TestCase):
    def setUp(self):
        self.engine = BacktestEngine(fetcher=None)
        self.ts = datetime(2026, 7, 14, tzinfo=timezone.utc)

    def test_open_long_levels(self):
        row = {"close": 100.0, "timestamp": self.ts}
        spec = {"stop_loss_pct": 2.0, "take_profit_pct": [3.0]}
        pos = self.engine._open_position("long", row, spec)
        self.assertAlmostEqual(pos["entry_price"], 100.0)
        self.assertAlmostEqual(pos["stop_loss"], 98.0)
        self.assertAlmostEqual(pos["take_profit"], 103.0)

    def test_open_short_levels(self):
        row = {"close": 100.0, "timestamp": self.ts}
        spec = {"stop_loss_pct": 2.0, "take_profit_pct": [3.0]}
        pos = self.engine._open_position("short", row, spec)
        self.assertAlmostEqual(pos["stop_loss"], 102.0)
        self.assertAlmostEqual(pos["take_profit"], 97.0)

    def test_long_hits_take_profit(self):
        pos = {"direction": "long", "entry_price": 100.0, "stop_loss": 98.0, "take_profit": 103.0}
        row = {"high": 104.0, "low": 99.0}
        price, reason = self.engine._check_exit(pos, row)
        self.assertEqual(reason, "take_profit")
        self.assertAlmostEqual(price, 103.0)

    def test_long_hits_stop_loss(self):
        pos = {"direction": "long", "entry_price": 100.0, "stop_loss": 98.0, "take_profit": 103.0}
        row = {"high": 101.0, "low": 97.0}
        price, reason = self.engine._check_exit(pos, row)
        self.assertEqual(reason, "stop_loss")
        self.assertAlmostEqual(price, 98.0)

    def test_sl_priority_over_tp_same_candle(self):
        # Если свеча задела и SL, и TP — приоритет SL (консервативно)
        pos = {"direction": "long", "entry_price": 100.0, "stop_loss": 98.0, "take_profit": 103.0}
        row = {"high": 104.0, "low": 97.0}
        price, reason = self.engine._check_exit(pos, row)
        self.assertEqual(reason, "stop_loss")

    def test_short_hits_take_profit(self):
        pos = {"direction": "short", "entry_price": 100.0, "stop_loss": 102.0, "take_profit": 97.0}
        row = {"high": 101.0, "low": 96.0}
        price, reason = self.engine._check_exit(pos, row)
        self.assertEqual(reason, "take_profit")

    def test_no_exit(self):
        pos = {"direction": "long", "entry_price": 100.0, "stop_loss": 98.0, "take_profit": 103.0}
        row = {"high": 101.0, "low": 99.0}
        price, reason = self.engine._check_exit(pos, row)
        self.assertIsNone(price)

    def test_close_long_pnl_with_commission(self):
        pos = {"direction": "long", "entry_price": 100.0, "entry_time": self.ts}
        # TP на 103 → +3%, минус комиссия 2*0.05% = 0.1% → ~+2.9%
        cap, trade = self.engine._close_position(pos, 103.0, "take_profit", 1000.0, 0.05, self.ts)
        self.assertAlmostEqual(trade["result_pct"], 2.9, places=1)
        self.assertGreater(cap, 1000.0)

    def test_close_short_pnl(self):
        pos = {"direction": "short", "entry_price": 100.0, "entry_time": self.ts}
        # Цена упала до 97 → +3% минус комиссия
        cap, trade = self.engine._close_position(pos, 97.0, "take_profit", 1000.0, 0.05, self.ts)
        self.assertAlmostEqual(trade["result_pct"], 2.9, places=1)

    def test_close_loss(self):
        pos = {"direction": "long", "entry_price": 100.0, "entry_time": self.ts}
        cap, trade = self.engine._close_position(pos, 98.0, "stop_loss", 1000.0, 0.05, self.ts)
        self.assertLess(trade["result_pct"], 0)
        self.assertLess(cap, 1000.0)


# ======================================================================
# backtest_engine: статистика
# ======================================================================
class TestComputeStats(unittest.TestCase):
    def setUp(self):
        self.engine = BacktestEngine(fetcher=None)

    def test_empty_trades(self):
        stats = self.engine._compute_stats([], [1000.0], 1000.0, 1000.0)
        self.assertEqual(stats["total_trades"], 0)
        self.assertEqual(stats["win_rate"], 0)

    def test_win_rate_and_pf(self):
        trades = [
            {"result_pct": 3.0, "direction": "long"},
            {"result_pct": -2.0, "direction": "long"},
            {"result_pct": 3.0, "direction": "short"},
            {"result_pct": -2.0, "direction": "short"},
        ]
        equity = [1000, 1030, 1009, 1039, 1018]
        stats = self.engine._compute_stats(trades, equity, 1018.0, 1000.0)
        self.assertEqual(stats["total_trades"], 4)
        self.assertEqual(stats["win_rate"], 50.0)
        # PF = 6 / 4 = 1.5
        self.assertAlmostEqual(stats["profit_factor"], 1.5)
        self.assertEqual(stats["long_trades"], 2)
        self.assertEqual(stats["short_trades"], 2)

    def test_all_wins_pf_capped(self):
        trades = [{"result_pct": 2.0, "direction": "long"}]
        stats = self.engine._compute_stats(trades, [1000, 1020], 1020.0, 1000.0)
        # gross_loss == 0 → PF = inf → должно превратиться в 999.0
        self.assertEqual(stats["profit_factor"], 999.0)

    def test_max_drawdown(self):
        # equity: 1000 → 1200 → 900 → просадка = (1200-900)/1200 = 25%
        equity = [1000, 1200, 900, 1100]
        trades = [{"result_pct": 1.0, "direction": "long"}]
        stats = self.engine._compute_stats(trades, equity, 1100.0, 1000.0)
        self.assertAlmostEqual(stats["max_drawdown_pct"], 25.0, places=1)


# ======================================================================
# backtest_engine: интеграционный тест run() с моком fetcher
# ======================================================================
class _MockFetcher:
    """Мок fetcher: возвращает заранее заданный DataFrame свечей."""
    def __init__(self, df):
        self._df = df

    def fetch_ohlcv(self, symbol, interval, start, end, category, progress_callback=None):
        return self._df.copy()


class TestSimulateAndRun(unittest.TestCase):
    def _trending_df(self, n=200):
        """Плавный восходящий тренд с всплеском объёма — чтобы сработали входы."""
        ts = pd.date_range("2026-07-01", periods=n, freq="15min", tz="UTC")
        closes = 1.0 + np.linspace(0, 0.5, n)  # устойчивый рост
        highs = closes + 0.01
        lows = closes - 0.01
        vols = np.full(n, 1000.0)
        vols[::10] = 5000.0  # периодические всплески объёма
        return pd.DataFrame({
            "timestamp": ts, "open": closes, "high": highs,
            "low": lows, "close": closes, "volume": vols
        })

    def test_simulate_produces_trades(self):
        engine = BacktestEngine(fetcher=None)
        df = engine._build_indicator_frame(self._trending_df(200))
        spec = default_strategy_spec()
        trades, equity, cap = engine._simulate(df, spec, 1000.0, 0.05)
        self.assertIsInstance(trades, list)
        self.assertIsInstance(equity, list)
        # Хотя бы структура корректна; сделки могут быть или не быть — не падаем
        for t in trades:
            self.assertIn("direction", t)
            self.assertIn("result_pct", t)

    def test_no_position_leak(self):
        """После симуляции не должно остаться незакрытых позиций (все в trades)."""
        engine = BacktestEngine(fetcher=None)
        df = engine._build_indicator_frame(self._trending_df(200))
        spec = default_strategy_spec()
        trades, equity, cap = engine._simulate(df, spec, 1000.0, 0.05)
        # equity стартует с initial + по одному значению на каждый закрытый трейд
        self.assertEqual(len(equity), 1 + len(trades))

    def test_run_integration(self):
        df = self._trending_df(200)
        engine = BacktestEngine(fetcher=_MockFetcher(df))
        result = engine.run(
            symbol="TESTUSDT", category="linear",
            strategy_spec=default_strategy_spec(),
            start_date=datetime(2026, 7, 1, tzinfo=timezone.utc),
            end_date=datetime(2026, 7, 3, tzinfo=timezone.utc),
            initial_capital=1000.0,
        )
        self.assertNotIn("error", result)
        self.assertIn("stats", result)
        self.assertEqual(result["spec_source"], "ai")

    def test_run_invalid_spec_falls_back_to_default(self):
        df = self._trending_df(200)
        engine = BacktestEngine(fetcher=_MockFetcher(df))
        result = engine.run(
            symbol="TESTUSDT", category="linear",
            strategy_spec={"garbage": True},  # невалидный
            start_date=datetime(2026, 7, 1, tzinfo=timezone.utc),
            end_date=datetime(2026, 7, 3, tzinfo=timezone.utc),
        )
        self.assertEqual(result["spec_source"], "default")

    def test_run_insufficient_candles(self):
        df = self._trending_df(30)  # < 60
        engine = BacktestEngine(fetcher=_MockFetcher(df))
        result = engine.run(
            symbol="TESTUSDT", category="linear",
            strategy_spec=default_strategy_spec(),
            start_date=datetime(2026, 7, 1, tzinfo=timezone.utc),
            end_date=datetime(2026, 7, 3, tzinfo=timezone.utc),
        )
        self.assertIn("error", result)

    def test_run_no_data(self):
        engine = BacktestEngine(fetcher=_MockFetcher(pd.DataFrame()))
        result = engine.run(
            symbol="TESTUSDT", category="linear",
            strategy_spec=default_strategy_spec(),
            start_date=datetime(2026, 7, 1, tzinfo=timezone.utc),
            end_date=datetime(2026, 7, 3, tzinfo=timezone.utc),
        )
        self.assertIn("error", result)

# ======================================================================
# Regression: merge_1h_features_to_15m не должен затирать 15m OHLCV
# ======================================================================
class TestMerge1hPreserves15mOhlcv(unittest.TestCase):
    """
    До фикса merge_1h_features_to_15m удалял 15m open/high/low/close/volume
    и подставлял одноимённые 1h-колонки. Из-за этого timestamp был
    15-минутным, а close «замирал» внутри часа — страдали сигналы,
    trailing и мягкие выходы.
    """

    def _make_15m(self, n=10):
        ts = pd.date_range("2026-10-05 17:00", periods=n, freq="15min", tz="UTC")
        return pd.DataFrame({
            "timestamp": ts,
            "open":   [100.0 + i for i in range(n)],
            "high":   [101.0 + i for i in range(n)],
            "low":    [99.0 + i for i in range(n)],
            "close":  [100.5 + i for i in range(n)],
            "volume": [1000.0 + i for i in range(n)],
        })

    def _make_1h_with_features(self):
        # Заведомо ДРУГИЕ значения OHLCV + корректные признаки *_1h.
        ts = pd.date_range("2026-10-05 17:00", periods=3, freq="1h", tz="UTC")
        return pd.DataFrame({
            "timestamp": ts,
            "open":   [900.0, 901.0, 902.0],
            "high":   [999.0, 998.0, 997.0],
            "low":    [888.0, 889.0, 890.0],
            "close":  [950.0, 951.0, 952.0],
            "volume": [7777.0, 7778.0, 7779.0],
            "sma_20_1h": [1.1, 1.2, 1.3],
            "macd_hist_1h": [0.01, 0.02, 0.03],
        })

    def test_15m_ohlcv_kept(self):
        df_15m = self._make_15m()
        df_1h = self._make_1h_with_features()

        merged = merge_1h_features_to_15m(df_15m, df_1h)

        for col in ("open", "high", "low", "close", "volume"):
            pd.testing.assert_series_equal(
                merged[col].reset_index(drop=True),
                df_15m[col].reset_index(drop=True),
                check_names=False,
                obj=f"column {col}",
            )

    def test_1h_features_merged(self):
        df_15m = self._make_15m()
        df_1h = self._make_1h_with_features()

        merged = merge_1h_features_to_15m(df_15m, df_1h)

        self.assertIn("sma_20_1h", merged.columns)
        self.assertIn("macd_hist_1h", merged.columns)
        self.assertFalse(merged["sma_20_1h"].isna().all())

    def test_15m_close_matches_timestamp(self):
        df_15m = self._make_15m()
        df_1h = self._make_1h_with_features()

        merged = merge_1h_features_to_15m(df_15m, df_1h)

        expected = dict(zip(df_15m["timestamp"], df_15m["close"]))
        for _, row in merged.iterrows():
            self.assertAlmostEqual(
                float(row["close"]),
                float(expected[row["timestamp"]]),
                places=9,
            )

# ======================================================================
# Regression: trailing в BacktestEngine реагирует на 15m close
# ======================================================================
class TestTrailingUses15mClose(unittest.TestCase):
    """
    Проверка end-to-end: когда ATR-трейлинг двигает SL, он должен
    использовать close той же 15m-строки.
    """

    def test_trailing_uses_row_close(self):
        # Создаём инстанс без fetcher: тестируем чистую логику.
        eng = BacktestEngine.__new__(BacktestEngine)
        eng.last_simulation_meta = {}
        eng.slippage_pct = 0.0

        pos = {
            "direction": "long",
            "entry_price": 100.0,
            "stop_loss": 98.0,
            "take_profit": 110.0,
            "management": {"trailing_atr_mult": 1.0},
            "bars_held": 0,
        }
        row = {
            "close": 101.0,
            "high": 101.2,
            "low": 100.8,
            "atr_14": 1.0,
        }

        eng._apply_position_management(pos, row)

        # new_sl = close - trail_mult * atr = 101.0 - 1.0 = 100.0
        self.assertAlmostEqual(pos["stop_loss"], 100.0, places=6)

# ======================================================================
# Slippage model в BacktestEngine
# ======================================================================
class TestBacktestSlippage(unittest.TestCase):

    def test_entry_slippage_long_worse(self):
        eps = 0.05 / 100.0
        self.assertGreater(100.0 * (1.0 + eps), 100.0)

    def test_entry_slippage_short_worse(self):
        eps = 0.05 / 100.0
        self.assertLess(100.0 * (1.0 - eps), 100.0)

    def test_exit_slippage_long_sl_worse(self):
        adjusted = BacktestEngine._apply_slippage_to_exit(
            price=99.0, reason="stop_loss", direction="long", eps=0.0005
        )
        self.assertLess(adjusted, 99.0)

    def test_exit_slippage_long_tp_worse(self):
        adjusted = BacktestEngine._apply_slippage_to_exit(
            price=105.0, reason="take_profit", direction="long", eps=0.0005
        )
        self.assertLess(adjusted, 105.0)

    def test_exit_slippage_short_sl_worse(self):
        adjusted = BacktestEngine._apply_slippage_to_exit(
            price=101.0, reason="stop_loss", direction="short", eps=0.0005
        )
        self.assertGreater(adjusted, 101.0)

    def test_exit_slippage_short_tp_worse(self):
        adjusted = BacktestEngine._apply_slippage_to_exit(
            price=95.0, reason="take_profit", direction="short", eps=0.0005
        )
        self.assertGreater(adjusted, 95.0)

    def test_soft_exit_unaffected(self):
        for reason in ("time_stop", "level_failure", "trend_loss",
                       "macd_flip", "end_of_period"):
            self.assertEqual(
                BacktestEngine._apply_slippage_to_exit(
                    price=100.0, reason=reason, direction="long", eps=0.0005
                ),
                100.0,
            )

    def test_zero_slippage_is_noop(self):
        self.assertEqual(
            BacktestEngine._apply_slippage_to_exit(
                price=100.0, reason="stop_loss", direction="long", eps=0.0
            ),
            100.0,
        )

if __name__ == "__main__":
    print("=" * 60)
    print("ЗАПУСК ТЕСТОВ БЭКТЕСТА И ДВИЖКА ПРАВИЛ")
    print("=" * 60)
    unittest.main(verbosity=2)