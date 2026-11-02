#!/usr/bin/env python3
"""Unit-тест синхронизации pivot: предыдущий день, как в backtest_engine."""
import unittest
import sys
import os
import numpy as np
import pandas as pd

from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from indicators import calculate_indicators


class TestPivotPreviousDay(unittest.TestCase):
    def test_pivot_uses_previous_day_not_window_max(self):
        # 1h свечи за 2 дня
        ts = pd.date_range("2026-07-14 00:00", periods=48, freq="1h", tz="UTC")
        day1 = ts < pd.Timestamp("2026-07-15", tz="UTC")
        df_1h = pd.DataFrame({
            "timestamp": ts,
            "open": np.where(day1, 94.0, 107.0),
            "high": np.where(day1, 100.0, 110.0),
            "low": np.where(day1, 90.0, 100.0),
            "close": np.where(day1, 95.0, 108.0),
            "volume": np.full(len(ts), 1000.0),
        })

        # 15m свечи только за последний день
        ts15 = pd.date_range("2026-07-15 00:00", periods=32, freq="15min", tz="UTC")
        df_15m = pd.DataFrame({
            "timestamp": ts15,
            "open": 107.0, "high": 108.0, "low": 106.0, "close": 108.0,
            "volume": np.full(len(ts15), 500.0),
        })

        result = calculate_indicators(df_15m, df_1h_for_pivot=df_1h, num_latest=30)

        # День 1: high=100, low=90, close=95
        self.assertAlmostEqual(result["pivot"], 95.0, places=4)
        self.assertAlmostEqual(result["resistance_1"], 100.0, places=4)
        self.assertAlmostEqual(result["support_1"], 90.0, places=4)

        # Это НЕ расчёт по дню 2 (pivot был бы 106)
        self.assertNotAlmostEqual(result["pivot"], 106.0, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)