"""
Unit-тесты для этапа 6:
резервирование holdout-периода и оценка его качества.

После рефакторинга логика вынесена в модули
optimization.optimizer_data_utils и optimization.optimizer_holdout,
поэтому тесты вызывают их статические методы напрямую.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from optimization.optimizer_data_utils import OptimizerDataUtils
from optimization.optimizer_holdout import OptimizerHoldout
from optimization.optimizer_robustness import OptimizerRobustness

def test_split_optimization_and_holdout_enabled():
    df = pd.DataFrame(
        {"close": range(10000)},
        index=pd.date_range("2025-01-01", periods=10000, freq="15min"),
    )

    optimization_df, holdout_df, meta = OptimizerDataUtils.split_optimization_and_holdout(
        df,
        holdout_ratio=0.15,
        min_holdout_candles=500,
        min_optimization_candles=3000,
    )

    assert meta["enabled"] is True
    assert len(optimization_df) + len(holdout_df) == len(df)
    assert len(holdout_df) >= 500
    assert len(optimization_df) >= 3000
    assert meta["holdout_candles"] == len(holdout_df)

def test_split_optimization_and_holdout_disabled_when_not_enough_data():
    df = pd.DataFrame(
        {"close": range(1200)},
        index=pd.date_range("2025-01-01", periods=1200, freq="15min"),
    )

    optimization_df, holdout_df, meta = OptimizerDataUtils.split_optimization_and_holdout(
        df,
        holdout_ratio=0.15,
        min_holdout_candles=500,
        min_optimization_candles=3000,
    )

    assert meta["enabled"] is False
    assert meta["reason"] == "not_enough_data"
    assert len(optimization_df) == len(df)
    assert len(holdout_df) == 0

def test_assess_holdout_quality_pass():
    holdout_stats = {
        "total_trades": 18,
        "return_pct": 6.2,
        "expectancy_pct": 0.22,
        "profit_factor": 1.35,
        "max_drawdown_pct": 4.1,
    }

    quality = OptimizerHoldout.assess_holdout_quality(holdout_stats, min_holdout_trades=10)

    assert quality["passed"] is True
    assert quality["warnings"] == []

def test_assess_holdout_quality_fail():
    holdout_stats = {
        "total_trades": 6,
        "return_pct": -1.5,
        "expectancy_pct": -0.05,
        "profit_factor": 0.92,
        "max_drawdown_pct": 12.0,
    }

    quality = OptimizerHoldout.assess_holdout_quality(holdout_stats, min_holdout_trades=10)

    assert quality["passed"] is False
    assert "На holdout мало сделок" in quality["warnings"]
    assert "На holdout expectancy <= 0" in quality["warnings"]
    assert "На holdout PF <= 1.05" in quality["warnings"]
    assert "На holdout итоговая доходность <= 0" in quality["warnings"]
    assert "На holdout повышенная просадка" in quality["warnings"]

def test_holdout_quality_and_robustness_can_coexist():
    holdout_stats = {
        "total_trades": 15,
        "return_pct": 4.0,
        "expectancy_pct": 0.12,
        "profit_factor": 1.2,
        "max_drawdown_pct": 5.0,
    }
    quality = OptimizerHoldout.assess_holdout_quality(holdout_stats, min_holdout_trades=10)

    window_robustness = {
        "windows_total": 3,
        "profitable_windows_ratio": 0.67,
        "positive_expectancy_windows_ratio": 0.67,
        "pf_pass_windows_ratio": 0.67,
        "median_window_return_pct": 0.6,
        "worst_window_return_pct": -3.0,
        "stability_score": 0.7,
    }
    robustness = OptimizerRobustness.assess_window_robustness(window_robustness)

    assert quality["passed"] is True
    assert robustness["status"] == "robust"

if __name__ == "__main__":
    test_split_optimization_and_holdout_enabled()
    test_split_optimization_and_holdout_disabled_when_not_enough_data()
    test_assess_holdout_quality_pass()
    test_assess_holdout_quality_fail()
    test_holdout_quality_and_robustness_can_coexist()
    print("✅ Все тесты прошли успешно")
