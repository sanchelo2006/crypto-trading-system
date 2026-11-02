"""
Unit-тесты для этапа 3:
мягкий штраф к selection_score за неустойчивость по окнам.

После рефакторинга логика вынесена в optimization.optimizer_robustness.
"""
import sys
import os

from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from optimization.optimizer_robustness import OptimizerRobustness

def test_window_stability_penalty_zero_for_strong_robustness():
    robust = {
        "windows_total": 3,
        "profitable_windows_ratio": 1.0,
        "positive_expectancy_windows_ratio": 1.0,
        "pf_pass_windows_ratio": 1.0,
        "worst_window_return_pct": 0.8,
        "return_std": 0.5,
    }

    penalty, reasons = OptimizerRobustness.window_stability_penalty(robust)

    assert penalty == 0.0
    assert reasons == []

def test_window_stability_penalty_positive_for_fragile_robustness():
    robust = {
        "windows_total": 3,
        "profitable_windows_ratio": 0.33,
        "positive_expectancy_windows_ratio": 0.33,
        "pf_pass_windows_ratio": 0.33,
        "worst_window_return_pct": -8.0,
        "return_std": 6.5,
    }

    penalty, reasons = OptimizerRobustness.window_stability_penalty(robust)

    assert penalty > 0.0
    assert len(reasons) >= 3

def test_window_stability_penalty_capped():
    robust = {
        "windows_total": 5,
        "profitable_windows_ratio": 0.0,
        "positive_expectancy_windows_ratio": 0.0,
        "pf_pass_windows_ratio": 0.0,
        "worst_window_return_pct": -20.0,
        "return_std": 20.0,
    }

    penalty, reasons = OptimizerRobustness.window_stability_penalty(robust)

    assert penalty == 3.0
    assert len(reasons) >= 4

def test_window_penalty_does_not_replace_robustness_assessment():
    robust = {
        "windows_total": 3,
        "profitable_windows_ratio": 0.67,
        "positive_expectancy_windows_ratio": 0.67,
        "pf_pass_windows_ratio": 0.67,
        "worst_window_return_pct": -3.0,
        "return_std": 1.0,
    }

    penalty, reasons = OptimizerRobustness.window_stability_penalty(robust)
    assessment = OptimizerRobustness.assess_window_robustness({
        **robust,
        "median_window_return_pct": 0.5,
        "stability_score": 0.7,
    })

    assert penalty >= 0.0
    assert assessment["status"] in ("robust", "fragile")

if __name__ == "__main__":
    test_window_stability_penalty_zero_for_strong_robustness()
    test_window_stability_penalty_positive_for_fragile_robustness()
    test_window_stability_penalty_capped()
    test_window_penalty_does_not_replace_robustness_assessment()
    print("✅ Все тесты прошли успешно")
