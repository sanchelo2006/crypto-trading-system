"""
Unit-тесты для этапа 1 оптимизатора:
проверяют расчёт устойчивости стратегии по окнам walk-forward
и сохранение этих метрик в компактной истории.

После рефакторинга методы вынесены в
optimization.optimizer_robustness и optimization.optimizer_summary.
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from optimization.optimizer_robustness import OptimizerRobustness
from optimization.optimizer_summary import OptimizerSummary

def test_compute_window_robustness_empty():
    result = OptimizerRobustness.compute_window_robustness([])

    assert result["windows_total"] == 0
    assert result["profitable_windows"] == 0
    assert result["positive_expectancy_windows"] == 0
    assert result["pf_pass_windows"] == 0
    assert result["median_window_return_pct"] == 0.0
    assert result["return_std"] == 0.0
    assert result["windows"] == []

def test_compute_window_robustness_basic():
    fold_results = [
        {"test_result": {"stats": {
            "return_pct": 10.0,
            "expectancy_pct": 0.4,
            "profit_factor": 1.5,
            "total_trades": 20,
            "max_drawdown_pct": 4.0,
        }}},
        {"test_result": {"stats": {
            "return_pct": -5.0,
            "expectancy_pct": -0.2,
            "profit_factor": 0.8,
            "total_trades": 18,
            "max_drawdown_pct": 6.0,
        }}},
        {"test_result": {"stats": {
            "return_pct": 3.0,
            "expectancy_pct": 0.1,
            "profit_factor": 1.1,
            "total_trades": 16,
            "max_drawdown_pct": 3.0,
        }}},
    ]

    result = OptimizerRobustness.compute_window_robustness(fold_results)

    assert result["windows_total"] == 3
    assert result["profitable_windows"] == 2
    assert result["positive_expectancy_windows"] == 2
    assert result["pf_pass_windows"] == 2
    assert result["profitable_windows_ratio"] == 0.6667
    assert result["positive_expectancy_windows_ratio"] == 0.6667
    assert result["pf_pass_windows_ratio"] == 0.6667
    assert result["worst_window_return_pct"] == -5.0
    assert result["best_window_return_pct"] == 10.0
    assert result["median_window_return_pct"] == 3.0
    assert len(result["windows"]) == 3

def test_compute_window_robustness_pf_threshold_strict():
    fold_results = [
        {"test_result": {"stats": {
            "return_pct": 1.0,
            "expectancy_pct": 0.01,
            "profit_factor": 1.0,
            "total_trades": 10,
            "max_drawdown_pct": 1.0,
        }}},
        {"test_result": {"stats": {
            "return_pct": 2.0,
            "expectancy_pct": 0.02,
            "profit_factor": 1.01,
            "total_trades": 10,
            "max_drawdown_pct": 1.0,
        }}},
    ]

    result = OptimizerRobustness.compute_window_robustness(fold_results)

    assert result["pf_pass_windows"] == 1
    assert result["pf_pass_windows_ratio"] == 0.5
    assert result["profitable_windows"] == 2
    assert result["positive_expectancy_windows"] == 2

def test_compact_history_includes_window_robustness():
    history = [{
        "iteration": 1,
        "test_trades": 20,
        "test_total_pnl_pct": 8.0,
        "test_return_pct": 7.5,
        "test_expectancy_pct": 0.2,
        "test_profit_factor": 1.3,
        "selection_score": 6.2,
        "acceptable": True,
        "changes_summary": "test",
        "window_robustness": {
            "profitable_windows_ratio": 0.75,
            "positive_expectancy_windows_ratio": 0.75,
            "pf_pass_windows_ratio": 0.5,
            "median_window_return_pct": 1.2,
            "worst_window_return_pct": -2.1,
            "return_std": 3.4,
        }
    }]

    compact = OptimizerSummary.compact_history(history)
    row = compact[0]

    assert row["profitable_windows_ratio"] == 0.75
    assert row["positive_expectancy_windows_ratio"] == 0.75
    assert row["pf_pass_windows_ratio"] == 0.5
    assert row["median_window_return_pct"] == 1.2
    assert row["worst_window_return_pct"] == -2.1
    assert row["return_std"] == 3.4

def test_compute_window_diagnostics_stable():
    robust = {
        "windows_total": 4,
        "profitable_windows_ratio": 0.75,
        "positive_expectancy_windows_ratio": 0.75,
        "pf_pass_windows_ratio": 0.75,
        "worst_window_return_pct": -3.0,
        "return_std": 2.5,
    }
    diag = OptimizerRobustness.compute_window_diagnostics(robust)

    assert diag["stability_score"] >= 0.7
    assert isinstance(diag["warnings"], list)
    assert isinstance(diag["recommendations"], list)

def test_compute_window_diagnostics_fragile():
    robust = {
        "windows_total": 4,
        "profitable_windows_ratio": 0.25,
        "positive_expectancy_windows_ratio": 0.25,
        "pf_pass_windows_ratio": 0.25,
        "worst_window_return_pct": -15.0,
        "return_std": 12.0,
    }
    diag = OptimizerRobustness.compute_window_diagnostics(robust)

    assert diag["stability_score"] <= 0.3
    assert len(diag["warnings"]) >= 3
    assert len(diag["recommendations"]) >= 2

def test_assess_window_robustness_robust():
    window_robustness = {
        "windows_total": 3,
        "profitable_windows_ratio": 0.67,
        "positive_expectancy_windows_ratio": 0.67,
        "pf_pass_windows_ratio": 0.67,
        "median_window_return_pct": 0.8,
        "worst_window_return_pct": -3.0,
        "stability_score": 0.72,
    }

    result = OptimizerRobustness.assess_window_robustness(window_robustness)

    assert result["status"] == "robust"
    assert result["passed"] is True
    assert result["fragile"] is False
    assert result["warnings"] == []

def test_assess_window_robustness_fragile():
    window_robustness = {
        "windows_total": 3,
        "profitable_windows_ratio": 0.5,
        "positive_expectancy_windows_ratio": 0.5,
        "pf_pass_windows_ratio": 0.5,
        "median_window_return_pct": -0.1,
        "worst_window_return_pct": -4.0,
        "stability_score": 0.4,
    }

    result = OptimizerRobustness.assess_window_robustness(window_robustness)

    assert result["status"] == "fragile"
    assert result["fragile"] is True
    assert len(result["warnings"]) >= 1

def test_assess_window_robustness_unstable():
    window_robustness = {
        "windows_total": 3,
        "profitable_windows_ratio": 0.33,
        "positive_expectancy_windows_ratio": 0.33,
        "pf_pass_windows_ratio": 0.33,
        "median_window_return_pct": -3.0,
        "worst_window_return_pct": -12.0,
        "stability_score": 0.2,
    }

    result = OptimizerRobustness.assess_window_robustness(window_robustness)

    assert result["status"] == "unstable"
    assert result["passed"] is False
    assert result["fragile"] is False
    assert len(result["warnings"]) >= 3

if __name__ == "__main__":
    test_compute_window_robustness_empty()
    test_compute_window_robustness_basic()
    test_compute_window_robustness_pf_threshold_strict()
    test_compact_history_includes_window_robustness()
    test_compute_window_diagnostics_stable()
    test_compute_window_diagnostics_fragile()
    test_assess_window_robustness_robust()
    test_assess_window_robustness_fragile()
    test_assess_window_robustness_unstable()

    print("✅ Все тесты прошли успешно")
