"""
Unit-тесты для этапа 4:
диагностика score breakdown без изменения логики отбора.
"""
import sys
import os

from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from optimization.backtest_optimizer import BacktestOptimizer


class DummyEngine:
    pass


class DummyAI:
    pass


class DummyMemory:
    pass


def _make_optimizer():
    return BacktestOptimizer(DummyEngine(), DummyAI(), DummyMemory())


def test_build_score_breakdown_contains_main_fields():
    opt = _make_optimizer()

    train_stats = {
        "total_trades": 40,
        "return_pct": 12.0,
        "expectancy_pct": 0.30,
        "profit_factor": 2.0,
        "max_drawdown_pct": 4.0,
    }
    test_stats = {
        "total_trades": 12,
        "return_pct": 3.5,
        "expectancy_pct": 0.22,
        "profit_factor": 1.5,
        "max_drawdown_pct": 2.5,
    }
    robust = {
        "windows_total": 3,
        "profitable_windows_ratio": 0.67,
        "positive_expectancy_windows_ratio": 0.67,
        "pf_pass_windows_ratio": 0.67,
        "median_window_return_pct": 1.0,
        "worst_window_return_pct": -0.5,
        "return_std": 0.8,
    }

    breakdown = opt._build_score_breakdown(
        train_stats=train_stats,
        test_stats=test_stats,
        train_trades=[],
        test_trades=[],
        raw_selection_score=-1.25,
        window_penalty=0.4,
        final_selection_score=-1.65,
        acceptable=True,
        window_robustness=robust,
    )

    assert breakdown["acceptable"] is True
    assert breakdown["raw_selection_score"] == -1.25
    assert breakdown["window_stability_penalty"] == 0.4
    assert breakdown["final_selection_score"] == -1.65
    assert breakdown["train_total_trades"] == 40
    assert breakdown["test_total_trades"] == 12
    assert breakdown["test_return_pct"] == 3.5
    assert breakdown["test_expectancy_pct"] == 0.22
    assert breakdown["test_profit_factor"] == 1.5
    assert breakdown["windows_total"] == 3


def test_build_score_breakdown_gap_metrics():
    opt = _make_optimizer()

    train_stats = {
        "total_trades": 50,
        "return_pct": 15.0,
        "expectancy_pct": 0.40,
        "profit_factor": 2.3,
        "max_drawdown_pct": 5.0,
    }
    test_stats = {
        "total_trades": 10,
        "return_pct": 4.0,
        "expectancy_pct": 0.10,
        "profit_factor": 1.2,
        "max_drawdown_pct": 3.0,
    }
    robust = {
        "windows_total": 3,
        "profitable_windows_ratio": 1.0,
        "positive_expectancy_windows_ratio": 1.0,
        "pf_pass_windows_ratio": 1.0,
        "median_window_return_pct": 1.1,
        "worst_window_return_pct": 0.5,
        "return_std": 0.3,
    }

    breakdown = opt._build_score_breakdown(
        train_stats=train_stats,
        test_stats=test_stats,
        train_trades=[],
        test_trades=[],
        raw_selection_score=-2.0,
        window_penalty=0.0,
        final_selection_score=-2.0,
        acceptable=True,
        window_robustness=robust,
    )

    assert breakdown["train_minus_test_return_pct"] == 11.0
    assert breakdown["train_minus_test_expectancy_pct"] == 0.3
    assert breakdown["train_minus_test_profit_factor"] == 1.1


if __name__ == "__main__":
    test_build_score_breakdown_contains_main_fields()
    test_build_score_breakdown_gap_metrics()
    print("✅ Все тесты этапа 4 прошли успешно")