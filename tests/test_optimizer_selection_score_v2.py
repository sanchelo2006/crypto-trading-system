"""
Unit-тесты для новой формулы raw selection score.

После рефакторинга логика вынесена в optimization.optimizer_scoring.
"""

from optimization.optimizer_scoring import OptimizerScoring

def test_stronger_test_candidate_gets_higher_raw_score():
    weaker_train = {
        "total_trades": 46,
        "return_pct": 12.0,
        "expectancy_pct": 0.2611,
        "profit_factor": 1.77,
        "max_drawdown_pct": 3.0,
    }
    weaker_test = {
        "total_trades": 18,
        "return_pct": 1.10,
        "expectancy_pct": 0.0667,
        "profit_factor": 1.18,
        "max_drawdown_pct": 1.64,
    }

    stronger_train = {
        "total_trades": 47,
        "return_pct": 3.40,
        "expectancy_pct": 0.0723,
        "profit_factor": 1.21,
        "max_drawdown_pct": 2.5,
    }
    stronger_test = {
        "total_trades": 14,
        "return_pct": 3.01,
        "expectancy_pct": 0.2443,
        "profit_factor": 1.44,
        "max_drawdown_pct": 2.74,
    }

    weaker_score = OptimizerScoring.compute_raw_selection_score_v2(
        train_stats=weaker_train,
        test_stats=weaker_test,
        test_trades=[],
        profit_concentration_func=OptimizerScoring.profit_concentration,
    )
    stronger_score = OptimizerScoring.compute_raw_selection_score_v2(
        train_stats=stronger_train,
        test_stats=stronger_test,
        test_trades=[],
        profit_concentration_func=OptimizerScoring.profit_concentration,
    )

    assert stronger_score > weaker_score

def test_gap_penalty_is_soft_not_destructive():
    # Большой разрыв TRAIN↔TEST (14.75% по доходности) должен смягчить
    # score, но не увести его в отрицательную зону: gap-штраф включается
    # только выше 10% разрыва и с малым коэффициентом.
    # Сделок >= 20, чтобы не примешивался отдельный штраф за малое число.
    train_stats = {
        "total_trades": 48,
        "return_pct": 0.25,
        "expectancy_pct": 0.0052,
        "profit_factor": 1.02,
        "max_drawdown_pct": 2.0,
    }
    test_stats = {
        "total_trades": 24,
        "return_pct": 15.0,
        "expectancy_pct": 0.1472,
        "profit_factor": 1.33,
        "max_drawdown_pct": 3.09,
    }

    score = OptimizerScoring.compute_raw_selection_score_v2(
        train_stats=train_stats,
        test_stats=test_stats,
        test_trades=[],
        profit_concentration_func=OptimizerScoring.profit_concentration,
    )

    assert score > 0

def test_acceptable_depends_on_quality_rules_not_score_sign():
    # 18 сделок на TEST: raw score уходит в минус из-за штрафа за малое
    # число сделок (<20), НО quality-правила (trades >= min, exp > 0,
    # pf > 1) выполнены — кандидат acceptable. Приёмка решается
    # правилами, а не знаком скоринга: score нужен для ранжирования.
    train_stats = {
        "total_trades": 48,
        "return_pct": 0.25,
        "expectancy_pct": 0.0052,
        "profit_factor": 1.02,
        "max_drawdown_pct": 2.0,
    }
    test_stats = {
        "total_trades": 18,
        "return_pct": 2.43,
        "expectancy_pct": 0.1472,
        "profit_factor": 1.33,
        "max_drawdown_pct": 3.09,
    }

    raw_score, acceptable, train_exp, test_exp = OptimizerScoring.score_and_flags(
        train_stats,
        test_stats,
        [],
        10,
        OptimizerScoring.profit_concentration,
    )

    assert acceptable is True
    assert train_exp > 0
    assert test_exp > 0
    assert raw_score < 0  # скоринг отрицательный, но приёмка по правилам

def test_unacceptable_when_test_pf_below_one():
    """
    Если TEST profit_factor <= 1.0, кандидат должен быть отвергнут
    (согласно правилу score_and_flags).
    """
    train_stats = {
        "total_trades": 45,
        "return_pct": 7.0,
        "expectancy_pct": 0.16,
        "profit_factor": 1.60,
        "max_drawdown_pct": 3.0,
    }
    test_stats = {
        "total_trades": 15,
        "return_pct": 0.20,
        "expectancy_pct": 0.0140,
        "profit_factor": 0.95,   # ← провал: PF < 1.0
        "max_drawdown_pct": 3.02,
    }

    raw_score, acceptable, _, _ = OptimizerScoring.score_and_flags(
        train_stats,
        test_stats,
        [],
        10,
        OptimizerScoring.profit_concentration,
    )

    assert raw_score is not None
    assert acceptable is False


def test_unacceptable_when_test_expectancy_not_positive():
    """
    Если TEST expectancy <= 0, кандидат должен быть отвергнут.
    """
    train_stats = {
        "total_trades": 45,
        "return_pct": 7.0,
        "expectancy_pct": 0.16,
        "profit_factor": 1.60,
        "max_drawdown_pct": 3.0,
    }
    test_stats = {
        "total_trades": 15,
        "return_pct": 0.20,
        "expectancy_pct": -0.0100,   # ← провал: expectancy < 0
        "profit_factor": 1.03,
        "max_drawdown_pct": 3.02,
    }

    raw_score, acceptable, _, _ = OptimizerScoring.score_and_flags(
        train_stats,
        test_stats,
        [],
        10,
        OptimizerScoring.profit_concentration,
    )

    assert raw_score is not None
    assert acceptable is False


def test_unacceptable_when_train_fails():
    """
    Если TRAIN провален (PF <= 1.0 или expectancy <= 0) — тоже reject.
    Это защищает от ситуации «повезло на TEST при отсутствии эджа на TRAIN».
    """
    train_stats = {
        "total_trades": 45,
        "return_pct": 0.5,
        "expectancy_pct": -0.0200,   # ← провал: train expectancy < 0
        "profit_factor": 1.02,
        "max_drawdown_pct": 3.0,
    }
    test_stats = {
        "total_trades": 15,
        "return_pct": 1.50,
        "expectancy_pct": 0.1000,
        "profit_factor": 1.30,
        "max_drawdown_pct": 2.00,
    }

    raw_score, acceptable, _, _ = OptimizerScoring.score_and_flags(
        train_stats,
        test_stats,
        [],
        10,
        OptimizerScoring.profit_concentration,
    )

    assert raw_score is not None
    assert acceptable is False

if __name__ == "__main__":
    test_stronger_test_candidate_gets_higher_raw_score()
    test_gap_penalty_is_soft_not_destructive()
    test_acceptable_depends_on_quality_rules_not_score_sign()
    test_unacceptable_when_test_pf_below_one()
    test_unacceptable_when_test_expectancy_not_positive()
    test_unacceptable_when_train_fails()
    print("✅ Все тесты этапа 5 прошли успешно")
