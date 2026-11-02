"""
Анализ устойчивости стратегии по walk-forward окнам.
"""

import logging
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Константы для оценки робастности
ROBUST_MIN_PROFITABLE_WINDOWS_RATIO = 2 / 3
ROBUST_MIN_POSITIVE_EXPECTANCY_WINDOWS_RATIO = 2 / 3
ROBUST_MIN_PF_PASS_WINDOWS_RATIO = 2 / 3
ROBUST_MIN_MEDIAN_WINDOW_RETURN_PCT = 0.0
ROBUST_MAX_WORST_WINDOW_RETURN_PCT = -5.0
ROBUST_MIN_STABILITY_SCORE = 0.5


class OptimizerRobustness:
    @staticmethod
    def compute_window_robustness(fold_results: List[Dict]) -> Dict:
        """
        Считает устойчивость стратегии по отдельным TEST-окнам walk-forward.

        Ничего не решает за оптимизатор, только возвращает диагностику:
        - сколько окон было прибыльными;
        - сколько окон имели положительный expectancy;
        - сколько окон имели PF > 1;
        - медианы и разброс по доходности/expectancy;
        - худшее и лучшее окно.
        """
        logger.info(f"[ROBUST DEBUG] Computing robustness for {len(fold_results)} folds")
        for idx, fold in enumerate(fold_results):
            test_result = fold.get('test_result', {})
            test_stats = test_result.get('stats', {}) or {}
            logger.info(
                f"[ROBUST DEBUG] Fold {idx+1}: "
                f"trades={test_stats.get('total_trades', 0)}, "
                f"return={test_stats.get('return_pct', 0):.2f}%, "
                f"pf={test_stats.get('profit_factor', 0):.2f}"
            )

        if not fold_results:
            logger.warning("[ROBUST DEBUG] fold_results is empty!")
            return {
                "windows_total": 0,
                "profitable_windows": 0,
                "profitable_windows_ratio": 0.0,
                "positive_expectancy_windows": 0,
                "positive_expectancy_windows_ratio": 0.0,
                "pf_pass_windows": 0,
                "pf_pass_windows_ratio": 0.0,
                "median_window_return_pct": 0.0,
                "median_window_expectancy_pct": 0.0,
                "return_std": 0.0,
                "expectancy_std": 0.0,
                "worst_window_return_pct": 0.0,
                "worst_window_expectancy_pct": 0.0,
                "best_window_return_pct": 0.0,
                "best_window_expectancy_pct": 0.0,
                "windows": [],
            }

        test_stats_list = []
        for fold in fold_results:
            stat = (fold.get("test_result") or {}).get("stats", {}) or {}
            test_stats_list.append(stat)

        returns = [float(stat.get("return_pct", 0) or 0) for stat in test_stats_list]
        expectancies = [
            float(stat.get("expectancy_pct", stat.get("avg_trade_pct", 0)) or 0)
            for stat in test_stats_list
        ]
        pfs = [float(stat.get("profit_factor", 0) or 0) for stat in test_stats_list]

        windows_total = len(test_stats_list)
        profitable_windows = sum(1 for x in returns if x > 0)
        positive_expectancy_windows = sum(1 for x in expectancies if x > 0)
        pf_pass_windows = sum(1 for x in pfs if x > 1.0)

        def _ratio(x: int, total: int) -> float:
            return round(x / total, 4) if total > 0 else 0.0

        windows = []
        for idx, stat in enumerate(test_stats_list, start=1):
            windows.append({
                "window_index": idx,
                "return_pct": round(float(stat.get("return_pct", 0) or 0), 4),
                "expectancy_pct": round(
                    float(stat.get("expectancy_pct", stat.get("avg_trade_pct", 0)) or 0), 4
                ),
                "profit_factor": round(float(stat.get("profit_factor", 0) or 0), 4),
                "total_trades": int(stat.get("total_trades", 0) or 0),
                "max_drawdown_pct": round(float(stat.get("max_drawdown_pct", 0) or 0), 4),
            })

        return {
            "windows_total": windows_total,
            "profitable_windows": profitable_windows,
            "profitable_windows_ratio": _ratio(profitable_windows, windows_total),
            "positive_expectancy_windows": positive_expectancy_windows,
            "positive_expectancy_windows_ratio": _ratio(positive_expectancy_windows, windows_total),
            "pf_pass_windows": pf_pass_windows,
            "pf_pass_windows_ratio": _ratio(pf_pass_windows, windows_total),
            "median_window_return_pct": round(float(pd.Series(returns).median()), 4) if returns else 0.0,
            "median_window_expectancy_pct": round(float(pd.Series(expectancies).median()), 4) if expectancies else 0.0,
            "return_std": round(float(pd.Series(returns).std(ddof=0)), 4) if returns else 0.0,
            "expectancy_std": round(float(pd.Series(expectancies).std(ddof=0)), 4) if expectancies else 0.0,
            "worst_window_return_pct": round(min(returns), 4) if returns else 0.0,
            "worst_window_expectancy_pct": round(min(expectancies), 4) if expectancies else 0.0,
            "best_window_return_pct": round(max(returns), 4) if returns else 0.0,
            "best_window_expectancy_pct": round(max(expectancies), 4) if expectancies else 0.0,
            "windows": windows,
        }

    @staticmethod
    def compute_window_diagnostics(window_robustness: Dict) -> Dict:
        """
        На основе window_robustness формирует human-readable diagnostics:
        - предупреждения о хрупкости,
        - оценку стабильности,
        - рекомендации по изменению стратегии.
        """
        warnings = []
        recommendations = []
        stability_score = 1.0  # 0..1, чем выше — тем устойчивее

        windows_total = window_robustness.get("windows_total", 0)
        if windows_total == 0:
            return {
                "stability_score": 0.0,
                "warnings": ["Нет данных по окнам"],
                "recommendations": ["Недостаточно walk-forward окон для оценки устойчивости"],
            }

        profitable_ratio = window_robustness.get("profitable_windows_ratio", 0)
        exp_ratio = window_robustness.get("positive_expectancy_windows_ratio", 0)
        pf_ratio = window_robustness.get("pf_pass_windows_ratio", 0)
        worst_return = window_robustness.get("worst_window_return_pct", 0)
        return_std = window_robustness.get("return_std", 0)
        median_return = window_robustness.get("median_window_return_pct", 0)

        # --- Штрафы за нестабильность ---
        if profitable_ratio < 0.5:
            warnings.append(f"Только {profitable_ratio:.0%} окон прибыльно — стратегия нестабильна")
            stability_score -= 0.25
        elif profitable_ratio < 0.7:
            warnings.append(f"Прибыльных окон {profitable_ratio:.0%} — есть риск нестабильности")
            stability_score -= 0.1

        if exp_ratio < 0.5:
            warnings.append(f"Только {exp_ratio:.0%} окон с положительным expectancy — стратегия хрупка")
            stability_score -= 0.2
        elif exp_ratio < 0.7:
            warnings.append(f"Положительный expectancy только в {exp_ratio:.0%} окон")
            stability_score -= 0.1

        if pf_ratio < 0.5:
            warnings.append(f"PF > 1 только в {pf_ratio:.0%} окон — стратегия не окупается")
            stability_score -= 0.2

        if worst_return < -10.0:
            warnings.append(f"Худшее окно {worst_return:+.1f}% — возможны катастрофические провалы")
            stability_score -= 0.2
        elif worst_return < -5.0:
            warnings.append(f"Худшее окно {worst_return:+.1f}% — есть риски просадки")
            stability_score -= 0.1

        if return_std > 10.0:
            warnings.append(f"Высокая волатильность доходности ({return_std:.1f}% std) — стратегия непредсказуема")
            stability_score -= 0.15
        elif return_std > 5.0:
            warnings.append(f"Умеренная волатильность доходности ({return_std:.1f}% std)")
            stability_score -= 0.05

        # --- Рекомендации ---
        if profitable_ratio < 0.5:
            recommendations.append("Ужесточить фильтры входа или увеличить stop-loss")
        if exp_ratio < 0.5:
            recommendations.append("Проверить логику входов: возможно, слишком много шумных сигналов")
        if pf_ratio < 0.5:
            recommendations.append("Увеличить take_profit или улучшить управление позицией")
        if worst_return < -5.0:
            recommendations.append("Рассмотреть более агрессивный breakeven или trailing stop")
        if return_std > 5.0:
            recommendations.append("Проверить стратегию на более длинной истории и дополнительном holdout-периоде")

        stability_score = max(0.0, min(1.0, stability_score))

        return {
            "stability_score": round(stability_score, 3),
            "warnings": warnings,
            "recommendations": recommendations,
        }

    @staticmethod
    def window_stability_penalty(window_robustness: Dict) -> Tuple[float, List[str]]:
        """
        Возвращает мягкий штраф к selection_score за неустойчивость по TEST-окнам.
        """
        windows_total = int(window_robustness.get("windows_total", 0) or 0)
        if windows_total <= 1:
            return 0.0, []

        penalty = 0.0
        reasons: List[str] = []

        profitable_ratio = float(window_robustness.get("profitable_windows_ratio", 0) or 0)
        exp_ratio = float(window_robustness.get("positive_expectancy_windows_ratio", 0) or 0)
        pf_ratio = float(window_robustness.get("pf_pass_windows_ratio", 0) or 0)
        worst_return = float(window_robustness.get("worst_window_return_pct", 0) or 0)
        return_std = float(window_robustness.get("return_std", 0) or 0)

        if profitable_ratio < 0.5:
            penalty += 1.0
            reasons.append("мало прибыльных окон")
        elif profitable_ratio < 0.67:
            penalty += 0.4
            reasons.append("пограничная доля прибыльных окон")

        if exp_ratio < 0.5:
            penalty += 0.9
            reasons.append("мало окон с положительным expectancy")
        elif exp_ratio < 0.67:
            penalty += 0.3
            reasons.append("пограничная доля окон с положительным expectancy")

        if pf_ratio < 0.5:
            penalty += 0.8
            reasons.append("мало окон с PF > 1")
        elif pf_ratio < 0.67:
            penalty += 0.3
            reasons.append("пограничная доля окон с PF > 1")

        if worst_return < -10.0:
            penalty += 0.8
            reasons.append("есть очень плохое окно")
        elif worst_return < -5.0:
            penalty += 0.4
            reasons.append("есть слабое худшее окно")

        if return_std > 10.0:
            penalty += 0.6
            reasons.append("очень высокий разброс доходности между окнами")
        elif return_std > 5.0:
            penalty += 0.3
            reasons.append("повышенный разброс доходности между окнами")

        penalty = round(min(penalty, 3.0), 4)
        return penalty, reasons

    @staticmethod
    def assess_window_robustness(
        window_robustness: Dict,
        *,
        min_profitable_windows_ratio: float = ROBUST_MIN_PROFITABLE_WINDOWS_RATIO,
        min_positive_expectancy_windows_ratio: float = ROBUST_MIN_POSITIVE_EXPECTANCY_WINDOWS_RATIO,
        min_pf_pass_windows_ratio: float = ROBUST_MIN_PF_PASS_WINDOWS_RATIO,
        min_median_window_return_pct: float = ROBUST_MIN_MEDIAN_WINDOW_RETURN_PCT,
        max_worst_window_return_pct: float = ROBUST_MAX_WORST_WINDOW_RETURN_PCT,
        min_stability_score: float = ROBUST_MIN_STABILITY_SCORE,
    ) -> Dict:
        """
        Оценивает устойчивость стратегии по walk-forward окнам.

        Статусы:
        - robust: стратегия выглядит устойчивой;
        - fragile: есть заметная хрупкость, но стратегия всё ещё пригодна;
        - unstable: стратегия неустойчива.
        """

        def _f(key: str, default: float = 0.0) -> float:
            try:
                return float(window_robustness.get(key, default) or default)
            except Exception:
                return float(default)

        windows_total = int(window_robustness.get("windows_total", 0) or 0)
        profitable_ratio = _f("profitable_windows_ratio", 0.0)
        expectancy_ratio = _f("positive_expectancy_windows_ratio", 0.0)
        pf_ratio = _f("pf_pass_windows_ratio", 0.0)
        median_return = _f("median_window_return_pct", 0.0)
        worst_return = _f("worst_window_return_pct", 0.0)
        stability_score = _f("stability_score", 0.0)

        warnings = []
        reasons = []

        if windows_total <= 0:
            return {
                "status": "unstable",
                "passed": False,
                "fragile": False,
                "warnings": ["Нет данных по окнам walk-forward"],
                "reasons": ["no_window_data"],
                "profitable_windows_ratio": profitable_ratio,
                "positive_expectancy_windows_ratio": expectancy_ratio,
                "pf_pass_windows_ratio": pf_ratio,
                "median_window_return_pct": median_return,
                "worst_window_return_pct": worst_return,
                "stability_score": stability_score,
            }

        if profitable_ratio < min_profitable_windows_ratio:
            warnings.append(f"Прибыльных окон недостаточно: {profitable_ratio:.0%}")
            reasons.append("profitable_windows_ratio_too_low")

        if expectancy_ratio < min_positive_expectancy_windows_ratio:
            warnings.append(f"Окон с положительным expectancy недостаточно: {expectancy_ratio:.0%}")
            reasons.append("positive_expectancy_windows_ratio_too_low")

        if pf_ratio < min_pf_pass_windows_ratio:
            warnings.append(f"Окон с PF > 1 недостаточно: {pf_ratio:.0%}")
            reasons.append("pf_pass_windows_ratio_too_low")

        if median_return < min_median_window_return_pct:
            warnings.append(f"Медианная доходность по окнам отрицательная: {median_return:+.2f}%")
            reasons.append("median_window_return_negative")

        if worst_return < max_worst_window_return_pct:
            warnings.append(f"Худшее окно слишком слабое: {worst_return:+.2f}%")
            reasons.append("worst_window_return_too_low")

        if stability_score < min_stability_score:
            warnings.append(f"Низкий stability_score: {stability_score:.3f}")
            reasons.append("stability_score_too_low")

        strong_pass = (
            profitable_ratio >= min_profitable_windows_ratio
            and expectancy_ratio >= min_positive_expectancy_windows_ratio
            and pf_ratio >= min_pf_pass_windows_ratio
            and median_return >= min_median_window_return_pct
            and worst_return >= max_worst_window_return_pct
            and stability_score >= min_stability_score
        )

        weak_pass = (
            windows_total >= 2
            and profitable_ratio >= 0.5
            and expectancy_ratio >= 0.5
            and pf_ratio >= 0.5
            and stability_score >= 0.35
        )

        if strong_pass:
            status = "robust"
            passed = True
            fragile = False
        elif weak_pass:
            status = "fragile"
            passed = True
            fragile = True
        else:
            status = "unstable"
            passed = False
            fragile = False

        return {
            "status": status,
            "passed": passed,
            "fragile": fragile,
            "warnings": warnings,
            "reasons": reasons,
            "profitable_windows_ratio": profitable_ratio,
            "positive_expectancy_windows_ratio": expectancy_ratio,
            "pf_pass_windows_ratio": pf_ratio,
            "median_window_return_pct": median_return,
            "worst_window_return_pct": worst_return,
            "stability_score": stability_score,
        }