"""
Analysis of strategy robustness across walk-forward windows.
"""
import logging
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Constants for robustness assessment
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
        Computes strategy robustness across individual TEST windows of walk-forward.
        Does not decide for the optimizer, only returns diagnostics:
        - how many windows were profitable;
        - how many windows had positive expectancy;
        - how many windows had PF > 1;
        - medians and dispersion of return/expectancy;
        - worst and best window.
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
        Based on window_robustness, forms human-readable diagnostics:
        - fragility warnings,
        - stability assessment,
        - recommendations for strategy changes.
        """
        warnings = []
        recommendations = []
        stability_score = 1.0  # 0..1, higher means more stable

        windows_total = window_robustness.get("windows_total", 0)
        if windows_total == 0:
            return {
                "stability_score": 0.0,
                "warnings": ["No window data"],
                "recommendations": ["Not enough walk-forward windows to assess robustness"],
            }

        profitable_ratio = window_robustness.get("profitable_windows_ratio", 0)
        exp_ratio = window_robustness.get("positive_expectancy_windows_ratio", 0)
        pf_ratio = window_robustness.get("pf_pass_windows_ratio", 0)
        worst_return = window_robustness.get("worst_window_return_pct", 0)
        return_std = window_robustness.get("return_std", 0)
        median_return = window_robustness.get("median_window_return_pct", 0)

        # --- Penalties for instability ---
        if profitable_ratio < 0.5:
            warnings.append(f"Only {profitable_ratio:.0%} of windows are profitable — strategy is unstable")
            stability_score -= 0.25
        elif profitable_ratio < 0.7:
            warnings.append(f"Profitable windows {profitable_ratio:.0%} — there is instability risk")
            stability_score -= 0.1

        if exp_ratio < 0.5:
            warnings.append(f"Only {exp_ratio:.0%} of windows have positive expectancy — strategy is fragile")
            stability_score -= 0.2
        elif exp_ratio < 0.7:
            warnings.append(f"Positive expectancy in only {exp_ratio:.0%} of windows")
            stability_score -= 0.1

        if pf_ratio < 0.5:
            warnings.append(f"PF > 1 in only {pf_ratio:.0%} of windows — strategy does not pay off")
            stability_score -= 0.2

        if worst_return < -10.0:
            warnings.append(f"Worst window {worst_return:+.1f}% — catastrophic drawdowns possible")
            stability_score -= 0.2
        elif worst_return < -5.0:
            warnings.append(f"Worst window {worst_return:+.1f}% — there are drawdown risks")
            stability_score -= 0.1

        if return_std > 10.0:
            warnings.append(f"Very high return volatility ({return_std:.1f}% std) — strategy is unpredictable")
            stability_score -= 0.15
        elif return_std > 5.0:
            warnings.append(f"Moderate return volatility ({return_std:.1f}% std)")
            stability_score -= 0.05

        # --- Recommendations ---
        if profitable_ratio < 0.5:
            recommendations.append("Tighten entry filters or increase stop-loss")
        if exp_ratio < 0.5:
            recommendations.append("Check entry logic: possibly too many noisy signals")
        if pf_ratio < 0.5:
            recommendations.append("Increase take_profit or improve position management")
        if worst_return < -5.0:
            recommendations.append("Consider more aggressive breakeven or trailing stop")
        if return_std > 5.0:
            recommendations.append("Test the strategy on longer history and an additional holdout period")

        stability_score = max(0.0, min(1.0, stability_score))

        return {
            "stability_score": round(stability_score, 3),
            "warnings": warnings,
            "recommendations": recommendations,
        }

    @staticmethod
    def window_stability_penalty(window_robustness: Dict) -> Tuple[float, List[str]]:
        """
        Returns a soft penalty to selection_score for instability across TEST windows.
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
            reasons.append("too few profitable windows")
        elif profitable_ratio < 0.67:
            penalty += 0.4
            reasons.append("borderline share of profitable windows")

        if exp_ratio < 0.5:
            penalty += 0.9
            reasons.append("too few windows with positive expectancy")
        elif exp_ratio < 0.67:
            penalty += 0.3
            reasons.append("borderline share of windows with positive expectancy")

        if pf_ratio < 0.5:
            penalty += 0.8
            reasons.append("too few windows with PF > 1")
        elif pf_ratio < 0.67:
            penalty += 0.3
            reasons.append("borderline share of windows with PF > 1")

        if worst_return < -10.0:
            penalty += 0.8
            reasons.append("there is a very bad window")
        elif worst_return < -5.0:
            penalty += 0.4
            reasons.append("there is a weak worst window")

        if return_std > 10.0:
            penalty += 0.6
            reasons.append("very high return dispersion between windows")
        elif return_std > 5.0:
            penalty += 0.3
            reasons.append("elevated return dispersion between windows")

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
        Assesses strategy robustness across walk-forward windows.

        Statuses:
        - robust: strategy looks robust;
        - fragile: there is noticeable fragility, but the strategy is still usable;
        - unstable: strategy is not robust.
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
                "warnings": ["No walk-forward window data"],
                "reasons": ["no_window_data"],
                "profitable_windows_ratio": profitable_ratio,
                "positive_expectancy_windows_ratio": expectancy_ratio,
                "pf_pass_windows_ratio": pf_ratio,
                "median_window_return_pct": median_return,
                "worst_window_return_pct": worst_return,
                "stability_score": stability_score,
            }

        if profitable_ratio < min_profitable_windows_ratio:
            warnings.append(f"Not enough profitable windows: {profitable_ratio:.0%}")
            reasons.append("profitable_windows_ratio_too_low")

        if expectancy_ratio < min_positive_expectancy_windows_ratio:
            warnings.append(f"Not enough windows with positive expectancy: {expectancy_ratio:.0%}")
            reasons.append("positive_expectancy_windows_ratio_too_low")

        if pf_ratio < min_pf_pass_windows_ratio:
            warnings.append(f"Not enough windows with PF > 1: {pf_ratio:.0%}")
            reasons.append("pf_pass_windows_ratio_too_low")

        if median_return < min_median_window_return_pct:
            warnings.append(f"Median window return is negative: {median_return:+.2f}%")
            reasons.append("median_window_return_negative")

        if worst_return < max_worst_window_return_pct:
            warnings.append(f"Worst window is too weak: {worst_return:+.2f}%")
            reasons.append("worst_window_return_too_low")

        if stability_score < min_stability_score:
            warnings.append(f"Low stability_score: {stability_score:.3f}")
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