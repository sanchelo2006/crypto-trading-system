"""
Holdout-валидация финального кандидата.
"""

import logging
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)

MIN_WEAK_HOLDOUT_TRADES = 5


class OptimizerHoldout:
    def __init__(self, engine):
        self.engine = engine

    def evaluate_spec_on_holdout(
        self,
        spec: Dict,
        rules: str,
        holdout_df,
        symbol: str,
        initial_capital: float,
        commission_pct: float,
    ) -> Dict:
        """
        Финальная оценка стратегии на holdout-периоде без TRAIN.
        Использует прямой прогон движка на одном DataFrame.
        """
        try:
            if holdout_df is None or len(holdout_df) == 0:
                return {"error": "Пустой holdout_df."}

            result = self.engine.run_on_dataframe(
                df=holdout_df,
                symbol=symbol,
                strategy_spec=spec,
                initial_capital=initial_capital,
                commission_pct=commission_pct,
                rules_text=rules,
            )

            if not result:
                return {"error": "Движок не вернул результат holdout-бэктеста."}

            if "error" in result:
                return {"error": result["error"]}

            stats = result.get("stats", {}) or {}
            trades = result.get("trades", []) or []

            return {
                "stats": stats,
                "trades": trades,
                "raw_result": result,
            }

        except Exception as e:
            logger.exception("Ошибка при финальной оценке HOLDOUT")
            return {"error": str(e)}

    @staticmethod
    def assess_holdout_quality(
        holdout_stats: Dict,
        min_holdout_trades: int = 10,
        min_weak_holdout_trades: int = MIN_WEAK_HOLDOUT_TRADES,
    ) -> Dict:
        """
        Диагностика качества на holdout.
        Статусы:
        - passed: метрики хорошие и сделок достаточно
        - weak_pass: метрики хорошие, но сделок мало
        - failed: метрики плохие или сделок слишком мало
        """

        def _f(key: str, default: float = 0.0) -> float:
            try:
                return float(holdout_stats.get(key, default) or default)
            except Exception:
                return float(default)

        def _i(key: str, default: int = 0) -> int:
            try:
                return int(holdout_stats.get(key, default) or default)
            except Exception:
                return int(default)

        trades = _i("total_trades", 0)
        expectancy = _f("expectancy_pct", _f("avg_trade_pct", 0.0))
        pf = _f("profit_factor", 0.0)
        ret = _f("return_pct", _f("total_pnl_pct", 0.0))
        dd = _f("max_drawdown_pct", 0.0)

        warnings = []
        recommendations = []

        quality_ok = (
            expectancy > 0
            and pf > 1.05
            and ret > 0
        )

        if trades < min_holdout_trades:
            warnings.append("На holdout мало сделок")
            recommendations.append("Проверить стратегию на более длинном внешнем периоде")

        if expectancy <= 0:
            warnings.append("На holdout expectancy <= 0")
            recommendations.append("Пересмотреть качество входов и управление позицией")

        if pf <= 1.05:
            warnings.append("На holdout PF <= 1.05")
            recommendations.append("Проверить соотношение прибыль/убыток и фильтры входа")

        if ret <= 0:
            warnings.append("На holdout итоговая доходность <= 0")
            recommendations.append("Проверить устойчивость стратегии вне оптимизационного периода")

        if dd > 10:
            warnings.append("На holdout повышенная просадка")
            recommendations.append("Снизить риск или ужесточить управление позицией")

        passed = trades >= min_holdout_trades and quality_ok
        weak_pass = (
            trades >= min_weak_holdout_trades
            and trades < min_holdout_trades
            and quality_ok
        )

        if passed:
            status = "passed"
        elif weak_pass:
            status = "weak_pass"
        else:
            status = "failed"

        return {
            "status": status,
            "passed": passed,
            "weak_pass": weak_pass,
            "warnings": warnings,
            "recommendations": recommendations,
            "total_trades": trades,
            "return_pct": ret,
            "expectancy_pct": expectancy,
            "profit_factor": pf,
            "max_drawdown_pct": dd,
        }

    @staticmethod
    def build_holdout_summary(
        *,
        holdout_stats: Dict,
        holdout_trades: List[Dict],
        holdout_quality: Dict,
        holdout_meta: Dict,
        holdout_period: Dict,
        profit_concentration_func,
    ) -> Dict:
        """Сводка результатов holdout."""
        return {
            "enabled": bool(holdout_meta.get("enabled", False)),
            "reason": holdout_meta.get("reason"),
            "period_start": holdout_period.get("start"),
            "period_end": holdout_period.get("end"),
            "candles": int(holdout_meta.get("holdout_candles", 0) or 0),
            "total_trades": int(holdout_stats.get("total_trades", 0) or 0),
            "return_pct": float(
                holdout_stats.get("return_pct", holdout_stats.get("total_pnl_pct", 0.0)) or 0.0
            ),
            "expectancy_pct": float(
                holdout_stats.get("expectancy_pct", holdout_stats.get("avg_trade_pct", 0.0)) or 0.0
            ),
            "profit_factor": float(holdout_stats.get("profit_factor", 0.0) or 0.0),
            "max_drawdown_pct": float(holdout_stats.get("max_drawdown_pct", 0.0) or 0.0),
            "profit_concentration": profit_concentration_func(holdout_trades),
            "status": holdout_quality.get("status", "failed"),
            "passed": bool(holdout_quality.get("passed", False)),
            "weak_pass": bool(holdout_quality.get("weak_pass", False)),
            "warnings": list(holdout_quality.get("warnings", []) or []),
            "recommendations": list(holdout_quality.get("recommendations", []) or []),
        }