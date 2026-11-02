"""
Скоринг и оценка пригодности стратегий.
"""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class OptimizerScoring:
    @staticmethod
    def compute_raw_selection_score_v2(
        *,
        train_stats: Dict,
        test_stats: Dict,
        test_trades: List[Dict],
        profit_concentration_func,
    ) -> float:
        """
        Новая мягкая формула raw score с защитой от артефактов компаундирования.

        Принципы:
        - TEST важнее TRAIN;
        - train/test gap штрафуется мягко, а не разрушительно;
        - концентрация прибыли штрафуется только при явном перекосе;
        - score отвечает за ранжирование, а не за hard reject.
        """

        # ВАЖНО: ограничиваем return_pct разумными пределами
        def _safe_return(stats: Dict) -> float:
            r = float(stats.get("return_pct", stats.get("total_pnl_pct", 0.0)) or 0.0)
            return max(-100.0, min(r, 2000.0))  # cap [-100%, +2000%]

        def _f(stats: Dict, key: str, default: float = 0.0) -> float:
            try:
                return float(stats.get(key, default) or default)
            except Exception:
                return float(default)

        def _i(stats: Dict, key: str, default: int = 0) -> int:
            try:
                return int(stats.get(key, default) or default)
            except Exception:
                return int(default)

        train_return = _safe_return(train_stats)
        test_return = _safe_return(test_stats)

        train_expectancy = _f(
            train_stats,
            "expectancy_pct",
            _f(train_stats, "avg_trade_pct", 0.0),
        )
        test_expectancy = _f(
            test_stats,
            "expectancy_pct",
            _f(test_stats, "avg_trade_pct", 0.0),
        )

        train_pf = _f(train_stats, "profit_factor", 0.0)
        test_pf = _f(test_stats, "profit_factor", 0.0)

        test_dd = _f(test_stats, "max_drawdown_pct", 0.0)

        train_trades_count = _i(train_stats, "total_trades", 0)
        test_trades_count = _i(test_stats, "total_trades", 0)

        # Ограничиваем gap разумными пределами
        gap_return = min(abs(train_return - test_return), 200.0)
        gap_expectancy = abs(train_expectancy - test_expectancy)

        test_profit_concentration = profit_concentration_func(test_trades)

        score = 0.0

        # 1) Главный приоритет — TEST
        # Базовая доходность и качество
        score += 3.0 * test_expectancy
        score += 1.5 * (min(test_pf, 2.5) - 1.0)
        score += 0.30 * test_return
        
        # ЖЕСТКИЙ БОНУС ЗА ЧАСТОТУ СДЕЛОК
        # Модель должна понимать: 40 сделок с PF=1.3 ЛУЧШЕ, чем 8 сделок с PF=3.0
        test_trades = min(test_trades_count, 80)
        if test_trades_count < 20:
            # ШТРАФ за малое число сделок (переобучение)
            score -= 5.0 
            score += 0.02 * test_trades  # Слабый бонус
        elif test_trades_count < 40:
            score += 0.10 * test_trades  # Средний бонус
        else:
            score += 0.25 * test_trades  # Мощный бонус за частоту

        # 2) TRAIN — только как подтверждение
        score += 1.5 * train_expectancy
        score += 0.8 * (min(train_pf, 2.5) - 1.0)
        score += 0.015 * min(train_trades_count, 80)

        # 3) Мягкие штрафы
        # Просадка: штрафуем только за DD выше 2%
        score -= 0.30 * max(test_dd - 2.0, 0.0)

        # Разрыв TRAIN↔TEST по доходности: только если реально заметный
        score -= 0.08 * max(gap_return - 10.0, 0.0)

        # Разрыв TRAIN↔TEST по expectancy: очень мягко
        score -= 4.0 * max(gap_expectancy - 0.15, 0.0)

        # Концентрация прибыли: штраф только если слишком высокая
        score -= 1.5 * max(test_profit_concentration - 0.60, 0.0)

        return round(score, 4)

    @staticmethod
    def score_and_flags(
        train_stats: Dict,
        test_stats: Dict,
        test_trades: List[Dict],
        min_test_trades: int,
        profit_concentration_func,
    ) -> Tuple[float, bool, float, float]:
        """
        Возвращает: (raw_selection_score, acceptable, train_expectancy, test_expectancy)
        """
        train_exp = float(
            train_stats.get(
                "expectancy_pct",
                train_stats.get("avg_trade_pct", 0.0),
            ) or 0.0
        )
        test_exp = float(
            test_stats.get(
                "expectancy_pct",
                test_stats.get("avg_trade_pct", 0.0),
            ) or 0.0
        )

        train_pf = float(train_stats.get("profit_factor", 0.0) or 0.0)
        test_pf = float(test_stats.get("profit_factor", 0.0) or 0.0)

        test_trade_count = int(test_stats.get("total_trades", 0) or 0)

        raw_selection_score = OptimizerScoring.compute_raw_selection_score_v2(
            train_stats=train_stats,
            test_stats=test_stats,
            test_trades=test_trades,
            profit_concentration_func=profit_concentration_func,
        )

        acceptable = (
            test_trade_count >= min_test_trades
            and train_exp > 0
            and test_exp > 0
            and train_pf > 1.0
            and test_pf > 1.0
        )

        return raw_selection_score, acceptable, train_exp, test_exp

    @staticmethod
    def selection_score(
        test_stats: Dict,
        test_trades: List[Dict],
        profit_concentration_func,
    ) -> float:
        """
        Единая целевая функция по TEST: суммарный P&L со штрафом за просадку.
        total_pnl = expectancy × число сделок, поэтому одновременно поощряются
        и качество, и частота сделок.
        """
        total_pnl = float(test_stats.get('total_pnl_pct', 0) or 0)
        drawdown = float(test_stats.get('max_drawdown_pct', 0) or 0)
        profit_factor = float(test_stats.get('profit_factor', 0) or 0)
        trades = int(test_stats.get('total_trades', 0) or 0)
        long_trades = int(test_stats.get('long_trades', 0) or 0)
        short_trades = int(test_stats.get('short_trades', 0) or 0)

        if trades == 0:
            return -999.0

        score = total_pnl - 0.5 * drawdown

        # Слабый бонус за устойчивость (PF с потолком).
        capped_pf = max(0.0, min(profit_factor, 10.0))
        score += (capped_pf - 1.0) * 0.5

        # Штраф за концентрацию прибыли на одной сделке.
        if test_trades and trades >= 8:
            concentration = profit_concentration_func(test_trades)
            if concentration > 0.5:
                score -= (concentration - 0.5) * 40.0

        # Штраф за сильный перекос по направлению.
        directional_total = long_trades + short_trades
        if directional_total >= 8:
            skew = abs(long_trades - short_trades) / directional_total
            if skew > 0.85:
                score -= (skew - 0.85) * 20.0

        return round(score, 4)

    @staticmethod
    def train_score(train_stats: Dict) -> float:
        """Score по TRAIN для локального поиска."""
        trades = int(train_stats.get('total_trades', 0) or 0)
        if trades == 0:
            return -999.0
        total_pnl = float(train_stats.get('total_pnl_pct', 0) or 0)
        drawdown = float(train_stats.get('max_drawdown_pct', 0) or 0)
        return total_pnl - 0.5 * drawdown

    @staticmethod
    def profit_concentration(trades: List[Dict]) -> float:
        """
        Доля самой крупной прибыльной сделки в суммарной прибыли (0..1).

        Высокое значение (например > 0.5) означает, что вся прибыль держится
        на одной-двух сделках — статистически ненадёжная стратегия.
        """
        profits = [
            float(t.get("result_pct", 0))
            for t in trades
            if float(t.get("result_pct", 0)) > 0
        ]
        total = sum(profits)
        if total <= 0:
            return 0.0
        return round(max(profits) / total, 3)

    @staticmethod
    def merge_stats(
        stats_list: List[Dict],
        trades_all: List[Dict],
    ) -> Dict:
        """
        Считает агрегированную статистику по объединённым сделкам.

        Доходность и максимальная просадка усредняются по окнам:
        у каждого walk-forward окна собственный виртуальный капитал,
        поэтому складывать return_pct между окнами некорректно.
        """
        if not stats_list:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl_pct': 0.0,
                'return_pct': 0.0,
                'max_drawdown_pct': 0.0,
                'profit_factor': 0.0,
                'avg_trade_pct': 0.0,
                'expectancy_pct': 0.0,
                'avg_win_pct': 0.0,
                'avg_loss_pct': 0.0,
                'long_trades': 0,
                'short_trades': 0,
            }

        total_trades = len(trades_all)

        wins = [
            trade for trade in trades_all
            if trade.get('result_pct', 0) > 0
        ]

        losses = [
            trade for trade in trades_all
            if trade.get('result_pct', 0) <= 0
        ]

        total_pnl = sum(
            trade.get('result_pct', 0)
            for trade in trades_all
        )

        gross_profit = sum(
            trade.get('result_pct', 0)
            for trade in wins
        )

        gross_loss = abs(sum(
            trade.get('result_pct', 0)
            for trade in losses
        ))

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = 10.0
        else:
            profit_factor = 0.0

        expectancy = (
            total_pnl / total_trades
            if total_trades > 0
            else 0.0
        )

        avg_win = (
            sum(trade.get('result_pct', 0) for trade in wins) / len(wins)
            if wins
            else 0.0
        )

        avg_loss = (
            abs(sum(trade.get('result_pct', 0) for trade in losses)) / len(losses)
            if losses
            else 0.0
        )

        # Компаундируем доходность последовательных окон вместо усреднения:
        # окна TEST идут хронологически друг за другом.
        compounded = 1.0
        for stat in stats_list:
            compounded *= 1.0 + float(stat.get('return_pct', 0) or 0) / 100.0
        total_return = (compounded - 1.0) * 100.0

        # Худшая просадка среди окон честнее среднего.
        worst_drawdown = max(
            float(stat.get('max_drawdown_pct', 0) or 0)
            for stat in stats_list
        )

        return {
            'total_trades': total_trades,
            'win_rate': round(
                len(wins) / total_trades * 100,
                1,
            ) if total_trades > 0 else 0.0,
            'total_pnl_pct': round(total_pnl, 2),
            'return_pct': round(total_return, 2),
            'max_drawdown_pct': round(worst_drawdown, 2),
            'profit_factor': round(profit_factor, 2),
            'avg_trade_pct': round(expectancy, 4),
            'expectancy_pct': round(expectancy, 4),
            'avg_win_pct': round(avg_win, 4),
            'avg_loss_pct': round(avg_loss, 4),
            'long_trades': sum(
                1 for trade in trades_all
                if trade.get('direction') == 'long'
            ),
            'short_trades': sum(
                1 for trade in trades_all
                if trade.get('direction') == 'short'
            ),
        }