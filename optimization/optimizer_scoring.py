"""
Scoring and strategy fitness evaluation.
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
        New soft raw score formula with protection against compounding artifacts.

        Principles:
        - TEST is more important than TRAIN;
        - train/test gap is penalized softly, not destructively;
        - profit concentration is penalized only with explicit skew;
        - score is responsible for ranking, not for hard reject.
        """

        # IMPORTANT: limit return_pct to reasonable bounds
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

        # Limit gap to reasonable bounds
        gap_return = min(abs(train_return - test_return), 200.0)
        gap_expectancy = abs(train_expectancy - test_expectancy)

        test_profit_concentration = profit_concentration_func(test_trades)

        score = 0.0

        # 1) Main priority — TEST
        # Base return and quality
        score += 3.0 * test_expectancy
        score += 1.5 * (min(test_pf, 2.5) - 1.0)
        score += 0.30 * test_return
        
        # HARD BONUS FOR TRADE FREQUENCY
        # The model must understand: 40 trades with PF=1.3 are BETTER than 8 trades with PF=3.0
        test_trades = min(test_trades_count, 80)
        if test_trades_count < 20:
            # PENALTY for small number of trades (overfitting)
            score -= 5.0 
            score += 0.02 * test_trades  # Weak bonus
        elif test_trades_count < 40:
            score += 0.10 * test_trades  # Medium bonus
        else:
            score += 0.25 * test_trades  # Strong bonus for frequency

        # 2) TRAIN — only as confirmation
        score += 1.5 * train_expectancy
        score += 0.8 * (min(train_pf, 2.5) - 1.0)
        score += 0.015 * min(train_trades_count, 80)

        # 3) Soft penalties
        # Drawdown: penalize only for DD above 2%
        score -= 0.30 * max(test_dd - 2.0, 0.0)

        # TRAIN↔TEST return gap: only if really noticeable
        score -= 0.08 * max(gap_return - 10.0, 0.0)

        # TRAIN↔TEST expectancy gap: very soft
        score -= 4.0 * max(gap_expectancy - 0.15, 0.0)

        # Profit concentration: penalty only if too high
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
        Returns: (raw_selection_score, acceptable, train_expectancy, test_expectancy)
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
        Unified objective function on TEST: total P&L with drawdown penalty.
        total_pnl = expectancy × number of trades, so both quality and trade
        frequency are rewarded simultaneously.
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

        # Weak bonus for stability (PF with ceiling).
        capped_pf = max(0.0, min(profit_factor, 10.0))
        score += (capped_pf - 1.0) * 0.5

        # Penalty for profit concentration on a single trade.
        if test_trades and trades >= 8:
            concentration = profit_concentration_func(test_trades)
            if concentration > 0.5:
                score -= (concentration - 0.5) * 40.0

        # Penalty for strong directional skew.
        directional_total = long_trades + short_trades
        if directional_total >= 8:
            skew = abs(long_trades - short_trades) / directional_total
            if skew > 0.85:
                score -= (skew - 0.85) * 20.0

        return round(score, 4)

    @staticmethod
    def train_score(train_stats: Dict) -> float:
        """Score on TRAIN for local search."""
        trades = int(train_stats.get('total_trades', 0) or 0)
        if trades == 0:
            return -999.0
        total_pnl = float(train_stats.get('total_pnl_pct', 0) or 0)
        drawdown = float(train_stats.get('max_drawdown_pct', 0) or 0)
        return total_pnl - 0.5 * drawdown

    @staticmethod
    def profit_concentration(trades: List[Dict]) -> float:
        """
        Share of the largest profitable trade in total profit (0..1).

        High value (e.g. > 0.5) means all profit depends on one or two trades —
        a statistically unreliable strategy.
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
        Calculates aggregated statistics on combined trades.

        Return and max drawdown are averaged across windows:
        each walk-forward window has its own virtual capital,
        so summing return_pct between windows is incorrect.
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

        # Compound return of consecutive windows instead of averaging:
        # TEST windows go chronologically one after another.
        compounded = 1.0
        for stat in stats_list:
            compounded *= 1.0 + float(stat.get('return_pct', 0) or 0) / 100.0
        total_return = (compounded - 1.0) * 100.0

        # Worst drawdown among windows is more honest than average.
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