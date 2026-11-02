"""
Формирование сводок, отчётов и trade summaries.
"""

import json
import logging
from typing import Dict, List

import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class OptimizerSummary:
    @staticmethod
    def derived(snapshot: Dict) -> Dict:
        """Вычисляемые метрики из entry_snapshot."""
        d = {}

        if not snapshot:
            return d

        vol = snapshot.get("volume")
        avg = snapshot.get("avg_volume_96")
        if vol is not None and avg not in (None, 0):
            d["vol_ratio"] = round(vol / avg, 3)

        close = snapshot.get("close")
        sma50 = snapshot.get("sma_50")
        if close is not None and sma50 not in (None, 0):
            d["dist_sma50_pct"] = round((close - sma50) / sma50 * 100, 3)

        if "rsi_14" in snapshot and snapshot.get("rsi_14") is not None:
            d["rsi_14"] = round(snapshot["rsi_14"], 2)

        if "adx_14" in snapshot and snapshot.get("adx_14") is not None:
            d["adx_14"] = round(snapshot["adx_14"], 2)

        if snapshot.get("macd") is not None and snapshot.get("macd_signal") is not None:
            d["macd_vs_signal"] = round(snapshot["macd"] - snapshot["macd_signal"], 5)

        if snapshot.get("atr_14") is not None:
            d["atr_14"] = round(snapshot["atr_14"], 6)

        return d

    @staticmethod
    def summarize_trades(trades: List[Dict], derived_func) -> Dict:
        """Сводка по сделкам для LLM (оптимизирована по токенам)."""
        if not trades:
            return {
                "total": 0,
                "note": "Сделок не было — условия входа слишком строгие, их нужно ОСЛАБИТЬ"
            }
        
        wins = [t for t in trades if t.get("result_pct", 0) > 0]
        losses = [t for t in trades if t.get("result_pct", 0) <= 0]

        def avg_metrics(group: List[Dict]) -> Dict:
            if not group:
                return {}
            derived = [derived_func(t.get("entry_snapshot", {})) for t in group]
            keys = set()
            for d in derived:
                keys.update(d.keys())
            out = {}
            for k in keys:
                vals = [d[k] for d in derived if k in d and d[k] is not None]
                if vals:
                    out[k] = round(sum(vals) / len(vals), 3)
            return out

        def count_by(group: List[Dict], field: str) -> Dict:
            res = {}
            for t in group:
                key = t.get(field, "?")
                res[key] = res.get(key, 0) + 1
            return res

        # ВМЕСТО списка худших сделок (worst_trades) передаем агрегированные профили
        loss_profile = avg_metrics(losses)
        win_profile = avg_metrics(wins)

        return {
            "total": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "winrate": round(len(wins) / len(trades) * 100, 1) if trades else 0.0,
            "loss_by_direction": count_by(losses, "direction"),
            "loss_by_exit": count_by(losses, "exit_reason"),
            "loss_profile": loss_profile,   # Средние индикаторы на входах, которые принесли убыток
            "win_profile": win_profile,     # Средние индикаторы на входах, которые принесли прибыль
        }

    @staticmethod
    def compact_history(history: List[Dict], limit: int = 10) -> List[Dict]:
        """Компактная история для LLM (без мусорных массивов)."""
        out = []
        for h in history[-limit:]:
            score_breakdown = h.get('score_breakdown') or {}
            row = {
                'iter': h.get('iteration'),
                'test_trades': h.get('test_trades'),
                'test_pnl': h.get('test_total_pnl_pct'),
                'test_pf': h.get('test_profit_factor'),
                'score': h.get('selection_score'),
                'accepted': h.get('acceptable'),
                'changes': (h.get('changes_summary') or '')[:150], # Обрезаем длинные описания
                'robust_status': score_breakdown.get('robustness_status', ''),
                'worst_window': score_breakdown.get('worst_window_return_pct', 0),
            }
            # Метрики оконной устойчивости — в плоский вид строки:
            # AI в промпте видит динамику стабильности между итерациями,
            # а не только итоговый score.
            row.update(h.get('window_robustness') or {})
            out.append(row)
        return out

    @staticmethod
    def save_round(
        csv_dir: Path,
        symbol: str,
        iteration,
        train_stats: Dict,
        test_stats: Dict,
        spec: Dict,
        rules: str,
        changes: str,
        train_trades: List[Dict],
        test_trades: List[Dict],
        selection_score: float = 0.0,
        acceptable: bool = False,
    ):
        """Сохраняет сводку и сделки одной итерации в CSV."""
        try:
            from datetime import datetime, timezone
            csv_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now(timezone.utc).strftime(
                '%Y-%m-%d %H:%M:%S'
            )

            summary_path = csv_dir / f'{symbol}_optimization_summary.csv'

            row = {
                'timestamp': timestamp,
                'iteration': iteration,

                'train_trades': train_stats.get('total_trades', 0),
                'test_trades': test_stats.get('total_trades', 0),

                'train_winrate': train_stats.get('win_rate', 0),
                'test_winrate': test_stats.get('win_rate', 0),

                'train_return_pct': train_stats.get('return_pct', 0),
                'test_return_pct': test_stats.get('return_pct', 0),

                'train_expectancy_pct': train_stats.get(
                    'expectancy_pct',
                    train_stats.get('avg_trade_pct', 0),
                ),
                'test_expectancy_pct': test_stats.get(
                    'expectancy_pct',
                    test_stats.get('avg_trade_pct', 0),
                ),

                'train_profit_factor': train_stats.get('profit_factor', 0),
                'test_profit_factor': test_stats.get('profit_factor', 0),

                'train_max_drawdown_pct': train_stats.get(
                    'max_drawdown_pct',
                    0,
                ),
                'test_max_drawdown_pct': test_stats.get(
                    'max_drawdown_pct',
                    0,
                ),

                'selection_score': selection_score,
                'acceptable': acceptable,

                'strategy_spec': json.dumps(
                    spec,
                    ensure_ascii=False,
                ),
                'rules': rules,
                'changes_summary': changes,
            }

            summary_df = pd.DataFrame([row])

            if summary_path.exists():
                summary_df.to_csv(
                    summary_path,
                    mode='a',
                    header=False,
                    index=False,
                    encoding='utf-8-sig',
                )
            else:
                summary_df.to_csv(
                    summary_path,
                    index=False,
                    encoding='utf-8-sig',
                )

            for label, trades in (
                ('train', train_trades),
                ('test', test_trades),
            ):
                if not trades:
                    continue

                trades_path = csv_dir / (
                    f'{symbol}_iteration_{iteration}_{label}_trades.csv'
                )

                rows = []

                for trade in trades:
                    rows.append({
                        'entry_time': trade.get('entry_time'),
                        'exit_time': trade.get('exit_time'),
                        'direction': trade.get('direction'),
                        'entry_price': trade.get('entry_price'),
                        'exit_price': trade.get('exit_price'),
                        'result_pct': trade.get('result_pct'),
                        'exit_reason': trade.get('exit_reason'),
                        'capital_after': trade.get('capital_after'),
                    })

                pd.DataFrame(rows).to_csv(
                    trades_path,
                    index=False,
                    encoding='utf-8-sig',
                )

        except Exception as e:
            logger.warning(
                f'[{symbol}] Не удалось сохранить CSV итерации {iteration}: {e}'
            )