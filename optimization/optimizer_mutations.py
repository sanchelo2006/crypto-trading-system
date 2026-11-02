"""
Мутации стратегий и локальный поиск.
"""

import copy
import json
import logging
import random
from typing import Dict, List, Optional

from strategy_spec import validate_strategy_spec, ensure_engine_defaults

logger = logging.getLogger(__name__)

LOCAL_SEARCH_MUTATIONS = 30


class OptimizerMutations:
    def __init__(self, engine):
        self.engine = engine

    @staticmethod
    def mutate_numeric(spec: Dict) -> Dict:
        """Случайная мутация числовых параметров spec."""
        s = copy.deepcopy(spec)

        def _mut_conditions(conds):
            for cond in conds or []:
                if not isinstance(cond, dict):
                    continue
                ctype = cond.get('type')

                if ctype == 'range' and random.random() < 0.6:
                    try:
                        lo = float(cond.get('min', 0)) + random.uniform(-8, 8)
                        hi = float(cond.get('max', 100)) + random.uniform(-8, 8)
                        lo, hi = sorted([max(0.0, lo), min(100.0, hi)])
                        if hi - lo < 5:
                            hi = min(100.0, lo + 5)
                        cond['min'], cond['max'] = round(lo, 1), round(hi, 1)
                    except Exception:
                        pass

                elif ctype == 'volume' and random.random() < 0.6:
                    try:
                        cond['mult'] = round(
                            max(0.8, float(cond.get('mult', 1.2))
                                + random.uniform(-0.4, 0.3)), 2)
                    except Exception:
                        pass

                elif ctype == 'compare':
                    if (cond.get('left') == 'adx_14'
                            and isinstance(cond.get('right'), (int, float))
                            and random.random() < 0.6):
                        cond['right'] = round(
                            max(10.0, float(cond['right'])
                                + random.uniform(-6, 6)), 1)
                    for side_key in ('left', 'right'):
                        op = cond.get(side_key)
                        if (isinstance(op, dict) and 'mult' in op
                                and random.random() < 0.4):
                            try:
                                op['mult'] = round(
                                    float(op['mult'])
                                    + random.uniform(-0.004, 0.004), 4)
                            except Exception:
                                pass

        for side in ('long', 'short'):
            block = s.get(side)
            if not isinstance(block, dict):
                continue

            # Коридор выходов: если задан exit_corridor, мутируем в его пределах.
            # Иначе — стандартные широкие диапазоны.
            corridor = s.get('exit_corridor')
            if 'stop_loss_atr_mult' in block and random.random() < 0.5:
                if corridor and 'stop_loss_atr_mult' in corridor:
                    lo, hi = corridor['stop_loss_atr_mult']
                    block['stop_loss_atr_mult'] = round(random.uniform(lo, hi), 2)
                else:
                    block['stop_loss_atr_mult'] = round(random.uniform(1.0, 2.5), 2)
            if 'take_profit_atr_mult' in block and random.random() < 0.5:
                if corridor and 'take_profit_atr_mult' in corridor:
                    lo, hi = corridor['take_profit_atr_mult']
                    block['take_profit_atr_mult'] = round(random.uniform(lo, hi), 2)
                else:
                    block['take_profit_atr_mult'] = round(random.uniform(2.0, 6.0), 2)

            mgmt = block.get('management')
            if isinstance(mgmt, dict):
                # Если коридор задан — НЕ мутируем breakeven и trailing
                # (лаборатория зафиксировала их на null).
                if not corridor:
                    if random.random() < 0.4:
                        mgmt['move_to_breakeven_at_tp_pct'] = random.choice(
                            [None, 0.3, 0.5, 0.7])
                    if random.random() < 0.4:
                        mgmt['trailing_atr_mult'] = random.choice(
                            [None, 2.0, 2.5, 3.0, 3.5])
                else:
                    # Холд можно мутировать в пределах коридора
                    if 'max_hold_candles' in corridor and random.random() < 0.4:
                        lo, hi = corridor['max_hold_candles']
                        mgmt['max_hold_candles'] = random.randint(int(lo), int(hi))

            if not s.get("frozen_entries"):
                _mut_conditions(block.get('entry'))
                for group in block.get('entry_any') or []:
                    _mut_conditions(group)

        _mut_conditions(s.get('filters'))
        return s

    def local_search(
        self, *, spec, rules, splits, full_df, symbol,
        initial_capital, commission_pct, n_mutations,
        rejected_fingerprints, base_train_score,
        evaluate_func, train_score_func, spec_fingerprint_func,
    ) -> Optional[tuple]:
        """
        Локальный поиск: N мутаций, лучшая по TRAIN-score.
        Возвращает (best_candidate, best_eval) или None.
        """
        best_candidate = None
        best_eval = None
        best_score = base_train_score

        for _ in range(n_mutations):
            candidate = ensure_engine_defaults(self.mutate_numeric(spec))
            if validate_strategy_spec(candidate) is not None:
                continue
            if spec_fingerprint_func(candidate) in rejected_fingerprints:
                continue

            evaluation = evaluate_func(
                spec=candidate, rules=rules, splits=splits, full_df=full_df,
                symbol=symbol, initial_capital=initial_capital,
                commission_pct=commission_pct,
            )
            if 'error' in evaluation:
                continue

            score = train_score_func(
                evaluation['aggregated']['train_stats'])
            if score > best_score:
                best_score = score
                best_candidate = candidate
                best_eval = evaluation

        if best_candidate is None:
            return None
        return best_candidate, best_eval