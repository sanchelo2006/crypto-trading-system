"""
Итеративный оптимизатор стратегии с train/test walk-forward разбиением.

После рефакторинга: основной координирующий класс,
вся логика вынесена в модули-помощники.
"""

import copy
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from strategy_spec import (
    validate_strategy_spec,
    default_strategy_spec,
    ensure_engine_defaults,
)

from optimization.optimizer_data_utils import OptimizerDataUtils
from optimization.optimizer_scoring import OptimizerScoring
from optimization.optimizer_robustness import (
    OptimizerRobustness,
    ROBUST_MIN_PROFITABLE_WINDOWS_RATIO,
    ROBUST_MIN_POSITIVE_EXPECTANCY_WINDOWS_RATIO,
    ROBUST_MIN_PF_PASS_WINDOWS_RATIO,
    ROBUST_MIN_MEDIAN_WINDOW_RETURN_PCT,
    ROBUST_MAX_WORST_WINDOW_RETURN_PCT,
    ROBUST_MIN_STABILITY_SCORE,
)
from optimization.optimizer_mutations import OptimizerMutations, LOCAL_SEARCH_MUTATIONS
from optimization.optimizer_summary import OptimizerSummary
from optimization.optimizer_holdout import OptimizerHoldout
from optimization.optimization_report import OptimizationReport

logger = logging.getLogger(__name__)

MIN_ACCEPT_TEST_TRADES = 10
PROGRAM_MUTATION_ATTEMPTS = 20
BLOCK_FRAGILE_BEST_CANDIDATE = True


class BacktestOptimizer:
    def __init__(self, engine, ai_client, memory):
        self.engine = engine
        self.ai_client = ai_client
        self.memory = memory

        # Инициализация модулей-помощников
        self.data_utils = OptimizerDataUtils(engine)
        self.scoring = OptimizerScoring()
        self.robustness = OptimizerRobustness()
        self.mutations = OptimizerMutations(engine)
        self.summary = OptimizerSummary()
        self.holdout = OptimizerHoldout(engine)

    @staticmethod
    def _spec_fingerprint(spec: Dict) -> str:
        """Отпечаток spec для дедупликации."""
        def _norm(x):
            if isinstance(x, dict):
                return {k: _norm(v) for k, v in sorted(x.items())}
            if isinstance(x, list):
                return [_norm(v) for v in x]
            if isinstance(x, bool):
                return x
            if isinstance(x, (int, float)):
                return round(float(x), 2)
            return x

        return json.dumps(_norm(spec), sort_keys=True, ensure_ascii=False)

    @staticmethod
    def _inherit_engine_controls(new_spec: Dict, base_spec: Dict) -> Dict:
        """Сохраняет служебные блоки движка (risk_controls, management, frozen_entries)."""
        if not isinstance(new_spec, dict):
            return base_spec
        merged = json.loads(json.dumps(new_spec, ensure_ascii=False))
        if isinstance(base_spec, dict):
            if "risk_controls" in base_spec and "risk_controls" not in merged:
                merged["risk_controls"] = base_spec["risk_controls"]
            if base_spec.get("frozen_entries"):
                merged["frozen_entries"] = True
            if "exit_corridor" in base_spec:
                merged["exit_corridor"] = base_spec["exit_corridor"]
            for side in ("long", "short"):
                base_side = base_spec.get(side)
                new_side = merged.get(side)
                if isinstance(base_side, dict) and isinstance(new_side, dict):
                    if "management" in base_side and "management" not in new_side:
                        new_side["management"] = base_side["management"]
        return merged

    @staticmethod
    def _enforce_frozen_entries(candidate: Dict, base: Dict) -> Dict:
        """
        Если base имеет флаг frozen_entries, принудительно восстанавливает
        условия входа из base в candidate. Риск-параметры кандидата сохраняются.
        """
        if not isinstance(base, dict) or not base.get("frozen_entries"):
            return candidate
        if not isinstance(candidate, dict):
            return candidate
        for side in ("long", "short"):
            base_side = base.get(side)
            cand_side = candidate.get(side)
            if isinstance(base_side, dict):
                if not isinstance(cand_side, dict):
                    candidate[side] = base_side
                else:
                    for key in ("entry", "entry_any"):
                        if key in base_side:
                            cand_side[key] = base_side[key]
                        elif key in cand_side:
                            del cand_side[key]
            elif side in candidate:
                del candidate[side]
        candidate["frozen_entries"] = True
        return candidate

    @staticmethod
    def _enforce_exit_corridor(candidate: Dict, base: Dict) -> Dict:
        """
        Если base имеет exit_corridor, обрезает стопы/тейки/холд в его пределы.
        Трейлинг и безубыток фиксируются на null (лабораторный запрет).
        """
        if not isinstance(base, dict) or "exit_corridor" not in base:
            return candidate
        if not isinstance(candidate, dict):
            return candidate
        
        corridor = base["exit_corridor"]
        for side in ("long", "short"):
            cand_side = candidate.get(side)
            if not isinstance(cand_side, dict):
                continue
            
            # Стоп
            if "stop_loss_atr_mult" in cand_side and "stop_loss_atr_mult" in corridor:
                lo, hi = corridor["stop_loss_atr_mult"]
                val = cand_side["stop_loss_atr_mult"]
                cand_side["stop_loss_atr_mult"] = round(max(lo, min(hi, val)), 2)
            
            # Тейк
            if "take_profit_atr_mult" in cand_side and "take_profit_atr_mult" in corridor:
                lo, hi = corridor["take_profit_atr_mult"]
                val = cand_side["take_profit_atr_mult"]
                cand_side["take_profit_atr_mult"] = round(max(lo, min(hi, val)), 2)
            
            # Менеджмент
            mgmt = cand_side.get("management")
            if isinstance(mgmt, dict):
                # Трейлинг и безубыток — всегда null (лабораторный запрет)
                mgmt["trailing_atr_mult"] = None
                mgmt["move_to_breakeven_at_tp_pct"] = None
                # Холд
                if "max_hold_candles" in mgmt and "max_hold_candles" in corridor:
                    lo, hi = corridor["max_hold_candles"]
                    val = mgmt["max_hold_candles"]
                    mgmt["max_hold_candles"] = max(int(lo), min(int(hi), int(val)))
        
        candidate["exit_corridor"] = corridor
        return candidate

    def _evaluate_spec_on_splits(
        self, *, spec, rules, splits, full_df, symbol,
        initial_capital, commission_pct
    ) -> Dict:
        """Оценка spec на всех TRAIN/TEST окнах."""
        fold_results = []
        for fold_index, split in enumerate(splits, start=1):
            train_df = full_df[
                (full_df['timestamp'] >= split['train_start']) &
                (full_df['timestamp'] < split['train_end'])
            ].copy()
            test_df = full_df[
                (full_df['timestamp'] >= split['test_start']) &
                (full_df['timestamp'] <= split['test_end'])
            ].copy()

            if len(train_df) < 60 or len(test_df) < 60:
                return {'error': (
                    f'Недостаточно свечей в окне {fold_index}: '
                    f'TRAIN={len(train_df)}, TEST={len(test_df)}.'
                )}

            train_result = self.engine.run_on_dataframe(
                df=train_df, symbol=symbol, strategy_spec=spec,
                initial_capital=initial_capital, commission_pct=commission_pct,
                rules_text=rules,
                start_date=split['train_start'], end_date=split['train_end'],
            )
            if 'error' in train_result:
                return {'error': f'TRAIN, окно {fold_index}: {train_result["error"]}'}

            test_result = self.engine.run_on_dataframe(
                df=test_df, symbol=symbol, strategy_spec=spec,
                initial_capital=initial_capital, commission_pct=commission_pct,
                rules_text=rules,
                start_date=split['test_start'], end_date=split['test_end'],
            )
            if 'error' in test_result:
                return {'error': f'TEST, окно {fold_index}: {test_result["error"]}'}

            fold_results.append({
                'split': split,
                'train_result': train_result,
                'test_result': test_result,
            })

        return {
            'fold_results': fold_results,
            'aggregated': self._aggregate_fold_results(fold_results),
        }

    @staticmethod
    def _aggregate_fold_results(fold_results: List[Dict]) -> Dict:
        """
        TEST агрегируется по всем окнам (они не пересекаются).
        TRAIN берётся ТОЛЬКО из последнего окна.
        """
        test_trades_all = []
        test_stats_list = []

        for fold in fold_results:
            test_result = fold.get('test_result', {})
            test_trades_all.extend(test_result.get('trades', []))
            test_stats_list.append(test_result.get('stats', {}))

        last_train = fold_results[-1].get('train_result', {})
        train_trades_all = list(last_train.get('trades', []))
        train_stats = OptimizerScoring.merge_stats(
            stats_list=[last_train.get('stats', {})],
            trades_all=train_trades_all,
        )

        test_stats = OptimizerScoring.merge_stats(
            stats_list=test_stats_list,
            trades_all=test_trades_all,
        )

        return {
            'train_stats': train_stats,
            'test_stats': test_stats,
            'train_trades': train_trades_all,
            'test_trades': test_trades_all,
        }

    def _build_score_breakdown(
        self,
        *,
        train_stats: Dict,
        test_stats: Dict,
        train_trades: List[Dict],
        test_trades: List[Dict],
        raw_selection_score: float,
        window_penalty: float,
        final_selection_score: float,
        acceptable: bool,
        window_robustness: Dict,
    ) -> Dict:
        """Подробная диагностика score."""
        train_return = float(train_stats.get("return_pct", 0) or 0)
        test_return = float(test_stats.get("return_pct", 0) or 0)

        train_expectancy = float(
            train_stats.get("expectancy_pct", train_stats.get("avg_trade_pct", 0)) or 0
        )
        test_expectancy = float(
            test_stats.get("expectancy_pct", test_stats.get("avg_trade_pct", 0)) or 0
        )

        train_pf = float(train_stats.get("profit_factor", 0) or 0)
        test_pf = float(test_stats.get("profit_factor", 0) or 0)

        train_dd = float(train_stats.get("max_drawdown_pct", 0) or 0)
        test_dd = float(test_stats.get("max_drawdown_pct", 0) or 0)

        train_trades_count = int(train_stats.get("total_trades", 0) or 0)
        test_trades_count = int(test_stats.get("total_trades", 0) or 0)

        train_minus_test_return = round(train_return - test_return, 4)
        train_minus_test_expectancy = round(train_expectancy - test_expectancy, 4)
        train_minus_test_pf = round(train_pf - test_pf, 4)

        train_profit_concentration = self.scoring.profit_concentration(train_trades)
        test_profit_concentration = self.scoring.profit_concentration(test_trades)

        return {
            "acceptable": acceptable,
            "raw_selection_score": round(float(raw_selection_score), 6),
            "window_stability_penalty": round(float(window_penalty), 6),
            "final_selection_score": round(float(final_selection_score), 6),

            "train_total_trades": train_trades_count,
            "test_total_trades": test_trades_count,

            "train_return_pct": round(train_return, 6),
            "test_return_pct": round(test_return, 6),

            "train_expectancy_pct": round(train_expectancy, 6),
            "test_expectancy_pct": round(test_expectancy, 6),

            "train_profit_factor": round(train_pf, 6),
            "test_profit_factor": round(test_pf, 6),

            "train_max_drawdown_pct": round(train_dd, 6),
            "test_max_drawdown_pct": round(test_dd, 6),

            "train_minus_test_return_pct": train_minus_test_return,
            "train_minus_test_expectancy_pct": train_minus_test_expectancy,
            "train_minus_test_profit_factor": train_minus_test_pf,

            "train_profit_concentration": train_profit_concentration,
            "test_profit_concentration": test_profit_concentration,

            "windows_total": int(window_robustness.get("windows_total", 0) or 0),
            "profitable_windows_ratio": float(
                window_robustness.get("profitable_windows_ratio", 0) or 0
            ),
            "positive_expectancy_windows_ratio": float(
                window_robustness.get("positive_expectancy_windows_ratio", 0) or 0
            ),
            "pf_pass_windows_ratio": float(
                window_robustness.get("pf_pass_windows_ratio", 0) or 0
            ),
            "median_window_return_pct": float(
                window_robustness.get("median_window_return_pct", 0) or 0
            ),
            "worst_window_return_pct": float(
                window_robustness.get("worst_window_return_pct", 0) or 0
            ),
            "window_return_std": float(
                window_robustness.get("return_std", 0) or 0
            ),
            "window_stability_score": float(
                window_robustness.get("stability_score", 0) or 0
            ),
        }

    def _build_optimization_summary(
        self, *, train_stats: Dict, test_stats: Dict,
        train_trades: List[Dict], test_trades: List[Dict],
        iteration: int, rejected_count: int, history: List[Dict],
        fold_results: List[Dict],
        robust_min_profitable_windows_ratio: float,
        robust_min_positive_expectancy_windows_ratio: float,
        robust_min_pf_pass_windows_ratio: float,
        robust_min_median_window_return_pct: float,
        robust_max_worst_window_return_pct: float,
        robust_min_stability_score: float,
    ) -> Dict:
        """Сводка для AI."""
        def compact(stats: Dict) -> Dict:
            return {
                "total_trades": stats.get("total_trades", 0),
                "win_rate": stats.get("win_rate", 0),
                "return_pct": stats.get("return_pct", 0),
                "total_pnl_pct": stats.get("total_pnl_pct", 0),
                "profit_factor": stats.get("profit_factor", 0),
                "avg_trade_pct": stats.get("avg_trade_pct", 0),
                "avg_win_pct": stats.get("avg_win_pct", 0),
                "avg_loss_pct": stats.get("avg_loss_pct", 0),
                "expectancy_pct": stats.get("expectancy_pct", stats.get("avg_trade_pct", 0)),
                "max_drawdown_pct": stats.get("max_drawdown_pct", 0),
                "long_trades": stats.get("long_trades", 0),
                "short_trades": stats.get("short_trades", 0),
            }

        window_robustness = self.robustness.compute_window_robustness(fold_results)
        window_diagnostics = self.robustness.compute_window_diagnostics(window_robustness)
        window_robustness["stability_score"] = window_diagnostics.get("stability_score", 0.0)

        robustness_assessment = self.robustness.assess_window_robustness(
            window_robustness,
            min_profitable_windows_ratio=robust_min_profitable_windows_ratio,
            min_positive_expectancy_windows_ratio=robust_min_positive_expectancy_windows_ratio,
            min_pf_pass_windows_ratio=robust_min_pf_pass_windows_ratio,
            min_median_window_return_pct=robust_min_median_window_return_pct,
            max_worst_window_return_pct=robust_max_worst_window_return_pct,
            min_stability_score=robust_min_stability_score,
        )

        window_penalty, window_penalty_reasons = self.robustness.window_stability_penalty(
            window_robustness
        )

        raw_selection_score, acceptable, _, _ = self.scoring.score_and_flags(
            train_stats,
            test_stats,
            test_trades,
            MIN_ACCEPT_TEST_TRADES,
            self.scoring.profit_concentration,
        )
        final_selection_score = raw_selection_score - window_penalty

        score_breakdown = self._build_score_breakdown(
            train_stats=train_stats,
            test_stats=test_stats,
            train_trades=train_trades,
            test_trades=test_trades,
            raw_selection_score=raw_selection_score,
            window_penalty=window_penalty,
            final_selection_score=final_selection_score,
            acceptable=acceptable,
            window_robustness=window_robustness,
        )
        score_breakdown["robustness_status"] = robustness_assessment.get("status", "unstable")
        score_breakdown["robustness_passed"] = robustness_assessment.get("passed", False)
        score_breakdown["robustness_fragile"] = robustness_assessment.get("fragile", False)
        score_breakdown["robustness_warnings"] = robustness_assessment.get("warnings", [])
        score_breakdown["robustness_reasons"] = robustness_assessment.get("reasons", [])

        # Собираем текстовые описания последних 3 НЕУДАЧНЫХ изменений вместо сырых JSON
        rejected_changes = []
        for h in reversed(history):
            if not h.get('acceptable') and h.get('changes_summary'):
                rejected_changes.append(f"[{h.get('iteration')}] {h['changes_summary']}")
            if len(rejected_changes) >= 3:
                break

        return {
            "iteration": iteration,
            "rejected_specs_count": rejected_count,
            "selection_goal": (
                "Максимизировать суммарный P&L на TEST (total_pnl_pct) при "
                "умеренной просадке. Доходность = expectancy × число сделок: "
                "и слишком мало сделок, и отрицательный expectancy — одинаково "
                "провал. Кандидат пригоден только при PF>1.1 на TEST, PF>1.0 на "
                "TRAIN, expectancy>0 на обоих и не менее 10 сделок на TEST. "
                "Изучи iteration_history: продолжай изменения, повышавшие "
                "selection_score, и не повторяй ухудшавшие."
            ),
            "train_stats": compact(train_stats),
            "test_stats": compact(test_stats),
            "diagnostics": {
                "train_minus_test_return_pct": round(
                    float(train_stats.get("return_pct", 0)) - float(test_stats.get("return_pct", 0)), 4
                ),
                "train_minus_test_expectancy_pct": round(
                    float(train_stats.get("expectancy_pct", 0)) - float(test_stats.get("expectancy_pct", 0)), 4
                ),
                "test_trade_count_warning": (
                    "too_few_trades" if int(test_stats.get("total_trades", 0)) < MIN_ACCEPT_TEST_TRADES else "ok"
                ),
                "window_stability_score": window_diagnostics.get("stability_score", 0),
                "robustness_status": robustness_assessment.get("status", "unstable"),
                "worst_window_return_pct": window_robustness.get("worst_window_return_pct", 0),
                "window_stability_penalty": window_penalty,
                "raw_selection_score": raw_selection_score,
                "final_selection_score": final_selection_score,
                "acceptable": acceptable,
            },
            "iteration_history": self.summary.compact_history(history),
            "rejected_changes": rejected_changes,  # <--- НОВОЕ ПОЛЕ
            "train_trade_summary": self.summary.summarize_trades(train_trades, self.summary.derived),
            "test_trade_summary": self.summary.summarize_trades(test_trades, self.summary.derived),
        }

    def run(
        self,
        symbol: str,
        category: str,
        max_iterations: int,
        initial_capital: float,
        commission_pct: float,
        csv_path: str,
        start_date: datetime,
        end_date: datetime,
        train_ratio: float = 0.7,
        walk_forward_folds: int = 6,
        phase0_screen: bool = False,
        min_train_trades: int = 0,
        min_test_trades: int = 0,
        seed_archetype: Optional[str] = None,
        freeze_entries: Optional[bool] = None,
        save_best_to_memory: bool = True,
        progress_callback: Optional[Callable] = None,
        local_search_mutations: int = LOCAL_SEARCH_MUTATIONS,
        allow_weak_holdout_save: bool = False,
        robust_min_profitable_windows_ratio: float = ROBUST_MIN_PROFITABLE_WINDOWS_RATIO,
        robust_min_positive_expectancy_windows_ratio: float = ROBUST_MIN_POSITIVE_EXPECTANCY_WINDOWS_RATIO,
        robust_min_pf_pass_windows_ratio: float = ROBUST_MIN_PF_PASS_WINDOWS_RATIO,
        robust_min_median_window_return_pct: float = ROBUST_MIN_MEDIAN_WINDOW_RETURN_PCT,
        robust_max_worst_window_return_pct: float = ROBUST_MAX_WORST_WINDOW_RETURN_PCT,
        robust_min_stability_score: float = ROBUST_MIN_STABILITY_SCORE,
        block_fragile_best_candidate: bool = BLOCK_FRAGILE_BEST_CANDIDATE,
    ) -> Dict:
        """
        Основной метод оптимизации с полным циклом:
        1. Загрузка данных
        2. Разбиение на optimization/holdout
        3. TRAIN/TEST walk-forward окна
        4. Итеративная оптимизация через AI + локальный поиск
        5. Holdout-проверка финального кандидата
        6. Сохранение лучшей стратегии
        """

        self.last_report_path = ""
        report = OptimizationReport(_ROOT, symbol)
        self.last_report_path = str(report.path)

        def emit(message: str):
            logger.info(f'[{symbol}] {message}')
            report.line(message)
            if progress_callback:
                progress_callback({'message': message})

        emit(f'Подробный отчёт оптимизации: {report.path}')

        cm = self.memory.get(symbol)
        min_test_trades = max(int(min_test_trades or 0), MIN_ACCEPT_TEST_TRADES)

        current_spec = cm.get_state().get('strategy_spec')
        if not current_spec or validate_strategy_spec(current_spec) is not None:
            current_spec = default_strategy_spec()
            current_spec = ensure_engine_defaults(current_spec)
        current_rules = cm.get_rules() or ''

        # Управление заморозкой входов из UI.
        # True  → принудительно заморозить входы.
        # False → принудительно разморозить (удалить флаг из спеки).
        # None  → использовать то, что уже в спеке (обратная совместимость).
        if freeze_entries is True:
            current_spec["frozen_entries"] = True
        elif freeze_entries is False:
            current_spec.pop("frozen_entries", None)

        # Явный сид-архетип требует Фазы 0: именно там берутся его лучшие
        # параметры на оптимизационном периоде.
        if seed_archetype and not phase0_screen:
            phase0_screen = True
            emit('Задан сид-архетип: включаю Фазу 0 для получения его лучших параметров.')

        csv_dir = Path(csv_path)

        emit(
            f'Оптимизация: период {start_date.strftime("%Y-%m-%d")} → '
            f'{end_date.strftime("%Y-%m-%d")}, итераций: {max_iterations}, '
            f'walk-forward окон: {walk_forward_folds}, '
            f'локальных мутаций на итерацию: {local_search_mutations}.'
        )

        # 1. Загрузка данных
        emit('Загрузка исторических свечей и расчёт индикаторов...')
        prepared = self.data_utils.load_prepared_dataframe(
            symbol=symbol, category=category,
            start_date=start_date, end_date=end_date,
        )
        if 'error' in prepared:
            return {'error': prepared['error'], 'history': []}

        full_df = prepared['df']
        emit(f'Исторические данные готовы: {prepared["candles_count"]} свечей.')

        # --- БАЗОВАЯ ЛИНИЯ: buy&hold за период (дёшево, всегда) ---
        from optimization.baselines import buy_and_hold
        bh = buy_and_hold(full_df)
        emit(
            f'BASELINE buy&hold за период: {bh["return_pct"]:+.2f}%, '
            f'DD={bh["max_drawdown_pct"]:.1f}%.'
        )
        report.section("BASELINE / BUY&HOLD", bh)

        # --- ФАЗА 0: скрининг архетипов (по запросу, долго) ---
        if phase0_screen:
            from optimization.screen_runner import run_screen
            screen_text, screen_results = run_screen(
                engine=self.engine,
                df=full_df,
                symbol=symbol,
                commission_pct=commission_pct,
                initial_capital=initial_capital,
            )
            for _line in screen_text.splitlines():
                emit(_line)
            report.section("ФАЗА 0 / СКРИНИНГ АРХЕТИПОВ", screen_text)

            # --- Выбор стартового архетипа ---
            # Явный сид из UI применяется БЕЗУСЛОВНО: пользователь сознательно
            # задаёт стартовую точку, и она не обязана обгонять buy&hold на
            # скрининге (иначе сид молча подменялся спекой из памяти).
            # Авто-выбор работает только когда сид не задан и требует PF > 1.0
            # и превосходства над базовой линией.
            bh_return = float(bh.get("return_pct", 0.0) or 0.0)

            forced_arch = None
            if seed_archetype:
                forced_arch = (screen_results.get("by_name") or {}).get(seed_archetype)
                if forced_arch is None:
                    emit(
                        f'ФАЗА 0: архетип {seed_archetype} не найден в результатах '
                        f'скрининга — обычный отбор.'
                    )

            best_arch = forced_arch if forced_arch is not None else screen_results.get("overall")

            beats_baseline = (
                best_arch is not None
                and best_arch["pf"] > 1.0
                and best_arch["return_pct"] > bh_return
            )

            if forced_arch is not None or beats_baseline:
                emit(
                    f'ФАЗА 0: старт с архетипа {best_arch["name"]} '
                    f'(P&L={best_arch["return_pct"]:+.2f}%, PF={best_arch["pf"]:.2f}, '
                    f'сделок={best_arch["trades"]}). AI будет улучшать его.'
                )
                current_spec = ensure_engine_defaults(best_arch["spec"])
                # Повторно применяем заморозку после Фазы 0, потому что
                # ensure_engine_defaults или скрининг могли потерять флаг.
                if freeze_entries is True:
                    current_spec["frozen_entries"] = True
                elif freeze_entries is False:
                    current_spec.pop("frozen_entries", None)
                current_rules = f"Стартовая стратегия на базе архетипа {best_arch['name']}."
            else:
                emit(
                    'ФАЗА 0: ни один архетип не превзошёл базовую линию с PF > 1.0. '
                    'Используем текущую стратегию из памяти.'
                )
        # 2. Разбиение на optimization/holdout
        optimization_df, holdout_df, holdout_meta = self.data_utils.split_optimization_and_holdout(
            full_df,
            holdout_ratio=0.15,
            min_holdout_candles=500,
            min_optimization_candles=3000,
        )

        holdout_period = self.data_utils.describe_frame_period(holdout_df)
        optimization_period = self.data_utils.describe_frame_period(optimization_df)

        if holdout_meta.get("enabled"):
            emit(
                f'HOLDOUT зарезервирован: '
                f'{holdout_meta.get("holdout_candles", 0)} свечей '
                f'({holdout_period.get("start")} → {holdout_period.get("end")}).'
            )
        else:
            emit(
                f'HOLDOUT отключён: {holdout_meta.get("reason", "unknown")}.'
            )

        emit(
            f'OPTIMIZATION период: '
            f'{optimization_period.get("start")} → {optimization_period.get("end")}.'
        )

        full_df = optimization_df

        optimization_start_dt, optimization_end_dt = self.data_utils.get_frame_boundary_datetimes(
            optimization_df,
            fallback_start=start_date,
            fallback_end=end_date,
        )

        # 3. TRAIN/TEST окна
        splits = self.data_utils.make_splits(
            start_date=optimization_start_dt,
            end_date=optimization_end_dt,
            train_ratio=train_ratio,
            walk_forward_folds=walk_forward_folds,
        )
        if not splits:
            return {'error': 'Не удалось сформировать TRAIN/TEST разбиение.',
                    'history': []}

        emit(f'Количество split-окон: {len(splits)}.')
        for idx, split in enumerate(splits, start=1):
            emit(
                f'Окно {idx}/{len(splits)}: '
                f'TRAIN {split["train_start"].strftime("%Y-%m-%d")} → '
                f'{split["train_end"].strftime("%Y-%m-%d")} | '
                f'TEST {split["test_start"].strftime("%Y-%m-%d")} → '
                f'{split["test_end"].strftime("%Y-%m-%d")}'
            )

        history = []
        best = None
        rejected_fingerprints = set()
        last_changes = 'исходный spec'

        def record_candidate(label, spec, rules, evaluation, changes):
            """Оценивает кандидата, пишет history/CSV, обновляет best."""
            nonlocal best

            agg = evaluation['aggregated']
            train_stats = agg['train_stats']
            test_stats = agg['test_stats']
            train_trades_all = agg['train_trades']
            test_trades_all = agg['test_trades']

            fold_results = evaluation.get('fold_results', [])
            window_robustness = self.robustness.compute_window_robustness(fold_results)

            raw_selection_score, acceptable, train_exp, test_exp = self.scoring.score_and_flags(
                train_stats, test_stats, test_trades_all, min_test_trades,
                self.scoring.profit_concentration,
            )

            window_diag = self.robustness.compute_window_diagnostics(window_robustness)
            window_robustness["stability_score"] = window_diag.get("stability_score", 0.0)
            window_penalty, penalty_reasons = self.robustness.window_stability_penalty(
                window_robustness
            )
            robustness_assessment = self.robustness.assess_window_robustness(
                window_robustness,
                min_profitable_windows_ratio=robust_min_profitable_windows_ratio,
                min_positive_expectancy_windows_ratio=robust_min_positive_expectancy_windows_ratio,
                min_pf_pass_windows_ratio=robust_min_pf_pass_windows_ratio,
                min_median_window_return_pct=robust_min_median_window_return_pct,
                max_worst_window_return_pct=robust_max_worst_window_return_pct,
                min_stability_score=robust_min_stability_score,
            )

            selection_score = raw_selection_score - window_penalty

            score_breakdown = self._build_score_breakdown(
                train_stats=train_stats,
                test_stats=test_stats,
                train_trades=train_trades_all,
                test_trades=test_trades_all,
                raw_selection_score=raw_selection_score,
                window_penalty=window_penalty,
                final_selection_score=selection_score,
                acceptable=acceptable,
                window_robustness=window_robustness,
            )
            score_breakdown["robustness_status"] = robustness_assessment.get("status", "unstable")
            score_breakdown["robustness_passed"] = robustness_assessment.get("passed", False)
            score_breakdown["robustness_fragile"] = robustness_assessment.get("fragile", False)
            score_breakdown["robustness_warnings"] = robustness_assessment.get("warnings", [])
            score_breakdown["robustness_reasons"] = robustness_assessment.get("reasons", [])

            soft_warnings = []
            if window_robustness.get("profitable_windows_ratio", 0) < 0.5:
                soft_warnings.append("Прибыльных окон < 50%")
            if window_robustness.get("positive_expectancy_windows_ratio", 0) < 0.5:
                soft_warnings.append("Окон с положительным expectancy < 50%")
            if window_robustness.get("pf_pass_windows_ratio", 0) < 0.5:
                soft_warnings.append("Окон с PF > 1 < 50%")
            if window_robustness.get("worst_window_return_pct", 0) < -10:
                soft_warnings.append(
                    f'Худшее окно {window_robustness.get("worst_window_return_pct", 0):+.1f}%'
                )

            history.append({
                'iteration': label,
                'train_trades': train_stats.get('total_trades', 0),
                'test_trades': test_stats.get('total_trades', 0),
                'train_winrate': train_stats.get('win_rate', 0),
                'test_winrate': test_stats.get('win_rate', 0),
                'train_return_pct': train_stats.get('return_pct', 0),
                'test_return_pct': test_stats.get('return_pct', 0),
                'train_expectancy_pct': train_exp,
                'test_expectancy_pct': test_exp,
                'train_profit_factor': train_stats.get('profit_factor', 0),
                'test_profit_factor': test_stats.get('profit_factor', 0),
                'train_max_drawdown_pct': train_stats.get('max_drawdown_pct', 0),
                'test_max_drawdown_pct': test_stats.get('max_drawdown_pct', 0),
                'test_total_pnl_pct': test_stats.get('total_pnl_pct', 0),
                'selection_score': selection_score,
                'raw_selection_score': raw_selection_score,
                'window_stability_penalty': window_penalty,
                'window_penalty_reasons': penalty_reasons,
                'score_breakdown': score_breakdown,
                'acceptable': acceptable,
                'changes_summary': changes,
                'spec': spec,
                'rules': rules,
                'window_robustness': window_robustness,
                'window_diagnostics': window_diag,
                'window_robustness_assessment': robustness_assessment,
                'soft_warnings': soft_warnings,
            })

            self.summary.save_round(
                csv_dir=csv_dir, symbol=symbol, iteration=label,
                train_stats=train_stats, test_stats=test_stats,
                spec=spec, rules=rules, changes=changes,
                train_trades=train_trades_all, test_trades=test_trades_all,
                selection_score=selection_score, acceptable=acceptable,
            )

            emit(
                f'[{label}] TRAIN: сделок={train_stats.get("total_trades", 0)}, '
                f'P&L={train_stats.get("total_pnl_pct", 0):+.2f}%, '
                f'Exp={train_exp:+.4f}%, PF={train_stats.get("profit_factor", 0):.2f}.'
            )
            emit(
                f'[{label}] TEST: сделок={test_stats.get("total_trades", 0)}, '
                f'P&L={test_stats.get("total_pnl_pct", 0):+.2f}%, '
                f'Exp={test_exp:+.4f}%, PF={test_stats.get("profit_factor", 0):.2f}, '
                f'DD={test_stats.get("max_drawdown_pct", 0):.2f}%, '
                f'RawScore={raw_selection_score:+.4f}, '
                f'WindowPenalty={window_penalty:+.4f}, '
                f'Score={selection_score:+.4f}, '
                f'{"ПРИНЯТ" if acceptable else "не принят"}.'
            )

            if soft_warnings:
                emit(f'[{label}] SOFT WARNINGS: ' + ' | '.join(soft_warnings))

            if window_diag.get("recommendations"):
                emit(
                    f'[{label}] RECOMMENDATIONS: ' +
                    ' | '.join(window_diag.get("recommendations", []))
                )

            if window_penalty > 0:
                emit(
                    f'[{label}] WINDOW PENALTY: {window_penalty:+.4f}'
                    + (
                        f' | причины: {" | ".join(penalty_reasons)}'
                        if penalty_reasons else ''
                    )
                )

            emit(
                f'[{label}] SCORE BREAKDOWN: '
                f'test_return={score_breakdown.get("test_return_pct", 0):+.2f}%, '
                f'test_exp={score_breakdown.get("test_expectancy_pct", 0):+.4f}%, '
                f'test_pf={score_breakdown.get("test_profit_factor", 0):.2f}, '
                f'test_dd={score_breakdown.get("test_max_drawdown_pct", 0):.2f}%, '
                f'gap_ret={score_breakdown.get("train_minus_test_return_pct", 0):+.2f}, '
                f'gap_exp={score_breakdown.get("train_minus_test_expectancy_pct", 0):+.4f}, '
                f'gap_pf={score_breakdown.get("train_minus_test_profit_factor", 0):+.2f}, '
                f'test_conc={score_breakdown.get("test_profit_concentration", 0):.2f}, '
                f'raw={score_breakdown.get("raw_selection_score", 0):+.4f}, '
                f'penalty={score_breakdown.get("window_stability_penalty", 0):+.4f}, '
                f'final={score_breakdown.get("final_selection_score", 0):+.4f}.'
            )

            emit(
                f'[{label}] ROBUSTNESS: '
                f'status={robustness_assessment.get("status", "unstable")}, '
                f'passed={robustness_assessment.get("passed", False)}, '
                f'fragile={robustness_assessment.get("fragile", False)}, '
                f'profitable={window_robustness.get("profitable_windows_ratio", 0):.0%}, '
                f'positive_exp={window_robustness.get("positive_expectancy_windows_ratio", 0):.0%}, '
                f'pf_pass={window_robustness.get("pf_pass_windows_ratio", 0):.0%}, '
                f'worst={window_robustness.get("worst_window_return_pct", 0):+.2f}%, '
                f'median={window_robustness.get("median_window_return_pct", 0):+.2f}%, '
                f'stability={window_robustness.get("stability_score", 0):.3f}'
            )

            can_be_best = acceptable
            is_fragile = robustness_assessment.get("fragile", False)

            if can_be_best and (best is None or selection_score > best['selection_score']):
                # Если новый кандидат fragile, а текущий best — robust, требуем заметного улучшения
                if is_fragile and best is not None:
                    current_robust = best.get("window_robustness_assessment", {}).get("status") == "robust"
                    if current_robust and selection_score < best['selection_score'] * 1.10:
                        # Fragile должен быть на 10% лучше, чтобы вытеснить robust
                        emit(f'[{label}] Fragile-кандидат не вытесняет robust best: нужно +10% по score.')
                        can_be_best = False

            if can_be_best and (best is None or selection_score > best['selection_score']):
                best = {
                    'iteration': label,
                    'selection_score': selection_score,
                    'spec': spec,
                    'rules': rules,
                    'train_winrate': train_stats.get('win_rate', 0),
                    'test_winrate': test_stats.get('win_rate', 0),
                    'aggregated_train_stats': train_stats,
                    'aggregated_test_stats': test_stats,
                    'train_trades': train_trades_all,
                    'test_trades': test_trades_all,
                    'train_result': evaluation['fold_results'][-1]['train_result'],
                    'test_result': evaluation['fold_results'][-1]['test_result'],
                    'fold_results': fold_results,
                    'window_robustness': window_robustness,
                    'window_diagnostics': window_diag,
                    'soft_warnings': soft_warnings,
                    'raw_selection_score': raw_selection_score,
                    'window_stability_penalty': window_penalty,
                    'window_penalty_reasons': penalty_reasons,
                    'score_breakdown': score_breakdown,
                    'window_robustness_assessment': robustness_assessment,
                    'holdout_summary': None,
                }
                emit(f'[{label}] Новый лучший кандидат по TEST-score.')

            return train_stats, test_stats, train_trades_all, test_trades_all, selection_score, fold_results

        # 4. Основной цикл итераций
        for iteration in range(1, max_iterations + 1):
            emit(f'Итерация {iteration}/{max_iterations}: прогон TRAIN/TEST...')
            rejected_fingerprints.add(self._spec_fingerprint(current_spec))

            evaluation = self._evaluate_spec_on_splits(
                spec=current_spec, rules=current_rules, splits=splits,
                full_df=full_df, symbol=symbol,
                initial_capital=initial_capital, commission_pct=commission_pct,
            )
            if 'error' in evaluation:
                return {'error': evaluation['error'], 'history': history}

            (
                train_stats,
                test_stats,
                train_trades_all,
                test_trades_all,
                selection_score,
                current_fold_results,
            ) = record_candidate(
                iteration, current_spec, current_rules, evaluation, last_changes)

            # Локальный поиск
            if local_search_mutations > 0:
                refined = self.mutations.local_search(
                    spec=current_spec, rules=current_rules, splits=splits,
                    full_df=full_df, symbol=symbol,
                    initial_capital=initial_capital,
                    commission_pct=commission_pct,
                    n_mutations=local_search_mutations,
                    rejected_fingerprints=rejected_fingerprints,
                    base_train_score=self.scoring.train_score(train_stats),
                    evaluate_func=self._evaluate_spec_on_splits,
                    train_score_func=self.scoring.train_score,
                    spec_fingerprint_func=self._spec_fingerprint,
                )
                if refined is not None:
                    r_spec, r_eval = refined
                    rejected_fingerprints.add(self._spec_fingerprint(r_spec))
                    label = f'{iteration}-local'
                    emit(f'Локальный поиск улучшил TRAIN-score, фиксирую {label}.')

                    (
                        train_stats,
                        test_stats,
                        train_trades_all,
                        test_trades_all,
                        selection_score,
                        current_fold_results,
                    ) = record_candidate(
                        label, r_spec, current_rules, r_eval, 'Локальная мутация числовых параметров.')

                    current_spec = r_spec

            if iteration >= max_iterations:
                emit('Достигнут лимит итераций оптимизации.')
                break

            # Hill-climbing
            base_spec = current_spec
            base_rules = current_rules
            base_train_stats = train_stats
            base_test_stats = test_stats
            base_train_trades = train_trades_all
            base_test_trades = test_trades_all
            base_fold_results = current_fold_results

            if best is not None:
                threshold = best['selection_score'] - max(
                    1.0, abs(best['selection_score']) * 0.3)
                if selection_score < threshold:
                    base_spec = best['spec']
                    base_rules = best['rules']
                    base_train_stats = best['aggregated_train_stats']
                    base_test_stats = best['aggregated_test_stats']
                    base_train_trades = best['train_trades']
                    base_test_trades = best['test_trades']
                    base_fold_results = best['fold_results']
                    emit('Откат к лучшему spec.')

            # AI формирует следующий spec
            summary = self._build_optimization_summary(
                train_stats=base_train_stats,
                test_stats=base_test_stats,
                train_trades=base_train_trades,
                test_trades=base_test_trades,
                iteration=iteration,
                rejected_count=len(rejected_fingerprints),
                history=history,
                fold_results=base_fold_results,
                robust_min_profitable_windows_ratio=robust_min_profitable_windows_ratio,
                robust_min_positive_expectancy_windows_ratio=robust_min_positive_expectancy_windows_ratio,
                robust_min_pf_pass_windows_ratio=robust_min_pf_pass_windows_ratio,
                robust_min_median_window_return_pct=robust_min_median_window_return_pct,
                robust_max_worst_window_return_pct=robust_max_worst_window_return_pct,
                robust_min_stability_score=robust_min_stability_score,
            )
            emit(f'Итерация {iteration}: AI анализирует результаты...')

            new_spec = None
            changes_summary = 'AI изменила strategy_spec.'
            new_rules_candidate = base_rules
            max_ai_attempts = 3

            for attempt in range(1, max_ai_attempts + 1):
                extra_instruction = "" if attempt == 1 else (
                    "Предыдущее предложение было невалидным или совпало с уже "
                    "проверенным. Внеси качественно иное изменение."
                )

                report.section(
                    f"ИТЕРАЦИЯ {iteration} / ВХОДНЫЕ ДАННЫЕ AI",
                    {
                        "base_spec": base_spec,
                        "base_rules": base_rules,
                        "analysis_summary": summary,
                        "rejected_fingerprints": list(rejected_fingerprints),
                        "extra_instruction": extra_instruction,
                    },
                )

                optimization_result = self.ai_client.optimize_strategy(
                    analysis_summary=summary,
                    current_spec=base_spec,
                    current_rules=base_rules,
                    rejected_specs=list(rejected_fingerprints),
                    extra_instruction=extra_instruction,
                )

                report.section(
                    f"ИТЕРАЦИЯ {iteration} / ОТВЕТ AI",
                    optimization_result,
                )

                if not optimization_result:
                    continue

                candidate_spec = optimization_result.get('strategy_spec')
                if not candidate_spec:
                    continue

                validation_error = validate_strategy_spec(candidate_spec)
                if validation_error is not None:
                    continue

                candidate_spec = self._inherit_engine_controls(
                    new_spec=candidate_spec, base_spec=base_spec)
                candidate_spec = self._enforce_frozen_entries(
                    candidate_spec, base_spec)
                candidate_spec = self._enforce_exit_corridor(
                    candidate_spec, base_spec)
                candidate_spec = ensure_engine_defaults(candidate_spec)
                # ensure_engine_defaults может стереть флаги — восстанавливаем из базы
                if base_spec.get("frozen_entries"):
                    candidate_spec["frozen_entries"] = True
                if "exit_corridor" in base_spec:
                    candidate_spec["exit_corridor"] = base_spec["exit_corridor"]

                if self._spec_fingerprint(candidate_spec) in rejected_fingerprints:
                    continue

                new_spec = candidate_spec
                new_rules_candidate = (
                    optimization_result.get('new_rules', base_rules) or base_rules)
                changes_summary = optimization_result.get(
                    'changes_summary', changes_summary)
                break

            # Fallback: программная мутация
            if new_spec is None:
                emit('AI не дала новый spec — применяю программную мутацию.')
                for _ in range(PROGRAM_MUTATION_ATTEMPTS):
                    candidate = ensure_engine_defaults(
                        self.mutations.mutate_numeric(base_spec))
                    candidate = self._enforce_frozen_entries(
                        candidate, base_spec)
                    candidate = self._enforce_exit_corridor(
                        candidate, base_spec)
                    err = validate_strategy_spec(candidate)
                    if err is not None:
                        emit(f'Fallback: невалидный кандидат: {err}')
                        continue
                    if self._spec_fingerprint(candidate) in rejected_fingerprints:
                        emit('Fallback: дубликат уже отклонённого кандидата.')
                        continue
                    new_spec = candidate
                    new_rules_candidate = base_rules
                    changes_summary = 'Программная мутация числовых параметров.'
                    break

            if new_spec is None:
                emit('Не удалось сформировать новый spec. Оптимизация остановлена.')
                break

            current_spec = new_spec
            current_rules = new_rules_candidate
            last_changes = changes_summary
            emit(f'Итерация {iteration}: новый spec принят.')

        # 5. Holdout-проверка
        holdout_summary = {
            "enabled": False,
            "reason": "not_run",
            "status": "not_run",
            "passed": False,
            "weak_pass": False,
            "warnings": [],
            "recommendations": [],
        }

        if best is not None and holdout_meta.get("enabled") and len(holdout_df) > 0:
            emit('Финальная HOLDOUT-проверка лучшего кандидата...')

            holdout_evaluation = self.holdout.evaluate_spec_on_holdout(
                spec=best["spec"],
                rules=best["rules"],
                holdout_df=holdout_df,
                symbol=symbol,
                initial_capital=initial_capital,
                commission_pct=commission_pct,
            )

            if "error" not in holdout_evaluation:
                holdout_stats = holdout_evaluation.get("stats", {}) or {}
                holdout_trades = holdout_evaluation.get("trades", []) or []

                holdout_quality = self.holdout.assess_holdout_quality(holdout_stats)

                holdout_summary = self.holdout.build_holdout_summary(
                    holdout_stats=holdout_stats,
                    holdout_trades=holdout_trades,
                    holdout_quality=holdout_quality,
                    holdout_meta=holdout_meta,
                    holdout_period=holdout_period,
                    profit_concentration_func=self.scoring.profit_concentration,
                )

                emit(
                    f'HOLDOUT: сделок={holdout_summary.get("total_trades", 0)}, '
                    f'P&L={holdout_summary.get("return_pct", 0):+.2f}%, '
                    f'Status={holdout_summary.get("status", "failed")}.'
                )

                best["holdout_summary"] = holdout_summary
            else:
                emit(f'HOLDOUT: ошибка — {holdout_evaluation.get("error")}')
                best["holdout_summary"] = {
                    "enabled": True,
                    "reason": "evaluation_error",
                    "error": holdout_evaluation.get("error"),
                    "status": "failed",
                }
        elif best is not None:
            best["holdout_summary"] = holdout_summary

        # 6. Сохранение лучшей стратегии
        holdout_status = holdout_summary.get("status", "not_run")
        holdout_blocks_save = (
            holdout_meta.get("enabled")
            and holdout_status == "failed"
        )
        holdout_is_weak_but_allowed = (
            holdout_meta.get("enabled")
            and holdout_status == "weak_pass"
            and allow_weak_holdout_save
        )
        holdout_is_weak_but_blocked = (
            holdout_meta.get("enabled")
            and holdout_status == "weak_pass"
            and not allow_weak_holdout_save
        )

        if save_best_to_memory and best is not None and not holdout_blocks_save and not holdout_is_weak_but_blocked:
            cm.update_state({'strategy_spec': best['spec']})
            cm.update_rules(best['rules'])
            cm.save()
            bt = best['aggregated_test_stats']
            emit(
                f'Сохранена лучшая стратегия: итерация {best["iteration"]}, '
                f'TEST P&L={bt.get("total_pnl_pct", 0):+.2f}%.'
            )
        elif best is None:
            emit('Ни одна стратегия не прошла требования.')
        elif holdout_is_weak_but_blocked:
            emit('Лучшая стратегия не сохранена: holdout имеет статус weak_pass.')
        elif holdout_blocks_save:
            emit('Лучшая стратегия не сохранена: holdout не пройден.')

        report.section("ФИНАЛ / ЛУЧШИЙ КАНДИДАТ", best)
        report.section("ФИНАЛ / HOLDOUT", holdout_summary)
        report.section("ФИНАЛ / ВСЯ ИСТОРИЯ ИТЕРАЦИЙ", history)
        emit(f'Отчёт сохранён: {report.path}')

        return {
            'history': history,
            'best': best,
            'splits': splits,
            'walk_forward_folds': walk_forward_folds,
            'train_ratio': train_ratio,
            'csv_dir': str(csv_dir),
            'holdout': holdout_summary,
            'report_path': str(report.path),
        }