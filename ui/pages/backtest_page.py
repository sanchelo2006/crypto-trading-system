"""
Страница бэктеста и оптимизации стратегии.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path

from nicegui import ui

import config
from ui.app_context import app_context


class BacktestPage:
    """
    UI-страница для:
    - одиночного бэктеста текущей strategy_spec;
    - итеративной оптимизации стратегии.

    Все зависимости передаются явно через конструктор.
    Это снижает связанность с nicegui_app.py и упрощает дальнейший рефакторинг.
    """

    def __init__(
        self,
        *,
        memory,
        backtest_engine,
        backtest_optimizer,
    ):
        self.memory = memory
        self.backtest_engine = backtest_engine
        self.backtest_optimizer = backtest_optimizer

        self.container = ui.column().classes("w-full")
        self._opt_status = {}
        self._opt_running = False
        #раньше прогресс выводился на странице, теперь сохранем в файл
        #self._progress_lines = []
        #self._last_rendered_count = 0
        self._ui_update_lock = threading.Lock()

        # Результаты, ожидающие безопасного рендера в UI-контексте.
        self._pending_backtest_result = None
        self._pending_backtest_status = None
        self._pending_optimization_result = None
        self._pending_optimization_status = None
        self._pending_status_text = None
        self._pending_reenable_buttons = False

        with self.container:
            self.build_ui()

        self.container.visible = False

        # Заполняем список архетипов для селекта сид-архетипа.
        try:
            from optimization.archetypes import ARCHETYPES
            options = {
                "memory": "Из памяти монеты (без скрининга)",
                "auto": "Авто по скринингу",
            }
            for arch in ARCHETYPES:
                options[arch["name"]] = arch["name"]
            self.seed_archetype_select.options = options
            self.seed_archetype_select.value = "memory"
            self.seed_archetype_select.update()
        except Exception:
            pass

    def build_ui(self):
        ui.label("📈 Бэктест и оптимизация стратегии").classes("text-h6 q-mb-md")

        ui.label(
            "Одиночный бэктест проверяет текущую strategy_spec. "
            "Оптимизация сначала один раз загружает исторические свечи, затем "
            "все итерации работают с ними локально без повторных сетевых загрузок."
        ).classes("text-caption text-grey q-mb-sm")

        with ui.row().classes("gap-4 items-end w-full"):
            self.symbol_select = ui.select(
                label="Монета",
                options=list(self.memory.get_all_symbols()),
                value=self.memory.get_all_symbols()[0] if self.memory.get_all_symbols() else None,
            ).classes("w-40")

            self.start_date_input = ui.input(
                "Дата начала",
                value=(datetime.now(timezone.utc) - timedelta(days=180)).strftime("%Y-%m-%d"),
                placeholder="YYYY-MM-DD",
            ).classes("w-40")

            self.end_date_input = ui.input(
                "Дата конца",
                value=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                placeholder="YYYY-MM-DD",
            ).classes("w-40")

            ui.label("Формат дат: YYYY-MM-DD").classes("text-caption text-grey")

        with ui.row().classes("gap-4 items-end q-mt-sm w-full"):
            self.capital_input = ui.number(
                "Начальный капитал ($)",
                value=1000.0,
                format="%.2f",
                min=1,
            ).classes("w-40")

            self.commission_input = ui.number(
                "Комиссия на сторону (%)",
                value=0.05,
                format="%.3f",
                min=0,
                step=0.01,
            ).classes("w-40")

            self.iterations_input = ui.number(
                "Итераций оптимизации",
                value=10,
                min=1,
                max=30,
                format="%d",
            ).classes("w-40")

            ui.label(
                "Win rate НЕ задаётся: критерий — положительный expectancy и "
                "Profit Factor на TRAIN и TEST."
            ).classes("text-caption text-grey")

        with ui.row().classes("gap-4 items-end q-mt-sm w-full"):
            self.train_ratio_input = ui.number(
                "Начальная доля TRAIN (%)",
                value=70,
                min=50,
                max=90,
                format="%d",
            ).classes("w-48")

            self.walk_forward_folds_input = ui.number(
                "TEST-окон walk-forward",
                value=6,
                min=3,
                max=8,
                format="%d",
            ).classes("w-48")

            self.phase0_screen_checkbox = ui.checkbox(
                "Фаза 0: скрининг архетипов (долго, ~3-5 мин)",
                value=False,
            ).classes("q-mt-sm")
            self.freeze_entries_checkbox = ui.checkbox(
                "Заморозить входы (оптимизировать только риск-параметры)",
                value=False,
            ).classes("q-mt-sm")
            ui.label(
                "При включении условия входа (entry/entry_any) не изменяются. "
                "Оптимизатор тюнит только стопы, тейки, трейлинг и фильтры."
            ).classes("text-caption text-grey")

            ui.label(
                "1 = один обычный TRAIN/TEST split. "
                "3 = три последовательных TEST-окна с расширяющимся TRAIN."
            ).classes("text-caption text-grey")

        # --- Сид-архетип ---
        with ui.row().classes("gap-4 items-end q-mt-sm w-full"):
            self.seed_archetype_select = ui.select(
                label="Сид-архетип (опционально)",
                options={"auto": "Авто по скринингу"},
                value="auto",
            ).classes("w-56")

            ui.label(
                "Сид-архетип — явный старт для оптимизации вместо авто-выбора по скринингу."
            ).classes("text-caption text-grey")

        with ui.expansion("📋 Доступные серии и подсказки", icon="list").classes("w-full q-mt-md"):
            ui.label("Доступные серии (15m и 1h):").classes("text-bold")

            ui.table(
                columns=[
                    {"name": "series", "label": "Серия", "field": "series", "align": "left"},
                    {"name": "description", "label": "Описание", "field": "description", "align": "left"},
                ],
                rows=[
                    {"series": "open, high, low, close, volume", "description": "Основные OHLCV"},
                    {"series": "sma_20, sma_50", "description": "Простые скользящие средние"},
                    {"series": "rsi_14", "description": "Индекс относительной силы"},
                    {"series": "atr_14", "description": "Средний истинный диапазон"},
                    {"series": "bb_upper, bb_middle, bb_lower", "description": "Полосы Боллинджера"},
                    {"series": "macd, macd_signal, macd_hist", "description": "MACD"},
                    {"series": "adx_14", "description": "Индекс направленного движения"},
                    {"series": "vwap", "description": "VWAP"},
                    {"series": "pivot, resistance_1..3, support_1..3", "description": "Пивот и уровни"},
                    {"series": "avg_volume_20, avg_volume_96", "description": "Средние объёмы"},
                    {"series": "sma_20_1h, sma_50_1h, rsi_14_1h, atr_14_1h", "description": "1h-индикаторы тренда"},
                    {"series": "bb_upper_1h, bb_middle_1h, bb_lower_1h", "description": "1h-полосы Боллинджера"},
                    {"series": "macd_1h, macd_signal_1h, macd_hist_1h", "description": "1h MACD"},
                    {"series": "adx_14_1h, vwap_1h", "description": "1h ADX и VWAP"},
                    {"series": "pivot_1h, resistance_1_1h..3, support_1_1h..3", "description": "1h уровни"},
                    {"series": "avg_volume_20_1h, avg_volume_96_1h", "description": "1h средние объёмы"},
                ],
                row_key="series",
            ).classes("w-full")

            ui.label(
                "💡 1h-серии нужны как фильтр старшего таймфрейма: "
                "они помогают отсеивать шум и повышают устойчивость входов."
            ).classes("text-caption text-grey q-mt-sm")

        with ui.row().classes("gap-4 items-end q-mt-sm w-full"):
            default_csv = str(config.DATA_DIR / "backtests")
            self.csv_path_input = ui.input(
                "Папка для CSV-отчётов",
                value=default_csv,
            ).classes("flex-1").style("min-width: 300px;")

        default_allow_weak_holdout = False
        if getattr(app_context, "app_settings", None) is not None:
            default_allow_weak_holdout = bool(
                app_context.app_settings.get("allow_weak_holdout_save", False)
            )

        with ui.row().classes("gap-4 items-center q-mt-sm w-full"):
            self.allow_weak_holdout_save_checkbox = ui.checkbox(
                "Разрешать сохранение стратегии при weak_pass holdout",
                value=default_allow_weak_holdout,
            )

        with ui.row().classes("gap-4 q-mt-md"):
            self.run_btn = ui.button(
                "🚀 Одиночный бэктест",
                on_click=self.run_backtest,
            ).classes("bg-secondary text-white")

            self.optimize_btn = ui.button(
                "♻️ Запустить оптимизацию",
                on_click=self.run_optimization,
            ).classes("bg-positive text-white")

        self.status_label = ui.label("").classes("text-caption q-mt-md")
        
        self.result_container = ui.column().classes("w-full q-mt-lg")

        

        # Таймер безопасного рендера результатов в UI-контексте.
        ui.timer(0.5, self._poll_results)

    def _poll_results(self):
        """
        Выполняется в UI-контексте.
        Рендерит результаты, которые воркеры положили в pending-поля.
        """
        with self._ui_update_lock:
            status_text = self._pending_status_text
            self._pending_status_text = None

            backtest_result = self._pending_backtest_result
            backtest_status = self._pending_backtest_status
            self._pending_backtest_result = None
            self._pending_backtest_status = None

            optimization_result = self._pending_optimization_result
            optimization_status = self._pending_optimization_status
            self._pending_optimization_result = None
            self._pending_optimization_status = None

            reenable = self._pending_reenable_buttons
            self._pending_reenable_buttons = False

        if status_text is not None:
            self.status_label.set_text(status_text)

        if backtest_result is not None:
            try:
                self._render_results(backtest_result)
                self.result_container.update()
            except Exception:
                logging.getLogger("nicegui").exception("Render backtest error")

        if backtest_status is not None:
            self.status_label.set_text(backtest_status)

        if optimization_result is not None:
            try:
                self._render_optimization(optimization_result)
                self.result_container.update()
            except Exception:
                logging.getLogger("nicegui").exception("Render optimization error")

        self.refresh()

        if optimization_status is not None:
            self.status_label.set_text(optimization_status)

        if reenable:
            self.run_btn.enabled = True
            self.optimize_btn.enabled = True

    def _get_common_params(self):
        symbol = self.symbol_select.value
        if not symbol:
            ui.notify("Выберите монету.", type="warning")
            return None

        try:
            start_date = datetime.strptime(
                (self.start_date_input.value or "").strip(), "%Y-%m-%d",
            ).replace(
                tzinfo=timezone.utc,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
            end_date = datetime.strptime(
                (self.end_date_input.value or "").strip(), "%Y-%m-%d",
            ).replace(
                tzinfo=timezone.utc,
                hour=23,
                minute=59,
                second=59,
                microsecond=999999,
            )
        except ValueError:
            ui.notify(
                "Неверный формат даты. Используйте формат YYYY-MM-DD.",
                type="warning",
            )
            return None

        if start_date >= end_date:
            ui.notify(
                "Дата начала должна быть раньше даты конца.",
                type="warning",
            )
            return None

        # Лёгкая проверка по launchTime через метаданные биржи.
        adjusted_start_date, adjusted_end_date, listing_warning = (
            self._adjust_period_to_listing_if_needed(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                category="linear",
            )
        )

        if adjusted_start_date >= adjusted_end_date:
            ui.notify(
                "После корректировки по дате начала торгов период стал недопустимым. "
                "Выберите более позднюю дату конца.",
                type="warning",
            )
            return None

        if adjusted_start_date != start_date:
            self.start_date_input.value = adjusted_start_date.strftime("%Y-%m-%d")
            self.start_date_input.update()

        if listing_warning:
            ui.notify(listing_warning, type="warning")

        start_date = adjusted_start_date
        end_date = adjusted_end_date

        min_period = timedelta(days=15)
        if end_date - start_date < min_period:
            ui.notify(
                "Период должен быть не менее 15 дней: индикаторам требуется история.",
                type="warning",
            )
            return None

        return {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": float(self.capital_input.value or 1000.0),
            "commission_pct": float(self.commission_input.value or 0.05),
        }

    def _get_current_strategy_payload(self, symbol: str):
        """
        Берёт актуальную стратегию из памяти монеты на момент запуска.
        """
        from strategy_spec import validate_strategy_spec

        cm = self.memory.get(symbol)
        state = cm.get_state()

        strategy_spec = state.get("strategy_spec")
        rules_text = cm.get_rules() or ""
        spec_source = "memory_best"

        if not strategy_spec:
            spec_source = "missing"
            return None, rules_text, spec_source

        err = validate_strategy_spec(strategy_spec)
        if err is not None:
            logging.getLogger("nicegui").warning(
                "[%s] strategy_spec в памяти невалиден: %s",
                symbol,
                err,
            )
            spec_source = "invalid"
            return None, rules_text, spec_source

        return strategy_spec, rules_text, spec_source

    def _adjust_period_to_listing_if_needed(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        category: str = "linear",
    ):
        """
        Если пользователь выбрал дату начала раньше листинга монеты на бирже,
        сдвигает start_date до даты начала торгов и возвращает предупреждение.

        Возвращает:
            adjusted_start_date, adjusted_end_date, warning_text_or_none
        """
        fetcher = getattr(self.backtest_engine, "fetcher", None)
        if fetcher is None:
            return start_date, end_date, None

        try:
            listing_start = fetcher.get_symbol_launch_time(symbol, category)
        except Exception:
            logging.getLogger("nicegui").exception(
                "Не удалось проверить дату начала торгов для %s", symbol
            )
            return start_date, end_date, None

        if listing_start is None:
            return start_date, end_date, None

        listing_day_start = listing_start.replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        if start_date < listing_day_start:
            warning = (
                f"{symbol} начала торговаться на бирже только с "
                f"{listing_day_start.strftime('%Y-%m-%d')}. "
                f"Начало периода автоматически скорректировано."
            )
            return listing_day_start, end_date, warning

        return start_date, end_date, None

    def _prepare_new_run(self, status_text: str, *, optimization: bool = False):
        """Подготавливает страницу к новому запуску бэктеста/оптимизации.

        - очищает старые результаты;
        - сбрасывает pending-поля;
        - при оптимизации очищает прогресс;
        - блокирует кнопки;
        - выставляет стартовый статус.
        """
        with self._ui_update_lock:
            self._pending_backtest_result = None
            self._pending_backtest_status = None
            self._pending_optimization_result = None
            self._pending_optimization_status = None
            self._pending_status_text = None
            self._pending_reenable_buttons = False

            if optimization:
                
                self._opt_running = True
            else:
                self._opt_running = False

        self.result_container.clear()

        

        self.status_label.set_text(status_text)
        self.run_btn.enabled = False
        self.optimize_btn.enabled = False

    def run_backtest(self):
        params = self._get_common_params()
        if not params:
            return

        symbol = params["symbol"]
        cm = self.memory.get(symbol)

        cm.reload()

        strategy_spec = cm.get_state().get("strategy_spec")
        rules_text = cm.get_rules()

        if not strategy_spec and not rules_text:
            ui.notify(
                f"Нет правил для {symbol}. Сначала инициализируйте монету.",
                type="warning",
            )
            return

        if not strategy_spec:
            ui.notify(
                "У монеты нет strategy_spec — будет использована стратегия по умолчанию.",
                type="info",
            )

        # Папку отчётов читаем в UI-потоке, воркер работает в отдельном треде
        # и не должен дёргать элементы UI.
        report_dir = (
            self.csv_path_input.value.strip()
            if self.csv_path_input.value
            else str(config.DATA_DIR / "backtests")
        )

        self._prepare_new_run("Загрузка данных и расчёт индикаторов...", optimization=False)

        def worker():
            try:
                result = self.backtest_engine.run(
                    symbol=symbol,
                    category="linear",
                    strategy_spec=strategy_spec,
                    start_date=params["start_date"],
                    end_date=params["end_date"],
                    initial_capital=params["initial_capital"],
                    commission_pct=params["commission_pct"],
                    rules_text=rules_text,
                )

                if "error" in result:
                    with self._ui_update_lock:
                        self._pending_status_text = f'Ошибка: {result["error"]}'
                    return

                # Сохраняем XLSX-отчёт. Ошибка экспорта не должна ломать UI.
                from utils.backtest_exporter import export_backtest_to_xlsx

                saved_path = export_backtest_to_xlsx(
                    result=result,
                    symbol=symbol,
                    output_dir=Path(report_dir),
                    start_date=params["start_date"],
                    end_date=params["end_date"],
                )

                src = result.get("spec_source", "ai")
                status = (
                    "Бэктест завершён "
                    f'(стратегия: {"из инициализации" if src == "ai" else "по умолчанию"})'
                )
                if saved_path is not None:
                    status += f". Отчёт: {saved_path}"
                else:
                    status += ". ⚠️ Не удалось сохранить XLSX-отчёт."

                with self._ui_update_lock:
                    self._pending_backtest_result = result
                    self._pending_backtest_status = status

            except Exception as e:
                logging.getLogger("nicegui").exception("Backtest error")
                with self._ui_update_lock:
                    self._pending_status_text = f"Ошибка бэктеста: {e}"
            finally:
                with self._ui_update_lock:
                    self._pending_reenable_buttons = True

        asyncio.create_task(asyncio.to_thread(worker))

    def run_optimization(self):
        params = self._get_common_params()
        if not params:
            return

        symbol = params["symbol"]
        cm = self.memory.get(symbol)
        strategy_spec = cm.get_state().get("strategy_spec")
        rules_text = cm.get_rules()

        if not strategy_spec and not rules_text:
            ui.notify(
                f"Для {symbol} нет strategy_spec и торговых правил. "
                "Сначала инициализируйте монету.",
                type="warning",
            )
            return

        max_iterations = int(self.iterations_input.value or 10)
        train_ratio = float(self.train_ratio_input.value or 70) / 100.0
        walk_forward_folds = int(self.walk_forward_folds_input.value or 1)

        csv_path = (
            self.csv_path_input.value.strip()
            if self.csv_path_input.value
            else str(config.DATA_DIR / "backtests")
        )

        robust_settings = {
            "robust_min_profitable_windows_ratio": float(
                app_context.app_settings.get("robust_min_profitable_windows_ratio", 0.67)
            ) if getattr(app_context, "app_settings", None) is not None else 0.67,

            "robust_min_positive_expectancy_windows_ratio": float(
                app_context.app_settings.get("robust_min_positive_expectancy_windows_ratio", 0.67)
            ) if getattr(app_context, "app_settings", None) is not None else 0.67,

            "robust_min_pf_pass_windows_ratio": float(
                app_context.app_settings.get("robust_min_pf_pass_windows_ratio", 0.67)
            ) if getattr(app_context, "app_settings", None) is not None else 0.67,

            "robust_min_median_window_return_pct": float(
                app_context.app_settings.get("robust_min_median_window_return_pct", 0.0)
            ) if getattr(app_context, "app_settings", None) is not None else 0.0,

            "robust_max_worst_window_return_pct": float(
                app_context.app_settings.get("robust_max_worst_window_return_pct", -5.0)
            ) if getattr(app_context, "app_settings", None) is not None else -5.0,

            "robust_min_stability_score": float(
                app_context.app_settings.get("robust_min_stability_score", 0.5)
            ) if getattr(app_context, "app_settings", None) is not None else 0.5,

            "block_fragile_best_candidate": bool(
                app_context.app_settings.get("block_fragile_best_candidate", True)
            ) if getattr(app_context, "app_settings", None) is not None else True,
        }

        self._prepare_new_run(
            "ИДЕТ ОПТИМИЗАЦИЯ",
            optimization=True,
        )

        def progress_cb(event: dict):
            # Прогресс в UI больше не выводится.
            # Полный ход оптимизации пишет OptimizationReport в .txt.
            pass

        def worker():
            try:
                # Сид-архетип
                seed_val = self.seed_archetype_select.value
                if seed_val == "memory":
                    seed_archetype = None
                    use_phase0 = False
                elif seed_val == "auto":
                    seed_archetype = None
                    use_phase0 = self.phase0_screen_checkbox.value
                else:
                    seed_archetype = seed_val
                    use_phase0 = True
                freeze_entries = self.freeze_entries_checkbox.value
                result = self.backtest_optimizer.run(
                    symbol=symbol,
                    category="linear",
                    max_iterations=max_iterations,
                    initial_capital=params["initial_capital"],
                    commission_pct=params["commission_pct"],
                    csv_path=csv_path,
                    start_date=params["start_date"],
                    end_date=params["end_date"],
                    train_ratio=train_ratio,
                    walk_forward_folds=walk_forward_folds,
                    save_best_to_memory=True,
                    phase0_screen=use_phase0,
                    freeze_entries=freeze_entries,
                    seed_archetype=seed_archetype,
                    progress_callback=progress_cb,
                    allow_weak_holdout_save=bool(
                        self.allow_weak_holdout_save_checkbox.value
                    ),
                    block_fragile_best_candidate=robust_settings["block_fragile_best_candidate"],
                    robust_min_profitable_windows_ratio=robust_settings["robust_min_profitable_windows_ratio"],
                    robust_min_positive_expectancy_windows_ratio=robust_settings["robust_min_positive_expectancy_windows_ratio"],
                    robust_min_pf_pass_windows_ratio=robust_settings["robust_min_pf_pass_windows_ratio"],
                    robust_min_median_window_return_pct=robust_settings["robust_min_median_window_return_pct"],
                    robust_max_worst_window_return_pct=robust_settings["robust_max_worst_window_return_pct"],
                    robust_min_stability_score=robust_settings["robust_min_stability_score"],
                )


                if "error" in result:
                    with self._ui_update_lock:
                        self._pending_status_text = f'Ошибка оптимизации: {result["error"]}'
                    return

                best = result.get("best")
                if best:
                    bt = best.get("aggregated_test_stats", {})
                    status = (
                        f'Оптимизация завершена. Лучшая итерация #{best.get("iteration", "?")}: '
                        f'TEST доходность={bt.get("return_pct", 0):+.2f}%, '
                        f'TEST expectancy={bt.get("expectancy_pct", 0):+.4f}%, '
                        f'TEST PF={bt.get("profit_factor", 0):.2f}. '
                        f'CSV: {result.get("csv_dir", "")}'
                    )
                else:
                    status = (
                        "Оптимизация завершена, но стратегия с эджем на TRAIN и TEST "
                        "не найдена."
                    )

                with self._ui_update_lock:
                    self._pending_optimization_result = result
                    self._pending_optimization_status = status

            except Exception as e:
                logging.getLogger("nicegui").exception("Optimization error")
                with self._ui_update_lock:
                    self._pending_status_text = f"Ошибка оптимизации: {e}"
            finally:
                self._opt_running = False
                with self._ui_update_lock:
                    self._pending_reenable_buttons = True

        asyncio.create_task(asyncio.to_thread(worker))

    def _render_results(self, result: dict):
        stats = result.get("stats", {})
        trades = result.get("trades", [])
        spec_used = result.get("strategy_spec_used", {})
        spec_source = result.get("spec_source", "ai")

        self.result_container.clear()

        with self.result_container:
            source_color = "positive" if spec_source == "ai" else "negative"
            source_text = (
                "✅ Стратегия из инициализации монеты"
                if spec_source == "ai"
                else "⚠️ Стратегия ПО УМОЛЧАНИЮ (spec монеты не найден/невалиден)"
            )
            ui.label(source_text).classes(f"text-{source_color} text-bold q-mt-md")

            with ui.expansion("🔍 Исполнявшаяся strategy_spec", icon="rule").classes("w-full"):
                ui.code(
                    _json.dumps(spec_used, ensure_ascii=False, indent=2),
                    language="json",
                ).classes("w-full")

            ui.label("📊 Статистика").classes("text-h6 q-mt-md")
            ui.table(
                columns=[
                    {"name": "metric", "label": "Метрика", "field": "metric", "align": "left"},
                    {"name": "value", "label": "Значение", "field": "value", "align": "right"},
                ],
                rows=[
                    {"metric": "Всего сделок", "value": str(stats.get("total_trades", 0))},
                    {
                        "metric": "Long / Short",
                        "value": f'{stats.get("long_trades", 0)} / {stats.get("short_trades", 0)}',
                    },
                    {"metric": "Win Rate", "value": f'{stats.get("win_rate", 0)}%'},
                    {
                        "metric": "Доходность (от капитала)",
                        "value": f'{stats.get("return_pct", 0):+.2f}%',
                    },
                    {
                        "metric": "Сумма % сделок (справочно)",
                        "value": f'{stats.get("total_pnl_pct", 0):+.2f}%',
                    },
                    {
                        "metric": "Итоговый капитал",
                        "value": f'${stats.get("final_capital", 0):,.2f}',
                    },
                    {
                        "metric": "Макс. просадка",
                        "value": f'{stats.get("max_drawdown_pct", 0)}%',
                    },
                    {
                        "metric": "Profit Factor",
                        "value": str(stats.get("profit_factor", 0)),
                    },
                    {
                        "metric": "Средняя сделка",
                        "value": f'{stats.get("avg_trade_pct", 0):+.4f}%',
                    },
                    {
                        "metric": "Expectancy",
                        "value": f'{stats.get("expectancy_pct", stats.get("avg_trade_pct", 0)):+.4f}%',
                    },
                    {
                        "metric": "Средний win",
                        "value": f'{stats.get("avg_win_pct", 0):+.4f}%',
                    },
                    {
                        "metric": "Средний loss",
                        "value": f'-{abs(stats.get("avg_loss_pct", 0)):.4f}%',
                    },
                    {
                        "metric": "Risk controls",
                        "value": "ON" if stats.get("risk_controls_enabled") else "OFF",
                    },
                    {
                        "metric": "Cooldown activations",
                        "value": str(stats.get("cooldown_activations", 0)),
                    },
                    {
                        "metric": "Skipped candles by cooldown",
                        "value": str(stats.get("skipped_candles_by_cooldown", 0)),
                    },
                ],
                row_key="metric",
            ).classes("w-full")

            if trades:
                ui.label(f"📋 Сделки ({len(trades)})").classes("text-h6 q-mt-md")

                for idx, t in enumerate(trades, 1):
                    entry_time = t.get("entry_time")
                    exit_time = t.get("exit_time")

                    entry_t = (
                        entry_time.strftime("%Y-%m-%d %H:%M")
                        if hasattr(entry_time, "strftime")
                        else str(entry_time)
                    )
                    exit_t = (
                        exit_time.strftime("%Y-%m-%d %H:%M")
                        if hasattr(exit_time, "strftime")
                        else str(exit_time)
                    )

                    header = (
                        f'#{idx} {str(t.get("direction", "?")).upper()} | '
                        f"{entry_t} → {exit_t} | "
                        f'${t.get("entry_price", 0):.4f} → ${t.get("exit_price", 0):.4f} | '
                        f'{t.get("result_pct", 0):+.2f}% ({t.get("exit_reason", "")})'
                    )

                    with ui.expansion(header, icon="trending_up").classes("w-full"):
                        entry_reasons = t.get("entry_reasons", [])
                        if entry_reasons:
                            ui.label("Условия входа").classes("text-bold")
                            for r in entry_reasons:
                                mark = "✓" if r.get("passed") else "✗"
                                color = "positive" if r.get("passed") else "negative"
                                ui.label(
                                    f'{mark} {r.get("text", "")} → {r.get("detail", "")}'
                                ).classes(f"text-{color}")

                        snapshot = t.get("entry_snapshot", {})
                        if snapshot:
                            ui.label("Снимок индикаторов на входе").classes("text-bold q-mt-md")
                            snapshot_rows = [
                                {"metric": k, "value": str(v)}
                                for k, v in snapshot.items()
                            ]
                            ui.table(
                                columns=[
                                    {"name": "metric", "label": "Параметр", "field": "metric", "align": "left"},
                                    {"name": "value", "label": "Значение", "field": "value", "align": "right"},
                                ],
                                rows=snapshot_rows,
                                row_key="metric",
                            ).classes("w-full")
            else:
                ui.label("Сделок за период не совершено.").classes("text-caption q-mt-md")

    def _render_optimization(self, result: dict):
        history = result.get("history", [])
        best = result.get("best")
        splits = result.get("splits", [])
        walk_forward_folds = result.get("walk_forward_folds", 1)
        train_ratio = result.get("train_ratio", 0.7)
        holdout = result.get("holdout_summary", result.get("holdout", {})) or {}

        def fmt_date(value):
            if hasattr(value, "strftime"):
                return value.strftime("%Y-%m-%d")
            return str(value)[:10] if value else "—"

        self.result_container.clear()

        with self.result_container:
            ui.label("♻️ Результаты оптимизации").classes("text-h6 q-mt-md")

            report_path = (
                result.get("report_path")
                or getattr(self.backtest_optimizer, "last_report_path", "")
            )
            if report_path:
                ui.label(f"📄 Подробный отчёт: {report_path}").classes(
                    "text-caption text-grey"
                )

            if walk_forward_folds <= 1:
                ui.label(
                    f"Режим проверки: один TRAIN/TEST split "
                    f"({int(train_ratio * 100)}% / {int((1 - train_ratio) * 100)}%)."
                ).classes("text-caption text-grey")
            else:
                ui.label(
                    f"Режим проверки: walk-forward, TEST-окон: {walk_forward_folds}. "
                    "TRAIN-выборка расширяется перед каждым последующим TEST-окном."
                ).classes("text-caption text-grey")

            if splits:
                with ui.expansion("🗓️ Использованные TRAIN / TEST окна", icon="date_range").classes("w-full"):
                    split_rows = []
                    for index, split in enumerate(splits, start=1):
                        split_rows.append(
                            {
                                "fold": index,
                                "train": f'{fmt_date(split.get("train_start"))} → {fmt_date(split.get("train_end"))}',
                                "test": f'{fmt_date(split.get("test_start"))} → {fmt_date(split.get("test_end"))}',
                            }
                        )

                    ui.table(
                        columns=[
                            {"name": "fold", "label": "Окно", "field": "fold", "align": "center"},
                            {"name": "train", "label": "TRAIN-период", "field": "train", "align": "left"},
                            {"name": "test", "label": "TEST-период", "field": "test", "align": "left"},
                        ],
                        rows=split_rows,
                        row_key="fold",
                    ).classes("w-full")

                ui.label(
                    "Кандидат считается пригодным при положительном expectancy "
                    "и Profit Factor > 1 на TRAIN и TEST. Win Rate сам по себе "
                    "не является критерием прибыльности."
                ).classes("text-caption text-grey q-mt-sm")

            if history:
                rows = []
                for item in history:
                    rows.append(
                        {
                            "iteration": item.get("iteration", 0),
                            "train_trades": item.get("train_trades", 0),
                            "test_trades": item.get("test_trades", 0),
                            "train_wr": f'{item.get("train_winrate", 0):.1f}%',
                            "test_wr": f'{item.get("test_winrate", 0):.1f}%',
                            "train_return": f'{item.get("train_return_pct", 0):+.2f}%',
                            "test_return": f'{item.get("test_return_pct", 0):+.2f}%',
                            "train_exp": f'{item.get("train_expectancy_pct", 0):+.4f}%',
                            "test_exp": f'{item.get("test_expectancy_pct", 0):+.4f}%',
                            "test_pf": f'{item.get("test_profit_factor", 0):.2f}',
                            "test_dd": f'{item.get("test_max_drawdown_pct", 0):.2f}%',
                            "rob_status": str(item.get("robustness_status", "—")),
                            "rob_pass": "Да" if item.get("robustness_passed", False) else "Нет",
                            "rob_fragile": "Да" if item.get("robustness_fragile", False) else "Нет",
                            "rob_score": f'{item.get("robustness_stability_score", 0):.3f}',
                            "score": f'{item.get("selection_score", 0):.4f}',
                            "status": "✅ Принят" if item.get("acceptable", False) else "— Не принят",
                        }
                    )

                ui.table(
                    columns=[
                        {"name": "iteration", "label": "Итер.", "field": "iteration", "align": "center"},
                        {"name": "train_trades", "label": "TRAIN сделок", "field": "train_trades", "align": "right"},
                        {"name": "test_trades", "label": "TEST сделок", "field": "test_trades", "align": "right"},
                        {"name": "train_wr", "label": "TRAIN WR", "field": "train_wr", "align": "right"},
                        {"name": "test_wr", "label": "TEST WR", "field": "test_wr", "align": "right"},
                        {"name": "train_return", "label": "TRAIN дох.", "field": "train_return", "align": "right"},
                        {"name": "test_return", "label": "TEST дох.", "field": "test_return", "align": "right"},
                        {"name": "train_exp", "label": "TRAIN Exp.", "field": "train_exp", "align": "right"},
                        {"name": "test_exp", "label": "TEST Exp.", "field": "test_exp", "align": "right"},
                        {"name": "test_pf", "label": "TEST PF", "field": "test_pf", "align": "right"},
                        {"name": "test_dd", "label": "TEST DD", "field": "test_dd", "align": "right"},
                        {"name": "rob_status", "label": "Robust status", "field": "rob_status", "align": "center"},
                        {"name": "rob_pass", "label": "Robust pass", "field": "rob_pass", "align": "center"},
                        {"name": "rob_fragile", "label": "Fragile", "field": "rob_fragile", "align": "center"},
                        {"name": "rob_score", "label": "Robust score", "field": "rob_score", "align": "right"},
                        {"name": "score", "label": "Score", "field": "score", "align": "right"},
                        {"name": "status", "label": "Статус", "field": "status", "align": "center"},
                    ],
                    rows=rows,
                    row_key="iteration",
                ).classes("w-full")
            else:
                ui.label("Нет завершённых итераций оптимизации.").classes("text-warning q-mt-md")

            if not best:
                ui.label(
                    "⚠️ Приемлемый вариант стратегии не найден. "
                    "Это означает не ошибку программы, а то, что ни одна проверенная "
                    "версия не прошла критерии по TEST-данным."
                ).classes("text-warning text-bold q-mt-md")
                return

            train_stats = best.get("aggregated_train_stats", {})
            test_stats = best.get("aggregated_test_stats", {})

            ui.label(
                f'🏆 Лучшая итерация: #{best.get("iteration", "?")} | '
                f'TEST WR={test_stats.get("win_rate", 0):.1f}% | '
                f'TEST доходность={test_stats.get("return_pct", 0):+.2f}% | '
                f'TEST expectancy={test_stats.get("expectancy_pct", 0):+.4f}% | '
                f'TEST PF={test_stats.get("profit_factor", 0):.2f}'
            ).classes("text-positive text-bold text-subtitle1 q-mt-md")

            robustness_assessment = best.get("window_robustness_assessment", {}) or {}
            if robustness_assessment:
                ui.label("🧱 Robust-оценка лучшего кандидата").classes("text-h6 q-mt-md")
                ui.table(
                    columns=[
                        {"name": "metric", "label": "Метрика", "field": "metric", "align": "left"},
                        {"name": "value", "label": "Значение", "field": "value", "align": "right"},
                    ],
                    rows=[
                        {"metric": "Статус", "value": str(robustness_assessment.get("status", "unstable"))},
                        {"metric": "Доля прибыльных окон", "value": f'{robustness_assessment.get("profitable_windows_ratio", 0):.0%}'},
                        {"metric": "Доля окон с положительным expectancy", "value": f'{robustness_assessment.get("positive_expectancy_windows_ratio", 0):.0%}'},
                        {"metric": "Доля окон с PF > 1", "value": f'{robustness_assessment.get("pf_pass_windows_ratio", 0):.0%}'},
                        {"metric": "Медианная доходность", "value": f'{robustness_assessment.get("median_window_return_pct", 0):+.2f}%'},
                        {"metric": "Худшее окно", "value": f'{robustness_assessment.get("worst_window_return_pct", 0):+.2f}%'},
                        {"metric": "Stability score", "value": f'{robustness_assessment.get("stability_score", robustness_assessment.get("window_stability_score", 0)):.3f}'},
                    ],
                    row_key="metric",
                ).classes("w-full")

                warnings = robustness_assessment.get("warnings", [])
                reasons = robustness_assessment.get("reasons", [])
                if warnings:
                    ui.label("Предупреждения robust-оценки").classes("text-bold q-mt-sm")
                    for warning in warnings:
                        ui.label(f"• {warning}").classes("text-warning")

                if reasons:
                    ui.label("Причины robust-оценки").classes("text-bold q-mt-sm")
                    for reason in reasons:
                        ui.label(f"• {reason}").classes("text-caption")

            last_history_item = history[-1] if history else {}
            if last_history_item:
                ui.label("🧱 Robust-оценка последней итерации").classes("text-h6 q-mt-md")
                ui.table(
                    columns=[
                        {"name": "metric", "label": "Метрика", "field": "metric", "align": "left"},
                        {"name": "value", "label": "Значение", "field": "value", "align": "right"},
                    ],
                    rows=[
                        {"metric": "Статус", "value": str(last_history_item.get("robustness_status", "unstable"))},
                        {"metric": "Passed", "value": "Да" if last_history_item.get("robustness_passed", False) else "Нет"},
                        {"metric": "Fragile", "value": "Да" if last_history_item.get("robustness_fragile", False) else "Нет"},
                        {"metric": "Stability score", "value": f'{last_history_item.get("robustness_stability_score", 0):.3f}'},
                    ],
                    row_key="metric",
                ).classes("w-full")

            gap = train_stats.get("win_rate", 0) - test_stats.get("win_rate", 0)
            if gap > 15:
                ui.label(
                    f"⚠️ TRAIN−TEST разрыв по Win Rate: {gap:.1f}%. "
                    "Есть риск переобучения."
                ).classes("text-warning")

            if test_stats.get("expectancy_pct", 0) <= 0:
                ui.label(
                    "⚠️ TEST expectancy не положительный. "
                    "Такую стратегию нельзя считать готовой к реальной торговле."
                ).classes("text-warning")

            ui.label("📊 Метрики лучшей стратегии").classes("text-h6 q-mt-md")
            ui.table(
                columns=[
                    {"name": "metric", "label": "Метрика", "field": "metric", "align": "left"},
                    {"name": "train", "label": "TRAIN", "field": "train", "align": "right"},
                    {"name": "test", "label": "TEST", "field": "test", "align": "right"},
                ],
                rows=[
                    {
                        "metric": "Количество сделок",
                        "train": str(train_stats.get("total_trades", 0)),
                        "test": str(test_stats.get("total_trades", 0)),
                    },
                    {
                        "metric": "Win Rate",
                        "train": f'{train_stats.get("win_rate", 0):.1f}%',
                        "test": f'{test_stats.get("win_rate", 0):.1f}%',
                    },
                    {
                        "metric": "Доходность",
                        "train": f'{train_stats.get("return_pct", 0):+.2f}%',
                        "test": f'{test_stats.get("return_pct", 0):+.2f}%',
                    },
                    {
                        "metric": "Expectancy / средняя сделка",
                        "train": f'{train_stats.get("expectancy_pct", 0):+.4f}%',
                        "test": f'{test_stats.get("expectancy_pct", 0):+.4f}%',
                    },
                    {
                        "metric": "Profit Factor",
                        "train": f'{train_stats.get("profit_factor", 0):.2f}',
                        "test": f'{test_stats.get("profit_factor", 0):.2f}',
                    },
                    {
                        "metric": "Максимальная просадка",
                        "train": f'{train_stats.get("max_drawdown_pct", 0):.2f}%',
                        "test": f'{test_stats.get("max_drawdown_pct", 0):.2f}%',
                    },
                    {
                        "metric": "Средняя прибыльная сделка",
                        "train": f'{train_stats.get("avg_win_pct", 0):+.4f}%',
                        "test": f'{test_stats.get("avg_win_pct", 0):+.4f}%',
                    },
                    {
                        "metric": "Средняя убыточная сделка",
                        "train": f'-{abs(train_stats.get("avg_loss_pct", 0)):.4f}%',
                        "test": f'-{abs(test_stats.get("avg_loss_pct", 0)):.4f}%',
                    },
                ],
                row_key="metric",
            ).classes("w-full")

            if holdout.get("enabled"):
                ui.label("🧪 Финальная HOLDOUT-проверка").classes("text-h6 q-mt-md")
                ui.table(
                    columns=[
                        {"name": "metric", "label": "Метрика", "field": "metric", "align": "left"},
                        {"name": "value", "label": "Значение", "field": "value", "align": "right"},
                    ],
                    rows=[
                        {"metric": "Статус", "value": str(holdout.get("status", "not_run"))},
                        {"metric": "Сделок", "value": str(holdout.get("total_trades", 0))},
                        {"metric": "Доходность", "value": f'{holdout.get("return_pct", 0):+.2f}%'},
                        {"metric": "Expectancy", "value": f'{holdout.get("expectancy_pct", 0):+.4f}%'},
                        {"metric": "Profit Factor", "value": f'{holdout.get("profit_factor", 0):.2f}'},
                        {"metric": "Макс. просадка", "value": f'{holdout.get("max_drawdown_pct", 0):.2f}%'},
                    ],
                    row_key="metric",
                ).classes("w-full")

                if holdout.get("warnings"):
                    ui.label("Предупреждения holdout").classes("text-bold q-mt-sm")
                    for warning in holdout.get("warnings", []):
                        ui.label(f"• {warning}").classes("text-warning")

                if holdout.get("recommendations"):
                    ui.label("Рекомендации holdout").classes("text-bold q-mt-sm")
                    for rec in holdout.get("recommendations", []):
                        ui.label(f"• {rec}").classes("text-caption")

            with ui.expansion("🔍 Лучшая strategy_spec", icon="rule").classes("w-full q-mt-md"):
                ui.code(
                    _json.dumps(best.get("spec", {}), ensure_ascii=False, indent=2),
                    language="json",
                ).classes("w-full")

            with ui.expansion("📐 Правила лучшей стратегии", icon="description").classes("w-full"):
                ui.markdown(best.get("rules") or "Текстовые правила отсутствуют.")

    def refresh(self):
        if not self.memory:
            return

        current_symbols = list(self.memory.get_all_symbols())

        self.symbol_select.options = current_symbols

        if self.symbol_select.value not in current_symbols:
            self.symbol_select.value = current_symbols[0] if current_symbols else None

        self.symbol_select.update()