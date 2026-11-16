"""
Backtest and strategy optimization page.
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
    UI page for:
    - a single backtest of the current strategy_spec;
    - iterative strategy optimization.

    All dependencies are passed explicitly through the constructor.
    This reduces coupling with nicegui_app.py and simplifies further refactoring.
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
        # Progress used to be shown on the page; now it is saved to a file
        #self._progress_lines = []
        #self._last_rendered_count = 0
        self._ui_update_lock = threading.Lock()

        # Results waiting for safe rendering in the UI context.
        self._pending_backtest_result = None
        self._pending_backtest_status = None
        self._pending_optimization_result = None
        self._pending_optimization_status = None
        self._pending_status_text = None
        self._pending_reenable_buttons = False

        with self.container:
            self.build_ui()

        self.container.visible = False

        # Fill the archetype list for the seed archetype select.
        try:
            from optimization.archetypes import ARCHETYPES
            options = {
                "memory": "From coin memory (no screening)",
                "auto": "Auto by screening",
            }
            for arch in ARCHETYPES:
                options[arch["name"]] = arch["name"]
            self.seed_archetype_select.options = options
            self.seed_archetype_select.value = "memory"
            self.seed_archetype_select.update()
        except Exception:
            pass

    def build_ui(self):
        ui.label("📈 Backtest and strategy optimization").classes("text-h6 q-mb-md")

        ui.label(
            "A single backtest checks the current strategy_spec. "
            "Optimization first loads historical candles once, then "
            "all iterations work with them locally without repeated network downloads."
        ).classes("text-caption text-grey q-mb-sm")

        with ui.row().classes("gap-4 items-end w-full"):
            self.symbol_select = ui.select(
                label="Coin",
                options=list(self.memory.get_all_symbols()),
                value=self.memory.get_all_symbols()[0] if self.memory.get_all_symbols() else None,
            ).classes("w-40")

            self.start_date_input = ui.input(
                "Start date",
                value=(datetime.now(timezone.utc) - timedelta(days=180)).strftime("%Y-%m-%d"),
                placeholder="YYYY-MM-DD",
            ).classes("w-40")

            self.end_date_input = ui.input(
                "End date",
                value=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                placeholder="YYYY-MM-DD",
            ).classes("w-40")

            ui.label("Date format: YYYY-MM-DD").classes("text-caption text-grey")

        with ui.row().classes("gap-4 items-end q-mt-sm w-full"):
            self.capital_input = ui.number(
                "Initial capital ($)",
                value=1000.0,
                format="%.2f",
                min=1,
            ).classes("w-40")

            self.commission_input = ui.number(
                "Commission per side (%)",
                value=0.05,
                format="%.3f",
                min=0,
                step=0.01,
            ).classes("w-40")

            self.iterations_input = ui.number(
                "Optimization iterations",
                value=10,
                min=1,
                max=30,
                format="%d",
            ).classes("w-40")

            ui.label(
                "Win rate is NOT set: the criterion is positive expectancy and "
                "Profit Factor on TRAIN and TEST."
            ).classes("text-caption text-grey")

        with ui.row().classes("gap-4 items-end q-mt-sm w-full"):
            self.train_ratio_input = ui.number(
                "Initial TRAIN share (%)",
                value=70,
                min=50,
                max=90,
                format="%d",
            ).classes("w-48")

            self.walk_forward_folds_input = ui.number(
                "Walk-forward TEST windows",
                value=6,
                min=3,
                max=8,
                format="%d",
            ).classes("w-48")

            self.phase0_screen_checkbox = ui.checkbox(
                "Phase 0: archetype screening (slow, ~3-5 min)",
                value=False,
            ).classes("q-mt-sm")
            self.freeze_entries_checkbox = ui.checkbox(
                "Freeze entries (optimize risk parameters only)",
                value=False,
            ).classes("q-mt-sm")
            ui.label(
                "When enabled, entry conditions (entry/entry_any) are not changed. "
                "The optimizer tunes only stops, takes, trailing and filters."
            ).classes("text-caption text-grey")

            ui.label(
                "1 = a single regular TRAIN/TEST split. "
                "3 = three consecutive TEST windows with an expanding TRAIN."
            ).classes("text-caption text-grey")

        # --- Seed archetype ---
        with ui.row().classes("gap-4 items-end q-mt-sm w-full"):
            self.seed_archetype_select = ui.select(
                label="Seed archetype (optional)",
                options={"auto": "Auto by screening"},
                value="auto",
            ).classes("w-56")

            ui.label(
                "A seed archetype is an explicit starting point for optimization instead of auto-selection by screening."
            ).classes("text-caption text-grey")

        with ui.expansion("📋 Available series and hints", icon="list").classes("w-full q-mt-md"):
            ui.label("Available series (15m and 1h):").classes("text-bold")

            ui.table(
                columns=[
                    {"name": "series", "label": "Series", "field": "series", "align": "left"},
                    {"name": "description", "label": "Description", "field": "description", "align": "left"},
                ],
                rows=[
                    {"series": "open, high, low, close, volume", "description": "Basic OHLCV"},
                    {"series": "sma_20, sma_50", "description": "Simple moving averages"},
                    {"series": "rsi_14", "description": "Relative Strength Index"},
                    {"series": "atr_14", "description": "Average True Range"},
                    {"series": "bb_upper, bb_middle, bb_lower", "description": "Bollinger Bands"},
                    {"series": "macd, macd_signal, macd_hist", "description": "MACD"},
                    {"series": "adx_14", "description": "Average Directional Index"},
                    {"series": "vwap", "description": "VWAP"},
                    {"series": "pivot, resistance_1..3, support_1..3", "description": "Pivot and levels"},
                    {"series": "avg_volume_20, avg_volume_96", "description": "Average volumes"},
                    {"series": "sma_20_1h, sma_50_1h, rsi_14_1h, atr_14_1h", "description": "1h trend indicators"},
                    {"series": "bb_upper_1h, bb_middle_1h, bb_lower_1h", "description": "1h Bollinger Bands"},
                    {"series": "macd_1h, macd_signal_1h, macd_hist_1h", "description": "1h MACD"},
                    {"series": "adx_14_1h, vwap_1h", "description": "1h ADX and VWAP"},
                    {"series": "pivot_1h, resistance_1_1h..3, support_1_1h..3", "description": "1h levels"},
                    {"series": "avg_volume_20_1h, avg_volume_96_1h", "description": "1h average volumes"},
                ],
                row_key="series",
            ).classes("w-full")

            ui.label(
                "💡 The 1h series act as a higher-timeframe filter: "
                "they help filter out noise and improve entry robustness."
            ).classes("text-caption text-grey q-mt-sm")

        with ui.row().classes("gap-4 items-end q-mt-sm w-full"):
            default_csv = str(config.DATA_DIR / "backtests")
            self.csv_path_input = ui.input(
                "Folder for CSV reports",
                value=default_csv,
            ).classes("flex-1").style("min-width: 300px;")

        default_allow_weak_holdout = False
        if getattr(app_context, "app_settings", None) is not None:
            default_allow_weak_holdout = bool(
                app_context.app_settings.get("allow_weak_holdout_save", False)
            )

        with ui.row().classes("gap-4 items-center q-mt-sm w-full"):
            self.allow_weak_holdout_save_checkbox = ui.checkbox(
                "Allow saving the strategy on weak_pass holdout",
                value=default_allow_weak_holdout,
            )

        with ui.row().classes("gap-4 q-mt-md"):
            self.run_btn = ui.button(
                "🚀 Single backtest",
                on_click=self.run_backtest,
            ).classes("bg-secondary text-white")

            self.optimize_btn = ui.button(
                "♻️ Run optimization",
                on_click=self.run_optimization,
            ).classes("bg-positive text-white")

        self.status_label = ui.label("").classes("text-caption q-mt-md")

        self.result_container = ui.column().classes("w-full q-mt-lg")

        # Timer for safe rendering of results in the UI context.
        ui.timer(0.5, self._poll_results)

    def _poll_results(self):
        """
        Runs in the UI context.
        Renders the results that the workers put into the pending fields.
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
            ui.notify("Select a coin.", type="warning")
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
                "Invalid date format. Use the YYYY-MM-DD format.",
                type="warning",
            )
            return None

        if start_date >= end_date:
            ui.notify(
                "The start date must be earlier than the end date.",
                type="warning",
            )
            return None

        # Lightweight check against launchTime via exchange metadata.
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
                "After adjusting to the trading start date the period became invalid. "
                "Choose a later end date.",
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
                "The period must be at least 15 days: the indicators need history.",
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
        Takes the current strategy from the coin's memory at the moment of the run.
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
                "[%s] strategy_spec in memory is invalid: %s",
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
        If the user picked a start date earlier than the coin's listing on the exchange,
        shifts start_date to the trading start date and returns a warning.

        Returns:
            adjusted_start_date, adjusted_end_date, warning_text_or_none
        """
        fetcher = getattr(self.backtest_engine, "fetcher", None)
        if fetcher is None:
            return start_date, end_date, None

        try:
            listing_start = fetcher.get_symbol_launch_time(symbol, category)
        except Exception:
            logging.getLogger("nicegui").exception(
                "Failed to check the trading start date for %s", symbol
            )
            return start_date, end_date, None

        if listing_start is None:
            return start_date, end_date, None

        listing_day_start = listing_start.replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        if start_date < listing_day_start:
            warning = (
                f"{symbol} started trading on the exchange only from "
                f"{listing_day_start.strftime('%Y-%m-%d')}. "
                f"The period start was adjusted automatically."
            )
            return listing_day_start, end_date, warning

        return start_date, end_date, None

    def _prepare_new_run(self, status_text: str, *, optimization: bool = False):
        """Prepares the page for a new backtest/optimization run.

        - clears old results;
        - resets the pending fields;
        - clears the progress on optimization;
        - disables the buttons;
        - sets the starting status.
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
                f"No rules for {symbol}. Initialize the coin first.",
                type="warning",
            )
            return

        if not strategy_spec:
            ui.notify(
                "The coin has no strategy_spec — the default strategy will be used.",
                type="info",
            )

        # The reports folder is read in the UI thread; the worker runs in a separate
        # thread and must not touch UI elements.
        report_dir = (
            self.csv_path_input.value.strip()
            if self.csv_path_input.value
            else str(config.DATA_DIR / "backtests")
        )

        self._prepare_new_run("Loading data and calculating indicators...", optimization=False)

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
                        self._pending_status_text = f'Error: {result["error"]}'
                    return

                # Save the XLSX report. An export error must not break the UI.
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
                    "Backtest completed "
                    f'(strategy: {"from initialization" if src == "ai" else "default"})'
                )
                if saved_path is not None:
                    status += f". Report: {saved_path}"
                else:
                    status += ". ⚠️ Failed to save the XLSX report."

                with self._ui_update_lock:
                    self._pending_backtest_result = result
                    self._pending_backtest_status = status

            except Exception as e:
                logging.getLogger("nicegui").exception("Backtest error")
                with self._ui_update_lock:
                    self._pending_status_text = f"Backtest error: {e}"
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
                f"{symbol} has no strategy_spec and no trading rules. "
                "Initialize the coin first.",
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
            "OPTIMIZATION IN PROGRESS",
            optimization=True,
        )

        def progress_cb(event: dict):
            # Progress is no longer shown in the UI.
            # The full optimization flow is written by OptimizationReport into a .txt.
            pass

        def worker():
            try:
                # Seed archetype
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
                        self._pending_status_text = f'Optimization error: {result["error"]}'
                    return

                best = result.get("best")
                if best:
                    bt = best.get("aggregated_test_stats", {})
                    status = (
                        f'Optimization completed. Best iteration #{best.get("iteration", "?")}: '
                        f'TEST return={bt.get("return_pct", 0):+.2f}%, '
                        f'TEST expectancy={bt.get("expectancy_pct", 0):+.4f}%, '
                        f'TEST PF={bt.get("profit_factor", 0):.2f}. '
                        f'CSV: {result.get("csv_dir", "")}'
                    )
                else:
                    status = (
                        "Optimization completed, but no strategy with an edge on TRAIN and TEST "
                        "was found."
                    )

                with self._ui_update_lock:
                    self._pending_optimization_result = result
                    self._pending_optimization_status = status

            except Exception as e:
                logging.getLogger("nicegui").exception("Optimization error")
                with self._ui_update_lock:
                    self._pending_status_text = f"Optimization error: {e}"
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
                "✅ Strategy from the coin initialization"
                if spec_source == "ai"
                else "⚠️ DEFAULT strategy (the coin's spec was not found or is invalid)"
            )
            ui.label(source_text).classes(f"text-{source_color} text-bold q-mt-md")

            with ui.expansion("🔍 Executed strategy_spec", icon="rule").classes("w-full"):
                ui.code(
                    _json.dumps(spec_used, ensure_ascii=False, indent=2),
                    language="json",
                ).classes("w-full")

            ui.label("📊 Statistics").classes("text-h6 q-mt-md")
            ui.table(
                columns=[
                    {"name": "metric", "label": "Metric", "field": "metric", "align": "left"},
                    {"name": "value", "label": "Value", "field": "value", "align": "right"},
                ],
                rows=[
                    {"metric": "Total trades", "value": str(stats.get("total_trades", 0))},
                    {
                        "metric": "Long / Short",
                        "value": f'{stats.get("long_trades", 0)} / {stats.get("short_trades", 0)}',
                    },
                    {"metric": "Win Rate", "value": f'{stats.get("win_rate", 0)}%'},
                    {
                        "metric": "Return (of capital)",
                        "value": f'{stats.get("return_pct", 0):+.2f}%',
                    },
                    {
                        "metric": "Sum of trade % (reference)",
                        "value": f'{stats.get("total_pnl_pct", 0):+.2f}%',
                    },
                    {
                        "metric": "Final capital",
                        "value": f'${stats.get("final_capital", 0):,.2f}',
                    },
                    {
                        "metric": "Max drawdown",
                        "value": f'{stats.get("max_drawdown_pct", 0)}%',
                    },
                    {
                        "metric": "Profit Factor",
                        "value": str(stats.get("profit_factor", 0)),
                    },
                    {
                        "metric": "Average trade",
                        "value": f'{stats.get("avg_trade_pct", 0):+.4f}%',
                    },
                    {
                        "metric": "Expectancy",
                        "value": f'{stats.get("expectancy_pct", stats.get("avg_trade_pct", 0)):+.4f}%',
                    },
                    {
                        "metric": "Average win",
                        "value": f'{stats.get("avg_win_pct", 0):+.4f}%',
                    },
                    {
                        "metric": "Average loss",
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
                ui.label(f"📋 Trades ({len(trades)})").classes("text-h6 q-mt-md")

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
                            ui.label("Entry conditions").classes("text-bold")
                            for r in entry_reasons:
                                mark = "✓" if r.get("passed") else "✗"
                                color = "positive" if r.get("passed") else "negative"
                                ui.label(
                                    f'{mark} {r.get("text", "")} → {r.get("detail", "")}'
                                ).classes(f"text-{color}")

                        snapshot = t.get("entry_snapshot", {})
                        if snapshot:
                            ui.label("Indicator snapshot at entry").classes("text-bold q-mt-md")
                            snapshot_rows = [
                                {"metric": k, "value": str(v)}
                                for k, v in snapshot.items()
                            ]
                            ui.table(
                                columns=[
                                    {"name": "metric", "label": "Parameter", "field": "metric", "align": "left"},
                                    {"name": "value", "label": "Value", "field": "value", "align": "right"},
                                ],
                                rows=snapshot_rows,
                                row_key="metric",
                            ).classes("w-full")
            else:
                ui.label("No trades were made during the period.").classes("text-caption q-mt-md")

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
            ui.label("♻️ Optimization results").classes("text-h6 q-mt-md")

            report_path = (
                result.get("report_path")
                or getattr(self.backtest_optimizer, "last_report_path", "")
            )
            if report_path:
                ui.label(f"📄 Detailed report: {report_path}").classes(
                    "text-caption text-grey"
                )

            if walk_forward_folds <= 1:
                ui.label(
                    f"Validation mode: a single TRAIN/TEST split "
                    f"({int(train_ratio * 100)}% / {int((1 - train_ratio) * 100)}%)."
                ).classes("text-caption text-grey")
            else:
                ui.label(
                    f"Validation mode: walk-forward, TEST windows: {walk_forward_folds}. "
                    "The TRAIN sample expands before each subsequent TEST window."
                ).classes("text-caption text-grey")

            if splits:
                with ui.expansion("🗓️ TRAIN / TEST windows used", icon="date_range").classes("w-full"):
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
                            {"name": "fold", "label": "Window", "field": "fold", "align": "center"},
                            {"name": "train", "label": "TRAIN period", "field": "train", "align": "left"},
                            {"name": "test", "label": "TEST period", "field": "test", "align": "left"},
                        ],
                        rows=split_rows,
                        row_key="fold",
                    ).classes("w-full")

                ui.label(
                    "A candidate is considered acceptable with positive expectancy "
                    "and Profit Factor > 1 on TRAIN and TEST. Win Rate by itself "
                    "is not a profitability criterion."
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
                            "rob_pass": "Yes" if item.get("robustness_passed", False) else "No",
                            "rob_fragile": "Yes" if item.get("robustness_fragile", False) else "No",
                            "rob_score": f'{item.get("robustness_stability_score", 0):.3f}',
                            "score": f'{item.get("selection_score", 0):.4f}',
                            "status": "✅ Accepted" if item.get("acceptable", False) else "— Not accepted",
                        }
                    )

                ui.table(
                    columns=[
                        {"name": "iteration", "label": "Iter.", "field": "iteration", "align": "center"},
                        {"name": "train_trades", "label": "TRAIN trades", "field": "train_trades", "align": "right"},
                        {"name": "test_trades", "label": "TEST trades", "field": "test_trades", "align": "right"},
                        {"name": "train_wr", "label": "TRAIN WR", "field": "train_wr", "align": "right"},
                        {"name": "test_wr", "label": "TEST WR", "field": "test_wr", "align": "right"},
                        {"name": "train_return", "label": "TRAIN return", "field": "train_return", "align": "right"},
                        {"name": "test_return", "label": "TEST return", "field": "test_return", "align": "right"},
                        {"name": "train_exp", "label": "TRAIN Exp.", "field": "train_exp", "align": "right"},
                        {"name": "test_exp", "label": "TEST Exp.", "field": "test_exp", "align": "right"},
                        {"name": "test_pf", "label": "TEST PF", "field": "test_pf", "align": "right"},
                        {"name": "test_dd", "label": "TEST DD", "field": "test_dd", "align": "right"},
                        {"name": "rob_status", "label": "Robust status", "field": "rob_status", "align": "center"},
                        {"name": "rob_pass", "label": "Robust pass", "field": "rob_pass", "align": "center"},
                        {"name": "rob_fragile", "label": "Fragile", "field": "rob_fragile", "align": "center"},
                        {"name": "rob_score", "label": "Robust score", "field": "rob_score", "align": "right"},
                        {"name": "score", "label": "Score", "field": "score", "align": "right"},
                        {"name": "status", "label": "Status", "field": "status", "align": "center"},
                    ],
                    rows=rows,
                    row_key="iteration",
                ).classes("w-full")
            else:
                ui.label("No completed optimization iterations.").classes("text-warning q-mt-md")

            if not best:
                ui.label(
                    "⚠️ No acceptable strategy variant was found. "
                    "This is not a program error: none of the tested versions "
                    "passed the criteria on the TEST data."
                ).classes("text-warning text-bold q-mt-md")
                return

            train_stats = best.get("aggregated_train_stats", {})
            test_stats = best.get("aggregated_test_stats", {})

            ui.label(
                f'🏆 Best iteration: #{best.get("iteration", "?")} | '
                f'TEST WR={test_stats.get("win_rate", 0):.1f}% | '
                f'TEST return={test_stats.get("return_pct", 0):+.2f}% | '
                f'TEST expectancy={test_stats.get("expectancy_pct", 0):+.4f}% | '
                f'TEST PF={test_stats.get("profit_factor", 0):.2f}'
            ).classes("text-positive text-bold text-subtitle1 q-mt-md")

            robustness_assessment = best.get("window_robustness_assessment", {}) or {}
            if robustness_assessment:
                ui.label("🧱 Robustness assessment of the best candidate").classes("text-h6 q-mt-md")
                ui.table(
                    columns=[
                        {"name": "metric", "label": "Metric", "field": "metric", "align": "left"},
                        {"name": "value", "label": "Value", "field": "value", "align": "right"},
                    ],
                    rows=[
                        {"metric": "Status", "value": str(robustness_assessment.get("status", "unstable"))},
                        {"metric": "Share of profitable windows", "value": f'{robustness_assessment.get("profitable_windows_ratio", 0):.0%}'},
                        {"metric": "Share of windows with positive expectancy", "value": f'{robustness_assessment.get("positive_expectancy_windows_ratio", 0):.0%}'},
                        {"metric": "Share of windows with PF > 1", "value": f'{robustness_assessment.get("pf_pass_windows_ratio", 0):.0%}'},
                        {"metric": "Median window return", "value": f'{robustness_assessment.get("median_window_return_pct", 0):+.2f}%'},
                        {"metric": "Worst window", "value": f'{robustness_assessment.get("worst_window_return_pct", 0):+.2f}%'},
                        {"metric": "Stability score", "value": f'{robustness_assessment.get("stability_score", robustness_assessment.get("window_stability_score", 0)):.3f}'},
                    ],
                    row_key="metric",
                ).classes("w-full")

                warnings = robustness_assessment.get("warnings", [])
                reasons = robustness_assessment.get("reasons", [])
                if warnings:
                    ui.label("Robustness assessment warnings").classes("text-bold q-mt-sm")
                    for warning in warnings:
                        ui.label(f"• {warning}").classes("text-warning")

                if reasons:
                    ui.label("Robustness assessment reasons").classes("text-bold q-mt-sm")
                    for reason in reasons:
                        ui.label(f"• {reason}").classes("text-caption")

            last_history_item = history[-1] if history else {}
            if last_history_item:
                ui.label("🧱 Robustness assessment of the last iteration").classes("text-h6 q-mt-md")
                ui.table(
                    columns=[
                        {"name": "metric", "label": "Metric", "field": "metric", "align": "left"},
                        {"name": "value", "label": "Value", "field": "value", "align": "right"},
                    ],
                    rows=[
                        {"metric": "Status", "value": str(last_history_item.get("robustness_status", "unstable"))},
                        {"metric": "Passed", "value": "Yes" if last_history_item.get("robustness_passed", False) else "No"},
                        {"metric": "Fragile", "value": "Yes" if last_history_item.get("robustness_fragile", False) else "No"},
                        {"metric": "Stability score", "value": f'{last_history_item.get("robustness_stability_score", 0):.3f}'},
                    ],
                    row_key="metric",
                ).classes("w-full")

            gap = train_stats.get("win_rate", 0) - test_stats.get("win_rate", 0)
            if gap > 15:
                ui.label(
                    f"⚠️ TRAIN−TEST Win Rate gap: {gap:.1f}%. "
                    "There is a risk of overfitting."
                ).classes("text-warning")

            if test_stats.get("expectancy_pct", 0) <= 0:
                ui.label(
                    "⚠️ TEST expectancy is not positive. "
                    "Such a strategy cannot be considered ready for live trading."
                ).classes("text-warning")

            ui.label("📊 Metrics of the best strategy").classes("text-h6 q-mt-md")
            ui.table(
                columns=[
                    {"name": "metric", "label": "Metric", "field": "metric", "align": "left"},
                    {"name": "train", "label": "TRAIN", "field": "train", "align": "right"},
                    {"name": "test", "label": "TEST", "field": "test", "align": "right"},
                ],
                rows=[
                    {
                        "metric": "Number of trades",
                        "train": str(train_stats.get("total_trades", 0)),
                        "test": str(test_stats.get("total_trades", 0)),
                    },
                    {
                        "metric": "Win Rate",
                        "train": f'{train_stats.get("win_rate", 0):.1f}%',
                        "test": f'{test_stats.get("win_rate", 0):.1f}%',
                    },
                    {
                        "metric": "Return",
                        "train": f'{train_stats.get("return_pct", 0):+.2f}%',
                        "test": f'{test_stats.get("return_pct", 0):+.2f}%',
                    },
                    {
                        "metric": "Expectancy / average trade",
                        "train": f'{train_stats.get("expectancy_pct", 0):+.4f}%',
                        "test": f'{test_stats.get("expectancy_pct", 0):+.4f}%',
                    },
                    {
                        "metric": "Profit Factor",
                        "train": f'{train_stats.get("profit_factor", 0):.2f}',
                        "test": f'{test_stats.get("profit_factor", 0):.2f}',
                    },
                    {
                        "metric": "Max drawdown",
                        "train": f'{train_stats.get("max_drawdown_pct", 0):.2f}%',
                        "test": f'{test_stats.get("max_drawdown_pct", 0):.2f}%',
                    },
                    {
                        "metric": "Average winning trade",
                        "train": f'{train_stats.get("avg_win_pct", 0):+.4f}%',
                        "test": f'{test_stats.get("avg_win_pct", 0):+.4f}%',
                    },
                    {
                        "metric": "Average losing trade",
                        "train": f'-{abs(train_stats.get("avg_loss_pct", 0)):.4f}%',
                        "test": f'-{abs(test_stats.get("avg_loss_pct", 0)):.4f}%',
                    },
                ],
                row_key="metric",
            ).classes("w-full")

            if holdout.get("enabled"):
                ui.label("🧪 Final HOLDOUT validation").classes("text-h6 q-mt-md")
                ui.table(
                    columns=[
                        {"name": "metric", "label": "Metric", "field": "metric", "align": "left"},
                        {"name": "value", "label": "Value", "field": "value", "align": "right"},
                    ],
                    rows=[
                        {"metric": "Status", "value": str(holdout.get("status", "not_run"))},
                        {"metric": "Trades", "value": str(holdout.get("total_trades", 0))},
                        {"metric": "Return", "value": f'{holdout.get("return_pct", 0):+.2f}%'},
                        {"metric": "Expectancy", "value": f'{holdout.get("expectancy_pct", 0):+.4f}%'},
                        {"metric": "Profit Factor", "value": f'{holdout.get("profit_factor", 0):.2f}'},
                        {"metric": "Max drawdown", "value": f'{holdout.get("max_drawdown_pct", 0):.2f}%'},
                    ],
                    row_key="metric",
                ).classes("w-full")

                if holdout.get("warnings"):
                    ui.label("Holdout warnings").classes("text-bold q-mt-sm")
                    for warning in holdout.get("warnings", []):
                        ui.label(f"• {warning}").classes("text-warning")

                if holdout.get("recommendations"):
                    ui.label("Holdout recommendations").classes("text-bold q-mt-sm")
                    for rec in holdout.get("recommendations", []):
                        ui.label(f"• {rec}").classes("text-caption")

            with ui.expansion("🔍 Best strategy_spec", icon="rule").classes("w-full q-mt-md"):
                ui.code(
                    _json.dumps(best.get("spec", {}), ensure_ascii=False, indent=2),
                    language="json",
                ).classes("w-full")

            with ui.expansion("📐 Rules of the best strategy", icon="description").classes("w-full"):
                ui.markdown(best.get("rules") or "No text rules available.")

    def refresh(self):
        if not self.memory:
            return

        current_symbols = list(self.memory.get_all_symbols())

        self.symbol_select.options = current_symbols

        if self.symbol_select.value not in current_symbols:
            self.symbol_select.value = current_symbols[0] if current_symbols else None

        self.symbol_select.update()