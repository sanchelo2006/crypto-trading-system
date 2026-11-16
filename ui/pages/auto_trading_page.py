"""
Auto trading page.

Allows the user to:
- Select coins for auto trading
- Configure capital shares, position size, leverage
- Set loss limits (daily / monthly)
- Specify the optimization period used on a monthly loss
- Start / stop trading
- Close positions (per coin or all)
- See the current trading state and logs
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta

from nicegui import ui

import config
from ui.app_context import app_context
from trading.executor import BybitExecutor
from trading.auto_trader import (
    AutoTraderConfig,
    is_auto_trading_running,
    get_all_states,
    start_auto_trading,
    stop_auto_trading,
    close_all_positions_for_symbol,
    close_all_positions_all_symbols,
)
from ui.auto_mode import AutoModeState

logger = logging.getLogger(__name__)


class AutoTradingPage:
    """
    Auto trading UI page.
    """

    def __init__(
        self,
        *,
        memory,
        fetcher,
        backtest_engine,
        backtest_optimizer,
        ai_client,
        app_settings,
        notify_fn,
    ):
        self.memory = memory
        self.fetcher = fetcher
        self.backtest_engine = backtest_engine
        self.backtest_optimizer = backtest_optimizer
        self.ai_client = ai_client
        self.app_settings = app_settings
        self.notify_fn = notify_fn

        # The executor is created once
        self.executor = BybitExecutor(
            testnet=config.BYBIT_TESTNET,
            default_leverage=10,
        )

        self.container = ui.column().classes("w-full")
        self._coin_configs: dict = {}  # symbol -> dict of UI inputs

        with self.container:
            self.build_ui()

        self.container.visible = False

        # State refresh timer
        ui.timer(3.0, self._refresh_state)

        # Balance refresh timer (once every 15 minutes)
        ui.timer(900.0, self._load_balance_async)

        # Load the balance right when the page opens (with a 0.5 sec delay)
        ui.timer(0.5, self._load_balance_async, once=True)

    def build_ui(self):
        # ================================================================
        # Header
        # ================================================================
        ui.label("🤖 Auto trading").classes("text-h5 q-mb-sm")
        ui.label(
            "Trading on the Bybit futures market (linear perpetual). "
            "The strategy is taken from the coin's memory (strategy_spec). "
            "At most 1 open position per coin at a time."
        ).classes("text-caption text-grey q-mb-md")

        # ================================================================
        # Conflict warning with auto mode
        # ================================================================
        self.conflict_warning = ui.label("").classes(
            "text-warning text-bold q-mb-md"
        ).style("display: none;")

        # ================================================================
        # Total capital
        # ================================================================
        with ui.card().classes("w-full q-mb-md"):
            ui.label("💰 Total capital").classes("text-subtitle1")

            with ui.row().classes("gap-4 items-center"):
                self.total_capital_input = ui.number(
                    "Capital for trading (USDT)",
                    value=1000.0,
                    min=1,
                    format="%.2f",
                ).classes("w-64")

                # Exchange balance label (refreshed every 30 sec)
                self.balance_label = ui.label("🔄 Loading balance...").classes(
                    "text-caption"
                )

                # Manual refresh button
                ui.button(
                    "🔄",
                    on_click=lambda: asyncio.create_task(
                        asyncio.to_thread(self._load_balance_from_exchange)
                    ),
                ).props("flat dense").classes("q-ml-xs").style(
                    "min-width: 32px; padding: 4px;"
                ).tooltip("Refresh balance from the exchange")

            with ui.row().classes("q-mt-xs"):
                self.testnet_label = ui.label("").classes("text-caption")
                self._update_testnet_label()

        # ================================================================
        # Coin selection and settings
        # ================================================================
        with ui.card().classes("w-full q-mb-md"):
            ui.label("📋 Coins for trading").classes("text-subtitle1")

            symbols = list(self.memory.get_all_symbols())

            for symbol in symbols:
                with ui.expansion(
                    f"⚙️ {symbol}",
                    icon="settings",
                ).classes("w-full q-mb-sm"):
                    with ui.column().classes("gap-2 w-full"):
                        with ui.row().classes("gap-4"):
                            capital_share = ui.number(
                                "Share of total capital (%)",
                                value=30.0,
                                min=5,
                                max=100,
                                format="%.1f",
                            ).classes("w-48")

                            position_size = ui.number(
                                "Position size (% of coin capital)",
                                value=10.0,
                                min=10,
                                max=100,
                                format="%.1f",
                            ).classes("w-56")

                            leverage = ui.number(
                                "Leverage (1–100)",
                                value=1,
                                min=1,
                                max=100,
                                format="%d",
                            ).classes("w-32")

                        with ui.row().classes("gap-4"):
                            daily_loss = ui.number(
                                "Daily loss limit (%)",
                                value=1.0,
                                min=1,
                                max=50,
                                format="%.1f",
                            ).classes("w-48")

                            monthly_loss = ui.number(
                                "Monthly loss limit (%)",
                                value=15.0,
                                min=5,
                                max=100,
                                format="%.1f",
                            ).classes("w-48")

                            opt_days = ui.number(
                                "Optimization period (days)",
                                value=180,
                                min=30,
                                max=365,
                                format="%d",
                            ).classes("w-48")

                        with ui.row().classes("gap-4"):
                            opt_iterations = ui.number(
                                "Optimization iterations",
                                value=8,
                                min=5,
                                max=30,
                                format="%d",
                            ).classes("w-48")

                            opt_folds = ui.number(
                                "TEST windows (walk-forward)",
                                value=3,
                                min=1,
                                max=8,
                                format="%d",
                            ).classes("w-48")

                    self._coin_configs[symbol] = {
                        "capital_share": capital_share,
                        "position_size": position_size,
                        "leverage": leverage,
                        "daily_loss": daily_loss,
                        "monthly_loss": monthly_loss,
                        "opt_days": opt_days,
                        "opt_iterations": opt_iterations,
                        "opt_folds": opt_folds,
                    }

        # ================================================================
        # Control buttons
        # ================================================================
        with ui.card().classes("w-full q-mb-md"):
            ui.label("🎮 Controls").classes("text-subtitle1 q-mb-sm")

            with ui.row().classes("gap-2 flex-wrap"):
                self.start_btn = ui.button(
                    "🚀 Start trading (selected coins)",
                    on_click=self._start_selected,
                ).classes("bg-positive text-white")

                self.stop_btn = ui.button(
                    "⏹ Stop trading (selected coins)",
                    on_click=self._stop_selected,
                ).classes("bg-negative text-white")

            with ui.row().classes("gap-2 flex-wrap q-mt-sm"):
                ui.button(
                    "🔒 Close all positions (all coins)",
                    on_click=self._close_all_positions,
                ).classes("bg-warning text-white")

                ui.button(
                    "🔒 Close positions (selected coins)",
                    on_click=self._close_selected_positions,
                ).classes("bg-orange text-white")

            # Coin selection checkboxes
            ui.label("Coin selection:").classes("text-bold q-mt-md")
            self.symbol_checkboxes = {}
            with ui.row().classes("gap-4"):
                for symbol in self.memory.get_all_symbols():
                    self.symbol_checkboxes[symbol] = ui.checkbox(
                        symbol, value=False
                    )

        # ================================================================
        # Current state
        # ================================================================
        with ui.card().classes("w-full q-mb-md"):
            ui.label("📊 Current state").classes("text-subtitle1 q-mb-sm")

            self.state_table = ui.table(
                columns=[
                    {"name": "symbol", "label": "Coin", "field": "symbol", "align": "left"},
                    {"name": "status", "label": "Status", "field": "status", "align": "left"},
                    {"name": "positions", "label": "Positions", "field": "positions", "align": "center"},
                    {"name": "capital", "label": "Capital", "field": "capital", "align": "right"},
                    {"name": "daily_pnl", "label": "Daily P&L", "field": "daily_pnl", "align": "right"},
                    {"name": "monthly_pnl", "label": "Monthly P&L", "field": "monthly_pnl", "align": "right"},
                    {"name": "message", "label": "Message", "field": "message", "align": "left"},
                ],
                rows=[],
                row_key="symbol",
            ).classes("w-full")

    # ----------------------------------------------------------------
    # Button handlers
    # ----------------------------------------------------------------

    def _get_selected_symbols(self) -> list:
        return [
            sym for sym, cb in self.symbol_checkboxes.items()
            if cb.value
        ]

    def _get_total_capital(self) -> float:
        return float(self.total_capital_input.value or 1000.0)

    def _start_selected(self):
        selected = self._get_selected_symbols()
        if not selected:
            ui.notify("Select at least one coin.", type="warning")
            return

        # Check for a conflict with auto mode
        for sym in selected:
            auto_state = AutoModeState.get_state(sym)
            if auto_state and auto_state.get("is_running"):
                ui.notify(
                    f"Auto mode is enabled for {sym}. "
                    "Stop it before starting auto trading.",
                    type="warning",
                )
                return

            if is_auto_trading_running(sym):
                ui.notify(
                    f"Auto trading for {sym} is already running.",
                    type="info",
                )
                return

        total_capital = self._get_total_capital()

        # Check the sum of capital shares
        total_share = 0.0
        for sym in selected:
            cfg_inputs = self._coin_configs.get(sym, {})
            share = float(cfg_inputs["capital_share"].value or 30)
            total_share += share

        if total_share > 100:
            ui.notify(
                f"Total capital share ({total_share}%) exceeds 100%.",
                type="warning",
            )
            return

        for sym in selected:
            cfg_inputs = self._coin_configs.get(sym, {})

            cfg = AutoTraderConfig(
                symbol=sym,
                category="linear",
                capital_share_pct=float(cfg_inputs["capital_share"].value or 30),
                position_size_pct=float(cfg_inputs["position_size"].value or 50),
                leverage=int(cfg_inputs["leverage"].value or 10),
                daily_loss_limit_pct=float(cfg_inputs["daily_loss"].value or 5),
                monthly_loss_limit_pct=float(cfg_inputs["monthly_loss"].value or 15),
                optimization_days_back=int(cfg_inputs["opt_days"].value or 90),
                optimization_iterations=int(cfg_inputs["opt_iterations"].value or 10),
                walk_forward_folds=int(cfg_inputs["opt_folds"].value or 3),
                initial_capital=total_capital * float(cfg_inputs["capital_share"].value or 30) / 100,
                commission_pct=0.05,
            )

            def worker(s=sym, c=cfg):
                try:
                    start_auto_trading(
                        c,
                        executor=self.executor,
                        fetcher=self.fetcher,
                        memory=self.memory,
                        backtest_engine=self.backtest_engine,
                        backtest_optimizer=self.backtest_optimizer,
                        ai_client=self.ai_client,
                        notify_fn=self.notify_fn,
                        total_capital_usdt=self._get_total_capital(),
                    )
                except Exception as e:
                    logger.exception(f"Auto trading start error {s}")
                    self.notify_fn(f"Start error {s}: {e}", "negative")

            asyncio.create_task(asyncio.to_thread(worker))

        ui.notify(f"Starting auto trading for: {', '.join(selected)}", type="positive")

    def _stop_selected(self):
        selected = self._get_selected_symbols()
        if not selected:
            ui.notify("Select at least one coin.", type="warning")
            return

        for sym in selected:
            if is_auto_trading_running(sym):
                stop_auto_trading(sym, close_positions=False, reason="manual")
            else:
                ui.notify(f"Auto trading for {sym} is not running.", type="info")

    def _close_all_positions(self):
        with ui.dialog() as dialog, ui.card():
            ui.label("Close ALL positions for ALL coins?")
            ui.label("This action cannot be undone!").classes("text-negative")

            with ui.row().classes("justify-end q-mt-md"):
                ui.button("Cancel", on_click=dialog.close)
                ui.button(
                    "Close all",
                    on_click=lambda: self._do_close_all(dialog),
                    color="negative",
                )

            dialog.open()

    def _do_close_all(self, dialog):
        dialog.close()
        close_all_positions_all_symbols()
        ui.notify("All positions closed.", type="warning")

    def _close_selected_positions(self):
        selected = self._get_selected_symbols()
        if not selected:
            ui.notify("Select at least one coin.", type="warning")
            return

        for sym in selected:
            close_all_positions_for_symbol(sym)

        ui.notify(f"Positions closed for: {', '.join(selected)}", type="info")

    # ----------------------------------------------------------------
    # State refresh
    # ----------------------------------------------------------------

    def _refresh_state(self):
        """Refreshes the state table (called by the timer)."""
        states = get_all_states()

        rows = []

        for sym, state in states.items():
            is_running = state.get("is_running", False)
            is_optimizing = state.get("is_optimizing", False)

            if is_optimizing:
                status = "🔄 Optimizing"
            elif is_running:
                status = "🟢 Running"
            else:
                status = "⏹ Stopped"

            # --- HEARTBEAT: check freshness ---
            last_iter_ts = state.get("_last_iteration_ts")
            stale = False
            if is_running and last_iter_ts:
                import time
                age = time.time() - last_iter_ts
                if age > 120:  # 2 minutes without updates
                    status = "🟡 No data (network error?)"
                    stale = True

            # Build the detailed status line
            details = []
            if state.get("last_iteration_utc"):
                details.append(f"Check: {state['last_iteration_utc']}")
            if state.get("last_candle_time"):
                details.append(f"Candle: {state['last_candle_time']}")
            if state.get("last_price"):
                details.append(f"Price: {state['last_price']:.4f}")
            if state.get("candles_processed"):
                details.append(f"Candles processed: {state['candles_processed']}")
            if state.get("last_indicators_summary"):
                details.append(state['last_indicators_summary'])

            message_text = state.get("status_message", "")
            if details:
                message_text += " | " + " | ".join(details)

            # Format PnL with percentages
            current_cap = state.get("current_capital", state.get("allocated_capital", 0))
            daily_pnl_usdt = state.get("daily_pnl", 0)
            monthly_pnl_usdt = state.get("monthly_pnl", 0)
            daily_pnl_pct = state.get("daily_pnl_pct", 0)
            monthly_pnl_pct = state.get("monthly_pnl_pct", 0)

            rows.append({
                "symbol": sym,
                "status": status,
                "positions": state.get("open_positions", 0),
                "capital": f"{current_cap:.2f} USDT",
                "daily_pnl": f"{daily_pnl_usdt:+.2f} USDT ({daily_pnl_pct:+.1f}%)",
                "monthly_pnl": f"{monthly_pnl_usdt:+.2f} USDT ({monthly_pnl_pct:+.1f}%)",
                "message": message_text,
            })

        self.state_table.rows = rows
        self.state_table.update()

        # Conflict warning
        self._check_conflicts()

    def _check_conflicts(self):
        """Checks conflicts between auto trading and auto mode."""
        warnings = []

        for sym in self.memory.get_all_symbols():
            auto_state = AutoModeState.get_state(sym)
            auto_mode_running = auto_state and auto_state.get("is_running", False)
            auto_trading = is_auto_trading_running(sym)

            if auto_mode_running and auto_trading:
                warnings.append(
                    f"⚠️ {sym}: auto trading and auto mode are enabled at the same time!"
                )
            elif auto_mode_running:
                warnings.append(
                    f"🔒 {sym}: auto mode is enabled — auto trading is unavailable."
                )
            elif auto_trading:
                warnings.append(
                    f"🔒 {sym}: auto trading is enabled — auto mode is unavailable."
                )

        if warnings:
            self.conflict_warning.set_text("\n".join(warnings))
            self.conflict_warning.style("display: block;")
        else:
            self.conflict_warning.set_text("")
            self.conflict_warning.style("display: none;")

    def _update_testnet_label(self):
        if config.BYBIT_TESTNET:
            self.testnet_label.set_text(
                "🧪 TESTNET — trading on a test account."
            )
            self.testnet_label.classes("text-warning")
        else:
            self.testnet_label.set_text(
                "🔴 MAINNET — trading on a real account!"
            )
            self.testnet_label.classes("text-negative text-bold")

    def _load_balance_from_exchange(self):
        """
        Loads the USDT balance from the exchange and updates the label.
        Does NOT overwrite the input field — the user may manually
        specify a smaller amount (for risk control).
        """
        success, balance = self.executor.get_usdt_balance()

        if success and balance is not None:
            self.balance_label.set_text(
                f"💰 Exchange balance: {balance:.2f} USDT"
            )
            self.balance_label.classes("text-positive text-bold")

            # Autofill the field on first load (if the user has not changed it)
            current_value = float(self.total_capital_input.value or 0)
            if current_value <= 1000.0:  # default value — update it
                self.total_capital_input.set_value(balance)
        else:
            self.balance_label.set_text(
                "⚠️ Failed to load the balance from the exchange"
            )
            self.balance_label.classes("text-warning")

    async def _load_balance_async(self):
        """Async wrapper for loading the balance (for ui.timer)."""
        await asyncio.to_thread(self._load_balance_from_exchange)

    def refresh(self):
        """Refreshes the coin list (when new coins are added)."""
        current_symbols = list(self.memory.get_all_symbols())
        for sym in current_symbols:
            if sym not in self.symbol_checkboxes:
                with ui.row():
                    self.symbol_checkboxes[sym] = ui.checkbox(sym, value=False)