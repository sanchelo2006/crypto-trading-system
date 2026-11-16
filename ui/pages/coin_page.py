"""
Page for a specific coin.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from nicegui import ui

import config


class CoinPage:
    """
    Management page for a single coin.

    This is the richest class of the interface:
    - coin initialization;
    - running morning / intraday packages;
    - starting/stopping auto mode;
    - manual position entry/exit;
    - viewing analysis, indicators, rules and the archive.

    At this stage the class is moved with almost no logic changes.
    The only systemic difference is that dependencies are passed explicitly
    into the constructor.
    """

    def __init__(
        self,
        *,
        symbol: str,
        memory,
        fetcher,
        ai_client,
        app_settings,
        auto_mode_cls,
        process_ai_response_fn,
        clear_coin_data_fn,
        parse_rules_from_text_fn,
        add_notification_fn,
        schedule_refresh_fn,
        set_manual_action_fn,
        clear_manual_action_fn,
        render_sidebar_buttons_fn,
        summary_page_getter,
        journal_page_getter,
    ):
        self.symbol = symbol

        self.memory = memory
        self.fetcher = fetcher
        self.ai_client = ai_client
        self.app_settings = app_settings
        self.auto_mode_cls = auto_mode_cls
        self.process_ai_response_fn = process_ai_response_fn
        self.clear_coin_data_fn = clear_coin_data_fn
        self.parse_rules_from_text_fn = parse_rules_from_text_fn
        self.add_notification_fn = add_notification_fn
        self.schedule_refresh_fn = schedule_refresh_fn
        self.set_manual_action_fn = set_manual_action_fn
        self.clear_manual_action_fn = clear_manual_action_fn
        self.render_sidebar_buttons_fn = render_sidebar_buttons_fn
        self.summary_page_getter = summary_page_getter
        self.journal_page_getter = journal_page_getter

        self.container = ui.column().classes("w-full")

        with self.container:
            self.build_ui()

        self.container.visible = False

    def build_ui(self):
        with ui.row().classes("items-center justify-between w-full"):
            ui.label(f"💰 {self.symbol}").classes("text-h4 text-dark")
            self.adx_label = ui.label("").classes("text-subtitle1 text-grey q-ml-md").style("margin-left: 10px;")

            with ui.row().classes("gap-2"):
                self.init_btn = ui.button(
                    "🪄 Initialize",
                    on_click=self.init_symbol,
                ).classes("bg-primary text-white")

                self.auto_btn = ui.button(
                    "🚀 Auto mode",
                    on_click=self.toggle_auto,
                ).classes("bg-positive text-white")

                ui.button("🔄 Refresh", on_click=self.refresh).classes("bg-secondary")
                ui.button("🗑 Clear", on_click=self.clear_dialog).classes("bg-negative text-white")

        with ui.row().classes("gap-2 w-full my-2"):
            ui.button("🌅 Morning package", on_click=self.run_morning).classes("bg-info text-white")
            ui.button("⏱ Intraday package", on_click=self.run_intraday).classes("bg-info text-white")
            ui.button("🔒 Close day", on_click=self.close_day).classes("bg-warning text-white")
            ui.button("🔓 Open day", on_click=self.open_day).classes("bg-positive text-white")

        with ui.tabs() as self.tabs:
            self.analysis_tab = ui.tab("📊 Analysis")
            self.rules_tab = ui.tab("📐 Rules")
            self.archive_tab = ui.tab("📦 Day archive")

        with ui.tab_panels(self.tabs, value=self.analysis_tab).classes("w-full"):
            with ui.tab_panel(self.analysis_tab):
                self.build_analysis_tab()
            with ui.tab_panel(self.rules_tab):
                self.build_rules_tab()
            with ui.tab_panel(self.archive_tab):
                self.build_archive_tab()

        self.refresh()

    def switch_to_archive_tab(self):
        self.tabs.set_value(self.archive_tab)

    def build_analysis_tab(self):
        with ui.row().classes("w-full gap-4").style("flex-wrap: nowrap; align-items: stretch;"):
            with ui.column().classes("flex-1").style("min-width: 0;"):
                self.analysis_card = ui.card().classes("w-full q-pa-md")
                with self.analysis_card:
                    ui.label("📊 Analysis").classes("text-h6")
                    self.analysis_table = ui.table(
                        columns=[
                            {"name": "field", "label": "Parameter", "field": "field", "align": "left"},
                            {"name": "value", "label": "Value", "field": "value", "align": "left"},
                        ],
                        rows=[],
                        row_key="field",
                    ).classes("w-full")
                    self.summary_label = ui.label("").classes("text-body1 q-mt-md")

            with ui.column().classes("flex-1").style("min-width: 0;"):
                self.signal_setup_card = ui.card().classes("w-full q-pa-md")
                with self.signal_setup_card:
                    ui.label("📐 Setup / Signal").classes("text-h6")
                    self.signal_setup_table = ui.table(
                        columns=[
                            {"name": "field", "label": "Parameter", "field": "field", "align": "left"},
                            {"name": "value", "label": "Value", "field": "value", "align": "left"},
                        ],
                        rows=[],
                        row_key="field",
                    ).classes("w-full")
                    self.signal_setup_conditions = ui.column().classes("w-full q-mt-md")
                self.signal_setup_card.visible = False

                self.indicators_card = ui.card().classes("w-full q-pa-md bg-blue-1")
                with self.indicators_card:
                    ui.label("📈 Indicators (last 5 values)").classes("text-h6")
                    self.indicators_table = ui.table(
                        columns=[{"name": "time", "label": "Time", "field": "time", "align": "left"}],
                        rows=[],
                        row_key="time",
                    ).classes("w-full")
                    self.volume_profile_container = ui.column().classes("w-full")
                self.indicators_card.visible = False

            with ui.column().classes("flex-1").style("min-width: 0;"):
                self.active_trade_card = ui.card().classes("w-full q-pa-md bg-grey-2")
                with self.active_trade_card:
                    ui.label("🔴 Active position").classes("text-h6")
                    self.active_trade_table = ui.table(
                        columns=[
                            {"name": "field", "label": "Parameter", "field": "field", "align": "left"},
                            {"name": "value", "label": "Value", "field": "value", "align": "left"},
                        ],
                        rows=[],
                        row_key="field",
                    ).classes("w-full")

                    ui.button(
                        "✏️ Edit",
                        on_click=self.edit_active_trade_dialog,
                    ).classes("bg-secondary text-white q-mb-sm")

                    with ui.row().classes("gap-2"):
                        self.close_price_input = ui.input("Exit price", value="0").classes("w-32")
                        self.close_position_size_input = ui.input("Position size (USDT)", value="0").classes("w-40")
                        self.close_reason = ui.select(
                            options=["manual", "stop_loss", "take_profit"],
                            value="manual",
                            label="Reason",
                        )
                        ui.button(
                            "🔒 Close position",
                            on_click=self.close_active_trade,
                        ).classes("bg-negative text-white")
                self.active_trade_card.visible = False

                self.entry_card = ui.card().classes("w-full q-pa-md")
                with self.entry_card:
                    ui.label("➕ Open position").classes("text-h6")
                    with ui.row().classes("gap-2 items-center"):
                        direction = ui.select(options=["long", "short"], value="long", label="Direction")
                        entry_price = ui.input("Entry price", value="0")
                        stop_loss = ui.input("Stop loss", value="0")
                        take_profit = ui.input("Take profit", value="0")
                        ui.button(
                            "✅ Enter",
                            on_click=lambda: self.open_trade(
                                direction.value,
                                entry_price.value,
                                stop_loss.value,
                                take_profit.value,
                            ),
                        ).classes("bg-positive text-white")
                self.entry_card.visible = False

                self.position_advice_card = ui.card().classes("w-full q-pa-md bg-blue-1")
                with self.position_advice_card:
                    ui.label("📋 Position advice").classes("text-h6")
                    self.position_advice_table = ui.table(
                        columns=[
                            {"name": "field", "label": "Parameter", "field": "field", "align": "left"},
                            {"name": "value", "label": "Value", "field": "value", "align": "left"},
                        ],
                        rows=[],
                        row_key="field",
                    ).classes("w-full")
                    self.position_advice_conditions = ui.column().classes("w-full q-mt-md")
                self.position_advice_card.visible = False

    def build_rules_tab(self):
        self.rules_table = ui.table(
            columns=[
                {"name": "num", "label": "#", "field": "num", "align": "left", "style": "width: 40px;"},
                {"name": "condition", "label": "Condition", "field": "condition", "align": "left", "style": "white-space: normal; max-width: 300px;"},
                {"name": "description", "label": "Description", "field": "description", "align": "left", "style": "white-space: normal; max-width: 400px;"},
            ],
            rows=[],
            row_key="num",
        ).classes("w-full")

        self.rules_text_area = ui.textarea("Rules (Markdown)").classes("w-full").props("outlined")
        ui.button("💾 Save rules", on_click=self.save_rules).classes("bg-positive text-white")

    def build_archive_tab(self):
        self.archive_table = ui.table(
            columns=[
                {"name": "date", "label": "Date", "field": "date", "align": "left"},
                {"name": "pnl", "label": "P&L", "field": "pnl", "align": "center"},
                {"name": "trades", "label": "Trades", "field": "trades", "align": "center"},
            ],
            rows=[],
            row_key="date",
        ).classes("w-full")

        self.archive_table.on("rowClick", lambda e: self.show_archive_day(e.args[1]["date"]))

    def refresh(self):
        if not self.memory:
            return

        cm = self.memory.get(self.symbol)
        state = cm.get_state()
        analysis = cm.get_last_analysis()

        self.init_btn.visible = not cm.is_initialized()

        # ----- ADX threshold -----
        market_stats = state.get("market_stats", {})
        adx_threshold = market_stats.get("adx_threshold")
        if adx_threshold is not None:
            self.adx_label.set_text(f"ADX threshold = {adx_threshold:.1f}")
        else:
            self.adx_label.set_text("ADX threshold = 20 (default)")

        # ----- Analysis -----
        rows = []
        if analysis:
            rows.append({"field": "Analysis time", "value": analysis.get("timestamp_utc", "N/A")})
            rows.append(
                {
                    "field": "Price",
                    "value": f"${analysis.get('current_price', 0):.4f}" if analysis.get("current_price") else "N/A",
                }
            )
            rows.append({"field": "24h change", "value": f"{analysis.get('price_change_24h_pct', 0):.2f}%"})
            rows.append({"field": "Order book", "value": analysis.get("orderbook_bias", "N/A")})
            rows.append({"field": "OI trend", "value": analysis.get("oi_trend", "N/A")})
            rows.append({"field": "Funding rate", "value": analysis.get("funding_rate_status", "N/A")})
            rows.append({"field": "Fear & Greed", "value": analysis.get("fear_greed_state", "N/A")})
            rows.append({"field": "L/S Ratio", "value": analysis.get("long_short_ratio_signal", "N/A")})
            rows.append({"field": "News", "value": analysis.get("news_impact", "N/A")})
            summary = analysis.get("summary", "")
            self.summary_label.set_text(f"📝 Summary: {summary}")
        else:
            rows.append({"field": "Analysis", "value": "No data"})
            self.summary_label.set_text("")

        self.analysis_table.rows = rows
        self.analysis_table.update()

        # ----- Setup / Signal -----
        active = state.get("active_trade")
        if not active:
            signal = state.get("last_signal")
            setup = state.get("last_setup") if not signal else None

            if signal and signal.get("confidence_pct", 0) > 0:
                self.signal_setup_card.visible = True
                self.signal_setup_conditions.clear()

                rows = [
                    {"field": "Type", "value": "🚨 SIGNAL"},
                    {"field": "Direction", "value": signal.get("direction", "?").upper()},
                    {"field": "Confidence", "value": f"{signal.get('confidence_pct', 0)}%"},
                    {"field": "Entry", "value": f"${signal.get('entry_price', 0):.4f}" if signal.get("entry_price") else "Market"},
                    {"field": "SL", "value": f"${signal.get('stop_loss', 0):.4f}"},
                    {"field": "TP", "value": ", ".join([f"${tp:.4f}" for tp in signal.get("take_profit", [])])},
                    {"field": "Rationale", "value": signal.get("signal_rationale", "")},
                ]
                self.signal_setup_table.rows = rows
                self.signal_setup_table.update()

                with self.signal_setup_conditions:
                    ui.label("Entry conditions:").classes("text-bold")
                    for cond in signal.get("entry_conditions", []):
                        with ui.row().classes("w-full"):
                            ui.label(f"• {cond}")

                    ui.label("Cancel conditions:").classes("text-bold q-mt-md")
                    for cond in signal.get("cancel_conditions", []):
                        with ui.row().classes("w-full"):
                            ui.label(f"• {cond}")

            elif setup:
                self.signal_setup_card.visible = True
                self.signal_setup_conditions.clear()

                rows = [
                    {"field": "Type", "value": "📐 SETUP"},
                    {"field": "Direction", "value": setup.get("direction", "?").upper()},
                    {"field": "Confidence", "value": f"{setup.get('confidence_pct', 0)}%"},
                    {"field": "Entry", "value": f"${setup.get('entry_price', 0):.4f}" if setup.get("entry_price") else "Market"},
                    {"field": "SL", "value": f"${setup.get('stop_loss', 0):.4f}"},
                    {"field": "TP", "value": ", ".join([f"${tp:.4f}" for tp in setup.get("take_profit", [])])},
                    {"field": "Rationale", "value": setup.get("rationale", "")},
                ]
                self.signal_setup_table.rows = rows
                self.signal_setup_table.update()

                with self.signal_setup_conditions:
                    ui.label("Entry conditions:").classes("text-bold")
                    for cond in setup.get("conditions", []):
                        with ui.row().classes("w-full"):
                            ui.label(f"• {cond}")

                    ui.label("Cancel conditions:").classes("text-bold q-mt-md")
                    for cond in setup.get("cancel_conditions", []):
                        with ui.row().classes("w-full"):
                            ui.label(f"• {cond}")
            else:
                self.signal_setup_card.visible = False
        else:
            self.signal_setup_card.visible = False

        # ----- Indicators -----
        indicators = state.get("indicators", {})
        timestamps = state.get("indicator_timestamps", [])

        if indicators and timestamps:
            self.indicators_card.visible = True

            latest_timestamps = timestamps[-5:] if len(timestamps) >= 5 else timestamps
            cols = [{"name": "time", "label": "Time", "field": "time", "align": "left"}]
            ind_names = sorted(k for k in indicators.keys() if k != "volume_profile")

            for ind_name in ind_names:
                if isinstance(indicators[ind_name], list):
                    cols.append({"name": ind_name, "label": ind_name, "field": ind_name, "align": "right"})

            self.indicators_table.columns = cols

            rows = []
            for i, ts in enumerate(latest_timestamps):
                row = {"time": ts}
                for ind_name in ind_names:
                    if isinstance(indicators[ind_name], list):
                        values = indicators[ind_name]
                        if len(values) >= len(latest_timestamps):
                            val = values[-len(latest_timestamps) + i]
                        else:
                            val = None

                        if isinstance(val, (int, float)):
                            row[ind_name] = f"{val:.4f}"
                        else:
                            row[ind_name] = str(val) if val is not None else ""

                rows.append(row)

            self.indicators_table.rows = rows
            self.indicators_table.update()

            self.volume_profile_container.clear()
            volume_profile = indicators.get("volume_profile")
            if volume_profile and isinstance(volume_profile, list) and len(volume_profile) > 0:
                with self.volume_profile_container:
                    ui.label("📊 Volume profile (top 5 levels)").classes("text-bold q-mt-md")
                    ui.table(
                        columns=[
                            {"name": "level", "label": "Price", "field": "level", "align": "right"},
                            {"name": "volume", "label": "Volume", "field": "volume", "align": "right"},
                        ],
                        rows=[
                            {"level": f"{v[0]:.4f}", "volume": f"{v[1]:.0f}"}
                            for v in volume_profile
                        ],
                        row_key="level",
                    ).classes("w-full")
        else:
            self.indicators_card.visible = False

        # ----- Active position -----
        if active:
            self.active_trade_card.visible = True
            self.entry_card.visible = False

            trade_rows = [
                {"field": "Direction", "value": active.get("direction", "?").upper()},
                {"field": "Entry price", "value": f"${active.get('entry_price', 0):.4f}"},
                {"field": "Stop loss", "value": f"${active.get('stop_loss', 0):.4f}"},
                {"field": "Take profit", "value": ", ".join([f"${tp:.4f}" for tp in active.get("take_profit", [])])},
                {"field": "Entry type", "value": active.get("entry_type", "manual")},
            ]
            self.active_trade_table.rows = trade_rows
            self.active_trade_table.update()

            try:
                cp = self.fetcher.get_current_price(self.symbol, "linear")
                if cp:
                    self.close_price_input.value = str(cp)
            except Exception:
                pass

            pos_update = state.get("last_position_update", {})
            if pos_update or analysis:
                self.position_advice_card.visible = True
                advice_rows = []

                if pos_update.get("advice"):
                    advice_rows.append({"field": "Advice", "value": pos_update["advice"]})
                if pos_update.get("new_stop_loss"):
                    advice_rows.append({"field": "New stop loss", "value": f"${pos_update['new_stop_loss']:.4f}"})
                if pos_update.get("partial_take_profit_pct"):
                    advice_rows.append({"field": "Partial take profit", "value": f"{pos_update['partial_take_profit_pct']}%"})

                if not advice_rows:
                    advice_rows.append({"field": "Status", "value": "Waiting for update..."})

                self.position_advice_table.rows = advice_rows
                self.position_advice_table.update()

                cancel_conditions = []
                setup = state.get("last_setup")
                if setup and setup.get("cancel_conditions"):
                    cancel_conditions = setup["cancel_conditions"]

                signal = state.get("last_signal")
                if signal and signal.get("cancel_conditions"):
                    cancel_conditions = signal["cancel_conditions"]

                self.position_advice_conditions.clear()
                if cancel_conditions:
                    with self.position_advice_conditions:
                        ui.label("Exit conditions:").classes("text-bold")
                        for cond in cancel_conditions:
                            with ui.row().classes("w-full"):
                                ui.label(f"• {cond}")
                else:
                    with self.position_advice_conditions:
                        ui.label("No exit conditions.").classes("text-caption")
            else:
                self.position_advice_card.visible = False
        else:
            self.active_trade_card.visible = False
            self.entry_card.visible = True
            self.position_advice_card.visible = False

        # ----- Rules -----
        tr_list = state.get("trading_rules_list", [])
        rm_list = state.get("risk_management_rules_list", [])
        rules_text = cm.get_rules()

        if not tr_list and not rm_list and rules_text:
            tr_list, rm_list = self.parse_rules_from_text_fn(rules_text)
            if tr_list or rm_list:
                cm.update_state(
                    {
                        "trading_rules_list": tr_list,
                        "risk_management_rules_list": rm_list,
                    }
                )

        rules_rows = []
        if tr_list:
            for i, rule in enumerate(tr_list, 1):
                parts = rule.split(":", 1) if ":" in rule else (
                    rule.split("---", 1) if "---" in rule else [rule, ""]
                )
                rules_rows.append(
                    {
                        "num": i,
                        "condition": parts[0].strip(),
                        "description": parts[1].strip() if len(parts) > 1 else "",
                    }
                )

        if rm_list:
            rules_rows.append({"num": "", "condition": "=== RISK MANAGEMENT ===", "description": ""})
            for i, rule in enumerate(rm_list, len(tr_list) + 1):
                parts = rule.split(":", 1) if ":" in rule else (
                    rule.split("---", 1) if "---" in rule else [rule, ""]
                )
                rules_rows.append(
                    {
                        "num": i,
                        "condition": parts[0].strip(),
                        "description": parts[1].strip() if len(parts) > 1 else "",
                    }
                )

        self.rules_table.rows = rules_rows
        self.rules_table.update()
        self.rules_text_area.set_value(rules_text if rules_text else "No rules.")

        # ----- Archive -----
        archive_dates = cm.get_archive_dates()
        archive_rows = []
        for date_str in archive_dates:
            pnl = cm.get_day_pnl(date_str)
            day_trades = cm.journal[cm.journal["date"] == date_str] if not cm.journal.empty else pd.DataFrame()
            trades_count = len(day_trades)
            archive_rows.append(
                {
                    "date": date_str,
                    "pnl": f"{pnl:+.2f}%",
                    "trades": str(trades_count),
                }
            )

        self.archive_table.rows = archive_rows
        self.archive_table.update()

        # ----- Auto mode -----
        auto_state = self.auto_mode_cls.get_state(self.symbol)
        if auto_state and auto_state.get("is_running"):
            self.auto_btn.set_text("⏹ Stop")
            self.auto_btn.props("color=negative")
        else:
            self.auto_btn.set_text("🚀 Auto mode")
            self.auto_btn.props("color=positive")

        summary_page = self.summary_page_getter()
        if summary_page:
            summary_page.refresh()

        journal_page = self.journal_page_getter()
        if journal_page:
            journal_page.refresh()

    def show_archive_day(self, date_str: str):
        cm = self.memory.get(self.symbol)
        data = cm.load_archive_day(date_str)
        if not data:
            ui.notify("Data not found", type="warning")
            return

        with ui.dialog() as dialog, ui.card().style("min-width: 600px; max-height: 80vh; overflow-y: auto;"):
            ui.label(f"Archive {self.symbol} — {date_str}").classes("text-h6")
            pnl = cm.get_day_pnl(date_str)
            ui.label(f"Daily P&L: {pnl:+.2f}%").classes("text-h6")

            if data.get("day_trades"):
                for trade in data["day_trades"]:
                    ui.label(
                        f"{trade.get('time_utc','')} {trade.get('direction','')} "
                        f"Entry: {trade.get('entry_price','')} "
                        f"Exit: {trade.get('exit_price','')} "
                        f"Result: {trade.get('result_pct','')}%"
                    ).classes("text-caption")

            ui.button("Close", on_click=dialog.close)
            dialog.open()

    def close_day(self):
        cm = self.memory.get(self.symbol)
        if cm.is_day_closed():
            ui.notify("Day is already closed", type="warning")
            return

        with ui.dialog() as dialog, ui.card():
            ui.label(f"Close the day for {self.symbol}?")

            with ui.row().classes("justify-end"):
                ui.button("Cancel", on_click=dialog.close)

                def do_close():
                    cm.close_day()
                    self.auto_mode_cls.stop(self.symbol)
                    self.refresh()
                    self.render_sidebar_buttons_fn()
                    ui.notify(f"{self.symbol}: day closed", type="positive")
                    dialog.close()

                ui.button("Close", on_click=do_close, color="negative")

            dialog.open()

    def open_day(self):
        cm = self.memory.get(self.symbol)
        if not cm.is_day_closed():
            ui.notify("Day is already open", type="info")
            return

        with ui.dialog() as dialog, ui.card():
            ui.label(f"Open a new day for {self.symbol}?")

            with ui.row().classes("justify-end"):
                ui.button("Cancel", on_click=dialog.close)

                def do_open():
                    cm.open_new_day()
                    self.refresh()
                    self.render_sidebar_buttons_fn()
                    ui.notify(f"{self.symbol}: day opened", type="positive")
                    dialog.close()

                ui.button("Open", on_click=do_open, color="positive")

            dialog.open()

    def toggle_auto(self):
        state = self.auto_mode_cls.get_state(self.symbol)
        if state and state.get("is_running"):
            self.auto_mode_cls.stop(self.symbol)
        else:
            cm = self.memory.get(self.symbol)
            if cm.is_day_closed():
                ui.notify("Day is closed. Open a new day to trade.", type="warning")
                return

            pc = next(
                (p for p in config.TRADING_PAIRS if p["symbol"] == self.symbol),
                {"symbol": self.symbol, "category": "linear"},
            )

            self.auto_mode_cls.start(
                self.symbol,
                pc["category"],
                update_callback=lambda: setattr(self, "needs_update", True),
            )

        self.refresh()

    def init_symbol(self):
        self.set_manual_action_fn(self.symbol, "🪄 Initialization...")

        def worker():
            try:
                cm = self.memory.get(self.symbol)

                pc = next(
                    (p for p in config.TRADING_PAIRS if p["symbol"] == self.symbol),
                    {"symbol": self.symbol, "category": "linear"},
                )

                try:
                    pkg, nt = self.fetcher.get_morning_package(
                        self.symbol,
                        pc["category"],
                        settings=self.app_settings.data,
                    )
                    if nt:
                        cm.set_last_candle_time(nt)
                except Exception:
                    pkg = {
                        "pair": self.symbol,
                        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "message_type": "init",
                        "data": {},
                    }

                from ui.init_validator import run_init_with_sanity
                from ui import services as ui_services

                engine = getattr(ui_services, "backtest_engine", None)
                category = pc["category"] if pc else "linear"

                r = run_init_with_sanity(
                    ai_client=self.ai_client,
                    engine=engine,
                    symbol=self.symbol,
                    category=category,
                    init_call=lambda: self.ai_client.init_symbol(pkg, self.symbol),
                    fetcher=self.fetcher,
                )
                if r:
                    self.process_ai_response_fn(r, cm, self.symbol)
                    cm.mark_initialized()
                    cm.open_new_day()
                    cm.save()
                    self.schedule_refresh_fn(self.symbol)
                    self.add_notification_fn(f"{self.symbol} initialized!", "positive")
                else:
                    self.add_notification_fn(f"Failed to initialize {self.symbol}", "warning")

            except Exception as e:
                self.add_notification_fn(f"Initialization error {self.symbol}: {e}", "negative")
                logging.getLogger("nicegui").exception("Init symbol error")
            finally:
                self.clear_manual_action_fn(self.symbol)

        asyncio.create_task(asyncio.to_thread(worker))

    def clear_dialog(self):
        with ui.dialog() as dialog, ui.card():
            ui.label("Clear the data for this coin?")

            with ui.row().classes("justify-end"):
                ui.button("Cancel", on_click=dialog.close)
                ui.button(
                    "Clear",
                    on_click=lambda: self.do_clear(dialog),
                    color="negative",
                )

            dialog.open()

    def do_clear(self, dialog):
        dialog.close()
        cm = self.memory.get(self.symbol)
        self.clear_coin_data_fn(
            coin_memory=cm,
            symbol=self.symbol,
            stop_auto_mode=self.auto_mode_cls.stop,
            mode="all",
        )
        self.refresh()
        ui.notify("Data cleared (except the journal).", type="warning")

    def run_morning(self):
        cm = self.memory.get(self.symbol)
        if cm.is_day_closed():
            ui.notify("Day is closed, morning package not sent", type="warning")
            return

        self.set_manual_action_fn(self.symbol, "🌅 Morning package...")

        def worker():
            try:
                cm = self.memory.get(self.symbol)
                if not cm.is_initialized():
                    self.add_notification_fn("Coin is not initialized", "warning")
                    return

                pc = next((p for p in config.TRADING_PAIRS if p["symbol"] == self.symbol), None)
                if not pc:
                    return

                pkg, nt = self.fetcher.get_morning_package(
                    self.symbol,
                    pc["category"],
                    settings=self.app_settings.data,
                )
                if nt:
                    cm.set_last_candle_time(nt)

                r = self.ai_client.analyze(
                    pkg,
                    cm.get_rules(),
                    cm.get_active_trade(),
                    is_initial=False,
                    behavior_rules=cm.get_behavior_rules(),
                )

                if r:
                    self.process_ai_response_fn(
                        r,
                        cm,
                        self.symbol,
                        has_active_trade=bool(cm.get_active_trade()),
                        data_package=pkg,
                    )

                cm.update_state({"last_morning_update_utc": datetime.now(timezone.utc).isoformat()})
                self.schedule_refresh_fn(self.symbol)

            except Exception as e:
                self.add_notification_fn(f"Error: {e}", "negative")
            finally:
                self.clear_manual_action_fn(self.symbol)

        asyncio.create_task(asyncio.to_thread(worker))

    def run_intraday(self):
        cm = self.memory.get(self.symbol)
        if cm.is_day_closed():
            ui.notify("Day is closed, intraday package not sent", type="warning")
            return

        self.set_manual_action_fn(self.symbol, "⏱ Intraday package...")

        def worker():
            try:
                cm = self.memory.get(self.symbol)
                if not cm.is_initialized():
                    return

                pc = next((p for p in config.TRADING_PAIRS if p["symbol"] == self.symbol), None)
                if not pc:
                    return

                lt = cm.get_last_candle_time()
                pkg, nt = self.fetcher.get_intraday_package(
                    self.symbol,
                    pc["category"],
                    from_time=lt,
                    settings=self.app_settings.data,
                )
                if nt:
                    cm.set_last_candle_time(nt)

                r = self.ai_client.analyze(
                    pkg,
                    cm.get_rules(),
                    cm.get_active_trade(),
                    morning_context=cm.get_morning_context(),
                    intraday_context=cm.get_intraday_context(),
                    behavior_rules=cm.get_behavior_rules(),
                )

                if r:
                    self.process_ai_response_fn(
                        r,
                        cm,
                        self.symbol,
                        has_active_trade=bool(cm.get_active_trade()),
                        data_package=pkg,
                    )

                cm.update_state({"last_update_utc": datetime.now(timezone.utc).isoformat()})
                self.schedule_refresh_fn(self.symbol)

            except Exception as e:
                self.add_notification_fn(f"Error: {e}", "negative")
            finally:
                self.clear_manual_action_fn(self.symbol)

        asyncio.create_task(asyncio.to_thread(worker))

    def open_trade(self, direction, entry_price_str, stop_loss_str, take_profit_str):
        try:
            entry_price = float(entry_price_str)
            stop_loss = float(stop_loss_str)
            take_profit = float(take_profit_str)
        except ValueError:
            ui.notify("Enter valid prices (numbers)", type="warning")
            return

        if entry_price <= 0 or stop_loss <= 0 or take_profit <= 0:
            ui.notify("Prices must be positive", type="warning")
            return

        if direction == "long":
            if stop_loss >= entry_price:
                ui.notify("For long, SL must be below the entry", type="warning")
                return
            if take_profit <= entry_price:
                ui.notify("For long, TP must be above the entry", type="warning")
                return
        else:
            if stop_loss <= entry_price:
                ui.notify("For short, SL must be above the entry", type="warning")
                return
            if take_profit >= entry_price:
                ui.notify("For short, TP must be below the entry", type="warning")
                return

        cm = self.memory.get(self.symbol)
        trade = {
            "direction": direction,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": [take_profit],
            "entry_type": "manual",
        }
        cm.set_active_trade(trade)

        pc = next((p for p in config.TRADING_PAIRS if p["symbol"] == self.symbol), None)
        if pc:
            state = self.auto_mode_cls.get_state(self.symbol)
            if not state or not state.get("is_running"):
                self.auto_mode_cls.start(
                    self.symbol,
                    pc["category"],
                    update_callback=lambda: setattr(self, "needs_update", True),
                    initial_mode="position_only",
                )

        self.set_manual_action_fn(self.symbol, "📊 Post-entry analysis...")

        def worker():
            try:
                pc = next((p for p in config.TRADING_PAIRS if p["symbol"] == self.symbol), None)
                if pc:
                    lt = cm.get_last_candle_time()
                    pkg, nt = self.fetcher.get_intraday_package(
                        self.symbol,
                        pc["category"],
                        from_time=lt,
                        settings=self.app_settings.data,
                    )
                    if nt:
                        cm.set_last_candle_time(nt)

                    r = self.ai_client.analyze(
                        pkg,
                        cm.get_rules(),
                        trade,
                        morning_context=cm.get_morning_context(),
                        intraday_context=cm.get_intraday_context(),
                        behavior_rules=cm.get_behavior_rules(),
                    )
                    if r:
                        self.process_ai_response_fn(
                            r,
                            cm,
                            self.symbol,
                            has_active_trade=True,
                            data_package=pkg,
                        )

                    self.schedule_refresh_fn(self.symbol)
                    self.add_notification_fn("Position opened. Advice received.", "info")

            except Exception as e:
                self.add_notification_fn(f"Analysis error: {e}", "negative")
            finally:
                self.clear_manual_action_fn(self.symbol)

        asyncio.create_task(asyncio.to_thread(worker))
        self.refresh()

    def edit_active_trade_dialog(self):
        cm = self.memory.get(self.symbol)
        active = cm.get_active_trade()
        if not active:
            ui.notify("No active position", type="warning")
            return

        current_tp = active.get("take_profit", [0])[0] if active.get("take_profit") else 0

        with ui.dialog() as dialog, ui.card().style("min-width: 300px;"):
            ui.label("Edit position").classes("text-h6")
            new_sl = ui.number("Stop loss", value=active.get("stop_loss", 0), format="%.4f")
            new_tp = ui.number("Take profit", value=current_tp, format="%.4f")

            with ui.row().classes("justify-end"):
                ui.button("Cancel", on_click=dialog.close)

                def do_save():
                    active["stop_loss"] = new_sl.value
                    active["take_profit"] = [new_tp.value]
                    cm.set_active_trade(active)

                    pc = next((p for p in config.TRADING_PAIRS if p["symbol"] == self.symbol), None)
                    if pc:
                        self.set_manual_action_fn(self.symbol, "📊 Updated position analysis...")

                        def worker():
                            try:
                                lt = cm.get_last_candle_time()
                                pkg, nt = self.fetcher.get_intraday_package(
                                    self.symbol,
                                    pc["category"],
                                    from_time=lt,
                                    settings=self.app_settings.data,
                                )
                                if nt:
                                    cm.set_last_candle_time(nt)

                                r = self.ai_client.analyze(
                                    pkg,
                                    cm.get_rules(),
                                    active,
                                    morning_context=cm.get_morning_context(),
                                    behavior_rules=cm.get_behavior_rules(),
                                    keep_history=False,
                                )

                                if r:
                                    self.process_ai_response_fn(
                                        r,
                                        cm,
                                        self.symbol,
                                        has_active_trade=True,
                                        data_package=pkg,
                                    )

                                self.schedule_refresh_fn(self.symbol)

                            except Exception as e:
                                self.add_notification_fn(f"Analysis error: {e}", "negative")
                            finally:
                                self.clear_manual_action_fn(self.symbol)

                        asyncio.create_task(asyncio.to_thread(worker))

                    dialog.close()
                    self.refresh()

                ui.button("Save", on_click=do_save, color="primary")

            dialog.open()

    def close_active_trade(self):
        cm = self.memory.get(self.symbol)
        active = cm.get_active_trade()
        if not active:
            ui.notify("No open position", type="warning")
            return

        try:
            exit_price = float(self.close_price_input.value)
        except ValueError:
            ui.notify("Enter a valid exit price (number)", type="warning")
            return

        if exit_price <= 0:
            ui.notify("Price must be positive", type="warning")
            return

        try:
            position_size = float(self.close_position_size_input.value)
        except ValueError:
            ui.notify("Enter a valid position size in USDT (number)", type="warning")
            return
        if position_size <= 0:
            ui.notify("Position size must be a positive number", type="warning")
            return

        direction = active.get("direction", "long")
        entry_price = active.get("entry_price", 0)
        if entry_price == 0:
            ui.notify("Error: entry price is unknown", type="error")
            return

        if direction == "long":
            result_pct = (exit_price - entry_price) / entry_price * 100
        else:
            result_pct = (entry_price - exit_price) / entry_price * 100

        exit_type = self.close_reason.value
        trade_entry = {
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "stop_loss": active.get("stop_loss", 0),
            "take_profit": active.get("take_profit", [0])[0] if active.get("take_profit") else 0,
            "result_pct": round(result_pct, 2),
            "reason": f"Close: {exit_type}",
            "exit_type": exit_type,
            "entry_type": active.get("entry_type", "manual"),
            "signal_snapshot": active.get("signal_snapshot", {}),
            "market_snapshot": active.get("market_snapshot", {}),
            "errors": "",
            "position_value_usdt": round(position_size, 2),
            "pnl_usdt": round(position_size * (result_pct / 100), 2),
        }

        cm.add_trade_entry(trade_entry)
        cm.set_active_trade(None)
        cm.update_state({"last_position_update": {}})
        cm.save()

        self.schedule_refresh_fn(self.symbol)
        self.add_notification_fn("Position closed", "positive")
        self.set_manual_action_fn(self.symbol, "📊 Post-trade update...")

        def post_close_analysis():
            try:
                pc = next((p for p in config.TRADING_PAIRS if p["symbol"] == self.symbol), None)
                if pc:
                    lt = cm.get_last_candle_time()
                    pkg, nt = self.fetcher.get_intraday_package(
                        self.symbol,
                        pc["category"],
                        from_time=lt,
                        settings=self.app_settings.data,
                    )
                    if nt:
                        cm.set_last_candle_time(nt)

                    r = self.ai_client.analyze(
                        pkg,
                        cm.get_rules(),
                        None,
                        trade_closed=trade_entry,
                        morning_context=cm.get_morning_context(),
                        intraday_context=cm.get_intraday_context(),
                        error_analysis=(result_pct < 0),
                        behavior_rules=cm.get_behavior_rules(),
                        keep_history=False,
                    )

                    if r:
                        self.process_ai_response_fn(
                            r,
                            cm,
                            self.symbol,
                            has_active_trade=False,
                            data_package=pkg,
                        )

                    self.schedule_refresh_fn(self.symbol)

            except Exception as e:
                self.add_notification_fn(f"Analysis error: {e}", "negative")
            finally:
                self.clear_manual_action_fn(self.symbol)

        asyncio.create_task(asyncio.to_thread(post_close_analysis))
        self.refresh()

    def save_rules(self):
        cm = self.memory.get(self.symbol)
        new_rules = self.rules_text_area.value
        cm.update_rules(new_rules)

        trading_list, risk_list = self.parse_rules_from_text_fn(new_rules)
        cm.update_state(
            {
                "trading_rules_list": trading_list,
                "risk_management_rules_list": risk_list,
            }
        )

        ui.notify("Rules saved!", type="positive")
        self.refresh()