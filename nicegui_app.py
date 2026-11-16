#!/usr/bin/env python3
"""
NiceGUI interface of the trading system.
Added tabs "Settings", "Backtest", "Auto trading", the _auto_threads lock,
and a lightweight intraday package when a setup/position exists.
Catch-up of missed candles on auto mode start is implemented.
HelpPage removed.
"""
import sys
import threading
import time
import json
import logging
import asyncio
import re
import socket
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple

from ui.ai_handlers import process_ai_response
from ui.auto_mode import AutoModeState

from ui.pages.settings_page import SettingsPage
from ui.pages.summary_page import SummaryPage
from ui.pages.journal_page import JournalPage
from ui.pages.chat_page import ChatPage
from ui.pages.backtest_page import BacktestPage
from ui.pages.coin_page import CoinPage
from ui.pages.auto_trading_page import AutoTradingPage
from ui.pages.statistics_page import StatisticsPage
from ui.pages.logs_page import LogsPage

from ui.app_context import app_context

# ADDED: import for conflict protection and auto-recovery
from trading.auto_trader import (
    is_auto_trading_running,
    _load_active_traders,
    AutoTraderConfig,
    start_auto_trading,
)

from trading.executor import BybitExecutor
import pandas as pd
from nicegui import ui

import config
from data_fetcher import BybitDataFetcher
from ai_client import DeepSeekClient
from memory_manager import MemoryManager, CoinMemory
from tool_registry import ToolExecutor, get_tool_definitions
from ui.queues import UiEventQueue
from ui.services import create_services
from ui.trade_helpers import (
    catch_up_historical_data,
    clear_all_journals,
    clear_coin_data,
    get_all_trades,
    parse_rules_from_text,
)

fetcher: Optional[BybitDataFetcher] = None
ai_client: Optional[DeepSeekClient] = None
memory: Optional[MemoryManager] = None

# At the first stage these types are intentionally not declared strictly:
# the instances are created in ui.services.create_services().
app_settings = None
backtest_engine = None
backtest_optimizer = None

_services_initialized = False

_auto_threads: Dict[str, dict] = {}
_auto_threads_lock = threading.Lock()

# ============================================================================
# UI events: notifications, page refreshes, chat, manual statuses.
#
# The actual thread-safe logic lives in ui.queues.UiEventQueue.
# These thin adapter functions are kept at the first stage to avoid changing
# dozens of existing calls across nicegui_app.py.
# ============================================================================

ui_events = UiEventQueue()

def queue_chat_message(text: str, is_assistant: bool = True):
    """Adds a message to the UI chat queue."""
    ui_events.queue_chat_message(text, is_assistant)


def queue_chat_loading(loading: bool, text: str = ""):
    """Adds a chat loading state change event."""
    ui_events.queue_chat_loading(loading, text)


def process_chat_queue(chat_page_instance):
    """Renders the accumulated chat messages in the UI context."""
    ui_events.process_chat_queue(chat_page_instance)


def set_manual_action(symbol: str, msg: str):
    """Sets the manual operation status for a coin."""
    ui_events.set_manual_action(symbol, msg)
    update_status()


def clear_manual_action(symbol: str):
    """Clears the manual operation status for a coin."""
    ui_events.clear_manual_action(symbol)
    update_status()


def get_manual_actions() -> Dict[str, str]:
    """Returns the current manual operation statuses."""
    return ui_events.get_manual_actions()


def add_notification(message: str, type: str = "positive"):
    """Adds a notification to be shown later in the UI thread."""
    ui_events.add_notification(message, type)


def schedule_refresh(symbol: str):
    """Queues a refresh of the coin's page in the UI."""
    ui_events.schedule_refresh(symbol)


def _notify_from_queue(message: str, notification_type: str):
    """Internal callback for UiEventQueue."""
    ui.notify(message, type=notification_type)


def _refresh_from_queue(symbol: str):
    """
    Refreshes a specific coin's page and the shared pages.

    Called by the UI queue only in the UI context.
    """
    if symbol in symbol_pages:
        symbol_pages[symbol].refresh()

    if summary_page:
        summary_page.refresh()

    if journal_page:
        journal_page.refresh()

    if backtest_page:
        backtest_page.refresh()

    if statistics_page:
        statistics_page.refresh()


def process_queues():
    """
    Applies the accumulated notifications and page refreshes.

    Called by the check_updates() timer in the UI context.
    """
    ui_events.process_ui_events(
        notify_callback=_notify_from_queue,
        refresh_callback=_refresh_from_queue,
    )


def init_services():
    """
    Creates the application services.

    At the first stage of the refactoring the global variables are kept for
    backward compatibility with the existing pages and AutoModeState.
    The actual service creation logic lives in ui.services.
    """
    global _services_initialized
    global fetcher
    global ai_client
    global memory
    global app_settings
    global backtest_engine
    global backtest_optimizer

    if _services_initialized:
        return True

    try:
        services = create_services()

        fetcher = services.fetcher
        ai_client = services.ai_client
        memory = services.memory
        app_settings = services.app_settings
        backtest_engine = services.backtest_engine
        backtest_optimizer = services.backtest_optimizer

        app_context.bind_services(
            fetcher=fetcher,
            ai_client=ai_client,
            memory=memory,
            app_settings=app_settings,
            backtest_engine=backtest_engine,
            backtest_optimizer=backtest_optimizer,
        )

        AutoModeState.configure(
            memory=memory,
            fetcher=fetcher,
            ai_client=ai_client,
            app_settings=app_settings,
            process_ai_response=process_ai_response,
            catch_up_historical_data=catch_up_historical_data,
            add_notification=add_notification,
            schedule_refresh=schedule_refresh,
        )

        _services_initialized = True

        logging.getLogger("nicegui").info(
            "Services initialized via ui.services"
        )

        return True

    except Exception as e:
        _services_initialized = False

        logging.getLogger("nicegui").exception(
            "Service initialization error: %s",
            e,
        )

        ui.notify(
            f"Initialization error: {e}",
            type="negative",
        )

        return False



def _restore_auto_traders():
    """
    Restores auto traders after an application restart.
    Reads the active traders file and starts them.
    Called ONCE at startup, after the services are initialized.
    """
    if not config.AUTO_TRADING_ENABLED:
        return

    saved = _load_active_traders()
    if not saved:
        logging.getLogger("nicegui").info("No saved auto traders to restore")
        return

    logger = logging.getLogger("nicegui")
    logger.info(f"Restoring auto traders: {len(saved)} records found")

    for symbol, cfg_data in saved.items():
        try:
            cfg = AutoTraderConfig(
                symbol=cfg_data["symbol"],
                category=cfg_data.get("category", "linear"),
                capital_share_pct=cfg_data.get("capital_share_pct", 30.0),
                position_size_pct=cfg_data.get("position_size_pct", 50.0),
                leverage=cfg_data.get("leverage", 10),
                daily_loss_limit_pct=cfg_data.get("daily_loss_limit_pct", 5.0),
                monthly_loss_limit_pct=cfg_data.get("monthly_loss_limit_pct", 15.0),
                optimization_days_back=cfg_data.get("optimization_days_back", 90),
                optimization_iterations=cfg_data.get("optimization_iterations", 10),
                walk_forward_folds=cfg_data.get("walk_forward_folds", 3),
                initial_capital=cfg_data.get("initial_capital", 1000.0),
                commission_pct=cfg_data.get("commission_pct", 0.05),
            )

            started = start_auto_trading(
                cfg,
                executor=BybitExecutor(
                    testnet=config.BYBIT_TESTNET,
                    default_leverage=cfg.leverage,
                ),
                fetcher=fetcher,
                memory=memory,
                backtest_engine=backtest_engine,
                backtest_optimizer=backtest_optimizer,
                ai_client=ai_client,
                notify_fn=add_notification,
                total_capital_usdt=cfg.initial_capital,
            )

            if started:
                logger.info(f"[{symbol}] Auto trader restored after restart")
            else:
                logger.warning(f"[{symbol}] Failed to restore the auto trader")
        except Exception:
            logger.exception(f"[{symbol}] Auto trader restore error")

symbol_pages: Dict[str, "CoinPage"] = {}
app_context.symbol_pages = symbol_pages

summary_page = None
journal_page = None
chat_page = None
settings_page = None
backtest_page = None
auto_trading_page = None
statistics_page = None
logs_page = None
status_label = None
symbol_list_container = None
main_container = None

def check_updates():
    process_queues()
    if chat_page:
        process_chat_queue(chat_page)
    if not memory:
        return
    for sym in memory.get_all_symbols():
        state = AutoModeState.get_state(sym)
        if state and state.get("needs_update"):
            state["needs_update"] = False
            if sym in symbol_pages:
                symbol_pages[sym].refresh()
    update_status()

def update_status():
    label = app_context.status_label or status_label
    if not label:
        return

    manual = get_manual_actions()
    if manual:
        lines = [f"{sym}: {msg}" for sym, msg in manual.items()]
        label.set_text("\n".join(lines))
        return

    current_memory = app_context.memory or memory
    if not current_memory:
        label.set_text("Auto mode is not running")
        return

    active = []
    for sym in current_memory.get_all_symbols():
        st = AutoModeState.get_state(sym)
        if st and st.get("is_running"):
            active.append(f"{sym}: {st.get('current_action','')}")

    label.set_text("\n".join(active) if active else "Auto mode is not running")

def create_app():
    # ================================================================
    # MOBILE ADAPTATION: global CSS for small screens
    # ================================================================
    ui.add_head_html(r'''
    <style>
        /* --- Viewport and base fonts --- */
        body {
            -webkit-text-size-adjust: 100%;
            font-size: 15px;
        }

        /* --- Mobile version (screens narrower than 768px) --- */
        @media (max-width: 768px) {
            /* Sidebar: turn it into a compact block on top */
            .mobile-sidebar-wrapper {
                flex-direction: column !important;
            }
            .mobile-sidebar {
                width: 100% !important;
                max-height: none !important;
                overflow-y: visible !important;
                padding: 8px !important;
            }
            .mobile-sidebar .q-btn {
                width: 100% !important;
                margin-bottom: 2px !important;
            }
            .mobile-main {
                padding: 8px !important;
            }

            /* Tables: horizontal scroll, do not break the width */
            .q-table {
                display: block !important;
                overflow-x: auto !important;
                white-space: nowrap !important;
                max-width: 100vw !important;
                font-size: 13px !important;
            }
            .q-table th, .q-table td {
                padding: 6px 8px !important;
            }

            /* Rows: wrap items onto a new line */
            .q-row, .nicegui-row {
                flex-wrap: wrap !important;
            }
            .q-row > *, .nicegui-row > * {
                min-width: 100% !important;
                width: 100% !important;
                margin-bottom: 4px !important;
            }

            /* Buttons: larger for comfortable tapping */
            .q-btn {
                min-height: 44px !important;
                font-size: 15px !important;
                padding: 8px 12px !important;
            }

            /* Inputs: larger and full width */
            .q-field, .q-input, input, select, textarea {
                font-size: 16px !important;  /* 16px so iOS does not zoom */
                min-height: 44px !important;
            }

            /* Cards: full width, less padding */
            .q-card {
                width: 100% !important;
                padding: 8px !important;
                margin-bottom: 8px !important;
            }

            /* Dialogs: full screen width */
            .q-dialog__inner {
                padding: 8px !important;
            }
            .q-dialog .q-card {
                min-width: auto !important;
                max-width: 95vw !important;
                width: 95vw !important;
            }

            /* Heading fonts */
            .text-h4 { font-size: 22px !important; }
            .text-h5 { font-size: 18px !important; }
            .text-h6 { font-size: 16px !important; }
            .text-subtitle1 { font-size: 15px !important; }
            .text-caption { font-size: 12px !important; }

            /* Expansions (collapsible blocks) */
            .q-expansion-item {
                font-size: 14px !important;
            }

            /* Chat: full-width messages */
            .nicegui-chat-message {
                max-width: 95% !important;
            }

            /* Tabs: more compact */
            .q-tabs__content {
                flex-wrap: wrap !important;
            }
            .q-tab {
                min-width: auto !important;
                padding: 4px 8px !important;
                font-size: 13px !important;
            }
        }

        /* --- Very small screens (phones < 480px) --- */
        @media (max-width: 480px) {
            body { font-size: 14px; }
            .q-table { font-size: 12px !important; }
            .q-btn { font-size: 14px !important; min-height: 40px !important; }
            .text-h4 { font-size: 20px !important; }
        }

        /* --- Tablets and desktop (> 768px) — left as is --- */
        @media (min-width: 769px) {
            /* Nothing is changed on desktop */
        }
    </style>
    ''')
    app_context.reset_pages()
    global main_container, summary_page, journal_page, chat_page
    global settings_page, backtest_page, auto_trading_page, statistics_page, logs_page, status_label, symbol_list_container  # CHANGED

    with ui.row().classes('bg-dark text-white p-4 w-full items-center justify-between'):
        ui.label('📊 Crypto Trading System').classes('text-2xl font-bold')
        status_label = ui.label('Auto mode is not running').classes('text-sm')

    with ui.row().classes('w-full items-stretch min-h-screen mobile-sidebar-wrapper'):
        with ui.column().classes('w-64 bg-grey-2 p-4 gap-2 mobile-sidebar').style('max-height: 100vh; overflow-y: auto;'):
            symbol_list_container = ui.column().classes('w-full gap-1')
            render_sidebar_buttons()

            ui.separator()
            with ui.row().classes('gap-2'):
                ui.button('➕ Add', on_click=add_symbol_dialog).classes('bg-primary text-white')
                ui.button('🗑 Remove', on_click=remove_symbol).classes('bg-negative text-white')

            ui.separator()
            ui.button('⏹ Stop all auto modes', on_click=stop_all_auto).classes('bg-warning text-white w-full')
            ui.separator()
            ui.button('🌅 Morning package (all)', on_click=run_morning_all).classes('bg-info text-white w-full')
            ui.button('⏱ Intraday (all)', on_click=run_intraday_all).classes('bg-info text-white w-full')
            ui.separator()
            ui.label(f'Model: {config.DEEPSEEK_MODEL}').classes('text-caption')
            ui.label(f'Testnet: {config.BYBIT_TESTNET}').classes('text-caption')

        with ui.column().classes('flex-1 p-4 overflow-y-auto mobile-main').style('width: 100%;'):
            main_container = ui.column().classes('w-full')
            with main_container:
                summary_page = SummaryPage(
                    memory=memory,
                    auto_mode_get_state=AutoModeState.get_state,
                )
                summary_page.container.visible = True

                journal_page = JournalPage(
                    memory=memory,
                    get_all_trades_fn=get_all_trades,
                    clear_all_journals_fn=clear_all_journals,
                )
                journal_page.container.visible = False

                chat_page = ChatPage(
                    memory=memory,
                    fetcher=fetcher,
                    ai_client=ai_client,
                    app_settings=app_settings,
                    queue_chat_message_fn=queue_chat_message,
                    queue_chat_loading_fn=queue_chat_loading,
                )
                chat_page.container.visible = False

                settings_page = SettingsPage(
                    app_settings=app_settings,
                )
                settings_page.container.visible = False

                backtest_page = BacktestPage(
                    memory=memory,
                    backtest_engine=backtest_engine,
                    backtest_optimizer=backtest_optimizer,
                )
                backtest_page.container.visible = False

                # ADDED: auto trading page creation
                if config.AUTO_TRADING_ENABLED:
                    auto_trading_page = AutoTradingPage(
                        memory=memory,
                        fetcher=fetcher,
                        backtest_engine=backtest_engine,
                        backtest_optimizer=backtest_optimizer,
                        ai_client=ai_client,
                        app_settings=app_settings,
                        notify_fn=add_notification,
                    )
                    auto_trading_page.container.visible = False

                # P&L statistics — always available (reads the trade logs).
                statistics_page = StatisticsPage(memory=memory)
                statistics_page.container.visible = False

                logs_page = LogsPage()
                logs_page.container.visible = False

                load_all_pages()

    app_context.bind_pages(
        summary_page=summary_page,
        journal_page=journal_page,
        chat_page=chat_page,
        settings_page=settings_page,
        logs_page=logs_page,
        backtest_page=backtest_page,
        auto_trading_page=auto_trading_page,
    )

    app_context.bind_shell(
        status_label=status_label,
        symbol_list_container=symbol_list_container,
        main_container=main_container,
    )

    ui.timer(2.0, callback=check_updates)

    # Restore auto traders after a restart (with a 10 sec delay
    # so the services have time to fully initialize)
    ui.timer(10.0, _restore_auto_traders, once=True)



def render_sidebar_buttons():
    current_memory = app_context.memory or memory
    container = app_context.symbol_list_container or symbol_list_container

    if not current_memory or container is None:
        return

    container.clear()
    with container:
        ui.label('⚙️ Controls').classes('text-h6')

        buttons_config = [
            ('📊 Summary', 'Summary'),
            ('📋 Trade journal', 'Journal'),
            ('💬 Chat', 'Chat'),
            ('⚙️ Settings', 'Settings'),
            ('📈 Backtest', 'Backtest'),
            ('📊 Statistics', 'Statistics'),
            ('📁 Logs', 'Logs'),
        ]

        if config.AUTO_TRADING_ENABLED:
            buttons_config.append(('🤖 Auto trading', 'Auto trading'))

        for btn_text, page_name in buttons_config:
            btn = ui.button(btn_text, on_click=lambda p=page_name: switch_page(p))
            btn.classes('bg-transparent text-dark text-left')
            btn.style('width: 100%; text-align: left; color: black !important;')

        ui.separator()
        ui.label('💰 Coins').classes('text-h6')

        for sym in current_memory.get_all_symbols():
            state = AutoModeState.get_state(sym)
            auto_mode_running = state and state.get("is_running")
            auto_trading = is_auto_trading_running(sym)  # ADDED

            # CHANGED: the icon accounts for both auto mode and auto trading
            if auto_trading:
                icon = "🤖"  # auto trading
            elif auto_mode_running:
                icon = "🟢"  # auto mode
            else:
                icon = "⚪"

            cm = current_memory.get(sym)
            closed = "🔒" if cm.is_day_closed() else ""
            btn = ui.button(f'{icon}{closed} {sym}', on_click=lambda s=sym: switch_page(s))
            btn.classes('bg-transparent text-dark text-left')
            btn.style('width: 100%; text-align: left; color: black !important;')

def switch_page(page_name: str, open_archive_tab: bool = False):
    # CHANGED: auto trading page added
    for p in [summary_page, journal_page, chat_page, settings_page, backtest_page, auto_trading_page, statistics_page, logs_page]:
        if p:
            p.container.visible = False
    for page in symbol_pages.values():
        page.container.visible = False

    if page_name == 'Summary' and summary_page:
        summary_page.container.visible = True
    elif page_name == 'Journal' and journal_page:
        journal_page.container.visible = True
        journal_page.refresh()
    elif page_name == 'Chat' and chat_page:
        chat_page.container.visible = True
    elif page_name == 'Settings' and settings_page:
        settings_page.container.visible = True
    elif page_name == 'Backtest' and backtest_page:
        backtest_page.container.visible = True
        backtest_page.refresh()
    # ADDED: auto trading page handling
    elif page_name == 'Auto trading' and auto_trading_page:
        auto_trading_page.container.visible = True
    elif page_name == 'Statistics' and statistics_page:
        statistics_page.container.visible = True
        statistics_page.refresh()
    elif page_name == 'Logs' and logs_page:
        logs_page.container.visible = True
    elif page_name in symbol_pages:
        page = symbol_pages[page_name]
        page.container.visible = True
        if open_archive_tab:
            page.switch_to_archive_tab()
        page.refresh()
    else:
        if summary_page:
            summary_page.container.visible = True

def load_all_pages():
    current_memory = app_context.memory or memory
    container = app_context.main_container or main_container

    if not current_memory or container is None:
        return

    with container:
        for sym in current_memory.get_all_symbols():
            if sym not in symbol_pages:
                page = CoinPage(
                    symbol=sym,
                    memory=current_memory,
                    fetcher=fetcher,
                    ai_client=ai_client,
                    app_settings=app_settings,
                    auto_mode_cls=AutoModeState,
                    process_ai_response_fn=process_ai_response,
                    clear_coin_data_fn=clear_coin_data,
                    parse_rules_from_text_fn=parse_rules_from_text,
                    add_notification_fn=add_notification,
                    schedule_refresh_fn=schedule_refresh,
                    set_manual_action_fn=set_manual_action,
                    clear_manual_action_fn=clear_manual_action,
                    render_sidebar_buttons_fn=render_sidebar_buttons,
                    summary_page_getter=lambda: app_context.summary_page or summary_page,
                    journal_page_getter=lambda: app_context.journal_page or journal_page,
                )
                symbol_pages[sym] = page
                page.container.visible = False

def stop_all_auto():
    AutoModeState.stop_all()
    add_notification('All auto modes stopped', 'warning')
    update_status()

def add_symbol_dialog():
    with ui.dialog() as dialog, ui.card().style('min-width: 450px;'):
        ui.label('Add coin').classes('text-h6')
        symbol_input = ui.input('Symbol', placeholder='XRPUSDT')
        with ui.row().classes('justify-end'):
            ui.button('Cancel', on_click=dialog.close)

            def do_add():
                symbol = symbol_input.value.strip().upper()
                if not symbol:
                    return
                if symbol in memory.get_all_symbols():
                    ui.notify('Already exists', type='warning')
                    return
                try:
                    price = fetcher.get_current_price(symbol, category='linear')
                    if price is None or price <= 0:
                        ui.notify(f'{symbol} not found', type='negative')
                        return
                except Exception as e:
                    ui.notify(f'Error: {e}', type='negative')
                    return
                memory.add_symbol(symbol)
                config.TRADING_PAIRS.append({"symbol": symbol, "category": "linear"})
                with main_container:
                    page = CoinPage(
                        symbol=symbol,
                        memory=memory,
                        fetcher=fetcher,
                        ai_client=ai_client,
                        app_settings=app_settings,
                        auto_mode_cls=AutoModeState,
                        process_ai_response_fn=process_ai_response,
                        clear_coin_data_fn=clear_coin_data,
                        parse_rules_from_text_fn=parse_rules_from_text,
                        add_notification_fn=add_notification,
                        schedule_refresh_fn=schedule_refresh,
                        set_manual_action_fn=set_manual_action,
                        clear_manual_action_fn=clear_manual_action,
                        render_sidebar_buttons_fn=render_sidebar_buttons,
                        summary_page_getter=lambda: summary_page,
                        journal_page_getter=lambda: journal_page,
                    )
                    symbol_pages[symbol] = page
                    page.container.visible = False
                render_sidebar_buttons()

                if backtest_page:
                    backtest_page.refresh()

                ui.notify(f'{symbol} added! Price: {price:.4f}', type='positive')
                dialog.close()

            ui.button('Add', on_click=do_add, color='primary')
    dialog.open()

def remove_symbol():
    if not memory:
        return
    symbols = memory.get_all_symbols()
    if not symbols:
        return
    with ui.dialog() as dialog, ui.card().style('min-width: 300px;'):
        ui.label('Remove coin').classes('text-h6')
        symbol_select = ui.select(options=symbols, label='Select')
        with ui.row().classes('justify-end'):
            ui.button('Cancel', on_click=dialog.close)

            def do_remove():
                sym = symbol_select.value
                if sym:
                    AutoModeState.stop(sym)
                    memory.remove_symbol(sym)
                    if sym in symbol_pages:
                        page = symbol_pages.pop(sym)
                        page.container.delete()
                    render_sidebar_buttons()
                    if backtest_page:
                        backtest_page.refresh()
                    ui.notify(f'{sym} removed', type='positive')
                    dialog.close()

            ui.button('Remove', on_click=do_remove, color='negative')
    dialog.open()

def run_morning_all():
    if not memory:
        return
    for sym in memory.get_all_symbols():
        # ADDED: auto trading check
        if is_auto_trading_running(sym):
            add_notification(
                f'[{sym}] Auto trading is enabled. The morning package is unavailable.',
                'warning'
            )
            continue
        set_manual_action(sym, "🌅 Morning...")

    def worker():
        try:
            for pc in config.TRADING_PAIRS:
                sym, cat = pc["symbol"], pc["category"]

                # ADDED: auto trading check
                if is_auto_trading_running(sym):
                    continue

                cm = memory.get(sym)
                if cm.is_day_closed():
                    add_notification(f'[{sym}] Day is closed, morning package not sent', 'warning')
                    continue
                if not cm.is_initialized():
                    continue
                try:
                    pkg, nt = fetcher.get_morning_package(sym, cat, settings=app_settings.data)
                    if nt:
                        cm.set_last_candle_time(nt)
                    r = ai_client.analyze(
                        pkg, cm.get_rules(), cm.get_active_trade(),
                        is_initial=cm.is_day_closed(),
                        behavior_rules=cm.get_behavior_rules()
                    )
                    if r:
                        process_ai_response(r, cm, sym, has_active_trade=bool(cm.get_active_trade()), data_package=pkg)
                        cm.update_state({"last_morning_update_utc": datetime.now(timezone.utc).isoformat()})
                        schedule_refresh(sym)
                except Exception as e:
                    add_notification(f'Error {sym}: {e}', 'negative')
                finally:
                    clear_manual_action(sym)
            add_notification('Morning (all) completed', 'positive')
        finally:
            for sym in memory.get_all_symbols():
                clear_manual_action(sym)

    asyncio.create_task(asyncio.to_thread(worker))

def run_intraday_all():
    if not memory:
        return
    for sym in memory.get_all_symbols():
        # ADDED: auto trading check
        if is_auto_trading_running(sym):
            add_notification(
                f'[{sym}] Auto trading is enabled. The intraday package is unavailable.',
                'warning'
            )
            continue
        set_manual_action(sym, "⏱ Intraday...")

    def worker():
        try:
            for pc in config.TRADING_PAIRS:
                sym, cat = pc["symbol"], pc["category"]

                # ADDED: auto trading check
                if is_auto_trading_running(sym):
                    continue

                cm = memory.get(sym)
                if cm.is_day_closed():
                    add_notification(f'[{sym}] Day is closed, intraday package not sent', 'warning')
                    continue
                if not cm.is_initialized():
                    continue
                try:
                    lt = cm.get_last_candle_time()
                    pkg, nt = fetcher.get_intraday_package(
                        sym, cat, from_time=lt, settings=app_settings.data
                    )
                    if nt:
                        cm.set_last_candle_time(nt)
                    r = ai_client.analyze(
                        pkg, cm.get_rules(), cm.get_active_trade(),
                        morning_context=cm.get_morning_context(),
                        intraday_context=cm.get_intraday_context(),
                        behavior_rules=cm.get_behavior_rules()
                    )
                    if r:
                        process_ai_response(r, cm, sym, has_active_trade=bool(cm.get_active_trade()), data_package=pkg)
                        cm.update_state({"last_update_utc": datetime.now(timezone.utc).isoformat()})
                        schedule_refresh(sym)
                except Exception as e:
                    add_notification(f'Error {sym}: {e}', 'negative')
                finally:
                    clear_manual_action(sym)
            add_notification('Intraday (all) completed', 'positive')
        finally:
            for sym in memory.get_all_symbols():
                clear_manual_action(sym)

    asyncio.create_task(asyncio.to_thread(worker))

def find_free_port(start_port: int = 8081, max_attempts: int = 10) -> int:
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"Failed to find a free port in the range {start_port}-{start_port+max_attempts-1}")

def main():
    from utils.logger_setup import setup_logging
    setup_logging()

    import atexit
    import faulthandler
    import signal
    import threading
    import traceback

    logger = logging.getLogger("nicegui")

    # --- 1. faulthandler: catches hard crashes (segfault, OOM) ---
    # The stack dump is written to a file if the process is killed not by a Python exception.
    try:
        from pathlib import Path
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        _crash_log = open(log_dir / "faulthandler.log", "a", encoding="utf-8")
        faulthandler.enable(file=_crash_log, all_threads=True)
    except Exception:
        pass  # not critical, continue without it

    # --- 2. Main excepthook: unhandled top-level exceptions ---
    def _excepthook(exc_type, exc_value, exc_tb):
        logger.critical(
            "UNHANDLED EXCEPTION (main thread):\n%s",
            "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
        )

    sys.excepthook = _excepthook

    # --- 3. threading.excepthook: exceptions in background threads ---
    # CRITICAL for auto trading — traders run in separate threads.
    def _thread_excepthook(args):
        logger.critical(
            "UNHANDLED EXCEPTION (thread %s):\n%s",
            args.thread.name if args.thread else "?",
            "".join(traceback.format_exception(
                args.exc_type, args.exc_value, args.exc_traceback
            )),
        )

    if hasattr(threading, "excepthook"):
        threading.excepthook = _thread_excepthook

    # --- 4. Termination signal handlers ---
    def _signal_handler(signum, frame):
        logger.warning(
            "RECEIVED SIGNAL %s — the process is being terminated externally.", signum
        )
        sys.exit(0)

    for sig_name in ("SIGTERM", "SIGINT", "SIGBREAK"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _signal_handler)
        except (ValueError, OSError):
            pass  # not all signals can be set outside the main thread

    # --- 5. atexit: normal shutdown ---
    def _onexit():
        logger.warning("The process is shutting down (atexit).")

    atexit.register(_onexit)

    # --- The rest of the logic is unchanged ---
    if not init_services():
        sys.exit(1)

    create_app()

    try:
        port = find_free_port(8081, 20)
        logger.info(f"Using port {port}")
        print(f"\n✅ Server started at http://localhost:{port}\n")
    except RuntimeError as e:
        logger.error(str(e))
        print(f"❌ {str(e)}")
        sys.exit(1)

    try:
        ui.run(title='Crypto Trading System', host='0.0.0.0',
               port=port, reload=False)
    except Exception:
        logger.critical("ui.run() finished with an exception", exc_info=True)
        raise

if __name__ == "__main__":
    main()