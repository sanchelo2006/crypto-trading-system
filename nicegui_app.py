#!/usr/bin/env python3
"""
NiceGUI интерфейс торговой системы.
Добавлены вкладки «Настройки», «Бэктест», «Автоторговля», блокировка _auto_threads,
облегчённый внутридневной пакет при наличии сетапа/позиции.
Реализована докачка пропущенных свечей при старте авторежима.
HelpPage удалена.
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
from ui.pages.research_lab_page import ResearchLabPage

from ui.app_context import app_context

# ДОБАВЛЕНО: импорт для защиты от конфликтов и автовосстановления
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

# На первом этапе эти типы намеренно не указываются строго:
# экземпляры создаются в ui.services.create_services().
app_settings = None
backtest_engine = None
backtest_optimizer = None

_services_initialized = False

_auto_threads: Dict[str, dict] = {}
_auto_threads_lock = threading.Lock()

# ============================================================================
# UI-события: уведомления, обновления страниц, чат, ручные статусы.
#
# Реальная потокобезопасная логика вынесена в ui.queues.UiEventQueue.
# Эти тонкие функции-адаптеры оставлены на первом этапе, чтобы не менять
# десятки существующих вызовов по всему nicegui_app.py.
# ============================================================================

ui_events = UiEventQueue()

def queue_chat_message(text: str, is_assistant: bool = True):
    """Добавляет сообщение в очередь UI-чата."""
    ui_events.queue_chat_message(text, is_assistant)


def queue_chat_loading(loading: bool, text: str = ""):
    """Добавляет событие изменения состояния загрузки чата."""
    ui_events.queue_chat_loading(loading, text)


def process_chat_queue(chat_page_instance):
    """Отрисовывает накопленные сообщения чата в UI-контексте."""
    ui_events.process_chat_queue(chat_page_instance)


def set_manual_action(symbol: str, msg: str):
    """Устанавливает ручной статус операции для монеты."""
    ui_events.set_manual_action(symbol, msg)
    update_status()


def clear_manual_action(symbol: str):
    """Очищает ручной статус операции для монеты."""
    ui_events.clear_manual_action(symbol)
    update_status()


def get_manual_actions() -> Dict[str, str]:
    """Возвращает текущие ручные статусы операций."""
    return ui_events.get_manual_actions()


def add_notification(message: str, type: str = "positive"):
    """Добавляет уведомление для последующего показа в UI-потоке."""
    ui_events.add_notification(message, type)


def schedule_refresh(symbol: str):
    """Ставит обновление страницы монеты в очередь UI."""
    ui_events.schedule_refresh(symbol)


def _notify_from_queue(message: str, notification_type: str):
    """Внутренний callback для UiEventQueue."""
    ui.notify(message, type=notification_type)


def _refresh_from_queue(symbol: str):
    """
    Обновляет страницу конкретной монеты и общие страницы.

    Вызывается UI-очередью только в UI-контексте.
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
    Применяет накопленные уведомления и обновления страниц.

    Вызывается таймером check_updates() в UI-контексте.
    """
    ui_events.process_ui_events(
        notify_callback=_notify_from_queue,
        refresh_callback=_refresh_from_queue,
    )


def init_services():
    """
    Создаёт сервисы приложения.

    На первом этапе рефакторинга глобальные переменные оставлены для
    обратной совместимости с существующими страницами и AutoModeState.
    Фактическая логика создания сервисов вынесена в ui.services.
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
            "Сервисы инициализированы через ui.services"
        )

        return True

    except Exception as e:
        _services_initialized = False

        logging.getLogger("nicegui").exception(
            "Ошибка инициализации сервисов: %s",
            e,
        )

        ui.notify(
            f"Ошибка инициализации: {e}",
            type="negative",
        )

        return False



def _restore_auto_traders():
    """
    Восстанавливает автотрейдеры после перезапуска приложения.
    Читает файл активных трейдеров и запускает их.
    Вызывается ОДИН раз при старте, после инициализации сервисов.
    """
    if not config.AUTO_TRADING_ENABLED:
        return

    saved = _load_active_traders()
    if not saved:
        logging.getLogger("nicegui").info("Нет сохранённых автотрейдеров для восстановления")
        return

    logger = logging.getLogger("nicegui")
    logger.info(f"Восстановление автотрейдеров: найдено {len(saved)} записей")

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
                logger.info(f"[{symbol}] Автотрейдер восстановлен после перезапуска")
            else:
                logger.warning(f"[{symbol}] Не удалось восстановить автотрейдер")
        except Exception:
            logger.exception(f"[{symbol}] Ошибка восстановления автотрейдера")

symbol_pages: Dict[str, "CoinPage"] = {}
app_context.symbol_pages = symbol_pages

summary_page = None
journal_page = None
chat_page = None
settings_page = None
backtest_page = None
auto_trading_page = None
statistics_page = None
research_lab_page = None
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
        label.set_text("Авторежим не запущен")
        return

    active = []
    for sym in current_memory.get_all_symbols():
        st = AutoModeState.get_state(sym)
        if st and st.get("is_running"):
            active.append(f"{sym}: {st.get('current_action','')}")

    label.set_text("\n".join(active) if active else "Авторежим не запущен")

def create_app():
    # ================================================================
    # МОБИЛЬНАЯ АДАПТАЦИЯ: глобальный CSS для маленьких экранов
    # ================================================================
    ui.add_head_html(r'''
    <style>
        /* --- Viewport и базовые шрифты --- */
        body {
            -webkit-text-size-adjust: 100%;
            font-size: 15px;
        }

        /* --- Мобильная версия (экраны уже 768px) --- */
        @media (max-width: 768px) {
            /* Сайдбар: превращаем в компактный блок сверху */
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

            /* Таблицы: горизонтальный скролл, не ломают ширину */
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

            /* Rows: перенос элементов на новую строку */
            .q-row, .nicegui-row {
                flex-wrap: wrap !important;
            }
            .q-row > *, .nicegui-row > * {
                min-width: 100% !important;
                width: 100% !important;
                margin-bottom: 4px !important;
            }

            /* Кнопки: крупнее для удобного тапа */
            .q-btn {
                min-height: 44px !important;
                font-size: 15px !important;
                padding: 8px 12px !important;
            }

            /* Инпуты: крупнее и на всю ширину */
            .q-field, .q-input, input, select, textarea {
                font-size: 16px !important;  /* 16px чтобы iOS не зумил */
                min-height: 44px !important;
            }

            /* Карточки: на всю ширину, меньше padding */
            .q-card {
                width: 100% !important;
                padding: 8px !important;
                margin-bottom: 8px !important;
            }

            /* Диалоги: на всю ширину экрана */
            .q-dialog__inner {
                padding: 8px !important;
            }
            .q-dialog .q-card {
                min-width: auto !important;
                max-width: 95vw !important;
                width: 95vw !important;
            }

            /* Шрифты заголовков */
            .text-h4 { font-size: 22px !important; }
            .text-h5 { font-size: 18px !important; }
            .text-h6 { font-size: 16px !important; }
            .text-subtitle1 { font-size: 15px !important; }
            .text-caption { font-size: 12px !important; }

            /* Expansions (сворачиваемые блоки) */
            .q-expansion-item {
                font-size: 14px !important;
            }

            /* Чат: сообщения на всю ширину */
            .nicegui-chat-message {
                max-width: 95% !important;
            }

            /* Табы: компактнее */
            .q-tabs__content {
                flex-wrap: wrap !important;
            }
            .q-tab {
                min-width: auto !important;
                padding: 4px 8px !important;
                font-size: 13px !important;
            }
        }

        /* --- Очень маленькие экраны (телефоны < 480px) --- */
        @media (max-width: 480px) {
            body { font-size: 14px; }
            .q-table { font-size: 12px !important; }
            .q-btn { font-size: 14px !important; min-height: 40px !important; }
            .text-h4 { font-size: 20px !important; }
        }

        /* --- Планшеты и десктоп (> 768px) — оставляем как есть --- */
        @media (min-width: 769px) {
            /* На десктопе ничего не меняем */
        }
    </style>
    ''')
    app_context.reset_pages()
    global main_container, summary_page, journal_page, chat_page
    global settings_page, backtest_page, auto_trading_page, statistics_page, research_lab_page, status_label, symbol_list_container  # ИЗМЕНЕНО

    with ui.row().classes('bg-dark text-white p-4 w-full items-center justify-between'):
        ui.label('📊 Crypto Trading System').classes('text-2xl font-bold')
        status_label = ui.label('Авторежим не запущен').classes('text-sm')

    with ui.row().classes('w-full items-stretch min-h-screen mobile-sidebar-wrapper'):
        with ui.column().classes('w-64 bg-grey-2 p-4 gap-2 mobile-sidebar').style('max-height: 100vh; overflow-y: auto;'):
            symbol_list_container = ui.column().classes('w-full gap-1')
            render_sidebar_buttons()

            ui.separator()
            with ui.row().classes('gap-2'):
                ui.button('➕ Добавить', on_click=add_symbol_dialog).classes('bg-primary text-white')
                ui.button('🗑 Удалить', on_click=remove_symbol).classes('bg-negative text-white')

            ui.separator()
            ui.button('⏹ Остановить все авторежимы', on_click=stop_all_auto).classes('bg-warning text-white w-full')
            ui.separator()
            ui.button('🌅 Утренний пакет (все)', on_click=run_morning_all).classes('bg-info text-white w-full')
            ui.button('⏱ Внутридневной (все)', on_click=run_intraday_all).classes('bg-info text-white w-full')
            ui.separator()
            ui.label(f'Модель: {config.DEEPSEEK_MODEL}').classes('text-caption')
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

                # ДОБАВЛЕНО: создание страницы автоторговли
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

                # Статистика P&L — доступна всегда (читает trade-логи).
                statistics_page = StatisticsPage(memory=memory)
                statistics_page.container.visible = False

                # Исследовательская лаборатория — портфельная симуляция
                research_lab_page = ResearchLabPage(backtest_engine=backtest_engine)
                research_lab_page.container.visible = False

                load_all_pages()

    app_context.bind_pages(
        summary_page=summary_page,
        journal_page=journal_page,
        chat_page=chat_page,
        settings_page=settings_page,
        backtest_page=backtest_page,
        auto_trading_page=auto_trading_page,
        research_lab_page=research_lab_page,
    )        

    app_context.bind_shell(
        status_label=status_label,
        symbol_list_container=symbol_list_container,
        main_container=main_container,
    )

    ui.timer(2.0, callback=check_updates)

    # Восстанавливаем автотрейдеры после перезапуска (с задержкой 10 сек,
    # чтобы сервисы успели полностью инициализироваться)
    ui.timer(10.0, _restore_auto_traders, once=True)



def render_sidebar_buttons():
    current_memory = app_context.memory or memory
    container = app_context.symbol_list_container or symbol_list_container

    if not current_memory or container is None:
        return

    container.clear()
    with container:
        ui.label('⚙️ Управление').classes('text-h6')
        
        buttons_config = [
            ('📊 Сводка', 'Сводка'),
            ('📋 Журнал сделок', 'Журнал'),
            ('💬 Чат', 'Чат'),
            ('⚙️ Настройки', 'Настройки'),
            ('📈 Бэктест', 'Бэктест'),
            ('📊 Статистика', 'Статистика'),
            ('🔬 Лаборатория', 'Лаборатория'),
        ]
        
        if config.AUTO_TRADING_ENABLED:
            buttons_config.append(('🤖 Автоторговля', 'Автоторговля'))
        
        for btn_text, page_name in buttons_config:
            btn = ui.button(btn_text, on_click=lambda p=page_name: switch_page(p))
            btn.classes('bg-transparent text-dark text-left')
            btn.style('width: 100%; text-align: left; color: black !important;')

        ui.separator()
        ui.label('💰 Монеты').classes('text-h6')

        for sym in current_memory.get_all_symbols():
            state = AutoModeState.get_state(sym)
            auto_mode_running = state and state.get("is_running")
            auto_trading = is_auto_trading_running(sym)  # ДОБАВЛЕНО
            
            # ИЗМЕНЕНО: иконка учитывает и авторежим, и автоторговлю
            if auto_trading:
                icon = "🤖"  # автоторговля
            elif auto_mode_running:
                icon = "🟢"  # авторежим
            else:
                icon = "⚪"
            
            cm = current_memory.get(sym)
            closed = "🔒" if cm.is_day_closed() else ""
            btn = ui.button(f'{icon}{closed} {sym}', on_click=lambda s=sym: switch_page(s))
            btn.classes('bg-transparent text-dark text-left')
            btn.style('width: 100%; text-align: left; color: black !important;')

def switch_page(page_name: str, open_archive_tab: bool = False):
    # ИЗМЕНЕНО: добавлена страница автоторговли
    for p in [summary_page, journal_page, chat_page, settings_page, backtest_page, auto_trading_page, statistics_page, research_lab_page]:
        if p:
            p.container.visible = False
    for page in symbol_pages.values():
        page.container.visible = False

    if page_name == 'Сводка' and summary_page:
        summary_page.container.visible = True
    elif page_name == 'Журнал' and journal_page:
        journal_page.container.visible = True
        journal_page.refresh()
    elif page_name == 'Чат' and chat_page:
        chat_page.container.visible = True
    elif page_name == 'Настройки' and settings_page:
        settings_page.container.visible = True
    elif page_name == 'Бэктест' and backtest_page:
        backtest_page.container.visible = True
        backtest_page.refresh()
    # ДОБАВЛЕНО: обработка страницы автоторговли
    elif page_name == 'Автоторговля' and auto_trading_page:
        auto_trading_page.container.visible = True
    elif page_name == 'Статистика' and statistics_page:
        statistics_page.container.visible = True
        statistics_page.refresh()
    elif page_name == 'Лаборатория' and research_lab_page:
        research_lab_page.container.visible = True
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
    add_notification('Все авторежимы остановлены', 'warning')
    update_status()

def add_symbol_dialog():
    with ui.dialog() as dialog, ui.card().style('min-width: 450px;'):
        ui.label('Добавить монету').classes('text-h6')
        symbol_input = ui.input('Символ', placeholder='XRPUSDT')
        with ui.row().classes('justify-end'):
            ui.button('Отмена', on_click=dialog.close)

            def do_add():
                symbol = symbol_input.value.strip().upper()
                if not symbol:
                    return
                if symbol in memory.get_all_symbols():
                    ui.notify('Уже есть', type='warning')
                    return
                try:
                    price = fetcher.get_current_price(symbol, category='linear')
                    if price is None or price <= 0:
                        ui.notify(f'{symbol} не найдена', type='negative')
                        return
                except Exception as e:
                    ui.notify(f'Ошибка: {e}', type='negative')
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

                ui.notify(f'{symbol} добавлена! Цена: {price:.4f}', type='positive')
                dialog.close()

            ui.button('Добавить', on_click=do_add, color='primary')
    dialog.open()

def remove_symbol():
    if not memory:
        return
    symbols = memory.get_all_symbols()
    if not symbols:
        return
    with ui.dialog() as dialog, ui.card().style('min-width: 300px;'):
        ui.label('Удалить монету').classes('text-h6')
        symbol_select = ui.select(options=symbols, label='Выберите')
        with ui.row().classes('justify-end'):
            ui.button('Отмена', on_click=dialog.close)

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
                    ui.notify(f'{sym} удалена', type='positive')
                    dialog.close()

            ui.button('Удалить', on_click=do_remove, color='negative')
    dialog.open()

def run_morning_all():
    if not memory:
        return
    for sym in memory.get_all_symbols():
        # ДОБАВЛЕНО: проверка автоторговли
        if is_auto_trading_running(sym):
            add_notification(
                f'[{sym}] Автоторговля включена. Утренний пакет недоступен.',
                'warning'
            )
            continue
        set_manual_action(sym, "🌅 Утро...")

    def worker():
        try:
            for pc in config.TRADING_PAIRS:
                sym, cat = pc["symbol"], pc["category"]
                
                # ДОБАВЛЕНО: проверка автоторговли
                if is_auto_trading_running(sym):
                    continue
                
                cm = memory.get(sym)
                if cm.is_day_closed():
                    add_notification(f'[{sym}] День закрыт, утренний пакет не отправлен', 'warning')
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
                    add_notification(f'Ошибка {sym}: {e}', 'negative')
                finally:
                    clear_manual_action(sym)
            add_notification('Утро (все) завершено', 'positive')
        finally:
            for sym in memory.get_all_symbols():
                clear_manual_action(sym)

    asyncio.create_task(asyncio.to_thread(worker))

def run_intraday_all():
    if not memory:
        return
    for sym in memory.get_all_symbols():
        # ДОБАВЛЕНО: проверка автоторговли
        if is_auto_trading_running(sym):
            add_notification(
                f'[{sym}] Автоторговля включена. Внутридневной пакет недоступен.',
                'warning'
            )
            continue
        set_manual_action(sym, "⏱ День...")

    def worker():
        try:
            for pc in config.TRADING_PAIRS:
                sym, cat = pc["symbol"], pc["category"]
                
                # ДОБАВЛЕНО: проверка автоторговли
                if is_auto_trading_running(sym):
                    continue
                
                cm = memory.get(sym)
                if cm.is_day_closed():
                    add_notification(f'[{sym}] День закрыт, внутридневной пакет не отправлен', 'warning')
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
                    add_notification(f'Ошибка {sym}: {e}', 'negative')
                finally:
                    clear_manual_action(sym)
            add_notification('День (все) завершён', 'positive')
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
    raise RuntimeError(f"Не удалось найти свободный порт в диапазоне {start_port}-{start_port+max_attempts-1}")

def main():
    from utils.logger_setup import setup_logging
    setup_logging()

    import atexit
    import faulthandler
    import signal
    import threading
    import traceback

    logger = logging.getLogger("nicegui")

    # --- 1. faulthandler: ловит жёсткие краши (segfault, OOM) ---
    # Дамп стека пишется в файл, если процесс убит не Python-исключением.
    try:
        from pathlib import Path
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        _crash_log = open(log_dir / "faulthandler.log", "a", encoding="utf-8")
        faulthandler.enable(file=_crash_log, all_threads=True)
    except Exception:
        pass  # не критично, продолжаем без него

    # --- 2. Главный excepthook: необработанные исключения верхнего уровня ---
    def _excepthook(exc_type, exc_value, exc_tb):
        logger.critical(
            "НЕОБРАБОТАННОЕ ИСКЛЮЧЕНИЕ (main thread):\n%s",
            "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
        )

    sys.excepthook = _excepthook

    # --- 3. threading.excepthook: исключения в фоновых потоках ---
    # КРИТИЧНО для автоторговли — трейдеры работают в отдельных потоках.
    def _thread_excepthook(args):
        logger.critical(
            "НЕОБРАБОТАННОЕ ИСКЛЮЧЕНИЕ (thread %s):\n%s",
            args.thread.name if args.thread else "?",
            "".join(traceback.format_exception(
                args.exc_type, args.exc_value, args.exc_traceback
            )),
        )

    if hasattr(threading, "excepthook"):
        threading.excepthook = _thread_excepthook

    # --- 4. Обработчики сигналов завершения ---
    def _signal_handler(signum, frame):
        logger.warning(
            "ПОЛУЧЕН СИГНАЛ %s — процесс завершается извне.", signum
        )
        sys.exit(0)

    for sig_name in ("SIGTERM", "SIGINT", "SIGBREAK"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _signal_handler)
        except (ValueError, OSError):
            pass  # не все сигналы ставятся вне главного потока

    # --- 5. atexit: штатное завершение ---
    def _onexit():
        logger.warning("Процесс завершается (atexit).")

    atexit.register(_onexit)

    # --- Остальная логика без изменений ---
    if not init_services():
        sys.exit(1)

    create_app()

    try:
        port = find_free_port(8081, 20)
        logger.info(f"Используется порт {port}")
        print(f"\n✅ Сервер запущен на http://localhost:{port}\n")
    except RuntimeError as e:
        logger.error(str(e))
        print(f"❌ {str(e)}")
        sys.exit(1)

    try:
        ui.run(title='Crypto Trading System', host='0.0.0.0',
               port=port, reload=False)
    except Exception:
        logger.critical("ui.run() завершился с исключением", exc_info=True)
        raise

if __name__ == "__main__":
    main()