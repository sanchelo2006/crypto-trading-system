"""
Страница автоматической торговли.

Позволяет пользователю:
- Выбрать монеты для автоторговли
- Настроить доли капитала, размер позиции, плечо
- Установить лимиты убытков (дневной / месячный)
- Указать период для оптимизации при месячном убытке
- Запускать / останавливать торговлю
- Закрывать позиции (по монете или все)
- Видеть текущее состояние торговли и логи
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
    UI-страница автоматической торговли.
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

        # Executor создаётся один раз
        self.executor = BybitExecutor(
            testnet=config.BYBIT_TESTNET,
            default_leverage=10,
        )

        self.container = ui.column().classes("w-full")
        self._coin_configs: dict = {}  # symbol -> dict of UI inputs

        with self.container:
            self.build_ui()

        self.container.visible = False

        # Таймер обновления состояния
        ui.timer(3.0, self._refresh_state)

        # Таймер обновления баланса (раз в 15 минут)
        ui.timer(900.0, self._load_balance_async)
        
        # Загружаем баланс сразу при открытии страницы (с задержкой 0.5 сек)
        ui.timer(0.5, self._load_balance_async, once=True)

    

    def build_ui(self):
        # ================================================================
        # Заголовок
        # ================================================================
        ui.label("🤖 Автоматическая торговля").classes("text-h5 q-mb-sm")
        ui.label(
            "Торговля на фьючерсном рынке Bybit (linear perpetual). "
            "Стратегия берётся из памяти монеты (strategy_spec). "
            "Максимум 1 одновременно открытая позиция на монету."
        ).classes("text-caption text-grey q-mb-md")

        # ================================================================
        # Предупреждение о конфликте с авторежимом
        # ================================================================
        self.conflict_warning = ui.label("").classes(
            "text-warning text-bold q-mb-md"
        ).style("display: none;")

        # ================================================================
        # Общий капитал
        # ================================================================
        with ui.card().classes("w-full q-mb-md"):
            ui.label("💰 Общий капитал").classes("text-subtitle1")

            with ui.row().classes("gap-4 items-center"):
                self.total_capital_input = ui.number(
                    "Капитал для торговли (USDT)",
                    value=1000.0,
                    min=1,
                    format="%.2f",
                ).classes("w-64")
                
                # Метка с балансом биржи (обновляется каждые 30 сек)
                self.balance_label = ui.label("🔄 Загрузка баланса...").classes(
                    "text-caption"
                )

                # Кнопка ручного обновления
                ui.button(
                    "🔄",
                    on_click=lambda: asyncio.create_task(
                        asyncio.to_thread(self._load_balance_from_exchange)
                    ),
                ).props("flat dense").classes("q-ml-xs").style(
                    "min-width: 32px; padding: 4px;"
                ).tooltip("Обновить баланс с биржи")

            with ui.row().classes("q-mt-xs"):
                self.testnet_label = ui.label("").classes("text-caption")
                self._update_testnet_label()

        # ================================================================
        # Выбор монет и настройки
        # ================================================================
        with ui.card().classes("w-full q-mb-md"):
            ui.label("📋 Монеты для торговли").classes("text-subtitle1")

            symbols = list(self.memory.get_all_symbols())

            for symbol in symbols:
                with ui.expansion(
                    f"⚙️ {symbol}",
                    icon="settings",
                ).classes("w-full q-mb-sm"):
                    with ui.column().classes("gap-2 w-full"):
                        with ui.row().classes("gap-4"):
                            capital_share = ui.number(
                                "Доля общего капитала (%)",
                                value=30.0,
                                min=5,
                                max=100,
                                format="%.1f",
                            ).classes("w-48")

                            position_size = ui.number(
                                "Размер позиции (% капитала монеты)",
                                value=10.0,
                                min=10,
                                max=100,
                                format="%.1f",
                            ).classes("w-56")

                            leverage = ui.number(
                                "Плечо (1–100)",
                                value=1,
                                min=1,
                                max=100,
                                format="%d",
                            ).classes("w-32")

                        with ui.row().classes("gap-4"):
                            daily_loss = ui.number(
                                "Дневной лимит убытка (%)",
                                value=1.0,
                                min=1,
                                max=50,
                                format="%.1f",
                            ).classes("w-48")

                            monthly_loss = ui.number(
                                "Месячный лимит убытка (%)",
                                value=15.0,
                                min=5,
                                max=100,
                                format="%.1f",
                            ).classes("w-48")

                            opt_days = ui.number(
                                "Период оптимизации (дней)",
                                value=180,
                                min=30,
                                max=365,
                                format="%d",
                            ).classes("w-48")

                        with ui.row().classes("gap-4"):
                            opt_iterations = ui.number(
                                "Итераций оптимизации",
                                value=8,
                                min=5,
                                max=30,
                                format="%d",
                            ).classes("w-48")

                            opt_folds = ui.number(
                                "TEST-окон (walk-forward)",
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
        # Кнопки управления
        # ================================================================
        with ui.card().classes("w-full q-mb-md"):
            ui.label("🎮 Управление").classes("text-subtitle1 q-mb-sm")

            with ui.row().classes("gap-2 flex-wrap"):
                self.start_btn = ui.button(
                    "🚀 Начать торговлю (выбранные монеты)",
                    on_click=self._start_selected,
                ).classes("bg-positive text-white")

                self.stop_btn = ui.button(
                    "⏹ Остановить торговлю (выбранные монеты)",
                    on_click=self._stop_selected,
                ).classes("bg-negative text-white")

            with ui.row().classes("gap-2 flex-wrap q-mt-sm"):
                ui.button(
                    "🔒 Закрыть все позиции (все монеты)",
                    on_click=self._close_all_positions,
                ).classes("bg-warning text-white")

                ui.button(
                    "🔒 Закрыть позиции (выбранные монеты)",
                    on_click=self._close_selected_positions,
                ).classes("bg-orange text-white")

            # Чекбоксы выбора монет
            ui.label("Выбор монет:").classes("text-bold q-mt-md")
            self.symbol_checkboxes = {}
            with ui.row().classes("gap-4"):
                for symbol in self.memory.get_all_symbols():
                    self.symbol_checkboxes[symbol] = ui.checkbox(
                        symbol, value=False
                    )

        # ================================================================
        # Текущее состояние
        # ================================================================
        with ui.card().classes("w-full q-mb-md"):
            ui.label("📊 Текущее состояние").classes("text-subtitle1 q-mb-sm")

            self.state_table = ui.table(
                columns=[
                    {"name": "symbol", "label": "Монета", "field": "symbol", "align": "left"},
                    {"name": "status", "label": "Статус", "field": "status", "align": "left"},
                    {"name": "positions", "label": "Позиции", "field": "positions", "align": "center"},
                    {"name": "capital", "label": "Капитал", "field": "capital", "align": "right"},
                    {"name": "daily_pnl", "label": "Дневной P&L", "field": "daily_pnl", "align": "right"},
                    {"name": "monthly_pnl", "label": "Месячный P&L", "field": "monthly_pnl", "align": "right"},
                    {"name": "message", "label": "Сообщение", "field": "message", "align": "left"},
                ],
                rows=[],
                row_key="symbol",
            ).classes("w-full")


    # ----------------------------------------------------------------
    # Обработчики кнопок
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
            ui.notify("Выберите хотя бы одну монету.", type="warning")
            return

        # Проверка конфликта с авторежимом
        for sym in selected:
            auto_state = AutoModeState.get_state(sym)
            if auto_state and auto_state.get("is_running"):
                ui.notify(
                    f"Авторежим включён для {sym}. "
                    "Остановите его перед запуском автоторговли.",
                    type="warning",
                )
                return

            if is_auto_trading_running(sym):
                ui.notify(
                    f"Автоторговля для {sym} уже запущена.",
                    type="info",
                )
                return

        total_capital = self._get_total_capital()

        # Проверка суммы долей
        total_share = 0.0
        for sym in selected:
            cfg_inputs = self._coin_configs.get(sym, {})
            share = float(cfg_inputs["capital_share"].value or 30)
            total_share += share

        if total_share > 100:
            ui.notify(
                f"Сумма долей капитала ({total_share}%) превышает 100%.",
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
                    logger.exception(f"Ошибка запуска автоторговли {s}")
                    self.notify_fn(f"Ошибка запуска {s}: {e}", "negative")

            asyncio.create_task(asyncio.to_thread(worker))

        ui.notify(f"Запуск автоторговли для: {', '.join(selected)}", type="positive")

    def _stop_selected(self):
        selected = self._get_selected_symbols()
        if not selected:
            ui.notify("Выберите хотя бы одну монету.", type="warning")
            return

        for sym in selected:
            if is_auto_trading_running(sym):
                stop_auto_trading(sym, close_positions=False, reason="manual")
            else:
                ui.notify(f"Автоторговля для {sym} не запущена.", type="info")

    def _close_all_positions(self):
        with ui.dialog() as dialog, ui.card():
            ui.label("Закрыть ВСЕ позиции по ВСЕМ монетам?")
            ui.label("Это действие необратимо!").classes("text-negative")

            with ui.row().classes("justify-end q-mt-md"):
                ui.button("Отмена", on_click=dialog.close)
                ui.button(
                    "Закрыть все",
                    on_click=lambda: self._do_close_all(dialog),
                    color="negative",
                )

            dialog.open()

    def _do_close_all(self, dialog):
        dialog.close()
        close_all_positions_all_symbols()
        ui.notify("Все позиции закрыты.", type="warning")

    def _close_selected_positions(self):
        selected = self._get_selected_symbols()
        if not selected:
            ui.notify("Выберите хотя бы одну монету.", type="warning")
            return

        for sym in selected:
            close_all_positions_for_symbol(sym)

        ui.notify(f"Позиции закрыты для: {', '.join(selected)}", type="info")

    # ----------------------------------------------------------------
    # Обновление состояния
    # ----------------------------------------------------------------

    def _refresh_state(self):
        """Обновляет таблицу состояния (вызывается по таймеру)."""
        states = get_all_states()

        rows = []
        

        for sym, state in states.items():
            is_running = state.get("is_running", False)
            is_optimizing = state.get("is_optimizing", False)

            if is_optimizing:
                status = "🔄 Оптимизация"
            elif is_running:
                status = "🟢 Работает"
            else:
                status = "⏹ Остановлен"
            
            # --- HEARTBEAT: проверяем свежесть ---
            last_iter_ts = state.get("_last_iteration_ts")
            stale = False
            if is_running and last_iter_ts:
                import time
                age = time.time() - last_iter_ts
                if age > 120:  # 2 минуты без обновлений
                    status = "🟡 Нет данных (ошибка сети?)"
                    stale = True

            # Формируем строку детального статуса
            details = []
            if state.get("last_iteration_utc"):
                details.append(f"Проверка: {state['last_iteration_utc']}")
            if state.get("last_candle_time"):
                details.append(f"Свеча: {state['last_candle_time']}")
            if state.get("last_price"):
                details.append(f"Цена: {state['last_price']:.4f}")
            if state.get("candles_processed"):
                details.append(f"Обработано свечей: {state['candles_processed']}")
            if state.get("last_indicators_summary"):
                details.append(state['last_indicators_summary'])
            
            message_text = state.get("status_message", "")
            if details:
                message_text += " | " + " | ".join(details)

            # Форматируем PnL с процентами
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

        # Предупреждение о конфликте
        self._check_conflicts()

    def _check_conflicts(self):
        """Проверяет конфликты между автоторговлей и авторежимом."""
        warnings = []

        for sym in self.memory.get_all_symbols():
            auto_state = AutoModeState.get_state(sym)
            auto_mode_running = auto_state and auto_state.get("is_running", False)
            auto_trading = is_auto_trading_running(sym)

            if auto_mode_running and auto_trading:
                warnings.append(
                    f"⚠️ {sym}: автоторговля и авторежим включены одновременно!"
                )
            elif auto_mode_running:
                warnings.append(
                    f"🔒 {sym}: авторежим включён — автоторговля недоступна."
                )
            elif auto_trading:
                warnings.append(
                    f"🔒 {sym}: автоторговля включена — авторежим недоступен."
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
                "🧪 TESTNET — торговля на тестовом счёте."
            )
            self.testnet_label.classes("text-warning")
        else:
            self.testnet_label.set_text(
                "🔴 MAINNET — торговля на реальном счёте!"
            )
            self.testnet_label.classes("text-negative text-bold")

    def _load_balance_from_exchange(self):
        """
        Загружает USDT-баланс с биржи и обновляет метку.
        НЕ перезаписывает поле ввода — пользователь может вручную
        указать меньшую сумму (для контроля риска).
        """
        success, balance = self.executor.get_usdt_balance()
        
        if success and balance is not None:
            self.balance_label.set_text(
                f"💰 Баланс на бирже: {balance:.2f} USDT"
            )
            self.balance_label.classes("text-positive text-bold")
            
            # Автозаполнение поля при первой загрузке (если пользователь не менял)
            current_value = float(self.total_capital_input.value or 0)
            if current_value <= 1000.0:  # дефолтное значение — обновляем
                self.total_capital_input.set_value(balance)
        else:
            self.balance_label.set_text(
                "⚠️ Не удалось загрузить баланс с биржи"
            )
            self.balance_label.classes("text-warning")

    async def _load_balance_async(self):
        """Асинхронная обёртка для загрузки баланса (для ui.timer)."""
        await asyncio.to_thread(self._load_balance_from_exchange)

    def refresh(self):
        """Обновляет список монет (при добавлении новых)."""
        current_symbols = list(self.memory.get_all_symbols())
        for sym in current_symbols:
            if sym not in self.symbol_checkboxes:
                with ui.row():
                    self.symbol_checkboxes[sym] = ui.checkbox(sym, value=False)