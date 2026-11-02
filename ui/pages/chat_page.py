"""
Страница общего чата с AI.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

from nicegui import ui

import config
from tool_registry import ToolExecutor, get_tool_definitions


class ChatPage:
    """
    Общий чат с AI по монетам.

    Страница получает зависимости явно:
    - memory;
    - fetcher;
    - ai_client;
    - app_settings;
    - очередь UI-сообщений (callbacks queue_chat_message / loading).
    """

    def __init__(
        self,
        *,
        memory,
        fetcher,
        ai_client,
        app_settings,
        queue_chat_message_fn,
        queue_chat_loading_fn,
    ):
        self.memory = memory
        self.fetcher = fetcher
        self.ai_client = ai_client
        self.app_settings = app_settings
        self.queue_chat_message_fn = queue_chat_message_fn
        self.queue_chat_loading_fn = queue_chat_loading_fn

        self.container = ui.column().classes("w-full")
        self.messages = []
        self.is_processing = False

        with self.container:
            self.build_ui()

        self.container.visible = False

    def build_ui(self):
        ui.label("💬 Общий чат с AI").classes("text-h6")

        with ui.row().classes("w-full justify-end"):
            ui.button("🗑 Очистить чат", on_click=self.clear_chat, color="negative")

        self.chat_container = ui.column().classes("w-full h-96 overflow-auto border rounded p-2")

        with ui.row().classes("w-full"):
            self.chat_input = ui.input("Сообщение", placeholder="Напишите...").classes("flex-1").style("width: 100%;")
            self.send_btn = ui.button("Отправить", on_click=self.send_message).classes("bg-primary text-white")

    def clear_chat(self):
        self.messages = []
        self.chat_container.clear()
        ui.notify("Чат очищен.", type="info")

    def _set_loading_ui(self, loading: bool, text: str = "Думаю..."):
        """
        Меняет состояние UI чата.

        text пока не используется визуально, но сигнатура сохранена
        для совместимости с очередью событий.
        """
        self.is_processing = loading
        self.send_btn.enabled = not loading
        self.chat_input.enabled = not loading

    def send_message(self):
        if self.is_processing:
            return

        message = self.chat_input.value.strip()
        if not message:
            return

        self.chat_input.value = ""
        self.messages.append(("user", message))

        with self.chat_container:
            ui.chat_message(text=message, name="Вы", sent=True)

        self._set_loading_ui(True)

        def worker():
            logger = logging.getLogger("nicegui")

            try:
                symbol = self.detect_symbol(message)
                if not symbol:
                    self.queue_chat_message_fn("❓ Не удалось определить монету.")
                    return

                cm = self.memory.get(symbol)
                if not cm.is_initialized():
                    self.queue_chat_message_fn(
                        f"⚠️ Монета {symbol} не инициализирована."
                    )
                    return

                pc = next(
                    (p for p in config.TRADING_PAIRS if p["symbol"] == symbol),
                    None,
                )
                if not pc:
                    return

                lt = cm.get_last_candle_time()
                pkg, nt = self.fetcher.get_intraday_package(
                    symbol,
                    pc["category"],
                    from_time=lt,
                    settings=self.app_settings.data,
                )

                if nt:
                    cm.set_last_candle_time(nt)

                tool_executor = ToolExecutor(cm, self.fetcher, symbol)

                reply = self.ai_client.chat_with_tools(
                    user_message=message,
                    data_package=pkg,
                    trading_rules=cm.get_rules(),
                    active_trade=cm.get_active_trade(),
                    morning_context=cm.get_morning_context(),
                    intraday_context=cm.get_intraday_context(),
                    last_analysis=cm.get_last_analysis(),
                    behavior_rules=cm.get_behavior_rules(),
                    tools=get_tool_definitions(),
                    tool_executor=tool_executor.execute,
                    max_iterations=5,
                )

                self.queue_chat_message_fn(reply)

            except Exception as e:
                logger.exception("Ошибка ChatPage.send_message worker")
                self.queue_chat_message_fn(f"❌ Ошибка: {e}")

            finally:
                self.queue_chat_loading_fn(False)

        asyncio.create_task(asyncio.to_thread(worker))

    def detect_symbol(self, message: str) -> Optional[str]:
        """
        Пытается определить символ монеты по тексту пользователя.
        """
        for sym in self.memory.get_all_symbols():
            if re.search(r"\b" + re.escape(sym) + r"\b", message, re.IGNORECASE):
                return sym
        return None