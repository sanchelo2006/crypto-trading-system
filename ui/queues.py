"""
Потокобезопасные UI-очереди.

Фоновые потоки не должны напрямую создавать или обновлять элементы NiceGUI.
Они помещают события в UiEventQueue, а главный UI-таймер забирает их
и применяет в UI-контексте.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Callable, Deque, Dict, Optional, Tuple


class UiEventQueue:
    """
    Потокобезопасный контейнер UI-событий.

    Используется как мост между фоновыми потоками и главным потоком NiceGUI.
    """

    def __init__(self) -> None:
        self._chat_messages: Deque[dict] = deque()
        self._notifications: Deque[Tuple[str, str]] = deque()
        self._refresh_symbols: Deque[str] = deque()
        self._manual_actions: Dict[str, str] = {}

        self._chat_lock = threading.Lock()
        self._queue_lock = threading.Lock()
        self._manual_actions_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Чат
    # ------------------------------------------------------------------

    def queue_chat_message(self, text: str, is_assistant: bool = True) -> None:
        """
        Добавляет обычное сообщение в очередь чата.

        Parameters
        ----------
        text:
            Текст сообщения.
        is_assistant:
            True — сообщение от AI, False — сообщение пользователя.
        """
        with self._chat_lock:
            self._chat_messages.append(
                {
                    "type": "message",
                    "text": text,
                    "assistant": is_assistant,
                }
            )

    def queue_chat_loading(self, loading: bool, text: str = "") -> None:
        """
        Добавляет событие изменения статуса загрузки чата.
        """
        with self._chat_lock:
            self._chat_messages.append(
                {
                    "type": "loading",
                    "loading": loading,
                    "text": text,
                }
            )

    def process_chat_queue(self, chat_page) -> None:
        """
        Применяет накопленные события чата в UI-контексте.

        chat_page намеренно не типизирован жёстко, чтобы модуль не зависел
        от конкретного класса ChatPage и не создавал циклический импорт.
        """
        if chat_page is None:
            return

        from nicegui import ui

        with self._chat_lock:
            while self._chat_messages:
                item = self._chat_messages.popleft()

                if item.get("type") == "loading":
                    chat_page._set_loading_ui(
                        item.get("loading", False),
                        item.get("text", ""),
                    )
                    continue

                with chat_page.chat_container:
                    ui.chat_message(
                        text=item.get("text", ""),
                        name="AI" if item.get("assistant", True) else "Вы",
                        sent=not item.get("assistant", True),
                    )

    # ------------------------------------------------------------------
    # Уведомления и refresh
    # ------------------------------------------------------------------

    def add_notification(self, message: str, notification_type: str = "positive") -> None:
        """
        Добавляет уведомление, которое позже покажет главный UI-поток.
        """
        with self._queue_lock:
            self._notifications.append((message, notification_type))

    def schedule_refresh(self, symbol: str) -> None:
        """
        Планирует обновление страницы конкретной монеты.
        """
        with self._queue_lock:
            self._refresh_symbols.append(symbol)

    def process_ui_events(
        self,
        notify_callback: Callable[[str, str], None],
        refresh_callback: Callable[[str], None],
    ) -> None:
        """
        Обрабатывает уведомления и запросы на обновление UI.

        Parameters
        ----------
        notify_callback:
            Функция вида notify_callback(message, notification_type).
            В nicegui_app.py обычно вызывает ui.notify(...).

        refresh_callback:
            Функция вида refresh_callback(symbol), которая обновляет
            страницу конкретной монеты и связанные сводные страницы.
        """
        notifications = []
        refresh_symbols = []

        with self._queue_lock:
            while self._notifications:
                notifications.append(self._notifications.popleft())

            while self._refresh_symbols:
                refresh_symbols.append(self._refresh_symbols.popleft())

        for message, notification_type in notifications:
            notify_callback(message, notification_type)

        for symbol in refresh_symbols:
            refresh_callback(symbol)

    # ------------------------------------------------------------------
    # Ручные статусы
    # ------------------------------------------------------------------

    def set_manual_action(self, symbol: str, message: str) -> None:
        """
        Устанавливает текст ручной операции для конкретной монеты.

        Примеры:
        - «🌅 Утренний пакет...»
        - «🪄 Инициализация...»
        - «📊 Анализ после входа...»
        """
        with self._manual_actions_lock:
            self._manual_actions[symbol] = message

    def clear_manual_action(self, symbol: str) -> None:
        """
        Удаляет ручной статус операции для монеты.
        """
        with self._manual_actions_lock:
            self._manual_actions.pop(symbol, None)

    def get_manual_actions(self) -> Dict[str, str]:
        """
        Возвращает копию текущих ручных статусов.
        """
        with self._manual_actions_lock:
            return self._manual_actions.copy()