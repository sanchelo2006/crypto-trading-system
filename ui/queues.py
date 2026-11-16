"""
Thread-safe UI queues.

Background threads must not create or update NiceGUI elements directly.
They put events into UiEventQueue, and the main UI timer picks them up
and applies them in the UI context.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Callable, Deque, Dict, Optional, Tuple


class UiEventQueue:
    """
    Thread-safe container of UI events.

    Used as a bridge between background threads and the main NiceGUI thread.
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
    # Chat
    # ------------------------------------------------------------------

    def queue_chat_message(self, text: str, is_assistant: bool = True) -> None:
        """
        Adds a regular message to the chat queue.

        Parameters
        ----------
        text:
            Message text.
        is_assistant:
            True — message from the AI, False — message from the user.
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
        Adds a chat loading status change event.
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
        Applies the accumulated chat events in the UI context.

        chat_page is intentionally not strictly typed, so that the module
        does not depend on the concrete ChatPage class and does not create
        a circular import.
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
                        name="AI" if item.get("assistant", True) else "You",
                        sent=not item.get("assistant", True),
                    )

    # ------------------------------------------------------------------
    # Notifications and refresh
    # ------------------------------------------------------------------

    def add_notification(self, message: str, notification_type: str = "positive") -> None:
        """
        Adds a notification that the main UI thread will show later.
        """
        with self._queue_lock:
            self._notifications.append((message, notification_type))

    def schedule_refresh(self, symbol: str) -> None:
        """
        Schedules a refresh of a specific coin's page.
        """
        with self._queue_lock:
            self._refresh_symbols.append(symbol)

    def process_ui_events(
        self,
        notify_callback: Callable[[str, str], None],
        refresh_callback: Callable[[str], None],
    ) -> None:
        """
        Processes notifications and UI refresh requests.

        Parameters
        ----------
        notify_callback:
            A function of the form notify_callback(message, notification_type).
            In nicegui_app.py it usually calls ui.notify(...).

        refresh_callback:
            A function of the form refresh_callback(symbol) that refreshes
            a specific coin's page and the related summary pages.
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
    # Manual statuses
    # ------------------------------------------------------------------

    def set_manual_action(self, symbol: str, message: str) -> None:
        """
        Sets the manual operation text for a specific coin.

        Examples:
        - "🌅 Morning batch..."
        - "🪄 Initialization..."
        - "📊 Post-entry analysis..."
        """
        with self._manual_actions_lock:
            self._manual_actions[symbol] = message

    def clear_manual_action(self, symbol: str) -> None:
        """
        Removes the manual operation status for a coin.
        """
        with self._manual_actions_lock:
            self._manual_actions.pop(symbol, None)

    def get_manual_actions(self) -> Dict[str, str]:
        """
        Returns a copy of the current manual statuses.
        """
        with self._manual_actions_lock:
            return self._manual_actions.copy()