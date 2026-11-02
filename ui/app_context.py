"""
Общий контекст приложения NiceGUI.

Здесь хранятся:
- сервисы;
- ссылки на страницы;
- ссылки на корневые UI-элементы;
- словарь страниц монет.

На данном этапе AppContext вводится мягко:
он сосуществует с уже существующими глобальными переменными и
становится единым контейнером состояния верхнего уровня.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AppContext:
    # --- Сервисы ---
    fetcher: Any = None
    ai_client: Any = None
    memory: Any = None
    app_settings: Any = None
    backtest_engine: Any = None
    backtest_optimizer: Any = None

    # --- Страницы ---
    summary_page: Any = None
    journal_page: Any = None
    chat_page: Any = None
    settings_page: Any = None
    backtest_page: Any = None
    auto_trading_page: Any = None
    research_lab_page: Any = None

    # --- UI-контейнеры верхнего уровня ---
    status_label: Any = None
    symbol_list_container: Any = None
    main_container: Any = None

    # --- Страницы монет ---
    symbol_pages: Dict[str, Any] = field(default_factory=dict)

    def reset_pages(self) -> None:
        """Сбрасывает ссылки на страницы и контейнеры UI."""
        self.summary_page = None
        self.journal_page = None
        self.chat_page = None
        self.settings_page = None
        self.backtest_page = None
        self.auto_trading_page = None
        self.research_lab_page = None
        self.status_label = None
        self.symbol_list_container = None
        self.main_container = None
        self.symbol_pages.clear()

    def bind_services(
        self,
        *,
        fetcher=None,
        ai_client=None,
        memory=None,
        app_settings=None,
        backtest_engine=None,
        backtest_optimizer=None,
    ) -> None:
        """Привязывает сервисы приложения."""
        self.fetcher = fetcher
        self.ai_client = ai_client
        self.memory = memory
        self.app_settings = app_settings
        self.backtest_engine = backtest_engine
        self.backtest_optimizer = backtest_optimizer

    def bind_shell(
        self,
        *,
        status_label=None,
        symbol_list_container=None,
        main_container=None,
    ) -> None:
        """Привязывает корневые UI-элементы."""
        self.status_label = status_label
        self.symbol_list_container = symbol_list_container
        self.main_container = main_container

    def bind_pages(
        self,
        *,
        summary_page=None,
        journal_page=None,
        chat_page=None,
        settings_page=None,
        backtest_page=None,
        auto_trading_page=None,
        research_lab_page=None,
    ) -> None:
        """Привязывает страницы верхнего уровня."""
        self.summary_page = summary_page
        self.journal_page = journal_page
        self.chat_page = chat_page
        self.settings_page = settings_page
        self.backtest_page = backtest_page
        self.auto_trading_page = auto_trading_page
        self.research_lab_page = research_lab_page


# Единый экземпляр контекста приложения.
app_context = AppContext()