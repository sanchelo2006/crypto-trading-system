"""
Shared application context for the NiceGUI interface.

Stores:
- services;
- references to pages;
- references to root UI elements;
- the dictionary of coin pages.

At this stage AppContext is introduced gradually:
it coexists with the already existing global variables and
becomes the single container of top-level state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AppContext:
    # --- Services ---
    fetcher: Any = None
    ai_client: Any = None
    memory: Any = None
    app_settings: Any = None
    backtest_engine: Any = None
    backtest_optimizer: Any = None

    # --- Pages ---
    summary_page: Any = None
    journal_page: Any = None
    chat_page: Any = None
    settings_page: Any = None
    logs_page: Any = None
    backtest_page: Any = None
    auto_trading_page: Any = None
    

    # --- Top-level UI containers ---
    status_label: Any = None
    symbol_list_container: Any = None
    main_container: Any = None

    # --- Coin pages ---
    symbol_pages: Dict[str, Any] = field(default_factory=dict)

    def reset_pages(self) -> None:
        """Resets references to pages and UI containers."""
        self.summary_page = None
        self.journal_page = None
        self.chat_page = None
        self.settings_page = None
        self.logs_page = None
        self.backtest_page = None
        self.auto_trading_page = None
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
        """Binds the application services."""
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
        """Binds the root UI elements."""
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
        logs_page=None,
        backtest_page=None,
        auto_trading_page=None,
    ) -> None:
        """Binds the top-level pages."""
        self.summary_page = summary_page
        self.journal_page = journal_page
        self.chat_page = chat_page
        self.settings_page = settings_page
        self.logs_page = logs_page
        self.backtest_page = backtest_page
        self.auto_trading_page = auto_trading_page


# Single instance of the application context.
app_context = AppContext()