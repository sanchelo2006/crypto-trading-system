"""
Creation and storage of the application's core services.

This module does not create UI elements and does not depend on NiceGUI.
It only assembles the application's infrastructure dependencies:
- market data loader;
- AI client;
- coin memory;
- settings;
- backtest engine;
- strategy optimizer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import config
from ai_client import DeepSeekClient
from app_settings import AppSettings
from backtest_engine import BacktestEngine
from optimization.backtest_optimizer import BacktestOptimizer
from data_fetcher import BybitDataFetcher
from memory_manager import MemoryManager
from trading.executor import BybitExecutor

logger = logging.getLogger(__name__)


@dataclass
class AppServices:
    """
    Set of the application's core services.

    At the first stage of the refactoring, nicegui_app.py may still keep
    references to these services in global variables for backward
    compatibility. Later this object will become the single AppContext.
    """

    fetcher: BybitDataFetcher
    ai_client: DeepSeekClient
    memory: MemoryManager
    app_settings: AppSettings
    backtest_engine: BacktestEngine
    backtest_optimizer: BacktestOptimizer
    bybit_executor: BybitExecutor

def create_services() -> AppServices:
    """
    Creates and wires together all services of the trading application.

    Exceptions are intentionally not suppressed: the calling code must decide
    how to show the error to the user — via the UI, a log, or by shutting down
    the application.
    """
    # Data comes from mainnet (or testnet if BYBIT_DATA_TESTNET=true).
    # Order execution follows BYBIT_TESTNET.
    fetcher = BybitDataFetcher(testnet=config.BYBIT_DATA_TESTNET)
    ai_client = DeepSeekClient(model=config.DEEPSEEK_MODEL)
    memory = MemoryManager()
    app_settings = AppSettings()

    backtest_engine = BacktestEngine(fetcher)
    backtest_optimizer = BacktestOptimizer(
        engine=backtest_engine,
        ai_client=ai_client,
        memory=memory,
    )

    bybit_executor = BybitExecutor(
        testnet=config.BYBIT_TESTNET,
        default_leverage=10,
    )

    logger.info("Application services created")

    return AppServices(
        fetcher=fetcher,
        ai_client=ai_client,
        memory=memory,
        app_settings=app_settings,
        backtest_engine=backtest_engine,
        backtest_optimizer=backtest_optimizer,
        bybit_executor=bybit_executor,
    )