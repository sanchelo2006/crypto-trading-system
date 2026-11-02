"""
Создание и хранение основных сервисов приложения.

Этот модуль не создаёт UI-элементы и не зависит от NiceGUI.
Он только собирает инфраструктурные зависимости приложения:
- загрузчик рыночных данных;
- AI-клиент;
- память монет;
- настройки;
- движок бэктеста;
- оптимизатор стратегии.
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
    Набор основных сервисов приложения.

    На первом этапе рефакторинга nicegui_app.py ещё может хранить
    ссылки на эти сервисы в глобальных переменных для обратной
    совместимости. Позже этот объект станет единым AppContext.
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
    Создаёт и связывает все сервисы торгового приложения.

    Исключения намеренно не подавляются: вызывающий код должен решить,
    как показать ошибку пользователю — через UI, лог или завершение
    приложения.
    """
    # Данные — с mainnet (или testnet, если BYBIT_DATA_TESTNET=true).
    # Исполнение ордеров — по BYBIT_TESTNET.
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

    logger.info("Сервисы приложения созданы")

    return AppServices(
        fetcher=fetcher,
        ai_client=ai_client,
        memory=memory,
        app_settings=app_settings,
        backtest_engine=backtest_engine,
        backtest_optimizer=backtest_optimizer,
        bybit_executor=bybit_executor,
    )