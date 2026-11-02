"""
Конфигурация торговой системы.
Оптимизировано для экономии токенов без потери качества анализа.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# --- Пути ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
PROMPTS_DIR = BASE_DIR / "prompts"

# --- API ключи ---
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# --- Единая модель DeepSeek ---
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# --- Bybit ---
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")

# Флаг исполнения ордеров: testnet или mainnet.
# Используется BybitExecutor.
BYBIT_TESTNET = os.getenv("BYBIT_TESTNET", "false").lower() == "true"

# Отдельный флаг ИСТОЧНИКА ДАННЫХ (OHLCV, свечи, стакан, трейды).
# Используется BybitDataFetcher — то есть бэктестом, оптимизацией,
# утренними/внутридневными пакетами и автоторговлей.
#
# Зачем разделять:
#  - testnet-история — синтетическая, ATR/ADX там искажены в разы;
#  - бэктест и оптимизацию нужно гонять на РЕАЛЬНЫХ данных mainnet;
#  - ордера при этом можно оставить на testnet, пока нет готовности
#    торговать реальными деньгами.
#
# Для полностью честной работы: оба флага одинаковые.
# Для безопасного тестирования сигналов: BYBIT_TESTNET=true, BYBIT_DATA_TESTNET=false.
BYBIT_DATA_TESTNET = os.getenv("BYBIT_DATA_TESTNET", "false").lower() == "true"

# --- Торгуемые пары ---
TRADING_PAIRS = [
    {"symbol": "XRPUSDT", "category": "linear"},
    {"symbol": "HYPEUSDT", "category": "linear"},
    {"symbol": "SOLUSDT", "category": "linear"},
    {"symbol": "ETHUSDT", "category": "linear"},
    {"symbol": "DOGEUSDT", "category": "linear"},
    {"symbol": "ARBUSDT", "category": "linear"},
]

# --- Таймфреймы ---
TIMEFRAME_1H = "60"
TIMEFRAME_15M = "15"
BYBIT_TIMEFRAME_MAP = {"60": "60", "15": "15", "1h": "60", "15m": "15"}
INTERVAL_SECONDS = {"60": 3600, "15": 900, "1h": 3600, "15m": 900}

# --- Параметры запросов ---
API_RETRY_COUNT = 3

# УМЕНЬШЕНО для экономии токенов:
MORNING_1H_CANDLES = 15       # было 30
MORNING_15M_CANDLES = 64      # было 96 (~16 часов = 64 свечи)
INTRADAY_1H_CANDLES = 15      # 1h свечи во внутридневном пакете
INTRADAY_15M_CANDLES = 30     # было 50
INTRADAY_HOURS_BACK = 8       # было 13 (30 × 15m = 7.5ч → 8ч с запасом)
DAYS_BACK_FOR_PIVOT = 5
ORDERBOOK_DEPTH = 10          # было 25
RECENT_TRADES_LIMIT = 100     # используется только для агрегации, не для CSV
FUNDAMENTALS_LIMIT = 1000

# --- Индикаторы (уменьшено для экономии токенов) ---
INDICATORS_LATEST_INTRADAY = 15   # было 30
INDICATORS_LATEST_MORNING = 20    # было 50
VWAP_HISTORICAL_DAYS = 5
PIVOT_HISTORICAL_DAYS = 5

# --- Расписание ---
SCHEDULE_SECOND = 5
SCHEDULE_MINUTES = "0,15,30,45"

# --- Риск-менеджмент ---
DEFAULT_MAX_STOP_LOSS_PCT = 2.0

# --- Логирование ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# --- Интервал обновления UI в авторежиме (секунды) ---
UI_REFRESH_INTERVAL = 5

# --- Автоторговля ---
AUTO_TRADING_ENABLED = os.getenv("AUTO_TRADING_ENABLED", "true").lower() == "true"

# --- Бэктест: модель проскальзывания ---
# Применяется «против сделки»:
#   вход по open: long платит дороже, short продаёт дешевле;
#   выход по SL/TP: цена выхода смещается в сторону убытка.
# Мягкие выходы (time_stop, level_failure, trend_loss, macd_flip,
# end_of_period) исполняются по уже наблюдаемой close-цене и
# проскальзыванию не подвержены.
# 0.05% ≈ средний taker-слиппедж на линейных перпетуалах Bybit.
BACKTEST_SLIPPAGE_PCT = float(os.getenv("BACKTEST_SLIPPAGE_PCT", "0.05"))
