"""
Trading system configuration.
Optimized to save tokens without losing analysis quality.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
PROMPTS_DIR = BASE_DIR / "prompts"

# --- API keys ---
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# --- Single DeepSeek model ---
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# --- Bybit ---
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")

# Order execution flag: testnet or mainnet.
# Used by BybitExecutor.
BYBIT_TESTNET = os.getenv("BYBIT_TESTNET", "false").lower() == "true"

# Separate flag for the DATA SOURCE (OHLCV, candles, orderbook, trades).
# Used by BybitDataFetcher — i.e. by backtest, optimization,
# morning/intraday packages and auto trading.
#
# Why they are separate:
#  - testnet history is synthetic, ATR/ADX there are distorted several times over;
#  - backtest and optimization must run on REAL mainnet data;
#  - orders can still stay on testnet until you are ready to trade real money.
#
# For fully consistent behavior: both flags identical.
# For safe signal testing: BYBIT_TESTNET=true, BYBIT_DATA_TESTNET=false.
BYBIT_DATA_TESTNET = os.getenv("BYBIT_DATA_TESTNET", "false").lower() == "true"

# --- Traded pairs ---
TRADING_PAIRS = [
    {"symbol": "XRPUSDT", "category": "linear"},
    {"symbol": "HYPEUSDT", "category": "linear"},
    {"symbol": "SOLUSDT", "category": "linear"},
    {"symbol": "ETHUSDT", "category": "linear"},
    {"symbol": "DOGEUSDT", "category": "linear"},
    {"symbol": "ARBUSDT", "category": "linear"},
]

# --- Timeframes ---
TIMEFRAME_1H = "60"
TIMEFRAME_15M = "15"
BYBIT_TIMEFRAME_MAP = {"60": "60", "15": "15", "1h": "60", "15m": "15"}
INTERVAL_SECONDS = {"60": 3600, "15": 900, "1h": 3600, "15m": 900}

# --- Request parameters ---
API_RETRY_COUNT = 3

# REDUCED to save tokens:
MORNING_1H_CANDLES = 15       # was 30
MORNING_15M_CANDLES = 64      # was 96 (~16 hours = 64 candles)
INTRADAY_1H_CANDLES = 15      # 1h candles in the intraday package
INTRADAY_15M_CANDLES = 30     # was 50
INTRADAY_HOURS_BACK = 8       # was 13 (30 x 15m = 7.5h -> 8h with margin)
DAYS_BACK_FOR_PIVOT = 5
ORDERBOOK_DEPTH = 10          # was 25
RECENT_TRADES_LIMIT = 100     # used for aggregation only, not for CSV
FUNDAMENTALS_LIMIT = 1000

# --- Indicators (reduced to save tokens) ---
INDICATORS_LATEST_INTRADAY = 15   # was 30
INDICATORS_LATEST_MORNING = 20    # was 50
VWAP_HISTORICAL_DAYS = 5
PIVOT_HISTORICAL_DAYS = 5

# --- Schedule ---
SCHEDULE_SECOND = 5
SCHEDULE_MINUTES = "0,15,30,45"

# --- Risk management ---
DEFAULT_MAX_STOP_LOSS_PCT = 2.0

# --- Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# --- UI refresh interval in auto mode (seconds) ---
UI_REFRESH_INTERVAL = 5

# --- Auto trading ---
AUTO_TRADING_ENABLED = os.getenv("AUTO_TRADING_ENABLED", "true").lower() == "true"

# --- Backtest: slippage model ---
# Applied "against the trade":
#   entry at open: long pays more, short sells cheaper;
#   exit at SL/TP: exit price is shifted toward a loss.
# Soft exits (time_stop, level_failure, trend_loss, macd_flip,
# end_of_period) are executed at the already observed close price
# and are not subject to slippage.
# 0.05% is roughly the average taker slippage on Bybit linear perpetuals.
BACKTEST_SLIPPAGE_PCT = float(os.getenv("BACKTEST_SLIPPAGE_PCT", "0.05"))