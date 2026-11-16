"""
UI-layer helper functions for working with coin data, journals,
rules and catching up on historical candles.
This module contains no NiceGUI page classes and creates no UI components.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

import config
from indicators import calculate_indicators
from memory_manager import CoinMemory, MemoryManager
from utils.helpers import df_to_csv_string

logger = logging.getLogger(__name__)


def catch_up_historical_data(
    fetcher,
    symbol: str,
    category: str,
    coin_memory: CoinMemory,
) -> Tuple[Optional[Dict], Optional[datetime]]:
    """
    Loads candles missed while auto mode was idle.

    Logic:
    - takes the time of the last intraday update;
    - if less than 15 minutes passed, no catch-up is needed;
    - loads 15m and 1h candles;
    - computes technical indicators;
    - builds a package in intraday_update format.

    Parameters
    ----------
    fetcher:
        BybitDataFetcher instance.
    symbol:
        Trading pair, e.g. XRPUSDT.
    category:
        Market category, usually linear.
    coin_memory:
        Memory of the specific coin.

    Returns
    -------
    tuple:
        (data_package, new_last_candle_time).
        If no catch-up is needed or data was not received: (None, None).
    """
    state = coin_memory.get_state()
    last_update_str = state.get("last_update_utc")
    if not last_update_str:
        last_update_str = state.get("last_morning_update_utc")
    if not last_update_str:
        return None, None

    try:
        last_update = datetime.fromisoformat(last_update_str)
    except (TypeError, ValueError):
        logger.warning(
            "[%s] Failed to parse last_update_utc: %r",
            symbol,
            last_update_str,
        )
        return None, None

    if last_update.tzinfo is None:
        last_update = last_update.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    if (now - last_update).total_seconds() < 900:
        return None, None

    logger.info(
        "[%s] Catching up candles: %s → %s",
        symbol,
        last_update.strftime("%Y-%m-%d %H:%M UTC"),
        now.strftime("%Y-%m-%d %H:%M UTC"),
    )

    prices_15m = fetcher.fetch_ohlcv(
        symbol=symbol,
        interval="15",
        start_date=last_update,
        end_date=now,
        category=category,
    )
    prices_1h = fetcher.fetch_ohlcv(
        symbol=symbol,
        interval="60",
        start_date=last_update - timedelta(days=2),
        end_date=now,
        category=category,
    )

    if prices_15m is None or prices_15m.empty:
        logger.warning("[%s] Failed to fetch 15m candles for catch-up", symbol)
        return None, None

    indicators, timestamps = calculate_indicators(
        prices_15m,
        df_1h_for_pivot=prices_1h,
        num_latest=config.INDICATORS_LATEST_INTRADAY,
        return_timestamps=True,
        include_volume_profile=True,
    )

    new_last_candle_time = prices_15m["timestamp"].max()

    market_stats = fetcher._compute_market_stats(
        prices_1h,
        prices_15m,
        pd.DataFrame(),
    )

    data = {
        "prices_1h_csv": (
            df_to_csv_string(
                prices_1h.tail(config.INTRADAY_1H_CANDLES),
                ["timestamp", "open", "high", "low", "close", "volume"],
            )
            if prices_1h is not None and not prices_1h.empty
            else ""
        ),
        "prices_15m_csv": df_to_csv_string(
            prices_15m.tail(config.INTRADAY_15M_CANDLES),
            ["timestamp", "open", "high", "low", "close", "volume"],
        ),
        "indicators": json.dumps(indicators, ensure_ascii=False),
        "indicators_timestamps": timestamps,
        "market_stats": json.dumps(market_stats, ensure_ascii=False),
    }

    package = {
        "pair": symbol,
        "timestamp_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "message_type": "intraday_update",
        "data": data,
    }
    return package, new_last_candle_time


def clear_coin_data(
    coin_memory: CoinMemory,
    symbol: str,
    stop_auto_mode: Callable[[str], None],
    mode: str = "all",
) -> None:
    """
    Clears the state of a specific coin.

    The trade journal is intentionally NOT cleared — this matches the
    previous behavior of the "Clear" button in the interface.

    Parameters
    ----------
    coin_memory:
        Coin memory.
    symbol:
        Coin symbol.
    stop_auto_mode:
        Auto-mode stop function, usually AutoModeState.stop.
    mode:
        "analysis" — clears analysis, setups, indicators and contexts.
        "all" — additionally clears rules, state and strategy_spec.
    """
    stop_auto_mode(symbol)

    if mode in ("analysis", "all"):
        coin_memory.set_last_analysis(None)
        coin_memory.update_state(
            {
                "last_signal": None,
                "last_setup": None,
                "last_update_utc": None,
                "last_morning_update_utc": None,
                "morning_context": "",
                "intraday_context": "",
                "last_position_update": {},
                "indicators": {},
                "indicator_timestamps": [],
                "last_key_events": [],
            }
        )

    if mode == "all":
        coin_memory.update_rules("")
        coin_memory.update_behavior_rules("")
        coin_memory.set_active_trade(None)
        coin_memory.update_state(
            {
                "is_initialized": False,
                "active_trade": None,
                "last_candle_time": None,
                "trading_rules_list": [],
                "risk_management_rules_list": [],
                "research": {},
                "strategy_spec": None,
                "indicators": {},
                "indicator_timestamps": [],
                "last_key_events": [],
            }
        )

    coin_memory.save()


def clear_all_journals(memory_manager: MemoryManager) -> None:
    """
    Completely clears trade journals for all coins.
    """
    for symbol in memory_manager.get_all_symbols():
        coin_memory = memory_manager.get(symbol)
        coin_memory.journal = coin_memory._empty_journal()
        coin_memory.save()

def get_auto_trading_trades() -> pd.DataFrame:
    """
    Reads all auto-trading trade logs from data/auto_trading/logs/
    and returns a DataFrame compatible with the manual journal format.
    """
    logs_dir = config.DATA_DIR / "auto_trading" / "logs"
    journal_columns = [
        "date", "time_utc", "pair", "direction", "entry_price",
        "stop_loss", "take_profit", "exit_price", "result_pct",
        "reason", "errors", "entry_type", "exit_type",
        "position_value_usdt", "pnl_usdt",
    ]
    if not logs_dir.exists():
        return pd.DataFrame(columns=journal_columns)

    frames = []
    for csv_file in logs_dir.glob("*_trades_*.csv"):
        # Extract symbol from filename: {SYMBOL}_trades_YYYY_Www.csv
        symbol = csv_file.stem.split("_trades_")[0]
        try:
            df = pd.read_csv(csv_file)
        except Exception:
            logger.warning(f"Failed to read auto-trading log: {csv_file}")
            continue
        if df.empty:
            continue

        df = df.copy()

        # Parse exit_time into date / time columns
        if "exit_time" in df.columns:
            exit_dt = pd.to_datetime(df["exit_time"], errors="coerce")
        else:
            exit_dt = pd.Series([pd.NaT] * len(df), index=df.index)
        df["date"] = exit_dt.dt.strftime("%Y-%m-%d").fillna("")
        df["time_utc"] = exit_dt.dt.strftime("%H:%M:%S").fillna("")

        df["pair"] = symbol
        df["errors"] = ""

        # Build reason / exit_type from exit_reason
        if "exit_reason" in df.columns:
            exit_reason = df["exit_reason"].fillna("").astype(str)
        else:
            exit_reason = pd.Series([""] * len(df), index=df.index)
        df["reason"] = "Auto: " + exit_reason
        df["exit_type"] = exit_reason

        # Safely select journal columns (missing ones become NaN)
        mapped = df.reindex(columns=journal_columns)
        frames.append(mapped)

    if not frames:
        return pd.DataFrame(columns=journal_columns)

    result = pd.concat(frames, ignore_index=True)
    numeric_cols = [
        "entry_price", "stop_loss", "take_profit", "exit_price",
        "result_pct", "position_value_usdt", "pnl_usdt",
    ]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)
    return result


def get_all_trades(memory_manager: MemoryManager) -> pd.DataFrame:
    """
    Collects trade journals of all coins into a single DataFrame.
    Includes both manual trades (from trade_journal.csv) and
    auto-trading trades (from data/auto_trading/logs/).
    """
    dataframes = []
    for symbol in memory_manager.get_all_symbols():
        coin_memory = memory_manager.get(symbol)
        journal_df = coin_memory.get_journal_df()
        if journal_df.empty:
            continue
        journal_df = journal_df.copy()
        if "pair" not in journal_df.columns:
            journal_df["pair"] = symbol
        dataframes.append(journal_df)

    # Add auto-trading trades
    auto_df = get_auto_trading_trades()
    if not auto_df.empty:
        dataframes.append(auto_df)

    if not dataframes:
        return pd.DataFrame(
            columns=[
                "pair",
                "date",
                "time_utc",
                "direction",
                "entry_price",
                "exit_price",
                "result_pct",
                "reason",
                "errors",
            ]
        )
    return pd.concat(dataframes, ignore_index=True)


def parse_rules_from_text(text: str) -> Tuple[List[str], List[str]]:
    """
    Splits Markdown rules text into:
    - trading rules;
    - risk-management rules.

    Section headers are detected case-insensitively by keyword, so all of
    these work: "## Trading Rules", "### Trading rules",
    "## Risk Management", "### Risk-Management".
    Legacy Russian headers are recognized as well.

    Supported list markers: "-", "*", "•", "1.", "2)", "10." ...
    """
    if not text or not text.strip():
        return [], []

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    trading_rules: List[str] = []
    risk_rules: List[str] = []
    current_section = "trading"

    def _strip_marker(line: str) -> str:
        rule = re.sub(r"^[-*•]\s+", "", line)
        rule = re.sub(r"^\d+[.)]\s+", "", rule)
        return rule.strip()

    for line in lines:
        if line.startswith("#"):
            lowered = line.lower()
            # Risk is checked first: "Risk Management Rules" contains both keywords.
            # Russian keywords are kept for rules files written before the migration.
            if "risk" in lowered or "риск" in lowered:
                current_section = "risk"
            elif "trading" in lowered or "rule" in lowered or "правил" in lowered:
                current_section = "trading"
            continue

        rule = _strip_marker(line)
        if not rule:
            continue

        if current_section == "trading":
            trading_rules.append(rule)
        else:
            risk_rules.append(rule)

    # Fallback for plain text without headers.
    if not trading_rules and not risk_rules and lines:
        for line in lines:
            if line.startswith("#"):
                continue
            rule = _strip_marker(line)
            if rule:
                trading_rules.append(rule)

    return trading_rules, risk_rules