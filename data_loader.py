"""
Loading local historical data from the backtest_data folder.

Structure:
backtest_data/
    XRPUSDT/
        XRPUSDT_15min_2024.csv
        XRPUSDT_15min_2025.csv
    SOLUSDT/
        SOLUSDT_15min_2024.csv
    ...

CSV format: timestamp, open, high, low, close, volume
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Project root folder
PROJECT_ROOT = Path(__file__).parent
BACKTEST_DATA_DIR = PROJECT_ROOT / "backtest_data"

# Expected columns AFTER normalization.
# In backtest_data files the time column is named datetime_utc —
# we rename it to timestamp at the loading boundary so that all the rest
# of the code (engine, tests) works with a single name.
REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}
TIME_COLUMN_ALIASES = {"datetime_utc": "timestamp"}


def get_available_symbols() -> List[str]:
    """
    Returns the list of available symbols (subfolders in backtest_data).

    Returns:
        List of coin names, e.g. ["XRPUSDT", "SOLUSDT", "HYPEUSDT"]
    """
    if not BACKTEST_DATA_DIR.exists():
        logger.warning(f"Folder {BACKTEST_DATA_DIR} not found")
        return []

    symbols = []
    for item in BACKTEST_DATA_DIR.iterdir():
        if item.is_dir():
            symbols.append(item.name)
    return sorted(symbols)


# The year in the file name is the last 4 digits: XRPUSDT_15m_2023.csv -> 2023
_YEAR_RE = re.compile(r"_(\d{4})$")


def _iter_symbol_files(symbol: str) -> List[Path]:
    """All csv files of the coin, sorted by name."""
    symbol_dir = BACKTEST_DATA_DIR / symbol
    if not symbol_dir.exists():
        return []
    return sorted(symbol_dir.glob("*.csv"))


def _year_from_file(file: Path) -> Optional[int]:
    """Extracts the year from the file name, or None."""
    m = _YEAR_RE.search(file.stem)
    return int(m.group(1)) if m else None


def _find_symbol_file(symbol: str, year: int) -> Optional[Path]:
    """
    Finds the coin's csv file for the given year.
    Does not depend on the interval token in the name (15m / 15min / ...).
    If there are several candidates, prefers 15-minute files.
    """
    candidates = [
        f for f in _iter_symbol_files(symbol) if _year_from_file(f) == year
    ]
    if not candidates:
        return None

    for f in candidates:
        if "15m" in f.stem or "15min" in f.stem:
            return f
    return candidates[0]


def get_available_years(symbol: str) -> List[int]:
    """
    Returns the list of available years for the coin.

    Args:
        symbol: Coin name, e.g. "XRPUSDT"

    Returns:
        List of years, e.g. [2023, 2024, 2025]
    """
    years = [_year_from_file(f) for f in _iter_symbol_files(symbol)]
    return sorted({y for y in years if y is not None})


def load_local_data(
    symbol: str,
    year: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Loads local data of the coin for the given year.

    Args:
        symbol: Coin name, e.g. "XRPUSDT"
        year: Data year, e.g. 2025
        start_date: Optional start date (format "2025-01-01")
        end_date: Optional end date (format "2025-12-31")

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume

    Raises:
        FileNotFoundError: If the file is not found
        ValueError: If the data format is invalid
    """
    file_path = _find_symbol_file(symbol, year)
    if file_path is None:
        raise FileNotFoundError(
            f"No csv file found for {symbol} for year {year} in "
            f"{BACKTEST_DATA_DIR / symbol}"
        )

    logger.info(f"Loading data: {file_path}")

    # Detect the delimiter from the first line (csv can use
    # either a comma or a tab).
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    sep = "\t" if "\t" in first_line else ","

    # Read the CSV
    df = pd.read_csv(file_path, sep=sep)

    # Normalize time column names (datetime_utc -> timestamp)
    df = df.rename(
        columns={
            old: new
            for old, new in TIME_COLUMN_ALIASES.items()
            if old in df.columns
        }
    )

    # Check that all required columns are present
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {file_path}: {missing}")

    # Parse timestamps (format 2023-01-01T00:00:00+00:00)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Cast numeric columns
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with NaN
    df = df.dropna(subset=numeric_cols)

    # Sort by time
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Filter by dates if provided
    if start_date:
        df = df[df["timestamp"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["timestamp"] <= pd.to_datetime(end_date)]

    logger.info(f"Loaded {len(df)} rows for {symbol} for year {year}")
    return df


def load_first_available_data(
    preferred_symbols: Optional[List[str]] = None,
) -> Tuple[str, int, pd.DataFrame]:
    """
    Loads the first available data.

    Args:
        preferred_symbols: Priority list of coins

    Returns:
        Tuple (symbol, year, DataFrame)

    Raises:
        RuntimeError: If no data is found
    """
    symbols = get_available_symbols()
    if not symbols:
        raise RuntimeError(
            f"No data in {BACKTEST_DATA_DIR}. "
            "Create subfolders with coins and CSV files."
        )

    # If there are preferred coins, look for them first
    if preferred_symbols:
        for sym in preferred_symbols:
            if sym in symbols:
                years = get_available_years(sym)
                if years:
                    df = load_local_data(sym, years[0])
                    return sym, years[0], df

    # Otherwise take the first available coin
    for sym in symbols:
        years = get_available_years(sym)
        if years:
            df = load_local_data(sym, years[0])
            return sym, years[0], df

    raise RuntimeError("Failed to load data from the available coins")