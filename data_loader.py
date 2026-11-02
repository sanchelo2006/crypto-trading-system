"""
Загрузка локальных исторических данных из папки backtest_data.

Структура:
    backtest_data/
        XRPUSDT/
            XRPUSDT_15min_2024.csv
            XRPUSDT_15min_2025.csv
        SOLUSDT/
            SOLUSDT_15min_2024.csv
            ...

Формат CSV: timestamp, open, high, low, close, volume
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Корневая папка проекта
PROJECT_ROOT = Path(__file__).parent
BACKTEST_DATA_DIR = PROJECT_ROOT / "backtest_data"

# Ожидаемые колонки ПОСЛЕ нормализации.
# В файлах backtest_data колонка времени называется datetime_utc —
# переименовываем её в timestamp на границе загрузки, чтобы весь
# остальной код (движок, тесты) работал с единым именем.
REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}
TIME_COLUMN_ALIASES = {"datetime_utc": "timestamp"}


def get_available_symbols() -> List[str]:
    """
    Возвращает список доступных символов (подпапки в backtest_data).
    
    Returns:
        Список имён монет, например ["XRPUSDT", "SOLUSDT", "HYPEUSDT"]
    """
    if not BACKTEST_DATA_DIR.exists():
        logger.warning(f"Папка {BACKTEST_DATA_DIR} не найдена")
        return []
    
    symbols = []
    for item in BACKTEST_DATA_DIR.iterdir():
        if item.is_dir():
            symbols.append(item.name)
    
    return sorted(symbols)


# Год в имени файла — это последние 4 цифры: XRPUSDT_15m_2023.csv -> 2023
_YEAR_RE = re.compile(r"_(\d{4})$")


def _iter_symbol_files(symbol: str) -> List[Path]:
    """Все csv-файлы монеты, отсортированные по имени."""
    symbol_dir = BACKTEST_DATA_DIR / symbol
    if not symbol_dir.exists():
        return []
    return sorted(symbol_dir.glob("*.csv"))


def _year_from_file(file: Path) -> Optional[int]:
    """Извлекает год из имени файла или None."""
    m = _YEAR_RE.search(file.stem)
    return int(m.group(1)) if m else None


def _find_symbol_file(symbol: str, year: int) -> Optional[Path]:
    """
    Находит csv-файл монеты за указанный год.
    Не зависит от токена интервала в имени (15m / 15min / ...).
    При нескольких кандидатах предпочитает 15-минутные файлы.
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
    Возвращает список доступных лет для монеты.
    
    Args:
        symbol: Имя монеты, например "XRPUSDT"
        
    Returns:
        Список годов, например [2023, 2024, 2025]
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
    Загружает локальные данные монеты за указанный год.
    
    Args:
        symbol: Имя монеты, например "XRPUSDT"
        year: Год данных, например 2025
        start_date: Опциональная дата начала (формат "2025-01-01")
        end_date: Опциональная дата конца (формат "2025-12-31")
        
    Returns:
        DataFrame с колонками: timestamp, open, high, low, close, volume
        
    Raises:
        FileNotFoundError: Если файл не найден
        ValueError: Если формат данных некорректен
    """
    file_path = _find_symbol_file(symbol, year)
    
    if file_path is None:
        raise FileNotFoundError(
            f"Не найден csv-файл {symbol} за {year} год в "
            f"{BACKTEST_DATA_DIR / symbol}"
        )
    
    logger.info(f"Загрузка данных: {file_path}")
    
    # Определяем разделитель по первой строке (csv может быть
    # как с запятой, так и с табуляцией).
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    sep = "\t" if "\t" in first_line else ","

    # Читаем CSV
    df = pd.read_csv(file_path, sep=sep)

    # Нормализуем имена колонок времени (datetime_utc -> timestamp)
    df = df.rename(
        columns={
            old: new
            for old, new in TIME_COLUMN_ALIASES.items()
            if old in df.columns
        }
    )

    # Проверяем наличие необходимых колонок
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Отсутствуют колонки в {file_path}: {missing}")

    # Парсим временные метки (формат 2023-01-01T00:00:00+00:00)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    
    # Приводим числовые колонки
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Удаляем строки с NaN
    df = df.dropna(subset=numeric_cols)
    
    # Сортируем по времени
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Фильтруем по датам, если указаны
    if start_date:
        df = df[df["timestamp"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["timestamp"] <= pd.to_datetime(end_date)]
    
    logger.info(f"Загружено {len(df)} строк для {symbol} за {year} год")
    return df


def load_first_available_data(
    preferred_symbols: Optional[List[str]] = None,
) -> Tuple[str, int, pd.DataFrame]:
    """
    Загружает первые доступные данные.
    
    Args:
        preferred_symbols: Приоритетный список монет
        
    Returns:
        Кортеж (символ, год, DataFrame)
        
    Raises:
        RuntimeError: Если данные не найдены
    """
    symbols = get_available_symbols()
    
    if not symbols:
        raise RuntimeError(
            f"Нет данных в {BACKTEST_DATA_DIR}. "
            "Создайте подпапки с монетами и CSV-файлы."
        )
    
    # Если есть приоритетные монеты, ищем их первыми
    if preferred_symbols:
        for sym in preferred_symbols:
            if sym in symbols:
                years = get_available_years(sym)
                if years:
                    df = load_local_data(sym, years[0])
                    return sym, years[0], df
    
    # Иначе берём первую доступную монету
    for sym in symbols:
        years = get_available_years(sym)
        if years:
            df = load_local_data(sym, years[0])
            return sym, years[0], df
    
    raise RuntimeError("Не удалось загрузить данные из доступных монет")