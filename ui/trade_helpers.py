"""
Вспомогательные функции UI-слоя для работы с данными монет, журналами,
правилами и докачкой исторических свечей.

Модуль не содержит классов страниц NiceGUI и не создаёт UI-компоненты.
"""

from __future__ import annotations

import json
import logging
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
    Загружает пропущенные свечи после простоя авторежима.

    Логика сохранена из прежнего nicegui_app.py:
    - берёт время последнего intraday-обновления;
    - если прошло менее 15 минут, докачка не нужна;
    - загружает 15m и 1h свечи;
    - рассчитывает технические индикаторы;
    - собирает пакет формата intraday_update.

    Parameters
    ----------
    fetcher:
        Экземпляр BybitDataFetcher.

    symbol:
        Торговая пара, например XRPUSDT.

    category:
        Категория рынка, обычно linear.

    coin_memory:
        Память конкретной монеты.

    Returns
    -------
    tuple:
        (data_package, new_last_candle_time).

        Если докачка не нужна или данные не получены:
        (None, None).
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
            "[%s] Не удалось распарсить last_update_utc: %r",
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
        "[%s] Докачка свечей: %s → %s",
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
        logger.warning("[%s] Не удалось получить 15m свечи для докачки", symbol)
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
    Очищает состояние конкретной монеты.

    Журнал сделок намеренно не очищается — это соответствует прежнему
    поведению кнопки «Очистить» в интерфейсе.

    Parameters
    ----------
    coin_memory:
        Память монеты.

    symbol:
        Символ монеты.

    stop_auto_mode:
        Функция остановки авторежима, обычно AutoModeState.stop.

    mode:
        "analysis" — очищает анализ, сетапы, индикаторы и контексты.
        "all" — дополнительно очищает правила, состояние и strategy_spec.
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
    Полностью очищает журналы сделок для всех монет.
    """
    for symbol in memory_manager.get_all_symbols():
        coin_memory = memory_manager.get(symbol)
        coin_memory.journal = coin_memory._empty_journal()
        coin_memory.save()


def get_all_trades(memory_manager: MemoryManager) -> pd.DataFrame:
    """
    Собирает журналы сделок всех монет в единый DataFrame.
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
    Разделяет Markdown-текст правил на:
    - торговые правила;
    - риск-правила.

    Поддерживаются заголовки:
    - ## Правила торговли
    - ### Правила торговли
    - ## Риск-менеджмент
    - ### Риск-менеджмент

    Также поддерживаются списки:
    - Правило
    * Правило
    1. Правило
    """
    if not text or not text.strip():
        return [], []

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    trading_rules: List[str] = []
    risk_rules: List[str] = []
    current_section = "trading"

    prefixes = [
        "- ",
        "* ",
        "1. ",
        "2. ",
        "3. ",
        "4. ",
        "5. ",
        "6. ",
        "7. ",
        "8. ",
        "9. ",
    ]

    for line in lines:
        if (
            line.startswith("## Риск-менеджмент")
            or line.startswith("### Риск-менеджмент")
        ):
            current_section = "risk"
            continue

        if (
            line.startswith("## Правила торговли")
            or line.startswith("### Правила торговли")
        ):
            current_section = "trading"
            continue

        if line.startswith("#"):
            continue

        rule = line

        for prefix in prefixes:
            if rule.startswith(prefix):
                rule = rule[len(prefix):]
                break

        if not rule:
            continue

        if current_section == "trading":
            trading_rules.append(rule)
        else:
            risk_rules.append(rule)

    # Fallback для обычного текста без заголовков.
    if not trading_rules and not risk_rules and lines:
        for line in lines:
            if line.startswith("#"):
                continue

            rule = line

            for prefix in prefixes:
                if rule.startswith(prefix):
                    rule = rule[len(prefix):]
                    break

            if rule:
                trading_rules.append(rule)

    return trading_rules, risk_rules