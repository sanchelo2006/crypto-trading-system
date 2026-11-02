"""
Обработка ответов AI-модели и применение их к памяти монеты.

Этот модуль не создаёт UI-компоненты и не зависит от страниц NiceGUI.
Он преобразует ответ модели в обновление состояния CoinMemory.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Optional

from memory_manager import CoinMemory
from ui.trade_helpers import parse_rules_from_text


logger = logging.getLogger(__name__)
key_logger = logging.getLogger("key_events")


def _ensure_safe_setup_fallback(
    response: Dict,
    symbol: str,
    has_active_trade: bool = False,
) -> None:
    """
    Если модель не вернула ни setup, ни signal, ни position_update, создаём
    безопасный fallback-setup, чтобы UI не остался без структуры данных.

    Логика полностью сохранена из прежнего nicegui_app.py.
    """
    action = response.get("action", "ANALYSIS")

    if (
        action != "INIT"
        and not response.get("setup")
        and not response.get("signal")
        and not response.get("position_update")
        and not has_active_trade
    ):
        logger.warning(
            "[%s] Ответ модели без setup/signal. Создаю пустой setup для безопасности.",
            symbol,
        )
        response["setup"] = {
            "direction": "long",
            "confidence_pct": 0,
            "entry_price": 0.0,
            "stop_loss": 0.0,
            "take_profit": [],
            "conditions": [
                "Модель не сформировала условия — перегрузка контекста или неопределённость рынка"
            ],
            "cancel_conditions": ["Ручное управление"],
            "rationale": "Автоматический fallback: модель не вернула сетап.",
        }


def _process_init_response(
    response: Dict,
    coin_memory: CoinMemory,
    symbol: str,
) -> str:
    """
    Обрабатывает INIT-ответ:
    - сохраняет research;
    - сохраняет strategy_spec;
    - сохраняет текстовые правила и списки правил.
    """
    if response.get("research"):
        coin_memory.update_state({"research": response["research"]})

    spec = response.get("strategy_spec")
    if spec:
        try:
            from strategy_spec import validate_strategy_spec, ensure_engine_defaults

            err = validate_strategy_spec(spec)
            if err is None:
                coin_memory.update_state(
                    {"strategy_spec": ensure_engine_defaults(spec)}
                )
                logger.info(
                    "[%s] strategy_spec сохранён (с дефолтами движка)",
                    symbol,
                )
            else:
                logger.warning(
                    "[%s] strategy_spec от AI невалиден: %s",
                    symbol,
                    err,
                )
        except Exception as e:
            logger.warning(
                "[%s] Ошибка валидации strategy_spec: %s",
                symbol,
                e,
            )

    trading = response.get("trading_rules_list", [])
    risk = response.get("risk_management_rules_list", [])

    if trading or risk:
        coin_memory.update_state(
            {
                "trading_rules_list": trading,
                "risk_management_rules_list": risk,
            }
        )

        rules_text = "## Правила торговли\n" + "\n".join(f"- {r}" for r in trading)
        if risk:
            rules_text += "\n\n## Риск-менеджмент\n" + "\n".join(f"- {r}" for r in risk)

        coin_memory.update_rules(rules_text)

    elif response.get("new_rules"):
        coin_memory.update_rules(response["new_rules"])
        trading_list, risk_list = parse_rules_from_text(response["new_rules"])
        coin_memory.update_state(
            {
                "trading_rules_list": trading_list,
                "risk_management_rules_list": risk_list,
            }
        )
    else:
        coin_memory.update_rules(
            f"# Базовые правила {symbol}\n\n"
            f"Повторите инициализацию или создайте правила вручную."
        )

    coin_memory.save()
    return "INIT"


def _save_data_package_state(
    coin_memory: CoinMemory,
    symbol: str,
    data_package: Optional[Dict],
) -> None:
    """
    Сохраняет в память монеты данные, пришедшие вместе с пакетом:
    - индикаторы;
    - временные метки индикаторов;
    - market_stats;
    - ключевые события.
    """
    if not data_package:
        logger.info("[%s] Нет data_package для сохранения индикаторов", symbol)
        return

    data = data_package.get("data", {})

    indicators = data.get("indicators")
    indicators_timestamps = data.get("indicators_timestamps")

    if indicators:
        if isinstance(indicators, str):
            try:
                indicators = json.loads(indicators)
            except Exception:
                indicators = {}

        if indicators:
            coin_memory.update_state(
                {
                    "indicators": indicators,
                    "indicator_timestamps": indicators_timestamps or [],
                }
            )
            logger.info(
                "[%s] Индикаторы сохранены: %s индикаторов, %s временных меток",
                symbol,
                len(indicators),
                len(indicators_timestamps or []),
            )

    market_stats_raw = data.get("market_stats", "{}")
    if market_stats_raw and market_stats_raw != "{}":
        try:
            stats = (
                json.loads(market_stats_raw)
                if isinstance(market_stats_raw, str)
                else market_stats_raw
            )
            coin_memory.update_state({"market_stats": stats})
        except Exception:
            pass

    key_events = data.get("key_events", "")
    if key_events:
        old_events = coin_memory.get_last_key_events()
        coin_memory.add_key_events(key_events)
        new_events = coin_memory.get_last_key_events()

        if new_events != old_events:
            key_logger.info(
                "[%s] Ключевые события обновлены (уникальных строк: %s)",
                symbol,
                len(new_events.splitlines()),
            )
        else:
            logger.debug("[%s] Ключевые события не изменились", symbol)


def process_ai_response(
    response: Dict,
    coin_memory: CoinMemory,
    symbol: str,
    has_active_trade: bool = False,
    data_package: Optional[Dict] = None,
) -> str:
    """
    Главный обработчик ответа AI.

    Parameters
    ----------
    response:
        JSON-ответ модели.

    coin_memory:
        Память конкретной монеты.

    symbol:
        Символ монеты.

    has_active_trade:
        Флаг, указывающий, есть ли открытая позиция в момент обработки.

    data_package:
        Пакет рыночных данных, на основе которого AI сформировал ответ.

    Returns
    -------
    str:
        Значение поля action из ответа модели.
    """
    _ensure_safe_setup_fallback(
        response=response,
        symbol=symbol,
        has_active_trade=has_active_trade,
    )

    action = response.get("action", "ANALYSIS")

    if action == "INIT":
        return _process_init_response(
            response=response,
            coin_memory=coin_memory,
            symbol=symbol,
        )

    analysis = response.get("analysis")
    if analysis:
        if "timestamp_utc" not in analysis:
            analysis["timestamp_utc"] = datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
        coin_memory.set_last_analysis(analysis)

    if response.get("position_update") and has_active_trade:
        coin_memory.update_state({"last_position_update": response["position_update"]})
    elif not has_active_trade:
        coin_memory.update_state({"last_position_update": {}})

    trade_entry = response.get("trade_log_entry")
    if trade_entry and not has_active_trade:
        if trade_entry.get("direction") and trade_entry.get("entry_price"):
            coin_memory.add_trade_entry(trade_entry)
        coin_memory.set_active_trade(None)
        coin_memory.update_state({"last_position_update": {}})
        coin_memory.save()

    if response.get("new_rules"):
        coin_memory.update_rules(response["new_rules"])
        trading_list, risk_list = parse_rules_from_text(response["new_rules"])
        coin_memory.update_state(
            {
                "trading_rules_list": trading_list,
                "risk_management_rules_list": risk_list,
            }
        )

    if response.get("new_behavior_rules"):
        coin_memory.append_behavior_rule(response["new_behavior_rules"])

    if response.get("signal"):
        signal = response["signal"]
        coin_memory.update_state(
            {
                "last_signal": signal if signal.get("confidence_pct", 0) > 0 else None
            }
        )

    if response.get("setup") and not has_active_trade:
        coin_memory.update_state({"last_setup": response["setup"]})
    elif has_active_trade:
        coin_memory.update_state({"last_setup": None})

    if action == "CANCEL_SIGNAL":
        coin_memory.update_state({"last_signal": None})

    if analysis and analysis.get("summary"):
        coin_memory.set_morning_context(f"Утренний анализ: {analysis['summary']}")

    _save_data_package_state(
        coin_memory=coin_memory,
        symbol=symbol,
        data_package=data_package,
    )

    coin_memory.update_state({"last_update_utc": datetime.now(timezone.utc).isoformat()})
    coin_memory.save()

    return action