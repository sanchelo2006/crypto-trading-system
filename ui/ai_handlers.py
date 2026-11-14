"""
Processing AI model responses and applying them to coin memory.

This module creates no UI components and does not depend on NiceGUI pages.
It converts a model response into a CoinMemory state update.
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
    If the model returned neither setup, nor signal, nor position_update,
    create a safe fallback setup so the UI is not left without a data structure.

    Logic fully preserved from the previous nicegui_app.py.
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
            "[%s] Model response without setup/signal. Creating an empty setup for safety.",
            symbol,
        )
        response["setup"] = {
            "direction": "long",
            "confidence_pct": 0,
            "entry_price": 0.0,
            "stop_loss": 0.0,
            "take_profit": [],
            "conditions": [
                "The model produced no conditions — context overload or market uncertainty"
            ],
            "cancel_conditions": ["Manual control"],
            "rationale": "Automatic fallback: the model returned no setup.",
        }


def _process_init_response(
    response: Dict,
    coin_memory: CoinMemory,
    symbol: str,
) -> str:
    """
    Handles an INIT response:
    - saves research;
    - saves strategy_spec;
    - saves text rules and rule lists.
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
                    "[%s] strategy_spec saved (with engine defaults)",
                    symbol,
                )
            else:
                logger.warning(
                    "[%s] strategy_spec from AI is invalid: %s",
                    symbol,
                    err,
                )
        except Exception as e:
            logger.warning(
                "[%s] strategy_spec validation error: %s",
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

        rules_text = "## Trading Rules\n" + "\n".join(f"- {r}" for r in trading)
        if risk:
            rules_text += "\n\n## Risk Management\n" + "\n".join(f"- {r}" for r in risk)

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
            f"# Basic rules for {symbol}\n\n"
            f"Run initialization again or create the rules manually."
        )

    coin_memory.save()
    return "INIT"


def _save_data_package_state(
    coin_memory: CoinMemory,
    symbol: str,
    data_package: Optional[Dict],
) -> None:
    """
    Saves into coin memory the data that came with the package:
    - indicators;
    - indicator timestamps;
    - market_stats;
    - key events.
    """
    if not data_package:
        logger.info("[%s] No data_package to save indicators", symbol)
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
                "[%s] Indicators saved: %s indicators, %s timestamps",
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
                "[%s] Key events updated (unique lines: %s)",
                symbol,
                len(new_events.splitlines()),
            )
        else:
            logger.debug("[%s] Key events unchanged", symbol)


def process_ai_response(
    response: Dict,
    coin_memory: CoinMemory,
    symbol: str,
    has_active_trade: bool = False,
    data_package: Optional[Dict] = None,
) -> str:
    """
    Main AI response handler.

    Parameters
    ----------
    response:
        JSON response from the model.

    coin_memory:
        Memory of the specific coin.

    symbol:
        Coin symbol.

    has_active_trade:
        Flag indicating whether a position is open at processing time.

    data_package:
        Market data package the AI based its response on.

    Returns
    -------
    str:
        Value of the action field from the model response.
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
        coin_memory.set_morning_context(f"Morning analysis: {analysis['summary']}")

    _save_data_package_state(
        coin_memory=coin_memory,
        symbol=symbol,
        data_package=data_package,
    )

    coin_memory.update_state({"last_update_utc": datetime.now(timezone.utc).isoformat()})
    coin_memory.save()

    return action