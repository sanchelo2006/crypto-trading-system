from __future__ import annotations

import sys
import os

from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
from unittest.mock import MagicMock

import pytest

import prompts.init_prompt as init_prompt
import prompts.optimize_prompt as optimize_prompt
import prompts.system_prompt as system_prompt
from ui.ai_handlers import process_ai_response
from ui.trade_helpers import parse_rules_from_text
from tool_registry import get_tool_definitions, ToolExecutor
from utils.helpers import safe_parse_json

def test_prompts_contain_1h_series_names():
    text = "\n".join([
    getattr(init_prompt, "INIT_PROMPT", ""),
    getattr(optimize_prompt, "OPTIMIZE_PROMPT", ""),
    getattr(system_prompt, "SYSTEM_PROMPT", ""),
])

    required_tokens = [
        "sma_20_1h",
        "sma_50_1h",
        "rsi_14_1h",
        "atr_14_1h",
        "bb_upper_1h",
        "bb_middle_1h",
        "bb_lower_1h",
        "macd_1h",
        "macd_signal_1h",
        "macd_hist_1h",
        "adx_14_1h",
        "vwap_1h",
        "pivot_1h",
        "resistance_1_1h",
        "support_1_1h",
        "avg_volume_20_1h",
        "avg_volume_96_1h",
    ]

    lower_text = text.lower()
    for token in required_tokens:
        assert token in lower_text, f"Не найден токен {token} в промптах"


def test_prompts_allow_higher_timeframe_logic():
    text = "\n".join([
    getattr(init_prompt, "INIT_PROMPT", ""),
    getattr(optimize_prompt, "OPTIMIZE_PROMPT", ""),
    getattr(system_prompt, "SYSTEM_PROMPT", ""),
])

    assert "старшего таймфрейма" in text
    assert "1h" in text
    assert "фильтрац" in text or "фильтр" in text


def test_parse_rules_from_text_splits_sections():
    src = """
## Правила торговли
- Входить только по подтверждённому пробою
- Избегать позднего входа

## Риск-менеджмент
- Риск на сделку не более 1%
- Не усреднять убыток
"""
    trading, risk = parse_rules_from_text(src)

    assert trading == [
        "Входить только по подтверждённому пробою",
        "Избегать позднего входа",
    ]
    assert risk == [
        "Риск на сделку не более 1%",
        "Не усреднять убыток",
    ]


def test_parse_rules_from_text_fallback_works():
    src = """
- Входить по тренду
- Не гнаться за импульсом
"""
    trading, risk = parse_rules_from_text(src)

    assert trading == ["Входить по тренду", "Не гнаться за импульсом"]
    assert risk == []


def test_safe_parse_json_parses_valid_json():
    payload = {"action": "INIT", "strategy_spec": {"long": {}, "short": {}}}
    result = safe_parse_json(json.dumps(payload, ensure_ascii=False))
    assert result == payload


def test_tool_registry_has_expected_tools():
    tools = get_tool_definitions()
    names = {t["function"]["name"] for t in tools}

    assert "get_current_rules" in names
    assert "update_setup" in names
    assert "get_market_analysis" in names
    assert "get_news_analysis" in names


def test_tool_executor_update_setup_works():
    cm = MagicMock()
    cm.get_rules.return_value = ""
    cm.get_state.return_value = {"trading_rules_list": [], "risk_management_rules_list": []}

    executor = ToolExecutor(cm, fetcher=None, symbol="BTCUSDT")
    result = executor.execute(
        "update_setup",
        {
            "direction": "long",
            "entry_price": 100.0,
            "stop_loss": 98.0,
            "take_profit": [104.0],
            "conditions": ["close above level"],
            "cancel_conditions": ["close below level"],
            "rationale": "test",
        },
    )

    assert "Сетап обновлён" in result
    cm.update_state.assert_called()


def test_process_ai_response_init_saves_rules_and_strategy_spec():
    cm = MagicMock()
    cm.get_last_key_events.return_value = ""
    cm.get_rules.return_value = ""
    cm.get_behavior_rules.return_value = ""
    cm.get_state.return_value = {}

    response = {
        "action": "INIT",
        "research": {"project_name": "Test", "sector": "L1"},
        "strategy_spec": {
            "long": {"entry": [], "stop_loss_pct": 2.0, "take_profit_pct": [4.0]},
            "short": {"entry": [], "stop_loss_pct": 2.0, "take_profit_pct": [4.0]},
            "filters": [],
        },
        "trading_rules_list": ["Правило 1"],
        "risk_management_rules_list": ["Риск 1"],
    }

    process_ai_response(response, cm, "ETHUSDT")

    assert cm.update_state.called
    assert cm.update_rules.called
    assert cm.save.called


def test_process_ai_response_adds_timestamp_to_analysis():
    cm = MagicMock()
    cm.get_last_key_events.return_value = ""
    cm.get_rules.return_value = ""
    cm.get_behavior_rules.return_value = ""
    cm.get_state.return_value = {}

    response = {
        "action": "ANALYSIS",
        "analysis": {
            "current_price": 100.0,
            "summary": "Тестовый анализ",
        },
        "setup": {
            "direction": "long",
            "confidence_pct": 10,
            "entry_price": 101.0,
            "stop_loss": 99.0,
            "take_profit": [103.0],
            "conditions": [],
            "cancel_conditions": [],
            "rationale": "test",
        },
    }

    process_ai_response(response, cm, "XRPUSDT")

    assert cm.set_last_analysis.called
    analysis_arg = cm.set_last_analysis.call_args[0][0]
    assert "timestamp_utc" in analysis_arg


def test_process_ai_response_fallback_setup_added_when_missing():
    cm = MagicMock()
    cm.get_last_key_events.return_value = ""
    cm.get_rules.return_value = ""
    cm.get_behavior_rules.return_value = ""
    cm.get_state.return_value = {}

    response = {
        "action": "ANALYSIS",
        "analysis": {"summary": "без сетапа"},
    }

    process_ai_response(response, cm, "SOLUSDT", has_active_trade=False)

    assert "setup" in response
    assert response["setup"]["direction"] == "long"
    assert response["setup"]["confidence_pct"] == 0


def test_prompts_do_not_forbid_1h():
    all_text = "\n".join([
        getattr(init_prompt, "INIT_PROMPT", ""),
        getattr(optimize_prompt, "OPTIMIZE_PROMPT", ""),
        getattr(system_prompt, "SYSTEM_PROMPT", ""),
    ])

    forbidden_markers = [
        "запрещены 1h",
        "не использовать 1h",
        "только 15m",
    ]

    for marker in forbidden_markers:
        assert marker not in all_text