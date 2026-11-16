"""Registry of tools (function calling) for the model."""
import logging
from typing import Dict, Any
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def get_tool_definitions() -> list:
    return [
        {"type": "function", "function": {"name": "get_current_rules", "description": "Get current trading rules.", "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {"name": "add_trading_rule", "description": "Add a new trading rule.", "parameters": {"type": "object", "properties": {"rule_text": {"type": "string", "description": "The text of the rule."}}, "required": ["rule_text"]}}},
        {"type": "function", "function": {"name": "update_trading_rules", "description": "Replace all trading rules.", "parameters": {"type": "object", "properties": {"rules_text": {"type": "string", "description": "The new text of the rules."}}, "required": ["rules_text"]}}},
        {"type": "function", "function": {"name": "get_current_setup", "description": "Get the current setup.", "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {"name": "update_setup", "description": "Update the setup (direction, levels, conditions).", "parameters": {"type": "object", "properties": {"direction": {"type": "string", "enum": ["long", "short"]}, "entry_price": {"type": "number"}, "stop_loss": {"type": "number"}, "take_profit": {"type": "array", "items": {"type": "number"}}, "conditions": {"type": "array", "items": {"type": "string"}}, "cancel_conditions": {"type": "array", "items": {"type": "string"}}, "rationale": {"type": "string"}}, "required": ["direction"]}}},
        {"type": "function", "function": {"name": "get_market_analysis", "description": "Get the latest market analysis.", "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {"name": "get_active_trade", "description": "Get information about the open position.", "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {"name": "get_trade_stats", "description": "Get trade statistics.", "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {"name": "get_news", "description": "Get fresh news for the coin.", "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {"name": "get_behavior_rules", "description": "Get behavior rules.", "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {"name": "add_behavior_rule", "description": "Add a behavior rule.", "parameters": {"type": "object", "properties": {"rule_text": {"type": "string", "description": "The text of the behavior rule."}}, "required": ["rule_text"]}}},
        {"type": "function", "function": {"name": "get_news_analysis", "description": "Get extended news analysis (formatted text).", "parameters": {"type": "object", "properties": {}, "required": []}}},
    ]


class ToolExecutor:
    def __init__(self, coin_memory, fetcher=None, symbol: str = ""):
        self.cm = coin_memory
        self.fetcher = fetcher
        self.symbol = symbol
        # News cache for the duration of a single chat call
        self._news_cache_local = None
        # Cache for extended analysis (to avoid re-computing sentiment)
        self._news_analysis_cache = None

    def execute(self, func_name: str, arguments: Dict[str, Any]) -> str:
        logger.info(f"[{self.symbol}] Tool call: {func_name}")
        handlers = {
            "get_current_rules": lambda a: f"Current rules:\n\n{self.cm.get_rules()}" if self.cm.get_rules() else "No rules have been created.",
            "add_trading_rule": self._add_rule,
            "update_trading_rules": self._update_rules,
            "get_current_setup": self._get_setup,
            "update_setup": self._update_setup,
            "get_market_analysis": self._get_analysis,
            "get_active_trade": self._get_trade,
            "get_trade_stats": self._get_stats,
            "get_news": self._get_news,
            "get_behavior_rules": lambda a: self.cm.get_behavior_rules() or "No behavior rules.",
            "add_behavior_rule": self._add_behavior,
            "get_news_analysis": self._get_news_analysis,
        }
        handler = handlers.get(func_name)
        if not handler: return f"Function {func_name} not found."
        try: return handler(arguments)
        except Exception as e: return f"Error in {func_name}: {e}"

    def _add_rule(self, a):
        rule_text = a.get("rule_text", "").strip()
        if not rule_text: return "Empty rule text."
        self.cm.append_rule(rule_text)
        current_list = self.cm.get_state().get("trading_rules_list", [])
        current_list.append(rule_text)
        self.cm.update_state({"trading_rules_list": current_list})
        return "Rule added."
        
    def _update_rules(self, a):
        rules_text = a.get("rules_text", "").strip()
        if not rules_text: return "Empty rules text."
        self.cm.update_rules(rules_text)
        self.cm.update_state({"trading_rules_list": [], "risk_management_rules_list": []})
        return "Rules updated."
        
    def _add_behavior(self, a):
        rule_text = a.get("rule_text", "").strip()
        if not rule_text: return "Empty text."
        self.cm.append_behavior_rule(rule_text)
        return "Behavior rule added."

    def _get_setup(self, a):
        s = self.cm.get_state()
        setup = s.get("last_setup")
        signal = s.get("last_signal")
        if signal and signal.get("confidence_pct", 0) > 0:
            return f"Signal: {signal.get('direction','')} (confidence {signal.get('confidence_pct',0)}%)"
        if setup:
            return (f"Setup: {setup.get('direction','')}\nEntry: {setup.get('entry_price','?')}\n"
                    f"SL: {setup.get('stop_loss','?')} TP: {setup.get('take_profit',[])}\n"
                    f"Conditions: {setup.get('conditions',[])}\nCancel: {setup.get('cancel_conditions',[])}")
        return "No setup."

    def _update_setup(self, a):
        new_setup = {
            "direction": a.get("direction"), "entry_price": a.get("entry_price"),
            "stop_loss": a.get("stop_loss"), "take_profit": a.get("take_profit", []),
            "conditions": a.get("conditions", []), "cancel_conditions": a.get("cancel_conditions", []),
            "rationale": a.get("rationale", ""), "confidence_pct": 0
        }
        self.cm.update_state({"last_setup": new_setup})
        return f"Setup updated: {new_setup['direction']}."

    def _get_analysis(self, a):
        an = self.cm.get_last_analysis()
        if not an: return "No analysis performed."
        return (f"Price: {an.get('current_price','?')}, 24h change: {an.get('price_change_24h_pct','?')}%\n"
                f"Bias: {an.get('orderbook_bias','?')}, OI: {an.get('oi_trend','?')}\n"
                f"Funding: {an.get('funding_rate_status','?')}, F&G: {an.get('fear_greed_state','?')}\n"
                f"News: {an.get('news_impact','?')}\nSummary: {an.get('summary','')}")

    def _get_trade(self, a):
        t = self.cm.get_active_trade()
        if not t: return "No open positions."
        return f"Position: {t.get('direction','?')}, Entry: {t.get('entry_price','?')}, SL: {t.get('stop_loss','?')}, TP: {t.get('take_profit',[])}"

    def _get_stats(self, a):
        s = self.cm.get_trade_stats()
        return (f"Trades: {s['total_trades']}, Win Rate: {s['win_rate']}%\n"
                f"P&L: {s['total_pnl_pct']:+.2f}%, Avg Win: {s['avg_win_pct']}%, Avg Loss: {s['avg_loss_pct']}%")

    def _get_news(self, a):
        """Returns news, using the cache if available."""
        if not self.fetcher: return "No access to news."
        try:
            # If cache exists, return it
            if self._news_cache_local is not None:
                logger.info(f"[{self.symbol}] Returning cached news")
                return self._news_cache_local

            # Otherwise load news
            df = self.fetcher.news_fetcher.get_news_for_symbol(self.symbol, max_items=10, hours_back=24)
            result = self.fetcher.news_fetcher.format_news_for_prompt(df)
            # Save to cache
            self._news_cache_local = result
            return result
        except Exception as e:
            return f"Error loading news: {e}"

    def _get_news_analysis(self, a):
        """Returns extended news analysis, using the cache for formatted text."""
        if not self.fetcher: return "No access to news."
        try:
            # If analysis cache exists, return it
            if self._news_analysis_cache is not None:
                logger.info(f"[{self.symbol}] Returning cached news analysis")
                return self._news_analysis_cache

            # Load data
            df = self.fetcher.news_fetcher.get_news_for_symbol(self.symbol, max_items=15, hours_back=24)
            if df.empty:
                result = "No news found."
                self._news_analysis_cache = result
                self._news_cache_local = result  # also save for get_news
                return result

            sentiment = self.fetcher.news_fetcher.assess_news_sentiment(df)
            news_text = self.fetcher.news_fetcher.format_news_for_prompt(df)
            result = f"=== EXTENDED NEWS ANALYSIS ===\nSentiment: {sentiment['impact']}\nPositive: {sentiment['positive_count']}, Negative: {sentiment['negative_count']}\n\n{news_text}"
            # Save to caches
            self._news_analysis_cache = result
            self._news_cache_local = news_text  # so get_news can use it
            return result
        except Exception as e:
            return f"Error: {e}"