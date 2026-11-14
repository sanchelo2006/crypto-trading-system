"""
Client for the DeepSeek API.
Includes: adaptive ADX threshold (from market_stats), timeframe validation,
support for the lightweight intraday package (intraday_light).
"""
import json
import logging
import time
import threading
import re
from typing import Optional, Dict, Any, Callable

from openai import OpenAI

import config

# Not used for now — replaced by a new function based on a new library
#from utils.helpers import safe_parse_json
from prompts.system_prompt import SYSTEM_PROMPT
from prompts.init_prompt import INIT_PROMPT

REVISE_INIT_PROMPT = """You are a professional crypto trader and trading-system architect.

### TASK
A strategy was created during coin initialization. The system then ran it on
~60 days of history and returned numeric diagnostics. Rebuild strategy_spec,
fixing ONLY the diagnosed defects.

### HARD RULES (must not be violated)
1. NEVER REMOVE REQUIRED FILTERS. For EVERY entry group of EVERY enabled side:
   - a volume condition referencing `volume`, `avg_volume_96`, `avg_volume_20`
     or `activity`;
   - a trend condition referencing `sma_50`, `sma_50_1h`, `sma_20_1h`, `adx_14`,
     `adx_14_1h`, `macd`, `macd_signal`, `macd_hist`, `macd_1h`,
     `macd_signal_1h`, `macd_hist_1h`, `low_32`, `high_32` or `drop_2_atr`;
   - a level/momentum condition referencing `resistance_1`, `support_1`,
     `pivot`, `close`, `rsi_14`, `macd_hist` or `drop_2_atr`;
   - a stop-loss (`stop_loss_atr_mult` or `stop_loss_pct`);
   - a take-profit (`take_profit_atr_mult` or `take_profit_pct`).
     `take_profit_pct` must be a list with exactly ONE number.
2. YOU MAY CHANGE:
   - numeric thresholds (RSI ranges, volume multipliers, ATR multipliers);
   - add a second OR group via `entry_any`;
   - `management` parameters (breakeven, trailing, max_hold_candles);
   - `risk_controls` parameters (cooldown).
3. STRUCTURAL LIMITS:
   - exactly 4 conditions per entry group (trend + momentum + volume + level);
   - at most 2 OR groups per side (`entry_any` has at most 2 elements);
   - stop-loss: 2.5-4.0 x ATR(15m), outside noise;
   - take-profit: at least 2 x the stop (reward/risk >= 2:1);
   - `filters` (global): at most 1 soft condition, or [].
4. Series names are NOT validated by the engine. A misspelled name silently
   evaluates to False forever and produces zero trades. Use only real names.

### SANITY TARGETS
- at least 1.5 trades per week (aim for 2-5);
- profit factor >= 1.0;
- share of quick losses (closed within 3 candles) <= 50%.

### EXAMPLES OF CORRECT FIXES
If frequency is too low:
- widen the RSI range: "min": 40, "max": 70 instead of "min": 52, "max": 60;
- lower the volume threshold: "mult": 0.8 instead of "mult": 1.3;
- add a second OR group in `entry_any` (level retest or pullback to sma_20).
If win rate is below the breakeven threshold:
- raise `take_profit_atr_mult` to 5.0-8.0 (increase payoff);
- raise `stop_loss_atr_mult` to 3.0-4.0 (stop outside noise);
- drop entries with little room to move (too close to the level).
If there are many quick losses (<= 3 candles):
- enter on the retest, not on the breakout:
  "low" <= "resistance_1" instead of "close" > "resistance_1";
- widen the stop: stop_loss_atr_mult 3.0 instead of 1.5;
- add a regime filter: "adx_14_1h" > 20 (trend only).

### EXECUTION REALITY
Signals are evaluated on the last CLOSED 15m candle and executed at the OPEN
of the NEXT candle. Round-trip cost is ~0.16-0.21%; targets below ~0.3% are
noise after costs.

### RESPONSE FORMAT (strict JSON, these keys)
{
  "strategy_spec": {
    "long": {
      "entry_any": [
        [
          {"type": "compare", "left": "close", "op": ">", "right": "sma_50_1h"},
          {"type": "compare", "left": "macd_hist", "op": ">", "right": 0},
          {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 1.0},
          {"type": "compare", "left": "low", "op": "<=", "right": {"name": "resistance_1", "mult": 1.003}}
        ],
        [
          {"type": "compare", "left": "close", "op": ">", "right": "sma_50_1h"},
          {"type": "compare", "left": "macd_hist", "op": ">", "right": 0},
          {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 0.8},
          {"type": "compare", "left": "low", "op": "<=", "right": "sma_20"}
        ]
      ],
      "stop_loss_atr_mult": 3.0,
      "take_profit_atr_mult": 6.0,
      "stop_loss_pct": 2.0,
      "take_profit_pct": [4.0],
      "management": {
        "move_to_breakeven_at_tp_pct": 0.5,
        "breakeven_offset_pct": 0.1,
        "trailing_atr_mult": null,
        "max_hold_candles": 120,
        "exit_on_level_failure": false,
        "level_field": "resistance_1",
        "level_failure_buffer_pct": 1.0,
        "exit_on_trend_loss": false,
        "exit_on_macd_flip": false
      }
    },
    "short": {
      "entry": [
        {"type": "compare", "left": "close", "op": "<", "right": "sma_50_1h"},
        {"type": "compare", "left": "macd_hist", "op": "<", "right": 0},
        {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 1.0},
        {"type": "compare", "left": "close", "op": "<", "right": "support_1"}
      ],
      "stop_loss_atr_mult": 3.0,
      "take_profit_atr_mult": 6.0,
      "stop_loss_pct": 2.0,
      "take_profit_pct": [4.0],
      "management": {
        "move_to_breakeven_at_tp_pct": 0.5,
        "breakeven_offset_pct": 0.1,
        "trailing_atr_mult": null,
        "max_hold_candles": 120,
        "exit_on_level_failure": false,
        "level_field": "support_1",
        "level_failure_buffer_pct": 1.0,
        "exit_on_trend_loss": false,
        "exit_on_macd_flip": false
      }
    },
    "filters": [],
    "risk_controls": {
      "enabled": true,
      "max_consecutive_losses": 4,
      "pause_candles_after_losses": 32,
      "max_losses_in_window": 6,
      "loss_window_trades": 10
    }
  },
  "trading_rules_list": ["", ""],
  "risk_management_rules_list": ["", ""],
  "new_rules": "",
  "changes_summary": ""
}

### CRITICAL
- Answer in English ONLY.
- NEVER remove the volume filter — it is mandatory.
- NEVER add comments to JSON, no trailing commas, no markdown fences.
- Return COMPLETE valid JSON parseable by json.loads() directly.
- Double quotes only, for keys and string values.
- Fill BOTH sides completely; never leave "..." placeholders.
- Every text rule must name the exact series and threshold used in the spec.
- Keep 3-8 items per rules list, each <= 160 characters.
- `changes_summary` <= 150 characters: what changed and why.
"""

from prompts.optimize_prompt import OPTIMIZE_PROMPT
from web_search import WebSearcher

try:
    from json_repair import repair_json
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False

logger = logging.getLogger(__name__)
token_logger = logging.getLogger("ai_client.token")
message_logger = logging.getLogger("ai_client.messages")

_last_logged_messages = {}
_LAST_LOG_TIMEOUT = 2.0
_log_lock = threading.Lock()


def _safe_json_loads(text: str) -> Optional[Dict]:
    """
    Safe JSON parsing with aggressive pre-processing and json_repair.
    """
    # 1. Strip markdown blocks ```json ... ```
    text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```\s*', '', text)

    # 2. Strip single-line comments //
    text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)

    # 3. Strip multi-line comments /* */
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

    # 4. Strip trailing commas (aggressively, several passes)
    for _ in range(3):
        text = re.sub(r',\s*([}\]])', r'\1', text)

    # 5. Fix empty strings in numeric fields: "" -> null
    text = re.sub(r':\s*""\s*([,}])', r': null\1', text)

    # 6. Convert single quotes to double quotes (if the model used them)
    text = re.sub(r"'([^']*)'(?=\s*[:},])", r'"\1"', text)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        if HAS_JSON_REPAIR:
            try:
                repaired = repair_json(text)
                logger.info(f"JSON repair succeeded: processed {len(text)} characters")
                return json.loads(repaired)
            except Exception as repair_error:
                logger.error(f"JSON repair failed: {repair_error}")
                # Last attempt: find the first { and the last }
                try:
                    start = text.find('{')
                    end = text.rfind('}')
                    if start >= 0 and end > start:
                        extracted = text[start:end+1]
                        return json.loads(extracted)
                except Exception:
                    pass
                return None
        logger.error(f"JSON parse error (json_repair unavailable): {e}")
        return None


class DeepSeekClient:
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or config.DEEPSEEK_API_KEY
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set.")
        self.model = model or config.DEEPSEEK_MODEL
        self.client = OpenAI(api_key=self.api_key, base_url=config.DEEPSEEK_BASE_URL)
        self.conversation_history: Dict[str, list] = {}
        self.web_searcher = WebSearcher()
        logger.info(f"DeepSeekClient created, model: {self.model}")

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def _log_data_sizes(self, data: dict, label: str = ""):
        for field in ["prices_1h_csv", "prices_15m_csv", "fundamentals_csv", "orderbook_bids_csv", "orderbook_asks_csv", "trades_summary", "news_text", "market_stats"]:
            content = data.get(field, "")
            if content:
                tokens = self._estimate_tokens(content)
                token_logger.info(f"{label} {field}: ~{tokens} tokens ({len(content)} chars)")

    def _log_message_summary(self, data_package: Dict, additional_info: str = ""):
        msg_type = data_package.get("message_type", "unknown")
        pair = data_package.get("pair", "UNKNOWN")
        ts = data_package.get("timestamp_utc", "")
        key = f"{pair}_{msg_type}_{additional_info}_{ts}"
        now = time.time()
        with _log_lock:
            if key in _last_logged_messages and (now - _last_logged_messages[key]) < _LAST_LOG_TIMEOUT:
                return
            _last_logged_messages[key] = now

        data = data_package.get("data", {})
        parts = [f"📩 [{msg_type}] {pair} {additional_info}"]
        parts.append("  - System prompt (included)")
        for item in ["indicators", "prices_1h_csv", "prices_15m_csv", "fundamentals_csv", "orderbook_bids_csv", "orderbook_asks_csv", "trades_summary", "news_text", "market_stats"]:
            if data.get(item):
                parts.append(f"  - {item}")
        if data.get("prices_1h_count"):
            parts.append(f"  - 1h candles: {data['prices_1h_count']}")
        if data.get("prices_15m_count"):
            parts.append(f"  - 15m candles: {data['prices_15m_count']}")
        if data_package.get("morning_context"):
            parts.append("  - Morning context (included)")
        if data_package.get("intraday_context"):
            parts.append("  - Intraday context (included)")
        message_logger.info("\n".join(parts))

    def _get_init_tools(self, symbol: str) -> list:
        return [{"type": "function", "function": {"name": "web_search", "description": f"Search the internet. Use it to find trading rules, risk-management practices and specifics of {symbol}.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}}, "required": ["query"]}}}]

    def _build_message(self, data_package: Dict, trading_rules: str = "", active_trade: Optional[Dict] = None, trade_closed: Optional[Dict] = None, is_initial: bool = False, morning_context: str = "", intraday_context: str = "", behavior_rules: str = "", error_analysis: bool = False) -> str:
        parts = []
        msg_type = data_package.get("message_type", "unknown")
        pair = data_package.get("pair", "UNKNOWN")
        ts = data_package.get("timestamp_utc", "")
        data = data_package.get("data", {})

        if msg_type == "intraday_light":
            parts.append("=== SETUP CHECK ===")
            parts.append(f"Pair: {pair} | Time (UTC): {ts}")
            parts.append("This is a lightweight package for checking the setup/position. Only fresh candles and updated metrics are provided.")
        else:
            parts.append(f"=== {'ERROR ANALYSIS' if error_analysis else 'ANALYSIS REQUEST'} ===")
            parts.append(f"Pair: {pair} | Time (UTC): {ts} | Type: {msg_type}")
            if error_analysis:
                parts.append("\nThis is an analysis of a closed trade. Diagnose the mistakes and propose changes to the rules.")

        self._log_data_sizes(data, f"[{pair}] {msg_type}")
        data_package['morning_context'] = morning_context
        data_package['intraday_context'] = intraday_context
        self._log_message_summary(data_package, additional_info=f"{'error_analysis' if error_analysis else ''}")

        if data.get("news_text"):
            parts.append("\n--- NEWS ---")
            parts.append(data["news_text"])
        if morning_context:
            parts.append("\n--- MORNING CONTEXT ---")
            parts.append(morning_context)
        if intraday_context:
            parts.append("\n--- PREVIOUS CONCLUSIONS ---")
            parts.append(intraday_context)
        if trading_rules:
            parts.append("\n--- TRADING RULES ---")
            parts.append(trading_rules)
        if behavior_rules:
            parts.append("\n--- BEHAVIOR RULES ---")
            parts.append(behavior_rules)
        if active_trade:
            parts.append("\n--- ACTIVE POSITION ---")
            parts.append(json.dumps(active_trade, ensure_ascii=False))
        if trade_closed:
            parts.append("\n--- TRADE CLOSED ---")
            parts.append(json.dumps(trade_closed, ensure_ascii=False))
        if data.get("indicators"):
            parts.append("\n--- INDICATORS (JSON) ---")
            parts.append(data["indicators"])
        if data.get("market_stats") and data["market_stats"] != "{}":
            parts.append("\n--- PRECOMPUTED METRICS (JSON) ---")
            parts.append(data["market_stats"])

        if msg_type != "intraday_light":
            parts.append("\n--- IMPORTANT: RESPONSE FORMAT ---")
            parts.append("You MUST return JSON with ALL fields. Even in an uncertain market always include 'setup' with conditions and cancel_conditions. For volumes use the ready numbers from market_stats, do NOT parse CSV.")

        parts.append("\n--- MARKET DATA ---")
        for field, label in [("prices_1h_csv", "1h candles"), ("prices_15m_csv", "15m candles"), ("fundamentals_csv", "Fundamentals"), ("orderbook_bids_csv", "Bids"), ("orderbook_asks_csv", "Asks")]:
            if data.get(field):
                parts.append(f"\n{label}:")
                parts.append(data[field])

        if data.get("trades_summary") and data["trades_summary"] != "{}":
            parts.append("\n--- TRADES (aggregated) ---")
            parts.append(data["trades_summary"])

        return "\n".join(parts)

    def _build_init_message(self, data_package: Dict, symbol: str) -> str:
        msg_type = data_package.get("message_type", "unknown")
        ts = data_package.get("timestamp_utc", "")
        data = data_package.get("data", {})
        base_asset = symbol.replace("USDT", "").replace("USDC", "").replace("BUSD", "")

        parts = ["=== COIN INITIALIZATION ===", f"Pair: {symbol} (base asset: {base_asset}) | Time (UTC): {ts}", f"\nBuild trading rules and risk-management rules for {symbol}.", "You have access to the web_search tool.", "Rules must be in English, without specific price values."]

        self._log_data_sizes(data, f"[{symbol}] INIT")
        self._log_message_summary(data_package, additional_info="INIT")

        if data.get("news_text"):
            parts.append("\n--- NEWS ---")
            parts.append(data["news_text"])
        if data.get("indicators"):
            parts.append("\n--- INDICATORS (JSON) ---")
            parts.append(data["indicators"])

        parts.append("\n--- MARKET DATA (CSV) ---")
        for field, label in [("prices_1h_csv", "1h candles"), ("prices_15m_csv", "15m candles"), ("fundamentals_csv", "Fundamentals"), ("orderbook_bids_csv", "Bids"), ("orderbook_asks_csv", "Asks")]:
            if data.get(field):
                parts.append(f"\n{label}:")
                parts.append(data[field])

        return "\n".join(parts)

    def init_symbol(self, data_package: Dict, symbol: str, model: str = None) -> Optional[Dict[str, Any]]:
        use_model = model or self.model
        user_message = self._build_init_message(data_package, symbol)
        init_prompt = INIT_PROMPT.replace("{symbol}", symbol)
        base_asset = symbol.replace("USDT", "").replace("USDC", "").replace("BUSD", "")

        messages = [{"role": "system", "content": init_prompt}, {"role": "user", "content": user_message}]
        tools = self._get_init_tools(symbol)

        for iteration in range(5):
            if iteration == 2:
                messages.append({"role": "user", "content": "You have gathered enough information. Now aggregate all data and produce the JSON with the rules. If you need ONE MORE search query for clarification, do it now, but on the next iteration you MUST return the finished JSON without calling tools."})
            elif iteration >= 3:
                messages.append({"role": "user", "content": "STOP SEARCHING. Return the JSON with the aggregated rules right now. No tool_calls. JSON only."})

            kwargs = {"model": use_model, "messages": messages, "max_tokens": 8000, "temperature": 0.5, "tools": tools if iteration < 3 else [], "tool_choice": "auto" if iteration < 3 else "none"}

            try:
                total_chars = sum(len(m["content"]) for m in messages)
                logger.info(f"Initializing {symbol} (iteration {iteration+1}): ~{total_chars} characters")
                response = self.client.chat.completions.create(**kwargs)

                if not response.choices:
                    return {"action": "INIT", "analysis": {"summary": "Empty API response", "current_price": 0}}

                msg = response.choices[0].message

                if msg.content and '<｜｜DSML｜｜tool_calls>' in msg.content:
                    logger.warning(f"[{symbol}] The model tried to call a tool after tools were disabled (iteration {iteration+1}). Returning fallback.")
                    return {"action": "INIT", "analysis": {"summary": "The model failed to finish initialization — it attempted a tool call after tools were disabled.", "current_price": 0}}

                if msg.content and len(msg.content.strip()) > 100:
                    result = _safe_json_loads(msg.content)
                    if result is not None and (result.get("trading_rules_list") or result.get("new_rules")):
                        logger.info(f"[{symbol}] INIT: the model returned finished rules (iteration {iteration+1})")
                        if "trading_rules_list" in result and "new_rules" not in result:
                            trading = "\n".join(result.get("trading_rules_list", []))
                            risk = "\n".join(result.get("risk_management_rules_list", []))
                            result["new_rules"] = trading + "\n\n" + risk
                        return result

                if msg.tool_calls:
                    logger.info(f"[{symbol}] INIT: the model called {len(msg.tool_calls)} tools (iteration {iteration+1})")
                    messages.append({"role": "assistant", "content": msg.content, "tool_calls": [{"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in msg.tool_calls]})
                    for tc in msg.tool_calls:
                        func_name = tc.function.name
                        try:
                            args = json.loads(tc.function.arguments or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        if func_name == "web_search":
                            query = args.get("query", "")
                            logger.info(f"[{symbol}] Search: {query}")
                            result = self.web_searcher.search_formatted(query, max_results=5)
                        else:
                            result = f"Unknown tool: {func_name}"
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                        logger.info(f"[{symbol}] Search result ({len(result)} characters)")
                    continue

                content = msg.content
                logger.info(f"Initialization response: {len(content) if content else 0} characters")
                if not content or len(content.strip()) == 0:
                    return {"action": "INIT", "analysis": {"summary": "Empty model response", "current_price": 0}}

                result = _safe_json_loads(content)
                if result is None:
                    logger.error(f"Failed to parse JSON from the INIT response. Raw response: {content[:500]}...")
                    return {"action": "INIT", "analysis": {"summary": "Failed to parse JSON of the model response.", "current_price": 0}}

                if "trading_rules_list" in result and "new_rules" not in result:
                    trading = "\n".join(result.get("trading_rules_list", []))
                    risk = "\n".join(result.get("risk_management_rules_list", []))
                    result["new_rules"] = trading + "\n\n" + risk
                return result

            except Exception as e:
                logger.error(f"Initialization error for {symbol} (iteration {iteration+1}): {e}")
                return {"action": "INIT", "analysis": {"summary": f"Error: {str(e)[:200]}", "current_price": 0}}

        return {"action": "INIT", "analysis": {"summary": "Initialization iteration limit reached.", "current_price": 0}}

    def _build_optimize_message(
        self,
        analysis_summary: Dict,
        current_spec: Dict,
        current_rules: str,
        rejected_specs: Optional[list] = None,  # Kept for compatibility; now carries rejected_changes
        extra_instruction: str = "",
    ) -> str:
        """
        Message for optimization. Compressed to save tokens.
        """
        # 1. HARD TRUNCATION OF TEXT RULES (they duplicate the spec and eat tokens)
        rules_text = (current_rules or "none").strip()
        if len(rules_text) > 800:
            rules_text = rules_text[:800] + "\n... [rules text truncated to save tokens, see strategy_spec]"

        parts = [
            "=== STRATEGY OPTIMIZATION ===",
            "--- CURRENT strategy_spec ---",
            json.dumps(current_spec, ensure_ascii=False),
            "",
            "--- CURRENT RULES (abridged) ---",
            rules_text,
            "",
            "--- BACKTEST RESULTS ---",
            json.dumps(analysis_summary, ensure_ascii=False),
            "",
            "--- GOAL ---",
            "Maximize TEST P&L (expectancy x number of trades). "
            "PF > 1.0 on TRAIN and TEST, expectancy > 0 on both. "
            "Look at 'loss_profile' and 'win_profile' in trade_summary: compare average "
            "indicators (RSI, ADX, volume) on losing and winning entries. "
            "If losses happen at low ADX or high RSI, add filters. "
            "Do not repeat changes from the 'rejected_changes' block."
        ]

        # 2. Instead of raw JSON of rejected specs we pass textual descriptions
        rejected_changes = analysis_summary.get("rejected_changes", [])
        if rejected_changes:
            parts.append("")
            parts.append("--- RECENT FAILED ATTEMPTS (DO NOT REPEAT) ---")
            for rc in rejected_changes:
                parts.append(f"- {rc}")

        # --- STEP 3: Dynamic injection of hard constraints for the AI ---
        test_stats = analysis_summary.get("test_stats", {})
        test_pf = float(test_stats.get("profit_factor", 0) or 0)
        test_trades = int(test_stats.get("total_trades", 0) or 0)

        ai_restrictions = []

        # 1. If the strategy is already profitable, forbid changing the core
        if test_pf > 1.1:
            ai_restrictions.append(
                "⚠️ CRITICAL: the current base is profitable (PF > 1.1). "
                "It is STRICTLY FORBIDDEN to change the entry logic ('entry' and 'entry_any'). "
                "You may change ONLY 'filters', 'management' and ATR multipliers "
                "(stop_loss_atr_mult, take_profit_atr_mult)."
            )

        # 2. If there are too few trades, require entry_any instead of extra conditions
        if test_trades < 30:
            ai_restrictions.append(
                f"⚠️ TOO FEW TRADES ({test_trades} < 30). "
                "To raise frequency you MUST use the 'entry_any' block and add "
                "at least 2 DIFFERENT logical groups (e.g. one for breakout, one for pullback). "
                "Do NOT add new conditions to the existing 'entry' — that kills frequency."
            )

        if test_pf > 1.1 and test_trades < 30:
            ai_restrictions.append(
                "⚠️ PRECEDENCE: both restrictions above apply, and the ENTRY FREEZE WINS. "
                "Do not touch 'entry'/'entry_any'. Raise frequency only through softer "
                "global filters, wider exits, a shorter time-stop or a shorter cooldown."
            )

        if ai_restrictions:
            dynamic_instruction = "\n".join(ai_restrictions)
            if extra_instruction:
                extra_instruction = dynamic_instruction + "\n" + extra_instruction
            else:
                extra_instruction = dynamic_instruction

        if extra_instruction:
            parts.append("")
            parts.append(f"--- ADDITIONAL REQUIREMENT ---\n{extra_instruction}")

        parts.extend([
            "",
            "Return the IMPROVED strategy_spec (including management and ATR multipliers), "
            "updated rules and changes_summary strictly as JSON. "
            "strategy_spec MUST genuinely differ from the current one."
        ])

        return "\n".join(parts)

    def optimize_strategy(
        self,
        analysis_summary: Dict,
        current_spec: Dict,
        current_rules: str,
        rejected_specs: Optional[list] = None,
        extra_instruction: str = "",
        model: str = None,
    ) -> Optional[Dict[str, Any]]:
        """
        One model request: TRAIN/TEST analysis -> improved strategy_spec + rules.
        Returns a dict with keys: strategy_spec, new_rules, changes_summary (or None).
        """
        use_model = model or self.model
        user_message = self._build_optimize_message(
            analysis_summary,
            current_spec,
            current_rules,
            rejected_specs=rejected_specs,
            extra_instruction=extra_instruction,
        )
        messages = [
            {"role": "system", "content": OPTIMIZE_PROMPT},
            {"role": "user", "content": user_message},
        ]
        try:
            total_chars = sum(len(m["content"]) for m in messages)
            logger.info(f"Strategy optimization: ~{total_chars} characters")
            response = self.client.chat.completions.create(
                model=use_model,
                messages=messages,
                max_tokens=16000,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            if not response.choices:
                return None
            content = response.choices[0].message.content
            if not content or not content.strip():
                return None
            parsed = _safe_json_loads(content)
            if parsed is None:
                logger.error(
                    f"optimize_strategy: failed to parse JSON. "
                    f"First 500 characters: {content[:500]}"
                )
            return parsed
        except Exception as e:
            logger.error(f"optimize_strategy error: {e}")
            return None

    def revise_init_strategy(
        self,
        symbol: str,
        current_spec: Dict,
        current_rules: str,
        diagnostics_text: str,
        research: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """One request: rebuild strategy_spec after the initialization sanity backtest."""
        parts = [
            f"=== REBUILDING STRATEGY FOR {symbol} AFTER THE SANITY BACKTEST ===",
            "",
            "--- CURRENT strategy_spec ---",
            json.dumps(current_spec, ensure_ascii=False),
            "",
            "--- CURRENT RULES (abridged) ---",
            (current_rules or "none")[:800],
            "",
            "--- BACKTEST DIAGNOSTICS ---",
            diagnostics_text,
            "",
            "Return STRICT JSON without comments and markdown blocks:",
            '{"strategy_spec": {...}, "trading_rules_list": [...], '
            '"risk_management_rules_list": [...], "new_rules": "...", '
            '"changes_summary": "..."}',
        ]
        messages = [
            {"role": "system", "content": REVISE_INIT_PROMPT},
            {"role": "user", "content": "\n".join(parts)},
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=16000,  # Increased for the full JSON
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            if not response.choices or not response.choices[0].message.content:
                logger.warning("[%s] revise_init_strategy: the model returned an empty response", symbol)
                return None
            content = response.choices[0].message.content
            parsed = _safe_json_loads(content)
            if parsed is None:
                logger.error("[%s] revise_init_strategy: failed to parse JSON", symbol)
            return parsed
        except Exception as e:
            logger.error("[%s] revise_init_strategy error: %s", symbol, e)
            return None

    @staticmethod
    def _detect_available_timeframes(data_package: Dict) -> set:
        available = set()
        data = data_package.get("data", {})
        if data.get("prices_1h_csv"):
            available.add("1h")
        if data.get("prices_15m_csv"):
            available.add("15m")
        return available

    @staticmethod
    def _validate_conditions(conditions: list, available_tf: set) -> tuple:
        if not conditions:
            return conditions, []
        tf_map = {
            "5m": "15m",
            "5min": "15m",
            "4h": "1h",
            "4hour": "1h",
            "1d": "1h",
            "day": "1h",
        }
        fixed_conditions = []
        warnings = []
        for cond in conditions:
            if not isinstance(cond, str):
                fixed_conditions.append(cond)
                continue
            original = cond
            match = re.search(r'(?:на\s+)?(\d+\s*(?:m|min|h|hour|d|day))\b', cond, re.IGNORECASE)
            if match:
                tf_raw = match.group(1).lower().replace(" ", "")
                tf_normalized = tf_raw.replace("min", "m").replace("hour", "h").replace("day", "d")
                if tf_normalized not in available_tf:
                    replacement = tf_map.get(tf_normalized, "15m")
                    cond = re.sub(
                        r'(?:на\s+)?' + re.escape(match.group(1)),
                        replacement,
                        cond,
                        count=1,
                        flags=re.IGNORECASE,
                    )
                    warnings.append(
                        f"Timeframe '{match.group(1)}' replaced with '{replacement}' "
                        f"(missing in the data). Original condition: '{original}'"
                    )
            fixed_conditions.append(cond)
        return fixed_conditions, warnings

    def _validate_response_timeframes(self, response: Dict, data_package: Dict) -> Dict:
        available_tf = self._detect_available_timeframes(data_package)
        all_warnings = []

        setup = response.get("setup")
        if setup:
            conditions, warns = self._validate_conditions(setup.get("conditions", []), available_tf)
            if warns:
                setup["conditions"] = conditions
                all_warnings.extend(warns)
            cancel_conds, cancel_warns = self._validate_conditions(
                setup.get("cancel_conditions", []), available_tf
            )
            if cancel_warns:
                setup["cancel_conditions"] = cancel_conds
                all_warnings.extend(cancel_warns)
            if all_warnings:
                prefix = setup.get("rationale", "")
                setup["rationale"] = (
                    prefix + (" | " if prefix else "") +
                    "⚠️ Timeframes adjusted: " + "; ".join(all_warnings)
                )

        signal = response.get("signal")
        if signal:
            entry_conds, entry_warns = self._validate_conditions(
                signal.get("entry_conditions", []), available_tf
            )
            if entry_warns:
                signal["entry_conditions"] = entry_conds
                all_warnings.extend(entry_warns)
            cancel_conds, cancel_warns = self._validate_conditions(
                signal.get("cancel_conditions", []), available_tf
            )
            if cancel_warns:
                signal["cancel_conditions"] = cancel_conds
                all_warnings.extend(cancel_warns)
            if all_warnings:
                prefix = signal.get("signal_rationale", "")
                signal["signal_rationale"] = (
                    prefix + (" | " if prefix else "") +
                    "⚠️ Timeframes adjusted: " + "; ".join(all_warnings)
                )

        if all_warnings:
            analysis = response.get("analysis")
            if analysis:
                summary = analysis.get("summary", "")
                analysis["summary"] = (
                    summary + (" " if summary else "") +
                    "[Timeframe auto-correction: " + "; ".join(all_warnings) + "]"
                )

        return response

    def analyze(
        self,
        data_package: Dict,
        trading_rules: str = "",
        active_trade: Optional[Dict] = None,
        trade_closed: Optional[Dict] = None,
        is_initial: bool = False,
        morning_context: str = "",
        intraday_context: str = "",
        behavior_rules: str = "",
        error_analysis: bool = False,
        keep_history: bool = True,
        model: str = None,
    ) -> Optional[Dict[str, Any]]:
        pair = data_package.get("pair", "UNKNOWN")
        use_model = model or self.model
        msg_type = data_package.get("message_type", "")

        user_message = self._build_message(
            data_package,
            trading_rules,
            active_trade,
            trade_closed,
            is_initial,
            morning_context,
            intraday_context,
            behavior_rules,
            error_analysis,
        )

        use_fresh_context = (
            not keep_history
            or "intraday" in msg_type
            or error_analysis
            or active_trade is None
        )

        if use_fresh_context:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
        else:
            if pair not in self.conversation_history:
                self.conversation_history[pair] = [{"role": "system", "content": SYSTEM_PROMPT}]
            self.conversation_history[pair].append({"role": "user", "content": user_message})
            if len(self.conversation_history[pair]) > 7:
                self.conversation_history[pair] = (
                    [self.conversation_history[pair][0]] + self.conversation_history[pair][-6:]
                )
            messages = self.conversation_history[pair]

        kwargs = {
            "model": use_model,
            "messages": messages,
            "max_tokens": 2000,
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
        }

        try:
            total_chars = sum(len(m["content"]) for m in messages)
            logger.info(
                f"Request to {use_model} for {pair}: ~{total_chars} characters, {len(messages)} messages"
            )
            response = self.client.chat.completions.create(**kwargs)

            if not response.choices:
                return {
                    "action": "ANALYSIS",
                    "analysis": {"summary": "Empty API response", "current_price": 0},
                }

            content = response.choices[0].message.content
            logger.info(f"Response: {len(content) if content else 0} characters")

            if not content or len(content.strip()) == 0:
                return {
                    "action": "ANALYSIS",
                    "analysis": {"summary": "Empty model response", "current_price": 0},
                }

            if keep_history and "intraday" not in msg_type and not error_analysis:
                if pair not in self.conversation_history:
                    self.conversation_history[pair] = []
                self.conversation_history[pair].append({"role": "assistant", "content": content})

            result = _safe_json_loads(content)
            if result is None:
                logger.error(f"[{pair}] analyze: failed to parse JSON. First 500 characters: {content[:500]}")
                return {
                    "action": "ANALYSIS",
                    "analysis": {
                        "summary": f"Non-JSON response (first 200 characters): {content[:200]}",
                        "current_price": 0,
                    },
                }

            result = self._validate_response_timeframes(result, data_package)
            return result

        except Exception as e:
            logger.exception(f"API error for {pair}")
            return {
                "action": "ANALYSIS",
                "analysis": {"summary": f"Error: {str(e)[:200]}", "current_price": 0},
            }

    def chat_with_tools(self, user_message: str, data_package: Dict, trading_rules: str = "", active_trade: Optional[Dict] = None, morning_context: str = "", intraday_context: str = "", last_analysis: Optional[Dict] = None, behavior_rules: str = "", tools: list = None, tool_executor: Callable = None, max_iterations: int = 5, model: str = None, cached_news: tuple = None) -> str:
        use_model = model or self.model
        pair = data_package.get("pair", "UNKNOWN")
        data = data_package.get("data", {})

        if cached_news:
            news_text, news_sentiment = cached_news
            data["news_text"] = news_text
            data["news_sentiment"] = news_sentiment

        data_package['morning_context'] = morning_context
        data_package['intraday_context'] = intraday_context
        self._log_message_summary(data_package, additional_info=f"CHAT: {user_message[:50]}...")

        chat_msg = f"=== CHAT FROM USER ===\nPair: {pair}\nUSER MESSAGE: {user_message}\nYou have access to tools. Answer in English."

        if morning_context:
            chat_msg += f"\n--- MORNING CONTEXT ---\n{morning_context}"
        if intraday_context:
            chat_msg += f"\n--- LATEST CONCLUSIONS ---\n{intraday_context}"
        if last_analysis:
            chat_msg += f"\n--- LAST ANALYSIS ---\n{json.dumps(last_analysis, ensure_ascii=False)}"
        if trading_rules:
            chat_msg += f"\n--- TRADING RULES ---\n{trading_rules}"
        if behavior_rules:
            chat_msg += f"\n--- BEHAVIOR RULES ---\n{behavior_rules}"
        if active_trade:
            chat_msg += f"\n--- ACTIVE POSITION ---\n{json.dumps(active_trade, ensure_ascii=False)}"
        if data.get("indicators"):
            chat_msg += f"\n--- INDICATORS (JSON) ---\n{data['indicators']}"
        if data.get("news_text"):
            chat_msg += f"\n--- NEWS ---\n{data['news_text']}"
        if data.get("market_stats") and data["market_stats"] != "{}":
            chat_msg += f"\n--- PRECOMPUTED METRICS (JSON) ---\n{data['market_stats']}"

        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": chat_msg}]
        tools = tools or []

        for iteration in range(max_iterations):
            kwargs = {"model": use_model, "messages": messages, "max_tokens": 2000, "temperature": 0.5, "tools": tools, "tool_choice": "auto"}
            try:
                response = self.client.chat.completions.create(**kwargs)
                if not response.choices:
                    return "The model did not respond."
                msg = response.choices[0].message
                if msg.tool_calls:
                    messages.append({"role": "assistant", "content": msg.content, "tool_calls": [{"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in msg.tool_calls]})
                    for tc in msg.tool_calls:
                        try:
                            args = json.loads(tc.function.arguments or "{}")
                        except Exception:
                            args = {}
                        result = tool_executor(tc.function.name, args) if tool_executor else "No tool handler configured."
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                    continue
                return (msg.content or "No response").strip()
            except Exception as e:
                logger.error(f"chat_with_tools error: {e}")
                return f"Error: {str(e)[:200]}"

        return "Tool iteration limit reached."

    def clear_history(self, pair: str = None):
        if pair:
            self.conversation_history.pop(pair, None)
        else:
            self.conversation_history.clear()