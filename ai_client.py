"""
Клиент для взаимодействия с DeepSeek API.
Добавлены: адаптивный порог ADX (из market_stats), валидация таймфреймов,
поддержка облегчённого внутридневного пакета (intraday_light).
"""
import json
import logging
import time
import threading
import re
from typing import Optional, Dict, Any, Callable
from openai import OpenAI

import config
#пока не используем, аместо нее новая функция с новой библиотекой
#from utils.helpers import safe_parse_json
from prompts.system_prompt import SYSTEM_PROMPT
from prompts.init_prompt import INIT_PROMPT
REVISE_INIT_PROMPT = """Ты — профессиональный криптотрейдер и системный разработчик торговых стратегий.

### ЗАДАЧА
Стратегия была создана при инициализации монеты, затем система прогнала её на ~60 днях истории и вернула числовую диагностику. Пересобери strategy_spec, устранив диагностированные дефекты.

### ЖЁСТКИЕ ПРАВИЛА (нельзя нарушать)
1. **НЕ УБИРАТЬ ОБЯЗАТЕЛЬНЫЕ ФИЛЬТРЫ:**
   - Объёмный фильтр (`type: volume`) ОБЯЗАТЕЛЕН в каждой группе входа (long и short)
   - Трендовый фильтр (sma_50/sma_50_1h/adx_14/macd_hist) ОБЯЗАТЕЛЕН
   - Условие привязки к уровню (resistance_1/support_1/sma_20/close) ОБЯЗАТЕЛЬНО

2. **МОЖНО МЕНЯТЬ:**
   - Числовые пороги (RSI-диапазоны, объёмные множители, ATR-множители стопов/тейков)
   - Добавлять вторые OR-группы входа (`entry_any`)
   - Параметры блока `management` (breakeven, trailing, max_hold_candles)
   - Параметры блока `risk_controls` (cooldown)

3. **СТРУКТУРНЫЕ ОГРАНИЧЕНИЯ:**
   - Не более 2 OR-групп входа на сторону (`entry_any` имеет максимум 2 элемента)
   - Не более 6-7 условий в каждой группе
   - Стоп-лосс: 2.5-4.0 x ATR15m (вне шума)
   - Тейк-профит: не ближе 2.5 x стопа (payoff >= 2.5:1)

### ПРИМЕРЫ ПРАВИЛЬНЫХ ИЗМЕНЕНИЙ
**Если частота слишком низкая:**
- Ослабить RSI-диапазон: `"min": 40, "max": 70` вместо `"min": 52, "max": 60`
- Уменьшить объёмный порог: `"mult": 0.8` вместо `"mult": 1.3`
- Добавить вторую OR-группу в `entry_any` (ретест уровня или откат к SMA20)

**Если WR ниже безубыточного порога:**
- Поднять `take_profit_atr_mult` до 4.0-5.0 (увеличить payoff)
- Увеличить `stop_loss_atr_mult` до 3.0-4.0 (стоп вне шума)
- Убрать входы с малым потенциалом движения (слишком близкие к уровню)

**Если много быстрых убытков (<= 3 свечи):**
- Входить на ретесте, а не на пробое: `"low" <= "resistance_1"` вместо `"close" > "resistance_1"`
- Расширить стоп: `stop_loss_atr_mult: 3.0` вместо `1.5`
- Добавить фильтр ADX: `"adx_14_1h" > 20` (только в тренде)

### ФОРМАТ ОТВЕТА (строгий JSON)
{
  "strategy_spec": {
    "long": {
      "entry_any": [
        [
          {"type": "compare", "left": "close", "op": ">", "right": "resistance_1"},
          {"type": "compare", "left": "low", "op": "<=", "right": {"name": "resistance_1", "mult": 1.003}},
          {"type": "compare", "left": "close", "op": ">", "right": "sma_50_1h"},
          {"type": "compare", "left": "macd_hist", "op": ">", "right": 0},
          {"type": "range", "value": "rsi_14", "min": 45, "max": 65},
          {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 1.0}
        ],
        [
          {"type": "compare", "left": "close", "op": ">", "right": "sma_50_1h"},
          {"type": "compare", "left": "low", "op": "<=", "right": "sma_20"},
          {"type": "compare", "left": "close", "op": ">", "right": "sma_20"},
          {"type": "compare", "left": "macd_hist", "op": ">", "right": 0},
          {"type": "range", "value": "rsi_14", "min": 40, "max": 60},
          {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 0.8}
        ]
      ],
      "stop_loss_atr_mult": 3.0,
      "take_profit_atr_mult": 4.5,
      "management": {
        "move_to_breakeven_at_tp_pct": 0.65,
        "breakeven_offset_pct": 0.0,
        "trailing_atr_mult": 2.5,
        "max_hold_candles": 128,
        "exit_on_level_failure": false,
        "exit_on_trend_loss": false,
        "exit_on_macd_flip": false
      }
    },
    "short": { ... аналогично для short ... },
    "filters": [ ... ],
    "risk_controls": { ... }
  },
  "trading_rules_list": [ ... ],
  "risk_management_rules_list": [ ... ],
  "new_rules": "...",
  "changes_summary": "Ослабил RSI до 45-65, добавил вторую группу (откат к SMA20), поднял стоп до 3.0 ATR, тейк до 4.5 ATR (payoff 1.5:1 → 1.5:1)"
}

### КРИТИЧЕСКИ ВАЖНО
- НИКОГДА не убирай объёмный фильтр — это обязательное условие
- НИКОГДА не добавляй комментарии в JSON
- Верни ПОЛНЫЙ валидный JSON, который можно напрямую распарсить json.loads()
- Все ключи и строковые значения только в двойных кавычках
- Минимум 6 правил торговли и 4 правила риск-менеджмента"""

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
    Безопасный парсинг JSON с агрессивной pre-обработкой и json_repair.
    """
    # 1. Удаляем markdown блоки ```json ... ```
    text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```\s*', '', text)
    
    # 2. Удаляем однострочные комментарии //
    text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
    
    # 3. Удаляем многострочные комментарии /* */
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    
    # 4. Удаляем trailing запятые (агрессивно, несколько проходов)
    for _ in range(3):
        text = re.sub(r',\s*([}\]])', r'\1', text)
    
    # 5. Фиксим пустые строки в числовых полях: "" → null
    text = re.sub(r':\s*""\s*([,}])', r': null\1', text)
    
    # 6. Фиксим одинарные кавычки на двойные (если модель их использовала)
    text = re.sub(r"'([^']*)'(?=\s*[:},])", r'"\1"', text)
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        if HAS_JSON_REPAIR:
            try:
                repaired = repair_json(text)
                logger.info(f"JSON repair успешен: исправлено {len(text)} символов")
                return json.loads(repaired)
            except Exception as repair_error:
                logger.error(f"JSON repair failed: {repair_error}")
                # Последняя попытка: ищем первый { и последний }
                try:
                    start = text.find('{')
                    end = text.rfind('}')
                    if start >= 0 and end > start:
                        extracted = text[start:end+1]
                        return json.loads(extracted)
                except Exception:
                    pass
                return None
        logger.error(f"JSON parse error (json_repair недоступен): {e}")
        return None

class DeepSeekClient:
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or config.DEEPSEEK_API_KEY
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY не задан.")
        self.model = model or config.DEEPSEEK_MODEL
        self.client = OpenAI(api_key=self.api_key, base_url=config.DEEPSEEK_BASE_URL)
        self.conversation_history: Dict[str, list] = {}
        self.web_searcher = WebSearcher()
        logger.info(f"DeepSeekClient создан, модель: {self.model}")

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def _log_data_sizes(self, data: dict, label: str = ""):
        for field in ["prices_1h_csv", "prices_15m_csv", "fundamentals_csv", "orderbook_bids_csv", "orderbook_asks_csv", "trades_summary", "news_text", "market_stats"]:
            content = data.get(field, "")
            if content:
                tokens = self._estimate_tokens(content)
                token_logger.info(f"{label} {field}: ~{tokens} токенов ({len(content)} символов)")

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
        return [{"type": "function", "function": {"name": "web_search", "description": f"Поиск информации в интернете. Используй для поиска правил торговли, риск-менеджмента и особенностей {symbol}.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Поисковый запрос"}}, "required": ["query"]}}}]

    def _build_message(self, data_package: Dict, trading_rules: str = "", active_trade: Optional[Dict] = None, trade_closed: Optional[Dict] = None, is_initial: bool = False, morning_context: str = "", intraday_context: str = "", behavior_rules: str = "", error_analysis: bool = False) -> str:
        parts = []
        msg_type = data_package.get("message_type", "unknown")
        pair = data_package.get("pair", "UNKNOWN")
        ts = data_package.get("timestamp_utc", "")
        data = data_package.get("data", {})
        if msg_type == "intraday_light":
            parts.append("=== ПРОВЕРКА СЕТАПА ===")
            parts.append(f"Пара: {pair} | Время (UTC): {ts}")
            parts.append("Это облегчённый пакет для проверки сетапа/позиции. Получи только свежие свечи и обновлённые метрики.")
        else:
            parts.append(f"=== {'АНАЛИЗ ОШИБОК' if error_analysis else 'АНАЛИТИЧЕСКИЙ ЗАПРОС'} ===")
            parts.append(f"Пара: {pair} | Время (UTC): {ts} | Тип: {msg_type}")
            if error_analysis:
                parts.append("\nЭто анализ закрытой сделки. Проанализируй ошибки и предложи изменения в правилах.")
        self._log_data_sizes(data, f"[{pair}] {msg_type}")
        data_package['morning_context'] = morning_context
        data_package['intraday_context'] = intraday_context
        self._log_message_summary(data_package, additional_info=f"{'error_analysis' if error_analysis else ''}")
        if data.get("news_text"):
            parts.append("\n--- НОВОСТИ ---")
            parts.append(data["news_text"])
        if morning_context:
            parts.append("\n--- УТРЕННИЙ КОНТЕКСТ ---")
            parts.append(morning_context)
        if intraday_context:
            parts.append("\n--- ПРЕДЫДУЩИЕ ВЫВОДЫ ---")
            parts.append(intraday_context)
        if trading_rules:
            parts.append("\n--- ПРАВИЛА ТОРГОВЛИ ---")
            parts.append(trading_rules)
        if behavior_rules:
            parts.append("\n--- ПОВЕДЕНЧЕСКИЕ ПРАВИЛА ---")
            parts.append(behavior_rules)
        if active_trade:
            parts.append("\n--- АКТИВНАЯ ПОЗИЦИЯ ---")
            parts.append(json.dumps(active_trade, ensure_ascii=False))
        if trade_closed:
            parts.append("\n--- СДЕЛКА ЗАКРЫТА ---")
            parts.append(json.dumps(trade_closed, ensure_ascii=False))
        if data.get("indicators"):
            parts.append("\n--- ИНДИКАТОРЫ (JSON) ---")
            parts.append(data["indicators"])
        if data.get("market_stats") and data["market_stats"] != "{}":
            parts.append("\n--- ПРЕДВЫЧИСЛЕННЫЕ МЕТРИКИ (JSON) ---")
            parts.append(data["market_stats"])
        if msg_type != "intraday_light":
            parts.append("\n--- ВНИМАНИЕ: ФОРМАТ ОТВЕТА ---")
            parts.append("Ты ОБЯЗАН вернуть JSON со ВСЕМИ полями. Даже при неопределённом рынке всегда включай 'setup' с conditions и cancel_conditions. Для объёмов используй готовые цифры из market_stats, НЕ парси CSV.")
        parts.append("\n--- РЫНОЧНЫЕ ДАННЫЕ ---")
        for field, label in [("prices_1h_csv", "Часовые свечи"), ("prices_15m_csv", "15m свечи"), ("fundamentals_csv", "Фундаментал"), ("orderbook_bids_csv", "Bids"), ("orderbook_asks_csv", "Asks")]:
            if data.get(field):
                parts.append(f"\n{label}:")
                parts.append(data[field])
        if data.get("trades_summary") and data["trades_summary"] != "{}":
            parts.append("\n--- ТРЕЙДЫ (агрегировано) ---")
            parts.append(data["trades_summary"])
        return "\n".join(parts)

    def _build_init_message(self, data_package: Dict, symbol: str) -> str:
        msg_type = data_package.get("message_type", "unknown")
        ts = data_package.get("timestamp_utc", "")
        data = data_package.get("data", {})
        base_asset = symbol.replace("USDT", "").replace("USDC", "").replace("BUSD", "")
        parts = ["=== ИНИЦИАЛИЗАЦИЯ МОНЕТЫ ===", f"Пара: {symbol} (базовый актив: {base_asset}) | Время (UTC): {ts}", f"\nСформируй правила торговли и риск-менеджмента для {symbol}.", "У тебя есть доступ к поисковому инструменту web_search.", "Правила должны быть на русском языке, без конкретных ценовых значений."]
        self._log_data_sizes(data, f"[{symbol}] INIT")
        self._log_message_summary(data_package, additional_info="INIT")
        if data.get("news_text"):
            parts.append("\n--- НОВОСТИ ---")
            parts.append(data["news_text"])
        if data.get("indicators"):
            parts.append("\n--- ИНДИКАТОРЫ (JSON) ---")
            parts.append(data["indicators"])
        parts.append("\n--- РЫНОЧНЫЕ ДАННЫЕ (CSV) ---")
        for field, label in [("prices_1h_csv", "Часовые свечи"), ("prices_15m_csv", "15m свечи"), ("fundamentals_csv", "Фундаментал"), ("orderbook_bids_csv", "Bids"), ("orderbook_asks_csv", "Asks")]:
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
                messages.append({"role": "user", "content": "Ты собрал достаточно информации. Теперь агрегируй все данные и сформируй JSON с правилами. Если тебе нужен ЕЩЁ ОДИН поисковый запрос для уточнения — сделай его, но на следующей итерации ОБЯЗАН вернуть готовый JSON без вызова инструментов."})
            elif iteration >= 3:
                messages.append({"role": "user", "content": "БОЛЬШЕ НЕ ИЩИ. Сейчас же верни JSON с агрегированными правилами. Никаких tool_calls. Только JSON."})
            kwargs = {"model": use_model, "messages": messages, "max_tokens": 8000, "temperature": 0.5, "tools": tools if iteration < 3 else [], "tool_choice": "auto" if iteration < 3 else "none"}
            try:
                total_chars = sum(len(m["content"]) for m in messages)
                logger.info(f"Инициализация {symbol} (итерация {iteration+1}): ~{total_chars} символов")
                response = self.client.chat.completions.create(**kwargs)
                if not response.choices:
                    return {"action": "INIT", "analysis": {"summary": "Пустой ответ API", "current_price": 0}}
                msg = response.choices[0].message
                if msg.content and '<｜｜DSML｜｜tool_calls>' in msg.content:
                    logger.warning(f"[{symbol}] Модель попыталась вызвать инструмент после отключения (итерация {iteration+1}). Возвращаю fallback.")
                    return {"action": "INIT", "analysis": {"summary": "Модель не смогла завершить инициализацию — попытка вызова инструмента после отключения.", "current_price": 0}}
                if msg.content and len(msg.content.strip()) > 100:
                    result = _safe_json_loads(msg.content)
                    if result is not None and (result.get("trading_rules_list") or result.get("new_rules")):
                        logger.info(f"[{symbol}] INIT: модель вернула готовые правила (итерация {iteration+1})")
                        if "trading_rules_list" in result and "new_rules" not in result:
                            trading = "\n".join(result.get("trading_rules_list", []))
                            risk = "\n".join(result.get("risk_management_rules_list", []))
                            result["new_rules"] = trading + "\n\n" + risk
                        return result
                if msg.tool_calls:
                    logger.info(f"[{symbol}] INIT: модель вызвала {len(msg.tool_calls)} инструментов (итерация {iteration+1})")
                    messages.append({"role": "assistant", "content": msg.content, "tool_calls": [{"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in msg.tool_calls]})
                    for tc in msg.tool_calls:
                        func_name = tc.function.name
                        try:
                            args = json.loads(tc.function.arguments or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        if func_name == "web_search":
                            query = args.get("query", "")
                            logger.info(f"[{symbol}] Поиск: {query}")
                            result = self.web_searcher.search_formatted(query, max_results=5)
                        else:
                            result = f"Неизвестный инструмент: {func_name}"
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                        logger.info(f"[{symbol}] Результат поиска ({len(result)} символов)")
                    continue
                content = msg.content
                logger.info(f"Ответ инициализации: {len(content) if content else 0} символов")
                if not content or len(content.strip()) == 0:
                    return {"action": "INIT", "analysis": {"summary": "Пустой ответ модели", "current_price": 0}}
                result = _safe_json_loads(content)
                if result is None:
                    logger.error(f"Не удалось распарсить JSON из INIT ответа. Сырой ответ: {content[:500]}...")
                    return {"action": "INIT", "analysis": {"summary": "Не удалось распарсить JSON ответа модели.", "current_price": 0}}
                if "trading_rules_list" in result and "new_rules" not in result:
                    trading = "\n".join(result.get("trading_rules_list", []))
                    risk = "\n".join(result.get("risk_management_rules_list", []))
                    result["new_rules"] = trading + "\n\n" + risk
                return result
            except Exception as e:
                logger.error(f"Ошибка инициализации {symbol} (итерация {iteration+1}): {e}")
                return {"action": "INIT", "analysis": {"summary": f"Ошибка: {str(e)[:200]}", "current_price": 0}}
        return {"action": "INIT", "analysis": {"summary": "Достигнут лимит итераций инициализации.", "current_price": 0}}

    def _build_optimize_message(
        self,
        analysis_summary: Dict,
        current_spec: Dict,
        current_rules: str,
        rejected_specs: Optional[list] = None, # Оставляем для совместимости, но теперь сюда придут rejected_changes
        extra_instruction: str = "",
    ) -> str:
        """
        Сообщение для оптимизации. Сжато для экономии токенов.
        """
        # 1. ЖЕСТКО ОБРЕЗАЕМ ТЕКСТОВЫЕ ПРАВИЛА (они дублируют spec и жрут токены)
        rules_text = (current_rules or "нет").strip()
        if len(rules_text) > 800:
            rules_text = rules_text[:800] + "\n... [текст правил обрезан для экономии токенов, смотри strategy_spec]"

        parts = [
            "=== ОПТИМИЗАЦИЯ СТРАТЕГИИ ===",
            "--- ТЕКУЩАЯ strategy_spec ---",
            json.dumps(current_spec, ensure_ascii=False),
            "",
            "--- ТЕКУЩИЕ ПРАВИЛА (сокращенно) ---",
            rules_text,
            "",
            "--- РЕЗУЛЬТАТЫ БЭКТЕСТА ---",
            json.dumps(analysis_summary, ensure_ascii=False),
            "",
            "--- ЦЕЛЬ ---",
            "Максимизируй TEST P&L (expectancy × число сделок). PF>1.1 на TEST. "
            "Смотри на 'loss_profile' и 'win_profile' в trade_summary: сравни средние "
            "индикаторы (RSI, ADX, volume) на убыточных и прибыльных входах. "
            "Если убытки происходят при низком ADX или высоком RSI — добавь фильтры. "
            "Не повторяй изменения из блока 'rejected_changes'."
        ]

        # 2. ВМЕСТО сырых JSON отвергнутых спеков, передаем текстовые описания ошибок
        rejected_changes = analysis_summary.get("rejected_changes", [])
        if rejected_changes:
            parts.append("")
            parts.append("--- ПОСЛЕДНИЕ НЕУДАЧНЫЕ ПОПЫТКИ (НЕ ПОВТОРЯЙ) ---")
            for rc in rejected_changes:
                parts.append(f"- {rc}")

        # --- ШАГ 3: Динамическая инъекция жестких ограничений для AI ---
        test_stats = analysis_summary.get("test_stats", {})
        test_pf = float(test_stats.get("profit_factor", 0) or 0)
        test_trades = int(test_stats.get("total_trades", 0) or 0)
        
        ai_restrictions = []
        # 1. Если стратегия уже прибыльна, запрещаем менять ядро (защита от переобучения)
        if test_pf > 1.1:
            ai_restrictions.append(
                "⚠️ КРИТИЧНО: Текущая база прибыльна (PF > 1.1). "
                "КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО менять логику входа (блоки 'entry' и 'entry_any'). "
                "Ты можешь менять ТОЛЬКО 'filters', 'management' и ATR-множители "
                "(stop_loss_atr_mult, take_profit_atr_mult)."
            )
        # 2. Если сделок мало, запрещаем добавлять условия в один entry, требуем entry_any
        if test_trades < 30:
            ai_restrictions.append(
                f"⚠️ СДЕЛОК СЛИШКОМ МАЛО ({test_trades} < 30). "
                "Чтобы увеличить частоту, ты ОБЯЗАН использовать блок 'entry_any' и добавить "
                "туда минимум 2 РАЗНЫЕ логические группы (например, одна на пробой, другая на откат). "
                "НЕ добавляй новые условия в существующий 'entry' — это убьет частоту."
            )
        
        if ai_restrictions:
            dynamic_instruction = "\n".join(ai_restrictions)
            if extra_instruction:
                extra_instruction = dynamic_instruction + "\n" + extra_instruction
            else:
                extra_instruction = dynamic_instruction

        if extra_instruction:
            parts.append("")
            parts.append(f"--- ДОПОЛНИТЕЛЬНОЕ ТРЕБОВАНИЕ ---\n{extra_instruction}")

        parts.extend([
            "",
            "Верни УЛУЧШЕННУЮ strategy_spec (включая management и ATR-множители), "
            "обновлённые правила и changes_summary строго в JSON. "
            "strategy_spec ОБЯЗАН реально отличаться от текущего."
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
        Один запрос к модели: анализ TRAIN/TEST → улучшенная strategy_spec + правила.
        Возвращает dict с ключами: strategy_spec, new_rules, changes_summary (или None).
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
            logger.info(f"Оптимизация стратегии: ~{total_chars} символов")

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
                    f"optimize_strategy: не удалось распарсить JSON. "
                    f"Первые 500 символов: {content[:500]}"
                )
            return parsed

        except Exception as e:
            logger.error(f"Ошибка optimize_strategy: {e}")
            return None

    def revise_init_strategy(
        self,
        symbol: str,
        current_spec: Dict,
        current_rules: str,
        diagnostics_text: str,
        research: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Один запрос: пересборка strategy_spec после санити-бэктеста инициализации."""
        parts = [
            f"=== ПЕРЕРАБОТКА СТРАТЕГИИ {symbol} ПОСЛЕ САНИТИ-БЭКТЕСТА ===",
            "",
            "--- ТЕКУЩАЯ strategy_spec ---",
            json.dumps(current_spec, ensure_ascii=False),
            "",
            "--- ТЕКУЩИЕ ПРАВИЛА (сокращённо) ---",
            (current_rules or "нет")[:800],
            "",
            "--- ДИАГНОСТИКА БЭКТЕСТА ---",
            diagnostics_text,
            "",
            "Верни СТРОГИЙ JSON без комментариев и markdown-блоков:",
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
                max_tokens=16000,  # Увеличено для полного JSON
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            if not response.choices or not response.choices[0].message.content:
                logger.warning("[%s] revise_init_strategy: модель вернула пустой ответ", symbol)
                return None
            content = response.choices[0].message.content
            parsed = _safe_json_loads(content)
            if parsed is None:
                logger.error("[%s] revise_init_strategy: не удалось распарсить JSON", symbol)
            return parsed
        except Exception as e:
            logger.error("[%s] Ошибка revise_init_strategy: %s", symbol, e)
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
                        f"Таймфрейм '{match.group(1)}' заменён на '{replacement}' "
                        f"(отсутствует в данных). Исходное условие: '{original}'"
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
                    "⚠️ Таймфреймы скорректированы: " + "; ".join(all_warnings)
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
                    "⚠️ Таймфреймы скорректированы: " + "; ".join(all_warnings)
                )

        if all_warnings:
            analysis = response.get("analysis")
            if analysis:
                summary = analysis.get("summary", "")
                analysis["summary"] = (
                    summary + (" " if summary else "") +
                    "[Автокоррекция таймфреймов: " + "; ".join(all_warnings) + "]"
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
                f"Запрос к {use_model} для {pair}: ~{total_chars} символов, {len(messages)} сообщений"
            )

            response = self.client.chat.completions.create(**kwargs)
            if not response.choices:
                return {
                    "action": "ANALYSIS",
                    "analysis": {"summary": "Пустой ответ API", "current_price": 0},
                }

            content = response.choices[0].message.content
            logger.info(f"Ответ: {len(content) if content else 0} символов")

            if not content or len(content.strip()) == 0:
                return {
                    "action": "ANALYSIS",
                    "analysis": {"summary": "Пустой ответ модели", "current_price": 0},
                }

            if keep_history and "intraday" not in msg_type and not error_analysis:
                if pair not in self.conversation_history:
                    self.conversation_history[pair] = []
                self.conversation_history[pair].append({"role": "assistant", "content": content})

            result = _safe_json_loads(content)
            if result is None:
                logger.error(f"[{pair}] analyze: не удалось распарсить JSON. Первые 500 символов: {content[:500]}")
                return {
                    "action": "ANALYSIS",
                    "analysis": {
                        "summary": f"Не-JSON ответ (первые 200 символов): {content[:200]}",
                        "current_price": 0,
                    },
                }

            result = self._validate_response_timeframes(result, data_package)
            return result

        except Exception as e:
            logger.exception(f"Ошибка API для {pair}")
            return {
                "action": "ANALYSIS",
                "analysis": {"summary": f"Ошибка: {str(e)[:200]}", "current_price": 0},
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
        chat_msg = f"=== ЧАТ ОТ ПОЛЬЗОВАТЕЛЯ ===\nПара: {pair}\nСООБЩЕНИЕ ПОЛЬЗОВАТЕЛЯ: {user_message}\nУ тебя есть доступ к инструментам (tools). Отвечай на русском языке."
        if morning_context:
            chat_msg += f"\n--- УТРЕННИЙ КОНТЕКСТ ---\n{morning_context}"
        if intraday_context:
            chat_msg += f"\n--- ПОСЛЕДНИЕ ВЫВОДЫ ---\n{intraday_context}"
        if last_analysis:
            chat_msg += f"\n--- ПОСЛЕДНИЙ АНАЛИЗ ---\n{json.dumps(last_analysis, ensure_ascii=False)}"
        if trading_rules:
            chat_msg += f"\n--- ПРАВИЛА ТОРГОВЛИ ---\n{trading_rules}"
        if behavior_rules:
            chat_msg += f"\n--- ПОВЕДЕНЧЕСКИЕ ПРАВИЛА ---\n{behavior_rules}"
        if active_trade:
            chat_msg += f"\n--- АКТИВНАЯ ПОЗИЦИЯ ---\n{json.dumps(active_trade, ensure_ascii=False)}"
        if data.get("indicators"):
            chat_msg += f"\n--- ИНДИКАТОРЫ (JSON) ---\n{data['indicators']}"
        if data.get("news_text"):
            chat_msg += f"\n--- НОВОСТИ ---\n{data['news_text']}"
        if data.get("market_stats") and data["market_stats"] != "{}":
            chat_msg += f"\n--- ПРЕДВЫЧИСЛЕННЫЕ МЕТРИКИ (JSON) ---\n{data['market_stats']}"
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": chat_msg}]
        tools = tools or []
        for iteration in range(max_iterations):
            kwargs = {"model": use_model, "messages": messages, "max_tokens": 2000, "temperature": 0.5, "tools": tools, "tool_choice": "auto"}
            try:
                response = self.client.chat.completions.create(**kwargs)
                if not response.choices:
                    return "Модель не ответила."
                msg = response.choices[0].message
                if msg.tool_calls:
                    messages.append({"role": "assistant", "content": msg.content, "tool_calls": [{"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in msg.tool_calls]})
                    for tc in msg.tool_calls:
                        try:
                            args = json.loads(tc.function.arguments or "{}")
                        except Exception:
                            args = {}
                        result = tool_executor(tc.function.name, args) if tool_executor else "Обработчик не задан."
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                    continue
                return (msg.content or "Нет ответа").strip()
            except Exception as e:
                logger.error(f"Ошибка chat_with_tools: {e}")
                return f"Ошибка: {str(e)[:200]}"
        return "Достигнут лимит итераций инструментов."

    def clear_history(self, pair: str = None):
        if pair:
            self.conversation_history.pop(pair, None)
        else:
            self.conversation_history.clear()
