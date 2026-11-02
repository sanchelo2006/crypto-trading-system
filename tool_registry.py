"""Реестр инструментов (function calling) для модели."""
import logging
from typing import Dict, Any
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

def get_tool_definitions() -> list:
    return [
        {"type": "function", "function": {"name": "get_current_rules", "description": "Получить текущие правила торговли.", "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {"name": "add_trading_rule", "description": "Добавить новое правило торговли.", "parameters": {"type": "object", "properties": {"rule_text": {"type": "string", "description": "Текст правила."}}, "required": ["rule_text"]}}},
        {"type": "function", "function": {"name": "update_trading_rules", "description": "Заменить все правила торговли.", "parameters": {"type": "object", "properties": {"rules_text": {"type": "string", "description": "Новый текст правил."}}, "required": ["rules_text"]}}},
        {"type": "function", "function": {"name": "get_current_setup", "description": "Получить текущий сетап.", "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {"name": "update_setup", "description": "Обновить сетап (направление, уровни, условия).", "parameters": {"type": "object", "properties": {"direction": {"type": "string", "enum": ["long", "short"]}, "entry_price": {"type": "number"}, "stop_loss": {"type": "number"}, "take_profit": {"type": "array", "items": {"type": "number"}}, "conditions": {"type": "array", "items": {"type": "string"}}, "cancel_conditions": {"type": "array", "items": {"type": "string"}}, "rationale": {"type": "string"}}, "required": ["direction"]}}},
        {"type": "function", "function": {"name": "get_market_analysis", "description": "Получить последний анализ рынка.", "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {"name": "get_active_trade", "description": "Получить информацию об открытой позиции.", "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {"name": "get_trade_stats", "description": "Получить статистику сделок.", "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {"name": "get_news", "description": "Получить свежие новости по монете.", "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {"name": "get_behavior_rules", "description": "Получить поведенческие правила.", "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {"name": "add_behavior_rule", "description": "Добавить поведенческое правило.", "parameters": {"type": "object", "properties": {"rule_text": {"type": "string", "description": "Текст правила поведения."}}, "required": ["rule_text"]}}},
        {"type": "function", "function": {"name": "get_news_analysis", "description": "Получить расширенный анализ новостей (форматированный текст).", "parameters": {"type": "object", "properties": {}, "required": []}}},
    ]

class ToolExecutor:
    def __init__(self, coin_memory, fetcher=None, symbol: str = ""):
        self.cm = coin_memory
        self.fetcher = fetcher
        self.symbol = symbol
        # Кеш для новостей на время одного вызова чата
        self._news_cache_local = None
        # Кеш для расширенного анализа (чтобы не пересчитывать тональность)
        self._news_analysis_cache = None

    def execute(self, func_name: str, arguments: Dict[str, Any]) -> str:
        logger.info(f"[{self.symbol}] Tool call: {func_name}")
        handlers = {
            "get_current_rules": lambda a: f"Текущие правила:\n\n{self.cm.get_rules()}" if self.cm.get_rules() else "Правила не созданы.",
            "add_trading_rule": self._add_rule,
            "update_trading_rules": self._update_rules,
            "get_current_setup": self._get_setup,
            "update_setup": self._update_setup,
            "get_market_analysis": self._get_analysis,
            "get_active_trade": self._get_trade,
            "get_trade_stats": self._get_stats,
            "get_news": self._get_news,
            "get_behavior_rules": lambda a: self.cm.get_behavior_rules() or "Поведенческие правила отсутствуют.",
            "add_behavior_rule": self._add_behavior,
            "get_news_analysis": self._get_news_analysis,
        }
        handler = handlers.get(func_name)
        if not handler: return f"Функция {func_name} не найдена."
        try: return handler(arguments)
        except Exception as e: return f"Ошибка {func_name}: {e}"

    def _add_rule(self, a):
        rule_text = a.get("rule_text", "").strip()
        if not rule_text: return "Пустой текст правила."
        self.cm.append_rule(rule_text)
        current_list = self.cm.get_state().get("trading_rules_list", [])
        current_list.append(rule_text)
        self.cm.update_state({"trading_rules_list": current_list})
        return "Правило добавлено."
        
    def _update_rules(self, a):
        rules_text = a.get("rules_text", "").strip()
        if not rules_text: return "Пустой текст правил."
        self.cm.update_rules(rules_text)
        self.cm.update_state({"trading_rules_list": [], "risk_management_rules_list": []})
        return "Правила обновлены."
        
    def _add_behavior(self, a):
        rule_text = a.get("rule_text", "").strip()
        if not rule_text: return "Пустой текст."
        self.cm.append_behavior_rule(rule_text)
        return "Поведенческое правило добавлено."

    def _get_setup(self, a):
        s = self.cm.get_state()
        setup = s.get("last_setup")
        signal = s.get("last_signal")
        if signal and signal.get("confidence_pct", 0) > 0:
            return f"Сигнал: {signal.get('direction','')} (уверенность {signal.get('confidence_pct',0)}%)"
        if setup:
            return (f"Сетап: {setup.get('direction','')}\nВход: {setup.get('entry_price','?')}\n"
                    f"SL: {setup.get('stop_loss','?')} TP: {setup.get('take_profit',[])}\n"
                    f"Условия: {setup.get('conditions',[])}\nОтмена: {setup.get('cancel_conditions',[])}")
        return "Сетап отсутствует."

    def _update_setup(self, a):
        new_setup = {
            "direction": a.get("direction"), "entry_price": a.get("entry_price"),
            "stop_loss": a.get("stop_loss"), "take_profit": a.get("take_profit", []),
            "conditions": a.get("conditions", []), "cancel_conditions": a.get("cancel_conditions", []),
            "rationale": a.get("rationale", ""), "confidence_pct": 0
        }
        self.cm.update_state({"last_setup": new_setup})
        return f"Сетап обновлён: {new_setup['direction']}."

    def _get_analysis(self, a):
        an = self.cm.get_last_analysis()
        if not an: return "Анализ не проводился."
        return (f"Цена: {an.get('current_price','?')}, Изменение 24ч: {an.get('price_change_24h_pct','?')}%\n"
                f"Bias: {an.get('orderbook_bias','?')}, OI: {an.get('oi_trend','?')}\n"
                f"Funding: {an.get('funding_rate_status','?')}, F&G: {an.get('fear_greed_state','?')}\n"
                f"Новости: {an.get('news_impact','?')}\nСводка: {an.get('summary','')}")

    def _get_trade(self, a):
        t = self.cm.get_active_trade()
        if not t: return "Открытых позиций нет."
        return f"Позиция: {t.get('direction','?')}, Вход: {t.get('entry_price','?')}, SL: {t.get('stop_loss','?')}, TP: {t.get('take_profit',[])}"

    def _get_stats(self, a):
        s = self.cm.get_trade_stats()
        return (f"Сделок: {s['total_trades']}, Win Rate: {s['win_rate']}%\n"
                f"P&L: {s['total_pnl_pct']:+.2f}%, Ср. Win: {s['avg_win_pct']}%, Ср. Loss: {s['avg_loss_pct']}%")

    def _get_news(self, a):
        """Возвращает новости, используя кеш, если он есть."""
        if not self.fetcher: return "Нет доступа к новостям."
        try:
            # Если есть кеш, возвращаем его
            if self._news_cache_local is not None:
                logger.info(f"[{self.symbol}] Возвращаем кешированные новости")
                return self._news_cache_local

            # Иначе загружаем новости
            df = self.fetcher.news_fetcher.get_news_for_symbol(self.symbol, max_items=10, hours_back=24)
            result = self.fetcher.news_fetcher.format_news_for_prompt(df)
            # Сохраняем в кеш
            self._news_cache_local = result
            return result
        except Exception as e:
            return f"Ошибка загрузки новостей: {e}"

    def _get_news_analysis(self, a):
        """Возвращает расширенный анализ новостей, используя кеш для форматированного текста."""
        if not self.fetcher: return "Нет доступа к новостям."
        try:
            # Если есть кеш для анализа, возвращаем его
            if self._news_analysis_cache is not None:
                logger.info(f"[{self.symbol}] Возвращаем кешированный анализ новостей")
                return self._news_analysis_cache

            # Загружаем данные
            df = self.fetcher.news_fetcher.get_news_for_symbol(self.symbol, max_items=15, hours_back=24)
            if df.empty:
                result = "Новостей не найдено."
                self._news_analysis_cache = result
                self._news_cache_local = result  # также сохраняем для get_news
                return result

            sentiment = self.fetcher.news_fetcher.assess_news_sentiment(df)
            news_text = self.fetcher.news_fetcher.format_news_for_prompt(df)
            result = f"=== РАСШИРЕННЫЙ АНАЛИЗ НОВОСТЕЙ ===\nТональность: {sentiment['impact']}\nПозитивных: {sentiment['positive_count']}, Негативных: {sentiment['negative_count']}\n\n{news_text}"
            # Сохраняем в кеши
            self._news_analysis_cache = result
            self._news_cache_local = news_text  # чтобы get_news мог использовать
            return result
        except Exception as e:
            return f"Ошибка: {e}"
