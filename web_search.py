"""
Поисковый инструмент для модели DeepSeek.
Использует DuckDuckGo Instant Answer API (бесплатно, без ключа).
"""
import logging
import requests
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class WebSearcher:
    """
    Поиск информации через DuckDuckGo.
    Возвращает текстовые результаты для использования моделью.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; TradingSystem/1.0; +https://example.com)"
        })

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Поиск через DuckDuckGo Instant Answer API.
        Возвращает список результатов: {title, snippet, url}.
        """
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        }

        results = []
        try:
            resp = self.session.get(url, params=params, timeout=10)
            data = resp.json()

            # Abstract (краткий ответ)
            if data.get("AbstractText"):
                results.append({
                    "title": data.get("AbstractSource", "DuckDuckGo"),
                    "snippet": data.get("AbstractText", ""),
                    "url": data.get("AbstractURL", "")
                })

            # Related topics
            for topic in data.get("RelatedTopics", [])[:max_results - len(results)]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({
                        "title": topic.get("FirstURL", "").split("/")[-1].replace("_", " ").title() if topic.get("FirstURL") else "Related",
                        "snippet": topic.get("Text", ""),
                        "url": topic.get("FirstURL", "")
                    })

            # Если DuckDuckGo не дал результатов, пробуем запасной вариант
            if not results:
                results = self._fallback_search(query, max_results)

        except Exception as e:
            logger.warning(f"Ошибка поиска DuckDuckGo для '{query}': {e}")
            results = self._fallback_search(query, max_results)

        return results[:max_results]

    def _fallback_search(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """
        Запасной поиск через Wikipedia API.
        """
        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": max_results,
            }
            resp = self.session.get(url, params=params, timeout=10)
            data = resp.json()
            results = []
            for item in data.get("query", {}).get("search", []):
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "url": f"https://en.wikipedia.org/wiki/{item.get('title', '').replace(' ', '_')}"
                })
            return results
        except Exception as e:
            logger.warning(f"Ошибка Wikipedia для '{query}': {e}")
            return []

    def search_formatted(self, query: str, max_results: int = 5) -> str:
        """
        Поиск с форматированным выводом для модели.
        Возвращает строку: Заголовок: ... Содержание: ... URL: ...
        """
        results = self.search(query, max_results)
        if not results:
            return f"По запросу '{query}' ничего не найдено."

        lines = [f"### Результаты поиска: {query}\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "Без названия")
            snippet = r.get("snippet", "")[:300]
            url = r.get("url", "")
            lines.append(f"{i}. **{title}**")
            lines.append(f"   {snippet}")
            if url:
                lines.append(f"   URL: {url}")
        return "\n".join(lines)

    def search_for_trading_rules(self, symbol: str, base_asset: str) -> str:
        """
        Специализированный поиск: собирает информацию о торговле конкретной монетой.
        Выполняет несколько запросов и объединяет результаты.
        """
        queries = [
            f"{base_asset} cryptocurrency trading strategy intraday 2025 2026",
            f"{base_asset} криптовалюта торговля стратегия внутридневная",
            f"{base_asset} risk management stop loss volatility",
            f"{base_asset} price drivers news impact analysis",
        ]

        all_results = []
        for query in queries:
            result = self.search_formatted(query, max_results=3)
            if "ничего не найдено" not in result:
                all_results.append(result)

        if not all_results:
            return f"Информация по {symbol} ({base_asset}) в поиске не найдена."

        return "\n\n".join(all_results)
