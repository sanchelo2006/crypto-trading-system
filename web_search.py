"""
Search tool for the DeepSeek model.
Uses the DuckDuckGo Instant Answer API (free, no key required).
"""
import logging
import requests
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


# Marker returned by search_formatted() when there are no results.
# search_for_trading_rules() checks it, so both places must stay in sync.
_NO_RESULTS_MARKER = "Nothing found"


class WebSearcher:
    """
    Information search via DuckDuckGo.
    Returns text results for use by the model.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; TradingSystem/1.0; +https://example.com)"
        })

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Search via the DuckDuckGo Instant Answer API.
        Returns a list of results: {title, snippet, url}.
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

            # Abstract (short answer)
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

            # If DuckDuckGo returned nothing, try the fallback
            if not results:
                results = self._fallback_search(query, max_results)

        except Exception as e:
            logger.warning(f"DuckDuckGo search error for '{query}': {e}")
            results = self._fallback_search(query, max_results)

        return results[:max_results]

    def _fallback_search(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """
        Fallback search via the Wikipedia API.
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
            logger.warning(f"Wikipedia error for '{query}': {e}")
            return []

    def search_formatted(self, query: str, max_results: int = 5) -> str:
        """
        Search with formatted output for the model.
        Returns a string: Title: ... Content: ... URL: ...
        """
        results = self.search(query, max_results)
        if not results:
            return f"{_NO_RESULTS_MARKER} for the query '{query}'."

        lines = [f"### Search results: {query}\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "Untitled")
            snippet = r.get("snippet", "")[:300]
            url = r.get("url", "")
            lines.append(f"{i}. **{title}**")
            lines.append(f"   {snippet}")
            if url:
                lines.append(f"   URL: {url}")
        return "\n".join(lines)

    def search_for_trading_rules(self, symbol: str, base_asset: str) -> str:
        """
        Specialized search: collects information about trading a specific coin.
        Runs several queries and merges the results.
        """
        queries = [
            f"{base_asset} cryptocurrency trading strategy intraday 2025 2026",
            f"{base_asset} risk management stop loss volatility",
            f"{base_asset} price drivers news impact analysis",
        ]

        all_results = []
        for query in queries:
            result = self.search_formatted(query, max_results=3)
            if _NO_RESULTS_MARKER not in result:
                all_results.append(result)

        if not all_results:
            return f"No search information found for {symbol} ({base_asset})."

        return "\n\n".join(all_results)