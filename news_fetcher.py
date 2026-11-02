"""
Модуль для получения и анализа крипто-новостей через RSS-ленты.
Использует CryptoBERT (ElKulako/cryptobert) для точного сентимент-анализа с fallback на ключевые слова.
Добавлена дедупликация по сходству заголовков.
"""
import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from difflib import SequenceMatcher

import feedparser
import pandas as pd

logger = logging.getLogger(__name__)

# ==================== Вспомогательные функции ====================
def _similar(a: str, b: str) -> float:
    """Возвращает коэффициент сходства двух строк (0.0 – 1.0)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def _deduplicate_articles(articles: List[Dict], threshold: float = 0.8) -> List[Dict]:
    """
    Группирует статьи с похожими заголовками и оставляет одну с пометкой о количестве источников.
    """
    if not articles:
        return articles
    used = [False] * len(articles)
    result = []
    for i, art in enumerate(articles):
        if used[i]:
            continue
        duplicates = [art]
        used[i] = True
        for j in range(i + 1, len(articles)):
            if not used[j] and _similar(art.get("title", ""), articles[j].get("title", "")) > threshold:
                duplicates.append(articles[j])
                used[j] = True
        if len(duplicates) > 1:
            rep = duplicates[0].copy()
            rep["duplicate_count"] = len(duplicates)
            rep["duplicate_sources"] = list(set(d.get("source", "?") for d in duplicates))
            rep["title"] = rep.get("title", "") + f" (повторяется в {len(duplicates)} источниках)"
            result.append(rep)
        else:
            result.append(art)
    return result

# ==================== CryptoBERT Sentiment Analyzer ====================
class CryptoBERTSentiment:
    """
    Сентимент-анализатор на основе CryptoBERT (ElKulako/cryptobert).
    Ленивая загрузка (синглтон), потокобезопасный.
    """
    _instance = None
    _pipeline = None
    _initialized = False
    _init_error = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load_model(self):
        if self._initialized:
            return self._pipeline is not None
        self._initialized = True
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            import torch

            model_name = "ElKulako/cryptobert"
            logger.info(f"Загрузка CryptoBERT: {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                return_all_scores=True,
                truncation=True,
                max_length=512
            )
            logger.info("CryptoBERT успешно загружен")
            return True
        except Exception as e:
            self._init_error = str(e)
            logger.warning(
                f"Не удалось загрузить CryptoBERT: {e}. "
                f"Будет использован fallback (ключевые слова)."
            )
            return False

    def analyze(self, text: str) -> Tuple[str, float]:
        """Возвращает (sentiment, confidence) для одного текста."""
        if not self._load_model():
            return "neutral", 0.0
        if not text or not text.strip():
            return "neutral", 0.0
        try:
            results = self._pipeline(text)
            # results — это список словарей. 
            # При return_all_scores=True каждый элемент списка — список всех меток с их скорами.
            # Формат: [[{'label': 'Bearish', 'score': 0.1}, {'label': 'Neutral', 'score': 0.8}, ...]]
            
            # Проверяем структуру
            if isinstance(results, list) and len(results) > 0:
                first_item = results[0]
                if isinstance(first_item, list):
                    # Это список словарей — стандартный формат return_all_scores
                    scores = first_item
                elif isinstance(first_item, dict) and 'label' in first_item:
                    # Это один словарь (без return_all_scores) — оборачиваем в список
                    scores = results
                else:
                    logger.warning(f"Неожиданный формат ответа CryptoBERT: {type(results)} {str(results)[:200]}")
                    return "neutral", 0.0
                
                best = max(scores, key=lambda x: x['score'] if isinstance(x, dict) else 0)
                if isinstance(best, dict):
                    label = best.get('label', '').lower()
                    score = best.get('score', 0.5)
                    if 'bearish' in label or 'negative' in label or 'label_0' in label:
                        return "negative", score
                    elif 'bullish' in label or 'positive' in label or 'label_1' in label:
                        return "positive", score
                    else:
                        return "neutral", score
            
            return "neutral", 0.0
        except Exception as e:
            logger.error(f"Ошибка анализа CryptoBERT: {e}")
            return "neutral", 0.0

    def analyze_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        if not self._load_model() or not texts:
            return [("neutral", 0.0)] * len(texts)
        results = []
        for text in texts:
            results.append(self.analyze(text))
        return results

# Глобальный экземпляр (синглтон)
cryptobert_sentiment = CryptoBERTSentiment()

# ==================== News Aggregator ====================
class NewsAggregator:
    RSS_FEEDS = {
        "beincrypto": "https://beincrypto.com/feed",
        "binance_blog": "https://www.binance.com/en/blog/feed",
        "bitcoin_com_news": "https://news.bitcoin.com/feed/",
        "bitfinex_blog": "https://blog.bitfinex.com/feed/",
        "blockworks": "https://blockworks.co/rss",
        "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss",
        "coingape": "https://coingape.com/feed/",
        "coingecko_learn": "https://www.coingecko.com/learn/feed",
        "cointelegraph": "https://cointelegraph.com/rss",
        "crypto_briefing": "https://cryptobriefing.com/feed/",
        "cryptopanic": "https://cryptopanic.com/api/v2/posts/?format=rss",
        "cryptoslate": "https://cryptoslate.com/feed/",
        "decrypt": "https://decrypt.co/feed",
        "kucoin_blog": "https://www.kucoin.com/blog/rss.xml",
        "newsbtc": "https://www.newsbtc.com/feed/",
        "the_block": "https://www.theblock.co/rss",
        "u_today": "https://u.today/rss",
    }

    SYMBOL_KEYWORDS = {
        "BTCUSDT": "Bitcoin OR BTC",
        "ETHUSDT": "Ethereum OR ETH",
        "SOLUSDT": "Solana OR SOL",
        "XRPUSDT": "Ripple OR XRP",
        "ADAUSDT": "Cardano OR ADA",
        "DOGEUSDT": "Dogecoin OR DOGE",
        "AVAXUSDT": "Avalanche OR AVAX",
        "DOTUSDT": "Polkadot OR DOT",
        "LINKUSDT": "Chainlink OR LINK",
        "MATICUSDT": "Polygon OR MATIC",
        "HYPEUSDT": "Hyperliquid OR HYPE",
    }

    def __init__(self, deduplicate: bool = True, use_ml: bool = True):
        self.deduplicate = deduplicate
        self.use_ml = use_ml

    @staticmethod
    def _extract_base_asset(symbol: str) -> str:
        return re.sub(r"(USDT|USDC|BUSD|USD)$", "", symbol).upper()

    def fetch_articles_from_feeds(self, limit: int = 50) -> List[Dict]:
        articles = []
        for source_name, feed_url in self.RSS_FEEDS.items():
            try:
                logger.debug(f"Парсинг RSS: {source_name}")
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:limit]:
                    articles.append({
                        "date": entry.get("published", datetime.now().isoformat()),
                        "title": entry.get("title", "No Title"),
                        "summary": entry.get("summary", ""),
                        "url": entry.get("link", ""),
                        "source": source_name
                    })
            except Exception as e:
                logger.warning(f"Ошибка парсинга {source_name}: {e}")
        logger.info(f"Всего статей из RSS: {len(articles)}")
        if self.deduplicate and articles:
            articles = _deduplicate_articles(articles)
            logger.info(f"После дедупликации: {len(articles)} статей")
        return articles

    def filter_articles_by_symbol(self, articles: List[Dict], coin_symbol: str) -> List[Dict]:
        coin_lower = coin_symbol.lower()
        filtered = [a for a in articles if coin_lower in a["title"].lower()]
        logger.info(f"После фильтрации по '{coin_symbol}': {len(filtered)} статей (из {len(articles)})")
        return filtered

    def _parse_date(self, date_val) -> datetime:
        if isinstance(date_val, datetime):
            if date_val.tzinfo is not None:
                return date_val.replace(tzinfo=None)
            return date_val
        try:
            from dateutil import parser
            dt = parser.parse(date_val)
            if dt.tzinfo is not None:
                return dt.replace(tzinfo=None)
            return dt
        except Exception:
            return datetime.now()

    @staticmethod
    def _extractive_summary(text: str, max_sentences: int = 3) -> str:
        if not text:
            return ""
        clean = re.sub(r"<[^>]+>", "", text)
        sentences = re.split(r"(?<=[.!?])\s+", clean)
        return " ".join(sentences[:max_sentences]) if sentences else clean[:200]

    def fetch_news(self, coin_symbol: str, source: str = "rss",
                   limit: int = 20, max_age_days: int = 30) -> pd.DataFrame:
        articles = self.fetch_articles_from_feeds(limit=limit * 2)
        if not articles:
            logger.warning(f"Нет статей из RSS-лент для {coin_symbol}")
            return pd.DataFrame()
        filtered = self.filter_articles_by_symbol(articles, coin_symbol)
        if max_age_days is not None:
            cutoff = datetime.now() - timedelta(days=max_age_days)
            filtered = [a for a in filtered if self._parse_date(a.get("date")) >= cutoff]
        logger.info(f"После фильтрации по дате (<{max_age_days} дн): {len(filtered)} статей")
        if not filtered:
            return pd.DataFrame()
        data = []
        for article in filtered[:limit]:
            text_to_summarize = article.get("summary", article["title"])
            summary = self._extractive_summary(text_to_summarize)
            data.append({
                "date": article.get("date", "N/A"),
                "title": article["title"],
                "summary": summary,
                "url": article.get("url", "N/A"),
                "source": article.get("source", "N/A"),
                "duplicate_count": article.get("duplicate_count", 0),
                "duplicate_sources": article.get("duplicate_sources", []),
            })
        logger.info(f"Новостей собрано для {coin_symbol}: {len(data)}")
        return pd.DataFrame(data)

    def get_news_for_symbol(self, symbol: str, max_items: int = 15,
                            hours_back: int = 24) -> pd.DataFrame:
        base = self._extract_base_asset(symbol)
        logger.info(f"Поиск новостей для {symbol} (базовый актив: {base})")
        max_age_days = max(1, hours_back // 24)
        df = self.fetch_news(base, limit=max_items, max_age_days=max_age_days)
        if not df.empty and hours_back < 24:
            cutoff = datetime.now() - timedelta(hours=hours_back)
            try:
                df = df[df["date"].apply(lambda d: self._parse_date(d) >= cutoff.replace(tzinfo=None))]
            except Exception:
                pass
        return df

    def format_news_for_prompt(self, df: pd.DataFrame) -> str:
        if df.empty:
            return "Новости не найдены за указанный период."
        lines = [f"### Новостной фон ({len(df)} новостей):\n"]
        for i, (_, row) in enumerate(df.iterrows(), 1):
            title = row.get("title", "Без заголовка")
            source = row.get("source", "?")
            date = row.get("date", "?")
            summary = row.get("summary", "")
            duplicate_count = row.get("duplicate_count", 0)
            try:
                if isinstance(date, datetime):
                    time_str = date.strftime("%H:%M UTC")
                else:
                    dt = datetime.fromisoformat(str(date).replace("Z", "+00:00"))
                    time_str = dt.strftime("%H:%M UTC")
            except Exception:
                time_str = str(date)[:16]
            lines.append(f"{i}. [{source}] {title} ({time_str})")
            if duplicate_count > 1:
                lines.append(f"   ⚠️ Эта новость повторяется в {duplicate_count} источниках")
            if summary:
                lines.append(f"   {summary}")
        return "\n".join(lines)

    def assess_news_sentiment(self, df: pd.DataFrame) -> Dict:
        if df.empty:
            return {
                "impact": "neutral",
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "keywords": [],
                "method": "none"
            }

        # Пытаемся использовать CryptoBERT
        if self.use_ml:
            try:
                texts = []
                for _, row in df.iterrows():
                    text = f"{row.get('title', '')} {row.get('summary', '')}"
                    texts.append(text.strip())
                results = cryptobert_sentiment.analyze_batch(texts)

                pos, neg, neu = 0, 0, 0
                for (s, _), (_, row) in zip(results, df.iterrows()):
                    w = max(1, row.get("duplicate_count", 0))
                    if s == "positive":
                        pos += w
                    elif s == "negative":
                        neg += w
                    else:
                        neu += w

                if neg > pos and neg >= 2:
                    impact = "negative"
                elif pos > neg and pos >= 2:
                    impact = "positive"
                else:
                    impact = "neutral"

                logger.info(f"Сентимент (CryptoBERT): {impact} (pos={pos}, neg={neg}, neu={neu})")
                return {
                    "impact": impact,
                    "positive_count": pos,
                    "negative_count": neg,
                    "neutral_count": neu,
                    "keywords": [],
                    "method": "cryptobert"
                }
            except Exception as e:
                logger.warning(f"Ошибка CryptoBERT анализа, использую fallback: {e}")

        # Fallback: ключевые слова
        return self._keyword_sentiment(df)

    def _keyword_sentiment(self, df: pd.DataFrame) -> Dict:
        positive_words = [
            "bullish", "surge", "rally", "breakout", "pump", "moon", "launch",
            "upgrade", "partnership", "adoption", "listing", "buy", "green",
            "record", "high", "gain", "profit", "growth", "support", "approval",
            "etf", "approved", "accumulation", "whale", "buying"
        ]
        negative_words = [
            "bearish", "crash", "dump", "hack", "exploit", "ban", "regulation",
            "sec", "lawsuit", "fine", "crackdown", "delist", "sell", "red",
            "loss", "decline", "drop", "fear", "liquidat", "bankrupt",
            "investigation", "warning", "risk"
        ]
        pos, neg = 0, 0
        all_keywords = []
        for _, row in df.iterrows():
            w = max(1, row.get("duplicate_count", 0))
            text = f"{row.get('title', '')} {row.get('summary', '')}".lower()
            pos_found = [kw for kw in positive_words if kw in text]
            neg_found = [kw for kw in negative_words if kw in text]
            if pos_found:
                pos += w
                all_keywords.extend(pos_found)
            if neg_found:
                neg += w
                all_keywords.extend(neg_found)
        total = len(df)
        neu = max(0, total - pos - neg)
        impact = "negative" if (neg > pos and neg >= 2) else (
            "positive" if (pos > neg and pos >= 2) else "neutral"
        )
        logger.info(f"Сентимент (keywords): {impact} (pos={pos}, neg={neg}, neu={neu})")
        return {
            "impact": impact,
            "positive_count": pos,
            "negative_count": neg,
            "neutral_count": neu,
            "keywords": list(set(all_keywords))[:10],
            "method": "keywords"
        }
