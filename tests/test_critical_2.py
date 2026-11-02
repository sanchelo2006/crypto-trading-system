#!/usr/bin/env python3
"""
Дополнительные unit-тесты для CryptoBERT-сентимента и агрегации трейдов.
Запуск: python test_critical_2.py
"""

import sys
import os
import unittest

import pandas as pd
import numpy as np

from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from news_fetcher import NewsAggregator, cryptobert_sentiment
from data_fetcher import BybitDataFetcher

class TestCryptoBERTSentiment(unittest.TestCase):
    """Тесты для CryptoBERT-анализатора и сентимента новостей."""

    def setUp(self):
        self.aggregator = NewsAggregator(deduplicate=False, use_ml=True)
        self.aggregator_keywords = NewsAggregator(deduplicate=False, use_ml=False)

    def _make_news_df(self, titles):
        """Создаёт тестовый DataFrame с новостями."""
        data = []
        for title in titles:
            data.append({
                "date": "2026-08-04 12:00",
                "title": title,
                "summary": f"Summary for {title}",
                "url": "http://example.com",
                "source": "test",
                "duplicate_count": 1,
                "duplicate_sources": ["test"],
            })
        return pd.DataFrame(data)

    def test_ml_returns_expected_format(self):
        """CryptoBERT должен вернуть словарь с ключами impact, method и счётчиками."""
        df = self._make_news_df([
            "Bitcoin surges 10% after ETF approval",
            "Market crashes amid regulatory fears",
        ])
        result = self.aggregator.assess_news_sentiment(df)
        self.assertIn("impact", result)
        self.assertIn("method", result)
        self.assertIn("positive_count", result)
        self.assertIn("negative_count", result)
        self.assertIn("neutral_count", result)
        # Метод должен быть cryptobert или keywords
        self.assertIn(result["method"], ["cryptobert", "keywords"])

    def test_empty_dataframe_returns_neutral(self):
        """Пустой DataFrame должен давать нейтральный результат."""
        result = self.aggregator.assess_news_sentiment(pd.DataFrame())
        self.assertEqual(result["impact"], "neutral")
        self.assertEqual(result["method"], "none")

    def test_keyword_fallback(self):
        """При отключённом ML должен работать keyword-анализ."""
        df = self._make_news_df([
            "Bitcoin bullish surge rally moon",
            "Market crash dump hack bearish",
        ])
        result = self.aggregator_keywords.assess_news_sentiment(df)
        self.assertEqual(result["method"], "keywords")
        self.assertGreater(result["positive_count"], 0)
        self.assertGreater(result["negative_count"], 0)

    def test_cryptobert_batch_analysis(self):
        """CryptoBERT должен уметь обрабатывать список текстов (даже если модель не загружена, возвращает neutral)."""
        texts = [
            "The company reported strong earnings and raised guidance.",
            "There are significant layoffs and the stock plummeted.",
            "The weather is nice today.",
        ]
        results = cryptobert_sentiment.analyze_batch(texts)
        self.assertEqual(len(results), 3)
        for sentiment, confidence in results:
            self.assertIn(sentiment, ["positive", "negative", "neutral"])
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)

class TestTradesAggregation(unittest.TestCase):
    """Тесты для агрегации трейдов."""

    def setUp(self):
        self.fetcher = BybitDataFetcher(testnet=True)

    def test_empty_trades(self):
        """Пустой DataFrame трейдов должен вернуть нулевой словарь."""
        result = self.fetcher._aggregate_trades(pd.DataFrame())
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["buy_volume"], 0)
        self.assertEqual(result["sell_volume"], 0)

    def test_aggregate_basic(self):
        """Агрегация с одним buy и одним sell."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-08-04 12:00", periods=2, freq="1s", tz="UTC"),
            "price": [1.5, 1.6],
            "volume": [100.0, 200.0],
            "side": ["Buy", "Sell"],
        })
        result = self.fetcher._aggregate_trades(df)
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["buy_volume"], 100.0)
        self.assertEqual(result["sell_volume"], 200.0)
        self.assertAlmostEqual(result["buy_pct"], 100 / 300 * 100, 1)

    def test_top_prices(self):
        """Проверка топ-цен."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-08-04 12:00", periods=3, freq="1s", tz="UTC"),
            "price": [1.5, 1.5, 1.6],
            "volume": [100.0, 50.0, 200.0],
            "side": ["Buy", "Sell", "Sell"],
        })
        result = self.fetcher._aggregate_trades(df)
        self.assertEqual(len(result["top_prices"]), 2)  # две уникальные цены
        # Цена 1.6 с объёмом 200 должна быть первой
        self.assertAlmostEqual(result["top_prices"][0]["price"], 1.6)
        self.assertAlmostEqual(result["top_prices"][0]["volume"], 200.0)

if __name__ == "__main__":
    print("=" * 60)
    print("ЗАПУСК ДОПОЛНИТЕЛЬНЫХ UNIT-ТЕСТОВ ТОРГОВОГО БОТА")
    print("=" * 60)
    unittest.main(verbosity=2)
