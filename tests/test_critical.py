#!/usr/bin/env python3
"""
Unit-тесты для критических функций торгового бота.
Запуск: python test_critical.py

Проверяемые функции:
  - safe_parse_json (utils/helpers.py)
  - _fix_json_strings (utils/helpers.py)
  - _compress_events (memory_manager.py)
  - parse_rules_from_text (nicegui_app.py)
  - calculate_indicators (indicators.py) – включён адаптивный Volume Profile
  - _compute_market_stats (data_fetcher.py)
  - _deduplicate_articles (news_fetcher.py)
  - _validate_conditions (ai_client.py)
"""

import sys
import os
import unittest
import json
import shutil
import pandas as pd
import numpy as np

from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import safe_parse_json, _fix_json_strings
from memory_manager import CoinMemory
from nicegui_app import parse_rules_from_text
from indicators import calculate_indicators, calculate_volume_profile
from news_fetcher import _deduplicate_articles
from ai_client import DeepSeekClient

class TestSafeParseJson(unittest.TestCase):
    def test_valid_json(self):
        result = safe_parse_json('{"action": "ANALYSIS", "current_price": 1.5}')
        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "ANALYSIS")
        self.assertEqual(result["current_price"], 1.5)

    def test_empty_string(self):
        self.assertIsNone(safe_parse_json(""))

    def test_none_input(self):
        self.assertIsNone(safe_parse_json(None))

    def test_plain_text_not_json(self):
        self.assertIsNone(safe_parse_json("Привет, я модель DeepSeek!"))

    def test_json_with_newlines_in_strings(self):
        raw = '{"summary": "Анализ проведён\\nв 13:00 UTC.\\nРынок стабилен."}'
        result = safe_parse_json(raw)
        self.assertIsNotNone(result)
        self.assertIn("Анализ проведён", result["summary"])

    def test_json_in_markdown_block(self):
        raw = (
            'Вот мой анализ:\n```json\n'
            '{"action": "SIGNAL", "direction": "long", "confidence_pct": 75}\n```\n'
            'Надеюсь, это поможет.'
        )
        result = safe_parse_json(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "SIGNAL")
        self.assertEqual(result["direction"], "long")
        self.assertEqual(result["confidence_pct"], 75)

    def test_unclosed_json(self):
        result = safe_parse_json('{"action": "ANALYSIS", "summary": "текст')
        if result is not None:
            self.assertIsInstance(result, dict)

    def test_json_with_extra_text(self):
        raw = 'Данные получены. {"price": 100.5, "volume": 1200} Конец сообщения.'
        result = safe_parse_json(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result["price"], 100.5)
        self.assertEqual(result["volume"], 1200)

class TestFixJsonStrings(unittest.TestCase):
    def test_already_valid(self):
        original = '{"key": "value", "num": 42}'
        fixed = _fix_json_strings(original)
        self.assertEqual(json.loads(fixed), json.loads(original))

    def test_newlines_inside_string(self):
        raw = '{"text": "строка\nс переносом"}'
        fixed = _fix_json_strings(raw)
        parsed = json.loads(fixed)
        self.assertIn("строка", parsed["text"])
        self.assertIn("с переносом", parsed["text"])

    def test_tabs_inside_string(self):
        raw = '{"text": "колонка1\tколонка2"}'
        fixed = _fix_json_strings(raw)
        parsed = json.loads(fixed)
        self.assertIn("колонка1", parsed["text"])
        self.assertIn("колонка2", parsed["text"])

    def test_nested_json_structure(self):
        raw = (
            '{"analysis": {"supports": [1.0, 2.0], '
            '"text": "многострочный\\nтекст"}}'
        )
        fixed = _fix_json_strings(raw)
        parsed = json.loads(fixed)
        self.assertEqual(parsed["analysis"]["supports"], [1.0, 2.0])

class TestCompressEvents(unittest.TestCase):
    def setUp(self):
        self.cm = CoinMemory("TEST_COMPRESSION")
        self._cleanup_path = self.cm.data_dir

    def tearDown(self):
        if self._cleanup_path.exists():
            shutil.rmtree(self._cleanup_path, ignore_errors=True)

    def test_empty_events(self):
        self.assertEqual(self.cm._compress_events([]), [])

    def test_single_event(self):
        event = "• 2026-07-14 10:00 UTC — касание pivot (1.2450)"
        result = self.cm._compress_events([event])
        self.assertEqual(len(result), 1)
        self.assertIn("касание pivot", result[0])

    def test_compression_grouping(self):
        events = [
            "• 2026-07-14 10:00 UTC — касание support_1 (1.2300)",
            "• 2026-07-14 10:15 UTC — касание support_1 (1.2300)",
            "• 2026-07-14 10:30 UTC — пробой support_1 (1.2300) вниз",
        ]
        result = self.cm._compress_events(events)
        self.assertEqual(len(result), 1)
        self.assertIn("support_1", result[0])
        self.assertIn("касаний", result[0])
        self.assertIn("пробоев", result[0])

    def test_different_levels_no_compression(self):
        events = [
            "• 2026-07-14 10:00 UTC — касание support_1 (1.2300)",
            "• 2026-07-14 10:05 UTC — касание support_2 (1.2200)",
            "• 2026-07-14 10:10 UTC — касание resistance_1 (1.2500)",
        ]
        self.assertEqual(len(self.cm._compress_events(events)), 3)

    def test_anomaly_volume_not_grouped(self):
        events = [
            "• 2026-07-14 10:00 UTC — аномальный объём (5000, в 2.5x выше среднего)",
            "• 2026-07-14 10:05 UTC — аномальный объём (4800, в 2.4x выше среднего)",
        ]
        self.assertEqual(len(self.cm._compress_events(events)), 2)

class TestParseRulesFromText(unittest.TestCase):
    def test_standard_rules_format(self):
        text = (
            "## Правила торговли\n"
            "- Правило 1: условие A\n"
            "- Правило 2: условие B\n\n"
            "## Риск-менеджмент\n"
            "- Риск-правило 1: стоп-лосс 2%\n"
            "- Риск-правило 2: максимум 5 сделок в день"
        )
        trading, risk = parse_rules_from_text(text)
        self.assertEqual(len(trading), 2)
        self.assertEqual(len(risk), 2)

    def test_empty_text(self):
        trading, risk = parse_rules_from_text("")
        self.assertEqual(trading, [])
        self.assertEqual(risk, [])

    def test_only_trading_rules(self):
        trading, risk = parse_rules_from_text("## Правила торговли\n- Только одно правило")
        self.assertEqual(len(trading), 1)
        self.assertEqual(len(risk), 0)

    def test_no_section_headers(self):
        trading, risk = parse_rules_from_text("- Правило 1\n- Правило 2\n- Правило 3")
        self.assertEqual(len(trading), 3)

    def test_numbered_list(self):
        trading, risk = parse_rules_from_text("## Правила торговли\n1. Первое правило\n2. Второе правило")
        self.assertEqual(len(trading), 2)

class TestCalculateIndicators(unittest.TestCase):
    def _make_test_df(self, n_candles=50, volatility=0.01):
        np.random.seed(42)
        base_price = 1.5
        dates = pd.date_range("2026-07-14", periods=n_candles, freq="15min", tz="UTC")
        closes = base_price + np.cumsum(np.random.randn(n_candles) * volatility)
        highs = closes + np.abs(np.random.randn(n_candles) * 0.005)
        lows = closes - np.abs(np.random.randn(n_candles) * 0.005)
        opens = closes + np.random.randn(n_candles) * 0.002
        volumes = np.abs(np.random.randn(n_candles) * 1000 + 5000)
        return pd.DataFrame({
            "timestamp": dates, "open": opens, "high": highs,
            "low": lows, "close": closes, "volume": volumes,
        })

    def test_basic_calculation(self):
        result = calculate_indicators(self._make_test_df(50), num_latest=30)
        self.assertIsInstance(result, dict)
        for k in ("rsi_14", "sma_20", "sma_50", "pivot", "adx_14"):
            self.assertIn(k, result)

    def test_empty_dataframe(self):
        self.assertEqual(calculate_indicators(pd.DataFrame()), {})

    def test_too_few_candles(self):
        self.assertEqual(calculate_indicators(self._make_test_df(10)), {})

    def test_with_timestamps(self):
        _, timestamps = calculate_indicators(self._make_test_df(50), num_latest=5, return_timestamps=True)
        self.assertEqual(len(timestamps), 5)

    def test_pivot_levels_are_floats(self):
        result = calculate_indicators(self._make_test_df(50))
        self.assertGreater(result["resistance_1"], result["pivot"])
        self.assertGreater(result["pivot"], result["support_1"])

    def test_vwap_daily_reset(self):
        timestamps = []
        for h in range(23, 24):
            for m in (0, 15, 30, 45):
                timestamps.append(pd.Timestamp(f"2026-07-14 {h}:{m:02d}", tz="UTC"))
        for h in range(0, 4):
            for m in (0, 15, 30, 45):
                timestamps.append(pd.Timestamp(f"2026-07-15 {h}:{m:02d}", tz="UTC"))
        timestamps = timestamps[:20]
        n = len(timestamps)
        prices_day1 = 1.0
        prices_day2 = 2.0
        closes = [prices_day1] * 4 + [prices_day2] * (n - 4)
        highs = [p + 0.05 for p in closes]
        lows = [p - 0.05 for p in closes]
        opens = closes
        df = pd.DataFrame({
            "timestamp": timestamps, "open": opens, "high": highs,
            "low": lows, "close": closes, "volume": [100] * n
        })
        result = calculate_indicators(df, num_latest=20)
        self.assertIn("vwap", result)
        self.assertTrue(len(result["vwap"]) >= 2)
        self.assertAlmostEqual(result["vwap"][-1], 2.0, delta=0.5)

    def test_pivot_from_1h_candles(self):
        df = self._make_test_df(50)
        dates = pd.date_range("2026-07-10", periods=120, freq="h", tz="UTC")
        closes = 1.5 + np.cumsum(np.random.randn(120) * 0.01)
        df_1h = pd.DataFrame({
            "timestamp": dates, "open": closes, "high": closes + 0.02,
            "low": closes - 0.02, "close": closes + np.random.randn(120) * 0.005,
            "volume": np.random.randint(1000, 5000, 120)
        })
        result = calculate_indicators(df, df_1h_for_pivot=df_1h)
        self.assertGreater(result["resistance_1"], result["pivot"])
        self.assertGreater(result["pivot"], result["support_1"])

    def test_macd_present_and_valid(self):
        result = calculate_indicators(self._make_test_df(50), num_latest=30)
        for key in ('macd', 'macd_signal', 'macd_hist'):
            self.assertIn(key, result, f"Индикатор {key} отсутствует")
            self.assertIsInstance(result[key], list)
            self.assertTrue(len(result[key]) > 0)
            self.assertTrue(all(isinstance(v, (int, float)) for v in result[key]))

    def test_adx_present_and_valid(self):
        result = calculate_indicators(self._make_test_df(50), num_latest=30)
        self.assertIn("adx_14", result)
        self.assertIsInstance(result["adx_14"], list)
        self.assertTrue(len(result["adx_14"]) > 0)
        for val in result["adx_14"]:
            self.assertGreaterEqual(val, 0)
            self.assertLessEqual(val, 100)

    def test_volume_profile_when_requested(self):
        result = calculate_indicators(self._make_test_df(50), num_latest=30, include_volume_profile=True)
        self.assertIn('volume_profile', result)
        vp = result['volume_profile']
        self.assertIsInstance(vp, list)
        self.assertTrue(len(vp) > 0)
        self.assertLessEqual(len(vp), 5)
        for entry in vp:
            self.assertIsInstance(entry, (list, tuple))
            self.assertEqual(len(entry), 2)
            self.assertIsInstance(entry[0], float)
            self.assertIsInstance(entry[1], float)

    def test_volume_profile_adaptive_bins(self):
        df_high = self._make_test_df(50, volatility=0.1)
        vp_high = calculate_volume_profile(df_high, top_n=5)
        df_low = self._make_test_df(50, volatility=0.001)
        vp_low = calculate_volume_profile(df_low, top_n=5)
        self.assertTrue(len(vp_high) > 0)
        self.assertTrue(len(vp_low) > 0)
        for vp in (vp_high, vp_low):
            for level, vol in vp:
                self.assertIsInstance(level, float)
                self.assertIsInstance(vol, float)

    def test_volume_profile_fixed_bins(self):
        df = self._make_test_df(50)
        vp = calculate_volume_profile(df, num_bins=12, top_n=3)
        self.assertEqual(len(vp), 3)

    def test_volume_profile_absent_when_not_requested(self):
        result = calculate_indicators(self._make_test_df(50), num_latest=30)
        self.assertNotIn('volume_profile', result)

    def test_volume_profile_empty_on_flat_data(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-07-15", periods=50, freq="15min", tz="UTC"),
            "open": 2.0, "high": 2.0, "low": 2.0, "close": 2.0,
            "volume": np.random.randint(500, 3000, 50)
        })
        result = calculate_indicators(df, num_latest=30, include_volume_profile=True)
        self.assertIn('volume_profile', result)
        self.assertTrue(len(result['volume_profile']) >= 1)

class TestComputeMarketStats(unittest.TestCase):
    def _make_1h_df(self, n=48):
        np.random.seed(42)
        dates = pd.date_range("2026-07-14 00:00", periods=n, freq="h", tz="UTC")
        closes = 1.5 + np.cumsum(np.random.randn(n) * 0.005)
        return pd.DataFrame({
            "timestamp": dates, "open": closes, "high": closes + 0.01,
            "low": closes - 0.01, "close": closes, "volume": np.random.randint(5000, 15000, n)
        })

    def _make_15m_df(self, n=96):
        dates = pd.date_range("2026-07-14 00:00", periods=n, freq="15min", tz="UTC")
        closes = 1.5 + np.cumsum(np.random.randn(n) * 0.002)
        return pd.DataFrame({
            "timestamp": dates, "open": closes, "high": closes + 0.005,
            "low": closes - 0.005, "close": closes, "volume": np.random.randint(500, 3000, n)
        })

    def test_basic_metrics(self):
        from data_fetcher import BybitDataFetcher
        fetcher = BybitDataFetcher(testnet=True)
        stats = fetcher._compute_market_stats(
            self._make_1h_df(), self._make_15m_df(), pd.DataFrame()
        )
        self.assertIn("avg_volume_1h_24h", stats)
        self.assertIn("avg_volume_15m_24h", stats)
        self.assertIn("last_15m_volume_vs_avg_pct", stats)
        self.assertIsInstance(stats["avg_volume_1h_24h"], (int, float))

    def test_empty_data(self):
        from data_fetcher import BybitDataFetcher
        fetcher = BybitDataFetcher(testnet=True)
        stats = fetcher._compute_market_stats(None, None, pd.DataFrame())
        self.assertEqual(stats, {})

    def test_oi_change(self):
        from data_fetcher import BybitDataFetcher
        fetcher = BybitDataFetcher(testnet=True)
        fund = pd.DataFrame({
            "timestamp": pd.date_range("2026-07-14 00:00", periods=4, freq="h", tz="UTC"),
            "Open_Interest": [1000.0, 1020.0, 1040.0, 1060.0]
        })
        stats = fetcher._compute_market_stats(
            self._make_1h_df(), self._make_15m_df(), fund
        )
        self.assertIn("oi_change_4h_pct", stats)
        self.assertAlmostEqual(stats["oi_change_4h_pct"], 6.0, places=1)

class TestNewsDeduplication(unittest.TestCase):
    def _make_articles(self, titles):
        return [
            {"title": t, "source": f"source_{i}", "date": "2026-07-15 10:00",
             "summary": f"Summary for {t}", "url": f"http://example.com/{i}"}
            for i, t in enumerate(titles)
        ]

    def test_no_duplicates(self):
        articles = self._make_articles([
            "Bitcoin hits new ATH",
            "Ethereum merge completed",
            "Solana network outage"
        ])
        result = _deduplicate_articles(articles)
        self.assertEqual(len(result), 3)

    def test_exact_duplicates(self):
        articles = self._make_articles([
            "SEC approves Bitcoin ETF",
            "SEC approves Bitcoin ETF",
            "SEC approves Bitcoin ETF"
        ])
        result = _deduplicate_articles(articles)
        self.assertEqual(len(result), 1)
        self.assertIn("повторяется в 3 источниках", result[0]["title"])
        self.assertEqual(result[0]["duplicate_count"], 3)

    def test_similar_titles(self):
        articles = self._make_articles([
            "Breaking: SEC Approves Spot Bitcoin ETF",
            "BREAKING: SEC approves spot Bitcoin ETF",
            "SEC Approves Spot Bitcoin ETF - Breaking News"
        ])
        result = _deduplicate_articles(articles)
        self.assertLessEqual(len(result), 2)

    def test_mixed_unique_and_duplicates(self):
        articles = self._make_articles([
            "Bitcoin price surges 10%",
            "Ethereum upgrade scheduled",
            "Bitcoin price surges 10%",
            "Ripple wins court case",
            "Ethereum upgrade scheduled",
            "Bitcoin price surges 10%",
        ])
        result = _deduplicate_articles(articles)
        self.assertEqual(len(result), 3)
        for r in result:
            if "Bitcoin" in r["title"]:
                self.assertEqual(r["duplicate_count"], 3)
            elif "Ethereum" in r["title"]:
                self.assertEqual(r["duplicate_count"], 2)
            elif "Ripple" in r["title"]:
                self.assertEqual(r.get("duplicate_count", 0), 0)

    def test_empty_list(self):
        self.assertEqual(_deduplicate_articles([]), [])

class TestValidateTimeframes(unittest.TestCase):
    def setUp(self):
        self.validator = DeepSeekClient._validate_conditions

    def test_valid_timeframe_no_change(self):
        conditions = ["На 15m графике пробой R1 с RSI > 60"]
        available = {"15m", "1h"}
        fixed, warnings = self.validator(conditions, available)
        self.assertEqual(fixed[0], conditions[0])
        self.assertEqual(len(warnings), 0)

    def test_5m_replaced_with_15m(self):
        conditions = ["На 5m графике отскок от поддержки"]
        available = {"15m", "1h"}
        fixed, warnings = self.validator(conditions, available)
        self.assertIn("15m", fixed[0])
        self.assertNotIn("на 5m", fixed[0].lower())
        self.assertEqual(len(warnings), 1)
        self.assertIn("5m", warnings[0])

    def test_4h_replaced_with_1h(self):
        conditions = ["На 4h графике закрытие выше SMA50"]
        available = {"15m", "1h"}
        fixed, warnings = self.validator(conditions, available)
        self.assertIn("1h", fixed[0])
        self.assertNotIn("4h", fixed[0])
        self.assertEqual(len(warnings), 1)

    def test_multiple_conditions_mixed(self):
        conditions = [
            "На 15m пробой с объёмом",
            "На 5m отскок от VWAP",
            "На 1h закрытие выше EMA50",
            "На 4h медвежья дивергенция MACD",
        ]
        available = {"15m", "1h"}
        fixed, warnings = self.validator(conditions, available)
        self.assertEqual(len(fixed), 4)
        self.assertIn("15m", fixed[1])
        self.assertIn("1h", fixed[3])
        self.assertEqual(len(warnings), 2)

    def test_no_timeframe_in_condition(self):
        conditions = ["RSI > 60 и объём выше среднего"]
        available = {"15m"}
        fixed, warnings = self.validator(conditions, available)
        self.assertEqual(fixed[0], conditions[0])
        self.assertEqual(len(warnings), 0)

    def test_empty_conditions(self):
        fixed, warnings = self.validator([], {"15m"})
        self.assertEqual(fixed, [])
        self.assertEqual(warnings, [])

if __name__ == "__main__":
    print("=" * 60)
    print("ЗАПУСК UNIT-ТЕСТОВ ТОРГОВОГО БОТА")
    print("=" * 60)
    unittest.main(verbosity=2)
