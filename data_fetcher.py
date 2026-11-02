"""
Загрузка рыночных данных с Bybit API.
Добавлены: предвычисленные метрики (market_stats), адаптивный порог ADX,
корреляция с BTC (с кешем), агрегированные трейды, 1h свечи в пакетах,
облегчённый внутридневной пакет для проверки сетапов.
"""
import time
import logging
import json
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pybit.unified_trading import HTTP

import config
from utils.helpers import df_to_csv_string, datetime_to_ts_ms
from news_fetcher import NewsAggregator
from indicators import calculate_indicators, calculate_indicators_morning

logger = logging.getLogger(__name__)

def _requests_retry_session(retries=3, backoff_factor=1.0):
    session = requests.Session()
    retry = Retry(
        total=retries, read=retries, connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; TradingSystem/1.0)"})
    return session

class FearGreedFetcher:
    def __init__(self):
        self._session = None

    def _get_session(self):
        if self._session is None:
            self._session = _requests_retry_session(retries=2, backoff_factor=2.0)
        return self._session

    def get_historical_data(self, start_time: int = None, end_time: int = None) -> pd.DataFrame:
        url = "https://api.alternative.me/fng/?limit=0"
        for attempt in range(3):
            try:
                session = self._get_session()
                resp = session.get(url, timeout=15)
                data = resp.json().get("data", [])
                if not data:
                    return pd.DataFrame()
                rows = []
                for item in data:
                    ts_sec = int(item.get("timestamp", 0))
                    if ts_sec == 0:
                        continue
                    dt = pd.to_datetime(ts_sec, unit="s", utc=True)
                    if start_time and dt.timestamp() * 1000 < start_time:
                        continue
                    if end_time and dt.timestamp() * 1000 > end_time:
                        continue
                    rows.append({
                        "timestamp": dt,
                        "Fear_Greed_Value": float(item.get("value", 50)),
                        "Market_Sentiment": item.get("value_classification", "")
                    })
                if not rows:
                    return pd.DataFrame()
                return pd.DataFrame(rows).sort_values("timestamp")
            except Exception as e:
                logger.warning(f"Fear & Greed попытка {attempt+1}/3: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Fear & Greed недоступен после 3 попыток: {e}")
        return pd.DataFrame()

class BybitFundamentals:
    def __init__(self, testnet: bool = False):
        self.session = HTTP(testnet=testnet)
        self.fg_fetcher = FearGreedFetcher()

    def _request_with_retry(self, request_func, max_retries: int = 3):
        last_error = None
        for attempt in range(max_retries):
            try:
                return request_func()
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        f"Сетевая ошибка Bybit API, попытка {attempt+1}/{max_retries}. "
                        f"Повтор через {wait}с: {e}"
                    )
                    time.sleep(wait)
        logger.exception(f"Исчерпаны попытки запроса к Bybit: {last_error}")
        raise last_error

    def fetch_open_interest(self, symbol: str, interval: str = "1h", limit: int = 200) -> pd.DataFrame:
        def _req():
            resp = self.session.get_open_interest(
                category="linear", symbol=symbol, intervalTime=interval, limit=limit
            )
            data = resp.get("result", {}).get("list", [])
            return self._list_to_df(data, "openInterest", ts_col="timestamp")
        try:
            return self._request_with_retry(_req, max_retries=2)
        except Exception:
            logger.exception(f"Не удалось загрузить OI для {symbol}")
            return pd.DataFrame(columns=["timestamp", "openInterest"])

    def fetch_funding_rate(self, symbol: str, start_time: int = None,
                           end_time: int = None, limit: int = 200) -> pd.DataFrame:
        def _req():
            resp = self.session.get_funding_rate_history(
                category="linear", symbol=symbol,
                startTime=start_time, endTime=end_time, limit=limit
            )
            data = resp.get("result", {}).get("list", [])
            return self._list_to_df(data, "fundingRate", ts_col="fundingRateTimestamp")
        try:
            return self._request_with_retry(_req, max_retries=2)
        except Exception:
            logger.exception(f"Не удалось загрузить funding rate для {symbol}")
            return pd.DataFrame(columns=["timestamp", "fundingRate"])

    def fetch_ls_ratio(self, symbol: str, period: str = "1h", limit: int = 200) -> pd.DataFrame:
        def _req():
            resp = self.session.get_long_short_ratio(
                category="linear", symbol=symbol, period=period, limit=limit
            )
            data = resp.get("result", {}).get("list", [])
            if not data:
                return pd.DataFrame()
            rows = []
            for item in data:
                buy = float(item.get("buyRatio", 0))
                sell = float(item.get("sellRatio", 0))
                ratio = buy / sell if sell != 0 else 0
                rows.append({"timestamp": int(item["timestamp"]), "longShortRatio": ratio})
            df = pd.DataFrame(rows)
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            return df[["timestamp", "longShortRatio"]] if not df.empty else pd.DataFrame()
        try:
            return self._request_with_retry(_req, max_retries=2)
        except Exception:
            logger.exception(f"Не удалось загрузить L/S ratio для {symbol}")
            return pd.DataFrame(columns=["timestamp", "longShortRatio"])

    def fetch_fear_greed(self, start_time: int = None, end_time: int = None) -> pd.DataFrame:
        return self.fg_fetcher.get_historical_data(start_time, end_time)

    def fetch_all_fundamentals(self, symbol: str, start_time: int = None, end_time: int = None,
                               interval: str = "1h", limit: int = 1000) -> pd.DataFrame:
        oi_df = self.fetch_open_interest(symbol, interval, limit)
        funding_df = self.fetch_funding_rate(symbol, start_time, end_time, limit)
        ls_df = self.fetch_ls_ratio(symbol, interval, limit)
        fg_df = self.fetch_fear_greed(start_time, end_time)
        start_dt = pd.to_datetime(start_time, unit="ms", utc=True) if start_time else None
        end_dt = pd.to_datetime(end_time, unit="ms", utc=True) if end_time else None

        def _filter_time(df, start, end):
            if df is None or df.empty:
                return df
            if "timestamp" not in df.columns:
                return df
            if df["timestamp"].dt.tz is None:
                df = df.copy()
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
            if start is not None:
                df = df[df["timestamp"] >= start]
            if end is not None:
                df = df[df["timestamp"] <= end]
            return df

        oi_df = _filter_time(oi_df, start_dt, end_dt)
        funding_df = _filter_time(funding_df, start_dt, end_dt)
        ls_df = _filter_time(ls_df, start_dt, end_dt)
        fg_df = _filter_time(fg_df, start_dt, end_dt)

        merged = oi_df if not oi_df.empty else pd.DataFrame(columns=["timestamp"])
        for other in [funding_df, ls_df, fg_df]:
            if other is not None and not other.empty:
                if merged.empty:
                    merged = other
                else:
                    merged = merged.merge(other, on="timestamp", how="outer")
        merged = merged.sort_values("timestamp").reset_index(drop=True)
        merged.rename(columns={
            "openInterest": "Open_Interest",
            "fundingRate": "Funding_Rate",
            "longShortRatio": "Long_Short_Ratio"
        }, inplace=True)
        for col in ["Open_Interest", "Funding_Rate", "Long_Short_Ratio", "Fear_Greed_Value"]:
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors="coerce")
        return merged

    def _list_to_df(self, records, value_col, ts_col="timestamp") -> pd.DataFrame:
        if not records:
            return pd.DataFrame(columns=["timestamp", value_col])
        df = pd.DataFrame(records)
        if ts_col in df.columns:
            df["timestamp"] = pd.to_datetime(df[ts_col].astype(int), unit="ms", utc=True)
        elif "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
        else:
            return pd.DataFrame()
        if value_col not in df.columns:
            return pd.DataFrame()
        df[value_col] = df[value_col].astype(float)
        return df[["timestamp", value_col]].copy()

class BybitDataFetcher:
    def __init__(self, testnet: bool = False):
        self.session = HTTP(testnet=testnet)
        self.fundamentals = BybitFundamentals(testnet=testnet)
        self.news_fetcher = NewsAggregator()
        self._btc_cache: Dict[str, Tuple[pd.DataFrame, datetime]] = {}

    def _get_cached_btc_ohlcv(self, interval: str, category: str = "linear",
                              ttl_minutes: int = 15) -> Optional[pd.DataFrame]:
        now = datetime.now(timezone.utc)
        cache_key = f"BTCUSDT_{interval}"
        if cache_key in self._btc_cache:
            cached_df, cached_time = self._btc_cache[cache_key]
            if (now - cached_time).total_seconds() < ttl_minutes * 60:
                return cached_df
        start_date = now - timedelta(hours=25)
        df = self.fetch_ohlcv("BTCUSDT", interval, start_date, now, category)
        if df is not None and not df.empty:
            self._btc_cache[cache_key] = (df, now)
        return df

    def _compute_market_stats(self, prices_1h, prices_15m, fundamentals) -> Dict:
        stats = {}
        if prices_1h is not None and not prices_1h.empty:
            df = prices_1h.tail(24)
            if len(df) >= 2:
                stats["avg_volume_1h_24h"] = round(df["volume"].mean(), 2)
                stats["max_volume_1h_24h"] = round(df["volume"].max(), 2)
                stats["min_volume_1h_24h"] = round(df["volume"].min(), 2)
                stats["price_change_24h_pct"] = round(
                    (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100, 2
                )
                stats["high_24h"] = round(df["high"].max(), 6)
                stats["low_24h"] = round(df["low"].min(), 6)
        if prices_15m is not None and not prices_15m.empty:
            df = prices_15m.tail(96)
            if len(df) >= 2:
                stats["avg_volume_15m_24h"] = round(df["volume"].mean(), 2)
                last_vol = df["volume"].iloc[-1]
                avg_vol = df["volume"].mean()
                stats["last_15m_volume_vs_avg_pct"] = round(
                    (last_vol - avg_vol) / avg_vol * 100, 2
                ) if avg_vol > 0 else 0
        if not fundamentals.empty:
            if "Open_Interest" in fundamentals.columns:
                oi = fundamentals["Open_Interest"].dropna()
                if len(oi) >= 2:
                    oi_change = (oi.iloc[-1] - oi.iloc[0]) / oi.iloc[0] * 100
                    stats["oi_change_4h_pct"] = round(oi_change, 2)
                    stats["current_oi"] = round(oi.iloc[-1], 2)
            if "Funding_Rate" in fundamentals.columns:
                fr = fundamentals["Funding_Rate"].dropna()
                if len(fr) > 0:
                    stats["current_funding_rate_pct"] = round(fr.iloc[-1] * 100, 4)
            if "Long_Short_Ratio" in fundamentals.columns:
                ls = fundamentals["Long_Short_Ratio"].dropna()
                if len(ls) > 0:
                    stats["current_ls_ratio"] = round(ls.iloc[-1], 2)
        return stats

    def _compute_adx_threshold(self, symbol: str, category: str = "linear",
                               days_back: int = 5, min_threshold: float = 12.0,
                               max_threshold: float = 30.0) -> Optional[float]:
        try:
            now = datetime.now(timezone.utc)
            start_date = now - timedelta(days=days_back)
            df = self.fetch_ohlcv(symbol, "60", start_date, now, category)
            if df is None or df.empty or len(df) < 24:
                return None
            import talib
            high = df['high'].values.astype(np.float64)
            low = df['low'].values.astype(np.float64)
            close = df['close'].values.astype(np.float64)
            adx = talib.ADX(high, low, close, timeperiod=14)
            adx = adx[~np.isnan(adx)]
            if len(adx) < 10:
                return None
            mean_adx = np.mean(adx)
            std_adx = np.std(adx)
            threshold = mean_adx - 0.5 * std_adx
            threshold = max(min_threshold, min(max_threshold, threshold))
            return round(float(threshold), 1)
        except Exception:
            logger.exception(f"Не удалось рассчитать адаптивный порог ADX для {symbol}")
            return None

    def _compute_btc_correlation(self, symbol: str, prices_15m: pd.DataFrame,
                                 category: str = "linear") -> Optional[float]:
        if symbol == "BTCUSDT" or prices_15m is None or prices_15m.empty:
            return None
        try:
            btc_df = self._get_cached_btc_ohlcv("15", category)
            if btc_df is None or btc_df.empty:
                return None
            merged = pd.merge(
                prices_15m[['timestamp', 'close']].rename(columns={'close': 'alt'}),
                btc_df[['timestamp', 'close']].rename(columns={'close': 'btc'}),
                on='timestamp', how='inner'
            )
            if len(merged) < 10:
                return None
            alt_ret = np.log(merged['alt'] / merged['alt'].shift(1)).dropna()
            btc_ret = np.log(merged['btc'] / merged['btc'].shift(1)).dropna()
            if len(alt_ret) < 5:
                return None
            corr = np.corrcoef(alt_ret, btc_ret)[0, 1]
            return round(float(corr), 2)
        except Exception:
            logger.exception(f"Не удалось рассчитать корреляцию с BTC для {symbol}")
            return None

    def _aggregate_trades(self, trades_df: pd.DataFrame) -> Dict:
        if trades_df.empty:
            return {"count": 0, "buy_volume": 0, "sell_volume": 0, "top_prices": []}
        buy_vol = trades_df[trades_df["side"] == "Buy"]["volume"].sum()
        sell_vol = trades_df[trades_df["side"] == "Sell"]["volume"].sum()
        total = buy_vol + sell_vol
        if "price" in trades_df.columns and "volume" in trades_df.columns:
            price_vol = trades_df.groupby("price")["volume"].sum().sort_values(ascending=False)
            top_prices = [
                {"price": round(p, 6), "volume": round(v, 2)}
                for p, v in price_vol.head(3).items()
            ]
        else:
            top_prices = []
        buy_pct = round(buy_vol / total * 100, 1) if total > 0 else 50
        return {
            "count": len(trades_df),
            "buy_volume": round(buy_vol, 2),
            "sell_volume": round(sell_vol, 2),
            "buy_pct": buy_pct,
            "top_prices": top_prices,
        }

    def fetch_ohlcv(self, symbol: str, interval: str, start_date: datetime,
                    end_date: datetime, category: str = "spot",
                    progress_callback=None) -> Optional[pd.DataFrame]:

        # --- ДИАГНОСТИКА: кто и с каким окном вызывает fetch_ohlcv ---
        import traceback
        stack = traceback.extract_stack()
        caller = "?"
        for frame in reversed(stack[:-1]):
            if "data_fetcher.py" not in frame.filename:
                caller = f"{frame.filename.split('TRADING_AGENT_4')[-1]}:{frame.lineno} {frame.name}"
                break
        try:
            window_h = (end_date - start_date).total_seconds() / 3600
        except Exception:
            window_h = -1
        logger.info(
            f"fetch_ohlcv START caller={caller} "
            f"symbol={symbol} interval={interval} window={window_h:.1f}h"
        )
        # --- /ДИАГНОСТИКА ---
        bybit_interval = config.BYBIT_TIMEFRAME_MAP.get(interval, interval)
        interval_seconds = config.INTERVAL_SECONDS.get(interval, 3600)
        interval_ms = interval_seconds * 1000
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)
        start_timestamp = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)
        all_candles = []
        current_end = end_timestamp
        max_retries = config.API_RETRY_COUNT
        total_needed = int((end_timestamp - start_timestamp) / interval_ms) + 100
        while current_end > start_timestamp:
            retry_count = 0
            while retry_count < max_retries:
                try:
                    remaining_ms = current_end - start_timestamp
                    fetch_limit = min(200, int(remaining_ms / interval_ms) + 10)
                    response = self.session.get_kline(
                        category=category, symbol=symbol,
                        interval=bybit_interval, end=current_end, limit=fetch_limit
                    )
                    if response["retCode"] == 0:
                        candles = response["result"]["list"]
                        if not candles:
                            current_end = start_timestamp
                            break
                        all_candles.extend(candles)
                        oldest_timestamp = int(candles[-1][0])
                        current_end = oldest_timestamp - 1
                        if progress_callback:
                            total_fetched = len(all_candles)
                            progress = min(100, int(total_fetched / total_needed * 100))
                            progress_callback(progress, total_fetched, total_needed)
                        if oldest_timestamp <= start_timestamp:
                            current_end = start_timestamp
                            break
                        time.sleep(0.2)
                        break
                    else:
                        retry_count += 1
                        if retry_count >= max_retries:
                            logger.error(
                                f"OHLCV: исчерпаны попытки по retCode!=0 "
                                f"(retCode={response.get('retCode')}, "
                                f"retMsg={response.get('retMsg')})"
                            )
                            return None
                        delay = min(30.0, 1.0 * (2 ** (retry_count - 1)))
                        logger.warning(
                            f"OHLCV: retCode!=0, повтор через {delay:.1f}с "
                            f"({retry_count}/{max_retries})"
                        )
                        time.sleep(delay)
                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        logger.exception(
                            f"OHLCV fetch failed после {max_retries} попыток: {e}"
                        )
                        return None
                    delay = min(30.0, 1.0 * (2 ** (retry_count - 1)))
                    logger.warning(
                        f"OHLCV: ошибка {type(e).__name__}, повтор через "
                        f"{delay:.1f}с ({retry_count}/{max_retries})"
                    )
                    time.sleep(delay)
        if not all_candles:
            return None
        df = pd.DataFrame(all_candles, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms', utc=True)
        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = df[col].astype(float)
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        return df

    def get_orderbook(self, category: str, symbol: str, limit: int = 10) -> dict:
        try:
            response = self.session.get_orderbook(category=category, symbol=symbol, limit=limit)
            if response["retCode"] != 0:
                return {}
            return response.get("result", {})
        except Exception:
            logger.exception(f"Не удалось загрузить orderbook для {symbol}")
            return {}

    def get_public_trading_records(self, category: str, symbol: str, limit: int = 100) -> pd.DataFrame:
        try:
            response = self.session.get_public_trade_history(
                category=category, symbol=symbol, limit=limit
            )
            if response["retCode"] != 0:
                return pd.DataFrame()
            data = response["result"]["list"]
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data)
            time_candidates = ['execTime', 'tradeTime', 'timestamp', 'time']
            time_col = next((c for c in time_candidates if c in df.columns), None)
            if time_col:
                df['timestamp'] = pd.to_datetime(
                    df[time_col].astype(int), unit='ms', utc=True, errors='coerce'
                )
            else:
                df['timestamp'] = pd.NaT
            df = df.dropna(subset=['timestamp'])
            if df.empty:
                return pd.DataFrame()
            df.rename(columns={'price': 'price', 'size': 'volume', 'side': 'side'}, inplace=True)
            df = df[['timestamp', 'price', 'volume', 'side']]
            df['price'] = df['price'].astype(float)
            df['volume'] = df['volume'].astype(float)
            return df
        except Exception:
            logger.exception(f"Не удалось загрузить трейды для {symbol}")
            return pd.DataFrame()

    def _extract_key_events(self, df_15m: pd.DataFrame, pivot_levels: dict) -> str:
        if df_15m.empty or len(df_15m) < 10:
            return ""
        df = df_15m.copy().sort_values('timestamp')
        avg_volume = df['volume'].mean()
        avg_range = (df['high'] - df['low']).mean()
        if avg_volume == 0 or avg_range == 0:
            return ""
        events = []
        levels = {
            'pivot': pivot_levels.get('pivot'),
            'r1': pivot_levels.get('resistance_1'),
            'r2': pivot_levels.get('resistance_2'),
            'r3': pivot_levels.get('resistance_3'),
            's1': pivot_levels.get('support_1'),
            's2': pivot_levels.get('support_2'),
            's3': pivot_levels.get('support_3'),
        }
        levels = {k: v for k, v in levels.items() if v is not None and not pd.isna(v)}
        for idx, row in df.iterrows():
            ts = row['timestamp'].strftime('%Y-%m-%d %H:%M UTC')
            volume = row['volume']
            high = row['high']
            low = row['low']
            close = row['close']
            candle_range = high - low
            if volume > 2 * avg_volume:
                events.append(f"{ts} — аномальный объём ({volume:.0f}, в {volume/avg_volume:.1f}x выше среднего)")
            if candle_range > 1.5 * avg_range:
                events.append(f"{ts} — широкий диапазон ({candle_range:.4f}, в {candle_range/avg_range:.1f}x выше среднего)")
            for name, level in levels.items():
                if level is None:
                    continue
                if low <= level <= high:
                    threshold = candle_range * 0.1 if candle_range > 0 else 0.0001
                    if close > level + threshold:
                        events.append(f"{ts} — пробой {name} ({level:.4f}) вверх")
                    elif close < level - threshold:
                        events.append(f"{ts} — пробой {name} ({level:.4f}) вниз")
                    else:
                        events.append(f"{ts} — касание {name} ({level:.4f})")
        if not events:
            return ""
        if len(events) > 10:
            events = events[-10:]
        return "Ключевые события за период:\n" + "\n".join(f"• {e}" for e in events)

    def _build_data_block(self, prices_1h, prices_15m, fundamentals, orderbook,
                          trades_summary: Dict, news_text, news_sentiment,
                          indicators_15m=None, indicators_1h=None,
                          indicators_timestamps=None, key_events="",
                          market_stats: Dict = None) -> Dict:
        data = {}
        if prices_1h is not None and not prices_1h.empty:
            data["prices_1h_csv"] = df_to_csv_string(prices_1h, ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if prices_15m is not None and not prices_15m.empty:
            data["prices_15m_csv"] = df_to_csv_string(prices_15m, ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if not fundamentals.empty:
            fund_cols = ['timestamp', 'Open_Interest', 'Funding_Rate', 'Fear_Greed_Value', 'Long_Short_Ratio']
            data["fundamentals_csv"] = df_to_csv_string(fundamentals, fund_cols)
        if orderbook:
            bids = orderbook.get("b", [])[:config.ORDERBOOK_DEPTH]
            asks = orderbook.get("a", [])[:config.ORDERBOOK_DEPTH]
            if bids:
                data["orderbook_bids_csv"] = df_to_csv_string(pd.DataFrame(bids, columns=["price", "volume"]))
            if asks:
                data["orderbook_asks_csv"] = df_to_csv_string(pd.DataFrame(asks, columns=["price", "volume"]))
        if trades_summary:
            data["trades_summary"] = json.dumps(trades_summary, ensure_ascii=False)
        else:
            data["trades_summary"] = "{}"
        if news_text:
            data["news_text"] = news_text
        if news_sentiment:
            data["news_sentiment"] = news_sentiment
        if prices_1h is not None:
            data["prices_1h_count"] = len(prices_1h)
        if prices_15m is not None:
            data["prices_15m_count"] = len(prices_15m)
        if indicators_15m:
            data["indicators"] = json.dumps(indicators_15m, ensure_ascii=False)
            if indicators_timestamps:
                data["indicators_timestamps"] = indicators_timestamps
        elif indicators_1h:
            data["indicators"] = json.dumps(indicators_1h, ensure_ascii=False)
            if indicators_timestamps:
                data["indicators_timestamps"] = indicators_timestamps
        else:
            data["indicators"] = "{}"
            data["indicators_timestamps"] = []
        if key_events:
            data["key_events"] = key_events
        if market_stats:
            data["market_stats"] = json.dumps(market_stats, ensure_ascii=False)
        else:
            data["market_stats"] = "{}"
        return data

    def get_morning_package(self, symbol: str, category: str,
                            settings: dict = None) -> Tuple[Dict, Optional[datetime]]:
        if settings is None:
            settings = {}
        days_back = settings.get("morning_days_back", 2)
        include_1h = settings.get("include_1h", True)
        include_15m = settings.get("include_15m", True)
        include_fund = settings.get("include_fundamentals", True)
        include_ob = settings.get("include_orderbook", True)
        include_trades = settings.get("include_trades", True)
        include_news = settings.get("include_news", True)
        news_hours = settings.get("news_hours_back_morning", 24)
        ob_depth = settings.get("orderbook_depth", config.ORDERBOOK_DEPTH)
        trades_limit = settings.get("recent_trades_limit", config.RECENT_TRADES_LIMIT)
        now = datetime.now(timezone.utc)
        start_date = now - timedelta(days=days_back)
        prices_1h = self.fetch_ohlcv(symbol, "60", start_date, now, category) if include_1h else None
        prices_15m = self.fetch_ohlcv(symbol, "15", start_date, now, category) if include_15m else None
        if prices_1h is not None and not prices_1h.empty and len(prices_1h) > config.MORNING_1H_CANDLES:
            prices_1h = prices_1h.tail(config.MORNING_1H_CANDLES)
        if prices_15m is not None and not prices_15m.empty and len(prices_15m) > config.MORNING_15M_CANDLES:
            prices_15m = prices_15m.tail(config.MORNING_15M_CANDLES)
        start_ts = datetime_to_ts_ms(start_date)
        end_ts = datetime_to_ts_ms(now)
        fundamentals = self.fundamentals.fetch_all_fundamentals(symbol, start_ts, end_ts, interval="1h", limit=config.FUNDAMENTALS_LIMIT) if include_fund else pd.DataFrame()
        orderbook = self.get_orderbook(category, symbol, ob_depth) if include_ob else {}
        trades = self.get_public_trading_records(category, symbol, trades_limit) if include_trades else pd.DataFrame()
        trades_summary = self._aggregate_trades(trades) if include_trades else {}
        news_df = self.news_fetcher.get_news_for_symbol(symbol, max_items=15, hours_back=news_hours) if include_news else pd.DataFrame()
        news_text = self.news_fetcher.format_news_for_prompt(news_df) if include_news else ""
        news_sentiment = self.news_fetcher.assess_news_sentiment(news_df) if include_news else {}
        indicators = {}
        timestamps = []
        key_events = ""
        if prices_15m is not None and not prices_15m.empty:
            if prices_1h is not None and not prices_1h.empty:
                indicators, timestamps = calculate_indicators_morning(prices_15m, prices_1h, num_latest=config.INDICATORS_LATEST_MORNING, return_timestamps=True)
            else:
                indicators, timestamps = calculate_indicators_morning(prices_15m, None, num_latest=config.INDICATORS_LATEST_MORNING, return_timestamps=True)
            pivot_levels = {'pivot': indicators.get('pivot'), 'resistance_1': indicators.get('resistance_1'), 'resistance_2': indicators.get('resistance_2'), 'resistance_3': indicators.get('resistance_3'), 'support_1': indicators.get('support_1'), 'support_2': indicators.get('support_2'), 'support_3': indicators.get('support_3')}
            key_events = self._extract_key_events(prices_15m, pivot_levels)
        market_stats = self._compute_market_stats(prices_1h, prices_15m, fundamentals)
        adx_threshold = self._compute_adx_threshold(symbol, category)
        market_stats["adx_threshold"] = adx_threshold if adx_threshold is not None else 20.0
        btc_corr = self._compute_btc_correlation(symbol, prices_15m, category)
        if btc_corr is not None:
            market_stats["btc_correlation_24h"] = btc_corr
        last_candle_time = prices_15m['timestamp'].max() if prices_15m is not None and not prices_15m.empty else None
        data = self._build_data_block(prices_1h, prices_15m, fundamentals, orderbook, trades_summary, news_text, news_sentiment, indicators_15m=indicators, indicators_timestamps=timestamps if timestamps else None, key_events=key_events, market_stats=market_stats)
        package = {"pair": symbol, "timestamp_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "message_type": "morning_package", "data": data}
        return package, last_candle_time

    def get_intraday_package(self, symbol: str, category: str,
                             from_time: Optional[datetime] = None,
                             settings: dict = None) -> Tuple[Dict, Optional[datetime]]:
        if settings is None:
            settings = {}
        candles_count = settings.get("intraday_candles_count", config.INTRADAY_15M_CANDLES)
        hours_back = max(8, int(candles_count * 0.25))
        include_news = settings.get("include_news", True)
        news_hours = settings.get("news_hours_back_intraday", 6)
        now = datetime.now(timezone.utc)
        end_date = now
        start_date = now - timedelta(hours=hours_back + 2)
        prices_15m = self.fetch_ohlcv(symbol, "15", start_date, now, category)
        if prices_15m is not None and len(prices_15m) < 20:
            logger.warning(f"Недостаточно свечей для индикаторов ({len(prices_15m)}), расширяем период до 24 часов")
            start_date = now - timedelta(hours=24)
            prices_15m = self.fetch_ohlcv(symbol, "15", start_date, now, category)
        if prices_15m is not None and not prices_15m.empty and len(prices_15m) > config.INTRADAY_15M_CANDLES:
            prices_15m = prices_15m.tail(config.INTRADAY_15M_CANDLES)
        start_1h = now - timedelta(days=config.DAYS_BACK_FOR_PIVOT)
        prices_1h = self.fetch_ohlcv(symbol, "60", start_1h, now, category)
        if prices_1h is not None and not prices_1h.empty and len(prices_1h) > config.INTRADAY_1H_CANDLES:
            prices_1h = prices_1h.tail(config.INTRADAY_1H_CANDLES)
        start_ts = datetime_to_ts_ms(start_date)
        end_ts = datetime_to_ts_ms(now)
        fundamentals = self.fundamentals.fetch_all_fundamentals(symbol, start_ts, end_ts, interval="15m", limit=200)
        orderbook = self.get_orderbook(category, symbol, settings.get("orderbook_depth", config.ORDERBOOK_DEPTH))
        trades = self.get_public_trading_records(category, symbol, settings.get("recent_trades_limit", config.RECENT_TRADES_LIMIT))
        trades_summary = self._aggregate_trades(trades)
        news_df = self.news_fetcher.get_news_for_symbol(symbol, max_items=5, hours_back=news_hours) if include_news else pd.DataFrame()
        news_text = self.news_fetcher.format_news_for_prompt(news_df) if include_news else ""
        news_sentiment = self.news_fetcher.assess_news_sentiment(news_df) if include_news else {}
        indicators = {}
        timestamps = []
        key_events = ""
        if prices_15m is not None and not prices_15m.empty:
            indicators, timestamps = calculate_indicators(prices_15m, df_1h_for_pivot=prices_1h, num_latest=config.INDICATORS_LATEST_INTRADAY, return_timestamps=True, include_volume_profile=True)
            logger.info(f"[{symbol}] Intraday индикаторы рассчитаны: {len(indicators)} индикаторов, {len(timestamps)} временных меток")
            pivot_levels = {'pivot': indicators.get('pivot'), 'resistance_1': indicators.get('resistance_1'), 'resistance_2': indicators.get('resistance_2'), 'resistance_3': indicators.get('resistance_3'), 'support_1': indicators.get('support_1'), 'support_2': indicators.get('support_2'), 'support_3': indicators.get('support_3')}
            key_events = self._extract_key_events(prices_15m, pivot_levels)
        else:
            logger.warning(f"[{symbol}] Нет 15m свечей для расчёта индикаторов")
        market_stats = self._compute_market_stats(prices_1h, prices_15m, fundamentals)
        if "adx_threshold" not in market_stats:
            adx_threshold = self._compute_adx_threshold(symbol, category)
            market_stats["adx_threshold"] = adx_threshold if adx_threshold is not None else 20.0
        btc_corr = self._compute_btc_correlation(symbol, prices_15m, category)
        if btc_corr is not None:
            market_stats["btc_correlation_24h"] = btc_corr
        new_last_time = prices_15m['timestamp'].max() if prices_15m is not None and not prices_15m.empty else None
        data = self._build_data_block(prices_1h, prices_15m, fundamentals, orderbook, trades_summary, news_text, news_sentiment, indicators_15m=indicators, indicators_timestamps=timestamps if timestamps else None, key_events=key_events, market_stats=market_stats)
        package = {"pair": symbol, "timestamp_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "message_type": "intraday_update", "data": data}
        return package, new_last_time

    def get_intraday_light_package(self, symbol: str, category: str,
                                   from_time: Optional[datetime] = None,
                                   settings: dict = None) -> Tuple[Dict, Optional[datetime]]:
        now = datetime.now(timezone.utc)
        start_15m = now - timedelta(hours=1)
        prices_15m = self.fetch_ohlcv(symbol, "15", start_15m, now, category)
        if prices_15m is not None and not prices_15m.empty:
            prices_15m = prices_15m.tail(2)
        start_1h = now - timedelta(hours=2)
        prices_1h = self.fetch_ohlcv(symbol, "60", start_1h, now, category)
        if prices_1h is not None and not prices_1h.empty:
            prices_1h = prices_1h.tail(2)
        news_df = self.news_fetcher.get_news_for_symbol(symbol, max_items=5, hours_back=0.25)
        news_text = self.news_fetcher.format_news_for_prompt(news_df) if not news_df.empty else ""
        news_sentiment = self.news_fetcher.assess_news_sentiment(news_df) if not news_df.empty else {}
        market_stats = self._compute_market_stats(prices_1h, prices_15m, pd.DataFrame())
        data = {
            "prices_1h_csv": df_to_csv_string(prices_1h, ['timestamp', 'open', 'high', 'low', 'close', 'volume']) if prices_1h is not None and not prices_1h.empty else "",
            "prices_15m_csv": df_to_csv_string(prices_15m, ['timestamp', 'open', 'high', 'low', 'close', 'volume']) if prices_15m is not None and not prices_15m.empty else "",
            "market_stats": json.dumps(market_stats, ensure_ascii=False),
            "news_text": news_text,
            "news_sentiment": news_sentiment,
        }
        new_last_time = prices_15m['timestamp'].max() if prices_15m is not None and not prices_15m.empty else None
        package = {"pair": symbol, "timestamp_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "message_type": "intraday_light", "data": data}
        return package, new_last_time

    def get_symbol_launch_time(self, symbol: str, category: str = "linear") -> Optional[datetime]:
        """
        Возвращает дату начала торгов инструмента по метаданным биржи (launchTime),
        если она доступна через get_instruments_info.

        Это лёгкий запрос, в отличие от загрузки большой истории свечей.
        """
        try:
            response = self.session.get_instruments_info(
                category=category,
                symbol=symbol,
            )
            if response.get("retCode") != 0:
                logger.warning(
                    f"get_instruments_info вернул retCode={response.get('retCode')} для {symbol}"
                )
                return None

            instruments = response.get("result", {}).get("list", [])
            if not instruments:
                return None

            item = instruments[0]
            launch_time = item.get("launchTime")
            if not launch_time:
                return None

            ts_ms = int(launch_time)
            return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        except Exception:
            logger.exception(f"Не удалось получить launchTime для {symbol}")
            return None

    def get_current_price(self, symbol: str, category: str) -> Optional[float]:
        ob = self.get_orderbook(category, symbol, limit=1)
        if ob:
            bids = ob.get("b", [])
            asks = ob.get("a", [])
            if bids and asks:
                return (float(bids[0][0]) + float(asks[0][0])) / 2
            elif bids:
                return float(bids[0][0])
            elif asks:
                return float(asks[0][0])
        return None
