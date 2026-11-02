"""
Расчёт технических индикаторов с использованием TA-Lib.
Добавлены MACD, объёмный профиль (адаптивные бины) и ADX.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

try:
    import talib
except ImportError:
    raise ImportError(
        "TA-Lib не установлен. Установите его:\n"
        "  Windows (рекомендуется): скачайте wheel с https://github.com/cgohlke/talib-build/releases\n"
        "  Или: conda install -c conda-forge ta-lib\n"
        "  Или: pip install TA-Lib (требуется C-библиотека)"
    )

def calculate_volume_profile(
    df: pd.DataFrame,
    num_bins: Optional[int] = None,
    top_n: int = 5,
    atr_period: int = 14,
    bin_width_atr_mult: float = 0.3
) -> List[Tuple[float, float]]:
    """
    Рассчитывает объёмный профиль: топ-N ценовых уровней с наибольшим объёмом.

    Если num_bins не задан, количество бинов определяется адаптивно:
    ширина бина = ATR * bin_width_atr_mult, но не менее 10 и не более 40 бинов.

    Параметры:
        df: DataFrame с колонками high, low, close, volume (и опционально timestamp)
        num_bins: фиксированное количество бинов (None = адаптивно)
        top_n: сколько уровней вернуть
        atr_period: период для расчёта ATR (если num_bins=None)
        bin_width_atr_mult: множитель ATR для ширины бина

    Возвращает:
        Список кортежей (price_level, volume), отсортированный по убыванию объёма.
    """
    if df.empty or 'volume' not in df.columns:
        return []

    price_high = df['high'].max()
    price_low = df['low'].min()
    if price_high == price_low:
        return [(price_high, df['volume'].sum())]

    if num_bins is None:
        # Адаптивный расчёт количества бинов на основе ATR
        high_arr = df['high'].values.astype(np.float64)
        low_arr = df['low'].values.astype(np.float64)
        close_arr = df['close'].values.astype(np.float64)
        atr = talib.ATR(high_arr, low_arr, close_arr, timeperiod=atr_period)
        atr_value = atr[-1] if not np.isnan(atr[-1]) and atr[-1] > 0 else (price_high - price_low) / 20
        bin_width = atr_value * bin_width_atr_mult
        if bin_width <= 0:
            bin_width = (price_high - price_low) / 20
        num_bins = max(10, min(40, int((price_high - price_low) / bin_width)))

    bins = np.linspace(price_low, price_high, num_bins + 1)
    volume_profile = np.zeros(num_bins)

    for i in range(len(df)):
        row = df.iloc[i]
        candle_low = row['low']
        candle_high = row['high']
        vol = row['volume']
        if candle_high == candle_low:
            bin_idx = np.digitize(candle_low, bins) - 1
            if 0 <= bin_idx < num_bins:
                volume_profile[bin_idx] += vol
        else:
            for j in range(num_bins):
                bin_low = bins[j]
                bin_high = bins[j + 1]
                overlap = max(0, min(candle_high, bin_high) - max(candle_low, bin_low))
                if overlap > 0:
                    volume_profile[j] += vol * (overlap / (candle_high - candle_low))

    top_indices = np.argsort(volume_profile)[-top_n:][::-1]
    result = []
    for idx in top_indices:
        if volume_profile[idx] > 0:
            center = (bins[idx] + bins[idx + 1]) / 2
            result.append((round(center, 6), round(volume_profile[idx], 2)))

    return result

def calculate_indicators(
    df_15m: pd.DataFrame,
    df_1h_for_pivot: pd.DataFrame = None,
    num_latest: int = 30,
    return_timestamps: bool = False,
    include_volume_profile: bool = False,
    volume_profile_bins: Optional[int] = None
) -> Dict[str, Any] | Tuple[Dict[str, Any], list]:
    """
    Рассчитывает технические индикаторы с помощью TA-Lib на основе 15-минутных свечей.
    VWAP сбрасывается каждый день в 00:00 UTC.
    Pivot Points рассчитываются по дневным свечам.
    Добавлены MACD, ADX и опционально объёмный профиль (адаптивные бины, если bins=None).

    Параметры:
        df_15m: DataFrame со свечами 15m
        df_1h_for_pivot: DataFrame со свечами 1h или дневными за 5 дней (опционально)
        num_latest: количество последних значений индикаторов для возврата
        return_timestamps: если True, возвращает также список временных меток
        include_volume_profile: если True, рассчитывает и добавляет объёмный профиль
        volume_profile_bins: количество бинов (None = адаптивно)
    """
    if df_15m.empty or len(df_15m) < 20:
        return ({}, []) if return_timestamps else {}

    df = df_15m.copy()
    if 'open' not in df.columns:
        if len(df.columns) >= 5:
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume'][:len(df.columns)]
        else:
            return ({}, []) if return_timestamps else {}

    open_price = df['open'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    close = df['close'].values.astype(np.float64)
    volume = df['volume'].values.astype(np.float64)

    # --- SMA ---
    sma20 = talib.SMA(close, timeperiod=20)
    sma50 = talib.SMA(close, timeperiod=50)

    # --- RSI ---
    rsi14 = talib.RSI(close, timeperiod=14)

    # --- ATR ---
    atr14 = talib.ATR(high, low, close, timeperiod=14)

    # --- Bollinger Bands ---
    bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

    # --- MACD ---
    macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

    # --- ADX ---
    adx_14 = talib.ADX(high, low, close, timeperiod=14)

    # --- VWAP с ресетом каждый день в 00:00 UTC ---
    typical = (high + low + close) / 3.0
    vwap_values = []
    cum_vol = 0.0
    cum_typical_vol = 0.0
    last_date = None
    for i in range(len(df)):
        current_date = None
        if 'timestamp' in df.columns:
            ts = df['timestamp'].iloc[i]
            if hasattr(ts, 'date'):
                current_date = ts.date()
            elif isinstance(ts, str):
                current_date = ts[:10]
        if last_date is not None and current_date != last_date:
            cum_vol = 0.0
            cum_typical_vol = 0.0
        last_date = current_date
        cum_vol += volume[i]
        cum_typical_vol += typical[i] * volume[i]
        vwap_values.append(cum_typical_vol / cum_vol if cum_vol > 0 else 0.0)
    vwap = np.array(vwap_values)

    if 'timestamp' in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # --- Pivot Points на основе дневных свечей ---
    pivot_df = None
    if df_1h_for_pivot is not None and not df_1h_for_pivot.empty:
        if 'timestamp' in df_1h_for_pivot.columns and len(df_1h_for_pivot) >= 2:
            t1 = df_1h_for_pivot['timestamp'].iloc[0]
            t2 = df_1h_for_pivot['timestamp'].iloc[1]
            if hasattr(t1, 'timestamp'):
                t1 = pd.Timestamp(t1)
                t2 = pd.Timestamp(t2)
                time_diff = (t2 - t1).total_seconds()
                if time_diff < 7200:
                    daily = df_1h_for_pivot.set_index('timestamp').resample('D').agg({
                        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
                    }).dropna()
                    pivot_df = daily.tail(5)
                else:
                    pivot_df = df_1h_for_pivot.tail(5)
            else:
                pivot_df = df_1h_for_pivot.tail(5)
        else:
            pivot_df = df_1h_for_pivot.tail(5)
    else:
        if 'timestamp' in df.columns:
            daily = df.set_index('timestamp').resample('D').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
            }).dropna()
            pivot_df = daily.tail(5)

    if pivot_df is not None and not pivot_df.empty:
        # Синхронизировано с backtest_engine._add_daily_pivots:
        # уровни строятся по ПРЕДЫДУЩЕМУ дню, а не по максимуму/минимуму окна.
        if len(pivot_df) >= 2:
            prev_day = pivot_df.iloc[-2]
        else:
            prev_day = pivot_df.iloc[-1]
        high_pivot = prev_day['high']
        low_pivot = prev_day['low']
        close_pivot = prev_day['close']
    else:
        high_pivot = df['high'].max()
        low_pivot = df['low'].min()
        close_pivot = df['close'].iloc[-1]

    pivot = (high_pivot + low_pivot + close_pivot) / 3.0
    r1 = 2 * pivot - low_pivot
    r2 = pivot + (high_pivot - low_pivot)
    r3 = high_pivot + 2 * (pivot - low_pivot)
    s1 = 2 * pivot - high_pivot
    s2 = pivot - (high_pivot - low_pivot)
    s3 = low_pivot - 2 * (high_pivot - pivot)

    # --- Берём последние num_latest значений ---
    def safe_tail(arr, n):
        arr = np.array(arr)
        if len(arr) == 0:
            return []
        return arr[-min(n, len(arr)):].tolist()

    result = {
        'sma_20': safe_tail(sma20, num_latest),
        'sma_50': safe_tail(sma50, num_latest),
        'rsi_14': safe_tail(rsi14, num_latest),
        'atr_14': safe_tail(atr14, num_latest),
        'bb_lower': safe_tail(bb_lower, num_latest),
        'bb_middle': safe_tail(bb_middle, num_latest),
        'bb_upper': safe_tail(bb_upper, num_latest),
        'macd': safe_tail(macd, num_latest),
        'macd_signal': safe_tail(macd_signal, num_latest),
        'macd_hist': safe_tail(macd_hist, num_latest),
        'adx_14': safe_tail(adx_14, num_latest),
        'vwap': safe_tail(vwap, num_latest),
        'pivot': float(pivot),
        'resistance_1': float(r1),
        'resistance_2': float(r2),
        'resistance_3': float(r3),
        'support_1': float(s1),
        'support_2': float(s2),
        'support_3': float(s3),
    }

    # Объёмный профиль (адаптивные бины, если volume_profile_bins не задан)
    if include_volume_profile:
        vp = calculate_volume_profile(df, num_bins=volume_profile_bins, top_n=5)
        result['volume_profile'] = vp

    # Очищаем от пустых списков и NaN
    cleaned = {}
    for k, v in result.items():
        if k == 'volume_profile':
            cleaned[k] = v
            continue
        if isinstance(v, list) and len(v) == 0:
            continue
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if isinstance(v, list):
            v = [float(x) for x in v if not np.isnan(x)]
            if not v:
                continue
        cleaned[k] = v

    if return_timestamps:
        timestamps = df['timestamp'].tail(num_latest).dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
        return cleaned, timestamps
    else:
        return cleaned

def calculate_indicators_morning(
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    num_latest: int = 50,
    return_timestamps: bool = False
) -> Dict[str, Any] | Tuple[Dict[str, Any], list]:
    """Утренний пакет – больше значений."""
    return calculate_indicators(df_15m, df_1h, num_latest, return_timestamps, include_volume_profile=True)

def resample_15m_to_1h(df_15m: pd.DataFrame) -> pd.DataFrame:
    """
    Преобразует 15m свечи в 1h свечи.
    Возвращает DataFrame с колонками:
    timestamp, open, high, low, close, volume
    """
    if df_15m is None or df_15m.empty:
        return pd.DataFrame()

    work = df_15m.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.dropna(subset=["timestamp"]).sort_values("timestamp")
    work = work.set_index("timestamp")

    df_1h = work.resample("1h", label="right", closed="right").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    df_1h = df_1h.dropna(subset=["open", "high", "low", "close"])
    df_1h = df_1h.reset_index()
    return df_1h


def compute_1h_indicators(df_1h: pd.DataFrame, talib=None) -> pd.DataFrame:
    """
    Считает индикаторы на 1h свечах.
    """
    if df_1h is None or df_1h.empty:
        return pd.DataFrame()

    if talib is None:
        raise RuntimeError("TA-Lib не установлен")

    df = df_1h.copy().reset_index(drop=True)

    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)

    df["sma_20_1h"] = talib.SMA(close, timeperiod=20)
    df["sma_50_1h"] = talib.SMA(close, timeperiod=50)
    df["rsi_14_1h"] = talib.RSI(close, timeperiod=14)
    df["atr_14_1h"] = talib.ATR(high, low, close, timeperiod=14)

    bb_u, bb_m, bb_l = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    df["bb_upper_1h"] = bb_u
    df["bb_middle_1h"] = bb_m
    df["bb_lower_1h"] = bb_l

    macd, macd_sig, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df["macd_1h"] = macd
    df["macd_signal_1h"] = macd_sig
    df["macd_hist_1h"] = macd_hist

    df["adx_14_1h"] = talib.ADX(high, low, close, timeperiod=14)

    df["avg_volume_20_1h"] = df["volume"].rolling(20).mean()
    df["avg_volume_96_1h"] = df["volume"].rolling(96).mean()

    df["vwap_1h"] = _compute_vwap_1h(df)
    _add_daily_pivots_1h(df)

    return df


def _compute_vwap_1h(df: pd.DataFrame) -> pd.Series:
    """
    VWAP для 1h-данных с дневным сбросом.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    day = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.date
    cum_tv = (typical * df["volume"]).groupby(day).cumsum()
    cum_v = df["volume"].groupby(day).cumsum()
    return cum_tv / cum_v.replace(0, np.nan)


def _add_daily_pivots_1h(df: pd.DataFrame) -> None:
    """
    Добавляет pivot_1h, resistance_1_1h..3, support_1_1h..3 на основе предыдущего дня.
    """
    df["_date"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.date

    daily = df.groupby("_date").agg(
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )

    prev = daily.shift(1)
    piv = (prev["high"] + prev["low"] + prev["close"]) / 3.0

    levels = pd.DataFrame(index=daily.index)
    levels["pivot_1h"] = piv
    levels["resistance_1_1h"] = 2 * piv - prev["low"]
    levels["resistance_2_1h"] = piv + (prev["high"] - prev["low"])
    levels["resistance_3_1h"] = prev["high"] + 2 * (piv - prev["low"])
    levels["support_1_1h"] = 2 * piv - prev["high"]
    levels["support_2_1h"] = piv - (prev["high"] - prev["low"])
    levels["support_3_1h"] = prev["low"] - 2 * (prev["high"] - piv)

    for col in levels.columns:
        df[col] = df["_date"].map(levels[col])

    df.drop(columns=["_date"], inplace=True)


def merge_1h_features_to_15m(df_15m: pd.DataFrame, df_1h_features: pd.DataFrame) -> pd.DataFrame:
    """
    Маппит 1h-признаки обратно на 15m через merge_asof.

    ВАЖНО: right (1h) содержит собственные OHLCV-колонки от
    resample_15m_to_1h (open/high/low/close/volume). Если их не отфильтровать,
    merge_asof затирает 15m OHLCV часовыми значениями: timestamp остаётся
    15-минутным, а close/high/low «замирают» внутри часа. Из-за этого
    сигналы, trailing-стоп и мягкие выходы принимали решения по устаревшей
    цене.

    Оставляем в right только признаки с суффиксом "_1h".
    """
    if df_15m is None or df_15m.empty:
        return df_15m

    if df_1h_features is None or df_1h_features.empty:
        return df_15m

    left = df_15m.copy()
    right = df_1h_features.copy()

    left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True, errors="coerce")
    right["timestamp"] = pd.to_datetime(right["timestamp"], utc=True, errors="coerce")

    left = left.sort_values("timestamp").reset_index(drop=True)
    right = right.sort_values("timestamp").reset_index(drop=True)

    # Только признаки старшего таймфрейма. OHLCV часовых свечей
    # НЕ мержим — иначе затираем 15m OHLCV.
    feature_cols = [
        c for c in right.columns
        if c != "timestamp" and c.endswith("_1h")
    ]
    if not feature_cols:
        return left

    right = right[["timestamp"] + feature_cols]
    left = left.drop(
        columns=[c for c in feature_cols if c in left.columns],
        errors="ignore",
    )

    merged = pd.merge_asof(
        left,
        right,
        on="timestamp",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged


def get_1h_feature_columns() -> List[str]:
    return [
        "sma_20_1h",
        "sma_50_1h",
        "rsi_14_1h",
        "atr_14_1h",
        "bb_upper_1h",
        "bb_middle_1h",
        "bb_lower_1h",
        "macd_1h",
        "macd_signal_1h",
        "macd_hist_1h",
        "adx_14_1h",
        "vwap_1h",
        "pivot_1h",
        "resistance_1_1h",
        "resistance_2_1h",
        "resistance_3_1h",
        "support_1_1h",
        "support_2_1h",
        "support_3_1h",
        "avg_volume_20_1h",
        "avg_volume_96_1h",
    ]