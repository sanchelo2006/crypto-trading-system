"""
Лаборатория свинг-трейдинга на старших таймфреймах (1H / 4H).

Гипотеза: трендовые стратегии на 1H/4H имеют лучший сигнал/шум,
чем на 15m, потому что тренды живут дольше, а комиссии съедают
меньшую долю от целевого движения.

Стратегии:
  1. trend_pullback  — вход на откате к EMA в сторону старшего тренда
  2. breakout        — пробой канала Дончиана (N-барный экстремум)
  3. momentum_bar    — сильная свеча с объёмом в направлении тренда

Запуск из корня проекта:
    python research/swing_lab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from data_loader import get_available_symbols, get_available_years, load_local_data
from research.impulse_lab import _simulate_fixed

# ------------------------------------------------------------------
# Параметры симуляции
# ------------------------------------------------------------------
# Таймфреймы для анализа
TIMEFRAMES = {
    "1h": "1h",
    "4h": "4h",
}

# Издержки (два сценария)
COST_TAKER = 0.28   # % round-trip
COST_MAKER = 0.04   # % round-trip

# Выходы (в ATR)
SL_ATR = 2.0        # стоп
TP_ATR = 3.5        # тейк
MAX_HOLD_BARS = {   # макс. удержание в барах (зависит от ТФ)
    "1h": 72,       # 3 дня
    "4h": 30,       # 5 дней
}

# Индикаторы
TREND_SMA = 50      # старший тренд
PULLBACK_EMA = 10   # откат к быстрой средней
DONCHIAN_N = 20     # канал Дончиана
VOLUME_SPIKE = 1.8  # множитель объёма для моментум-бара
ATR_PERIOD = 14

EXCLUDE_DIRS = {"funding", "long_short", "open_interest"}

# Критерии робастности
MIN_DATASETS = 6
MIN_TRADES_TOTAL = 100
MIN_FRAC_POSITIVE = 0.6


# ------------------------------------------------------------------
# Загрузка и ресэмплинг
# ------------------------------------------------------------------
def load_symbol_full(symbol: str) -> pd.DataFrame | None:
    """Все годы символа одним фреймом на 15m."""
    frames = []
    for year in get_available_years(symbol):
        try:
            frames.append(load_local_data(symbol, year))
        except FileNotFoundError:
            continue
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = (df.sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last")
            .reset_index(drop=True))
    return df if len(df) > 2000 else None


def resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Ресэмплинг 15m → freq с корректной агрегацией OHLCV."""
    out = (
        df.set_index("timestamp")
        .resample(freq)
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        .dropna()
        .reset_index()
    )
    return out


# ------------------------------------------------------------------
# Индикаторы
# ------------------------------------------------------------------
def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def add_swing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет индикаторы для свинг-стратегий."""
    d = df.copy()
    close = d["close"]

    # ATR
    d["atr_14"] = compute_atr(d)

    # Трендовые средние
    d["sma_trend"] = close.rolling(TREND_SMA, min_periods=TREND_SMA).mean()
    d["ema_pullback"] = close.ewm(span=PULLBACK_EMA, adjust=False).mean()

    # Канал Дончиана (пробой)
    d["donchian_high"] = d["high"].rolling(DONCHIAN_N, min_periods=DONCHIAN_N).max()
    d["donchian_low"] = d["low"].rolling(DONCHIAN_N, min_periods=DONCHIAN_N).min()

    # Объёмный всплеск
    d["avg_volume"] = d["volume"].rolling(DONCHIAN_N, min_periods=DONCHIAN_N).mean()
    d["vol_ratio"] = d["volume"] / d["avg_volume"].replace(0, np.nan)

    # Свечной диапазон в ATR
    d["range_atr"] = (d["high"] - d["low"]) / d["atr_14"].replace(0, np.nan)

    # Позиция закрытия в свече
    candle_range = (d["high"] - d["low"]).replace(0, np.nan)
    d["close_position"] = (close - d["low"]) / candle_range

    # Скорость движения (в единицах ATR)
    d["vel_1"] = close.diff(1) / d["atr_14"].replace(0, np.nan)
    d["vel_3"] = close.diff(3) / d["atr_14"].replace(0, np.nan)

    return d


# ------------------------------------------------------------------
# Генерация сигналов
# ------------------------------------------------------------------
def generate_signals(d: pd.DataFrame, tf: str) -> list:
    """
    Возвращает список кортежей (имя, маска, направление).
    Направление: +1 = лонг, -1 = шорт.
    """
    close = d["close"].to_numpy(float)
    sma_trend = d["sma_trend"].to_numpy(float)
    ema_pull = d["ema_pullback"].to_numpy(float)
    donchian_high = d["donchian_high"].to_numpy(float)
    donchian_low = d["donchian_low"].to_numpy(float)
    vol_ratio = d["vol_ratio"].to_numpy(float)
    close_pos = d["close_position"].to_numpy(float)
    vel_1 = d["vel_1"].to_numpy(float)
    vel_3 = d["vel_3"].to_numpy(float)
    range_atr = d["range_atr"].to_numpy(float)
    low = d["low"].to_numpy(float)
    high = d["high"].to_numpy(float)

    sigs = []

    # === 1. Trend Pullback ===
    # Лонг: цена выше SMA(50), откат к EMA(10), свеча закрылась в верхней половине
    uptrend = close > sma_trend
    downtrend = close < sma_trend
    near_ema_high = (high >= ema_pull * 0.998) & (close > ema_pull)
    near_ema_low = (low <= ema_pull * 1.002) & (close < ema_pull)
    bullish_close = close_pos > 0.6
    bearish_close = close_pos < 0.4

    pullback_long = uptrend & near_ema_high & bullish_close
    pullback_short = downtrend & near_ema_low & bearish_close

    sigs.append(("pullback_long", pullback_long, +1))
    sigs.append(("pullback_short", pullback_short, -1))

    # === 2. Breakout (Donchian) ===
    # Пробой максимума/минимума за N баров
    breakout_long = close > np.roll(donchian_high, 1)
    breakout_short = close < np.roll(donchian_low, 1)
    # Первые N баров невалидны
    breakout_long[:DONCHIAN_N + 1] = False
    breakout_short[:DONCHIAN_N + 1] = False

    sigs.append(("breakout_long", breakout_long, +1))
    sigs.append(("breakout_short", breakout_short, -1))

    # === 3. Momentum Bar ===
    # Сильная свеча с объёмом в направлении тренда
    vol_spike = vol_ratio > VOLUME_SPIKE
    strong_bull = (vel_1 > 1.0) & (close_pos > 0.7)
    strong_bear = (vel_1 < -1.0) & (close_pos < 0.3)

    momentum_long = uptrend & vol_spike & strong_bull
    momentum_short = downtrend & vol_spike & strong_bear

    sigs.append(("momentum_long", momentum_long, +1))
    sigs.append(("momentum_short", momentum_short, -1))

    # === 4. Momentum без фильтра тренда (двунаправленный) ===
    sigs.append(("momentum_any_long", vol_spike & strong_bull, +1))
    sigs.append(("momentum_any_short", vol_spike & strong_bear, -1))

    return sigs


# ------------------------------------------------------------------
# Прогон по одному датасету
# ------------------------------------------------------------------
def run_swing_lab(d: pd.DataFrame, tf: str, cost_pct: float) -> list:
    """Прогоняет все сигналы на одном датасете."""
    sigs = generate_signals(d, tf)
    max_hold = MAX_HOLD_BARS.get(tf, 30)
    rows = []

    for name, mask, direction in sigs:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue

        result = _simulate_fixed(
            d, idx, direction,
            sl_atr=SL_ATR,
            tp_atr=TP_ATR,
            max_hold=max_hold,
            cost=cost_pct,
        )

        rows.append({
            "signal": name,
            "n": len(idx),
            "n_traded": result.get("n_traded", 0),
            "wr": result.get("wr"),
            "pf": result.get("pf"),
            "exp": result.get("exp"),
            "tp_rate": result.get("tp_rate"),
            "sl_rate": result.get("sl_rate"),
            "med_bars": result.get("med_bars"),
        })

    return rows


# ------------------------------------------------------------------
# Основной прогон
# ------------------------------------------------------------------
def main():
    symbols = [s for s in get_available_symbols() if s not in EXCLUDE_DIRS]
    print("SWING LAB — трендовые стратегии на старших таймфреймах")
    print(f"Монет: {len(symbols)}")
    print(f"Таймфреймы: {list(TIMEFRAMES.keys())}")
    print(f"Выходы: SL={SL_ATR} ATR, TP={TP_ATR} ATR")
    print(f"Издержки: тейкер={COST_TAKER}%, мейкер={COST_MAKER}%")
    print()

    all_rows = []

    for symbol in symbols:
        df_15m = load_symbol_full(symbol)
        if df_15m is None:
            print(f"  {symbol}: нет данных, пропуск")
            continue

        for tf_label, tf_freq in TIMEFRAMES.items():
            df_tf = resample_ohlcv(df_15m, tf_freq)
            if len(df_tf) < 200:
                continue

            df_tf = add_swing_indicators(df_tf)
            n_years = len(df_tf) / (365.25 * (24 if tf_label == "1h" else 6))

            # Прогон с тейкером
            rows_taker = run_swing_lab(df_tf, tf_label, COST_TAKER)
            for r in rows_taker:
                r["symbol"] = symbol
                r["tf"] = tf_label
                r["year"] = df_tf["timestamp"].iloc[0].year if len(df_tf) else 0
                r["cost_type"] = "taker"
                r["n_years"] = round(n_years, 2)
                all_rows.append(r)

            # Прогон с мейкером
            rows_maker = run_swing_lab(df_tf, tf_label, COST_MAKER)
            for r in rows_maker:
                r["symbol"] = symbol
                r["tf"] = tf_label
                r["year"] = df_tf["timestamp"].iloc[0].year if len(df_tf) else 0
                r["cost_type"] = "maker"
                r["n_years"] = round(n_years, 2)
                all_rows.append(r)

        print(f"  {symbol}: {len(df_15m)} баров 15m → обработано")

    if not all_rows:
        print("Нет результатов.")
        return

    df_all = pd.DataFrame(all_rows)

    # ------------------------------------------------------------------
    # Сводка по сигналам и таймфреймам (тейкер)
    # ------------------------------------------------------------------
    print(f"\n{'='*110}")
    print("СВОДКА ПО СИГНАЛАМ (тейкер 0.28%, среднее по монетам)")
    print(f"{'='*110}")

    taker = df_all[df_all["cost_type"] == "taker"]
    for tf_label in TIMEFRAMES:
        sub = taker[taker["tf"] == tf_label]
        if sub.empty:
            continue
        agg = sub.groupby("signal").agg({
            "exp": "mean",
            "n_traded": "sum",
            "wr": "mean",
            "pf": lambda x: x.replace(999.0, np.nan).mean(),
            "tp_rate": "mean",
            "med_bars": "mean",
        }).round(3).sort_values("exp", ascending=False)
        print(f"\n  --- {tf_label} ---")
        print(agg.to_string())

    # ------------------------------------------------------------------
    # Сводка мейкер
    # ------------------------------------------------------------------
    print(f"\n{'='*110}")
    print("СВОДКА ПО СИГНАЛАМ (мейкер 0.04%, среднее по монетам)")
    print(f"{'='*110}")

    maker = df_all[df_all["cost_type"] == "maker"]
    for tf_label in TIMEFRAMES:
        sub = maker[maker["tf"] == tf_label]
        if sub.empty:
            continue
        agg = sub.groupby("signal").agg({
            "exp": "mean",
            "n_traded": "sum",
            "wr": "mean",
            "pf": lambda x: x.replace(999.0, np.nan).mean(),
            "tp_rate": "mean",
            "med_bars": "mean",
        }).round(3).sort_values("exp", ascending=False)
        print(f"\n  --- {tf_label} ---")
        print(agg.to_string())

    # ------------------------------------------------------------------
    # Проверка робастности
    # ------------------------------------------------------------------
    print(f"\n{'='*110}")
    print("ПРОВЕРКА РОБАСТНОСТИ")
    print(f"{'='*110}")
    print(f"Критерии: датасетов >= {MIN_DATASETS}, сделок >= {MIN_TRADES_TOTAL}, "
          f"доля положительных >= {MIN_FRAC_POSITIVE}")

    for cost_label in ("taker", "maker"):
        sub = df_all[df_all["cost_type"] == cost_label]
        print(f"\n  --- {cost_label} ---")

        robust_signals = []
        for (signal, tf), grp in sub.groupby(["signal", "tf"]):
            n_datasets = grp["symbol"].nunique()
            total_trades = grp["n_traded"].sum()
            frac_pos = (grp["exp"] > 0).mean() if len(grp) > 0 else 0
            mean_exp = grp["exp"].mean()

            is_robust = (
                n_datasets >= MIN_DATASETS
                and total_trades >= MIN_TRADES_TOTAL
                and frac_pos >= MIN_FRAC_POSITIVE
                and mean_exp > 0
            )

            if is_robust:
                robust_signals.append({
                    "signal": signal,
                    "tf": tf,
                    "datasets": n_datasets,
                    "trades": total_trades,
                    "exp_mean": round(mean_exp, 3),
                    "frac_pos": round(frac_pos, 2),
                })

        if robust_signals:
            df_robust = pd.DataFrame(robust_signals)
            print("  РОБАСТНЫЕ СИГНАЛЫ:")
            print(df_robust.to_string(index=False))
        else:
            print("  Нет робастных сигналов.")

    # ------------------------------------------------------------------
    # Вердикт
    # ------------------------------------------------------------------
    print(f"\n{'='*110}")
    print("ВЕРДИКТ")
    print(f"{'='*110}")

    for cost_label, cost_val in (("тейкер", COST_TAKER), ("мейкер", COST_MAKER)):
        sub = df_all[df_all["cost_type"] == cost_label]
        if sub.empty:
            continue
        positive = sub[sub["exp"] > 0]
        total_signals = sub["signal"].nunique() * sub["tf"].nunique()
        positive_combos = positive.groupby(["signal", "tf"]).size().count()

        print(f"\n  {cost_label} ({cost_val}%):")
        print(f"    Комбинаций сигнал×ТФ: {total_signals}")
        print(f"    Комбинаций с + ожиданием: {positive_combos}")
        if total_signals > 0:
            print(f"    Доля положительных: {positive_combos/total_signals*100:.0f}%")


if __name__ == "__main__":
    main()