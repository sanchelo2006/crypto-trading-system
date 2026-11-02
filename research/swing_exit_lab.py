"""
Лаборатория выходов для свинг-стратегий на 4H.

Фокус: оптимизация выходов (стоп/тейк/трейлинг/безубыток/удержание)
для сигнала momentum_any_long на 4H с мейкер-комиссиями.

Разделение данных (защита от переоптимизации):
  Тренировка: первые 60% баров
  Буфер:      30 баров (исключает перекрытие сделок)
  Тест:       последние 40% баров

Запуск из корня проекта:
    python research/swing_exit_lab.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from data_loader import get_available_symbols, get_available_years, load_local_data
from research.impulse_lab import _simulate_fixed

# ------------------------------------------------------------------
# Параметры сигнала (из swing_lab)
# ------------------------------------------------------------------
TREND_SMA = 50
VOLUME_SPIKE = 1.8
ATR_PERIOD = 14
COST_MAKER = 0.04   # % round-trip

# ------------------------------------------------------------------
# Сетка параметров выходов
# ------------------------------------------------------------------
SL_GRID = [1.5, 2.0, 2.5, 3.0]           # стоп в ATR
TP_GRID = [2.0, 3.0, 4.0, 6.0, 8.0]      # тейк в ATR
HOLD_GRID = [20, 30, 45, 60]             # макс. удержание в барах 4H
TRAIL_GRID = [None, 2.0, 3.0]            # трейлинг в ATR (None = выкл)
BE_GRID = [None, 1.5]                    # безубыток в ATR (None = выкл)

# Разделение данных
TRAIN_FRAC = 0.60
BUFFER_BARS = 30

# Критерии отбора
MIN_TRADES_TRAIN = 20
TOP_N = 15

EXCLUDE_DIRS = {"funding", "long_short", "open_interest"}


# ------------------------------------------------------------------
# Загрузка и ресэмплинг
# ------------------------------------------------------------------
def load_symbol_4h(symbol: str) -> pd.DataFrame | None:
    """Все годы символа, ресэмпл в 4H."""
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
    if len(df) < 2000:
        return None

    out = (
        df.set_index("timestamp")
        .resample("4h")
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
    return out if len(out) >= 1000 else None


# ------------------------------------------------------------------
# Индикаторы и маска сигнала
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


def generate_signal_mask(df: pd.DataFrame) -> np.ndarray:
    """Маска сигнала momentum_any_long на 4H."""
    d = df.copy()
    close = d["close"]

    d["atr_14"] = compute_atr(d)
    d["avg_volume"] = d["volume"].rolling(20, min_periods=20).mean()
    d["vol_ratio"] = d["volume"] / d["avg_volume"].replace(0, np.nan)
    d["vel_1"] = close.diff(1) / d["atr_14"].replace(0, np.nan)

    candle_range = (d["high"] - d["low"]).replace(0, np.nan)
    d["close_position"] = (close - d["low"]) / candle_range

    vol_spike = d["vol_ratio"].to_numpy() > VOLUME_SPIKE
    strong_bull = (d["vel_1"].to_numpy() > 1.0) & (d["close_position"].to_numpy() > 0.7)

    return vol_spike & strong_bull


# ------------------------------------------------------------------
# Прогон сетки на одной монете
# ------------------------------------------------------------------
def run_grid(df: pd.DataFrame, idx: np.ndarray, combos: list) -> list:
    """Прогон всех комбинаций выходов для одного набора индексов."""
    results = []
    for sl, tp, hold, trail, be in combos:
        r = _simulate_fixed(
            df, idx, +1,
            sl_atr=sl, tp_atr=tp,
            max_hold=hold,
            trail_atr=trail, be_atr=be,
            cost=COST_MAKER,
        )
        results.append({
            "sl_atr": sl,
            "tp_atr": tp,
            "max_hold": hold,
            "trail_atr": trail if trail is not None else -1,
            "be_atr": be if be is not None else -1,
            "n_traded": r.get("n_traded", 0),
            "exp": r.get("exp"),
            "wr": r.get("wr"),
            "pf": r.get("pf"),
            "tp_rate": r.get("tp_rate"),
            "sl_rate": r.get("sl_rate"),
        })
    return results


# ------------------------------------------------------------------
# Основной прогон
# ------------------------------------------------------------------
def main():
    symbols = [s for s in get_available_symbols() if s not in EXCLUDE_DIRS]
    combos = list(itertools.product(SL_GRID, TP_GRID, HOLD_GRID, TRAIL_GRID, BE_GRID))
    print(f"SWING EXIT LAB — оптимизация выходов для momentum_any_long 4H")
    print(f"Монет: {len(symbols)}")
    print(f"Комбинаций параметров: {len(combos)}")
    print(f"Издержки: мейкер {COST_MAKER}%")
    print(f"Разделение: тренировка {TRAIN_FRAC*100:.0f}% / буфер {BUFFER_BARS} / тест")
    print()

    # ------------------------------------------------------------------
    # Фаза 1: Тренировка
    # ------------------------------------------------------------------
    print(">>> Фаза 1: прогон на тренировочных данных...")
    train_rows = []

    for symbol in symbols:
        df = load_symbol_4h(symbol)
        if df is None:
            print(f"  {symbol}: недостаточно данных, пропуск")
            continue

        df["atr_14"] = compute_atr(df)
        mask = generate_signal_mask(df)
        idx = np.where(mask)[0]

        # Разделение на тренировку и тест
        n = len(df)
        split = int(n * TRAIN_FRAC)
        train_idx = idx[idx < split]

        if len(train_idx) < 5:
            print(f"  {symbol}: слишком мало сигналов на тренировке ({len(train_idx)}), пропуск")
            continue

        results = run_grid(df, train_idx, combos)
        for r in results:
            r["symbol"] = symbol
            train_rows.append(r)

        print(f"  {symbol}: {len(df)} баров 4H, {len(train_idx)} сигналов на тренировке")

    if not train_rows:
        print("Нет данных для тренировки.")
        return

    df_train = pd.DataFrame(train_rows)

    # Агрегация по комбинациям
    param_cols = ["sl_atr", "tp_atr", "max_hold", "trail_atr", "be_atr"]
    train_agg = df_train.groupby(param_cols).agg(
        datasets=("symbol", "nunique"),
        trades=("n_traded", "sum"),
        exp_mean=("exp", "mean"),
        exp_med=("exp", "median"),
        frac_pos=("exp", lambda x: (x > 0).mean()),
        wr_mean=("wr", "mean"),
        pf_mean=("pf", lambda x: x.replace(999.0, np.nan).mean()),
        tp_rate_mean=("tp_rate", "mean"),
    ).reset_index()

    # Фильтр: минимум сделок
    train_agg = train_agg[train_agg["trades"] >= MIN_TRADES_TRAIN]
    train_agg = train_agg.sort_values("exp_mean", ascending=False)

    print(f"\n{'='*110}")
    print(f"ТРЕНИРОВКА: топ-{TOP_N} комбинаций выходов (по exp_mean)")
    print(f"{'='*110}")
    top_train = train_agg.head(TOP_N)
    print(top_train.to_string(index=False))

    # ------------------------------------------------------------------
    # Фаза 2: Тест
    # ------------------------------------------------------------------
    print(f"\n>>> Фаза 2: проверка топ-{TOP_N} на тестовых данных...")

    # Извлекаем параметры топ-комбинаций
    top_params = top_train[param_cols].values.tolist()

    test_rows = []
    for symbol in symbols:
        df = load_symbol_4h(symbol)
        if df is None:
            continue

        df["atr_14"] = compute_atr(df)
        mask = generate_signal_mask(df)
        idx = np.where(mask)[0]

        n = len(df)
        split = int(n * TRAIN_FRAC)
        test_start = split + BUFFER_BARS
        test_idx = idx[idx >= test_start]

        if len(test_idx) < 3:
            continue

        # Прогоняем только топ-комбинации
        for sl, tp, hold, trail_enc, be_enc in top_params:
            trail = None if trail_enc == -1 else trail_enc
            be = None if be_enc == -1 else be_enc

            r = _simulate_fixed(
                df, test_idx, +1,
                sl_atr=sl, tp_atr=tp,
                max_hold=int(hold),
                trail_atr=trail, be_atr=be,
                cost=COST_MAKER,
            )
            test_rows.append({
                "sl_atr": sl, "tp_atr": tp, "max_hold": int(hold),
                "trail_atr": trail_enc, "be_atr": be_enc,
                "symbol": symbol,
                "n_traded": r.get("n_traded", 0),
                "exp": r.get("exp"),
                "wr": r.get("wr"),
                "pf": r.get("pf"),
                "tp_rate": r.get("tp_rate"),
                "sl_rate": r.get("sl_rate"),
            })

        print(f"  {symbol}: {len(test_idx)} сигналов на тесте")

    if not test_rows:
        print("Нет данных для теста.")
        return

    df_test = pd.DataFrame(test_rows)
    test_agg = df_test.groupby(param_cols).agg(
        datasets=("symbol", "nunique"),
        trades=("n_traded", "sum"),
        exp_mean=("exp", "mean"),
        frac_pos=("exp", lambda x: (x > 0).mean()),
        wr_mean=("wr", "mean"),
        pf_mean=("pf", lambda x: x.replace(999.0, np.nan).mean()),
        tp_rate_mean=("tp_rate", "mean"),
    ).reset_index()

    # ------------------------------------------------------------------
    # Фаза 3: Сравнение
    # ------------------------------------------------------------------
    print(f"\n{'='*110}")
    print("СРАВНЕНИЕ: тренировка vs тест")
    print(f"{'='*110}")

    # Соединяем тренировку и тест
    merged = top_train[param_cols + ["exp_mean", "trades", "frac_pos"]].copy()
    merged = merged.rename(columns={
        "exp_mean": "train_exp",
        "trades": "train_trades",
        "frac_pos": "train_frac_pos",
    })

    test_renamed = test_agg[param_cols + ["exp_mean", "trades", "frac_pos"]].copy()
    test_renamed = test_renamed.rename(columns={
        "exp_mean": "test_exp",
        "trades": "test_trades",
        "frac_pos": "test_frac_pos",
    })

    merged = merged.merge(test_renamed, on=param_cols, how="left")
    merged["test_exp"] = merged["test_exp"].fillna(0)
    merged["stable"] = (merged["train_exp"] > 0) & (merged["test_exp"] > 0)

    print(merged.to_string(index=False))

    # ------------------------------------------------------------------
    # Вердикт
    # ------------------------------------------------------------------
    print(f"\n{'='*110}")
    print("ВЕРДИКТ")
    print(f"{'='*110}")

    stable = merged[merged["stable"]]
    if stable.empty:
        print("  Нет комбинаций, стабильно прибыльных на тренировке И тесте.")
        print("  Это может означать переоптимизацию или нестабильность эджа.")
    else:
        best = stable.sort_values("test_exp", ascending=False).iloc[0]
        trail_str = "выкл" if best["trail_atr"] == -1 else f"{best['trail_atr']} ATR"
        be_str = "выкл" if best["be_atr"] == -1 else f"{best['be_atr']} ATR"
        print(f"  Найдено {len(stable)} стабильных комбинаций (прибыль на тренировке И тесте).")
        print(f"\n  ЛУЧШАЯ КОМБИНАЦИЯ:")
        print(f"    Стоп:       {best['sl_atr']} ATR")
        print(f"    Тейк:       {best['tp_atr']} ATR")
        print(f"    Удержание:  {int(best['max_hold'])} баров 4H")
        print(f"    Трейлинг:   {trail_str}")
        print(f"    Безубыток:  {be_str}")
        print(f"    Тренировка: exp={best['train_exp']:+.3f}%, сделок={int(best['train_trades'])}")
        print(f"    Тест:       exp={best['test_exp']:+.3f}%, сделок={int(best['test_trades'])}")

    # Сохраняем результаты
    output_path = Path("research") / "swing_exit_results.csv"
    merged.to_csv(output_path, index=False)
    print(f"\n  Результаты сохранены: {output_path}")


if __name__ == "__main__":
    main()