"""
Сетка / волатильность — ШАГ 1: есть ли в данных сырьё для mean-reversion?

Grid зарабатывает на осцилляциях и теряет на трендах. Прежде чем строить
симулятор сетки, измеряем свойства данных тремя независимыми способами.

1. АВТОКОРРЕЛЯЦИЯ лог-доходностей (лаги 1..16 баров).
     acf < 0 -> цена возвращается назад -> сырьё для сетки ЕСТЬ
     acf ~ 0 -> случайное блуждание     -> сетка платит только комиссии
     acf > 0 -> трендовость             -> сетка активно убыточна

2. EFFICIENCY RATIO: ER = |чистое смещение| / (полная длина пути).
     ER -> 1: чистый тренд.  ER -> 0: чистая пила.
     Считаем медиану и долю времени в "пиле" (ER < 0.15 / < 0.25).

3. ZIGZAG-ЁМКОСТЬ: сколько раз цена совершила завершённый разворот
   амплитудой >= g. Это число потенциальных round-trips сетки с шагом g.
   Плюс МАКСИМАЛЬНАЯ амплитуда разворота — это "убийца сетки":
   сколько уровней инвентаря накопилось бы до первого отката.

Запуск из корня проекта:
    python research/grid_feasibility_lab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from data_loader import get_available_symbols, get_available_years, load_local_data

# ------------------------------------------------------------------
# Параметры измерения
# ------------------------------------------------------------------
# Издержки на ОДНУ сторону, % (для оценки round-trip умножаем на 2)
COST_TAKER = 0.10
COST_MAKER = 0.02

# Шаги сетки, которые оцениваем (%)
GRID_SPACINGS_PCT = [0.2, 0.4, 0.8, 1.5]

# Лаги для автокорреляции (в барах)
ACF_LAGS = [1, 2, 4, 8, 16]

# Окна для Efficiency Ratio: (метка, число баров 15m)
ER_WINDOWS = {
    "6h": 24,
    "24h": 96,
    "72h": 288,
    "7d": 672,
}

# Пороги "пилы" для ER
ER_CHOP_STRICT = 0.15
ER_CHOP_LOOSE = 0.25

# Таймфреймы для анализа
TIMEFRAMES = {"15m": None, "1h": "1h"}


# ------------------------------------------------------------------
# Загрузка
# ------------------------------------------------------------------
def load_symbol_full(symbol: str) -> pd.DataFrame | None:
    """Все доступные годы символа одним фреймом на 15m."""
    years = get_available_years(symbol)
    if not years:
        return None

    dfs = []
    for y in years:
        try:
            dfs.append(load_local_data(symbol, y))
        except FileNotFoundError:
            continue
    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = (
        df.sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )
    return df if len(df) > 1000 else None


def resample_tf(df: pd.DataFrame, freq: str | None) -> pd.DataFrame:
    """Ресэмплинг 15m -> freq. None означает 'оставить 15m'."""
    if freq is None:
        return df
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
# Измерение 1: автокорреляция
# ------------------------------------------------------------------
def measure_acf(close: np.ndarray, lags=ACF_LAGS) -> dict:
    """Автокорреляция лог-доходностей. Отрицательная = mean-reversion."""
    r = np.diff(np.log(close))
    r = r[np.isfinite(r)]
    out = {}
    for lag in lags:
        if len(r) <= lag + 10:
            out[f"acf_{lag}"] = np.nan
            continue
        a, b = r[:-lag], r[lag:]
        if a.std() == 0 or b.std() == 0:
            out[f"acf_{lag}"] = np.nan
            continue
        out[f"acf_{lag}"] = float(np.corrcoef(a, b)[0, 1])
    return out


# ------------------------------------------------------------------
# Измерение 2: Efficiency Ratio
# ------------------------------------------------------------------
def measure_er(close: np.ndarray) -> dict:
    """Распределение ER по окнам. Низкий ER = пила = хорошо для сетки."""
    s = pd.Series(close)
    out = {}
    for label, w in ER_WINDOWS.items():
        if len(s) <= w:
            continue
        net = (s - s.shift(w)).abs()
        path = s.diff().abs().rolling(w, min_periods=w).sum()
        er = (net / path.replace(0.0, np.nan)).dropna().to_numpy()
        if len(er) == 0:
            continue
        out[f"er_{label}_med"] = float(np.median(er))
        out[f"er_{label}_chop15"] = float((er < ER_CHOP_STRICT).mean())
        out[f"er_{label}_chop25"] = float((er < ER_CHOP_LOOSE).mean())
    return out


# ------------------------------------------------------------------
# Измерение 3: zigzag — ёмкость сетки и риск тренда
# ------------------------------------------------------------------
def zigzag_stats(close: np.ndarray, threshold_pct: float) -> dict:
    """Считает завершённые развороты амплитудой >= threshold.

    Возвращает:
      swings         — число разворотов (потенциальные round-trips сетки)
      max_swing_pct  — максимальная амплитуда разворота
                       (сколько уровней инвентаря накопилось бы)
      sum_swing_pct  — суммарная амплитуда всех разворотов
                       (валовый потенциал сетки в %)
    """
    n = len(close)
    if n < 3:
        return {"swings": 0, "max_swing_pct": 0.0, "sum_swing_pct": 0.0}

    g = threshold_pct / 100.0
    ref = float(close[0])
    extreme = ref
    direction = 0          # 0 не определено, +1 вверх, -1 вниз
    swings = 0
    max_swing = 0.0
    sum_swing = 0.0
    last_turn = ref

    for p in close[1:]:
        p = float(p)
        if direction == 0:
            if p >= ref * (1 + g):
                direction, extreme, last_turn = 1, p, ref
            elif p <= ref * (1 - g):
                direction, extreme, last_turn = -1, p, ref
            else:
                ref = p
            continue

        if direction == 1:
            if p > extreme:
                extreme = p
            elif p <= extreme * (1 - g):
                amp = abs(extreme - last_turn) / last_turn * 100.0
                swings += 1
                sum_swing += amp
                max_swing = max(max_swing, amp)
                direction, extreme, last_turn = -1, p, extreme
        else:
            if p < extreme:
                extreme = p
            elif p >= extreme * (1 + g):
                amp = abs(last_turn - extreme) / last_turn * 100.0
                swings += 1
                sum_swing += amp
                max_swing = max(max_swing, amp)
                direction, extreme, last_turn = 1, p, extreme

    return {
        "swings": swings,
        "max_swing_pct": round(max_swing, 2),
        "sum_swing_pct": round(sum_swing, 2),
    }


def evaluate_spacing(close: np.ndarray, spacing_pct: float, n_days: float,
                     cost_per_side_pct: float) -> dict:
    """Оценка round-trip экономики сетки с заданным шагом."""
    zz = zigzag_stats(close, spacing_pct)
    swings = zz["swings"]
    if swings == 0 or n_days <= 0:
        return {"swings": 0, "net_per_year_pct": 0.0,
                "max_inventory_levels": 0, "verdict": "нет разворотов"}

    # Один разворот zigzag = один round-trip сетки:
    #   валовая прибыль = spacing_pct
    #   издержки        = 2 * cost_per_side (покупка + продажа)
    gross = swings * spacing_pct
    costs = swings * 2.0 * cost_per_side_pct
    net = gross - costs
    net_per_year = net / n_days * 365.25

    # Максимум уровней инвентаря, который накопился бы в худшем тренде
    max_levels = int(round(zz["max_swing_pct"] / spacing_pct))

    return {
        "swings": swings,
        "gross_pct": round(gross, 1),
        "costs_pct": round(costs, 1),
        "net_pct": round(net, 1),
        "net_per_year_pct": round(net_per_year, 1),
        "max_inventory_levels": max_levels,
        "max_inventory_loss_pct": round(
            max_levels * spacing_pct / 2.0, 2
        ),
    }


# ------------------------------------------------------------------
# Анализ одной монеты
# ------------------------------------------------------------------
def analyze_symbol(symbol: str, df: pd.DataFrame) -> list[dict]:
    rows = []
    for tf_label, tf_freq in TIMEFRAMES.items():
        d = resample_tf(df, tf_freq)
        if len(d) < 1000:
            continue

        close = d["close"].to_numpy(float)
        n_days = len(d) / (96.0 if tf_freq is None else 24.0)

        rec = {"symbol": symbol, "tf": tf_label, "bars": len(d),
               "days": round(n_days, 1)}
        rec.update(measure_acf(close))
        rec.update(measure_er(close))

        for sp in GRID_SPACINGS_PCT:
            for cost_label, cost in (("taker", COST_TAKER), ("maker", COST_MAKER)):
                r = evaluate_spacing(close, sp, n_days, cost)
                for k, v in r.items():
                    rec[f"g{sp}_{cost_label}_{k}"] = v
        rows.append(rec)
    return rows


# ------------------------------------------------------------------
# Основной прогон
# ------------------------------------------------------------------
def main():
    symbols = get_available_symbols()
    print("GRID FEASIBILITY LAB — ШАГ 1: измерение свойств данных")
    print(f"Монет: {len(symbols)}")
    print(f"Таймфреймы: {list(TIMEFRAMES)}")
    print(f"Шаги сетки: {GRID_SPACINGS_PCT}%")
    print(f"Издержки: тейкер {COST_TAKER}%/сторона, мейкер {COST_MAKER}%/сторона")
    print()

    all_rows = []
    for sym in symbols:
        df = load_symbol_full(sym)
        if df is None:
            print(f"  {sym}: недостаточно данных, пропуск")
            continue
        rows = analyze_symbol(sym, df)
        all_rows.extend(rows)
        print(f"  {sym}: {len(df)} баров 15m, "
              f"{df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()}")

    if not all_rows:
        print("Нет данных для анализа.")
        return

    df_all = pd.DataFrame(all_rows)

    # ---------------- Автокорреляция ----------------
    print(f"\n{'='*110}")
    print("1. АВТОКОРРЕЛЯЦИЯ лог-доходностей (среднее по монетам)")
    print("   acf < 0 = mean-reversion = сырьё для сетки")
    print(f"{'='*110}")
    acf_cols = [f"acf_{l}" for l in ACF_LAGS]
    print(df_all.groupby("tf")[acf_cols].mean().round(4).to_string())

    # ---------------- Efficiency Ratio ----------------
    print(f"\n{'='*110}")
    print("2. EFFICIENCY RATIO (медиана по монетам)")
    print("   chop15/chop25 = доля времени, когда ER < 0.15 / < 0.25 (режим пилы)")
    print(f"{'='*110}")
    for tf in df_all["tf"].unique():
        sub = df_all[df_all["tf"] == tf]
        cols = [c for c in sub.columns if c.startswith("er_")]
        print(f"\n  --- {tf} ---")
        print(sub[cols].median().round(3).to_string())

    # ---------------- Round-trip экономика ----------------
    print(f"\n{'='*110}")
    print("3. ЭКОНОМИКА СЕТКИ: net_per_year_pct (ВЕРХНЯЯ оценка, без потерь инвентаря)")
    print("   max_inventory_levels — сколько уровней накопилось бы в худшем тренде")
    print(f"{'='*110}")
    for tf in df_all["tf"].unique():
        sub = df_all[df_all["tf"] == tf]
        print(f"\n  --- {tf} ---")
        rows_out = []
        for sp in GRID_SPACINGS_PCT:
            for cost_label in ("taker", "maker"):
                net_col = f"g{sp}_{cost_label}_net_per_year_pct"
                lvl_col = f"g{sp}_{cost_label}_max_inventory_levels"
                sw_col = f"g{sp}_{cost_label}_swings"
                if net_col not in sub.columns:
                    continue
                rows_out.append({
                    "spacing_%": sp,
                    "cost": cost_label,
                    "swings_med": int(sub[sw_col].median()),
                    "net_per_year_%_med": round(float(sub[net_col].median()), 1),
                    "net_per_year_%_min": round(float(sub[net_col].min()), 1),
                    "frac_positive": round(float((sub[net_col] > 0).mean()), 2),
                    "max_inv_levels_med": int(sub[lvl_col].median()),
                })
        print(pd.DataFrame(rows_out).to_string(index=False))

    # ---------------- Вердикт ----------------
    print(f"\n{'='*110}")
    print("ВЕРДИКТ ШАГА 1")
    print(f"{'='*110}")

    for tf in df_all["tf"].unique():
        sub = df_all[df_all["tf"] == tf]
        acf1 = float(sub["acf_1"].mean())
        acf4 = float(sub["acf_4"].mean())
        er_med = float(sub["er_24h_med"].median()) if "er_24h_med" in sub else np.nan
        chop_frac = float(sub["er_24h_chop25"].median()) if "er_24h_chop25" in sub else np.nan

        print(f"\n  --- {tf} ---")
        print(f"  acf(1) = {acf1:+.4f}   acf(4) = {acf4:+.4f}")
        print(f"  ER(24h) медиана = {er_med:.3f}")
        print(f"  Доля времени в пиле (ER<0.25) = {chop_frac*100:.1f}%")

        mean_rev = (acf1 < -0.02) or (acf4 < -0.02)
        choppy = (not np.isnan(er_med)) and er_med < 0.30

        if mean_rev and choppy:
            print("  ✅ СЫРЬЁ ЕСТЬ: отрицательная автокорреляция + режим пилы.")
            print("     Переход к ШАГУ 2 (симулятор сетки) оправдан.")
        elif choppy:
            print("  ⚠️ СМЕШАННО: пила есть, но автокорреляция слабая.")
            print("     Сетка возможна ТОЛЬКО с режимным ER-фильтром.")
        else:
            print("  ❌ СЫРЬЯ НЕТ: данные трендовые, сетка будет терять.")
            print("     Переход к ШАГУ 2 не оправдан на этом таймфрейме.")

    print(f"\n{'='*110}")
    print("ВАЖНО: net_per_year_pct — ВЕРХНЯЯ граница. Она НЕ учитывает")
    print("убыток от накопления инвентаря в тренде. Реальный результат")
    print("даст только ШАГ 2 — событийный симулятор сетки с mark-to-market.")
    print(f"{'='*110}")


if __name__ == "__main__":
    main()