"""
D1: отчёт качества детектора режимов.
Сравнивает метки детектора (прошлое) с оракулом по будущему окну
и отвечает: угадывается ли режим, как ведёт себя цена после метки,
сколько режима спрятано в unknown, какова латентность и стабильность.
Запуск: python check_regime_quality.py
"""
import numpy as np
import pandas as pd

from data_loader import load_local_data
from data_fetcher import BybitDataFetcher
from backtest_engine import BacktestEngine

WINDOW = 8             # было 32: пробуем 2 часа вместо 8
TREND_ER = 0.40        # было 0.30: для короткого окна требуем больше трендовости
TREND_MOVE_ATR = 2.0   # было 4.0: для 2 часов 4 ATR — слишком много
RANGE_MOVE_ATR = 1.5   # было 2.0
RANGE_ER = 0.25        # оставляем

PAIRS = [("XRPUSDT", 2024), ("XRPUSDT", 2025), ("SOLUSDT", 2024), ("SOLUSDT", 2025)]


def oracle_labels(df: pd.DataFrame, window: int = WINDOW) -> np.ndarray:
    """Оракульная метка по БУДУЩЕМУ окну [t, t+window)."""
    close = df["close"].to_numpy(dtype=float)
    atr = df["atr_14"].to_numpy(dtype=float)
    n = len(close)
    abs_diff = np.abs(np.diff(close, prepend=close[0]))
    cum_abs = np.cumsum(abs_diff)
    labels = np.full(n, "other", dtype=object)
    for t in range(n - window):
        net = close[t + window] - close[t]
        path = cum_abs[t + window] - cum_abs[t]
        if path <= 0:
            continue
        er = abs(net) / path
        a = atr[t]
        if not np.isfinite(a) or a <= 0:
            continue
        move_atr = abs(net) / a
        if er >= TREND_ER and move_atr >= TREND_MOVE_ATR:
            labels[t] = "trend"
        elif move_atr <= RANGE_MOVE_ATR and er <= RANGE_ER:
            labels[t] = "range"
    return labels


def forward_stats(df: pd.DataFrame, regime: pd.Series, window: int = WINDOW) -> pd.DataFrame:
    """Поведение следующих window свечей по меткам детектора."""
    close = df["close"].to_numpy(dtype=float)
    atr = df["atr_14"].to_numpy(dtype=float)
    reg = regime.to_numpy()
    rows = []
    for label in ("trend", "range", "unknown"):
        idx = np.where(reg == label)[0]
        idx = idx[idx < len(close) - window]
        if len(idx) == 0:
            continue
        ers, moves, near = [], [], []
        for t in idx:
            seg = close[t:t + window + 1]
            net = seg[-1] - seg[0]
            path = np.abs(np.diff(seg)).sum()
            a = atr[t]
            ers.append(abs(net) / path if path > 0 else 0.0)
            moves.append(net / a if np.isfinite(a) and a > 0 else np.nan)
            if np.isfinite(a) and a > 0:
                near.append(float(np.mean(np.abs(seg - close[t]) <= 0.5 * a)))
        rows.append({
            "label": label,
            "n": len(idx),
            "fwd_ER_med": round(float(np.median(ers)), 3),
            "fwd_move_atr_med": round(float(np.nanmedian(np.abs(moves))), 2),
            "time_near_entry": round(float(np.nanmean(near)), 3) if near else None,
        })
    return pd.DataFrame(rows)


def latency(regime: np.ndarray, oracle: np.ndarray, label: str, window: int = WINDOW):
    """Задержка реакции детектора на начало оракульного отрезка."""
    delays = []
    n = len(oracle)
    t = 0
    while t < n:
        if oracle[t] == label:
            t0 = t
            while t < n and oracle[t] == label:
                t += 1
            found = np.nan
            for k in range(t0, min(t, t0 + window)):
                if regime[k] == label:
                    found = k - t0
                    break
            delays.append(found)
        else:
            t += 1
    delays = np.array(delays, dtype=float)
    if len(delays) == 0:
        return None
    det = delays[np.isfinite(delays)]
    return {
        "stretches": len(delays),
        "detected_pct": round(100 * len(det) / len(delays), 1),
        "delay_median": round(float(np.median(det)), 1) if len(det) else None,
        "delay_p90": round(float(np.percentile(det, 90)), 1) if len(det) else None,
    }


def run_stats(s: pd.Series) -> dict:
    changes = s != s.shift()
    lengths = s.groupby(changes.cumsum()).size()
    return {
        "avg_run": round(float(lengths.mean()), 1),
        "median_run": float(lengths.median()),
        "flips_per_1000": round(1000 * float(changes.sum()) / len(s), 1),
    }


def report(symbol: str, year: int, df: pd.DataFrame):
    regime = df["regime"]
    raw = df.get("regime_raw", regime)
    oracle = oracle_labels(df)
    reg = regime.to_numpy()

    print("\n" + "=" * 88)
    print(f"D1 ОТЧЁТ: {symbol} {year} | свечей: {len(df)}")
    print("=" * 88)
    print("Распределение меток детектора:")
    print(regime.value_counts(normalize=True).round(3).to_string())
    print("Распределение оракула (будущее окно 8ч):")
    print(pd.Series(oracle).value_counts(normalize=True).round(3).to_string())

    # --- Precision / recall против оракула ---
    mask_labeled = oracle != "other"
    prec = {}
    rec = {}
    for label in ("trend", "range"):
        det_pos = (reg == label)
        orc_pos = (oracle == label)
        prec[label] = float((det_pos & orc_pos).sum() / det_pos.sum()) if det_pos.sum() else 0.0
        rec[label] = float((det_pos & orc_pos).sum() / orc_pos.sum()) if orc_pos.sum() else 0.0
    print("\nТочность детектора против оракула:")
    for label in ("trend", "range"):
        print(f"  {label:5s} precision={prec[label]:.2f}  recall={rec[label]:.2f}")

    # --- Что спрятано в unknown ---
    unk = reg == "unknown"
    if unk.sum():
        orc_unk = pd.Series(oracle[unk]).value_counts(normalize=True).round(3)
        print("\nСостав unknown по оракулу:")
        print(orc_unk.to_string())

    # --- Поведение вперёд по меткам ---
    print("\nПоведение следующих 32 свечей по меткам детектора:")
    print(forward_stats(df, regime).to_string(index=False))

    # --- Латентность ---
    for label in ("trend", "range"):
        lat = latency(reg, oracle, label)
        if lat:
            print(f"\nЛатентность детектора ({label}): отрезков={lat['stretches']}, "
                  f"замечено={lat['detected_pct']}%, "
                  f"задержка медиана={lat['delay_median']} свечей, p90={lat['delay_p90']}")

    # --- Стабильность ---
    print("\nСтабильность меток:")
    print("  raw      :", run_stats(raw))
    print("  smoothed :", run_stats(regime))

    # --- Вердикты ---
    fs = forward_stats(df, regime).set_index("label")
    er_t = fs.loc["trend", "fwd_ER_med"] if "trend" in fs.index else 0.0
    er_r = fs.loc["range", "fwd_ER_med"] if "range" in fs.index else 0.0
    lat_t = latency(reg, oracle, "trend")
    checks = [
        ("precision(trend) >= 0.60", prec["trend"] >= 0.60),
        ("precision(range) >= 0.60", prec["range"] >= 0.60),
        ("fwd_ER(trend) >= 1.5 x fwd_ER(range)", er_t >= 1.5 * er_r if er_r else False),
        ("латентность trend медиана <= 8 свечей",
         bool(lat_t and lat_t["delay_median"] is not None and lat_t["delay_median"] <= 8)),
    ]
    print("\nВердикты:")
    for name, ok in checks:
        print(f"  [{'OK ' if ok else 'WARN'}] {name}")


def main():
    engine = BacktestEngine(BybitDataFetcher(testnet=True))
    for symbol, year in PAIRS:
        try:
            df = load_local_data(symbol, year)
        except FileNotFoundError:
            print(f"\n=== {symbol} {year}: нет локальных данных, пропуск ===")
            continue
        df = engine._build_indicator_frame(df)
        report(symbol, year, df)


if __name__ == "__main__":
    main()