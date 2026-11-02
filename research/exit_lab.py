"""
Лаборатория выходов: выращивание механизмов выхода для замороженного сигнала.

Вход заморожен: m2s2_vol2.0 (vel_2 < -2.0 + vol_ratio > 2.0, шорт).
Варьируем ТОЛЬКО параметры выхода: стоп, тейк, тайм-стоп, трейлинг, безубыток.
Критерий истины: кросс-активная робастность (все датасеты), не один датасет.

Запуск из корня проекта:  python research/exit_lab.py
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path

# Гарантируем импорт модулей корня проекта независимо от способа запуска.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from data_loader import get_available_symbols, get_available_years, load_local_data
from data_fetcher import BybitDataFetcher
from backtest_engine import BacktestEngine
import impulse_lab as lab

DIRECTION = -1  # шорт


def signal_mask(d: pd.DataFrame) -> np.ndarray:
    """m2s2_vol2.0: падение >2 ATR за 2 свечи + объём > 2.0x среднего.

    Использует колонки, созданные в lab.add_features (vel_2, vol_ratio).
    """
    return ((d["vel_2"] < -2.0) & (d["vol_ratio"] > 2.0)).to_numpy()


def build_grid() -> list:
    """Сетка параметров выхода, разбитая на этапы для интерпретируемости."""
    grid = []
    # Этап 1: стоп × тейк × тайм-стоп (без трейлинга и безубытка)
    for sl in (0.75, 1.0, 1.5, 2.0):
        for tp in (1.5, 2.0, 3.0, 4.0):
            for hold in (8, 16, 32, 64):
                grid.append({"sl_atr": sl, "tp_atr": tp, "max_hold": hold,
                             "trail_atr": None, "be_atr": None})
    # Этап 2: трейлинг поверх широких стопов (фиксированный hold=16)
    for sl in (1.0, 1.5, 2.0):
        for tp in (2.0, 3.0, 4.0):
            for trail in (1.0, 1.5, 2.0):
                grid.append({"sl_atr": sl, "tp_atr": tp, "max_hold": 16,
                             "trail_atr": trail, "be_atr": None})
    # Этап 3: безубыток поверх трейлинга
    for sl in (1.0, 1.5):
        for tp in (2.0, 3.0):
            for trail in (1.0, 1.5):
                for be in (0.5, 0.75, 1.0):
                    grid.append({"sl_atr": sl, "tp_atr": tp, "max_hold": 16,
                                 "trail_atr": trail, "be_atr": be})
    return grid


def main():
    engine = BacktestEngine(BybitDataFetcher(testnet=True))
    pairs = [(s, y) for s in get_available_symbols() for y in get_available_years(s)]
    grid = build_grid()
    print(f"Датасетов: {len(pairs)}")
    print(f"Комбинаций выходов: {len(grid)}")

    # {(sl, tp, hold, trail, be): [exp по датасетам]}
    agg = defaultdict(list)
    agg_pf = defaultdict(list)
    agg_n = defaultdict(int)

    for symbol, year in pairs:
        df = engine._build_indicator_frame(load_local_data(symbol, year))
        d = lab.add_features(df)
        mask = signal_mask(d)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue

        for params in grid:
            res = lab._simulate_fixed(
                d, idx, DIRECTION,
                sl_atr=params["sl_atr"],
                tp_atr=params["tp_atr"],
                max_hold=params["max_hold"],
                trail_atr=params["trail_atr"],
                be_atr=params["be_atr"],
            )
            if res["n_traded"] > 0 and res["exp"] is not None:
                key = (params["sl_atr"], params["tp_atr"], params["max_hold"],
                       params["trail_atr"], params["be_atr"])
                agg[key].append(res["exp"])
                agg_pf[key].append(res["pf"])
                agg_n[key] += res["n_traded"]

    # === Агрегация ===
    rows = []
    for key, exps in agg.items():
        sl, tp, hold, trail, be = key
        exps_arr = np.array(exps)
        pfs_arr = np.array(agg_pf[key])
        rows.append({
            "sl_atr": sl, "tp_atr": tp, "max_hold": hold,
            "trail_atr": trail, "be_atr": be,
            "n_datasets": len(exps_arr),
            "trades_total": agg_n[key],
            "exp_mean": round(float(exps_arr.mean()), 4),
            "frac_pos": round(float((exps_arr > 0).mean()), 2),
            "pf_mean": round(float(pfs_arr.mean()), 2),
        })

    result = pd.DataFrame(rows).sort_values("exp_mean", ascending=False)

    # === Базовая линия (текущие параметры лаборатории) ===
    print("\n=== БАЗОВАЯ ЛИНИЯ (sl=0.75, tp=1.5, hold=8, без трейлинга/безубытка) ===")
    base = result[(result["sl_atr"] == 0.75) & (result["tp_atr"] == 1.5) &
                  (result["max_hold"] == 8) & (result["trail_atr"].isna()) &
                  (result["be_atr"].isna())]
    if not base.empty:
        print(base.to_string(index=False))
    else:
        print("(базовая линия не найдена в сетке)")

    # === ТОП-15 ===
    print("\n=== ТОП-15 КОМБИНАЦИЙ ВЫХОДОВ (по exp_mean) ===")
    print(result.head(15).to_string(index=False))

    # === Влияние трейлинга (при прочих равных: sl=1.5, tp=3.0, hold=16) ===
    print("\n=== ВЛИЯНИЕ ТРЕЙЛИНГА (sl=1.5, tp=3.0, hold=16) ===")
    no_trail = result[(result["sl_atr"] == 1.5) & (result["tp_atr"] == 3.0) &
                      (result["max_hold"] == 16) & (result["trail_atr"].isna()) &
                      (result["be_atr"].isna())]
    with_trail = result[(result["sl_atr"] == 1.5) & (result["tp_atr"] == 3.0) &
                        (result["max_hold"] == 16) & (result["be_atr"].isna()) &
                        (result["trail_atr"].notna())]
    if not no_trail.empty:
        nt = no_trail.iloc[0]
        print(f"  Без трейлинга: exp={nt['exp_mean']:+.4f}%  frac_pos={nt['frac_pos']:.2f}")
        for _, wt in with_trail.iterrows():
            print(f"  Трейлинг {wt['trail_atr']}: exp={wt['exp_mean']:+.4f}%  "
                  f"frac_pos={wt['frac_pos']:.2f}")

    # === Влияние безубытка (при sl=1.0, tp=2.0, trail=1.0, hold=16) ===
    print("\n=== ВЛИЯНИЕ БЕЗУБЫТКА (sl=1.0, tp=2.0, trail=1.0, hold=16) ===")
    no_be = result[(result["sl_atr"] == 1.0) & (result["tp_atr"] == 2.0) &
                   (result["max_hold"] == 16) & (result["trail_atr"] == 1.0) &
                   (result["be_atr"].isna())]
    with_be = result[(result["sl_atr"] == 1.0) & (result["tp_atr"] == 2.0) &
                     (result["max_hold"] == 16) & (result["trail_atr"] == 1.0) &
                     (result["be_atr"].notna())]
    if not no_be.empty:
        nb = no_be.iloc[0]
        print(f"  Без безубытка: exp={nb['exp_mean']:+.4f}%  frac_pos={nb['frac_pos']:.2f}")
        for _, wb in with_be.iterrows():
            print(f"  Безубыток {wb['be_atr']}: exp={wb['exp_mean']:+.4f}%  "
                  f"frac_pos={wb['frac_pos']:.2f}")

    # === Сохранение ===
    out_path = Path(__file__).resolve().parent / "exit_lab_results.csv"
    result.to_csv(out_path, index=False)
    print(f"\nПолные результаты: {out_path} ({len(result)} строк)")


if __name__ == "__main__":
    main()