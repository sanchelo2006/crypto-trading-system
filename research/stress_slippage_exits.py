"""
Стресс-тест исполнения для двух сигналов с новыми выходами.

Сигналы:
- m2s2_vol2.0:     drop_2_atr < -2.0 + volume > 2.0*avg_volume_96
- m2s2_rsi_le30:   drop_2_atr < -2.0 + RSI ≤ 30 (перепроданность)

Для каждого сигнала сравниваем три конфигурации выходов:
- Базовая (sl=0.75, tp=1.5, hold=8)
- Интрадей (sl=2.0, tp=1.5, hold=8) — широкая защита от отскока
- Позиционный (sl=2.0, tp=1.5, hold=64) — максимальный горизонт

Запуск: python research/stress_slippage_exits.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from data_loader import get_available_symbols, get_available_years, load_local_data
from data_fetcher import BybitDataFetcher
from backtest_engine import BacktestEngine
import research.impulse_lab as lab

SLIPS = (0.0, 0.03, 0.05, 0.10)  # round-trip проскальзывание, %

# Два сигнала для сравнения
SIGNALS = {
    "m2s2_vol2.0": {
        "desc": "шорт после обвала с объёмом >2x",
        "mask_fn": lambda d: ((d["vel_2"] < -2.0) & (d["vol_ratio"] > 2.0)).to_numpy(),
    },
    "m2s2_rsi_le30": {
        "desc": "шорт после обвала при RSI ≤ 30 (перепроданность)",
        "mask_fn": lambda d: ((d["vel_2"] < -2.0) & (d["rsi_14"] <= 30)).to_numpy(),
    },
}

# Три конфигурации выходов для сравнения
EXIT_CONFIGS = {
    "Базовая (sl=0.75, tp=1.5, hold=8)": {
        "sl_atr": 0.75, "tp_atr": 1.5, "max_hold": 8,
        "trail_atr": None, "be_atr": None,
    },
    "Интрадей (sl=2.0, tp=1.5, hold=8)": {
        "sl_atr": 2.0, "tp_atr": 1.5, "max_hold": 8,
        "trail_atr": None, "be_atr": None,
    },
    "Позиционный (sl=2.0, tp=1.5, hold=64)": {
        "sl_atr": 2.0, "tp_atr": 1.5, "max_hold": 64,
        "trail_atr": None, "be_atr": None,
    },
}


def main():
    engine = BacktestEngine(BybitDataFetcher(testnet=True))
    pairs = [(s, y) for s in get_available_symbols() for y in get_available_years(s)]
    print(f"Датасетов: {len(pairs)}")

    for signal_name, signal_info in SIGNALS.items():
        print(f"\n{'='*72}")
        print(f"  СИГНАЛ: {signal_name} ({signal_info['desc']})")
        print(f"{'='*72}")

        for slip in SLIPS:
            lab.SLIPPAGE_PCT = slip
            print(f"\n  --- SLIPPAGE {slip:.2f}% (поверх комиссии {lab.FEE_PCT}%) ---")

            for config_name, params in EXIT_CONFIGS.items():
                all_exps = []
                all_pfs = []
                all_frac_pos = []
                total_trades = 0

                for symbol, year in pairs:
                    df = engine._build_indicator_frame(load_local_data(symbol, year))
                    d = lab.add_features(df)
                    mask = signal_info["mask_fn"](d)
                    idx = np.where(mask)[0]
                    if len(idx) == 0:
                        continue

                    res = lab._simulate_fixed(
                        d, idx, -1,  # direction=-1 (шорт)
                        sl_atr=params["sl_atr"],
                        tp_atr=params["tp_atr"],
                        max_hold=params["max_hold"],
                        trail_atr=params["trail_atr"],
                        be_atr=params["be_atr"],
                    )

                    if res["n_traded"] > 0 and res["exp"] is not None:
                        all_exps.append(res["exp"])
                        all_pfs.append(res["pf"])
                        all_frac_pos.append(res["exp"] > 0)
                        total_trades += res["n_traded"]

                if all_exps:
                    exp_mean = np.mean(all_exps)
                    pf_mean = np.mean(all_pfs)
                    frac_pos = np.mean(all_frac_pos)
                    print(f"  {config_name:40s} exp={exp_mean:+.4f}%  "
                          f"frac={frac_pos:.2f}  pf={pf_mean:.2f}  trades={total_trades}")
                else:
                    print(f"  {config_name:40s} нет сделок")


if __name__ == "__main__":
    main()