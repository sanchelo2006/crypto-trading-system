"""
Стресс-тест исполнения: выживает ли эдж сигнала при проскальзывании.
В данных Bybit гэпов нет (open[t] == close[t-1] всегда), поэтому
стоимость входа моделируем явно через SLIPPAGE_PCT (round-trip, поверх комиссии).
Запуск: python research/stress_slippage.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_loader import get_available_symbols, get_available_years, load_local_data
from data_fetcher import BybitDataFetcher
from backtest_engine import BacktestEngine
import research.impulse_lab as lab

SLIPS = (0.0, 0.03, 0.05, 0.10)          # round-trip проскальзывание, %
SIGNALS = ("m2s2_vol2.0", "m2s2_vol1.5", "mom2_short_t2.0")


def main():
    engine = BacktestEngine(BybitDataFetcher(testnet=True))
    pairs = [(s, y) for s in get_available_symbols() for y in get_available_years(s)]
    print(f"Датасетов: {len(pairs)}")

    for slip in SLIPS:
        lab.SLIPPAGE_PCT = slip
        rows = []
        for symbol, year in pairs:
            df = engine._build_indicator_frame(load_local_data(symbol, year))
            for r in lab.run_lab(df):
                if r["signal"] in SIGNALS and r.get("exp") is not None:
                    rows.append({"symbol": symbol, "year": year, "signal": r["signal"],
                                 "exp": r["exp"], "pf": r["pf"], "n": r["n_traded"]})
        d = pd.DataFrame(rows)
        print(f"\n=== SLIPPAGE {slip:.2f}% round-trip (поверх комиссии 0.11%) ===")
        for sig, g in d.groupby("signal"):
            print(f"{sig:18s} exp_mean={g['exp'].mean():+.3f}%  "
                  f"frac_pos={(g['exp'] > 0).mean():.2f}  "
                  f"pf_mean={g['pf'].mean():.2f}  trades={int(g['n'].sum())}")


if __name__ == "__main__":
    main()