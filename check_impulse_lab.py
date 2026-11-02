"""
Прогон лаборатории импульсов по ВСЕМ монетам и годам
из backtest_data + кросс-активная сводка и критерий робастности.
Результаты сохраняются в текстовые файлы (читаются без Excel):
- research/leaderboard.txt — накопительные строки всех итераций (TSV);
- research/lab_report.txt  — полный отчёт последнего прогона (перезапись).
Запуск: python check_impulse_lab.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from data_loader import get_available_symbols, get_available_years, load_local_data
from data_fetcher import BybitDataFetcher
from backtest_engine import BacktestEngine
from research.impulse_lab import run_lab

RESEARCH_DIR = Path("research")
LEADERBOARD = RESEARCH_DIR / "leaderboard.txt"   # TSV, дописывается
REPORT = RESEARCH_DIR / "lab_report.txt"         # отчёт последнего прогона

# Критерии робастности сигнала между датасетами
MIN_DATASETS = 6
MIN_TRADES_TOTAL = 300
MIN_FRAC_POSITIVE = 0.7

_REPORT_LINES: list = []

def out(text: str = ""):
    """Печатает в консоль и одновременно копит в отчёт прогона."""
    print(text)
    _REPORT_LINES.append(text)

def collect_pairs():
    pairs = []
    for sym in get_available_symbols():
        for year in get_available_years(sym):
            pairs.append((sym, year))
    return pairs

def cross_asset_summary(df_lb: pd.DataFrame) -> pd.DataFrame:
    """Агрегат по направленным сигналам между датасетами."""
    d = df_lb[df_lb["exp"].notna()].copy()
    rows = []
    for signal, g in d.groupby("signal"):
        exp = pd.to_numeric(g["exp"], errors="coerce")
        pf = pd.to_numeric(g["pf"], errors="coerce").replace(999.0, np.nan)
        lift = pd.to_numeric(g["lift"], errors="coerce")
        wr = pd.to_numeric(g["wr"], errors="coerce")
        mfe = pd.to_numeric(g["mfe_mean"], errors="coerce") if "mfe_mean" in g.columns else pd.Series(dtype=float)
        mae = pd.to_numeric(g["mae_mean"], errors="coerce") if "mae_mean" in g.columns else pd.Series(dtype=float)
        
        n_ds = len(g)
        trades = int(pd.to_numeric(g["n_traded"], errors="coerce").sum())
        
        rows.append({
            "signal": signal,
            "datasets": n_ds,
            "trades": trades,
            "exp_mean": round(float(exp.mean()), 3),
            "exp_med": round(float(exp.median()), 3),
            "frac_pos": round(float((exp > 0).mean()), 2),
            "lift_mean": round(float(lift.mean()), 2),
            "pf_mean": round(float(pf.mean()), 2),
            "wr_mean": round(float(wr.mean()), 1),
            "mfe_mean": round(float(mfe.mean()), 3) if len(mfe) > 0 and mfe.notna().any() else None,
            "mae_mean": round(float(mae.mean()), 3) if len(mae) > 0 and mae.notna().any() else None,
            "robust": bool(
                n_ds >= MIN_DATASETS
                and trades >= MIN_TRADES_TOTAL
                and (exp > 0).mean() >= MIN_FRAC_POSITIVE
                and exp.mean() > 0
            ),
        })
    return pd.DataFrame(rows).sort_values("exp_mean", ascending=False)

def cross_asset_states(df_lb: pd.DataFrame) -> pd.DataFrame:
    """Агрегат по состояниям (state_*): концентрируют ли они импульсы."""
    d = df_lb[(df_lb["exp"].isna()) & (df_lb["precision"].notna())]
    rows = []
    for signal, g in d.groupby("signal"):
        lift = pd.to_numeric(g["lift"], errors="coerce")
        rows.append({
            "signal": signal,
            "datasets": len(g),
            "lift_mean": round(float(lift.mean()), 2),
            "lift_min": round(float(lift.min()), 2),
            "prec_mean": round(float(pd.to_numeric(g["precision"], errors="coerce").mean()), 3),
        })
    if not rows:
        return pd.DataFrame(
            columns=["signal", "datasets", "lift_mean", "lift_min", "prec_mean"]
        )
    return pd.DataFrame(rows).sort_values("lift_mean", ascending=False)


def yearly_summary(df_lb: pd.DataFrame) -> pd.DataFrame:
    """Разбивка по (сигнал, год): средний результат в каждом отдельном году."""
    d = df_lb[df_lb["exp"].notna()].copy()
    d["exp"] = pd.to_numeric(d["exp"], errors="coerce")
    d["n_traded"] = pd.to_numeric(d["n_traded"], errors="coerce")

    rows = []
    for (signal, year), g in d.groupby(["signal", "year"]):
        exp = g["exp"]
        rows.append({
            "signal": signal,
            "year": year,
            "datasets": len(g),
            "trades": int(g["n_traded"].sum()),
            "exp_mean": round(float(exp.mean()), 3),
            "frac_pos": round(float((exp > 0).mean()), 2),
        })
    return pd.DataFrame(rows)


def yearly_robustness(df_lb: pd.DataFrame) -> pd.DataFrame:
    """Сводка режимной независимости: в скольких годах сигнал прибыльный.

    Год считается прибыльным, если средний exp > 0 И доля прибыльных
    датасетов в этом году >= 0.5. Критерий режимной независимости:
    сигнал должен быть прибыльным в КАЖДЫЙ год, а не в среднем.
    """
    ys = yearly_summary(df_lb)
    if ys.empty:
        return pd.DataFrame(columns=[
            "signal", "years_total", "years_positive", "years_frac",
            "min_year_exp", "worst_year", "all_years_positive",
        ])

    rows = []
    for signal, g in ys.groupby("signal"):
        years_total = len(g)
        positive = g[(g["exp_mean"] > 0) & (g["frac_pos"] >= 0.5)]
        years_positive = len(positive)
        min_row = g.loc[g["exp_mean"].idxmin()]
        rows.append({
            "signal": signal,
            "years_total": years_total,
            "years_positive": years_positive,
            "years_frac": round(years_positive / years_total, 2) if years_total else 0.0,
            "min_year_exp": min_row["exp_mean"],
            "worst_year": min_row["year"],
            "all_years_positive": bool(years_positive == years_total),
        })
    return pd.DataFrame(rows).sort_values(
        ["all_years_positive", "years_frac", "min_year_exp"],
        ascending=[False, False, False],
    )

def main():
    engine = BacktestEngine(BybitDataFetcher(testnet=True))
    pairs = collect_pairs()
    out(f"Датасетов в исследовании: {len(pairs)}")
    
    all_rows = []
    for symbol, year in pairs:
        try:
            df = load_local_data(symbol, year)
        except FileNotFoundError:
            out(f"{symbol} {year}: нет данных, пропуск")
            continue
        
        if len(df) < 5000:
            out(f"{symbol} {year}: мало свечей ({len(df)}), пропуск")
            continue
        
        df = engine._build_indicator_frame(df)
        rows = run_lab(df, symbol=symbol)
        
        for r in rows:
            r["symbol"] = symbol
            r["year"] = year
        
        all_rows.extend(rows)
        
        t = pd.DataFrame([r for r in rows if r.get("n_traded")])
        out(f"\n{'=' * 100}")
        out(f"{symbol} {year} | свечей: {len(df)}")
        out("=" * 100)
        if not t.empty:
            out(t.to_string(index=False))
        else:
            out("(направленных сигналов нет)")
    
    if not all_rows:
        return
    
    # --- Лидерборд: TSV, дописывается ---
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    df_lb = pd.DataFrame(all_rows)
    write_header = not LEADERBOARD.exists()
    df_lb.to_csv(LEADERBOARD, mode="a", header=write_header, index=False, sep="\t")
    out(f"\nЛидерборд дописан: {LEADERBOARD} (+{len(df_lb)} строк)")
    
    # --- Кросс-активные сводки ---
    summary = cross_asset_summary(df_lb)
    out(f"\n{'=' * 100}")
    out("КРОСС-АКТИВНАЯ СВОДКА: направленные сигналы, сортировка по exp_mean")
    out("=" * 100)
    out(summary.to_string(index=False))
    
    states = cross_asset_states(df_lb)
    out("\nКРОСС-АКТИВНАЯ СВОДКА: состояния (концентрация импульсов)")
    out(states.to_string(index=False))
    
    robust = summary[summary["robust"]]
    out(
        f"\nРобастные сигналы (datasets>={MIN_DATASETS}, "
        f"trades>={MIN_TRADES_TOTAL}, frac_pos>={MIN_FRAC_POSITIVE}, exp_mean>0):"
    )
    if robust.empty:
        out("  НЕТ — ни один сигнал не устойчив между монетами в этой итерации")
    else:
        out(robust.to_string(index=False))

    # --- Разбивка по годам: режимная независимость ---
    yr = yearly_robustness(df_lb)
    out(f"\n{'=' * 100}")
    out("РАЗБИВКА ПО ГОДАМ: режимная независимость сигналов")
    out("(all_years_positive=True — сигнал прибыльный в каждый отдельный год)")
    out("=" * 100)
    if yr.empty:
        out("(нет данных)")
    else:
        out(yr.to_string(index=False))

    # Детальная разбивка по годам — только для робастных сигналов
    if not robust.empty:
        ys = yearly_summary(df_lb)
        robust_signals = set(robust["signal"])
        ys_robust = ys[ys["signal"].isin(robust_signals)]
        if not ys_robust.empty:
            out(f"\n{'=' * 100}")
            out("ДЕТАЛЬНАЯ РАЗБИВКА ПО ГОДАМ (только робастные сигналы)")
            out("=" * 100)
            out(ys_robust.to_string(index=False))
    
    # --- Отчёт прогона в текстовый файл ---
    REPORT.write_text("\n".join(_REPORT_LINES), encoding="utf-8")
    print(f"Отчёт прогона сохранён: {REPORT}")

if __name__ == "__main__":
    main()