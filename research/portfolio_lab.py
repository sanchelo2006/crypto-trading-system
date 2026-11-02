"""
Портфельная лаборатория + стресс-тест проскальзывания.

Идея: вместо одного сигнала используем портфель из нескольких
шорт-сигналов с равными весами (1/N капитала на сигнал).

Стресс-тест: портфель прогоняется при разных уровнях round-trip
проскальзывания поверх комиссии. Критерий устойчивости:
  - при slip=0.05%: overall exp >= +0.05% И all_years_positive=True;
  - при slip=0.10%: overall exp > 0 (пессимистичный запас).

Запуск: python research/portfolio_lab.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from data_loader import get_available_symbols, get_available_years, load_local_data
from data_fetcher import BybitDataFetcher
from backtest_engine import BacktestEngine
import research.impulse_lab as lab

# === ВЫХОДЫ ИЗ ЛАБОРАТОРИИ (лабораторный оптимум) ===
SL_ATR = 2.0     # широкий стоп переживает отскок
TP_ATR = 1.5     # быстрый тейк фиксирует продолжение
MAX_HOLD = 8     # интрадей (2 часа)
DIRECTION = -1   # все сигналы шорт

# === УРОВНИ СТРЕСС-ТЕСТА ===
SLIPS = (0.0, 0.03, 0.05, 0.10)   # round-trip проскальзывание, %
DETAIL_SLIP = 0.05                 # уровень, для которого печатаем таблицу по датасетам

# Порог прохождения стресс-теста (из паспорта проекта)
STRESS_PASS_EXP = 0.05             # при slip=0.05% overall exp должен быть >= этого

# === ПОРТФЕЛЬ: сигналы для одновременной торговли ===
PORTFOLIO = {
    "m2s2_vol2.0": {
        "desc": "обвал >2ATR + объём >2x",
        "mask_fn": lambda d: (
            (d["vel_2"].to_numpy() < -2.0)
            & ((d["volume"] / d["avg_volume_96"]).to_numpy() > 2.0)
        ),
    },
    "act_short": {
        "desc": "аномальная активность + моментум вниз",
        "mask_fn": lambda d: (
            (d["activity"].to_numpy() > 3.0)
            & (d["vel_2"].to_numpy() < -1.0)
        ),
    },
    "mom2_short_t2.0": {
        "desc": "чистый моментум вниз >2ATR",
        "mask_fn": lambda d: (d["vel_2"].to_numpy() < -2.0),
    },
}


def prepare_datasets():
    """Загружает данные и кэширует индикаторы/маски ОДИН раз.

    Возвращает список (symbol, year, d, idx_map), где idx_map —
    готовые индексы сигналов для каждого датасета.
    """
    engine = BacktestEngine(BybitDataFetcher(testnet=True))
    pairs = [(s, y) for s in get_available_symbols() for y in get_available_years(s)]
    cache = []
    for symbol, year in pairs:
        try:
            df = load_local_data(symbol, year)
        except FileNotFoundError:
            print(f"{symbol} {year}: нет данных, пропуск")
            continue
        if len(df) < 5000:
            print(f"{symbol} {year}: мало свечей ({len(df)}), пропуск")
            continue
        df = engine._build_indicator_frame(df)
        d = lab.add_features(df)
        idx_map = {}
        for sig_name, sig_info in PORTFOLIO.items():
            idx = np.where(sig_info["mask_fn"](d))[0]
            if len(idx) > 0:
                idx_map[sig_name] = idx
        if idx_map:
            cache.append((symbol, year, d, idx_map))
    return cache


def run_portfolio(cache, slip):
    """Прогон портфеля при заданном проскальзывании.

    Меняет lab.SLIPPAGE_PCT (глобальная переменная, которую читает
    _simulate_fixed как cost = FEE_PCT + SLIPPAGE_PCT).
    Возвращает {(signal, symbol, year): exp}.
    """
    lab.SLIPPAGE_PCT = slip
    results = {}
    for symbol, year, d, idx_map in cache:
        for sig_name, idx in idx_map.items():
            res = lab._simulate_fixed(
                d, idx, DIRECTION,
                sl_atr=SL_ATR, tp_atr=TP_ATR, max_hold=MAX_HOLD,
            )
            if res["n_traded"] > 0 and res["exp"] is not None:
                results[(sig_name, symbol, year)] = res["exp"]
    return results


def aggregate(results, cache):
    """Портфельные метрики: среднее по сигналам для каждого датасета.

    Возвращает:
      per_dataset: {(symbol, year): портфельный exp}
      per_year:    {year: [exp по датасетам]}
    """
    datasets = sorted(set((s, y) for (s, y, _, _) in cache))
    per_dataset = {}
    for symbol, year in datasets:
        exps = [results[(sig, symbol, year)]
                for sig in PORTFOLIO
                if (sig, symbol, year) in results]
        if exps:
            per_dataset[(symbol, year)] = float(np.mean(exps))
    per_year = {}
    for (symbol, year), exp in per_dataset.items():
        per_year.setdefault(year, []).append(exp)
    return per_dataset, per_year


def year_verdict(per_year):
    """Сводка режимной независимости по годам."""
    lines = []
    all_positive = True
    for year in sorted(per_year.keys()):
        exps = per_year[year]
        mean_exp = float(np.mean(exps))
        frac_pos = float(np.mean([e > 0 for e in exps]))
        status = "✅" if mean_exp > 0 else "❌"
        if mean_exp <= 0:
            all_positive = False
        lines.append(
            f"  {year}: exp={mean_exp:+.4f}%  frac_pos={frac_pos:.2f}  "
            f"датасетов={len(exps)}  {status}"
        )
    all_exps = [e for exps in per_year.values() for e in exps]
    overall_mean = float(np.mean(all_exps))
    overall_frac = float(np.mean([e > 0 for e in all_exps]))
    return lines, all_positive, overall_mean, overall_frac


def main():
    print("Подготовка датасетов (индикаторы + маски сигналов)...")
    cache = prepare_datasets()
    print(f"Датасетов: {len(cache)}")
    print(f"Портфель ({len(PORTFOLIO)} сигналов): {', '.join(PORTFOLIO.keys())}")
    print(f"Выходы: sl={SL_ATR}, tp={TP_ATR}, hold={MAX_HOLD}")
    print(f"Комиссия: {lab.FEE_PCT}% round-trip (фиксированная)")

    summary_rows = []

    for slip in SLIPS:
        print(f"\n{'='*90}")
        print(f"СТРЕСС-ТЕСТ: SLIPPAGE {slip:.2f}% round-trip "
              f"(суммарные издержки {lab.FEE_PCT + slip:.2f}%)")
        print(f"{'='*90}")

        results = run_portfolio(cache, slip)
        per_dataset, per_year = aggregate(results, cache)
        year_lines, all_positive, overall_mean, overall_frac = year_verdict(per_year)

        for line in year_lines:
            print(line)
        print(f"\n  ВСЕ ГОДЫ: exp={overall_mean:+.4f}%  frac_pos={overall_frac:.2f}")
        print(f"  Режимная независимость: "
              f"{'✅ ДА — прибыльный в каждый год' if all_positive else '❌ НЕТ — есть убыточные годы'}")

        summary_rows.append({
            "slip": slip,
            "total_cost": lab.FEE_PCT + slip,
            "overall_exp": overall_mean,
            "overall_frac_pos": overall_frac,
            "all_years_positive": all_positive,
        })

        # Полная таблица по датасетам — только для ключевого уровня
        if slip == DETAIL_SLIP:
            print(f"\n  ДЕТАЛИЗАЦИЯ ПО ДАТАСЕТАМ (slip={slip:.2f}%):")
            print(f"  {'Датасет':22s} {'ПОРТФЕЛЬ':>12s}")
            print("  " + "-" * 40)
            for (symbol, year), exp in sorted(per_dataset.items()):
                status = "✅" if exp > 0 else "❌"
                print(f"  {symbol} {year:>6}   {exp:>+11.3f}% {status}")

    # === ИТОГОВАЯ СВОДКА СТРЕСС-ТЕСТА ===
    print(f"\n{'='*90}")
    print("ИТОГОВАЯ СВОДКА СТРЕСС-ТЕСТА")
    print(f"{'='*90}")
    print(f"{'slip':>6s} {'издержки':>10s} {'overall_exp':>13s} "
          f"{'frac_pos':>10s} {'all_years_pos':>15s}")
    for row in summary_rows:
        print(f"{row['slip']:>5.2f}% {row['total_cost']:>9.2f}% "
              f"{row['overall_exp']:>+12.4f}% {row['overall_frac_pos']:>10.2f} "
              f"{'✅' if row['all_years_positive'] else '❌':>14s}")

    # === ВЕРДИКТ ===
    stress_row = next((r for r in summary_rows if r["slip"] == DETAIL_SLIP), None)
    pessimistic_row = next((r for r in summary_rows if r["slip"] == max(SLIPS)), None)

    print(f"\n{'='*90}")
    print("ВЕРДИКТ")
    print(f"{'='*90}")
    if stress_row and stress_row["overall_exp"] >= STRESS_PASS_EXP and stress_row["all_years_positive"]:
        print(f"✅ ПОРТФЕЛЬ ПРОХОДИТ стресс-тест: при slip={DETAIL_SLIP:.2f}% "
              f"exp={stress_row['overall_exp']:+.4f}% (порог {STRESS_PASS_EXP}%), "
              f"режимная независимость сохранена.")
    elif stress_row:
        print(f"⚠️ ПОРТФЕЛЬ НЕ ПРОХОДИТ стресс-тест при slip={DETAIL_SLIP:.2f}%: "
              f"exp={stress_row['overall_exp']:+.4f}%, "
              f"all_years_positive={stress_row['all_years_positive']}.")
    if pessimistic_row:
        if pessimistic_row["overall_exp"] > 0:
            print(f"✅ Пессимистичный сценарий (slip={max(SLIPS):.2f}%): "
                  f"exp={pessimistic_row['overall_exp']:+.4f}% — запас прочности есть.")
        else:
            print(f"❌ Пессимистичный сценарий (slip={max(SLIPS):.2f}%): "
                  f"exp={pessimistic_row['overall_exp']:+.4f}% — портфель умирает.")


if __name__ == "__main__":
    main()