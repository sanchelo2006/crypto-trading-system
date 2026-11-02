"""
Поиск оптимальных троек монет для портфельной торговли.

Идея: из 6 альткоинов (ADA, ARB, AVAX, DOGE, SOL, XRP) найти тройки,
которые вместе дают максимальную прибыль и режимную независимость.

Перебираем все C(6,3) = 20 троек, для каждой считаем:
  - Средний exp по всем датасетам тройки
  - Режимную независимость (all_years_positive)
  - Количество прибыльных датасетов

Результат: топ-5 троек, отсортированных по overall exp.

Запуск: python research/portfolio_trios.py
"""
from __future__ import annotations
import sys
from pathlib import Path
from itertools import combinations

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from data_loader import get_available_symbols, get_available_years, load_local_data
from data_fetcher import BybitDataFetcher
from backtest_engine import BacktestEngine
import research.impulse_lab as lab

# === ВЫХОДЫ ИЗ ЛАБОРАТОРИИ (лабораторный оптимум) ===
SL_ATR = 2.0
TP_ATR = 1.5
MAX_HOLD = 8
DIRECTION = -1
SLIPPAGE = 0.05  # round-trip проскальзывание, %

# === МОНЕТЫ ДЛЯ АНАЛИЗА (исключены BTC, BNB, HYPE) ===
SYMBOLS = ["ADAUSDT", "ARBUSDT", "AVAXUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT"]

# === ПОРТФЕЛЬ: сигналы для одновременной торговли ===
PORTFOLIO = {
    "m2s2_vol2.0": {
        "mask_fn": lambda d: (
            (d["vel_2"].to_numpy() < -2.0)
            & ((d["volume"] / d["avg_volume_96"]).to_numpy() > 2.0)
        ),
    },
    "act_short": {
        "mask_fn": lambda d: (
            (d["activity"].to_numpy() > 3.0)
            & (d["vel_2"].to_numpy() < -1.0)
        ),
    },
    "mom2_short_t2.0": {
        "mask_fn": lambda d: (d["vel_2"].to_numpy() < -2.0),
    },
}


def prepare_datasets():
    """Загружает данные только для SYMBOLS."""
    engine = BacktestEngine(BybitDataFetcher(testnet=True))
    cache = []
    for symbol in SYMBOLS:
        for year in get_available_years(symbol):
            try:
                df = load_local_data(symbol, year)
            except FileNotFoundError:
                continue
            if len(df) < 5000:
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
    """Прогон портфеля при заданном проскальзывании."""
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


def evaluate_trio(trio_symbols, results, cache):
    """Оценивает тройку монет: средний exp и режимная независимость."""
    datasets = [(s, y) for (s, y, _, _) in cache if s in trio_symbols]
    if not datasets:
        return None

    per_dataset = {}
    for symbol, year in datasets:
        exps = [results[(sig, symbol, year)]
                for sig in PORTFOLIO
                if (sig, symbol, year) in results]
        if exps:
            per_dataset[(symbol, year)] = float(np.mean(exps))

    # Разбивка по годам
    per_year = {}
    for (symbol, year), exp in per_dataset.items():
        per_year.setdefault(year, []).append(exp)

    # Проверка режимной независимости
    all_positive = True
    for year, exps in per_year.items():
        if np.mean(exps) <= 0:
            all_positive = False
            break

    all_exps = list(per_dataset.values())
    overall_mean = float(np.mean(all_exps)) if all_exps else 0.0
    overall_frac = float(np.mean([e > 0 for e in all_exps])) if all_exps else 0.0

    return {
        "trio": trio_symbols,
        "datasets": len(per_dataset),
        "overall_exp": overall_mean,
        "overall_frac_pos": overall_frac,
        "all_years_positive": all_positive,
        "per_year": per_year,
    }


def main():
    print("Подготовка датасетов...")
    cache = prepare_datasets()
    print(f"Датасетов: {len(cache)}")
    print(f"Монеты: {', '.join(SYMBOLS)}")
    print(f"Портфель ({len(PORTFOLIO)} сигналов): {', '.join(PORTFOLIO.keys())}")
    print(f"Выходы: sl={SL_ATR}, tp={TP_ATR}, hold={MAX_HOLD}, slip={SLIPPAGE}%")

    results = run_portfolio(cache, SLIPPAGE)

    # Перебор всех троек
    print(f"\nПеребор всех C({len(SYMBOLS)},3) = {len(list(combinations(SYMBOLS, 3)))} троек...")
    trio_results = []
    for trio in combinations(SYMBOLS, 3):
        eval_result = evaluate_trio(trio, results, cache)
        if eval_result:
            trio_results.append(eval_result)

    # Сортировка по overall_exp
    trio_results.sort(key=lambda x: x["overall_exp"], reverse=True)

    # Вывод топ-10
    print(f"\n{'='*90}")
    print("ТОП-10 ТРОЕК (отсортированы по overall exp)")
    print(f"{'='*90}")
    print(f"{'#':>3s} {'Тройка':35s} {'Датасетов':>10s} {'Exp':>10s} "
          f"{'Frac':>8s} {'All years+':>12s}")
    print("-" * 90)

    for i, tr in enumerate(trio_results[:10], 1):
        trio_str = " + ".join(tr["trio"])
        status = "✅" if tr["all_years_positive"] else "❌"
        print(f"{i:>3d} {trio_str:35s} {tr['datasets']:>10d} "
              f"{tr['overall_exp']:>+9.4f}% {tr['overall_frac_pos']:>7.2f} "
              f"{status:>11s}")

    # Детальная разбивка по годам для топ-3
    print(f"\n{'='*90}")
    print("ДЕТАЛЬНАЯ РАЗБИВКА ПО ГОДАМ (топ-3 тройки)")
    print(f"{'='*90}")
    for i, tr in enumerate(trio_results[:3], 1):
        trio_str = " + ".join(tr["trio"])
        print(f"\n{i}. {trio_str} (exp={tr['overall_exp']:+.4f}%)")
        for year in sorted(tr["per_year"].keys()):
            exps = tr["per_year"][year]
            mean_exp = float(np.mean(exps))
            frac_pos = float(np.mean([e > 0 for e in exps]))
            status = "✅" if mean_exp > 0 else "❌"
            print(f"   {year}: exp={mean_exp:+.4f}%  frac_pos={frac_pos:.2f}  "
                  f"датасетов={len(exps)}  {status}")

    # Рекомендация
    print(f"\n{'='*90}")
    print("РЕКОМЕНДАЦИЯ")
    print(f"{'='*90}")
    if trio_results:
        best = trio_results[0]
        if best["all_years_positive"]:
            print(f"✅ Лучшая тройка: {' + '.join(best['trio'])}")
            print(f"   Exp: {best['overall_exp']:+.4f}%")
            print(f"   Режимная независимость: ДА (прибыльный в каждый год)")
            print(f"\n   Эта тройка готова для production.")
        else:
            print(f"⚠️ Лучшая тройка: {' + '.join(best['trio'])}")
            print(f"   Exp: {best['overall_exp']:+.4f}%")
            print(f"   Режимная независимость: НЕТ (есть убыточные годы)")
            print(f"\n   Рекомендуется добавить еще монет для улучшения диверсификации.")


if __name__ == "__main__":
    main()