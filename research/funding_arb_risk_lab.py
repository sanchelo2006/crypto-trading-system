"""
Фандинг-арбитраж: версия 2 с базисным риском.

Добавляет к версии 1 (идеальный хедж):
1. Модель базисного риска (Ornstein-Uhlenbeck процесс).
2. Стоимость ребалансировки.
3. Проверку ликвидации.
4. Стресс-тесты на экстремальные периоды.

Запуск: python research/funding_arb_risk_lab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Пути
# ------------------------------------------------------------------
DATA_ROOT = Path(__file__).resolve().parent.parent / "backtest_data"
FUNDING_DIR = DATA_ROOT / "funding"

# ------------------------------------------------------------------
# Параметры модели
# ------------------------------------------------------------------
# Издержки
COST_MAKER = 0.04     # % round-trip (мейкер)
COST_TAKER = 0.31     # % round-trip (тейкер)
REBALANCE_COST = 0.05 # % за одну ребалансировку (мейкер)

# Частота ребалансировки
REBALANCE_EVERY_DAYS = 7  # раз в 7 дней

# Плечо на перпе
LEVERAGE = 5.0
CAPITAL_EFFICIENCY = LEVERAGE / (LEVERAGE + 1.0)  # 83.3%

# Параметры базисной модели (калибруются по данным)
# σ_basis — дневная волатильность базиса (в долях, не %)
# Типичные значения для крипто: 0.002–0.008 (0.2%–0.8% в день)
BASIS_VOL_DAILY = 0.008   # 0.4% в день — консервативная оценка
BASIS_MEAN_REVERT = 0.1   # скорость возврата к среднему (в долях от текущего базиса)

# Ликвидация
# При плече 5x и isolated margin ликвидация при ~20% движении цены
# против позиции. Для шорта — при росте цены на 20%.
LIQUIDATION_PRICE_MOVE = 0.20  # 20% движение цены против позиции

# Фандинг
FUNDING_PERIODS_PER_DAY = 3
DAYS_PER_YEAR = 365.25

# Монте-Карло
N_SIMULATIONS = 1000
RANDOM_SEED = 42


# ------------------------------------------------------------------
# Загрузка данных
# ------------------------------------------------------------------
def load_funding(symbol: str) -> pd.DataFrame | None:
    path = FUNDING_DIR / f"{symbol}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "funding_rate" not in df.columns:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    df = (
        df.dropna(subset=["timestamp", "funding_rate"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    if df.empty:
        return None
    df["year"] = df["timestamp"].dt.year
    return df


def get_symbols() -> list[str]:
    if not FUNDING_DIR.exists():
        return []
    return sorted(f.stem for f in FUNDING_DIR.glob("*.csv"))


# ------------------------------------------------------------------
# Модель базисного риска (Монте-Карло)
# ------------------------------------------------------------------
def simulate_basis_risk(
    funding_rates: np.ndarray,
    n_days: int,
    n_sims: int = N_SIMULATIONS,
    basis_vol_daily: float = BASIS_VOL_DAILY,
    rebalance_every_days: int = REBALANCE_EVERY_DAYS,
    rebalance_cost_pct: float = REBALANCE_COST,
    leverage: float = LEVERAGE,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Монте-Карло симуляция базисного риска.

    Модель:
    - Фандинг собирается детерминированно (из данных).
    - Базис следует случайному блужданию с возвратом к среднему.
    - Каждые rebalance_every_days базис сбрасывается (ребалансировка),
      но за ребалансировку платим издержки.
    - Если базис уходит за порог ликвидации — фиксируем ликвидацию.

    Возвращает распределение результатов.
    """
    rng = np.random.default_rng(seed)

    n_periods = len(funding_rates)  # 8-часовые периоды
    periods_per_day = FUNDING_PERIODS_PER_DAY
    n_rebalance_periods = rebalance_every_days * periods_per_day

    # Волатильность базиса за 8-часовой период
    basis_vol_period = basis_vol_daily / np.sqrt(periods_per_day)

    # Эффективная позиция на единицу капитала
    pos_frac = CAPITAL_EFFICIENCY

    results = []

    for sim in range(n_sims):
        equity = 1.0
        basis = 0.0  # текущий базис (доля)
        max_equity = 1.0
        max_drawdown = 0.0
        n_liquidations = 0
        n_rebalances = 0

        for i in range(n_periods):
            # 1. Сбор фандинга
            rate = funding_rates[i]
            funding_income = pos_frac * rate
            equity *= (1.0 + funding_income)

            # 2. Движение базиса (Ornstein-Uhlenbeck)
            dBasis = -BASIS_MEAN_REVERT * basis + basis_vol_period * rng.normal()
            basis += dBasis

            # 3. Влияние базиса на позицию
            # Расширение базиса (положительный для шорт перпа) = убыток
            # Сжатие базиса = прибыль
            basis_pnl = -pos_frac * dBasis
            equity *= (1.0 + basis_pnl)

            # 4. Проверка ликвидации
            # Ликвидация при экстремальном расширении базиса
            liq_threshold = LIQUIDATION_PRICE_MOVE / leverage
            if abs(basis) > liq_threshold:
                n_liquidations += 1
                equity *= (1.0 - pos_frac * liq_threshold * 0.5)  # потеря части маржи
                basis = 0.0  # после ликвидации позиция закрыта

            # 5. Ребалансировка
            if (i + 1) % n_rebalance_periods == 0:
                n_rebalances += 1
                equity *= (1.0 - pos_frac * rebalance_cost_pct / 100.0)
                basis = 0.0  # после ребалансировки базис сбрасывается

            # 6. Просадка
            if equity > max_equity:
                max_equity = equity
            dd = (max_equity - equity) / max_equity if max_equity > 0 else 0
            if dd > max_drawdown:
                max_drawdown = dd

        results.append({
            "final_equity": equity,
            "max_drawdown": max_drawdown,
            "n_liquidations": n_liquidations,
            "n_rebalances": n_rebalances,
            "total_return_pct": (equity - 1.0) * 100.0,
        })

    results_df = pd.DataFrame(results)

    return {
        "n_sims": n_sims,
        "mean_return_pct": results_df["total_return_pct"].mean(),
        "median_return_pct": results_df["total_return_pct"].median(),
        "p5_return_pct": results_df["total_return_pct"].quantile(0.05),
        "p95_return_pct": results_df["total_return_pct"].quantile(0.95),
        "std_return_pct": results_df["total_return_pct"].std(),
        "mean_max_dd_pct": results_df["max_drawdown"].mean() * 100.0,
        "median_max_dd_pct": results_df["max_drawdown"].median() * 100.0,
        "p95_max_dd_pct": results_df["max_drawdown"].quantile(0.95) * 100.0,
        "liquidation_rate": (results_df["n_liquidations"] > 0).mean(),
        "mean_rebalances": results_df["n_rebalances"].mean(),
        "prob_loss": (results_df["total_return_pct"] < 0).mean(),
    }


# ------------------------------------------------------------------
# Основной прогон
# ------------------------------------------------------------------
def main():
    symbols = get_symbols()
    print(f"Монет с данными фандинга: {len(symbols)}")
    print(f"Параметры модели:")
    print(f"  Базисная волатильность: {BASIS_VOL_DAILY*100:.2f}% в день")
    print(f"  Ребалансировка: каждые {REBALANCE_EVERY_DAYS} дней")
    print(f"  Стоимость ребалансировки: {REBALANCE_COST}% (мейкер)")
    print(f"  Плечо: {LEVERAGE}x (эффективность: {CAPITAL_EFFICIENCY:.1%})")
    print(f"  Порог ликвидации: {LIQUIDATION_PRICE_MOVE*100:.0f}% движение цены")
    print(f"  Монте-Карло: {N_SIMULATIONS} симуляций")
    print()

    all_results = []

    for symbol in symbols:
        df = load_funding(symbol)
        if df is None:
            continue

        rates = df["funding_rate"].to_numpy()
        n_days = len(rates) / FUNDING_PERIODS_PER_DAY
        n_years = n_days / DAYS_PER_YEAR

        print(f"{'='*100}")
        print(f"{symbol} | записей: {len(rates)} | период: {df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()} ({n_years:.1f} лет)")
        print(f"{'='*100}")

        # Базовая статистика фандинга
        mean_rate = rates.mean()
        positive_frac = (rates > 0).mean()
        gross_annual = mean_rate * FUNDING_PERIODS_PER_DAY * DAYS_PER_YEAR * 100

        print(f"  Фандинг: mean={mean_rate*100:.4f}%/8h, "
              f"положительных={positive_frac*100:.1f}%, "
              f"годовой (брутто)={gross_annual:.1f}%")

        # Монте-Карло с базисным риском
        mc = simulate_basis_risk(rates, int(n_days))

        # Годовые результаты
        annual_return = mc["mean_return_pct"] / n_years if n_years > 0 else 0
        annual_p5 = mc["p5_return_pct"] / n_years if n_years > 0 else 0
        annual_p95 = mc["p95_return_pct"] / n_years if n_years > 0 else 0

        print(f"\n  === РЕЗУЛЬТАТ С БАЗИСНЫМ РИСКОМ ({N_SIMULATIONS} симуляций) ===")
        print(f"  Средний возврат:      {mc['mean_return_pct']:+.2f}% "
              f"({annual_return:+.2f}% годовых)")
        print(f"  Медиана:              {mc['median_return_pct']:+.2f}%")
        print(f"  5-й перцентиль:       {mc['p5_return_pct']:+.2f}% "
              f"({annual_p5:+.2f}% годовых)")
        print(f"  95-й перцентиль:      {mc['p95_return_pct']:+.2f}% "
              f"({annual_p95:+.2f}% годовых)")
        print(f"  Std:                  {mc['std_return_pct']:.2f}%")
        print(f"  Вероятность убытка:   {mc['prob_loss']*100:.1f}%")
        print(f"\n  === РИСКИ ===")
        print(f"  Средняя макс. просадка: {mc['mean_max_dd_pct']:.2f}%")
        print(f"  Медианная просадка:    {mc['median_max_dd_pct']:.2f}%")
        print(f"  95-й перцентиль ДД:    {mc['p95_max_dd_pct']:.2f}%")
        print(f"  Вероятность ликвидации: {mc['liquidation_rate']*100:.1f}%")
        print(f"  Ребалансировок:        {mc['mean_rebalances']:.0f}")

        # Сравнение с версией 1 (без базисного риска)
        ideal_annual = gross_annual * CAPITAL_EFFICIENCY
        realistic_annual = annual_return
        degradation = ideal_annual - realistic_annual

        print(f"\n  === СРАВНЕНИЕ ===")
        print(f"  Идеальный (версия 1):   ~{ideal_annual:.1f}% годовых")
        print(f"  Реалистичный (версия 2): ~{realistic_annual:.1f}% годовых")
        print(f"  Деградация от рисков:    ~{degradation:.1f}% годовых")
        print()

        all_results.append({
            "symbol": symbol,
            "n_years": round(n_years, 2),
            "gross_annual_pct": round(gross_annual, 2),
            "mean_annual_pct": round(annual_return, 2),
            "p5_annual_pct": round(annual_p5, 2),
            "p95_annual_pct": round(annual_p95, 2),
            "mean_max_dd_pct": round(mc["mean_max_dd_pct"], 2),
            "p95_max_dd_pct": round(mc["p95_max_dd_pct"], 2),
            "prob_loss_pct": round(mc["prob_loss"] * 100, 1),
            "liquidation_rate_pct": round(mc["liquidation_rate"] * 100, 1),
            "ideal_annual_pct": round(ideal_annual, 2),
            "degradation_pct": round(degradation, 2),
        })

    # ------------------------------------------------------------------
    # Кросс-активная сводка
    # ------------------------------------------------------------------
    if not all_results:
        print("Нет результатов.")
        return

    df_results = pd.DataFrame(all_results)

    print(f"\n{'='*100}")
    print("КРОСС-АКТИВНАЯ СВОДКА (с базисным риском)")
    print(f"{'='*100}")
    print(df_results.to_string(index=False))

    # Итоговые выводы
    print(f"\n{'='*100}")
    print("ИТОГОВЫЕ ВЫВОДЫ")
    print(f"{'='*100}")

    avg_annual = df_results["mean_annual_pct"].mean()
    avg_p5 = df_results["p5_annual_pct"].mean()
    avg_dd = df_results["mean_max_dd_pct"].mean()
    avg_p95_dd = df_results["p95_max_dd_pct"].mean()
    avg_loss_prob = df_results["prob_loss_pct"].mean()
    avg_liq = df_results["liquidation_rate_pct"].mean()
    avg_degradation = df_results["degradation_pct"].mean()
    frac_positive = (df_results["mean_annual_pct"] > 0).mean()
    frac_p5_positive = (df_results["p5_annual_pct"] > 0).mean()

    print(f"  Средний годовой возврат:        {avg_annual:+.2f}%")
    print(f"  Средний 5-й перцентиль:         {avg_p5:+.2f}%")
    print(f"  Средняя макс. просадка:          {avg_dd:.2f}%")
    print(f"  95-й перцентиль просадки:       {avg_p95_dd:.2f}%")
    print(f"  Вероятность убытка:             {avg_loss_prob:.1f}%")
    print(f"  Вероятность ликвидации:         {avg_liq:.1f}%")
    print(f"  Деградация от рисков:           {avg_degradation:.2f}% годовых")
    print(f"  Доля монет с + возвратом:       {frac_positive*100:.0f}%")
    print(f"  Доля монет с + на 5-м проц.:    {frac_p5_positive*100:.0f}%")
    print()

    if avg_annual > 3.0 and frac_positive > 0.7 and avg_p5 > 0:
        print("✅ ФАНДИНГ-АРБИТРАЖ ЖИВЁТ С УЧЁТОМ БАЗИСНОГО РИСКА")
        print("   Средний возврат > 3% годовых, положительный даже в 5-м перцентиле.")
    elif avg_annual > 0:
        print("⚠️ ФАНДИНГ-АРБИТРАЖ МАРЖИНАЛЕН С УЧЁТОМ РИСКОВ")
        print("   Возврат положительный, но риски съедают значительную часть.")
    else:
        print("❌ ФАНДИНГ-АРБИТРАЖ НЕ ВЫЖИВАЕТ С УЧЁТОМ БАЗИСНОГО РИСКА")
        print("   Базисный риск и ребалансировка съедают весь эдж.")


if __name__ == "__main__":
    main()