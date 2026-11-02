"""
Лаборатория фандинг-арбитража.

Структурный эдж: сбор фандинга на дельта-нейтральной позиции.
Не предсказываем направление цены — собираем платёж за кредитное плечо.

Модель (версия 1 — идеальный хедж):
- Шорт перпа + лонг спот = дельта-нейтрально.
- Движение цены компенсируется (не моделируем).
- Доход = фандинг, полученный каждые 8 часов.
- Расход = издержки входа/выхода.

Стратегии:
1. always_in     — всегда в позиции, собираем весь фандинг.
2. threshold     — вход когда фандинг > порога, выход когда < 0.
3. rolling_mean  — вход когда фандинг > скользящего среднего.
4. percentile    — вход когда фандинг в верхнем перцентиле.

Запуск: python research/funding_arb_lab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Пути и параметры
# ------------------------------------------------------------------
DATA_ROOT = Path(__file__).resolve().parent.parent / "backtest_data"
FUNDING_DIR = DATA_ROOT / "funding"

# Издержки (в % от позиции)
# Тейкер: вход ~0.155% + выход ~0.155% = 0.31% round-trip
# Мейкер: вход ~0.02% + выход ~0.02% = 0.04% round-trip
COST_TAKER = 0.31   # % round-trip (спот + перп, тейкер)
COST_MAKER = 0.04   # % round-trip (спот + перп, мейкер)

# Плечо на перпе (влияет на эффективность капитала)
# При плече L: капитал = позиция * (1 + 1/L)
# Эффективная позиция на единицу капитала = L / (L + 1)
LEVERAGE = 5.0

# Фандинг начисляется 3 раза в день (каждые 8 часов)
FUNDING_PERIODS_PER_DAY = 3
DAYS_PER_YEAR = 365.25


# ------------------------------------------------------------------
# Загрузка данных
# ------------------------------------------------------------------
def load_funding(symbol: str) -> pd.DataFrame | None:
    """Загружает данные фандинга для монеты."""
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
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )
    if df.empty:
        return None

    # Добавляем год для разбивки
    df["year"] = df["timestamp"].dt.year
    return df


def get_available_funding_symbols() -> list[str]:
    """Список монет с данными фандинга."""
    if not FUNDING_DIR.exists():
        return []
    symbols = []
    for f in sorted(FUNDING_DIR.glob("*.csv")):
        symbols.append(f.stem)
    return symbols


# ------------------------------------------------------------------
# Симуляция стратегий
# ------------------------------------------------------------------
def simulate_funding_arb(
    funding_rates: np.ndarray,
    strategy: str,
    entry_threshold: float = 0.0001,
    exit_threshold: float = 0.0,
    rolling_window: int = 90,   # 90 * 8h = 30 дней
    percentile: float = 0.75,
    cost_pct: float = COST_TAKER,
    leverage: float = LEVERAGE,
) -> dict:
    """
    Симулирует фандинг-арбитраж на массиве ставок фандинга.

    Каждый элемент массива — фандинг за один 8-часовой период.
    Положительный фандинг = лонги платят шортам.
    Мы шортим перп → получаем положительный фандинг.

    Возвращает словарь с метриками.
    """
    n = len(funding_rates)
    if n == 0:
        return {"error": "empty"}

    # Эффективная позиция на единицу капитала
    # При плече L: капитал = позиция * (1 + 1/L)
    # Позиция = капитал * L / (L + 1)
    capital_efficiency = leverage / (leverage + 1.0)

    # Состояние
    in_position = False
    equity = 1.0  # нормализованный капитал (начинаем с 1.0)
    equity_curve = [1.0]
    total_funding_collected = 0.0
    total_costs_paid = 0.0
    n_entries = 0
    n_exits = 0
    periods_in_market = 0

    # Для скользящего среднего
    rolling_sum = 0.0
    rolling_count = 0

    # Для перцентиля — храним последние N значений
    recent_rates = []

    cost_frac = cost_pct / 100.0  # переводим % в долю

    for i in range(n):
        rate = funding_rates[i]

        # Обновляем скользящее окно
        rolling_sum += rate
        rolling_count += 1
        if rolling_count > rolling_window:
            rolling_sum -= funding_rates[i - rolling_window]
            rolling_count -= 1

        recent_rates.append(rate)
        if len(recent_rates) > rolling_window:
            recent_rates.pop(0)

        # Определяем сигнал входа/выхода
        should_enter = False
        should_exit = False

        if strategy == "always_in":
            should_enter = not in_position
            should_exit = False  # никогда не выходим

        elif strategy == "threshold":
            if not in_position and rate > entry_threshold:
                should_enter = True
            elif in_position and rate < exit_threshold:
                should_exit = True

        elif strategy == "rolling_mean":
            if rolling_count >= 10:  # минимум 10 наблюдений
                rolling_avg = rolling_sum / rolling_count
                if not in_position and rate > rolling_avg:
                    should_enter = True
                elif in_position and rate < rolling_avg * 0.5:
                    should_exit = True

        elif strategy == "percentile":
            if len(recent_rates) >= 30:  # минимум 30 наблюдений
                threshold = np.percentile(recent_rates, percentile * 100)
                if not in_position and rate > threshold:
                    should_enter = True
                elif in_position and rate < np.percentile(recent_rates, 25):
                    should_exit = True

        # Исполнение входа
        if should_enter and not in_position:
            in_position = True
            n_entries += 1
            # Платим издержки входа
            entry_cost = capital_efficiency * cost_frac / 2.0  # половина round-trip
            equity *= (1.0 - entry_cost)
            total_costs_paid += entry_cost

        # Исполнение выхода
        if should_exit and in_position:
            in_position = False
            n_exits += 1
            # Платим издержки выхода
            exit_cost = capital_efficiency * cost_frac / 2.0  # половина round-trip
            equity *= (1.0 - exit_cost)
            total_costs_paid += exit_cost

        # Сбор фандинга (если в позиции)
        if in_position:
            periods_in_market += 1
            # Фандинг: положительный = мы получаем (шорт перпа)
            # Отрицательный = мы платим
            funding_income = capital_efficiency * rate
            equity *= (1.0 + funding_income)
            total_funding_collected += funding_income

        equity_curve.append(equity)

    # Закрываем позицию в конце (если открыта)
    if in_position:
        exit_cost = capital_efficiency * cost_frac / 2.0
        equity *= (1.0 - exit_cost)
        total_costs_paid += exit_cost
        n_exits += 1
        equity_curve[-1] = equity

    # Метрики
    equity_arr = np.array(equity_curve)
    total_return = equity - 1.0
    n_days = n / FUNDING_PERIODS_PER_DAY
    n_years = n_days / DAYS_PER_YEAR

    if n_years > 0 and equity > 0:
        annualized_return = (equity ** (1.0 / n_years)) - 1.0
    else:
        annualized_return = 0.0

    # Максимальная просадка
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr - peak) / peak
    max_drawdown = abs(float(drawdown.min())) if len(drawdown) > 0 else 0.0

    # Sharpe (по 8-часовым периодам, годовая)
    returns = np.diff(equity_arr) / equity_arr[:-1]
    if len(returns) > 1 and np.std(returns) > 0:
        periods_per_year = FUNDING_PERIODS_PER_DAY * DAYS_PER_YEAR
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(periods_per_year)
    else:
        sharpe = 0.0

    time_in_market = periods_in_market / n if n > 0 else 0.0

    return {
        "total_return_pct": round(total_return * 100, 3),
        "annualized_return_pct": round(annualized_return * 100, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 3),
        "sharpe": round(sharpe, 2),
        "funding_collected_pct": round(total_funding_collected * 100, 3),
        "costs_paid_pct": round(total_costs_paid * 100, 3),
        "n_entries": n_entries,
        "n_exits": n_exits,
        "time_in_market_pct": round(time_in_market * 100, 1),
        "n_periods": n,
        "n_days": round(n_days, 1),
        "n_years": round(n_years, 2),
    }


# ------------------------------------------------------------------
# Агрегация по годам
# ------------------------------------------------------------------
def simulate_by_year(
    df: pd.DataFrame,
    strategy: str,
    cost_pct: float,
    **kwargs,
) -> pd.DataFrame:
    """Симулирует стратегию по годам для одного символа."""
    rows = []
    for year, group in df.groupby("year"):
        rates = group["funding_rate"].to_numpy()
        result = simulate_funding_arb(
            rates, strategy=strategy, cost_pct=cost_pct, **kwargs
        )
        result["year"] = year
        result["symbol"] = df.attrs.get("symbol", "?")
        rows.append(result)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Основной прогон
# ------------------------------------------------------------------
STRATEGIES = {
    "always_in": {
        "desc": "Всегда в позиции (собираем весь фандинг)",
        "params": {},
    },
    "threshold_001": {
        "desc": "Вход при фандинге > 0.01%, выход при < 0",
        "params": {"entry_threshold": 0.0001, "exit_threshold": 0.0},
    },
    "threshold_003": {
        "desc": "Вход при фандинге > 0.03%, выход при < 0.01%",
        "params": {"entry_threshold": 0.0003, "exit_threshold": 0.0001},
    },
    "rolling_mean": {
        "desc": "Вход при фандинге > скользящего среднего (30 дней)",
        "params": {"rolling_window": 90},
    },
    "percentile_75": {
        "desc": "Вход при фандинге > 75-го перцентиля (30 дней)",
        "params": {"percentile": 0.75, "rolling_window": 90},
    },
}


def main():
    symbols = get_available_funding_symbols()
    print(f"Монет с данными фандинга: {len(symbols)}")
    print(f"Издержки: тейкер={COST_TAKER}%, мейкер={COST_MAKER}%")
    print(f"Плечо: {LEVERAGE}x (эффективность капитала: {LEVERAGE/(LEVERAGE+1):.1%})")
    print()

    all_results = []

    for symbol in symbols:
        df = load_funding(symbol)
        if df is None:
            continue
        df.attrs["symbol"] = symbol

        print(f"{'='*100}")
        print(f"{symbol} | записей: {len(df)} | период: {df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()}")
        print(f"{'='*100}")

        # Общая статистика фандинга
        rates = df["funding_rate"]
        print(f"  Фандинг: mean={rates.mean()*100:.4f}%, "
              f"median={rates.median()*100:.4f}%, "
              f"std={rates.std()*100:.4f}%, "
              f"min={rates.min()*100:.4f}%, "
              f"max={rates.max()*100:.4f}%")
        print(f"  Доля положительных периодов: {(rates > 0).mean()*100:.1f}%")
        print(f"  Годовой фандинг (если всегда в позиции): "
              f"{rates.mean() * FUNDING_PERIODS_PER_DAY * DAYS_PER_YEAR * 100:.1f}%")
        print()

        for strat_name, strat_info in STRATEGIES.items():
            # Тейкер
            result_taker = simulate_funding_arb(
                rates.to_numpy(),
                strategy=strat_name,
                cost_pct=COST_TAKER,
                **strat_info["params"],
            )
            result_taker["symbol"] = symbol
            result_taker["strategy"] = strat_name
            result_taker["cost_type"] = "taker"
            all_results.append(result_taker)

            # Мейкер
            result_maker = simulate_funding_arb(
                rates.to_numpy(),
                strategy=strat_name,
                cost_pct=COST_MAKER,
                **strat_info["params"],
            )
            result_maker["symbol"] = symbol
            result_maker["strategy"] = strat_name
            result_maker["cost_type"] = "maker"
            all_results.append(result_maker)

        # Печатаем результаты для этого символа
        sym_results = [r for r in all_results if r["symbol"] == symbol]
        print(f"  {'Стратегия':<20} {'Издержки':<8} {'Годовой %':<12} {'Макс.ДД %':<12} "
              f"{'Sharpe':<8} {'Входов':<8} {'В рынке %':<10}")
        print(f"  {'-'*90}")
        for r in sym_results:
            print(f"  {r['strategy']:<20} {r['cost_type']:<8} "
                  f"{r['annualized_return_pct']:>+10.2f}% "
                  f"{r['max_drawdown_pct']:>10.3f}% "
                  f"{r['sharpe']:>7.2f} "
                  f"{r['n_entries']:>7} "
                  f"{r['time_in_market_pct']:>9.1f}%")
        print()

    # ------------------------------------------------------------------
    # Кросс-активная сводка
    # ------------------------------------------------------------------
    if not all_results:
        print("Нет результатов.")
        return

    df_results = pd.DataFrame(all_results)

    print(f"\n{'='*100}")
    print("КРОСС-АКТИВНАЯ СВОДКА (среднее по всем монетам)")
    print(f"{'='*100}")

    summary = df_results.groupby(["strategy", "cost_type"]).agg({
        "annualized_return_pct": "mean",
        "max_drawdown_pct": "mean",
        "sharpe": "mean",
        "time_in_market_pct": "mean",
        "n_entries": "mean",
        "total_return_pct": "mean",
    }).round(3).reset_index()

    summary = summary.sort_values(
        ["cost_type", "annualized_return_pct"],
        ascending=[True, False],
    )

    print(summary.to_string(index=False))

    # ------------------------------------------------------------------
    # Разбивка по годам (для лучшей стратегии)
    # ------------------------------------------------------------------
    print(f"\n{'='*100}")
    print("РАЗБИВКА ПО ГОДАМ (always_in, тейкер)")
    print(f"{'='*100}")

    year_rows = []
    for symbol in symbols:
        df = load_funding(symbol)
        if df is None:
            continue
        df.attrs["symbol"] = symbol
        year_df = simulate_by_year(df, "always_in", cost_pct=COST_TAKER)
        year_rows.append(year_df)

    if year_rows:
        df_years = pd.concat(year_rows, ignore_index=True)
        year_summary = df_years.groupby("year").agg({
            "annualized_return_pct": "mean",
            "max_drawdown_pct": "mean",
            "sharpe": "mean",
            "funding_collected_pct": "mean",
            "costs_paid_pct": "mean",
        }).round(3).reset_index()
        print(year_summary.to_string(index=False))

    # ------------------------------------------------------------------
    # Вердикт
    # ------------------------------------------------------------------
    print(f"\n{'='*100}")
    print("ВЕРДИКТ")
    print(f"{'='*100}")

    # Лучшая стратегия с мейкером
    maker_results = df_results[df_results["cost_type"] == "maker"]
    if not maker_results.empty:
        best = maker_results.groupby("strategy")["annualized_return_pct"].mean().idxmax()
        best_data = maker_results[maker_results["strategy"] == best]
        avg_ann = best_data["annualized_return_pct"].mean()
        avg_dd = best_data["max_drawdown_pct"].mean()
        avg_sharpe = best_data["sharpe"].mean()
        frac_positive = (best_data["annualized_return_pct"] > 0).mean()

        print(f"Лучшая стратегия (мейкер): {best}")
        print(f"  Средний годовой возврат: {avg_ann:+.2f}%")
        print(f"  Средняя макс. просадка: {avg_dd:.3f}%")
        print(f"  Средний Sharpe: {avg_sharpe:.2f}")
        print(f"  Доля монет с положительным возвратом: {frac_positive*100:.0f}%")
        print()
        if avg_ann > 3.0 and frac_positive > 0.7:
            print("✅ ФАНДИНГ-АРБИТРАЖ ИМЕЕТ СТРУКТУРНЫЙ ЭДЖ")
            print("   Средний годовой возврат > 3% на большинстве монет.")
            print("   Это НЕ зависит от предсказания направления цены.")
        elif avg_ann > 0:
            print("⚠️ ФАНДИНГ-АРБИТРАЖ МАРЖИНАЛЕН")
            print("   Возврат положительный, но низкий или нестабильный.")
        else:
            print("❌ ФАНДИНГ-АРБИТРАЖ НЕ РАБОТАЕТ")
            print("   Издержки превышают доход от фандинга.")


if __name__ == "__main__":
    main()