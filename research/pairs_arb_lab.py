"""
Лаборатория парного статистического арбитража.

Проверяет все пары монет на коинтеграцию и торгует mean-reversion спреда.
Для коинтегрированной пары (A, B):
  spread = log(A) - β * log(B)
  вход:   |z-score(spread)| > entry_z
  выход:  |z-score(spread)| < exit_z
  стоп:   |z-score(spread)| > stop_z

Запуск: python research/pairs_arb_lab.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

from data_loader import get_available_symbols, get_available_years, load_local_data

# ------------------------------------------------------------------
# Параметры стратегии
# ------------------------------------------------------------------
# Издержки на каждую ногу (мейкер)
COST_PER_LEG_PCT = 0.05  # 0.05% на ногу
TOTAL_COST_PCT = COST_PER_LEG_PCT * 2  # 0.10% round-trip на пару

# Таймфрейм: ресэмплируем 15m → 1h для уменьшения шума
RESAMPLE_FREQ = "1h"

# Параметры z-score
ZSCORE_WINDOW_DAYS = 20  # окно для z-score (дней)

# Пороги стратегии
ENTRY_Z = 2.0    # вход при отклонении > 2σ
EXIT_Z = 0.5     # выход при возврате к 0.5σ
STOP_Z = 4.0     # стоп при экстремальном отклонении

# Walk-forward: доля данных на тренировку коинтеграции
TRAIN_FRAC = 0.5

# Статистический порог коинтеграции (p-value теста Engle-Granger)
COINT_PVALUE_THRESHOLD = 0.05

# Минимум сделок для валидного результата
MIN_TRADES = 10


# ------------------------------------------------------------------
# Загрузка и ресэмплинг
# ------------------------------------------------------------------
def load_symbol_hourly(symbol: str) -> pd.DataFrame | None:
    """Загружает все годы символа и ресэмплит в 1h."""
    years = get_available_years(symbol)
    if not years:
        return None

    dfs = []
    for year in years:
        try:
            df = load_local_data(symbol, year)
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    df = df.set_index("timestamp")

    # Ресэмплинг в 1h
    hourly = df.resample(RESAMPLE_FREQ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    return hourly


# ------------------------------------------------------------------
# Проверка коинтеграции
# ------------------------------------------------------------------
def check_cointegration(price_a: np.ndarray, price_b: np.ndarray) -> dict:
    """Тест Engle-Granger на коинтеграцию двух ценовых рядов.

    Возвращает: p-value, beta (хедж-рейт), stationarity p-value спреда.
    """
    if len(price_a) < 200 or len(price_b) < 200:
        return {"coint_pvalue": 1.0, "beta": 1.0, "adf_pvalue": 1.0, "valid": False}

    log_a = np.log(price_a)
    log_b = np.log(price_b)

    # Engle-Granger тест (coint из statsmodels)
    try:
        score, pvalue, crit_values = coint(log_a, log_b)
    except Exception:
        return {"coint_pvalue": 1.0, "beta": 1.0, "adf_pvalue": 1.0, "valid": False}

    if pvalue > COINT_PVALUE_THRESHOLD:
        return {"coint_pvalue": pvalue, "beta": 1.0, "adf_pvalue": 1.0, "valid": False}

    # Оценка хедж-рейта через OLS: log_a = β * log_b + const + ε
    # β = cov(log_a, log_b) / var(log_b)
    cov = np.cov(log_a, log_b)[0, 1]
    var_b = np.var(log_b)
    if var_b == 0:
        return {"coint_pvalue": pvalue, "beta": 1.0, "adf_pvalue": 1.0, "valid": False}

    beta = cov / var_b

    # Построение спреда и проверка стационарности (ADF)
    spread = log_a - beta * log_b
    try:
        adf_stat, adf_pvalue, *_ = adfuller(spread, maxlag=1)
    except Exception:
        adf_pvalue = 1.0

    valid = (pvalue < COINT_PVALUE_THRESHOLD) and (adf_pvalue < 0.05)

    return {
        "coint_pvalue": pvalue,
        "beta": float(beta),
        "adf_pvalue": adf_pvalue,
        "valid": valid,
    }


# ------------------------------------------------------------------
# Симуляция торговли на спреде
# ------------------------------------------------------------------
def simulate_pair_trading(
    price_a: np.ndarray,
    price_b: np.ndarray,
    beta: float,
    zscore_window: int,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    cost_pct: float,
) -> dict:
    """Симулирует торговлю на mean-reversion спреда.

    Вход в позицию:
      z > entry_z  → шорт A, лонг B
      z < -entry_z → лонг A, шорт B
    Выход:
      |z| < exit_z (возврат к среднему)
      |z| > stop_z (стоп)
    """
    n = len(price_a)
    if n < zscore_window + 10:
        return {"n_trades": 0, "total_pnl_pct": 0.0, "trades": []}

    log_a = np.log(price_a)
    log_b = np.log(price_b)
    spread = log_a - beta * log_b

    # Rolling mean и std для z-score
    mean = pd.Series(spread).rolling(zscore_window, min_periods=zscore_window).mean().to_numpy()
    std = pd.Series(spread).rolling(zscore_window, min_periods=zscore_window).std().to_numpy()

    # Защита от нулевого std
    std = np.where(std == 0, np.nan, std)
    z = (spread - mean) / std

    trades = []
    in_position = False
    position_direction = 0  # +1: лонг A / шорт B; -1: шорт A / лонг B
    entry_spread = 0.0

    cost_frac = cost_pct / 100.0

    for i in range(zscore_window, n):
        zi = z[i]
        if np.isnan(zi):
            continue

        if not in_position:
            # Ищем вход
            if zi > entry_z:
                in_position = True
                position_direction = -1  # шорт A, лонг B
                entry_spread = spread[i]
            elif zi < -entry_z:
                in_position = True
                position_direction = +1  # лонг A, шорт B
                entry_spread = spread[i]
        else:
            # В позиции: ищем выход
            exit_reason = None
            if position_direction == -1:
                # Шорт A, лонг B: прибыль когда z уменьшается
                if zi < exit_z:
                    exit_reason = "mean_revert"
                elif zi > stop_z:
                    exit_reason = "stop"
            else:
                # Лонг A, шорт B: прибыль когда z увеличивается
                if zi > -exit_z:
                    exit_reason = "mean_revert"
                elif zi < -stop_z:
                    exit_reason = "stop"

            if exit_reason is not None:
                # P&L в логарифмических пунктах спреда
                spread_pnl = (spread[i] - entry_spread) * position_direction
                pnl_pct = spread_pnl * 100.0 - cost_pct  # вычитаем издержки

                trades.append({
                    "entry_idx": i - 1,
                    "exit_idx": i,
                    "direction": position_direction,
                    "entry_z": z[i - 1] if i > 0 else 0,
                    "exit_z": zi,
                    "pnl_pct": float(pnl_pct),
                    "exit_reason": exit_reason,
                    "holding_bars": i - (i - 1),
                })

                in_position = False
                position_direction = 0

    if not trades:
        return {"n_trades": 0, "total_pnl_pct": 0.0, "trades": []}

    pnls = np.array([t["pnl_pct"] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    return {
        "n_trades": len(trades),
        "n_wins": len(wins),
        "n_losses": len(losses),
        "win_rate": 100.0 * len(wins) / len(trades) if len(trades) else 0,
        "total_pnl_pct": float(pnls.sum()),
        "mean_pnl_pct": float(pnls.mean()),
        "median_pnl_pct": float(np.median(pnls)),
        "max_win_pct": float(pnls.max()),
        "max_loss_pct": float(pnls.min()),
        "sharpe_like": float(pnls.mean() / pnls.std()) if pnls.std() > 0 else 0.0,
        "trades": trades,
    }


# ------------------------------------------------------------------
# Основной прогон
# ------------------------------------------------------------------
def main():
    symbols = get_available_symbols()
    print(f"Монет с данными: {len(symbols)}")
    print(f"Параметры:")
    print(f"  Таймфрейм: {RESAMPLE_FREQ}")
    print(f"  Z-score окно: {ZSCORE_WINDOW_DAYS} дней")
    print(f"  Вход |z| > {ENTRY_Z}, выход |z| < {EXIT_Z}, стоп |z| > {STOP_Z}")
    print(f"  Издержки: {TOTAL_COST_PCT}% на сделку (мейкер, обе ноги)")
    print(f"  Walk-forward: тренировка на первых {TRAIN_FRAC*100:.0f}% данных")
    print()

    # Загружаем часовые данные для всех символов
    print("Загрузка и ресэмплинг данных...")
    hourly_data: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = load_symbol_hourly(sym)
        if df is not None and len(df) > 500:
            hourly_data[sym] = df
            print(f"  {sym}: {len(df)} часов с {df.index[0].date()} по {df.index[-1].date()}")

    print(f"\nЗагружено {len(hourly_data)} монет")
    print()

    # Все возможные пары
    pairs = list(combinations(hourly_data.keys(), 2))
    print(f"Пар для проверки: {len(pairs)}")
    print()

    results = []

    for sym_a, sym_b in pairs:
        df_a = hourly_data[sym_a]
        df_b = hourly_data[sym_b]

        # Синхронизируем по времени (inner join)
        merged = df_a[["close"]].join(df_b[["close"]], how="inner", lsuffix="_a", rsuffix="_b")
        merged = merged.dropna()

        if len(merged) < 1000:
            continue

        prices_a = merged["close_a"].to_numpy()
        prices_b = merged["close_b"].to_numpy()

        # Разделение на train/test
        split_idx = int(len(merged) * TRAIN_FRAC)

        # Проверка коинтеграции на TRAIN
        coint_result = check_cointegration(prices_a[:split_idx], prices_b[:split_idx])

        if not coint_result["valid"]:
            results.append({
                "symbol_a": sym_a,
                "symbol_b": sym_b,
                "coint_pvalue": coint_result["coint_pvalue"],
                "beta": coint_result["beta"],
                "coint_valid": False,
                "n_trades": 0,
                "n_wins": 0,
                "win_rate": 0.0,
                "total_pnl_pct": 0.0,
                "annualized_pct": 0.0,
                "mean_pnl_pct": 0.0,
                "sharpe_like": 0.0,
                "test_years": 0.0,
            })
            continue

        # Тестирование стратегии на TEST
        test_a = prices_a[split_idx:]
        test_b = prices_b[split_idx:]
        beta = coint_result["beta"]
        hours_per_day = 24
        zscore_window = ZSCORE_WINDOW_DAYS * hours_per_day

        trade_result = simulate_pair_trading(
            test_a, test_b, beta,
            zscore_window=zscore_window,
            entry_z=ENTRY_Z,
            exit_z=EXIT_Z,
            stop_z=STOP_Z,
            cost_pct=TOTAL_COST_PCT,
        )

        # Годовая доходность
        n_test_hours = len(test_a)
        n_test_years = n_test_hours / (DAYS_PER_YEAR := 365.25) / hours_per_day
        annualized = trade_result["total_pnl_pct"] / n_test_years if n_test_years > 0 else 0

        results.append({
            "symbol_a": sym_a,
            "symbol_b": sym_b,
            "coint_pvalue": round(coint_result["coint_pvalue"], 4),
            "beta": round(beta, 4),
            "coint_valid": True,
            "n_trades": trade_result["n_trades"],
            "n_wins": trade_result.get("n_wins", 0),
            "win_rate": round(trade_result.get("win_rate", 0), 1),
            "total_pnl_pct": round(trade_result["total_pnl_pct"], 3),
            "annualized_pct": round(annualized, 2),
            "mean_pnl_pct": round(trade_result.get("mean_pnl_pct", 0), 4),
            "sharpe_like": round(trade_result.get("sharpe_like", 0), 3),
            "test_years": round(n_test_years, 2),
        })

    # ------------------------------------------------------------------
    # Вывод результатов
    # ------------------------------------------------------------------
    df_results = pd.DataFrame(results)

    # Сортируем: сначала коинтегрированные, потом по annualized
    df_results = df_results.sort_values(
        ["coint_valid", "annualized_pct"],
        ascending=[False, False],
    )

    print(f"{'='*120}")
    print("ВСЕ ПАРЫ (отсортированы по коинтеграции и годовой доходности)")
    print(f"{'='*120}")

    display_cols = [
        "symbol_a", "symbol_b", "coint_pvalue", "beta", "coint_valid",
        "n_trades", "win_rate", "total_pnl_pct", "annualized_pct", "sharpe_like",
    ]
    print(df_results[display_cols].to_string(index=False))

    # Только коинтегрированные пары
    coint_pairs = df_results[df_results["coint_valid"]]

    print(f"\n{'='*120}")
    print(f"КОИНТЕГРИРОВАННЫЕ ПАРЫ ({len(coint_pairs)} из {len(df_results)})")
    print(f"{'='*120}")

    if coint_pairs.empty:
        print("  Нет ни одной коинтегрированной пары на тренировочном окне")
        print("  Это означает, что на часовом таймфрейме стабильных пар нет")
    else:
        print(coint_pairs[display_cols].to_string(index=False))

    # Робастные пары
    robust = coint_pairs[
        (coint_pairs["n_trades"] >= MIN_TRADES)
        & (coint_pairs["annualized_pct"] > TOTAL_COST_PCT * 10)  # хотя бы 1% годовых после издержек
        & (coint_pairs["win_rate"] > 50)
    ]

    print(f"\n{'='*120}")
    print(f"РОБАСТНЫЕ ПАРЫ (сделок>={MIN_TRADES}, годовой>1%, win rate>50%)")
    print(f"{'='*120}")

    if robust.empty:
        print("  Нет робастных пар")
    else:
        print(robust[display_cols].to_string(index=False))

    # Итоговый вердикт
    print(f"\n{'='*120}")
    print("ИТОГОВЫЙ ВЕРДИКТ")
    print(f"{'='*120}")

    n_coint = len(coint_pairs)
    n_robust = len(robust)
    if n_robust > 0:
        best = robust.iloc[0]
        print(f"  ✅ Парный арбитраж ИМЕЕТ эдж")
        print(f"     Коинтегрированных пар: {n_coint}")
        print(f"     Робастных пар: {n_robust}")
        print(f"     Лучшая пара: {best['symbol_a']}/{best['symbol_b']}")
        print(f"       Годовой возврат: {best['annualized_pct']:+.2f}%")
        print(f"       Win rate: {best['win_rate']:.1f}%")
        print(f"       Sharpe-like: {best['sharpe_like']:.3f}")
    elif n_coint > 0:
        print(f"  ⚠️ Коинтегрированные пары есть ({n_coint}), но не робастны")
        print(f"     Стратегия не переживает издержки и шум")
    else:
        print(f"  ❌ На часовом таймфрейме нет стабильных пар")
        print(f"     Ни одна из {len(df_results)} пар не прошла тест Engle-Granger")


if __name__ == "__main__":
    main()