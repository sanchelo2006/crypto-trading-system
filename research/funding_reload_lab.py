"""
Лаборатория фандинг-релоад: структурные события вокруг выплаты фандинга.

Фандинг на Bybit выплачивается каждые 8 часов: 00:00, 08:00, 16:00 UTC.
Гипотеза: вокруг этих моментов происходит предсказуемое движение цены,
связанное с закрытием/открытием позиций для получения или избегания выплаты.

Стратегии (4 направления × пороги фандинга):
  pre_short:   шорт ПЕРЕД выплатой при положительном фандинге
               (лонги закрываются, чтобы не платить → давление вниз)
  pre_long:    лонг ПЕРЕД выплатой при отрицательном фандинге
               (шорты закрываются, чтобы не платить → давление вверх)
  post_long:   лонг ПОСЛЕ выплаты при положительном фандинге
               (давление снято → лонги возвращаются → релоад вверх)
  post_short:  шорт ПОСЛЕ выплаты при отрицательном фандинге
               (давление снято → шорты возвращаются → релоад вниз)

Тайминг (согласован с impulse_lab._simulate_fixed, вход = open[s+1]):
  i      = индекс свечи, открывающейся в момент выплаты фандинга;
  PRE    : сигнал s = i-2 → вход open[i-1] (свеча перед выплатой),
           удержание 1 свеча → выход ровно в момент выплаты;
  POST   : сигнал s = i-1 → вход open[i] (свеча выплаты),
           удержание 2 свечи → выход через 30 минут после выплаты.

ВАЖНО: симулятор моделирует ТОЛЬКО движение цены (SL/TP/тайм-аут).
Сама выплата фандинга в P&L не учитывается — мы торгуем именно
ценовую реакцию вокруг выплаты, а не собираем фандинг.

Запуск из корня проекта:  python research/funding_reload_lab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from data_loader import get_available_symbols, get_available_years, load_local_data
import research.impulse_lab as lab

# ------------------------------------------------------------------
# Параметры
# ------------------------------------------------------------------
BACKTEST_DATA_DIR = Path(__file__).resolve().parent.parent / "backtest_data"
FUNDING_DIR = BACKTEST_DATA_DIR / "funding"

# Пороги фандинга (в долях, не процентах). 0.0001 = 0.01% = дефолт Bybit.
THRESHOLDS = [0.0001, 0.0002, 0.0003, 0.0005]

# Выходы в ATR (страховочные; основной выход — по времени)
SL_ATR = 1.5
TP_ATR = 1.5

# Удержание зависит от фазы:
#   PRE  — 1 свеча (выход ровно в момент выплаты)
#   POST — 2 свечи (30 минут после выплаты)
HOLD = {
    "pre_short": 1,
    "pre_long": 1,
    "post_long": 2,
    "post_short": 2,
}

DIRECTIONS = {
    "pre_short": -1,
    "pre_long": +1,
    "post_long": +1,
    "post_short": -1,
}

# Издержки: два сценария (см. паспорт проекта)
COST_TAKER = 0.28   # % round-trip (тейкер)
COST_MAKER = 0.04   # % round-trip (мейкер)

ATR_PERIOD = 14


# ------------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------------
def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """Стандартный ATR без зависимости от TA-Lib."""
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def load_funding(symbol: str) -> pd.DataFrame | None:
    """Загружает funding-данные монеты (единый файл за все годы)."""
    path = FUNDING_DIR / f"{symbol}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "funding_rate" not in df.columns:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    df = df.dropna(subset=["timestamp", "funding_rate"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df if not df.empty else None


def build_event_indices(ohlcv: pd.DataFrame, funding: pd.DataFrame, threshold: float) -> dict:
    """
    Строит массивы сигнальных индексов для 4 стратегий при данном пороге.

    Возвращает dict: {"pre_short": [...], "pre_long": [...],
                      "post_long": [...], "post_short": [...]}.
    """
    ts_to_idx = {ts: pos for pos, ts in enumerate(ohlcv["timestamp"])}

    acc = {name: [] for name in DIRECTIONS}

    for _, row in funding.iterrows():
        i = ts_to_idx.get(row["timestamp"])
        if i is None:
            continue
        rate = row["funding_rate"]

        if rate > threshold:
            # положительный фандинг: лонги платят
            if i - 2 >= 0:
                acc["pre_short"].append(i - 2)   # вход open[i-1]
            if i - 1 >= 0:
                acc["post_long"].append(i - 1)   # вход open[i]
        elif rate < -threshold:
            # отрицательный фандинг: шорты платят
            if i - 2 >= 0:
                acc["pre_long"].append(i - 2)    # вход open[i-1]
            if i - 1 >= 0:
                acc["post_short"].append(i - 1)  # вход open[i]

    return {name: np.array(v, dtype=int) for name, v in acc.items()}


# ------------------------------------------------------------------
# Прогон
# ------------------------------------------------------------------
def run_for_cost(cost_pct: float) -> pd.DataFrame:
    """Прогон по всем (монета, год, порог, стратегия) при заданных издержках."""
    # Издержки задаём через глобальные переменные impulse_lab —
    # тот же приём, что использует portfolio_lab.py.
    lab.FEE_PCT = cost_pct
    lab.SLIPPAGE_PCT = 0.0

    rows = []
    symbols = [
        s for s in get_available_symbols()
        if (FUNDING_DIR / f"{s}.csv").exists()
    ]
    print(f"  Монет с funding-данными: {len(symbols)}")

    for symbol in symbols:
        funding_all = load_funding(symbol)
        if funding_all is None:
            continue

        for year in get_available_years(symbol):
            try:
                df = load_local_data(symbol, year)
            except FileNotFoundError:
                continue
            if len(df) < 100:
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
            df["atr_14"] = compute_atr(df)

            # funding за этот год
            funding = funding_all[funding_all["timestamp"].dt.year == year]
            if funding.empty:
                continue

            for threshold in THRESHOLDS:
                idx_map = build_event_indices(df, funding, threshold)
                for sig_name, idx in idx_map.items():
                    if len(idx) == 0:
                        continue
                    res = lab._simulate_fixed(
                        df,
                        idx,
                        DIRECTIONS[sig_name],
                        sl_atr=SL_ATR,
                        tp_atr=TP_ATR,
                        max_hold=HOLD[sig_name],
                    )
                    if res["n_traded"] == 0 or res["exp"] is None:
                        continue
                    rows.append(
                        {
                            "symbol": symbol,
                            "year": year,
                            "threshold": threshold,
                            "signal": sig_name,
                            "n_traded": res["n_traded"],
                            "wr": res["wr"],
                            "pf": res["pf"],
                            "exp": res["exp"],
                        }
                    )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Кросс-активная сводка по (сигнал, порог)."""
    print(f"\n{'='*100}")
    print(f"КРОСС-АКТИВНАЯ СВОДКА ({label})")
    print(f"{'='*100}")
    if df.empty:
        print("  Нет сделок")
        return pd.DataFrame()

    agg = (
        df.groupby(["signal", "threshold"])
        .agg(
            datasets=("exp", "count"),
            trades=("n_traded", "sum"),
            exp_mean=("exp", "mean"),
            frac_pos=("exp", lambda x: round((x > 0).mean(), 2)),
            wr_mean=("wr", "mean"),
            pf_mean=("pf", lambda x: x.replace(999.0, np.nan).mean()),
        )
        .reset_index()
        .sort_values("exp_mean", ascending=False)
    )
    print(agg.to_string(index=False))
    return agg


def yearly_breakdown(df: pd.DataFrame, label: str):
    """Разбивка по годам для проверки режимной независимости."""
    if df.empty:
        return
    print(f"\n{'='*100}")
    print(f"РАЗБИВКА ПО ГОДАМ ({label})")
    print(f"{'='*100}")
    yr = (
        df.groupby(["signal", "threshold", "year"])
        .agg(exp_mean=("exp", "mean"), trades=("n_traded", "sum"))
        .reset_index()
    )
    pivot = yr.pivot_table(
        index=["signal", "threshold"], columns="year", values="exp_mean"
    ).round(3)
    print(pivot.to_string())


def verdict(df: pd.DataFrame, label: str):
    """Финальный вердикт по сценарию издержек."""
    print(f"\n--- ВЕРДИКТ ({label}) ---")
    if df.empty:
        print("  Нет сделок")
        return
    agg = (
        df.groupby(["signal", "threshold"])
        .agg(
            trades=("n_traded", "sum"),
            exp_mean=("exp", "mean"),
            frac_pos=("exp", lambda x: (x > 0).mean()),
        )
    )
    robust = agg[
        (agg["exp_mean"] > 0)
        & (agg["frac_pos"] >= 0.6)
        & (agg["trades"] >= 100)
    ]
    if robust.empty:
        print("  Нет робастных сигналов (exp>0, frac_pos>=0.6, trades>=100)")
    else:
        print("  Робастные сигналы:")
        print(robust.to_string())


def main():
    print("FUNDING RELOAD LAB")
    print(f"Выходы: SL={SL_ATR} ATR, TP={TP_ATR} ATR, hold={HOLD}")
    print(f"Пороги фандинга: {THRESHOLDS}")
    print(f"Издержки: тейкер={COST_TAKER}%, мейкер={COST_MAKER}%")

    print(f"\n>>> Прогон с издержками тейкера {COST_TAKER}% ...")
    df_taker = run_for_cost(COST_TAKER)
    print(f">>> Прогон с издержками мейкера {COST_MAKER}% ...")
    df_maker = run_for_cost(COST_MAKER)

    agg_taker = summarize(df_taker, f"ТЕЙКЕР {COST_TAKER}%")
    agg_maker = summarize(df_maker, f"МЕЙКЕР {COST_MAKER}%")

    yearly_breakdown(df_maker, f"МЕЙКЕР {COST_MAKER}%")

    print(f"\n{'='*100}")
    print("ИТОГОВЫЙ ВЕРДИКТ")
    print(f"{'='*100}")
    verdict(df_taker, f"ТЕЙКЕР {COST_TAKER}%")
    verdict(df_maker, f"МЕЙКЕР {COST_MAKER}%")


if __name__ == "__main__":
    main()