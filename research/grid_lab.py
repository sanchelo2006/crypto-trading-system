"""
Лаборатория сеточной торговли с режимным переключением (Шаг 2 + гибрид).
ИСПРАВЛЕННАЯ ВЕРСИЯ: корректный учёт капитала и сделок.

Режимный детектор: Efficiency Ratio (доля чистого смещения в полном пути).
  низкий ER  -> рынок в пиле  -> работает СЕТКА
  высокий ER -> рынок в тренде -> сетка останавливается

Три конфигурации для сравнения:
  A. naive_grid : сетка активна всегда (базовая линия, максимальный риск)
  B. flat_only  : сетка только во флэте; в тренде — выход в кэш
  C. hybrid     : сетка во флэте; в тренде — следование за трендом

Запуск из корня проекта:
    python research/grid_lab.py
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from data_loader import get_available_symbols, get_available_years, load_local_data

# ------------------------------------------------------------------
# Параметры сетки (зафиксированы заранее — не оптимизируем)
# ------------------------------------------------------------------
SPACING_PCT = 0.4          # шаг сетки в % (лучший по Шагу 1)
LEVELS_PER_SIDE = 10       # уровней в каждую сторону от якоря
ORDER_SIZE_PCT = 1.0       # размер одного лота в % от капитала

# Параметры режимного детектора (на основе выводов Шага 1)
ER_WINDOW = 96             # окно для Efficiency Ratio (24 часа на 15m)
FLAT_THRESHOLD = 0.15      # ER ниже этого — флэт
TREND_THRESHOLD = 0.25     # ER выше этого — тренд (гистерезис)

# Параметры трендовой части (конфиг C)
TREND_SIZE_PCT = 30.0      # доля капитала на трендовую позицию
TREND_STOP_ATR = 3.0       # стоп трендовой позиции в ATR

# Сценарии издержек на ОДНУ сторону
COSTS = {
    "maker": 0.02,         # лимитные ордера сетки
    "taker": 0.10,         # консервативный сценарий
}

ATR_PERIOD = 14
INITIAL_CAPITAL = 10_000.0

EXCLUDE_DIRS = {"funding", "long_short", "open_interest"}


# ------------------------------------------------------------------
# Загрузка данных
# ------------------------------------------------------------------
def load_symbol_full(symbol: str) -> pd.DataFrame | None:
    frames = []
    for year in get_available_years(symbol):
        try:
            frames.append(load_local_data(symbol, year))
        except FileNotFoundError:
            continue
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = (df.sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last")
            .reset_index(drop=True))
    return df if len(df) > 2000 else None


# ------------------------------------------------------------------
# Индикаторы
# ------------------------------------------------------------------
def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> np.ndarray:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean().to_numpy()


def compute_er(close: np.ndarray, window: int = ER_WINDOW) -> np.ndarray:
    s = pd.Series(close)
    net = (s - s.shift(window)).abs()
    path = s.diff().abs().rolling(window, min_periods=window).sum()
    return (net / path.replace(0.0, np.nan)).to_numpy()


def detect_regimes(er: np.ndarray, flat_thr: float, trend_thr: float) -> np.ndarray:
    n = len(er)
    regimes = np.zeros(n, dtype=bool)
    in_flat = True
    for i in range(n):
        e = er[i]
        if np.isnan(e):
            regimes[i] = in_flat
            continue
        if in_flat and e > trend_thr:
            in_flat = False
        elif not in_flat and e < flat_thr:
            in_flat = True
        regimes[i] = in_flat
    return regimes


def trend_direction(close: np.ndarray, i: int, window: int = ER_WINDOW) -> int:
    if i - window < 0:
        return 0
    net = close[i] - close[i - window]
    if net > 0:
        return 1
    if net < 0:
        return -1
    return 0


# ------------------------------------------------------------------
# Сетка (ИСПРАВЛЕНО)
# ------------------------------------------------------------------
@dataclass
class GridState:
    anchor: float
    spacing: float
    levels_per_side: int
    order_usdt: float
    inventory: dict = field(default_factory=dict)
    realized_pnl: float = 0.0
    closed_cycles: int = 0        # НОВОЕ: счётчик нормальных циклов buy→sell

    def level_price(self, k: int) -> float:
        return self.anchor * (1.0 + k * self.spacing)

    def reset(self, new_anchor: float):
        self.anchor = new_anchor
        self.inventory = {}

    def unrealized_value(self, price: float) -> float:
        return sum(qty * price for qty, _ in self.inventory.values())

    def inventory_count(self) -> int:
        return len(self.inventory)

    def close_all(self, price: float, cost: float) -> float:
        """Закрывает весь инвентарь. ВОЗВРАЩАЕТ выручку в cash."""
        pnl = 0.0
        cash_return = 0.0
        for k, (qty, entry_price) in list(self.inventory.items()):
            proceeds = qty * price * (1.0 - cost)
            pnl += proceeds - qty * entry_price
            cash_return += proceeds
        self.inventory = {}
        self.realized_pnl += pnl
        return cash_return


def process_segment(grid: GridState, p_start: float, p_end: float,
                    cost: float, cash: float) -> float:
    if abs(p_end - p_start) < 1e-12:
        return cash

    direction = 1 if p_end > p_start else -1
    lo, hi = min(p_start, p_end), max(p_start, p_end)

    k_min = math.ceil((lo / grid.anchor - 1.0) / grid.spacing)
    k_max = math.floor((hi / grid.anchor - 1.0) / grid.spacing)

    if direction == -1:
        level_range = range(k_max, k_min - 1, -1)
    else:
        level_range = range(k_min, k_max + 1)

    for k in level_range:
        level_price = grid.level_price(k)
        if not (lo <= level_price <= hi):
            continue

        if direction == -1:
            if k < 0 and k not in grid.inventory:
                if cash >= grid.order_usdt * (1.0 + cost):
                    qty = grid.order_usdt / level_price
                    cash -= grid.order_usdt * (1.0 + cost)
                    grid.inventory[k] = (qty, level_price)
        else:
            src = k - 1
            if src in grid.inventory:
                qty, entry_price = grid.inventory.pop(src)
                proceeds = qty * level_price * (1.0 - cost)
                grid.realized_pnl += proceeds - qty * entry_price
                cash += proceeds
                grid.closed_cycles += 1    # НОВОЕ: считаем цикл

    return cash


def bar_price_path(open_: float, high: float, low: float, close: float):
    if close >= open_:
        return [open_, low, high, close]
    else:
        return [open_, high, low, close]


# ------------------------------------------------------------------
# Основная симуляция (ИСПРАВЛЕНО)
# ------------------------------------------------------------------
def simulate(df: pd.DataFrame, mode: str, cost: float) -> dict:
    close = df["close"].to_numpy(float)
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    open_ = df["open"].to_numpy(float)
    atr = compute_atr(df)
    er = compute_er(close)
    regimes = detect_regimes(er, FLAT_THRESHOLD, TREND_THRESHOLD)

    n = len(close)
    order_usdt = INITIAL_CAPITAL * (ORDER_SIZE_PCT / 100.0)
    spacing = SPACING_PCT / 100.0

    cash = INITIAL_CAPITAL
    grid = GridState(
        anchor=close[ER_WINDOW] if n > ER_WINDOW else close[0],
        spacing=spacing,
        levels_per_side=LEVELS_PER_SIDE,
        order_usdt=order_usdt,
    )
    grid_active = mode == "naive_grid"

    trend_pos = {"qty": 0.0, "direction": 0, "entry": 0.0, "stop": 0.0}
    trend_trades = 0

    equity_curve = np.zeros(n)
    regime_switches = 0
    last_regime_flat = regimes[ER_WINDOW] if n > ER_WINDOW else True

    for i in range(ER_WINDOW, n):
        is_flat = regimes[i]

        # ----- Переключение режимов -----
        if is_flat != last_regime_flat:
            regime_switches += 1

            if not is_flat:
                # ФЛЭТ -> ТРЕНД: останавливаем сетку
                if grid_active and grid.inventory_count() > 0:
                    inv_count = grid.inventory_count()
                    cash += grid.close_all(close[i], cost)  # ИСПРАВЛЕНО: деньги в cash
                    trend_trades += inv_count               # считаем принудительные закрытия
                grid_active = False

                if mode == "hybrid":
                    direction = trend_direction(close, i)
                    if direction != 0 and atr[i] > 0 and trend_pos["qty"] == 0.0:
                        pos_value = INITIAL_CAPITAL * (TREND_SIZE_PCT / 100.0)
                        qty = pos_value / close[i]
                        cash -= pos_value * cost
                        trend_pos = {
                            "qty": qty * direction,
                            "direction": direction,
                            "entry": close[i],
                            "stop": close[i] - direction * TREND_STOP_ATR * atr[i],
                        }
            else:
                # ТРЕНД -> ФЛЭТ: закрываем тренд, запускаем сетку
                if mode == "hybrid" and trend_pos["qty"] != 0.0:
                    direction = trend_pos["direction"]
                    exit_price = close[i]
                    qty_abs = abs(trend_pos["qty"])
                    proceeds = qty_abs * exit_price * (1.0 - cost)
                    cash += proceeds
                    trend_trades += 1
                    trend_pos = {"qty": 0.0, "direction": 0, "entry": 0.0, "stop": 0.0}

                grid.reset(close[i])
                grid_active = True

            last_regime_flat = is_flat

        # ----- Проверка стопа трендовой позиции -----
        if mode == "hybrid" and trend_pos["qty"] != 0.0:
            direction = trend_pos["direction"]
            hit_stop = (direction == 1 and low[i] <= trend_pos["stop"]) or \
                       (direction == -1 and high[i] >= trend_pos["stop"])
            if hit_stop:
                exit_price = trend_pos["stop"]
                qty_abs = abs(trend_pos["qty"])
                proceeds = qty_abs * exit_price * (1.0 - cost)
                cash += proceeds
                trend_trades += 1
                trend_pos = {"qty": 0.0, "direction": 0, "entry": 0.0, "stop": 0.0}

        # ----- Обработка сетки на баре -----
        if grid_active:
            path = bar_price_path(open_[i], high[i], low[i], close[i])
            for j in range(len(path) - 1):
                cash = process_segment(grid, path[j], path[j + 1], cost, cash)

        # ----- Mark-to-market -----
        equity = cash + grid.unrealized_value(close[i])
        if mode == "hybrid" and trend_pos["qty"] != 0.0:
            direction = trend_pos["direction"]
            equity += direction * (close[i] - trend_pos["entry"]) * abs(trend_pos["qty"])
        equity_curve[i] = equity

    # ----- Закрытие в конце периода -----
    if grid_active and grid.inventory_count() > 0:
        inv_count = grid.inventory_count()
        cash += grid.close_all(close[-1], cost)
        trend_trades += inv_count

    final_equity = cash
    if mode == "hybrid" and trend_pos["qty"] != 0.0:
        direction = trend_pos["direction"]
        qty_abs = abs(trend_pos["qty"])
        proceeds = qty_abs * close[-1] * (1.0 - cost)
        cash += proceeds
        final_equity = cash
        trend_trades += 1

    # ----- Метрики -----
    valid = equity_curve[ER_WINDOW:]
    total_return_pct = (final_equity / INITIAL_CAPITAL - 1.0) * 100.0
    n_years = len(valid) / (96.0 * 365.25)
    annualized_pct = ((final_equity / INITIAL_CAPITAL) ** (1.0 / max(n_years, 0.01)) - 1.0) * 100.0 \
        if final_equity > 0 else -100.0

    peak = np.maximum.accumulate(np.where(valid > 0, valid, INITIAL_CAPITAL))
    drawdown = (peak - np.where(valid > 0, valid, INITIAL_CAPITAL)) / peak
    max_dd_pct = float(drawdown.max() * 100.0) if len(drawdown) else 0.0

    flat_frac = float(regimes[ER_WINDOW:].mean())

    # ИТОГОВОЕ число сделок: циклы сетки + принудительные закрытия + трендовые
    total_trades = grid.closed_cycles + trend_trades

    return {
        "total_return_pct": round(total_return_pct, 2),
        "annualized_pct": round(annualized_pct, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "final_equity": round(final_equity, 2),
        "total_trades": total_trades,
        "grid_cycles": grid.closed_cycles,
        "regime_switches": regime_switches,
        "flat_time_pct": round(flat_frac * 100.0, 1),
        "realized_grid_pnl": round(grid.realized_pnl, 2),
    }


# ------------------------------------------------------------------
# Основной прогон
# ------------------------------------------------------------------
MODES = ["naive_grid", "flat_only", "hybrid"]
MODE_LABELS = {
    "naive_grid": "A. Сетка всегда (риск)",
    "flat_only": "B. Сетка только во флэте",
    "hybrid": "C. Гибрид (сетка + тренд)",
}


def main():
    symbols = [s for s in get_available_symbols() if s not in EXCLUDE_DIRS]
    print("GRID LAB — сетка с режимным переключением (ИСПРАВЛЕННАЯ ВЕРСИЯ)")
    print(f"Монет: {len(symbols)}")
    print(f"Шаг сетки: {SPACING_PCT}% | уровней: {LEVELS_PER_SIDE} | ER окно: {ER_WINDOW}")
    print(f"Флэт: ER < {FLAT_THRESHOLD} | Тренд: ER > {TREND_THRESHOLD}")
    print()

    all_rows = []

    for symbol in symbols:
        df = load_symbol_full(symbol)
        if df is None:
            print(f"  {symbol}: нет данных, пропуск")
            continue
        n_years = len(df) / (96.0 * 365.25)
        print(f"  {symbol}: {len(df)} баров, {n_years:.1f} лет")

        for cost_label, cost in COSTS.items():
            for mode in MODES:
                result = simulate(df, mode, cost)
                row = {
                    "symbol": symbol,
                    "mode": mode,
                    "cost": cost_label,
                    **result,
                }
                all_rows.append(row)

    if not all_rows:
        print("Нет результатов.")
        return

    df_all = pd.DataFrame(all_rows)

    # ------------------------------------------------------------------
    # Сводка по режимам
    # ------------------------------------------------------------------
    print(f"\n{'='*110}")
    print("СВОДКА ПО РЕЖИМАМ (среднее по монетам, издержки=мейкер 0.02%)")
    print(f"{'='*110}")

    maker = df_all[df_all["cost"] == "maker"]
    summary = maker.groupby("mode").agg({
        "annualized_pct": "mean",
        "max_drawdown_pct": "mean",
        "total_trades": "mean",
        "grid_cycles": "mean",
        "realized_grid_pnl": "mean",
        "flat_time_pct": "mean",
        "regime_switches": "mean",
    }).round(2).reindex(MODES)
    summary.index = [MODE_LABELS.get(m, m) for m in summary.index]
    print(summary.to_string())

    # ------------------------------------------------------------------
    # Детализация по монетам (мейкер)
    # ------------------------------------------------------------------
    print(f"\n{'='*110}")
    print("ДЕТАЛИЗАЦИЯ ПО МОНЕТАМ (мейкер 0.02%)")
    print(f"{'='*110}")

    for mode in MODES:
        sub = maker[maker["mode"] == mode]
        print(f"\n  --- {MODE_LABELS[mode]} ---")
        cols = ["symbol", "annualized_pct", "max_drawdown_pct",
                "total_trades", "grid_cycles", "realized_grid_pnl",
                "flat_time_pct", "regime_switches"]
        print(sub[cols].to_string(index=False))

    # ------------------------------------------------------------------
    # Вердикт
    # ------------------------------------------------------------------
    print(f"\n{'='*110}")
    print("ВЕРДИКТ")
    print(f"{'='*110}")

    naive = maker[maker["mode"] == "naive_grid"]
    flat = maker[maker["mode"] == "flat_only"]
    hybrid = maker[maker["mode"] == "hybrid"]

    for name, grp in [("Наивная сетка", naive),
                      ("Сетка только во флэте", flat),
                      ("Гибрид", hybrid)]:
        if grp.empty:
            continue
        avg_ret = grp["annualized_pct"].mean()
        avg_dd = grp["max_drawdown_pct"].mean()
        avg_pnl = grp["realized_grid_pnl"].mean()
        frac_pos = (grp["annualized_pct"] > 0).mean()
        frac_dd_ok = (grp["max_drawdown_pct"] < 20.0).mean()
        print(f"\n  {name}:")
        print(f"    Средний годовой возврат:  {avg_ret:+.2f}%")
        print(f"    Средняя макс. просадка:    {avg_dd:.1f}%")
        print(f"    Средний realized PnL сетки: {avg_pnl:+.2f} USDT")
        print(f"    Доля монет с + возвратом: {frac_pos*100:.0f}%")
        print(f"    Доля монет с ДД < 20%:    {frac_dd_ok*100:.0f}%")

    print(f"\n{'='*110}")
    print("КАК ЧИТАТЬ РЕЗУЛЬТАТ:")
    print("  realized_grid_pnl — чистая прибыль сетки от осцилляций (без учёта")
    print("  нереализованных убытков инвентаря). Если он положительный —")
    print("  mean-reversion в данных ЕСТЬ, и сетка на нём зарабатывает.")
    print("  annualized_pct — полный результат с учётом всех убытков.")
    print(f"{'='*110}")


if __name__ == "__main__":
    main()