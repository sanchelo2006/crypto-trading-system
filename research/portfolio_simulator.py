"""
Портфельный симулятор для исследовательской лаборатории.

Строит кривые доходности по монетам и портфелю в целом с учётом:
- распределения долей капитала между монетами;
- плеча и размера позиции по каждой монете;
- комиссий и проскальзывания;
- выбранных стратегий (входов) и параметров выходов.

Модель (упрощения первой версии):
- На каждой монете сделки исполняются последовательно (одна позиция).
- Если выбрано несколько стратегий, их маски объединяются по ИЛИ
  (аналог entry_any в реальном спеке) отдельно для лонга и шорта.
- Кривая портфеля = взвешенная сумма кривых монет (без ребалансировки).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from data_loader import get_available_symbols, get_available_years, load_local_data
import research.impulse_lab as lab


# ----------------------------------------------------------------
# Реестр стратегий (входов).
# Каждая стратегия: направление + функция маски от лабораторного фрейма `d`.
# Маски строятся на колонках, которые создаёт lab.add_features.
# ----------------------------------------------------------------
STRATEGY_REGISTRY: Dict[str, Dict] = {
    "mom2_short_t2.0": {
        "desc": "Чистый моментум-шорт: падение > 2 ATR за 2 свечи",
        "direction": -1,
        "mask": lambda d: d["vel_2"].to_numpy() < -2.0,
    },
    "m2s2_vol1.5": {
        "desc": "Моментум-шорт + объём > 1.5x",
        "direction": -1,
        "mask": lambda d: (d["vel_2"].to_numpy() < -2.0) & (d["vol_ratio"].to_numpy() > 1.5),
    },
    "m2s2_vol2.0": {
        "desc": "Моментум-шорт + объём > 2.0x (лидер лаборатории)",
        "direction": -1,
        "mask": lambda d: (d["vel_2"].to_numpy() < -2.0) & (d["vol_ratio"].to_numpy() > 2.0),
    },
    "act_short": {
        "desc": "Аномальная активность > 3.0 + падение > 1 ATR",
        "direction": -1,
        "mask": lambda d: (d["activity"].to_numpy() > 3.0) & (d["vel_2"].to_numpy() < -1.0),
    },
    "act_long": {
        "desc": "Аномальная активность > 3.0 + рост > 1 ATR",
        "direction": 1,
        "mask": lambda d: (d["activity"].to_numpy() > 3.0) & (d["vel_2"].to_numpy() > 1.0),
    },
    "cont_long_t2.0": {
        "desc": "Лонг-моментум: рост > 2 ATR за 2 свечи",
        "direction": 1,
        "mask": lambda d: d["vel_2"].to_numpy() > 2.0,
    },
    "climax_bear": {
        "desc": "Кульминационная медвежья свеча (диапазон + объём + закрытие у лоя)",
        "direction": -1,
        "mask": lambda d: (d["range_atr"].to_numpy() > 1.5) & (d["vol_ratio"].to_numpy() > 2.0) & (d["close_position"].to_numpy() < 0.2),
    },
    "climax_bull": {
        "desc": "Кульминационная бычья свеча (диапазон + объём + закрытие у хая)",
        "direction": 1,
        "mask": lambda d: (d["range_atr"].to_numpy() > 1.5) & (d["vol_ratio"].to_numpy() > 2.0) & (d["close_position"].to_numpy() > 0.8),
    },
}


# ----------------------------------------------------------------
# Конфигурация
# ----------------------------------------------------------------
@dataclass
class SymbolConfig:
    symbol: str
    capital_share_pct: float = 33.0     # доля от общего капитала
    position_size_pct: float = 50.0     # размер позиции в % от капитала монеты
    leverage: float = 1.0               # плечо


@dataclass
class SimConfig:
    initial_capital: float = 1000.0
    fee_pct: float = 0.11               # round-trip комиссия
    slippage_pct: float = 0.0           # round-trip проскальзывание
    sl_atr: float = 2.0                 # стоп в ATR
    tp_atr: float = 1.5                 # тейк в ATR
    max_hold: int = 8                   # макс. время удержания (свечей)
    reinvest: bool = True               # компаундинг прибыли
    symbols: List[SymbolConfig] = field(default_factory=list)
    strategies: List[str] = field(default_factory=lambda: ["m2s2_vol2.0"])


# ----------------------------------------------------------------
# Кэш подготовленных фреймов (индикаторы дороги в расчёте)
# ----------------------------------------------------------------
_PREPARED_CACHE: Dict[str, pd.DataFrame] = {}


def prepare_symbol(symbol: str, engine) -> Optional[pd.DataFrame]:
    """Загружает все годы монеты, считает индикаторы, кэширует."""
    if symbol in _PREPARED_CACHE:
        return _PREPARED_CACHE[symbol]

    frames = []
    for year in get_available_years(symbol):
        try:
            frames.append(load_local_data(symbol, year))
        except FileNotFoundError:
            continue
    if not frames:
        return None

    full = pd.concat(frames, ignore_index=True)
    full["timestamp"] = pd.to_datetime(full["timestamp"], utc=True, errors="coerce")
    full = (
        full.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"])
        .reset_index(drop=True)
    )
    if len(full) < 5000:
        return None

    df = engine._build_indicator_frame(full)
    d = lab.add_features(df)
    _PREPARED_CACHE[symbol] = d
    return d


def clear_cache():
    _PREPARED_CACHE.clear()


# ----------------------------------------------------------------
# Симуляция сделок (с датами и ценами)
# ----------------------------------------------------------------
def simulate_symbol(d: pd.DataFrame, signal_specs: List[tuple],
                    sl_atr: float, tp_atr: float, max_hold: int,
                    cost: float) -> tuple:
    """Последовательная симуляция одной монеты: максимум одна позиция.

    signal_specs: [(имя_стратегии, mask, direction), ...] в порядке приоритета.
    Сигналы сливаются в один поток по времени. Пока позиция открыта,
    все новые сигналы отбрасываются — как в backtest_engine._simulate.

    Возвращает (trades, stats).
    """
    close = d["close"].to_numpy(float)
    high = d["high"].to_numpy(float)
    low = d["low"].to_numpy(float)
    opn = d["open"].to_numpy(float)
    atr = d["atr_14"].to_numpy(float)
    ts = pd.to_datetime(d["timestamp"])
    n = len(close)

    # Один поток: (бар сигнала, приоритет, направление, имя стратегии)
    candidates = []
    raw_hits = {}
    for priority, (name, mask, direction) in enumerate(signal_specs):
        idx = np.flatnonzero(np.asarray(mask, dtype=bool))
        raw_hits[name] = int(len(idx))
        for s in idx:
            candidates.append((int(s), priority, direction, name))
    candidates.sort(key=lambda x: (x[0], x[1]))

    trades = []
    taken_by_strategy = {name: 0 for name, _, _ in signal_specs}
    skipped_overlap = 0
    skipped_warmup = 0
    last_exit_idx = -1

    for s, priority, direction, sname in candidates:
        if s + 1 >= n or s + max_hold >= n:
            skipped_warmup += 1
            continue

        # КЛЮЧЕВОЕ: сигнал на баре выхода или раньше — отбрасываем.
        # В живом движке после закрытия позиции стоит continue,
        # поэтому новый вход возможен только на следующем баре.
        if s <= last_exit_idx:
            skipped_overlap += 1
            continue

        # ATR намеренно оставлен на сигнальной свече, как в прежней
        # версии: не смешиваем два изменения в одном прогоне.
        a = atr[s]
        if not np.isfinite(a) or a <= 0:
            skipped_warmup += 1
            continue

        entry = opn[s + 1]
        sl = entry - direction * sl_atr * a
        tp = entry + direction * tp_atr * a
        exit_idx = min(s + max_hold, n - 1)
        res, exit_price, reason = None, None, "time"

        for k in range(s + 1, min(s + 1 + max_hold, n)):
            lk, hk = low[k], high[k]
            if direction == 1:
                if lk <= sl:
                    res, exit_idx, exit_price, reason = (sl - entry) / entry * 100 - cost, k, sl, "sl"
                    break
                if hk >= tp:
                    res, exit_idx, exit_price, reason = (tp - entry) / entry * 100 - cost, k, tp, "tp"
                    break
            else:
                if hk >= sl:
                    res, exit_idx, exit_price, reason = (entry - sl) / entry * 100 - cost, k, sl, "sl"
                    break
                if lk <= tp:
                    res, exit_idx, exit_price, reason = (entry - tp) / entry * 100 - cost, k, tp, "tp"
                    break

        if exit_price is None:
            exit_idx = min(s + max_hold, n - 1)
            exit_price = close[exit_idx]
            res = (exit_price - entry) / entry * 100 * direction - cost

        last_exit_idx = exit_idx
        taken_by_strategy[sname] += 1
        trades.append({
            "entry_time": ts[s + 1],
            "exit_time": ts[exit_idx],
            "direction": direction,
            "entry_price": float(entry),
            "exit_price": float(exit_price),
            "result_pct": float(res),
            "bars_held": int(exit_idx - s),
            "exit_reason": reason,
            "strategy": sname,
        })

    stats = {
        "signals_total": len(candidates),
        "raw_hits": raw_hits,
        "taken_by_strategy": taken_by_strategy,
        "skipped_overlap": skipped_overlap,
        "skipped_warmup": skipped_warmup,
    }
    return trades, stats


# ----------------------------------------------------------------
# Кривые капитала
# ----------------------------------------------------------------
def build_symbol_curve(trades: List[Dict], allocated: float,
                       position_size_pct: float, leverage: float,
                       reinvest: bool = True):
    """Кривая капитала монеты. Капитал не может стать отрицательным."""
    if not trades:
        return [], allocated
    pos_frac = position_size_pct / 100.0
    capital = allocated
    curve = [(min(t["entry_time"] for t in trades), capital)]

    for t in sorted(trades, key=lambda x: x["exit_time"]):
        if capital <= 0:
            break  # банкрот: торговать больше нечем
        trade_return_frac = t["result_pct"] / 100.0 * pos_frac * leverage
        base = capital if reinvest else allocated
        capital = max(0.0, capital + base * trade_return_frac)
        curve.append((t["exit_time"], capital))

    return curve, capital


def build_portfolio_curve(curves_by_symbol: Dict[str, List],
                          initial_by_symbol: Dict[str, float]):
    """Общая кривая портфеля = сумма текущих капиталов монет."""
    current = dict(initial_by_symbol)
    events = []
    for sym, curve in curves_by_symbol.items():
        for (t, cap) in curve:
            events.append((t, sym, cap))
    events.sort(key=lambda e: e[0])

    portfolio_curve = []
    if events:
        portfolio_curve.append((events[0][0], sum(initial_by_symbol.values())))
    for (t, sym, cap) in events:
        current[sym] = cap
        portfolio_curve.append((t, sum(current.values())))
    return portfolio_curve


# ----------------------------------------------------------------
# Главная функция симуляции
# ----------------------------------------------------------------
def run_simulation(engine, cfg: SimConfig) -> Dict:
    """Портфельная симуляция: одна позиция на монету, без перекрытий."""
    cost = cfg.fee_pct + cfg.slippage_pct
    result = {"symbols": {}, "portfolio_curve": [], "errors": [],
              "unallocated_cash": 0.0}
    curves = {}
    initials = {}

    # Приоритет как в живом движке: сначала LONG, потом SHORT.
    ordered = (
        [s for s in cfg.strategies
         if STRATEGY_REGISTRY.get(s, {}).get("direction") == 1]
        + [s for s in cfg.strategies
           if STRATEGY_REGISTRY.get(s, {}).get("direction") == -1]
    )

    for scfg in cfg.symbols:
        symbol = scfg.symbol
        d = prepare_symbol(symbol, engine)
        if d is None:
            result["errors"].append(f"{symbol}: недостаточно данных")
            continue

        allocated = cfg.initial_capital * scfg.capital_share_pct / 100.0
        initials[symbol] = allocated

        signal_specs = []
        for sname in ordered:
            sdef = STRATEGY_REGISTRY.get(sname)
            if sdef:
                signal_specs.append((sname, sdef["mask"](d), sdef["direction"]))

        trades, stats = simulate_symbol(
            d, signal_specs, cfg.sl_atr, cfg.tp_atr, cfg.max_hold, cost
        )

        curve, final_capital = build_symbol_curve(
            trades, allocated, scfg.position_size_pct, scfg.leverage, cfg.reinvest
        )
        curves[symbol] = curve

        # Max Drawdown по кривой капитала монеты
        peak, mdd = allocated, 0.0
        for _, cap in curve:
            peak = max(peak, cap)
            if peak > 0:
                mdd = max(mdd, (peak - cap) / peak * 100.0)

        exps = [t["result_pct"] for t in trades]
        wins = [e for e in exps if e > 0]
        result["symbols"][symbol] = {
            "trades": len(trades),
            "win_rate": round(100 * len(wins) / len(trades), 1) if trades else 0.0,
            "exp_mean": round(float(np.mean(exps)), 4) if exps else 0.0,
            "final_capital": round(final_capital, 2),
            "pnl": round(final_capital - allocated, 2),
            "pnl_pct": round((final_capital / allocated - 1) * 100, 2) if allocated else 0.0,
            "max_drawdown_pct": round(mdd, 2),
            "bankrupt": bool(trades and final_capital <= 0),
            "effective_leverage": round(scfg.position_size_pct / 100.0 * scfg.leverage, 2),
            "curve": curve,
            "trades_detail": trades,
            "stats": stats,
        }

    # Незанятый кэш участвует в капитале портфеля, но не торгуется
    unallocated = max(0.0, cfg.initial_capital - sum(initials.values()))
    result["unallocated_cash"] = round(unallocated, 2)

    port_curve = build_portfolio_curve(curves, initials)
    result["portfolio_curve"] = [(t, cap + unallocated) for t, cap in port_curve]
    return result

# ----------------------------------------------------------------
# Сохранение подробного текстового отчёта
# ----------------------------------------------------------------
LAB_RUNS_DIR = Path(__file__).resolve().parent / "lab_runs"


def save_simulation_report(result: Dict, cfg: SimConfig) -> Path:
    """Сохраняет подробный текстовый отчёт симуляции.

    Формат — структурированный текст с таблицами, удобный для
    последующего анализа. Возвращает путь к созданному файлу.
    """
    LAB_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = LAB_RUNS_DIR / f"lab_{stamp}.txt"

    lines: List[str] = []
    w = lines.append
    sep = "=" * 90
    sub = "-" * 90

    symbols = result.get("symbols", {})
    errors = result.get("errors", [])

    # Собираем все сделки для агрегаций
    all_trades: List[Dict] = []
    for sym, s in symbols.items():
        for t in s.get("trades_detail", []):
            t2 = dict(t)
            t2["symbol"] = sym
            all_trades.append(t2)

    total_final = sum(s["final_capital"] for s in symbols.values()) \
        + result.get("unallocated_cash", 0.0)
    total_pnl = total_final - cfg.initial_capital
    total_pnl_pct = (total_final / cfg.initial_capital - 1) * 100 if cfg.initial_capital else 0.0
    all_exps = [t["result_pct"] for t in all_trades]

    # === Шапка ===
    w(sep)
    w("ЛАБОРАТОРНАЯ СИМУЛЯЦИЯ ПОРТФЕЛЯ")
    w(sep)
    w(f"Время запуска (UTC):   {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    w("")
    w("ПАРАМЕТРЫ ПОРТФЕЛЯ:")
    w(f"  Начальный капитал:     ${cfg.initial_capital:,.2f}")
    w(f"  Комиссия (round-trip): {cfg.fee_pct}%")
    w(f"  Проскальзывание:       {cfg.slippage_pct}%")
    w(f"  Выходы:                SL={cfg.sl_atr} ATR, TP={cfg.tp_atr} ATR, MaxHold={cfg.max_hold} свечей")
    w(f"  Реинвестирование:      {'вкл' if cfg.reinvest else 'выкл'}")
    w(f"  Стратегии:             {', '.join(cfg.strategies)}")
    w("")

    # === Конфигурация монет ===
    w(sub)
    w("КОНФИГУРАЦИЯ МОНЕТ")
    w(sub)
    w(f"{'Монета':<12} | {'Доля %':>8} | {'Позиция %':>10} | {'Плечо':>6}")
    w(sub)
    for scfg in cfg.symbols:
        w(f"{scfg.symbol:<12} | {scfg.capital_share_pct:>8.1f} | "
          f"{scfg.position_size_pct:>10.1f} | {scfg.leverage:>6.1f}")
    w("")

    # === Итоги портфеля ===
    w(sub)
    w("ИТОГИ ПОРТФЕЛЯ")
    w(sub)
    w(f"  Начальный капитал:   ${cfg.initial_capital:,.2f}")
    w(f"  Конечный капитал:    ${total_final:,.2f}")
    w(f"  Общий P&L:           {total_pnl:+,.2f}$ ({total_pnl_pct:+.2f}%)")
    w(f"  Всего сделок:        {len(all_trades)}")
    if all_exps:
        wins = [e for e in all_exps if e > 0]
        w(f"  Средний exp_mean:    {np.mean(all_exps):+.4f}% (по всем сделкам)")
        w(f"  Общий Win Rate:      {100*len(wins)/len(all_exps):.1f}%")
        w(f"  Лучшая сделка:       {max(all_exps):+.4f}%")
        w(f"  Худшая сделка:       {min(all_exps):+.4f}%")
    w("")

    # === Диагностика реализма ===
    w(sub)
    w("ДИАГНОСТИКА РЕАЛИЗМА")
    w(sub)
    w(f"  Издержки на сделку: {cfg.fee_pct + cfg.slippage_pct:.3f}% "
      f"(комиссия {cfg.fee_pct}% + слиппедж {cfg.slippage_pct}%)")
    w("  ВНИМАНИЕ: Bybit taker 0.10%/сторона = 0.20% round-trip.")
    w(f"  Незанятый кэш: ${result.get('unallocated_cash', 0.0):,.2f}")
    w("")
    for sym, s in symbols.items():
        st = s.get("stats", {})
        w(f"  {sym}: сигналов={st.get('signals_total', 0)}, "
          f"перекрываний отброшено={st.get('skipped_overlap', 0)}, "
          f"прогретв отброшено={st.get('skipped_warmup', 0)}, "
          f"сделок={s['trades']}")
        w(f"    MDD={s.get('max_drawdown_pct', 0):.2f}%, "
          f"эффективное плечо={s.get('effective_leverage', 0):.2f}x, "
          f"банкротство={'ДА' if s.get('bankrupt') else 'нет'}")
        w(f"    сырые срабатывания по стратегиям: {st.get('raw_hits', {})}")
        w(f"    принятые сделки по стратегиям: {st.get('taken_by_strategy', {})})".replace(")", "", 1) if False else
          f"    принятые сделки по стратегиям: {st.get('taken_by_strategy', {})}")
    w("")

    # === Результаты по монетам ===
    w(sub)
    w("РЕЗУЛЬТАТЫ ПО МОНЕТАМ")
    w(sub)
    w(f"{'Монета':<12} | {'Сделок':>6} | {'WinRate':>8} | {'exp_mean':>10} | "
      f"{'Кон.капитал':>13} | {'P&L':>12} | {'P&L %':>8}")
    w(sub)
    for sym, s in symbols.items():
        w(f"{sym:<12} | {s['trades']:>6} | {s['win_rate']:>7.1f}% | "
          f"{s['exp_mean']:>+9.4f}% | ${s['final_capital']:>12,.2f} | "
          f"{s['pnl']:>+11,.2f}$ | {s['pnl_pct']:>+7.2f}%")
    w("")

    # === Разбивка по годам (портфель) ===
    by_year: Dict[int, List[float]] = {}
    for t in all_trades:
        y = t["exit_time"].year
        by_year.setdefault(y, []).append(t["result_pct"])

    w(sub)
    w("РАЗБИВКА ПО ГОДАМ (весь портфель)")
    w(sub)
    w(f"{'Год':>6} | {'Сделок':>6} | {'Средний exp':>12} | {'Сумма рез-тов':>14}")
    w(sub)
    for year in sorted(by_year.keys()):
        exps = by_year[year]
        w(f"{year:>6} | {len(exps):>6} | {np.mean(exps):>+11.4f}% | {sum(exps):>+13.2f}%")
    w("")

    # === Разбивка по монетам и годам ===
    by_sym_year: Dict[tuple, List[float]] = {}
    for t in all_trades:
        key = (t["symbol"], t["exit_time"].year)
        by_sym_year.setdefault(key, []).append(t["result_pct"])

    w(sub)
    w("РАЗБИВКА ПО МОНЕТАМ И ГОДАМ")
    w(sub)
    w(f"{'Монета':<12} | {'Год':>6} | {'Сделок':>6} | {'Средний exp':>12} | {'Сумма':>12}")
    w(sub)
    for (sym, year) in sorted(by_sym_year.keys()):
        exps = by_sym_year[(sym, year)]
        w(f"{sym:<12} | {year:>6} | {len(exps):>6} | {np.mean(exps):>+11.4f}% | {sum(exps):>+11.2f}%")
    w("")

    # === Статистика причин выхода ===
    w(sub)
    w("СТАТИСТИКА ПРИЧИН ВЫХОДА")
    w(sub)
    w(f"{'Монета':<12} | {'Стоп-лосс':>10} | {'Тейк':>10} | {'Тайм-стоп':>10}")
    w(sub)
    for sym, s in symbols.items():
        reasons = [t["exit_reason"] for t in s.get("trades_detail", [])]
        sl = reasons.count("sl")
        tp = reasons.count("tp")
        tm = reasons.count("time")
        w(f"{sym:<12} | {sl:>10} | {tp:>10} | {tm:>10}")
    w("")

    # === Ошибки/предупреждения ===
    if errors:
        w(sub)
        w("ПРЕДУПРЕЖДЕНИЯ")
        w(sub)
        for e in errors:
            w(f"  ⚠ {e}")
        w("")

    # === Полный список сделок ===
    w(sub)
    w("СПИСОК СДЕЛОК (компактно)")
    w(sub)
    w(f"{'Монета':<12} | {'Стратегия':<18} | {'Вход (время)':<17} | {'Выход (время)':<17} | {'Напр.':>5} | "
      f"{'Цена входа':>12} | {'Цена выхода':>12} | {'Рез-т':>8} | {'Баров':>5} | {'Причина':>8}")
    w(sub)
    for t in sorted(all_trades, key=lambda x: (x["symbol"], x["exit_time"])):
        dir_str = "LONG" if t["direction"] == 1 else "SHORT"
        w(f"{t['symbol']:<12} | {t.get('strategy', ''):<18} | {t['entry_time'].strftime('%Y-%m-%d %H:%M'):<17} | "
          f"{t['exit_time'].strftime('%Y-%m-%d %H:%M'):<17} | {dir_str:>5} | "
          f"{t['entry_price']:>12.6f} | {t['exit_price']:>12.6f} | "
          f"{t['result_pct']:>+7.3f}% | {t['bars_held']:>5} | {t['exit_reason']:>8}")

    # Запись файла
    path.write_text("\n".join(lines), encoding="utf-8")
    return path