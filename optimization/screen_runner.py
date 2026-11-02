"""
Фаза 0: скрининг простых архетипов и базовой линии.

Цель: ДО оптимизации понять, есть ли на символе вообще эдж,
в каких режимах/годах он живёт и сильнее ли он, чем «просто держать монету».
Скрининг информационный: ничего не отбирает и не сохраняет.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import pandas as pd

from optimization.archetypes import ARCHETYPES
from optimization.baselines import buy_and_hold

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Вызов движка. Сигнатура: run_on_dataframe(df, symbol, strategy_spec,
# initial_capital, commission_pct, rules_text="", start_date=None, end_date=None).
# Возвращает {"trades": [...], "stats": {...}}.
# ------------------------------------------------------------------
def _call_engine(engine, df, symbol: str, spec: Dict, capital: float, commission: float):
    result = engine.run_on_dataframe(
        df=df,
        symbol=symbol,
        strategy_spec=spec,
        initial_capital=capital,
        commission_pct=commission,
    )
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"engine error: {result['error']}")
    return result


def _normalize(res) -> Tuple[Dict, List[Dict]]:
    if isinstance(res, dict):
        return res.get("stats", {}) or {}, res.get("trades", []) or []
    if isinstance(res, tuple):
        stats, trades = res
        return (stats or {}), (trades or [])
    return {}, []


def _metrics(trades: List[Dict], stats: Dict) -> Dict:
    res = [float(t.get("result_pct", 0.0)) for t in trades]
    n = len(res)
    wins = [r for r in res if r > 0]
    losses = [r for r in res if r <= 0]
    gw, gl = sum(wins), abs(sum(losses))
    pf = (gw / gl) if gl > 0 else (999.0 if gw > 0 else 0.0)
    return {
        "trades": n,
        "winrate": round(100.0 * len(wins) / n, 1) if n else 0.0,
        "return_pct": round(float(stats.get("return_pct", sum(res))), 2),
        "expectancy_pct": round(sum(res) / n, 3) if n else 0.0,
        "profit_factor": round(pf, 2),
        "max_drawdown_pct": round(float(stats.get("max_drawdown_pct", 0.0)), 2),
    }


def _agg(pairs: List[Tuple[str, float]]) -> Dict:
    n = len(pairs)
    if not n:
        return {"n": 0, "sum_pct": 0.0, "pf": 0.0}
    s = sum(p for _, p in pairs)
    gw = sum(p for _, p in pairs if p > 0)
    gl = abs(sum(p for _, p in pairs if p <= 0))
    pf = (gw / gl) if gl > 0 else (999.0 if gw > 0 else 0.0)
    return {"n": n, "sum_pct": round(s, 2), "pf": round(pf, 2)}


def _fmt(name: str, m: Dict) -> str:
    return (
        f"{name:26s} trades={m['trades']:4d} WR={m['winrate']:5.1f}% "
        f"Ret={m['return_pct']:+8.2f}% Exp={m['expectancy_pct']:+7.3f}% "
        f"PF={m['profit_factor']:6.2f} DD={m['max_drawdown_pct']:6.2f}%"
    )


def run_screen(engine, df, symbol: str, commission_pct: float, initial_capital: float) -> Tuple[str, Dict]:
    # Сырой df из CSV не содержит индикаторов: без adx_14 режим
    # вырождался в unknown. Считаем индикаторы ОДИН раз на весь
    # скрининг — заодно движок не будет пересчитывать их при каждом
    # вызове архетипа (ускорение скрининга в разы).
    if not engine._has_precomputed_indicators(df):
        df = engine._build_indicator_frame(df)

    lines = ["=" * 96]
    lines.append("ФАЗА 0: СКРИНИНГ АРХЕТИПОВ И БАЗОВОЙ ЛИНИИ")
    lines.append("=" * 96)
    bh = buy_and_hold(df)
    lines.append(_fmt("BUY&HOLD (база)", bh))
    lines.append("-" * 96)

    best_overall = None
    by_name: Dict[str, Dict] = {}

    for arch in ARCHETYPES:
        best_m, best_p = None, None
        for p in arch["grid"]():
            try:
                stats, trades = _normalize(
                    _call_engine(engine, df, symbol, arch["build"](p), initial_capital, commission_pct)
                )
            except Exception as exc:
                logger.warning("screen: %s, params=%s -> ошибка: %s", arch["name"], p, exc)
                continue
            m = _metrics(trades, stats)
            if m["trades"] < 10:
                continue
            if best_m is None or m["return_pct"] > best_m["return_pct"]:
                best_m, best_p = m, p

        if best_m is None:
            lines.append(f"{arch['name']:26s} НЕТ СДЕЛОК ни на одном наборе параметров")
            continue

        lines.append(_fmt(arch["name"], best_m) + f"  params={best_p}")
        

        # --- Сбор результатов для автоматического выбора ---
        candidate = {
            "name": arch["name"],
            "spec": arch["build"](best_p),
            "params": best_p,
            "return_pct": best_m["return_pct"],
            "pf": best_m["profit_factor"],
            "trades": best_m["trades"],
        }

        by_name[arch["name"]] = candidate

        if best_overall is None or best_m["return_pct"] > best_overall["return_pct"]:
            best_overall = candidate


    lines.append("-" * 96)
    lines.append("Скрининг информационный. В оптимизацию уходит только архетип,")
    lines.append("который положителен по годам и лучше базовой линии.")
    lines.append("=" * 96)

    results = {
        "overall": best_overall,
        "by_name": by_name,
        "buy_and_hold": bh,
    }
    return "\n".join(lines), results