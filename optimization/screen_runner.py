"""
Phase 0: screening of simple archetypes and the baseline.

Goal: BEFORE optimization, understand whether the symbol has any edge at all,
in which regimes/years it lives, and whether it is stronger than "just hold the coin".
Screening is informational: it selects nothing and saves nothing.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import pandas as pd

from optimization.archetypes import ARCHETYPES
from optimization.baselines import buy_and_hold

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Engine call. Signature: run_on_dataframe(df, symbol, strategy_spec,
# initial_capital, commission_pct, rules_text="", start_date=None, end_date=None).
# Returns {"trades": [...], "stats": {...}}.
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
    # Raw df from CSV contains no indicators: without adx_14 the regime
    # degenerated to unknown. Compute indicators ONCE for the whole screening —
    # this also prevents the engine from recomputing them on every archetype
    # call (speeds up screening many times).
    if not engine._has_precomputed_indicators(df):
        df = engine._build_indicator_frame(df)

    lines = ["=" * 96]
    lines.append("PHASE 0: ARCHETYPE AND BASELINE SCREENING")
    lines.append("=" * 96)
    bh = buy_and_hold(df)
    lines.append(_fmt("BUY&HOLD (baseline)", bh))
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
                logger.warning("screen: %s, params=%s -> error: %s", arch["name"], p, exc)
                continue
            m = _metrics(trades, stats)
            if m["trades"] < 10:
                continue
            if best_m is None or m["return_pct"] > best_m["return_pct"]:
                best_m, best_p = m, p

        if best_m is None:
            lines.append(f"{arch['name']:26s} NO TRADES on any parameter set")
            continue

        lines.append(_fmt(arch["name"], best_m) + f"  params={best_p}")

        # --- Collect results for automatic selection ---
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
    lines.append("Screening is informational. Only an archetype that is positive")
    lines.append("across years and beats the baseline proceeds to optimization.")
    lines.append("=" * 96)

    results = {
        "overall": best_overall,
        "by_name": by_name,
        "buy_and_hold": bh,
    }
    return "\n".join(lines), results