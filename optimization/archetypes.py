"""
Phase 0 archetypes: fixed trading ideas with a small number of parameters.
Each archetype has:
- build(p) -> strategy_spec (dict);
- grid()   -> list of parameter sets for a coarse sweep.
The condition structure is hard-coded: only 2-3 numeric knobs are exposed
(stop, take, one filter). This is an overfitting barrier: the idea cannot be
"rewritten", only risk can be tuned to volatility.

Two idea families:
- A1-A5: impulse/trend (breakouts, pullbacks, momentum);
- R1-R4: mean-reversion (return to the mean inside a range).
Every entry group must pass the strategy_spec barrier:
volume filter + trend filter + level/impulse tie-in.
"""
from __future__ import annotations
from typing import Dict, List


def _mgmt() -> Dict:
    return {
        "move_to_breakeven_at_tp_pct": None,
        "breakeven_offset_pct": 0.0,
        "trailing_atr_mult": None,
        "max_hold_candles": 192,
        "exit_on_level_failure": False,
        "exit_on_trend_loss": False,
        "exit_on_macd_flip": False,
    }


def _side(conds: List[Dict], s: float, t: float) -> Dict:
    return {
        "entry": conds,
        "stop_loss_atr_mult": s,
        "take_profit_atr_mult": t,
        "management": _mgmt(),
    }


def _risk() -> Dict:
    return {
        "enabled": True,
        "max_consecutive_losses": 8,
        "pause_candles_after_losses": 16,
        "max_losses_in_window": 10,
        "loss_window_trades": 20,
    }


# ------------------------------------------------------------------
# A1. 1h trend + pullback to SMA20 on 15m (swing)
# ------------------------------------------------------------------
def _build_pullback(p: Dict) -> Dict:
    s, t, vm = p["stop"], p["tp"], p["vol"]
    return {
        "long": _side([
            {"type": "compare", "left": "close", "op": ">", "right": "sma_50_1h"},
            {"type": "compare", "left": "macd_hist_1h", "op": ">", "right": 0},
            {"type": "compare", "left": "low", "op": "<=", "right": "sma_20"},
            {"type": "compare", "left": "close", "op": ">", "right": "sma_20"},
            {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": vm},
        ], s, t),
        "short": _side([
            {"type": "compare", "left": "close", "op": "<", "right": "sma_50_1h"},
            {"type": "compare", "left": "macd_hist_1h", "op": "<", "right": 0},
            {"type": "compare", "left": "high", "op": ">=", "right": "sma_20"},
            {"type": "compare", "left": "close", "op": "<", "right": "sma_20"},
            {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": vm},
        ], s, t),
        "filters": [],
        "risk_controls": _risk(),
    }


def _grid_pullback() -> List[Dict]:
    return [
        {"stop": s, "tp": t, "vol": v}
        for s in (3.0, 4.5, 6.0)
        for t in (6.0, 9.0, 12.0)
        for v in (1.0, 1.3)
    ]


# ------------------------------------------------------------------
# A2. Volatility breakout (Bollinger) along the 1h trend
# Volume is fixed at 1.2: a breakout without volume is a false breakout.
# ------------------------------------------------------------------
def _build_breakout(p: Dict) -> Dict:
    s, t, ax = p["stop"], p["tp"], p["adx"]
    return {
        "long": _side([
            {"type": "compare", "left": "close", "op": ">", "right": "sma_50_1h"},
            {"type": "compare", "left": "close", "op": ">", "right": "bb_upper"},
            {"type": "compare", "left": "adx_14_1h", "op": ">", "right": ax},
            {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 1.2},
        ], s, t),
        "short": _side([
            {"type": "compare", "left": "close", "op": "<", "right": "sma_50_1h"},
            {"type": "compare", "left": "close", "op": "<", "right": "bb_lower"},
            {"type": "compare", "left": "adx_14_1h", "op": ">", "right": ax},
            {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 1.2},
        ], s, t),
        "filters": [],
        "risk_controls": _risk(),
    }


def _grid_breakout() -> List[Dict]:
    return [
        {"stop": s, "tp": t, "adx": a}
        for s in (3.0, 4.5, 6.0)
        for t in (6.0, 9.0, 12.0)
        for a in (18.0, 22.0)
    ]


# ------------------------------------------------------------------
# A3. Mean-reversion from an RSI extreme INSIDE the 1h trend
# ------------------------------------------------------------------
def _build_meanrev(p: Dict) -> Dict:
    s, t, r = p["stop"], p["tp"], p["rsi"]
    return {
        "long": _side([
            {"type": "compare", "left": "close", "op": ">", "right": "sma_50_1h"},
            {"type": "compare", "left": "rsi_14", "op": "<", "right": r},
            {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 1.0},
        ], s, t),
        "short": _side([
            {"type": "compare", "left": "close", "op": "<", "right": "sma_50_1h"},
            {"type": "compare", "left": "rsi_14", "op": ">", "right": 100.0 - r},
            {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 1.0},
        ], s, t),
        "filters": [],
        "risk_controls": _risk(),
    }


def _grid_meanrev() -> List[Dict]:
    return [
        {"stop": s, "tp": t, "rsi": r}
        for s in (2.5, 3.5, 4.5)
        for t in (4.0, 6.0, 8.0)
        for r in (30.0, 35.0, 40.0)
    ]


# ------------------------------------------------------------------
# A4. Pure 1h trend momentum (as simple as possible)
# ------------------------------------------------------------------
def _build_momentum(p: Dict) -> Dict:
    s, t = p["stop"], p["tp"]
    return {
        "long": _side([
            {"type": "compare", "left": "close", "op": ">", "right": "sma_50_1h"},
            {"type": "compare", "left": "macd_hist_1h", "op": ">", "right": 0},
            {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 1.0},
        ], s, t),
        "short": _side([
            {"type": "compare", "left": "close", "op": "<", "right": "sma_50_1h"},
            {"type": "compare", "left": "macd_hist_1h", "op": "<", "right": 0},
            {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 1.0},
        ], s, t),
        "filters": [],
        "risk_controls": _risk(),
    }


def _grid_momentum() -> List[Dict]:
    return [
        {"stop": s, "tp": t}
        for s in (4.0, 6.0, 8.0)
        for t in (8.0, 12.0, 16.0)
    ]


# ------------------------------------------------------------------
# R1. Mean-reversion from Bollinger bands in a flat market
# Entry — price returning inside the channel after crossing the boundary.
# ------------------------------------------------------------------
def _build_bb_meanrev(p: Dict) -> Dict:
    s, t, ax = p["stop"], p["tp"], p["adx"]
    return {
        "long": _side([
            {"type": "cross", "a": "close", "b": "bb_lower", "dir": "above"},
            {"type": "range", "value": "rsi_14", "min": 15.0, "max": 45.0},
            {"type": "compare", "left": "adx_14", "op": "<", "right": ax},
            {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 0.8},
        ], s, t),
        "short": _side([
            {"type": "cross", "a": "close", "b": "bb_upper", "dir": "below"},
            {"type": "range", "value": "rsi_14", "min": 55.0, "max": 85.0},
            {"type": "compare", "left": "adx_14", "op": "<", "right": ax},
            {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 0.8},
        ], s, t),
        "filters": [],
        "risk_controls": _risk(),
    }


def _grid_bb_meanrev() -> List[Dict]:
    return [
        {"stop": s, "tp": t, "adx": a}
        for s in (2.5, 3.5, 4.5)
        for t in (4.0, 6.0, 8.0)
        for a in (18.0, 22.0, 26.0)
    ]


# ------------------------------------------------------------------
# R2. RSI reversion from an extreme in a flat market (target — vwap)
# ------------------------------------------------------------------
def _build_rsi_reversion(p: Dict) -> Dict:
    s, t, ax, r = p["stop"], p["tp"], p["adx"], p["rsi"]
    return {
        "long": _side([
            {"type": "range", "value": "rsi_14", "min": 10.0, "max": r},
            {"type": "compare", "left": "close", "op": "<", "right": "vwap"},
            {"type": "compare", "left": "adx_14", "op": "<", "right": ax},
            {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 0.8},
        ], s, t),
        "short": _side([
            {"type": "range", "value": "rsi_14", "min": 100.0 - r, "max": 90.0},
            {"type": "compare", "left": "close", "op": ">", "right": "vwap"},
            {"type": "compare", "left": "adx_14", "op": "<", "right": ax},
            {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 0.8},
        ], s, t),
        "filters": [],
        "risk_controls": _risk(),
    }


def _grid_rsi_reversion() -> List[Dict]:
    return [
        {"stop": s, "tp": t, "adx": a, "rsi": r}
        for s in (2.5, 3.5, 4.5)
        for t in (4.0, 6.0, 8.0)
        for a in (18.0, 22.0, 26.0)
        for r in (25.0, 30.0, 35.0)
    ]


# ------------------------------------------------------------------
# R3. Bounce off range boundaries (support_1 / resistance_1) in a flat market
# Entry in a narrow zone near the level, but NOT after the level has been broken.
# ------------------------------------------------------------------
def _build_range_boundaries(p: Dict) -> Dict:
    s, t, ax = p["stop"], p["tp"], p["adx"]
    return {
        "long": _side([
            {"type": "compare", "left": "close", "op": "<=", "right": {"name": "support_1", "mult": 1.005}},
            {"type": "compare", "left": "close", "op": ">=", "right": {"name": "support_1", "mult": 0.995}},
            {"type": "compare", "left": "adx_14", "op": "<", "right": ax},
            {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 0.8},
        ], s, t),
        "short": _side([
            {"type": "compare", "left": "close", "op": ">=", "right": {"name": "resistance_1", "mult": 0.995}},
            {"type": "compare", "left": "close", "op": "<=", "right": {"name": "resistance_1", "mult": 1.005}},
            {"type": "compare", "left": "adx_14", "op": "<", "right": ax},
            {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 0.8},
        ], s, t),
        "filters": [],
        "risk_controls": _risk(),
    }


def _grid_range_boundaries() -> List[Dict]:
    return [
        {"stop": s, "tp": t, "adx": a}
        for s in (2.5, 3.5, 4.5)
        for t in (4.0, 6.0, 8.0)
        for a in (18.0, 22.0, 26.0)
    ]


# ------------------------------------------------------------------
# R4. Reversion to VWAP after an extension in a flat market
# ------------------------------------------------------------------
def _build_vwap_reversion(p: Dict) -> Dict:
    s, t, ax = p["stop"], p["tp"], p["adx"]
    return {
        "long": _side([
            {"type": "compare", "left": "close", "op": "<=", "right": {"name": "vwap", "mult": 0.990}},
            {"type": "compare", "left": "close", "op": ">=", "right": {"name": "vwap", "mult": 0.975}},
            {"type": "compare", "left": "adx_14", "op": "<", "right": ax},
            {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 0.8},
        ], s, t),
        "short": _side([
            {"type": "compare", "left": "close", "op": ">=", "right": {"name": "vwap", "mult": 1.010}},
            {"type": "compare", "left": "close", "op": "<=", "right": {"name": "vwap", "mult": 1.025}},
            {"type": "compare", "left": "adx_14", "op": "<", "right": ax},
            {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 0.8},
        ], s, t),
        "filters": [],
        "risk_controls": _risk(),
    }


def _grid_vwap_reversion() -> List[Dict]:
    return [
        {"stop": s, "tp": t, "adx": a}
        for s in (2.5, 3.5, 4.5)
        for t in (4.0, 6.0, 8.0)
        for a in (18.0, 22.0, 26.0)
    ]


# ------------------------------------------------------------------
# A5. Breakout of the 32-candle extreme with a volume spike and a strong close
# (Based on the successful results of impulse_lab iteration 3, no lookahead)
# ------------------------------------------------------------------
def _build_strong_vol_breakout(p: Dict) -> Dict:
    s, t, vm, h = p["stop"], p["tp"], p["vol"], p["hold"]

    long_side = _side([
        {"type": "compare", "left": "close", "op": ">", "right": "high_32"},
        {"type": "compare", "left": "close_position", "op": ">", "right": 0.6},
        {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": vm},
    ], s, t)
    long_side["management"]["max_hold_candles"] = h

    short_side = _side([
        {"type": "compare", "left": "close", "op": "<", "right": "low_32"},
        {"type": "compare", "left": "close_position", "op": "<", "right": 0.4},
        {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": vm},
    ], s, t)
    short_side["management"]["max_hold_candles"] = h

    return {
        "long": long_side,
        "short": short_side,
        "filters": [],
        "risk_controls": _risk(),
    }


def _grid_strong_vol_breakout() -> List[Dict]:
    # Aligned with impulse_lab (iteration 3): narrow ATR stops,
    # short horizon. 3*3*2*2 = 36 combinations.
    return [
        {"stop": s, "tp": t, "vol": v, "hold": h}
        for s in (0.75, 1.0, 1.5)
        for t in (1.5, 2.25, 3.0)
        for v in (1.2, 1.5, 2.0)
        for h in (8, 16)
    ]


# ------------------------------------------------------------------
# A6. Momentum short: continuation DOWN after a volume crash.
# Hypothesis from impulse_lab iter 5 (m2s2_vol2.0): a drop >2 ATR over
# 2 candles + volume > 2.0*avg_volume_96. Long breakouts and long momentum
# on 15m are unprofitable always and everywhere (empirical conclusion of
# iteration 4), so the archetype is SHORT-ONLY — the "long" key is NOT
# included in the spec. This is not a regime gate: the strategy_spec
# validator allows the absence of one side, and the engine simply does not
# generate signals for an empty block.
# ------------------------------------------------------------------
def _build_a6_short_momentum(p: Dict) -> Dict:
    s, t, vm, h = p["stop"], p["tp"], p["vol"], p["hold"]
    short_side = _side([
        {"type": "compare", "left": "drop_2_atr", "op": "<", "right": -2.0},
        {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": vm},
    ], s, t)
    short_side["management"]["max_hold_candles"] = h
    # Exit laboratory: trailing and breakeven are HARMFUL for this signal
    # (they knock you out on the bounce after the crash). Fixed to null.
    short_side["management"]["trailing_atr_mult"] = None
    short_side["management"]["move_to_breakeven_at_tp_pct"] = None
    return {
        "short": short_side,
        "filters": [],
        "risk_controls": _risk(),
        "frozen_entries": True,
        # Exit corridor around the laboratory optimum (Option 2).
        # The optimizer may move parameters ONLY within these bounds.
        "exit_corridor": {
            "stop_loss_atr_mult": [1.8, 2.2],
            "take_profit_atr_mult": [1.3, 1.7],
            "max_hold_candles": [6, 12],
        },
    }


def _grid_a6_short_momentum() -> List[Dict]:
    # Exits fixed by the exit laboratory (34 datasets + stress test):
    #   stop=2.0 — a wide stop survives the bounce after the crash;
    #   tp=1.5   — a fast take fixes the continuation before the reversal;
    #   hold=8   — intraday (2 hours), no overnight carry.
    # Laboratory optimum: exp=+0.19% at slip=0.05%, frac_pos=0.91.
    # Narrow stops (0.75-1.0) get knocked out by the bounce — DO NOT use.
    return [{"stop": 2.0, "tp": 1.5, "vol": 2.0, "hold": 8}]


# ------------------------------------------------------------------
# Archetype registry. Each builder returns a clean spec without service
# wrappers: the idea structure is hard-coded, only the numeric knobs from
# grid() are exposed.
# ------------------------------------------------------------------
ARCHETYPES = [
    {"name": "A1_TREND_PULLBACK",      "build": _build_pullback,             "grid": _grid_pullback},
    {"name": "A2_VOL_BREAKOUT",        "build": _build_breakout,             "grid": _grid_breakout},
    {"name": "A3_MEAN_REVERSION",      "build": _build_meanrev,              "grid": _grid_meanrev},
    {"name": "A4_TREND_MOMENTUM",      "build": _build_momentum,             "grid": _grid_momentum},
    {"name": "A5_STRONG_VOL_BREAKOUT", "build": _build_strong_vol_breakout,  "grid": _grid_strong_vol_breakout},
    {"name": "A6_SHORT_MOMENTUM",      "build": _build_a6_short_momentum,    "grid": _grid_a6_short_momentum},
    {"name": "R1_BB_MEAN_REVERSION",   "build": _build_bb_meanrev,           "grid": _grid_bb_meanrev},
    {"name": "R2_RSI_REVERSION",       "build": _build_rsi_reversion,        "grid": _grid_rsi_reversion},
    {"name": "R3_RANGE_BOUNDARIES",    "build": _build_range_boundaries,     "grid": _grid_range_boundaries},
    {"name": "R4_VWAP_REVERSION",      "build": _build_vwap_reversion,       "grid": _grid_vwap_reversion},
]