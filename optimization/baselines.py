"""Базовая линия Фазы 0: что даёт сам рынок за тот же период."""
from __future__ import annotations
from typing import Dict


def buy_and_hold(df) -> Dict:
    close = df["close"]
    ret = (float(close.iloc[-1]) / float(close.iloc[0]) - 1.0) * 100.0
    eq = close / close.iloc[0]
    dd = float(((eq / eq.cummax()) - 1.0).min()) * 100.0
    return {
        "name": "BUY&HOLD",
        "return_pct": round(ret, 2),
        "max_drawdown_pct": round(abs(dd), 2),
        "trades": 1,
        "winrate": 0.0,
        "expectancy_pct": 0.0,
        "profit_factor": 0.0,
    }