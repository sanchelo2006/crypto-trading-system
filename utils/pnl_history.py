"""
Aggregation of historical monthly P&L from auto-trading trade logs.

Data source: CSV files data/auto_trading/logs/{SYMBOL}_trades_YYYY_MM.csv
maintained by TradeLogger. They contain the pnl_usdt and result_pct columns
for every closed trade. No separate history file is needed.

The module is read-only: it writes nothing, only reads and aggregates.

Capital at month end (option a): current_capital from risk_state.json is
the capital NOW (at the end of the latest month with data). We walk backwards,
subtracting the P&L of later months. If there were deposits or withdrawals,
the value is approximate.
"""
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import config

logger = logging.getLogger(__name__)

LOGS_DIR = config.DATA_DIR / "auto_trading" / "logs"
RISK_STATE_FILE = config.DATA_DIR / "auto_trading" / "risk_state.json"


def _iter_trade_csvs(symbol: str) -> List[Path]:
    """List of the coin's trade CSV files, sorted by name (= month)."""
    if not LOGS_DIR.exists():
        return []
    pattern = f"{symbol}_trades_*.csv"
    return sorted(LOGS_DIR.glob(pattern))


def _month_from_filename(path: Path, symbol: str) -> Optional[str]:
    """Extracts YYYY-MM from the name {SYMBOL}_trades_YYYY_MM.csv."""
    stem = path.stem
    prefix = f"{symbol}_trades_"
    if not stem.startswith(prefix):
        return None
    suffix = stem[len(prefix):]          # "2026_10"
    parts = suffix.split("_")
    if len(parts) != 2:
        return None
    return f"{parts[0]}-{parts[1]}"


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_month_csv(path: Path) -> Dict:
    """Reads one monthly CSV. Returns {"trades": int, "pnl_usdt": float}."""
    trades = 0
    pnl_sum = 0.0
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                trades += 1
                pnl_sum += _safe_float(row.get("pnl_usdt", 0.0))
    except Exception:
        logger.exception(f"Failed to read trade log {path}")
        return {"trades": 0, "pnl_usdt": 0.0}
    return {"trades": trades, "pnl_usdt": pnl_sum}


def _load_current_capital(symbol: str) -> Optional[float]:
    """The coin's current_capital from risk_state.json (None if absent)."""
    if not RISK_STATE_FILE.exists():
        return None
    try:
        with open(RISK_STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
        entry = state.get(symbol)
        if not isinstance(entry, dict):
            return None
        cap = entry.get("current_capital")
        return float(cap) if cap is not None else None
    except Exception:
        logger.exception(f"Failed to read risk_state for {symbol}")
        return None


def get_monthly_pnl(symbol: str) -> List[Dict]:
    """
    Monthly records for one coin, sorted by month ascending:
    [{"month": "2026-09", "pnl_usdt": -1.23, "trades": 5, "capital_end": 24.5}, ...]
    capital_end = None if there is no capital data.
    """
    files = _iter_trade_csvs(symbol)
    if not files:
        return []

    months: Dict[str, Dict] = {}
    for path in files:
        month = _month_from_filename(path, symbol)
        if month is None:
            continue
        agg = _read_month_csv(path)
        months[month] = {
            "month": month,
            "pnl_usdt": round(agg["pnl_usdt"], 2),
            "trades": agg["trades"],
        }

    if not months:
        return []

    result = [months[m] for m in sorted(months.keys())]

    current_cap = _load_current_capital(symbol)
    if current_cap is not None:
        cap = current_cap
        for rec in reversed(result):
            rec["capital_end"] = round(cap, 2)
            cap -= rec["pnl_usdt"]
    else:
        for rec in result:
            rec["capital_end"] = None

    return result


def get_all_symbols_monthly(symbols: List[str]) -> List[Dict]:
    """
    Aggregated monthly P&L for a list of coins.
    capital_end is the sum of the coins' capital at month end.
    """
    per_symbol = {s: get_monthly_pnl(s) for s in symbols}

    all_months = set()
    for rows in per_symbol.values():
        for r in rows:
            all_months.add(r["month"])

    if not all_months:
        return []

    result = []
    for month in sorted(all_months):
        total_pnl = 0.0
        total_trades = 0
        total_cap = 0.0
        cap_known = False

        for sym in symbols:
            rows = per_symbol.get(sym, [])
            rec = next((r for r in rows if r["month"] == month), None)
            if rec is None:
                continue
            total_pnl += rec["pnl_usdt"]
            total_trades += rec["trades"]
            if rec["capital_end"] is not None:
                total_cap += rec["capital_end"]
                cap_known = True

        result.append({
            "month": month,
            "pnl_usdt": round(total_pnl, 2),
            "trades": total_trades,
            "capital_end": round(total_cap, 2) if cap_known else None,
        })

    return result