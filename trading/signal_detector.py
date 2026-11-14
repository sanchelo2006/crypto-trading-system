"""
Detection of trading signals from strategy_spec.
Reads current indicators and market data, checks entry conditions
using an algorithm identical to backtest_engine.py.
"""
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

from strategy_spec import evaluate_conditions, explain_conditions

logger = logging.getLogger(__name__)


class SignalDetector:
    """
    Signal detector based on strategy_spec.
    Uses the same logic as the backtest engine:
    - global filters
    - entry / entry_any blocks
    - Long and Short support
    """

    def __init__(self):
        pass

    def detect_signals(
        self,
        strategy_spec: Dict,
        current_indicators: Dict,
        prev_indicators: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Checks Long and Short entry conditions.

        Args:
            strategy_spec: strategy specification
            current_indicators: current indicators (last candle)
            prev_indicators: indicators of the previous candle

        Returns:
            List of signals: [{"direction": "long"/"short", "reasons": [...]}]
        """
        signals = []

        if not strategy_spec:
            return signals

        if prev_indicators is None:
            prev_indicators = {}

        # Global filters
        filters = strategy_spec.get("filters") or []
        if filters and not evaluate_conditions(filters, current_indicators, prev_indicators):
            logger.debug("Global filters not passed — no signals")
            return signals

        # Long
        long_spec = strategy_spec.get("long") or {}
        matched_long = self._check_entry(long_spec, current_indicators, prev_indicators)
        if matched_long is not None:
            signals.append({
                "direction": "long",
                "reasons": explain_conditions(matched_long, current_indicators, prev_indicators),
                "side_spec": long_spec,
            })

            logger.info("LONG signal: %s", explain_conditions(matched_long, current_indicators, prev_indicators))
        else:
            # Debug: entry conditions did not pass
            logger.debug("LONG not triggered: conditions not met. Check entry/entry_any in the long spec.")
            # Optional: dump the conditions themselves for debugging
            entry_conditions = long_spec.get("entry_any") or [long_spec.get("entry")]
            logger.debug("Long entry_any: %s", entry_conditions)

        # Short
        short_spec = strategy_spec.get("short") or {}
        matched_short = self._check_entry(short_spec, current_indicators, prev_indicators)
        if matched_short is not None:
            signals.append({
                "direction": "short",
                "reasons": explain_conditions(matched_short, current_indicators, prev_indicators),
                "side_spec": short_spec,
            })

            logger.info("SHORT signal: %s", explain_conditions(matched_short, current_indicators, prev_indicators))
        else:
            logger.debug("SHORT not triggered: conditions not met.")
            # Optional: dump the conditions themselves for debugging
            entry_conditions = short_spec.get("entry_any") or [short_spec.get("entry")]
            logger.debug("Short entry_any: %s", entry_conditions)

        return signals

    @staticmethod
    def _check_entry(side_spec: Dict, row: Dict, prev: Dict) -> Optional[List]:
        """
        Checks entry for side_spec (long/short).

        Priority: entry_any (list of OR groups). If absent — the legacy entry field.
        Full backward compatibility with older specs.

        Returns:
            list of conditions of the group that fired (for explain), or None
        """
        groups = side_spec.get("entry_any")
        if isinstance(groups, list) and groups:
            for group in groups:
                if isinstance(group, list) and group and evaluate_conditions(group, row, prev):
                    return group
            return None

        entry = side_spec.get("entry")
        if isinstance(entry, list) and entry and evaluate_conditions(entry, row, prev):
            return entry

        return None