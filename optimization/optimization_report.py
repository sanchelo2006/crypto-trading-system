"""
Detailed text report of a single optimization.

Written in streaming mode (append):
  - does not lose data on crash or optimization cancellation;
  - does not depend on the UI;
  - does not affect prompt size and token consumption.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class OptimizationReport:

    # Fields with large data arrays. In the text report they are replaced
    # with a summary string "<N items>". Full data is saved
    # to CSV files via save_round().
    _SKIP_LIST_FIELDS = frozenset({
        "trades", "train_trades", "test_trades",
        "fold_results",
        "rejected_fingerprints", "rejected_specs",
    })

    # Fields with long text that are truncated to _MAX_TEXT_LEN.
    _TRUNCATE_FIELDS = frozenset({
        "reasoning",
    })

    _MAX_TEXT_LEN = 500

    def __init__(self, root_dir: Path, symbol: str):
        self.dir = Path(root_dir) / "optimization_reports"
        self.dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_symbol = "".join(
            ch if ch.isalnum() or ch in "_-" else "_" for ch in symbol
        )
        self.path = self.dir / f"{safe_symbol}_{stamp}.txt"

        self._append(
            "=" * 100 + "\n"
            f"OPTIMIZATION {symbol}\n"
            f"UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
            f"Report file: {self.path}\n"
            + "=" * 100 + "\n\n"
        )

    def _append(self, text: str) -> None:
        # A report write error must NEVER crash the optimization.
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            logger.exception("Failed to write report: %s", self.path)

    def line(self, message: str) -> None:
        """One progress line (what currently goes to the log)."""
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self._append(f"[{stamp}] {message}\n")

    def _sanitize(self, data: Any) -> Any:
        """Recursively removes heavy fields from data for the report.

        - Trade lists and window results are replaced with a counter.
        - Long texts (reasoning) are truncated.
        - Other data passes through unchanged.
        """
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if key in self._SKIP_LIST_FIELDS and isinstance(value, list):
                    result[key] = f"<{len(value)} items — full data in CSV>"
                elif (key in self._TRUNCATE_FIELDS
                      and isinstance(value, str)
                      and len(value) > self._MAX_TEXT_LEN):
                    result[key] = (value[:self._MAX_TEXT_LEN]
                                   + f"... (truncated, {len(value)} chars total)")
                else:
                    result[key] = self._sanitize(value)
            return result
        elif isinstance(data, list):
            return [self._sanitize(item) for item in data]
        return data


    def section(self, title: str, data: Any) -> None:
        """Structured block: spec, metrics, AI response, etc.

        Data passes through _sanitize(): heavy arrays (trades, window
        results) are replaced with counters, long texts are truncated.
        """
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        try:
            clean = self._sanitize(data)
            body = json.dumps(clean, ensure_ascii=False, indent=2, default=str)
        except Exception as exc:
            body = f"<failed to serialize: {exc}>\n{data!r}"

        self._append(
            f"\n{'-' * 100}\n"
            f"[{stamp}] {title}\n"
            f"{'-' * 100}\n"
            f"{body}\n"
        )