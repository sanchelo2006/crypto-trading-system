"""
Auto-trading risk controls: daily and monthly losses per coin.

Tracks realized and unrealized P&L. When limits are hit, it initiates
a trading halt (daily) or strategy optimization (monthly).

State is saved to disk so it survives application restarts.

IMPORTANT: limits are calculated from the coin's CURRENT capital (dynamically),
not from the initial capital.
"""

import json
import logging
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional

import config

logger = logging.getLogger(__name__)

STATE_FILE = config.DATA_DIR / "auto_trading" / "risk_state.json"

# Serializes file access from multiple coin threads.
# Without it, threads can overwrite each other's updates and read the file
# while another thread is truncating it.
_state_file_lock = threading.Lock()


class RiskManager:
    """
    Risk controller for a single coin.

    Tracks:
    - daily realized P&L (from closed trades today)
    - monthly realized P&L (from closed trades this month)
    - unrealized P&L of open positions (for full daily limit control)
    - CURRENT coin capital (recalculated after every trade)
    """

    def __init__(
        self,
        symbol: str,
        daily_loss_limit_pct: float,
        monthly_loss_limit_pct: float,
        allocated_capital_usdt: float,
    ):
        """
        Args:
            symbol: trading pair
            daily_loss_limit_pct: daily loss limit as % of the coin's CURRENT capital
            monthly_loss_limit_pct: monthly loss limit as % of the coin's CURRENT capital
            allocated_capital_usdt: initial allocated coin capital in USDT
        """
        self.symbol = symbol
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.monthly_loss_limit_pct = monthly_loss_limit_pct
        
        # CURRENT capital (changes after every trade)
        self.current_capital = allocated_capital_usdt
        
        # Initial capital (for logging only)
        self.initial_capital = allocated_capital_usdt

        # Realized P&L in USDT
        self.daily_realized_pnl = 0.0
        self.monthly_realized_pnl = 0.0

        # Accounting dates
        self.current_date = datetime.now(timezone.utc).date()
        self.current_month = self.current_date.strftime("%Y-%m")

        # Triggers (so they don't fire repeatedly)
        self.daily_stop_triggered = False
        self.monthly_optimization_triggered = False

        # Load saved state
        self._load_state()

    # ----------------------------------------------------------------
    # P&L accounting
    # ----------------------------------------------------------------

    def record_closed_trade(self, pnl_pct: float, position_value_usdt: float):
        """
        Records a closed trade.

        Args:
            pnl_pct: trade result in %
            position_value_usdt: position value in USDT (for absolute P&L calculation)
        """
        self._check_rollover()

        pnl_usdt = position_value_usdt * (pnl_pct / 100.0)
        self.daily_realized_pnl += pnl_usdt
        self.monthly_realized_pnl += pnl_usdt
        
        # UPDATE CURRENT CAPITAL
        self.current_capital += pnl_usdt

        logger.info(
            f"[{self.symbol}] Trade recorded: {pnl_usdt:+.2f} USDT. "
            f"Current capital: {self.current_capital:.2f} USDT, "
            f"Day: {self.daily_realized_pnl:+.2f} USDT, "
            f"Month: {self.monthly_realized_pnl:+.2f} USDT"
        )

        self._save_state()

    def update_capital(self, new_capital: float):
        """
        Updates the coin's current capital (e.g. after a deposit).
        
        Args:
            new_capital: new capital in USDT
        """
        old = self.current_capital
        self.current_capital = new_capital
        logger.info(f"[{self.symbol}] Capital updated: {old:.2f} → {new_capital:.2f} USDT")
        self._save_state()

    def _check_rollover(self):
        """Checks if a new day/month has started and resets counters."""
        now = datetime.now(timezone.utc).date()

        # New day
        if now != self.current_date:
            logger.info(f"[{self.symbol}] New day. Resetting daily counter.")
            self.current_date = now
            self.daily_realized_pnl = 0.0
            self.daily_stop_triggered = False

        # New month
        new_month = now.strftime("%Y-%m")
        if new_month != self.current_month:
            logger.info(f"[{self.symbol}] New month. Resetting monthly counter.")
            self.current_month = new_month
            self.monthly_realized_pnl = 0.0
            self.monthly_optimization_triggered = False

        self._save_state()

    # ----------------------------------------------------------------
    # Limit checks
    # ----------------------------------------------------------------

    def get_total_daily_loss_pct(self, unrealized_pnl_usdt: float) -> float:
        """
        Returns the total daily loss as % of CURRENT capital
        (realized + unrealized).
        """
        total_usdt = self.daily_realized_pnl + unrealized_pnl_usdt
        if self.current_capital <= 0:
            return 0.0
        return (total_usdt / self.current_capital) * 100.0

    def get_total_monthly_loss_pct(self) -> float:
        """Returns the monthly realized loss as % of CURRENT capital."""
        if self.current_capital <= 0:
            return 0.0
        return (self.monthly_realized_pnl / self.current_capital) * 100.0

    def should_stop_daily(self, unrealized_pnl_usdt: float) -> bool:
        """
        Checks if the daily loss limit has been reached.

        Fires once (flag) to avoid duplicate stops.
        """
        if self.daily_stop_triggered:
            return False

        total_loss_pct = self.get_total_daily_loss_pct(unrealized_pnl_usdt)

        # The limit is given as a positive number (e.g. 5 = 5% loss)
        if total_loss_pct <= -abs(self.daily_loss_limit_pct):
            logger.warning(
                f"[{self.symbol}] Daily loss limit reached: "
                f"{total_loss_pct:.2f}% of current capital {self.current_capital:.2f} USDT "
                f"(limit -{self.daily_loss_limit_pct}%)"
            )
            self.daily_stop_triggered = True
            self._save_state()
            return True

        return False

    def should_optimize_monthly(self) -> bool:
        """
        Checks if the monthly loss limit has been reached to trigger optimization.

        Fires once (flag) to avoid repeated optimization triggers.
        """
        if self.monthly_optimization_triggered:
            return False

        monthly_loss_pct = self.get_total_monthly_loss_pct()

        if monthly_loss_pct <= -abs(self.monthly_loss_limit_pct):
            logger.warning(
                f"[{self.symbol}] Monthly loss limit reached: "
                f"{monthly_loss_pct:.2f}% of current capital {self.current_capital:.2f} USDT "
                f"(limit -{self.monthly_loss_limit_pct}%). "
                "Strategy optimization required."
            )
            self.monthly_optimization_triggered = True
            self._save_state()
            return True

        return False

    # ----------------------------------------------------------------
    # State persistence
    # ----------------------------------------------------------------

    def _save_state(self):
        """
        Saves the risk manager state to disk.

        Atomically (temp file + replace) and under a lock: a crash during write
        or a parallel write from another coin will not leave an empty file.
        """
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

            with _state_file_lock:
                state = {}
                if STATE_FILE.exists():
                    try:
                        with open(STATE_FILE, "r", encoding="utf-8") as f:
                            content = f.read().strip()
                        if content:
                            state = json.loads(content)
                    except (json.JSONDecodeError, OSError) as e:
                        logger.warning(
                            f"[{self.symbol}] risk_state.json corrupted ({e}). "
                            "Starting from a clean state."
                        )
                        state = {}

                state[self.symbol] = {
                    "daily_realized_pnl": self.daily_realized_pnl,
                    "monthly_realized_pnl": self.monthly_realized_pnl,
                    "current_capital": self.current_capital,
                    "initial_capital": self.initial_capital,
                    "allocated_capital_setting": self.initial_capital,
                    "current_date": str(self.current_date),
                    "current_month": self.current_month,
                    "daily_stop_triggered": self.daily_stop_triggered,
                    "monthly_optimization_triggered": self.monthly_optimization_triggered,
                }

                tmp_path = STATE_FILE.with_suffix(".json.tmp")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(state, f, ensure_ascii=False, indent=2)
                tmp_path.replace(STATE_FILE)

        except Exception as e:
            logger.error(f"Failed to save risk manager state: {e}")

    def _log_alloc_change_if_available(self, old_initial: float) -> None:
        """Hook for when the risk manager wants to notify the UI.
        Currently just writes to the logger, but kept as a separate method
        so notify_fn can be connected later without modifying _load_state.
        """
        pass

    def _load_state(self):
        """Loads the risk manager state from disk."""
        try:
            with _state_file_lock:
                if not STATE_FILE.exists():
                    return

                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                if not content:
                    logger.info(
                        f"[{self.symbol}] risk_state.json is empty — "
                        "using initial state"
                    )
                    return

                try:
                    state = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"[{self.symbol}] risk_state.json corrupted ({e}). "
                        "Using initial state."
                    )
                    return

                symbol_state = state.get(self.symbol)
                if not symbol_state:
                    return

                # Restore only if dates are current
                saved_date = symbol_state.get("current_date")
                saved_month = symbol_state.get("current_month")

                today = datetime.now(timezone.utc).date()
                this_month = today.strftime("%Y-%m")

                # Detect settings change.
                # self.initial_capital already holds the value from the constructor
                # (the new allocated_capital_usdt). If the file contains a different
                # value, the user changed the capital share, position size or another
                # parameter affecting allocation.
                stored_initial = symbol_state.get("initial_capital")
                stored_current = symbol_state.get("current_capital")

                settings_changed = (
                    stored_initial is not None
                    and abs(float(stored_initial) - float(self.initial_capital)) > 1e-6
                )

                if settings_changed and stored_current is not None:
                    # Adjust current_capital by the allocation delta.
                    # Example: base was 100 USDT, now 200. Accumulated P&L
                    # (current - initial = +5) is preserved, new balance = 205.
                    delta = float(self.initial_capital) - float(stored_initial)
                    self.current_capital = float(stored_current) + delta
                    logger.warning(
                        f"[{self.symbol}] Settings changed: "
                        f"capital base {stored_initial:.2f} → "
                        f"{self.initial_capital:.2f} USDT (delta {delta:+.2f}). "
                        f"Capital recalculated: {self.current_capital:.2f} USDT."
                    )
                    self._log_alloc_change_if_available(stored_initial)
                elif stored_current is not None:
                    self.current_capital = float(stored_current)

                # initial_capital is already up to date from the constructor.

                if saved_date == str(today):
                    self.daily_realized_pnl = symbol_state.get("daily_realized_pnl", 0.0)
                    self.daily_stop_triggered = symbol_state.get("daily_stop_triggered", False)

                if saved_month == this_month:
                    self.monthly_realized_pnl = symbol_state.get("monthly_realized_pnl", 0.0)
                    self.monthly_optimization_triggered = symbol_state.get(
                        "monthly_optimization_triggered", False
                    )

                logger.info(
                    f"[{self.symbol}] Risk manager state loaded: "
                    f"current capital {self.current_capital:.2f} USDT, "
                    f"day {self.daily_realized_pnl:+.2f} USDT, "
                    f"month {self.monthly_realized_pnl:+.2f} USDT"
                )

        except Exception as e:
            logger.error(f"Failed to load risk manager state: {e}")

    def reset_monthly_optimization_flag(self):
        """
        Resets the monthly optimization flag after successful completion.
        Allows optimization to trigger again in the same month if losses continue.
        """
        self.monthly_optimization_triggered = False
        self._save_state()