"""
Main auto-trading loop.

Manages background threads for each coin:
- monitoring new candles (15m)
- detecting signals based on strategy_spec
- opening positions with limit controls
- monitoring open positions and detecting closures
- controlling daily and monthly loss limits
- triggering strategy optimization on monthly limit
- handling network and balance errors

Limits:
- Maximum 1 simultaneously open position per coin
- On daily loss limit — close positions, stop trading, wait for confirmation
- On monthly loss limit — pause trading, run optimization,
  automatically resume trading after completion
"""
import math
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Callable, Dict, List, Optional

import config
from trading.executor import BybitExecutor
from trading.signal_detector import SignalDetector
from trading.position_manager import PositionManager
from trading.risk_manager import RiskManager
from trading.trade_logger import TradeLogger
from strategy_spec import evaluate_conditions
from ui.auto_mode import AutoModeState

logger = logging.getLogger(__name__)

# Global registry of active auto-traders by coin
_active_traders: Dict[str, "AutoTraderSymbol"] = {}
_active_traders_lock = threading.Lock()

MAX_POSITIONS_PER_SYMBOL = 1
# Limit of consecutive NON-network exceptions (code bugs, etc.)
# after which the trader stops itself.
# Network failures (ConnectionError) are NOT counted in this counter and
# never stop trading automatically.
NETWORK_RETRY_LIMIT = 60

# Defaults for auto-sanitize strategy_spec.
# Applied only if the field is MISSING or None.
DEFAULT_TRAILING_ATR_MULT = 2.0        # ATR multiplier for trailing stop
DEFAULT_MAX_HOLD_CANDLES = 48          # 48 × 15m = 12 hours

# Maximum gap between saving and loading risk_controls.
# If the application was stopped longer than this, the state is reset:
# cooldown would have expired anyway, and a "loss streak" from a week ago
# is no longer a streak. 24 hours = 96 candles of 15m.
RISK_CONTROLS_MAX_STALENESS_HOURS = 24

# Minimum interval between full candle history reloads.
# The auto-trading cycle runs more often (every 15 sec) for fast position
# monitoring, but a full 15m+1h history download on every iteration hits
# Bybit rate limits. Data updates by candles (15m), so reloading them more
# often than every few minutes makes no sense.
DATA_FETCH_MIN_INTERVAL_SEC = 240  # 4 minutes


def is_auto_trading_running(symbol: Optional[str] = None) -> bool:
    """
    Checks if auto-trading is running.
    
    If a symbol is provided, checks the specific coin.
    Otherwise, checks if at least one is running.
    
    Used to prevent simultaneous launch with auto mode.
    """
    with _active_traders_lock:
        if symbol is None:
            return any(t.is_running for t in _active_traders.values())
        
        trader = _active_traders.get(symbol)
        return trader is not None and trader.is_running


def get_all_states() -> Dict[str, Dict]:
    """Returns states of all auto-traders (for UI)."""
    with _active_traders_lock:
        return {
            symbol: trader.get_state()
            for symbol, trader in _active_traders.items()
        }

# File for auto-restoring traders after restart
_ACTIVE_TRADERS_FILE = config.DATA_DIR / "auto_trading" / "active_traders.json"


def _save_active_trader(symbol: str, cfg: AutoTraderConfig) -> None:
    """Saves the configuration of a running trader for auto-restore."""
    try:
        _ACTIVE_TRADERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if _ACTIVE_TRADERS_FILE.exists():
            try:
                with open(_ACTIVE_TRADERS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = {}
        data[symbol] = {
            "symbol": cfg.symbol,
            "category": cfg.category,
            "capital_share_pct": cfg.capital_share_pct,
            "position_size_pct": cfg.position_size_pct,
            "leverage": cfg.leverage,
            "daily_loss_limit_pct": cfg.daily_loss_limit_pct,
            "monthly_loss_limit_pct": cfg.monthly_loss_limit_pct,
            "optimization_days_back": cfg.optimization_days_back,
            "optimization_iterations": cfg.optimization_iterations,
            "walk_forward_folds": cfg.walk_forward_folds,
            "initial_capital": cfg.initial_capital,
            "commission_pct": cfg.commission_pct,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        tmp = _ACTIVE_TRADERS_FILE.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, _ACTIVE_TRADERS_FILE)
    except Exception:
        logger.exception(f"[{symbol}] Failed to save active trader")


def _remove_active_trader(symbol: str) -> None:
    """Removes the trader from the auto-restore file."""
    try:
        if not _ACTIVE_TRADERS_FILE.exists():
            return
        with open(_ACTIVE_TRADERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if symbol not in data:
            return
        del data[symbol]
        tmp = _ACTIVE_TRADERS_FILE.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, _ACTIVE_TRADERS_FILE)
    except Exception:
        logger.exception(f"[{symbol}] Failed to remove trader from file")


def _load_active_traders() -> Dict[str, dict]:
    """Loads the list of traders for auto-restore."""
    if not _ACTIVE_TRADERS_FILE.exists():
        return {}
    try:
        with open(_ACTIVE_TRADERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to read active traders file")
        return {}


class AutoTraderConfig:
    """Auto-trading configuration for a single coin."""
    
    def __init__(
        self,
        symbol: str,
        category: str = "linear",
        capital_share_pct: float = 30.0,
        position_size_pct: float = 50.0,
        leverage: int = 10,
        daily_loss_limit_pct: float = 1.0,
        monthly_loss_limit_pct: float = 15.0,
        optimization_days_back: int = 90,
        optimization_iterations: int = 10,
        walk_forward_folds: int = 3,
        initial_capital: float = 1000.0,
        commission_pct: float = 0.05,
    ):
        self.symbol = symbol
        self.category = category
        self.capital_share_pct = capital_share_pct
        self.position_size_pct = position_size_pct
        self.leverage = leverage
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.monthly_loss_limit_pct = monthly_loss_limit_pct
        self.optimization_days_back = optimization_days_back
        self.optimization_iterations = optimization_iterations
        self.walk_forward_folds = walk_forward_folds
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct


class AutoTraderSymbol:
    """
    Auto-trader for a single coin. Runs in its own background thread.
    """
    
    def __init__(
        self,
        cfg: AutoTraderConfig,
        *,
        executor: BybitExecutor,
        fetcher,
        memory,
        backtest_engine,
        backtest_optimizer,
        ai_client,
        notify_fn: Callable[[str, str], None],
        total_capital_usdt: float,
    ):
        self.cfg = cfg
        self.executor = executor
        self.fetcher = fetcher
        self.memory = memory
        self.backtest_engine = backtest_engine
        self.backtest_optimizer = backtest_optimizer
        self.ai_client = ai_client
        self.notify_fn = notify_fn
        self.total_capital = total_capital_usdt

        self.symbol = cfg.symbol

        # Allocated capital for the coin
        self.allocated_capital = total_capital_usdt * (cfg.capital_share_pct / 100.0)

        # Taker fee rate (in percent). Default is standard for Bybit
        # linear perpetuals, overwritten after get_fee_rates().
        self._taker_fee_pct: float = 0.055

        # Components
        self.signal_detector = SignalDetector()
        self.position_manager = PositionManager(
            executor, self.symbol, cfg.category,
            taker_fee_pct=self._taker_fee_pct,
        )
        self.risk_manager = RiskManager(
            symbol=self.symbol,
            daily_loss_limit_pct=cfg.daily_loss_limit_pct,
            monthly_loss_limit_pct=cfg.monthly_loss_limit_pct,
            allocated_capital_usdt=self.allocated_capital,
        )
        self.trade_logger = TradeLogger(self.symbol)

        # State
        self.is_running = False
        self.is_optimizing = False
        self.consecutive_network_errors = 0
        self._non_network_errors = 0
        self.last_processed_candle_time = None
        self.status_message = "Initializing..."
        self.log_lines: List[str] = []
        self._thread: Optional[threading.Thread] = None

        # IDs of already consumed Closed-PnL records
        self._consumed_closed_pnl_ids: set = set()

        # --- risk_controls (pause after loss streaks) ---
        self._consecutive_losses = 0
        self._recent_loss_flags: List[int] = []
        self._cooldown_remaining = 0  # how many candles remain paused
        
        # --- HEARTBEAT for UI ---
        self.last_iteration_utc: Optional[datetime] = None
        self.last_candle_time = None
        self.last_price: Optional[float] = None
        self.last_indicators_summary = ""
        self.candles_processed = 0

        # Cache of last successfully loaded indicators
        self._cached_indicators: Optional[Dict] = None
        self._cached_prev_indicators: Optional[Dict] = None
        self._cached_candle_time = None
        self._cached_at_utc: Optional[datetime] = None

        # Time of the last network history download (for throttling)
        self._last_data_fetch_utc: Optional[datetime] = None

        # --- Persistent position state ---
        # The file survives application restart: on startup we restore
        # the list of open positions to distinguish "our" positions from external
        # (orphan) ones, and also to catch up on closures after connection failure.
        self._state_dir = config.DATA_DIR / "auto_trading" / "positions"
        self._state_path = self._state_dir / f"{self.symbol}.json"

        # --- Limit order tracking ---
        # Store active limit orders for fill monitoring.
        # Format: [{ "order_id", "direction", "qty", "price", "side_spec",
        #            "reasons", "indicators", "candle_ts", "placed_at" }]
        self._pending_limit_orders: List[Dict] = []

        # Load saved state BEFORE the first _loop_iteration.
        self._load_state()
    
    # ----------------------------------------------------------------
    # Lifecycle management
    # ----------------------------------------------------------------
    
    def start(self):
        """
        Starts the background trading thread.

        Returns:
            True — trader actually started.
            False — start rejected (already running / auto mode /
                    settings do not allow opening a position).
        """
        if self.is_running:
            return False

        # Protection against simultaneous launch with auto mode
        auto_state = AutoModeState.get_state(self.symbol)
        if auto_state and auto_state.get("is_running"):
            self.notify_fn(
                f"[{self.symbol}] Cannot start auto-trading: "
                "auto mode is already enabled.",
                "warning",
            )
            return False

        # Sanitize strategy_spec: fill missing fields in management with safe
        # defaults (max_hold_candles, and if no management at all — trailing_atr_mult).
        # Done BEFORE starting the loop so live trading immediately works
        # with the full spec.
        try:
            self._sanitize_strategy_spec()
        except Exception:
            logger.exception(
                f"[{self.symbol}] Error sanitizing strategy_spec"
            )
            # Do not block startup due to sanitization failure.
        
        # Pre-check settings: can we even open a trade that passes
        # the exchange minimum. Do this BEFORE starting the loop.
        # If settings do not allow opening a position — refuse to start
        # instead of spinning the loop in vain (see SOLUSDT in logs).
        try:
            ok, price = self.executor.get_ticker_price(self.symbol)
            if ok and price:
                valid, warning = self.executor.validate_symbol_settings(
                    symbol=self.symbol,
                    capital_usdt=self.allocated_capital,
                    leverage=self.cfg.leverage,
                    position_size_pct=self.cfg.position_size_pct,
                    entry_price=price,
                )
                if not valid:
                    msg = warning or "Settings do not allow opening a position."
                    self._log_event("CONFIG_REJECT", msg)
                    self._notify(
                        f"[{self.symbol}] Auto-trading NOT started: {msg}",
                        "negative",
                    )
                    self.status_message = f"Start rejected: {msg}"
                    return False
        except Exception:
            logger.exception(f"[{self.symbol}] Pre-check settings error")

        self.is_running = True
        # Save for auto-restore after restart
        _save_active_trader(self.symbol, self.cfg)
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        self._log_event("START", f"Auto-trading started for {self.symbol}")
        self._notify(f"Auto-trading {self.symbol} started", "positive")
        return True
    
    def stop(self, close_positions: bool = False, reason: str = "manual"):
        """Stops trading."""
        self.is_running = False
        self.status_message = f"Stopped ({reason})"
        
        if close_positions:
            self._close_all_positions(reason=f"stop_{reason}")
        
        self._log_event("STOP", f"Auto-trading stopped: {reason}")
        self._notify(f"Auto-trading {self.symbol} stopped", "warning")
        
        with _active_traders_lock:
            _active_traders.pop(self.symbol, None)
        
        # Remove from auto-restore file
        _remove_active_trader(self.symbol)
    
    # ----------------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------------
    
    def _run_loop(self):
        """Main background trading loop."""
        logger.info(f"[{self.symbol}] Auto-trading thread started")

        # Initialize leverage and margin mode
        if not self._setup_leverage():
            self.stop(reason="leverage_setup_failed")
            return

        while self.is_running:
            try:
                self._loop_iteration()

            except ConnectionError as e:
                # NETWORK FAILURE. NEVER stop trading automatically —
                # only the user can do this via button. Wait for connection
                # restoration and try again.
                #
                # Exponential backoff with 300 sec ceiling:
                # 15 → 30 → 60 → 120 → 240 → 300 → 300 → ...
                # This way we do not spam the exchange during long outages.
                self.consecutive_network_errors += 1
                # Network failure is not a code bug. Reset the "other" error
                # counter so the trader does not stop due to alternation:
                # one code error → network gone → another error → ...
                self._non_network_errors = 0
                self._log_event(
                    "NETWORK",
                    f"Network error #{self.consecutive_network_errors}: {e}. "
                    "Trading continues, waiting for connection restoration.",
                )
                backoff = min(
                    15 * (2 ** min(self.consecutive_network_errors - 1, 4)),
                    300,
                )
                time.sleep(backoff)
                continue

            except Exception as e:
                # OTHER ERRORS (code bugs, unexpected states).
                # Count separately and stop only when the limit is exhausted,
                # to avoid looping on broken code.
                logger.exception(f"[{self.symbol}] Error in auto-trading loop")
                self._non_network_errors += 1
                self._log_event(
                    "ERROR",
                    f"Loop error ({self._non_network_errors}/"
                    f"{NETWORK_RETRY_LIMIT}): {e}",
                )

                if self._non_network_errors >= NETWORK_RETRY_LIMIT:
                    self._notify(
                        f"[{self.symbol}] {NETWORK_RETRY_LIMIT} non-network "
                        "errors in a row. Trading stopped.",
                        "negative",
                    )
                    self.stop(close_positions=False, reason="repeated_errors")
                    return

                time.sleep(60)
                continue

            else:
                # Successful iteration — reset both counters. After a long
                # connection outage, the next successful iteration will call
                # _monitor_positions → _detect_closed_positions, and all
                # trades closed while offline will be logged (Closed-PnL
                # is stored on the exchange for up to 7 days).
                self.consecutive_network_errors = 0
                self._non_network_errors = 0

            time.sleep(15)  # Pause between iterations

        logger.info(f"[{self.symbol}] Auto-trading thread finished")
    
    def _setup_leverage(self) -> bool:
        """Sets leverage and margin mode for the coin."""
        self.status_message = "Setting leverage..."
        
        success, err = self.executor.set_leverage(self.symbol, self.cfg.leverage)
        if not success:
            self._log_event("ERROR", f"Failed to set leverage: {err}")
            self._notify(f"[{self.symbol}] Failed to set leverage", "negative")
            return False

        # Taker fee rate — needed both for deduction on close and for
        # optimization (backtest should use the same rate).
        try:
            ok, rates = self.executor.get_fee_rates(self.symbol)
            if ok and rates:
                self._taker_fee_pct = float(rates["taker_pct"])
                self.position_manager.set_taker_fee_pct(self._taker_fee_pct)
                self._log_event(
                    "FEE_RATE",
                    f"Taker fee {self.symbol}: "
                    f"{self._taker_fee_pct:.4f}%",
                )
            else:
                self._log_event(
                    "FEE_RATE",
                    f"Failed to get taker rate ({rates}). "
                    f"Using default {self._taker_fee_pct:.4f}%",
                )
        except Exception:
            logger.exception(
                f"[{self.symbol}] Error getting taker rate, "
                f"using {self._taker_fee_pct:.4f}%"
            )

        return True

    def _sanitize_strategy_spec(self) -> bool:
        """
        Fills strategy_spec with safe defaults before starting trading.

        What we fix (only if the field is MISSING or None):
          - management.max_hold_candles → DEFAULT_MAX_HOLD_CANDLES
            Protection against stuck trades. Does not depend on the coin,
            depends on the timeframe (15m × 48 = 12 hours).

          - management.trailing_atr_mult → DEFAULT_TRAILING_ATR_MULT
            ONLY if management has no other management mechanism
            (breakeven / exit_on_*). If the model consciously chose
            move_to_breakeven_at_tp_pct (HYPEUSDT case) — we do not force
            trailing, that's its decision.

        Does NOT overwrite existing values. Does NOT touch other fields.

        Returns:
            True if spec was changed and saved.
            False if no changes were needed.
        """
        try:
            cm = self.memory.get(self.symbol)
            state = cm.get_state() or {}
            spec = state.get("strategy_spec")

            if not isinstance(spec, dict):
                return False

            changed_fields: List[str] = []

            for side_key in ("long", "short"):
                side = spec.get(side_key)
                if not isinstance(side, dict):
                    continue

                mgmt = side.get("management")
                if not isinstance(mgmt, dict):
                    mgmt = {}
                    side["management"] = mgmt

                # --- 1. max_hold_candles: add ALWAYS if not set ---
                # This is protection against stuck trades, not strategy.
                # Lives regardless of the coin.
                #
                # IMPORTANT: remember if time-stop was set CONSCIOUSLY
                # (before sanitization). If the strategy itself set
                # max_hold_candles — that's already an exit mechanism, and
                # forcing trailing on top of it would break swing strategies
                # where trailing kills trades on pullbacks.
                had_explicit_max_hold = mgmt.get("max_hold_candles") is not None

                if mgmt.get("max_hold_candles") is None:
                    mgmt["max_hold_candles"] = DEFAULT_MAX_HOLD_CANDLES
                    changed_fields.append(
                        f"{side_key}.management.max_hold_candles"
                    )

                # --- 2. trailing_atr_mult: only if there's no other ---
                #     management mechanism ---
                # Consciously set max_hold_candles counts as an exit
                # mechanism — in this case we do not force trailing.
                has_any_management = had_explicit_max_hold or any(
                    mgmt.get(key) not in (None, False, 0)
                    for key in (
                        "move_to_breakeven_at_tp_pct",
                        "trailing_atr_mult",
                        "breakeven_offset_pct",
                        "exit_on_level_failure",
                        "exit_on_trend_loss",
                        "exit_on_macd_flip",
                    )
                )

                if (
                    not has_any_management
                    and mgmt.get("trailing_atr_mult") is None
                ):
                    mgmt["trailing_atr_mult"] = DEFAULT_TRAILING_ATR_MULT
                    changed_fields.append(
                        f"{side_key}.management.trailing_atr_mult"
                    )

                if (
                    not has_any_management
                    and mgmt.get("trailing_atr_mult") is None
                ):
                    mgmt["trailing_atr_mult"] = DEFAULT_TRAILING_ATR_MULT
                    changed_fields.append(
                        f"{side_key}.management.trailing_atr_mult"
                    )

            if not changed_fields:
                return False

            cm.update_state({"strategy_spec": spec})
            cm.save()

            logger.info(
                f"[{self.symbol}] strategy_spec filled: "
                f"{', '.join(changed_fields)}"
            )
            self._log_event(
                "SPEC_SANITIZED",
                f"strategy_spec filled with defaults: "
                f"{', '.join(changed_fields)}",
            )
            self._notify(
                f"[{self.symbol}] Defaults added to strategy_spec: "
                f"{', '.join(changed_fields)}",
                "warning",
            )
            return True

        except Exception:
            logger.exception(
                f"[{self.symbol}] Error sanitizing strategy_spec"
            )
            return False
    
    def _loop_iteration(self):
        """One iteration of the main loop."""
        if self.is_optimizing:
            self.status_message = "Optimizing strategy..."
            self._monitor_positions()
            return

        self.status_message = "Monitoring market..."

        # 1. Load fresh candles and indicators — NO MORE OFTEN THAN EVERY N SECONDS.
        #    Between loads we use the cache. This is critical: a full history
        #    download = ~10-12 requests, and if done every 15 sec, Bybit
        #    responds with 10006 (rate limit).
        now_utc = datetime.now(timezone.utc)
        should_fetch = (
            self._last_data_fetch_utc is None
            or (now_utc - self._last_data_fetch_utc).total_seconds()
               >= DATA_FETCH_MIN_INTERVAL_SEC
        )

        if should_fetch:
            indicators, candle_time, prev_indicators = self._fetch_indicators()
            self._last_data_fetch_utc = now_utc

            if indicators is None:
                # Try to use cache — the trader continues monitoring
                # already open positions but does not generate new signals.
                cache_age = None
                if self._cached_at_utc is not None:
                    cache_age = (now_utc - self._cached_at_utc).total_seconds()

                if self._cached_indicators is None or cache_age is None or cache_age > 1800:
                    self.consecutive_network_errors += 1
                    self._log_event(
                        "NETWORK",
                        f"Indicators unavailable, cache missing or stale. "
                        f"Error {self.consecutive_network_errors}/{NETWORK_RETRY_LIMIT}",
                    )
                    raise ConnectionError("Failed to load indicators")

                self.consecutive_network_errors += 1
                self._log_event(
                    "NETWORK",
                    f"Indicators unavailable, using cache ({int(cache_age)}s). "
                    f"Error {self.consecutive_network_errors}/{NETWORK_RETRY_LIMIT}",
                )
                indicators = self._cached_indicators
                prev_indicators = self._cached_prev_indicators
                candle_time = self._cached_candle_time
            else:
                # Reset counter on successful load
                self.consecutive_network_errors = 0
                self._cached_indicators = indicators
                self._cached_prev_indicators = prev_indicators
                self._cached_candle_time = candle_time
                self._cached_at_utc = now_utc
        else:
            # Return cache without network request.
            indicators = self._cached_indicators
            prev_indicators = self._cached_prev_indicators
            candle_time = self._cached_candle_time

            if indicators is None:
                # First load has not happened yet — skip iteration.
                # This can only happen at the very start.
                time.sleep(5)
                return
        
        # --- HEARTBEAT ---
        self.last_iteration_utc = datetime.now(timezone.utc)
        self.last_candle_time = candle_time
        self.last_price = indicators.get("close")
        
        parts = []
        for name in ("rsi_14", "adx_14", "atr_14"):
            v = indicators.get(name)
            if v is not None:
                try:
                    parts.append(f"{name}={float(v):.2f}")
                except (TypeError, ValueError):
                    pass
        self.last_indicators_summary = ", ".join(parts)

        # 2. Monitor open positions (management + detect closures)
        self._monitor_positions(indicators=indicators)

        # 2.5. Monitor limit orders
        self._monitor_pending_limit_orders()    

        # 3. Check if a new day/month has started
        self.risk_manager._check_rollover()
        
        # 4. CHECK RISK LIMITS ALWAYS (even if no positions)
        unrealized = (
            self._get_unrealized_pnl()
            if self.position_manager.positions else 0.0
        )
        
        if self.risk_manager.should_stop_daily(unrealized):
            self._handle_daily_stop()
            return

        # 5. Check monthly limit (trigger optimization)
        if self.risk_manager.should_optimize_monthly():
            self._trigger_optimization()
            return

        # 6. Process signals only on a new closed candle.
        #    CANDLE events are no longer written to events: they duplicated
        #    SIGNAL_CHECK on every candle and polluted the log.
        if candle_time != self.last_processed_candle_time:
            self.last_processed_candle_time = candle_time
            self.candles_processed += 1
            self._process_signals(indicators, prev_indicators)
    
    # ----------------------------------------------------------------
    # Indicators
    # ----------------------------------------------------------------
    
    def _fetch_indicators(self):
        """
        Loads fresh candles and calculates indicators.

        Returns:
            (current indicators, last candle time, previous indicators)
            or (None, None, None) on error

        Load only 15m history for 5 days:
          - indicators need at most ~100 candles (SMA50, avg_volume_96);
          - daily pivots require at least 2-3 full days;
          - 1h features are computed by resampling inside _build_indicator_frame,
            a separate 1h download is NOT needed and only eats rate limit.
        """
        try:
            now = datetime.now(timezone.utc)
            start_date = now - timedelta(days=14)

            prices_15m = self.fetcher.fetch_ohlcv(
                symbol=self.symbol,
                interval="15",
                start_date=start_date,
                end_date=now,
                category=self.cfg.category,
            )

            if prices_15m is None or prices_15m.empty:
                return None, None, None

            df = self.backtest_engine._build_indicator_frame(prices_15m)


            if len(df) < 2:
                return None, None, None

            # The last candle may be unclosed — use the second-to-last
            records = df.to_dict("records")
            current = records[-2]
            prev = records[-3] if len(records) >= 3 else {}

            candle_time = current.get("timestamp")

            return current, candle_time, prev

        except Exception as e:
            logger.exception(f"[{self.symbol}] Error loading indicators")
            raise ConnectionError(f"Error loading indicators: {e}")
    
    # ----------------------------------------------------------------
    # Signals and position opening
    # ----------------------------------------------------------------
    
    def _process_signals(self, indicators: Dict, prev_indicators: Dict):
        """Processes trading signals."""
        cm = self.memory.get(self.symbol)
        strategy_spec = cm.get_state().get("strategy_spec")

        if not strategy_spec:
            self._log_event(
                "ERROR",
                "No strategy_spec for the coin. Trading impossible.",
            )
            self.stop(reason="no_strategy")
            return

        # --- Check cooldown after loss streak ---
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            self._log_event(
                "COOLDOWN",
                f"Pause after loss streak. Candles remaining: {self._cooldown_remaining}",
            )
            # Persist the decremented value so restart does not roll back cooldown.
            self._save_state()
            return


        # --- ENTRY CONDITION DIAGNOSTICS ---
        # Log everything in one group (tied to the candle): global filters
        # + LONG + SHORT. This glues all signal events of one iteration
        # into one block in the HTML log.
        candle_group = (
            str(self.last_candle_time)
            if self.last_candle_time is not None
            else None
        )
        self._log_signal_diagnostics(
            strategy_spec, indicators, prev_indicators, group=candle_group,
        )

        signals = self.signal_detector.detect_signals(
            strategy_spec=strategy_spec,
            current_indicators=indicators,
            prev_indicators=prev_indicators,
        )


        
        if not signals:
            return
        
        # Check open position limit
        open_count = len(self.position_manager.positions)
        available_slots = MAX_POSITIONS_PER_SYMBOL - open_count
        
        if available_slots <= 0:
            self._log_event(
                "OPEN",
                f"Signal exists, but position limit ({MAX_POSITIONS_PER_SYMBOL}) reached. Skipping.",
            )
            return
        
        for signal in signals[:available_slots]:
            self._open_position_from_signal(signal, indicators)

    def _get_risk_controls(self) -> Dict:
        """
        Reads risk_controls from strategy_spec.
        Normalizes values the same way as backtest_engine._get_risk_controls.
        """
        try:
            cm = self.memory.get(self.symbol)
            spec = cm.get_state().get("strategy_spec") or {}
        except Exception:
            return {"enabled": False}

        raw = spec.get("risk_controls") or {}
        if not bool(raw.get("enabled", False)):
            return {"enabled": False}

        return {
            "enabled": True,
            "max_consecutive_losses": max(1, int(raw.get("max_consecutive_losses", 3))),
            "pause_candles_after_losses": max(1, int(raw.get("pause_candles_after_losses", 96))),
            "max_losses_in_window": max(1, int(raw.get("max_losses_in_window", 5))),
            "loss_window_trades": max(1, int(raw.get("loss_window_trades", 10))),
        }


    def _should_pause_after_loss(self, result_pct: float, controls: Dict) -> bool:
        """
        Decides whether to activate a pause after closing a losing trade.
        Logic identical to backtest_engine._should_pause_after_loss.
        """
        if not controls.get("enabled"):
            return False
        if result_pct > 0:
            return False

        if self._consecutive_losses >= controls["max_consecutive_losses"]:
            return True

        window = controls["loss_window_trades"]
        max_losses = controls["max_losses_in_window"]
        last_window = self._recent_loss_flags[-window:]
        if len(last_window) >= window and sum(last_window) >= max_losses:
            return True

        return False

    def _log_signal_diagnostics(
        self,
        strategy_spec: Dict,
        indicators: Dict,
        prev: Dict,
        group: Optional[str] = None,
    ):
        """
        Logs entry condition checks for LONG and SHORT.

        Supports both specification forms:
          - "entry": [cond, ...] — one group, all conditions AND;
          - "entry_any": [[cond, ...], [cond, ...]] — OR groups.

        Args:
            group: group key for the HTML separator. Usually the candle
                timestamp, so FILTERS + LONG + SHORT glue into one visual block.
        """
        from strategy_spec import evaluate_conditions

        def _check_and_format(conditions: List, label: str) -> str:
            if not conditions:
                return f"{label}: no conditions"

            passed = []
            failed = []

            for cond in conditions:
                if not isinstance(cond, dict):
                    continue

                result = evaluate_conditions([cond], indicators, prev)
                cond_type = cond.get("type", "unknown")

                if cond_type == "compare":
                    left_key = cond.get("left")
                    op = cond.get("op", "?")
                    right = cond.get("right")
                    left_val = indicators.get(left_key, "N/A")

                    if isinstance(right, dict):
                        right_key = right.get("name")
                        mult = right.get("mult", 1.0)
                        right_val = indicators.get(right_key, "N/A")
                        if right_val != "N/A":
                            try:
                                right_val = float(right_val) * float(mult)
                            except Exception:
                                pass
                        desc = (
                            f"{left_key}({left_val}) {op} "
                            f"{right_key}*{mult}({right_val})"
                        )
                    else:
                        desc = f"{left_key}({left_val}) {op} {right}"

                    (passed if result else failed).append(
                        ("✓ " if result else "✗ ") + desc
                    )

                elif cond_type == "range":
                    value_key = cond.get("value")
                    min_val = cond.get("min")
                    max_val = cond.get("max")
                    actual = indicators.get(value_key, "N/A")
                    desc = f"{value_key}({actual}) in [{min_val}, {max_val}]"
                    (passed if result else failed).append(
                        ("✓ " if result else "✗ ") + desc
                    )

                elif cond_type == "volume":
                    ref = cond.get("ref")
                    mult = cond.get("mult", 1.0)
                    vol = indicators.get("volume", "N/A")
                    ref_vol = indicators.get(ref, "N/A")
                    desc = f"volume({vol}) > {ref}({ref_vol})*{mult}"
                    (passed if result else failed).append(
                        ("✓ " if result else "✗ ") + desc
                    )

                elif cond_type == "slope":
                    value_key = cond.get("value")
                    direction = cond.get("dir")
                    desc = f"slope({value_key}) → {direction}"
                    (passed if result else failed).append(
                        ("✓ " if result else "✗ ") + desc
                    )

                else:
                    desc = f"{cond_type}"
                    (passed if result else failed).append(
                        ("✓ " if result else "✗ ") + desc
                    )

            summary = f"{label}: {len(passed)}/{len(conditions)} conditions"
            if passed:
                summary += f"\n  PASSED: {'; '.join(passed[:3])}"
                if len(passed) > 3:
                    summary += f" ... and {len(passed) - 3} more"
            if failed:
                summary += f"\n  FAILED: {'; '.join(failed[:3])}"
                if len(failed) > 3:
                    summary += f" ... and {len(failed) - 3} more"

            return summary

        # --- Global filters ---
        filters = strategy_spec.get("filters") or []
        if filters:
            filters_result = evaluate_conditions(filters, indicators, prev)
            filters_summary = _check_and_format(filters, "FILTERS")
            self._log_event(
                "SIGNAL_CHECK",
                f"Global filters: "
                f"{'PASSED' if filters_result else 'FAILED'}\n"
                f"{filters_summary}",
                group=group,
            )
            if not filters_result:
                return

        # --- LONG and SHORT ---
        for side in ("long", "short"):
            side_spec = strategy_spec.get(side) or {}
            label = side.upper()

            groups = side_spec.get("entry_any")
            if isinstance(groups, list) and groups:
                for gidx, grp in enumerate(groups, start=1):
                    if not isinstance(grp, list) or not grp:
                        continue
                    grp_label = f"{label} entry_any [{gidx}/{len(groups)}]"
                    grp_summary = _check_and_format(grp, grp_label)
                    grp_passed = evaluate_conditions(grp, indicators, prev)
                    self._log_event(
                        "SIGNAL_CHECK",
                        f"{label}: {'SIGNAL' if grp_passed else 'NO SIGNAL'}\n"
                        f"{grp_summary}",
                        group=group,
                    )
                continue

            entry = side_spec.get("entry") or []
            if entry:
                entry_summary = _check_and_format(entry, f"{label} ENTRY")
                entry_passed = evaluate_conditions(entry, indicators, prev)
                self._log_event(
                    "SIGNAL_CHECK",
                    f"{label}: {'SIGNAL' if entry_passed else 'NO SIGNAL'}\n"
                    f"{entry_summary}",
                    group=group,
                )
    
    def _open_position_from_signal(self, signal: Dict, indicators: Dict):
        """Opens a position on a signal."""
        direction = signal["direction"]
        side_spec = signal["side_spec"]
        reasons = signal["reasons"]
        
        # Get current price
        success, current_price = self.executor.get_ticker_price(self.symbol)
        if not success or current_price is None:
            self._log_event("ERROR", f"Failed to get price for opening: {current_price}")
            return
        
        # Calculate position size
        position_value = self.allocated_capital * (self.cfg.position_size_pct / 100.0)
        
        success, qty = self.executor.calculate_position_size(
            symbol=self.symbol,
            capital_usdt=position_value,
            leverage=self.cfg.leverage,
            entry_price=current_price,
        )

        if not success or qty is None or qty <= 0:
            # qty error from calculate_position_size may contain the specific
            # reason (e.g. notional < min). Propagate it to the log and UI.
            detail = qty if isinstance(qty, str) else "Failed to calculate position size"
            self._log_event("ORDER_TOO_SMALL", detail)
            self._notify(
                f"[{self.symbol}] Order will not pass: {detail}",
                "warning",
            )
            return
        
        # Determine entry mode: limit (maker) or market (taker)
        use_limit_entry = self._should_use_limit_entry(indicators)

        if use_limit_entry:
            # LIMIT ENTRY (maker, low fee)
            success, order_id_or_err, limit_price = (
                self.position_manager.open_position_limit(
                    direction=direction,
                    qty=qty,
                    side_spec=side_spec,
                    entry_reasons=reasons,
                    indicators=indicators,
                    entry_candle_ts=self.last_candle_time,
                    price_offset_ticks=0,
                    timeout_seconds=900,  # 15 minutes
                )
            )
            if success:
                # Save order for monitoring
                self._pending_limit_orders.append({
                    "order_id": order_id_or_err,
                    "direction": direction,
                    "qty": qty,
                    "price": float(limit_price),
                    "side_spec": side_spec,
                    "reasons": reasons,
                    "indicators": dict(indicators),
                    "candle_ts": self.last_candle_time,
                    "placed_at": datetime.now(timezone.utc),
                })
                candle_group = (
                    str(self.last_candle_time)
                    if self.last_candle_time is not None
                    else None
                )
                self._log_event(
                    "OPEN",
                    f"Limit order placed: {direction.upper()} {qty} "
                    f"@ {limit_price} (Post-Only)",
                    group=candle_group,
                )
                self._save_state()
                return
            else:
                # Limit entry failed — fallback to market
                self._log_event(
                    "WARNING",
                    f"Limit entry failed ({order_id_or_err}). "
                    f"Switching to market.",
                )
                success, current_price = self.executor.get_ticker_price(self.symbol)
                if not success or current_price is None:
                    self._log_event("ERROR", f"Failed to get price: {current_price}")
                    return

        # MARKET ENTRY (taker, as before)
        success, err = self.position_manager.open_position(
            direction=direction,
            qty=qty,
            current_price=current_price,
            side_spec=side_spec,
            entry_reasons=reasons,
            indicators=indicators,
            entry_candle_ts=self.last_candle_time,
        )
        
        if success:
            # Save state immediately after successful opening — so on restart
            # the position is in the file and not counted as orphan.
            self._save_state()

            candle_group = (
                str(self.last_candle_time)
                if self.last_candle_time is not None
                else None
            )
            self._log_event(
                "OPEN",
                f"Position opened: {direction.upper()} {qty} @ {current_price:.4f}",
                group=candle_group,
            )

            # Check if protection was set successfully.
            # PositionManager puts the error in the tpsl_error field of the last position.
            last_pos = (
                self.position_manager.positions[-1]
                if self.position_manager.positions else None
            )
            tpsl_error = last_pos.get("tpsl_error") if last_pos else None

            if tpsl_error:
                self._log_event(
                    "TPSL_FAILED",
                    f"TP/SL NOT set for {direction.upper()} position: {tpsl_error}. "
                    f"Position opened WITHOUT protection.",
                )
                self._notify(
                    f"[{self.symbol}] WARNING: {direction.upper()} opened without TP/SL. "
                    f"Check the position manually.",
                    "negative",
                )
            else:
                self._notify(
                    f"[{self.symbol}] Opened {direction.upper()} position @ {current_price:.4f}",
                    "positive",
                )
        else:
            self._log_event("ERROR", f"Failed to open position: {err}")
            
            # If the reason is insufficient funds, stop trading
            if err and ("insufficient" in str(err).lower() or "balance" in str(err).lower()):
                self._notify(
                    f"[{self.symbol}] Insufficient funds. Trading stopped.",
                    "negative",
                )
                self._log_event("INSUFFICIENT_BALANCE", "Insufficient funds to open position")
                self.stop(close_positions=False, reason="insufficient_balance")

    def _should_use_limit_entry(self, indicators: Dict) -> bool:
        """
        Determines whether to use limit entry (maker).

        Limit entry is used when:
        1. Strategy does not require immediate entry (swing, not scalping).
        2. Volatility is not extreme (otherwise the order will not fill).

        For swing strategies on 4H — always limit.
        For scalping on 15m — market.
        """
        # If the spec specifies a 4h or higher timeframe — limit entry
        # Determine by ATR magnitude: on 4H ATR is significantly larger
        atr = indicators.get("atr_14")
        close = indicators.get("close")
        if atr and close and close > 0:
            atr_pct = (atr / close) * 100
            # On 15m ATR is usually < 0.5%, on 4H > 1%
            if atr_pct > 0.8:
                return True

        # Default — market entry (safer)
        return False
    
    # ----------------------------------------------------------------
    # Position monitoring
    # ----------------------------------------------------------------
    
    def _monitor_positions(self, indicators: Optional[Dict] = None):
        """
        Monitors open positions:
        0) Synchronization with the exchange (picks up orphan positions after restart)
        1) Management (breakeven, trailing) — updates SL on the exchange
        2) Closure detection
        """
        # 0) Always synchronize local state with the exchange.
        #    This is needed so the UI shows the real number of positions even
        #    after trader restart (when the local list is empty but positions
        #    remain on the exchange).
        try:
            self.position_manager.sync_with_exchange()
        except Exception:
            logger.exception(f"[{self.symbol}] sync_with_exchange error")

        if not self.position_manager.positions:
            return

        # 1) Manage open positions by management rules
        if indicators is not None:
            self._apply_live_management(indicators)

        # 2) Synchronization with the exchange and closure detection
        self._detect_closed_positions()

    def _monitor_pending_limit_orders(self):
        """
        Checks the status of limit orders:
        - Filled → finalize position opening
        - Timeout → cancel order, skip signal
        """
        if not self._pending_limit_orders:
            return

        now = datetime.now(timezone.utc)
        still_pending = []

        for order_info in self._pending_limit_orders:
            order_id = order_info["order_id"]
            placed_at = order_info["placed_at"]
            timeout = 900  # 15 minutes

            # Check timeout
            elapsed = (now - placed_at).total_seconds()
            if elapsed > timeout:
                self.executor.cancel_order(self.symbol, order_id)
                self._log_event(
                    "ORDER_TIMEOUT",
                    f"Limit order {order_id} not filled in {timeout}s. Cancelled.",
                )
                continue

            # Check order status
            success, order_data = self.executor.get_order_info(self.symbol, order_id)
            if not success or order_data is None:
                still_pending.append(order_info)
                continue

            status = order_data.get("orderStatus", "")

            if status == "Filled":
                # Order filled → finalize position opening
                filled_qty = float(order_data.get("cumExecQty", 0) or 0)
                avg_price = float(order_data.get("avgPrice", 0) or 0)

                if filled_qty > 0 and avg_price > 0:
                    success, err = self.position_manager.finalize_limit_entry(
                        direction=order_info["direction"],
                        qty=filled_qty,
                        entry_price=avg_price,
                        side_spec=order_info["side_spec"],
                        entry_reasons=order_info["reasons"],
                        indicators=order_info["indicators"],
                        entry_candle_ts=order_info["candle_ts"],
                        order_id=order_id,
                    )
                    if success:
                        self._save_state()
                        self._log_event(
                            "OPEN",
                            f"Limit position opened: "
                            f"{order_info['direction'].upper()} {filled_qty} "
                            f"@ {avg_price:.4f}",
                        )
                        self._notify(
                            f"[{self.symbol}] Limit entry filled "
                            f"@ {avg_price:.4f}",
                            "positive",
                        )
                    else:
                        self._log_event(
                            "ERROR",
                            f"Error finalizing limit entry: {err}",
                        )
                # Do not add to still_pending — order processed

            elif status == "Cancelled":
                self._log_event(
                    "ORDER_CANCELLED",
                    f"Limit order {order_id} was cancelled by the exchange.",
                )
                # Do not add to still_pending

            elif status in ("New", "PartiallyFilled"):
                still_pending.append(order_info)

            else:
                # Unknown status — leave pending
                still_pending.append(order_info)

        self._pending_limit_orders = still_pending
        
    def _apply_live_management(self, candle: Dict):
        """
        Applies management rules to open positions:
        1) SL management (breakeven, trailing) — set_trading_stop on the exchange.
        2) Soft exits (time_stop, exit_on_*) — when triggered, the position
           is closed by a market order.

        Both actions are performed once per candle.
        """
        candle_ts = candle.get("timestamp")
        if candle_ts is None:
            return

        for position in list(self.position_manager.positions):
            # Do not manage orphan positions — they have no real SL/TP.
            if position.get("is_orphan"):
                continue

            last_ts = position.get("_last_managed_candle_ts")
            if last_ts is not None and candle_ts <= last_ts:
                continue  # this candle for this position is already processed

            try:
                self._apply_management_to_position(position, candle)
            except Exception:
                logger.exception(
                    f"[{self.symbol}] Error managing position "
                    f"{position.get('position_id')}"
                )

            # --- Soft exits (time_stop, exit_on_*) ---
            try:
                exit_price, reason = self.position_manager.check_soft_exit(
                    position, candle
                )
                if exit_price is not None and reason:
                    self._close_by_soft_exit(position, reason, exit_price)
            except Exception:
                logger.exception(
                    f"[{self.symbol}] Error checking soft exit for "
                    f"{position.get('position_id')}"
                )

            position["_last_managed_candle_ts"] = candle_ts

    def _apply_management_to_position(self, position: Dict, candle: Dict):
        """
        Recalculates SL for one position and, if it changed, sends the
        update to the exchange via set_trading_stop.
        TP is not touched — left as it was at opening.
        """
        mgmt = position.get("management", {}) or {}
        if not mgmt:
            return

        direction = position["direction"]
        entry = position["entry_price"]
        take = position["take_profit"]
        current_sl = position["stop_loss"]

        new_sl: Optional[float] = None
        breakeven_applied = False
        trailing_applied = False

        # --- 1. Breakeven ---
        trigger = mgmt.get("move_to_breakeven_at_tp_pct")
        if trigger is not None:
            try:
                trigger = float(trigger)
            except (TypeError, ValueError):
                trigger = None

            if trigger is not None and 0 < trigger <= 1:
                try:
                    offset = float(mgmt.get("breakeven_offset_pct", 0.0)) / 100.0
                except (TypeError, ValueError):
                    offset = 0.0

                if direction == "long":
                    trigger_price = entry + (take - entry) * trigger
                    be_price = entry * (1 + offset)
                    high = candle.get("high")
                    if high is not None and high >= trigger_price:
                        if current_sl < be_price:
                            new_sl = be_price
                            breakeven_applied = True
                else:
                    trigger_price = entry - (entry - take) * trigger
                    be_price = entry * (1 - offset)
                    low = candle.get("low")
                    if low is not None and low <= trigger_price:
                        if current_sl > be_price:
                            new_sl = be_price
                            breakeven_applied = True

        # --- 2. Trailing ATR ---
        trail_mult = mgmt.get("trailing_atr_mult")
        if trail_mult:
            try:
                trail_mult = float(trail_mult)
            except (TypeError, ValueError):
                trail_mult = None

            atr = candle.get("atr_14")
            close = candle.get("close")
            atr_ok = (
                atr is not None
                and not (isinstance(atr, float) and math.isnan(atr))
                and atr > 0
            )

            if trail_mult and trail_mult > 0 and atr_ok and close is not None:
                if direction == "long":
                    trail_sl = float(close) - trail_mult * float(atr)
                    if trail_sl > current_sl:
                        if new_sl is None or trail_sl > new_sl:
                            new_sl = trail_sl
                            trailing_applied = True
                else:
                    trail_sl = float(close) + trail_mult * float(atr)
                    if trail_sl < current_sl:
                        if new_sl is None or trail_sl < new_sl:
                            new_sl = trail_sl
                            trailing_applied = True

        # --- 3. Checks before sending to the exchange ---
        if new_sl is None:
            return

        # Get tickSize. If unavailable — DO NOT send set_trading_stop
        # with an unrounded SL: the exchange may reject such a request,
        # and SL will silently remain old. Better to skip the candle and
        # try on the next one than to create the appearance of an update.
        success_info, info = self.executor.get_instrument_info(self.symbol)

        if not success_info or not info or not info.get("tickSize"):
            self._log_event(
                "ERROR",
                f"SL not updated ({direction.upper()}): tickSize unavailable "
                f"({info}). Skipping until next candle.",
            )
            return

        new_sl = self.executor.round_price(new_sl, info["tickSize"])

        # IMPORTANT: compare the ALREADY ROUNDED new_sl with the old SL.
        # Previously the comparison was before rounding: "raw" 88.8798 != 88.88
        # passed the request through, after rounding the exchange saw the same
        # 88.88 and responded with 34040 — and so every candle.
        if new_sl == current_sl:
            return

        # Check SL side relative to the current price.
        # The exchange will reject SL if it's on the wrong side of LastPrice
        # (ErrCode 10001): for long SL must be BELOW price, for short — ABOVE.
        # Take close of the last closed candle as a proxy for current price —
        # a separate ticker request is redundant.
        close_price = candle.get("close")
        if close_price is not None:
            if direction == "long" and new_sl >= close_price:
                self._log_event(
                    "MGMT",
                    f"SL skipped ({direction.upper()}): {new_sl:.4f} >= "
                    f"close {close_price:.4f}. Waiting for next candle.",
                )
                return
            if direction == "short" and new_sl <= close_price:
                self._log_event(
                    "MGMT",
                    f"SL skipped ({direction.upper()}): {new_sl:.4f} <= "
                    f"close {close_price:.4f}. Waiting for next candle.",
                )
                return

        success, err = self.executor.set_trading_stop(
            symbol=self.symbol,
            stop_loss=new_sl,
            take_profit=take,
        )

        if success:
            old = current_sl
            position["stop_loss"] = new_sl
            self._save_state()

            # Determine the reason for the shift for a separate sl_updates log.
            if breakeven_applied and trailing_applied:
                reason = "breakeven+trailing"
            elif breakeven_applied:
                reason = "breakeven"
            elif trailing_applied:
                reason = "trailing"
            else:
                reason = ""

            # Separate entry in sl_updates with position_id — so the log
            # can unambiguously link SL movement to a specific trade.
            try:
                self.trade_logger.log_sl_update(
                    position_id=position.get("position_id", ""),
                    direction=direction,
                    old_sl=old,
                    new_sl=new_sl,
                    reason=reason,
                    extra={
                        "atr_14": candle.get("atr_14"),
                        "close": candle.get("close"),
                        "candle_ts": str(candle.get("timestamp") or ""),
                    },
                )
            except Exception:
                logger.exception(
                    f"[{self.symbol}] Failed to write sl_update "
                    f"for {position.get('position_id')}"
                )

            self._log_event(
                "MGMT",
                f"SL updated ({direction.upper()}): {old:.4f} → {new_sl:.4f} "
                f"[{reason or 'n/a'}]",
            )
        else:
            self._log_event(
                "ERROR",
                f"Failed to update SL ({direction.UPPER()}): {err}",
            )

    def _close_by_soft_exit(self, position: Dict, reason: str, exit_price: float):
        """
        Closes a position on a soft exit (time_stop / exit_on_*).

        Sends a reduce-only market order via PositionManager.close_position,
        then registers the trade itself in risk_manager and trade_logger,
        as _close_all_positions does.
        """
        self._log_event(
            "SOFT_EXIT",
            f"Soft exit ({position['direction'].upper()}): "
            f"{reason} @ ~{exit_price:.4f}",
        )

        success, result = self.position_manager.close_position(
            position, reason, current_price=exit_price
        )
        if not success:
            self._log_event(
                "ERROR",
                f"Failed to close on soft exit ({reason}): {result}",
            )
            return

        trade = result
        position_value = trade["qty"] * trade["entry_price"]
        self.risk_manager.record_closed_trade(
            trade["result_pct"], position_value
        )
        trade["position_value_usdt"] = round(position_value, 2)
        trade["pnl_usdt"] = round(
            position_value * (trade["result_pct"] / 100.0), 2
        )
        self.trade_logger.log_trade(trade)

        # risk_controls accounting (as in _register_closed_position)
        controls = self._get_risk_controls()
        pnl_pct = trade["result_pct"]
        if pnl_pct <= 0:
            self._consecutive_losses += 1
            self._recent_loss_flags.append(1)
        else:
            self._consecutive_losses = 0
            self._recent_loss_flags.append(0)
        window = controls.get("loss_window_trades", 10)
        if window > 0 and len(self._recent_loss_flags) > window:
            self._recent_loss_flags = self._recent_loss_flags[-window:]
        if self._should_pause_after_loss(pnl_pct, controls):
            self._cooldown_remaining = controls["pause_candles_after_losses"]
            self._log_event(
                "COOLDOWN_ACTIVATED",
                f"Activated pause for {self._cooldown_remaining} candles",
            )

        self._save_state()

        close_event_type = "CLOSE_WIN" if pnl_pct > 0 else "CLOSE_LOSS"
        self._log_event(
            close_event_type,
            f"Position closed ({reason}): "
            f"{trade['direction'].upper()} @ {trade['exit_price']:.4f}, "
            f"PnL={pnl_pct:+.2f}% "
            f"({trade['pnl_usdt']:+.2f} USDT)",
        )
        
    
    def _detect_closed_positions(self):
        """
        Detects positions closed on the exchange (by order execution).
        
        If the position size on the exchange became 0 — log the closure.
        """
        success, exchange_positions = self.executor.get_position_info(self.symbol)
        if not success:
            return
        
        # Build a map: size on the exchange by direction
        exchange_size_by_side = {}
        for pos in exchange_positions or []:
            side = pos.get("side", "").lower()
            size = float(pos.get("size", 0))
            if side in ("buy", "sell"):
                exchange_size_by_side[side] = size
        
        for position in list(self.position_manager.positions):

            # GRACE PERIOD: the exchange registers the position with a delay.
            # Do not check closure in the first 2 minutes after opening.
            # For orphans, entry_time = tracker restart time, grace is also
            # appropriate — the exchange may not have had time to return fresh data.
            opened_at = position.get("entry_time")
            if opened_at is not None:
                try:
                    if isinstance(opened_at, str):
                        opened_at = datetime.fromisoformat(opened_at)
                    if (datetime.now(timezone.utc) - opened_at).total_seconds() < 120:
                        continue
                except Exception:
                    pass

            direction = position["direction"]
            side_key = "buy" if direction == "long" else "sell"
            exchange_size = exchange_size_by_side.get(side_key, 0)

            # If there is no position in this direction on the exchange (or it's
            # radically smaller than locally recorded) — consider it closed.
            # This also works for orphans: their qty = size from the exchange,
            # so 0 < qty*0.5 → True immediately after the position disappears.
            if exchange_size < position["qty"] * 0.5:
                self._register_closed_position(position)
    
    def _register_closed_position(self, position: Dict):
        """
        Registers a position closure.

        Exact data is taken from the exchange's Closed-PnL API. The record is
        matched by:
        - side (Buy/Sell),
        - size match (±50%),
        - time (not earlier than entry time minus 60 sec),
        - records that were already used earlier are discarded.

        In risk_manager we register EXACTLY ONCE per real closure.
        """
        direction = position["direction"]
        entry_price = position["entry_price"]
        qty = position["qty"]

        pnl_usdt = None
        exit_price = None
        order_id = None

        # 1) Find a matching record in Closed-PnL
        success, closed = self.executor.get_closed_pnl(self.symbol, limit=20)
        if success and closed:
            expected_side = "Buy" if direction == "long" else "Sell"

            entry_ts = None
            if isinstance(position.get("entry_time"), datetime):
                entry_ts = position["entry_time"].timestamp()
            elif isinstance(position.get("entry_time"), str):
                try:
                    entry_ts = datetime.fromisoformat(position["entry_time"]).timestamp()
                except Exception:
                    entry_ts = None

            best = None  # (rec_ts, rec)
            for rec in closed:
                try:
                    rec_id = rec.get("orderId")
                    if rec_id and rec_id in self._consumed_closed_pnl_ids:
                        continue

                    rec_side = str(rec.get("side", "")).capitalize()
                    if rec_side != expected_side:
                        continue

                    rec_qty = float(rec.get("qty", 0) or 0)
                    if qty > 0 and rec_qty > 0:
                        ratio = rec_qty / qty
                        if ratio < 0.5 or ratio > 1.5:
                            continue

                    rec_ts = int(rec.get("updatedTime", 0) or 0) / 1000.0
                    if entry_ts is not None and rec_ts < entry_ts - 60:
                        continue

                    if best is None or rec_ts > best[0]:
                        best = (rec_ts, rec)
                except (TypeError, ValueError):
                    continue

            if best is not None:
                latest = best[1]
                try:
                    pnl_usdt = float(latest.get("closedPnl", 0) or 0)
                    exit_price = float(latest.get("avgExitPrice", 0) or 0) or None
                    closed_qty = float(latest.get("qty", 0) or 0)
                    if closed_qty > 0:
                        qty = closed_qty
                    avg_entry = float(latest.get("avgEntryPrice", 0) or 0)
                    if avg_entry > 0:
                        entry_price = avg_entry

                    order_id = latest.get("orderId")
                    if order_id:
                        self._consumed_closed_pnl_ids.add(order_id)
                except (TypeError, ValueError):
                    pnl_usdt = None
                    exit_price = None

        # 2) Fallback — ticker (only if the exchange did not return data)
        if exit_price is None:
            ok, current_price = self.executor.get_ticker_price(self.symbol)
            exit_price = current_price if ok and current_price else entry_price

        position_value = qty * entry_price

        # 3) Calculate percentage
        if pnl_usdt is not None and position_value > 0:
            pnl_pct = pnl_usdt / position_value * 100.0
        else:
            if direction == "long":
                pnl_pct = (exit_price - entry_price) / entry_price * 100.0
            else:
                pnl_pct = (entry_price - exit_price) / entry_price * 100.0

        # Commissions: taker on entry and exit. Bybit closedPnl does NOT
        # include trading commissions — subtract manually so the result
        # matches backtest_engine._close_position.
        pnl_pct -= 2.0 * self._taker_fee_pct
        pnl_usdt = position_value * (pnl_pct / 100.0)

        # 4) Determine exit_reason
        exit_reason = self._infer_exit_reason(position, exit_price)

        # 5) Register EXACTLY ONCE in the risk manager
        self.risk_manager.record_closed_trade(pnl_pct, position_value)

        # 6) Log the trade
        trade = {
            "position_id": position["position_id"],
            "entry_time": position["entry_time"],
            "exit_time": datetime.now(timezone.utc),
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "qty": qty,
            "position_value_usdt": round(position_value, 2),
            "result_pct": round(pnl_pct, 2),
            "pnl_usdt": round(pnl_usdt, 2),
            "exit_reason": exit_reason,
            "stop_loss": position.get("stop_loss"),
            "take_profit": position.get("take_profit"),
            "initial_stop_loss": position.get(
                "initial_stop_loss", position.get("stop_loss")
            ),
            "initial_take_profit": position.get(
                "initial_take_profit", position.get("take_profit")
            ),
            "entry_reasons": position.get("entry_reasons", []),
        }
        self.trade_logger.log_trade(trade)

        # 7) Accounting for risk_controls (see Fix 4)
        controls = self._get_risk_controls()
        if pnl_pct <= 0:
            self._consecutive_losses += 1
            self._recent_loss_flags.append(1)
        else:
            self._consecutive_losses = 0
            self._recent_loss_flags.append(0)

        window = controls.get("loss_window_trades", 10)
        if window > 0 and len(self._recent_loss_flags) > window:
            self._recent_loss_flags = self._recent_loss_flags[-window:]

        if self._should_pause_after_loss(pnl_pct, controls):
            self._cooldown_remaining = controls["pause_candles_after_losses"]
            self._log_event(
                "COOLDOWN_ACTIVATED",
                f"Activated pause for {self._cooldown_remaining} candles",
            )

        # 8) Remove from local state and synchronously save to disk.
        #    Important: first persist the REMOVAL, then log CLOSE.
        #    If the process crashes between these steps, on the next start
        #    the position will already be gone from the file and no duplicate
        #    CLOSE will occur.
        self.position_manager.positions = [
            p for p in self.position_manager.positions
            if p["position_id"] != position["position_id"]
        ]
        self._save_state()

        # Split into two event_type: the HTML log will color profitable
        # trades green, losing trades red.
        close_event_type = "CLOSE_WIN" if pnl_pct > 0 else "CLOSE_LOSS"

        self._log_event(
            close_event_type,
            f"Position closed: {direction.upper()} @ {exit_price:.4f}, "
            f"PnL={pnl_pct:+.2f}% ({pnl_usdt:+.2f} USDT), reason={exit_reason}",
        )

    def _infer_exit_reason(self, position: Dict, exit_price: float) -> str:
        """
        Determines the exit reason by comparing the exit price with known
        position levels. The exact reason does not come in Closed-PnL,
        but this is sufficient for diagnostics.
        """
        if exit_price is None:
            return "unknown"

        def _close(a, b, tol_pct: float = 0.3) -> bool:
            if not a or not b:
                return False
            return abs(a - b) / a * 100.0 <= tol_pct

        sl = position.get("stop_loss")
        tp = position.get("take_profit")
        entry = position.get("entry_price")

        if _close(exit_price, sl):
            # If SL ≈ entry (within 0.2%) — that's breakeven
            if _close(exit_price, entry, tol_pct=0.2):
                return "breakeven"
            return "stop_loss"
        if _close(exit_price, tp):
            return "take_profit"
        return "manual_or_other"
    
    def _get_unrealized_pnl(self) -> float:
        """Returns the unrealized P&L of open positions in USDT."""
        success, exchange_positions = self.executor.get_position_info(self.symbol)
        if not success:
            return 0.0
        
        total_unrealized = 0.0
        for pos in exchange_positions or []:
            try:
                unrealized = float(pos.get("unrealisedPnl", 0))
                total_unrealized += unrealized
            except (TypeError, ValueError):
                continue
        
        return total_unrealized
    
    def _handle_daily_stop(self):
        """Handles the daily loss limit."""
        self._log_event(
            "DAILY_STOP",
            f"Daily loss limit reached. Closing positions and stopping trading.",
        )
        self._notify(
            f"[{self.symbol}] Daily loss limit reached. "
            "Positions closed, trading stopped. To resume, press 'Start trading'.",
            "negative",
        )
        
        self._close_all_positions(reason="daily_loss_limit")
        self.stop(reason="daily_loss_limit")
    
    # ----------------------------------------------------------------
    # Optimization
    # ----------------------------------------------------------------
    
    def _trigger_optimization(self):
        """Triggers strategy optimization on monthly loss limit."""
        if self.is_optimizing:
            return
        
        self.is_optimizing = True
        self.status_message = "Starting optimization..."
        
        self._log_event(
            "OPTIMIZATION_START",
            f"Monthly loss limit reached. Starting optimization "
            f"for the last {self.cfg.optimization_days_back} days.",
        )
        self._notify(
            f"[{self.symbol}] Monthly loss limit reached. "
            "Strategy optimization started. Trading paused.",
            "warning",
        )
        
        # Run optimization in a separate thread to not block monitoring
        threading.Thread(target=self._run_optimization, daemon=True).start()
    
    def _run_optimization(self):
        """Performs strategy optimization and resumes trading."""
        try:
            now = datetime.now(timezone.utc)
            start_date = now - timedelta(days=self.cfg.optimization_days_back)
            end_date = now
            
            result = self.backtest_optimizer.run(
                symbol=self.symbol,
                category=self.cfg.category,
                max_iterations=self.cfg.optimization_iterations,
                initial_capital=self.cfg.initial_capital,
                commission_pct=self._taker_fee_pct,
                csv_path=str(config.DATA_DIR / "auto_trading" / "optimization"),
                start_date=start_date,
                end_date=end_date,
                walk_forward_folds=self.cfg.walk_forward_folds,
                save_best_to_memory=True,
            )
            
            # Determine if a new strategy was saved
            best = result.get("best")
            new_strategy_saved = best is not None
            
            # Log the optimization result
            log_result = {
                "trigger_reason": "monthly_loss_limit",
                "history": result.get("history", []),
                "best": best,
                "new_strategy_saved": new_strategy_saved,
            }
            self.trade_logger.log_optimization(log_result)
            
            if new_strategy_saved:
                self._log_event(
                    "OPTIMIZATION_DONE",
                    "Optimization completed successfully. New strategy saved.",
                )
                self._notify(
                    f"[{self.symbol}] Optimization completed. "
                    "New strategy applied, trading resuming.",
                    "positive",
                )
            else:
                self._log_event(
                    "OPTIMIZATION_DONE",
                    "Optimization completed, but no suitable strategy found.",
                )
                self._notify(
                    f"[{self.symbol}] Optimization completed, but no new strategy found. "
                    "Trading continues with the old strategy.",
                    "warning",
                )
            
            # Do NOT reset the flag — optimization will trigger again only
            # on month change (_check_rollover will reset it automatically)
            self.is_optimizing = False
            self.status_message = "Trading resumed after optimization"
        
        except Exception as e:
            logger.exception(f"[{self.symbol}] Optimization error")
            self._log_event("ERROR", f"Optimization error: {e}")
            self._notify(f"[{self.symbol}] Optimization error: {e}", "negative")
            self.is_optimizing = False
    
    # ----------------------------------------------------------------
    # Closing positions
    # ----------------------------------------------------------------
    
    def _close_all_positions(self, reason: str = "manual"):
        """Closes all open positions."""
        if not self.position_manager.positions:
            return
        
        self.status_message = "Closing positions..."
        
        trades = self.position_manager.close_all_positions(reason=reason)
        
        for trade in trades:
            # Register in the risk manager
            position_value = trade["qty"] * trade["entry_price"]
            self.risk_manager.record_closed_trade(trade["result_pct"], position_value)
            
            # Log the trade
            trade["position_value_usdt"] = round(position_value, 2)
            trade["pnl_usdt"] = round(position_value * (trade["result_pct"] / 100.0), 2)
            self.trade_logger.log_trade(trade)

        self._save_state()
        self._log_event(
            "CLOSE",
            f"Closed {len(trades)} positions for reason: {reason}",
        )
    
    def close_all_public(self):
        """Public method to close all positions (called from UI)."""
        self._close_all_positions(reason="manual_from_ui")
        self._notify(f"[{self.symbol}] All positions closed manually", "info")
    
    def close_positions_public(self):
        """Synonym for close_all_public (close by coin)."""
        self.close_all_public()
    
    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------
    
    def _log_event(
        self,
        event_type: str,
        message: str,
        details: Optional[str] = None,
        group: Optional[str] = None,
    ):
        """Logs an event to file and to the local log."""
        self.trade_logger.log_event(event_type, message, details, group=group)
        self.log_lines.append(
            f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}"
        )
        # Limit the local log size
        if len(self.log_lines) > 100:
            self.log_lines = self.log_lines[-100:]
    
    def _notify(self, message: str, ntype: str):
        """Sends a notification to the UI."""
        try:
            self.notify_fn(message, ntype)
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")

    # ----------------------------------------------------------------
    # Position persistence
    # ----------------------------------------------------------------

    def _serialize_position(self, pos: Dict) -> Dict:
        """Prepares a position for JSON serialization."""
        out = {}
        for key, value in pos.items():
            # Transient fields (e.g. _last_managed_candle_ts) are not saved.
            if key.startswith("_"):
                continue
            if isinstance(value, datetime):
                out[key] = {"__dt__": value.isoformat()}
            else:
                out[key] = value
        return out

    @staticmethod
    def _deserialize_position(raw: Dict) -> Dict:
        """Reverse transformation from JSON."""
        out = {}
        for key, value in raw.items():
            if isinstance(value, dict) and "__dt__" in value:
                try:
                    out[key] = datetime.fromisoformat(value["__dt__"])
                except (TypeError, ValueError):
                    out[key] = None
            else:
                out[key] = value
        return out

    def _save_state(self):
        """
        Atomically saves the list of positions, consumed Closed-PnL ids,
        and risk_controls state (loss streak, window, cooldown).
        A write failure must not break trading.
        """
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)

            # Serialize limit orders
            serialized_orders = []
            for order in self._pending_limit_orders:
                serialized_orders.append({
                    "order_id": order["order_id"],
                    "direction": order["direction"],
                    "qty": order["qty"],
                    "price": order["price"],
                    "placed_at": order["placed_at"].isoformat(),
                    # side_spec, reasons, indicators are not serialized —
                    # on restart the order will be cancelled, position will not open.
                })

            payload = {
                "symbol": self.symbol,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "positions": [
                    self._serialize_position(p)
                    for p in self.position_manager.positions
                ],
                "pending_limit_orders": serialized_orders,
                "consumed_closed_pnl_ids": sorted(self._consumed_closed_pnl_ids),
                "risk_controls_state": {
                    "consecutive_losses": int(self._consecutive_losses),
                    "recent_loss_flags": list(self._recent_loss_flags),
                    "cooldown_remaining": int(self._cooldown_remaining),
                },
            }

            tmp_path = self._state_path.with_suffix(".json.tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            os.replace(tmp_path, self._state_path)
        except Exception:
            logger.exception(
                f"[{self.symbol}] Failed to save position state"
            )

    def _load_state(self):
        """
        Loads saved positions from disk.
        Restored positions are marked as "ours" (is_orphan is not set),
        because they have saved bybit_order_id and real SL/TP.
        """
        if not self._state_path.exists():
            return

        try:
            with open(self._state_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            positions_raw = payload.get("positions", []) or []
            loaded = [self._deserialize_position(p) for p in positions_raw]

            # Protection against broken format: a position must have required fields.
            valid = []
            for pos in loaded:
                if not all(
                    k in pos for k in (
                        "position_id", "direction", "entry_price", "qty",
                        "stop_loss", "take_profit", "entry_time",
                    )
                ):
                    logger.warning(
                        f"[{self.symbol}] Skipped invalid position record: {pos}"
                    )
                    continue
                # orphan flag should not be True for restored from disk:
                # only "our" positions get into the file.
                pos["is_orphan"] = bool(pos.get("is_orphan", False))
                valid.append(pos)

            self.position_manager.positions = valid

            consumed = payload.get("consumed_closed_pnl_ids", []) or []
            self._consumed_closed_pnl_ids = set(str(x) for x in consumed)

            # Restore risk_controls only if the file is fresh.
            # If the application was turned off longer than
            # RISK_CONTROLS_MAX_STALENESS_HOURS, the state is reset:
            # cooldown in candles does not tick when the bot is off, and the
            # loss streak "wears off" naturally.
            rc = payload.get("risk_controls_state") or {}
            rc_restored = False
            rc_stale = False
            age_hours = None

            if rc:
                saved_at_str = payload.get("saved_at")
                saved_at_dt = None
                if saved_at_str:
                    try:
                        saved_at_dt = datetime.fromisoformat(saved_at_str)
                        if saved_at_dt.tzinfo is None:
                            saved_at_dt = saved_at_dt.replace(tzinfo=timezone.utc)
                    except (TypeError, ValueError):
                        saved_at_dt = None

                if saved_at_dt is not None:
                    now_utc = datetime.now(timezone.utc)
                    age_hours = (now_utc - saved_at_dt).total_seconds() / 3600.0

                if (
                    age_hours is not None
                    and age_hours > RISK_CONTROLS_MAX_STALENESS_HOURS
                ):
                    rc_stale = True
                    logger.info(
                        f"[{self.symbol}] risk_controls_state is stale "
                        f"({age_hours:.1f}h > "
                        f"{RISK_CONTROLS_MAX_STALENESS_HOURS}h). "
                        f"Reset: consecutive_losses, recent_loss_flags, "
                        f"cooldown_remaining."
                    )
                    self._log_event(
                        "RISK_CONTROLS_RESET",
                        f"risk_controls state reset "
                        f"(pause {age_hours:.1f}h without trading).",
                    )
                else:
                    # Fresh state — restore.
                    try:
                        self._consecutive_losses = int(
                            rc.get("consecutive_losses", 0)
                        )
                    except (TypeError, ValueError):
                        self._consecutive_losses = 0
                    flags_raw = rc.get("recent_loss_flags", []) or []
                    self._recent_loss_flags = [
                        int(x) for x in flags_raw
                        if x in (0, 1, "0", "1")
                    ]
                    try:
                        self._cooldown_remaining = int(
                            rc.get("cooldown_remaining", 0)
                        )
                    except (TypeError, ValueError):
                        self._cooldown_remaining = 0
                    rc_restored = True

            # Restore limit orders: on restart we CANNOT guarantee that
            # indicators and spec are current. Therefore we simply cancel
            # all unfinished orders.
            pending_orders = payload.get("pending_limit_orders", []) or []
            for po in pending_orders:
                oid = po.get("order_id")
                if oid:
                    self.executor.cancel_order(self.symbol, oid)
                    self._log_event(
                        "ORDER_CANCELLED",
                        f"Order {oid} cancelled on restart.",
                    )
            self._pending_limit_orders = []        

            if valid or consumed or rc_restored:
                logger.info(
                    f"[{self.symbol}] Restored from state on disk: "
                    f"positions={len(valid)}, "
                    f"consumed_closed_pnl_ids={len(consumed)}, "
                    f"consecutive_losses={self._consecutive_losses}, "
                    f"cooldown_remaining={self._cooldown_remaining}"
                )
                detail = (
                    f"Restored from file: {len(valid)} positions, "
                    f"{len(consumed)} consumed Closed-PnL records"
                )
                if rc_restored:
                    detail += (
                        f", loss streak={self._consecutive_losses}, "
                        f"cooldown={self._cooldown_remaining} candles"
                    )
                elif rc_stale:
                    detail += ", risk controls reset (stale)"
                detail += "."
                self._log_event("STATE_LOADED", detail)

            # ONE-TIME diagnostics by orderId: if the saved order is already
            # cancelled/rejected by the exchange — warn the user.
            # Do this only on load, not every iteration, to not waste rate limit.
            for pos in valid:
                order_id = pos.get("bybit_order_id")
                if not order_id:
                    continue
                ok, info = self.executor.get_order_info(self.symbol, order_id)
                if ok and info:
                    status = str(info.get("orderStatus", ""))
                    if status in ("Cancelled", "Rejected", "Deactivated"):
                        logger.warning(
                            f"[{self.symbol}] Saved order {order_id} "
                            f"for position {pos['position_id']} has status "
                            f"{status} — worth checking the position manually."
                        )
                        self._log_event(
                            "STATE_WARNING",
                            f"Order {order_id} in status {status}. "
                            "Check the position manually.",
                        )
        except Exception:
            logger.exception(
                f"[{self.symbol}] Failed to load position state"
            )
    
    def get_state(self) -> Dict:
        """Returns the current state of the trader (for UI)."""
        return {
            "symbol": self.symbol,
            "is_running": self.is_running,
            "is_optimizing": self.is_optimizing,
            "status_message": self.status_message,
            
            # --- HEARTBEAT ---
            "last_iteration_utc": (
                self.last_iteration_utc.strftime("%H:%M:%S") + " UTC"
                if self.last_iteration_utc else None
            ),
            "last_candle_time": (
                str(self.last_candle_time)[:16] if self.last_candle_time else None
            ),
            "last_price": self.last_price,
            "last_indicators_summary": self.last_indicators_summary,
            "candles_processed": self.candles_processed,
            "_last_iteration_ts": (
                self.last_iteration_utc.timestamp() if self.last_iteration_utc else None
            ),
            
            "open_positions": len(self.position_manager.positions),
            "allocated_capital": round(self.allocated_capital, 2),
            "current_capital": round(self.risk_manager.current_capital, 2),
            "daily_pnl": round(self.risk_manager.daily_realized_pnl, 2),
            "monthly_pnl": round(self.risk_manager.monthly_realized_pnl, 2),
            "daily_pnl_pct": round(self.risk_manager.get_total_daily_loss_pct(0.0), 2),
            "monthly_pnl_pct": round(self.risk_manager.get_total_monthly_loss_pct(), 2),
            "daily_loss_limit": self.cfg.daily_loss_limit_pct,
            "monthly_loss_limit": self.cfg.monthly_loss_limit_pct,
            "log_lines": self.log_lines[-20:],
        }

    @staticmethod
    def _format_condition(cond: Dict, indicators: Dict) -> str:
        """Formats a condition for the log."""
        try:
            ctype = cond.get("type")
            if ctype == "compare":
                left = cond.get("left")
                op = cond.get("op", "?")
                right = cond.get("right")
                left_val = indicators.get(left, "?")
                if isinstance(right, str):
                    right_val = indicators.get(right, "?")
                    return f"{left}({left_val}) {op} {right}({right_val})"
                elif isinstance(right, dict):
                    ref = right.get("name")
                    mult = right.get("mult", 1.0)
                    ref_val = indicators.get(ref, 0) * mult
                    return f"{left}({left_val}) {op} {ref}*{mult}({ref_val:.4f})"
                else:
                    return f"{left}({left_val}) {op} {right}"
            elif ctype == "range":
                val = cond.get("value")
                vmin = cond.get("min", 0)
                vmax = cond.get("max", 100)
                val_v = indicators.get(val, "?")
                return f"{val}({val_v}) in [{vmin}, {vmax}]"
            elif ctype == "volume":
                ref = cond.get("ref")
                mult = cond.get("mult", 1.0)
                vol = indicators.get("volume", 0)
                ref_v = indicators.get(ref, 0) * mult
                return f"volume({vol}) > {ref}*{mult}({ref_v:.0f})"
            elif ctype == "slope":
                val = cond.get("value")
                direction = cond.get("dir", "?")
                return f"slope({val}) {direction}"
            else:
                return f"[{ctype}]"
        except Exception:
            return "[format_error]"


# ----------------------------------------------------------------
# Global management functions
# ----------------------------------------------------------------

def start_auto_trading(
    cfg: AutoTraderConfig,
    *,
    executor,
    fetcher,
    memory,
    backtest_engine,
    backtest_optimizer,
    ai_client,
    notify_fn,
    total_capital_usdt: float,
) -> bool:
    """
    Starts auto-trading for a coin.

    Returns True if the trader actually started.
    False — if start was rejected (already running / auto mode /
    settings do not allow opening a position).
    """
    with _active_traders_lock:
        existing = _active_traders.get(cfg.symbol)
        if existing and existing.is_running:
            return False

        trader = AutoTraderSymbol(
            cfg=cfg,
            executor=executor,
            fetcher=fetcher,
            memory=memory,
            backtest_engine=backtest_engine,
            backtest_optimizer=backtest_optimizer,
            ai_client=ai_client,
            notify_fn=notify_fn,
            total_capital_usdt=total_capital_usdt,
        )

        _active_traders[cfg.symbol] = trader

    started = trader.start()

    if not started:
        # Remove from registry so the UI does not show it as running.
        with _active_traders_lock:
            if _active_traders.get(cfg.symbol) is trader:
                _active_traders.pop(cfg.symbol, None)

    return started


def stop_auto_trading(symbol: str, close_positions: bool = False, reason: str = "manual"):
    """Stops auto-trading for a coin."""
    with _active_traders_lock:
        trader = _active_traders.get(symbol)
    
    if trader:
        trader.stop(close_positions=close_positions, reason=reason)


def close_all_positions_for_symbol(symbol: str):
    """Closes all positions for a coin (without stopping trading)."""
    with _active_traders_lock:
        trader = _active_traders.get(symbol)
    
    if trader:
        trader.close_all_public()


def close_all_positions_all_symbols():
    """Closes all positions for all coins."""
    with _active_traders_lock:
        traders = list(_active_traders.values())
    
    for trader in traders:
        trader.close_all_public()