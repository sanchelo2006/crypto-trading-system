"""
Position management: opening, closing, monitoring, exit by TP/SL.
Stores the state of open positions for each coin and synchronizes it
with the real positions on the exchange.
"""
import logging
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from trading.executor import BybitExecutor

logger = logging.getLogger(__name__)

# Snapshot of indicators at entry (for analysis)
SNAPSHOT_KEYS = [
    "close", "rsi_14", "adx_14", "volume", "avg_volume_96", "avg_volume_20",
    "sma_20", "sma_50", "macd", "macd_signal", "macd_hist", "atr_14", "vwap",
    "pivot", "resistance_1", "support_1",
]


def _to_dt(ts):
    """Converts timestamp (str / datetime / pd.Timestamp) to tz-aware datetime UTC."""
    if ts is None:
        return None
    try:
        if isinstance(ts, datetime):
            dt = ts
        else:
            dt = datetime.fromisoformat(str(ts))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _compute_bars_held(pos: Dict, row: Dict) -> Optional[int]:
    """
    Computes bars_held for live trading from entry_candle_ts.
    Returns None if the data is insufficient (position without entry_candle_ts
    or unknown candle timestamp).
    """
    entry_dt = _to_dt(pos.get("entry_candle_ts"))
    curr_dt = _to_dt(row.get("timestamp"))
    if entry_dt is None or curr_dt is None:
        return None
    try:
        delta_min = (curr_dt - entry_dt).total_seconds() / 60.0
        return int(delta_min // 15)
    except Exception:
        return None


class PositionManager:
    """
    Position manager for one coin.
    Responsible for:
    - Opening positions (market order + TP/SL placement)
    - Closing positions (by reason: SL, TP, manual, stop-out)
    - Monitoring (fetching current state from the exchange)
    - Checking exits (by management rules from strategy_spec)
    """

    def __init__(
        self,
        executor: BybitExecutor,
        symbol: str,
        category: str = "linear",
        taker_fee_pct: float = 0.055,
    ):
        self.executor = executor
        self.symbol = symbol
        self.category = category
        # Taker fee rate for deducting commissions on close.
        # Value is overwritten by AutoTraderSymbol after get_fee_rates() at start.
        self.taker_fee_pct = float(taker_fee_pct)

        # Local state: list of open positions
        # Each position is a dict with fields:
        # {
        #   "position_id": str (generated locally),
        #   "direction": "long" | "short",
        #   "entry_price": float,
        #   "qty": float,
        #   "stop_loss": float,
        #   "take_profit": float,
        #   "entry_time": datetime,
        #   "entry_reasons": List[str],
        #   "entry_snapshot": Dict,
        #   "management": Dict,
        #   "bybit_position_idx": int (for hedge mode),
        # }
        self.positions: List[Dict] = []

    def set_taker_fee_pct(self, value: float) -> None:
        """Updates the taker rate (called from AutoTraderSymbol after get_fee_rates)."""
        try:
            v = float(value)
            if v >= 0:
                self.taker_fee_pct = v
        except (TypeError, ValueError):
            pass

    def sync_with_exchange(self) -> Tuple[bool, Optional[str]]:
        """
        Synchronizes local state with the exchange.

        The only task of this method is to ADD orphan positions
        (what exists on the exchange but is not tracked locally, e.g.
        after the trader restarts). REMOVAL of local positions is NOT done here —
        that is handled by auto_trader._detect_closed_positions, which knows
        how to properly log a CLOSE event and fetch Closed-PnL.

        Returns:
            (success, error_message or None)
        """
        success, result = self.executor.get_position_info(self.symbol)
        if not success:
            return False, result

        # Collect non-zero positions from the exchange: side -> {size, avgPrice}
        exchange_by_side: Dict[str, Dict] = {}
        for p in (result or []):
            try:
                size = float(p.get("size", 0) or 0)
            except (TypeError, ValueError):
                size = 0
            if size <= 0:
                continue
            side = str(p.get("side", "")).lower()
            if side not in ("buy", "sell"):
                continue
            try:
                avg_price = float(p.get("avgPrice", 0) or 0)
            except (TypeError, ValueError):
                avg_price = 0.0
            exchange_by_side[side] = {"size": size, "avg_price": avg_price}

        # Add orphan positions: what exists on the exchange but is not tracked
        # locally. If a position was restored from a state file —
        # it is already in self.positions and will NOT be marked as orphan.
        existing_sides = {
            ("buy" if p["direction"] == "long" else "sell")
            for p in self.positions
        }
        for side_key, info in exchange_by_side.items():
            if side_key in existing_sides:
                continue

            direction = "long" if side_key == "buy" else "short"
            orphan = {
                "position_id": (
                    f"{self.symbol}_orphan_"
                    f"{int(datetime.now(timezone.utc).timestamp())}_{side_key}"
                ),
                "direction": direction,
                "entry_price": info["avg_price"] or 0.0,
                "qty": info["size"],
                "stop_loss": 0.0,        # unknown, do not touch
                "take_profit": 0.0,      # unknown, do not touch
                "entry_time": datetime.now(timezone.utc),
                "entry_reasons": [],
                "entry_snapshot": {},
                "management": {},
                "is_orphan": True,
                "tpsl_error": "external_position_detected",
            }
            logger.warning(
                f"[{self.symbol}] External position detected on the exchange "
                f"(side={side_key}, size={info['size']}). Added as orphan."
            )
            self.positions.append(orphan)

        return True, None

    def open_position(
        self,
        direction: str,  # "long" | "short"
        qty: float,
        current_price: float,
        side_spec: Dict,
        entry_reasons: List[str],
        indicators: Dict,
        entry_candle_ts=None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Opens a new position.
        Steps:
        1. Places a market order
        2. Computes TP/SL
        3. Sets TP/SL via set_trading_stop
        4. Saves the position to local state

        Returns:
            (success, error_message or None)
        """
        # 0. Safety net: if a position for this symbol ALREADY exists on the exchange —
        #    do not open a new one. Protects against double entry when local state
        #    is out of sync with the exchange (e.g. after trader restart
        #    or after an exception in the middle of open_position).
        success_check, exchange_positions = self.executor.get_position_info(self.symbol)
        if success_check and exchange_positions:
            for p in exchange_positions:
                try:
                    size = float(p.get("size", 0) or 0)
                except (TypeError, ValueError):
                    size = 0
                if size > 0:
                    logger.warning(
                        f"[{self.symbol}] Open rejected: a position already exists "
                        f"on the exchange (side={p.get('side')}, size={size})."
                    )
                    return False, "position_already_exists_on_exchange"

        # Order direction for opening a position
        side = "Buy" if direction == "long" else "Sell"

        # 1. Open the position with a market order
        success, result = self.executor.place_market_order(
            symbol=self.symbol,
            side=side,
            qty=qty,
            reduce_only=False,
        )
        if not success:
            logger.error(f"[{self.symbol}] Failed to open position: {result}")
            return False, result

        order_id = result.get("orderId")

        # Learn the ACTUAL fill (partial fill + slippage)
        success_o, order_info = self.executor.get_order_info(self.symbol, order_id)
        if success_o and order_info:
            filled_qty = float(order_info.get("cumExecQty", 0) or 0)
            avg_price = float(order_info.get("avgPrice", 0) or 0)
            if filled_qty > 0:
                qty = filled_qty          # actual size, not calculated
            if avg_price > 0:
                current_price = avg_price  # actual average entry price

        # if nothing was filled — do not open the position
        if qty <= 0:
            logger.error(f"[{self.symbol}] Order was not filled")
            return False, "order_not_filled"

        # 2. Compute TP/SL
        stop_loss, take_profit = self._compute_tp_sl(
            direction, current_price, side_spec, indicators
        )

        # 3. Set TP/SL
        tpsl_error = None
        success_sl, error_sl = self.executor.set_trading_stop(
            symbol=self.symbol,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        if not success_sl:
            tpsl_error = error_sl
            logger.warning(
                f"[{self.symbol}] Failed to set TP/SL: {error_sl}. "
                "Position is open but without protection."
            )

        # 4. Save to local state
        snapshot = {}
        for k in SNAPSHOT_KEYS:
            v = indicators.get(k)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                snapshot[k] = round(float(v), 6)

        position = {
            "position_id": f"{self.symbol}_{int(datetime.now(timezone.utc).timestamp())}",
            "direction": direction,
            "entry_price": current_price,
            "qty": qty,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            # Remember the INITIAL levels — they do not change
            # even when trailing/breakeven move stop_loss.
            "initial_stop_loss": stop_loss,
            "initial_take_profit": take_profit,
            "entry_time": datetime.now(timezone.utc),
            # Timestamp of the candle on which the signal fired. Needed for
            # computing bars_held in live trading (time-stop).
            "entry_candle_ts": (
                str(entry_candle_ts) if entry_candle_ts is not None else None
            ),
            "entry_reasons": entry_reasons,
            "entry_snapshot": snapshot,
            "management": side_spec.get("management", {}),
            "bybit_order_id": order_id,
            "tpsl_error": tpsl_error,
        }
        self.positions.append(position)

        logger.info(
            f"[{self.symbol}] Position opened: {direction.upper()} {qty} @ {current_price:.4f}, "
            f"SL={stop_loss:.4f}, TP={take_profit:.4f}"
        )
        return True, None

    def open_position_limit(
        self,
        direction: str,
        qty: float,
        side_spec: Dict,
        entry_reasons: List[str],
        indicators: Dict,
        entry_candle_ts=None,
        price_offset_ticks: int = 0,
        timeout_seconds: int = 900,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Opens a position with a limit order (maker).

        Algorithm:
        1. Gets best bid/ask from the order book.
        2. Computes the limit price:
           - Long: best_bid + offset (buy at bid or slightly above)
           - Short: best_ask - offset (sell at ask or slightly below)
        3. Places a Post-Only limit order.
        4. Returns order_id for monitoring.

        The position is NOT added to self.positions immediately.
        The calling code (auto_trader) must monitor the fill and
        call _finalize_limit_entry() after execution.

        Args:
            direction: "long" or "short"
            qty: size in contracts
            side_spec: side spec (for TP/SL computation)
            entry_reasons: entry reasons
            indicators: current indicators
            entry_candle_ts: timestamp of the signal candle
            price_offset_ticks: price shift in ticks from the best price
            timeout_seconds: fill wait timeout

        Returns:
            (success, order_id or error, limit_price)
        """
        # 0. Check whether a position already exists on the exchange
        success_check, exchange_positions = self.executor.get_position_info(self.symbol)
        if success_check and exchange_positions:
            for p in exchange_positions:
                try:
                    size = float(p.get("size", 0) or 0)
                except (TypeError, ValueError):
                    size = 0
                if size > 0:
                    return False, "position_already_exists_on_exchange", None

        # 1. Get best bid/ask
        success, bid_ask = self.executor.get_best_bid_ask(self.symbol)
        if not success:
            return False, f"No order book: {bid_ask}", None
        best_bid, best_ask = bid_ask

        # 2. Get tickSize for rounding
        success_info, info = self.executor.get_instrument_info(self.symbol)
        if not success_info or not info:
            return False, f"No instrument parameters: {info}", None
        tick_size = info.get("tickSize", 0.01)

        # 3. Compute the limit price
        if direction == "long":
            # Buy at bid (maker). If the price rises — it will not be filled.
            raw_price = best_bid + (price_offset_ticks * tick_size)
            limit_price = self.executor.round_price(raw_price, tick_size)
            side = "Buy"
        else:
            # Sell at ask (maker). If the price falls — it will not be filled.
            raw_price = best_ask - (price_offset_ticks * tick_size)
            limit_price = self.executor.round_price(raw_price, tick_size)
            side = "Sell"

        # 4. Place a Post-Only limit order
        success, result = self.executor.place_limit_order(
            symbol=self.symbol,
            side=side,
            qty=qty,
            price=limit_price,
            post_only=True,
        )
        if not success:
            if isinstance(result, dict) and result.get("error") == "post_only_rejected":
                # The order would become taker. Try with a more passive price.
                logger.warning(
                    f"[{self.symbol}] Post-Only rejected. Trying more passive."
                )
                # Step back 1 tick deeper into the order book
                if direction == "long":
                    limit_price -= tick_size
                else:
                    limit_price += tick_size
                limit_price = self.executor.round_price(limit_price, tick_size)
                success, result = self.executor.place_limit_order(
                    symbol=self.symbol,
                    side=side,
                    qty=qty,
                    price=limit_price,
                    post_only=True,
                )
                if not success:
                    return False, f"Second Post-Only rejected: {result}", None
            else:
                return False, str(result), None

        order_id = result.get("orderId")
        if not order_id:
            return False, "No orderId in response", None

        logger.info(
            f"[{self.symbol}] Limit order placed: {direction} {qty} @ {limit_price} "
            f"(orderId={order_id}, timeout={timeout_seconds}s)"
        )
        return True, order_id, str(limit_price)

    def finalize_limit_entry(
        self,
        direction: str,
        qty: float,
        entry_price: float,
        side_spec: Dict,
        entry_reasons: List[str],
        indicators: Dict,
        entry_candle_ts=None,
        order_id: str = "",
    ) -> Tuple[bool, Optional[str]]:
        """
        Completes opening a position after a limit order is filled.
        Called from auto_trader after fill confirmation.
        Sets TP/SL and adds the position to the local list.

        Returns:
            (success, error_message or None)
        """
        # Compute TP/SL
        stop_loss, take_profit = self._compute_tp_sl(
            direction, entry_price, side_spec, indicators
        )

        # Set TP/SL on the exchange
        tpsl_error = None
        success_sl, error_sl = self.executor.set_trading_stop(
            symbol=self.symbol,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        if not success_sl:
            tpsl_error = error_sl
            logger.warning(
                f"[{self.symbol}] Failed to set TP/SL: {error_sl}. "
                "Position is open without protection."
            )

        # Build the indicator snapshot
        snapshot = {}
        for k in SNAPSHOT_KEYS:
            v = indicators.get(k)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                snapshot[k] = round(float(v), 6)

        position = {
            "position_id": f"{self.symbol}_{int(datetime.now(timezone.utc).timestamp())}",
            "direction": direction,
            "entry_price": entry_price,
            "qty": qty,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "initial_stop_loss": stop_loss,
            "initial_take_profit": take_profit,
            "entry_time": datetime.now(timezone.utc),
            "entry_candle_ts": (
                str(entry_candle_ts) if entry_candle_ts is not None else None
            ),
            "entry_reasons": entry_reasons,
            "entry_snapshot": snapshot,
            "management": side_spec.get("management", {}),
            "bybit_order_id": order_id,
            "tpsl_error": tpsl_error,
            "entry_type": "limit",
        }
        self.positions.append(position)

        logger.info(
            f"[{self.symbol}] Limit position opened: {direction.upper()} "
            f"{qty} @ {entry_price:.4f}, SL={stop_loss:.4f}, TP={take_profit:.4f}"
        )
        return True, None

    def close_position(
        self,
        position: Dict,
        reason: str,
        current_price: Optional[float] = None,
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Closes a position.

        Closes the REAL position size from the exchange, not the one recorded
        locally (a market order could have been partially filled).
        """
        direction = position["direction"]
        qty = position["qty"]
        expected_side = "buy" if direction == "long" else "sell"

        # 1. Learn the actual position size on the exchange
        exchange_qty = None
        success_pos, exchange_positions = self.executor.get_position_info(self.symbol)
        if success_pos and exchange_positions:
            for p in exchange_positions:
                if str(p.get("side", "")).lower() == expected_side:
                    try:
                        exchange_qty = float(p.get("size", 0) or 0)
                    except (TypeError, ValueError):
                        exchange_qty = None
                    break

        if exchange_qty is not None:
            if exchange_qty <= 0:
                # The position is already gone on the exchange — do not send a second order.
                logger.warning(
                    f"[{self.symbol}] Position {position['position_id']} is absent "
                    "on the exchange, not sending a reduce-only order."
                )
                self.positions = [
                    p for p in self.positions
                    if p["position_id"] != position["position_id"]
                ]
                return False, "already_closed_on_exchange"
            qty = exchange_qty  # close exactly as much as actually exists

        # 2. Exit price — fetch it right before sending the order if not provided
        if current_price is None:
            ok, price = self.executor.get_ticker_price(self.symbol)
            if ok and price:
                current_price = price

        # 3. Send a reduce-only market order
        side = "Sell" if direction == "long" else "Buy"
        success, result = self.executor.place_market_order(
            symbol=self.symbol,
            side=side,
            qty=qty,
            reduce_only=True,
        )
        if not success:
            logger.error(f"[{self.symbol}] Failed to close position: {result}")
            return False, result

        # 4. Compute PnL
        entry_price = position["entry_price"]
        exit_price = current_price if current_price else entry_price

        if direction == "long":
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100

        # Commissions: taker on entry and exit. The same is done in
        # backtest_engine._close_position — so the result matches.
        pnl_pct -= 2.0 * self.taker_fee_pct

        trade = {
            "symbol": self.symbol,
            "position_id": position["position_id"],
            "entry_time": position["entry_time"],
            "exit_time": datetime.now(timezone.utc),
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "qty": qty,
            "result_pct": round(pnl_pct, 2),
            "exit_reason": reason,
            "stop_loss": position.get("stop_loss"),
            "take_profit": position.get("take_profit"),
            "initial_stop_loss": position.get(
                "initial_stop_loss", position.get("stop_loss")
            ),
            "initial_take_profit": position.get(
                "initial_take_profit", position.get("take_profit")
            ),
            "entry_reasons": position.get("entry_reasons", []),
            "entry_snapshot": position.get("entry_snapshot", {}),
        }

        # Remove from local state
        self.positions = [p for p in self.positions if p["position_id"] != position["position_id"]]

        logger.info(
            f"[{self.symbol}] Position closed: {direction.upper()} {qty} @ {exit_price:.4f}, "
            f"PnL={pnl_pct:+.2f}%, reason={reason}"
        )
        return True, trade

    def close_all_positions(self, reason: str = "manual_stop") -> List[Dict]:
        """
        Closes all open positions.

        Returns:
            List of closed trades (trade_dict)
        """
        trades = []
        for position in list(self.positions):
            success, result = self.close_position(position, reason)
            if success:
                trades.append(result)
            else:
                logger.error(
                    f"[{self.symbol}] Failed to close position {position['position_id']}: {result}"
                )
        return trades

    def check_exits(
        self,
        current_indicators: Dict,
    ) -> List[Tuple[Dict, str, float]]:
        """
        Checks exit conditions for all open positions.

        Returns:
            List of [(position, exit_reason, exit_price), ...]
        """
        exits = []
        for position in self.positions:
            exit_price, reason = self._check_exit(position, current_indicators)
            if exit_price is not None:
                exits.append((position, reason, exit_price))
        return exits

    @staticmethod
    def _check_hard_exit(pos: Dict, row: Dict) -> Tuple[Optional[float], str]:
        """Hard exits (SL/TP). Used only in backtest —
        in live trading SL/TP is set by the exchange."""
        if pos["direction"] == "long":
            if row.get("low") is not None and row["low"] <= pos["stop_loss"]:
                return pos["stop_loss"], "stop_loss"
            if row.get("high") is not None and row["high"] >= pos["take_profit"]:
                return pos["take_profit"], "take_profit"
        else:
            if row.get("high") is not None and row["high"] >= pos["stop_loss"]:
                return pos["stop_loss"], "stop_loss"
            if row.get("low") is not None and row["low"] <= pos["take_profit"]:
                return pos["take_profit"], "take_profit"
        return None, ""

    @staticmethod
    def _check_management_exits(pos: Dict, row: Dict) -> Tuple[Optional[float], str]:
        """
        Soft exits by management: time_stop, exit_on_level_failure,
        exit_on_trend_loss, exit_on_macd_flip.
        Logic is identical to backtest_engine._check_exit. For time_stop
        it uses:
          - pos["bars_held"] if present (backtest path);
          - otherwise bars_held is computed from entry_candle_ts and row["timestamp"]
            (live trading path).
        """
        mgmt = pos.get("management", {}) or {}
        close_price = row.get("close")
        if close_price is None:
            return None, ""

        # --- Time-stop ---
        max_hold = mgmt.get("max_hold_candles")
        if max_hold:
            try:
                max_hold = int(max_hold)
                bars_held = pos.get("bars_held")
                if bars_held is None:
                    bars_held = _compute_bars_held(pos, row)
                if bars_held is not None and bars_held >= max_hold:
                    return float(close_price), "time_stop"
            except (TypeError, ValueError):
                pass

        buffer_pct = float(mgmt.get("level_failure_buffer_pct", 1.0)) / 100.0

        def _valid(v):
            return v is not None and not (isinstance(v, float) and math.isnan(v))

        if pos["direction"] == "long":
            if mgmt.get("exit_on_level_failure"):
                lvl = row.get(mgmt.get("level_field", "resistance_1"))
                if _valid(lvl) and close_price < lvl * (1 - buffer_pct):
                    return float(close_price), "level_failure"
            if mgmt.get("exit_on_trend_loss"):
                sma50 = row.get("sma_50")
                if _valid(sma50) and close_price < sma50:
                    return float(close_price), "trend_loss"
            if mgmt.get("exit_on_macd_flip"):
                mh = row.get("macd_hist")
                if _valid(mh) and mh < 0:
                    return float(close_price), "macd_flip"
        else:
            if mgmt.get("exit_on_level_failure"):
                lvl = row.get(mgmt.get("level_field", "support_1"))
                if _valid(lvl) and close_price > lvl * (1 + buffer_pct):
                    return float(close_price), "level_failure"
            if mgmt.get("exit_on_trend_loss"):
                sma50 = row.get("sma_50")
                if _valid(sma50) and close_price > sma50:
                    return float(close_price), "trend_loss"
            if mgmt.get("exit_on_macd_flip"):
                mh = row.get("macd_hist")
                if _valid(mh) and mh > 0:
                    return float(close_price), "macd_flip"
        return None, ""

    def _check_exit(self, pos: Dict, row: Dict) -> Tuple[Optional[float], str]:
        """Full check (used in backtest and tests)."""
        price, reason = self._check_hard_exit(pos, row)
        if price is not None:
            return price, reason
        return self._check_management_exits(pos, row)

    def check_soft_exit(self, position: Dict, candle: Dict) -> Tuple[Optional[float], str]:
        """
        Public method for the live trader: soft exits only.
        SL/TP is handled by the exchange, they cannot be re-checked.
        """
        return self._check_management_exits(position, candle)

    def _compute_tp_sl(
        self,
        direction: str,
        entry: float,
        side_spec: Dict,
        indicators: Dict,
    ) -> Tuple[float, float]:
        """
        Computes TP and SL for a new position.
        Priority — ATR stops (as in backtest).
        Prices are rounded to the instrument tickSize: otherwise Bybit will reject
        set_trading_stop, and the position will remain unprotected.
        """
        # Use 4H ATR if available (swing strategies),
        # otherwise standard 15m ATR
        atr = indicators.get("atr_14_4h") or indicators.get("atr_14", 0)
        atr_valid = (
            atr is not None
            and not (isinstance(atr, float) and math.isnan(atr))
            and atr > 0
        )

        sl_atr_mult = side_spec.get("stop_loss_atr_mult")
        tp_atr_mult = side_spec.get("take_profit_atr_mult")

        stop_dist = None
        take_dist = None
        if atr_valid and sl_atr_mult is not None:
            try:
                stop_dist = float(sl_atr_mult) * float(atr)
            except (TypeError, ValueError):
                stop_dist = None
        if atr_valid and tp_atr_mult is not None:
            try:
                take_dist = float(tp_atr_mult) * float(atr)
            except (TypeError, ValueError):
                take_dist = None

        # Fallback — fixed percentages
        if stop_dist is None:
            sl_pct = float(side_spec.get("stop_loss_pct", 2.0)) / 100.0
            stop_dist = entry * sl_pct

        # Fallback — if ATR did not produce a value, take percentages from spec.
        if take_dist is None:
            tp_list = side_spec.get("take_profit_pct", [3.0]) or [3.0]
            tp_pct = float(tp_list[0]) / 100.0
            take_dist = entry * tp_pct

        # stop/take computation ALWAYS runs — regardless of whether
        # take_dist came from ATR or from percentages.
        if direction == "long":
            stop = entry - stop_dist
            take = entry + take_dist
        else:
            stop = entry + stop_dist
            take = entry - take_dist

        # PROTECTION: limit SL and TP to reasonable percentages of entry.
        # ATR can be abnormally inflated due to outliers in OHLCV (especially on testnet),
        # and then SL/TP fly 10-20x further than normal. Better to have
        # a conservative stop than to get -16% on one trade.
        MAX_STOP_PCT = 2.5    # stop no further than 2.5% from entry
        MAX_TAKE_PCT = 10.0   # take no further than 10% from entry
        if direction == "long":
            stop = max(stop, entry * (1 - MAX_STOP_PCT / 100.0))
            take = min(take, entry * (1 + MAX_TAKE_PCT / 100.0))
        else:
            stop = min(stop, entry * (1 + MAX_STOP_PCT / 100.0))
            take = max(take, entry * (1 - MAX_TAKE_PCT / 100.0))

        # Round to the instrument tickSize.
        success, info = self.executor.get_instrument_info(self.symbol)
        if success and info:
            tick = info.get("tickSize")
            if tick:
                stop = self.executor.round_price(stop, tick)
                take = self.executor.round_price(take, tick)
        return stop, take