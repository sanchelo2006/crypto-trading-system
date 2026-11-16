"""
Order execution via Bybit Unified Trading API for futures trading.

Specifics of linear perpetuals (USDT-margin):
- Category: linear
- Leverage is set separately
- Margin mode: isolated (default) or cross
- Position mode: one-way (position in one direction)
- Position size is specified in contracts (qty), not in USDT
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from decimal import Decimal, ROUND_DOWN, ROUND_UP

from pybit.unified_trading import HTTP

import config

logger = logging.getLogger(__name__)


class BybitError(Exception):
    """Exception for Bybit API errors."""
    pass


class BybitExecutor:
    """
    Low-level adapter for working with Bybit futures (linear perpetual).
    
    All methods return (success: bool, result/error).
    """
    
    def __init__(
        self,
        testnet: bool = False,
        api_key: str = None,
        api_secret: str = None,
        default_leverage: int = 10,
        margin_mode: str = "ISOLATED_MARGIN",  # or "REGULAR_MARGIN" (cross)
    ):
        self.testnet = testnet
        self.api_key = api_key or config.BYBIT_API_KEY
        self.api_secret = api_secret or config.BYBIT_API_SECRET
        self.default_leverage = default_leverage
        self.margin_mode = margin_mode
        
        if not self.api_key or not self.api_secret:
            raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set")
        
        self.session = HTTP(
            testnet=testnet,
            api_key=self.api_key,
            api_secret=self.api_secret,
        )
        
        # Cache of instrument parameters (tickSize, qtyStep, minOrderQty)
        self._instruments_cache: Dict[str, Dict] = {}
        
        logger.info(
            f"BybitExecutor initialized ({'testnet' if testnet else 'mainnet'}), "
            f"leverage={default_leverage}, margin_mode={margin_mode}"
        )
    
    # ----------------------------------------------------------------
    # Instrument information
    # ----------------------------------------------------------------
    
    def get_instrument_info(self, symbol: str) -> Tuple[bool, Optional[Dict]]:
        """
        Gets trading instrument parameters.

        Returns:
            (success, info_dict or error)

        info_dict contains:
            - tickSize: minimum price step
            - qtyStep: minimum quantity step
            - minOrderQty: minimum order size
            - maxOrderQty: maximum order size
            - minNotionalValue: minimum order value in USDT

        Fault tolerance logic:
        1. If the cache already has data — return immediately, no network call.
        2. On miss — retry with exponential backoff.
           Bybit testnet periodically returns ErrCode 10016 (Internal System Error)
           and 10000 (Server Timeout) — transient errors, usually pass
           on the 2-3rd attempt.
        3. If all attempts failed but the cache HAS data (e.g. received earlier,
           and now the network is down again) — return it.
           tickSize/qtyStep for an instrument rarely change,
           so old data is absolutely valid.
        4. If there is nothing at all — return (False, error).
        """
        # 1. Fast path — data in cache
        if symbol in self._instruments_cache:
            return True, self._instruments_cache[symbol]

        last_error = "Unknown error"

        # 2. Retry with exponential backoff
        for attempt in range(3):
            try:
                response = self.session.get_instruments_info(
                    category="linear",
                    symbol=symbol,
                )

                if response.get("retCode") != 0:
                    last_error = response.get("retMsg", "Unknown error")
                    logger.warning(
                        f"Info for {symbol}: retCode={response.get('retCode')}, "
                        f"{last_error} (attempt {attempt + 1}/3)"
                    )
                    if attempt < 2:
                        time.sleep(1.0 * (2 ** attempt))
                    continue

                instruments = response.get("result", {}).get("list", [])
                if not instruments:
                    last_error = f"Instrument {symbol} not found"
                    logger.warning(
                        f"Info for {symbol}: empty instrument list"
                    )
                    return False, last_error

                info = instruments[0]
                lot_size_filter = info.get("lotSizeFilter", {})
                price_filter = info.get("priceFilter", {})

                instrument_data = {
                    "symbol": symbol,
                    "tickSize": float(price_filter.get("tickSize", "0.01")),
                    "qtyStep": float(lot_size_filter.get("qtyStep", "0.001")),
                    "minOrderQty": float(lot_size_filter.get("minOrderQty", "0.001")),
                    "maxOrderQty": float(lot_size_filter.get("maxOrderQty", "1000000")),
                    "minNotionalValue": float(lot_size_filter.get("minNotionalValue", "5")),
                }

                self._instruments_cache[symbol] = instrument_data

                logger.info(
                    f"Info for {symbol}: tickSize={instrument_data['tickSize']}, "
                    f"qtyStep={instrument_data['qtyStep']}, "
                    f"minOrderQty={instrument_data['minOrderQty']}"
                )

                return True, instrument_data

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Info for {symbol}: exception {type(e).__name__} "
                    f"({last_error}), attempt {attempt + 1}/3"
                )
                if attempt < 2:
                    time.sleep(1.0 * (2 ** attempt))

        # 3. All attempts failed.
        # If the cache has old data — return it.
        if symbol in self._instruments_cache:
            logger.warning(
                f"Info for {symbol}: failed to get fresh data "
                f"({last_error}). Using cache."
            )
            return True, self._instruments_cache[symbol]

        # 4. Nothing at all
        logger.error(
            f"Info for {symbol}: final failure after 3 attempts. "
            f"Reason: {last_error}"
        )
        return False, last_error
    
    # ----------------------------------------------------------------
    # Balance and positions
    # ----------------------------------------------------------------
    
    def get_wallet_balance(self, account_type: str = "UNIFIED") -> Tuple[bool, Optional[Dict]]:
        """
        Gets wallet balance.
        
        Returns:
            (success, balance_dict or error_message)
        """
        try:
            response = self.session.get_wallet_balance(accountType=account_type)
            
            if response.get("retCode") != 0:
                error = response.get("retMsg", "Unknown error")
                logger.error(f"Error getting balance: {error}")
                return False, error
            
            result = response.get("result", {})
            return True, result
        
        except Exception as e:
            logger.exception("Exception getting balance")
            return False, str(e)
    
    def get_usdt_balance(self) -> Tuple[bool, Optional[float]]:
        """
        Returns the wallet USDT balance.
        availableToWithdraw is deprecated for UTA 2.0 (always ""), so
        take the first non-empty from availableToWithdraw → walletBalance → equity.
        """
        success, result = self.get_wallet_balance()
        if not success:
            return False, None
        try:
            accounts = result.get("list") or []
            if not accounts:
                return True, 0.0
            for coin in accounts[0].get("coin", []) or []:
                if coin.get("coin") != "USDT":
                    continue
                for key in ("availableToWithdraw", "walletBalance", "equity"):
                    raw = coin.get(key)
                    if raw in (None, ""):
                        continue
                    try:
                        value = float(raw)
                        #previously there was terminal logging here - but it pollutes
                        return True, value
                    except (TypeError, ValueError):
                        continue
                return True, 0.0
            return True, 0.0
        except Exception:
            logger.exception("Error parsing USDT balance")
            return False, None
    
    def get_position_info(self, symbol: str) -> Tuple[bool, Optional[List[Dict]]]:
        """
        Gets information about open positions for the symbol.
        
        Returns:
            (success, list_of_positions or error)
        """
        try:
            response = self.session.get_positions(
                category="linear",
                symbol=symbol,
            )
            
            if response.get("retCode") != 0:
                error = response.get("retMsg", "Unknown error")
                logger.error(f"Error getting positions {symbol}: {error}")
                return False, error
            
            positions = response.get("result", {}).get("list", [])
            return True, positions
        
        except Exception as e:
            logger.exception(f"Exception getting positions {symbol}")
            return False, str(e)
    
    # ----------------------------------------------------------------
    # Leverage and margin setup
    # ----------------------------------------------------------------
    
    def set_leverage(self, symbol: str, leverage: int) -> Tuple[bool, Optional[str]]:
        try:
            response = self.session.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage),
            )
            ret_code = response.get("retCode")
            if ret_code == 0 or ret_code == 110043:
                logger.info(f"Leverage for {symbol}: {leverage}x")
                return True, "ok"
            error = response.get("retMsg", "Unknown error")
            logger.error(f"Error setting leverage {symbol}: {error}")
            return False, error
        except Exception as e:
            # pybit throws InvalidRequestError; 110043 = leverage already set
            code = getattr(e, "status_code", None)
            if code == 110043 or "leverage not modified" in str(e).lower():
                logger.info(f"Leverage for {symbol} already set: {leverage}x")
                return True, "ok"
            logger.exception(f"Exception setting leverage {symbol}")
            return False, str(e)
    
    def switch_margin_mode(self, symbol: str, margin_mode: str) -> Tuple[bool, Optional[str]]:
        """
        Switches margin mode.
        
        Args:
            symbol: trading pair
            margin_mode: "ISOLATED_MARGIN" or "REGULAR_MARGIN" (cross)
        
        Returns:
            (success, result or error)
        """
        try:
            # Important: switching is only possible without open positions
            response = self.session.switch_margin_mode(
                category="linear",
                symbol=symbol,
                tradeMode=0 if margin_mode == "REGULAR_MARGIN" else 1,
                buyLeverage=str(self.default_leverage),
                sellLeverage=str(self.default_leverage),
            )
            
            if response.get("retCode") != 0:
                error = response.get("retMsg", "Unknown error")
                # If the error is "leverage not modified" — that's normal (already set)
                if "leverage not modified" in error.lower():
                    logger.info(f"Margin mode for {symbol} already set: {margin_mode}")
                    return True, "ok"
                logger.error(f"Error switching margin mode {symbol}: {error}")
                return False, error
            
            logger.info(f"Margin mode set for {symbol}: {margin_mode}")
            return True, "ok"
        
        except Exception as e:
            logger.exception(f"Exception switching margin mode {symbol}")
            return False, str(e)
    
    # ----------------------------------------------------------------
    # Opening and closing positions
    # ----------------------------------------------------------------
    
    def place_market_order(
        self,
        symbol: str,
        side: str,  # "Buy" or "Sell"
        qty: float,
        reduce_only: bool = False,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Places a market order on the futures market.
        
        Args:
            symbol: trading pair
            side: "Buy" (long) or "Sell" (short/close long)
            qty: number of contracts
            reduce_only: True to close a position
            stop_loss: stop loss price (optional)
            take_profit: take profit price (optional)
        
        Returns:
            (success, order_result or error)
        """
        try:
            params = {
                "category": "linear",
                "symbol": symbol,
                "side": side,
                "orderType": "Market",
                "qty": str(qty),
                "reduceOnly": reduce_only,
                "positionIdx": 0,  # one-way mode
            }
            
            # Add TP/SL to the order (if specified)
            if stop_loss is not None:
                params["stopLoss"] = str(stop_loss)
            
            if take_profit is not None:
                params["takeProfit"] = str(take_profit)
            
            response = self.session.place_order(**params)
            
            if response.get("retCode") != 0:
                error = response.get("retMsg", "Unknown error")
                logger.error(f"Error placing market order {symbol}: {error}")
                return False, error
            
            result = response.get("result", {})
            logger.info(
                f"Market order placed: {symbol} {side} {qty} "
                f"(orderId={result.get('orderId')})"
            )
            return True, result
        
        except Exception as e:
            logger.exception(f"Exception placing market order {symbol}")
            return False, str(e)

    def place_limit_order(
        self,
        symbol: str,
        side: str,  # "Buy" or "Sell"
        qty: float,
        price: float,
        reduce_only: bool = False,
        post_only: bool = True,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Places a limit order.

        When post_only=True the order is guaranteed to execute as maker.
        If it would execute as taker — the exchange rejects it (code 110016).

        Args:
            symbol: trading pair
            side: "Buy" (long) or "Sell" (short)
            qty: number of contracts
            price: limit order price
            reduce_only: True to close a position
            post_only: True = maker only (rejected if it would become taker)
            stop_loss: stop loss price (optional)
            take_profit: take profit price (optional)

        Returns:
            (success, order_result or error)
        """
        try:
            params = {
                "category": "linear",
                "symbol": symbol,
                "side": side,
                "orderType": "Limit",
                "qty": str(qty),
                "price": str(price),
                "reduceOnly": reduce_only,
                "positionIdx": 0,
                "timeInForce": "PostOnly" if post_only else "GTC",
            }

            if stop_loss is not None:
                params["stopLoss"] = str(stop_loss)
            if take_profit is not None:
                params["takeProfit"] = str(take_profit)

            response = self.session.place_order(**params)

            if response.get("retCode") != 0:
                error = response.get("retMsg", "Unknown error")
                ret_code = response.get("retCode")

                # 110016 = order would have executed as taker → rejected
                if ret_code == 110016:
                    logger.warning(
                        f"Limit order {symbol} rejected (would become taker): {error}"
                    )
                    return False, {"error": "post_only_rejected", "detail": error}

                logger.error(
                    f"Error placing limit order {symbol}: {error}"
                )
                return False, error

            result = response.get("result", {})
            logger.info(
                f"Limit order placed: {symbol} {side} {qty} @ {price} "
                f"(orderId={result.get('orderId')}, postOnly={post_only})"
            )
            return True, result

        except Exception as e:
            logger.exception(f"Exception placing limit order {symbol}")
            return False, str(e)
    
    def cancel_order(
        self,
        symbol: str,
        order_id: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Cancels an open order by orderId.

        Returns:
            (success, error or None)
        """
        try:
            response = self.session.cancel_order(
                category="linear",
                symbol=symbol,
                orderId=order_id,
            )
            if response.get("retCode") != 0:
                error = response.get("retMsg", "Unknown error")
                logger.warning(f"Failed to cancel order {order_id}: {error}")
                return False, error
            logger.info(f"Order {order_id} cancelled for {symbol}")
            return True, None
        except Exception as e:
            logger.exception(f"Exception cancelling order {order_id}")
            return False, str(e)

    def get_open_orders(
        self,
        symbol: str,
    ) -> Tuple[bool, Optional[List[Dict]]]:
        """
        Returns the list of open orders for the symbol.

        Returns:
            (success, list_of_orders or error)
        """
        try:
            response = self.session.get_open_orders(
                category="linear",
                symbol=symbol,
            )
            if response.get("retCode") != 0:
                error = response.get("retMsg", "Unknown error")
                return False, error
            orders = response.get("result", {}).get("list", [])
            return True, orders
        except Exception as e:
            logger.exception(f"Exception getting orders {symbol}")
            return False, str(e)

    def get_best_bid_ask(
        self,
        symbol: str,
    ) -> Tuple[bool, Optional[Tuple[float, float]]]:
        """
        Returns the best bid and ask prices from the order book.

        Returns:
            (success, (best_bid, best_ask) or error)
        """
        try:
            response = self.session.get_tickers(
                category="linear",
                symbol=symbol,
            )
            if response.get("retCode") != 0:
                error = response.get("retMsg", "Unknown error")
                return False, error

            tickers = response.get("result", {}).get("list", [])
            if not tickers:
                return False, f"No ticker data for {symbol}"

            ticker = tickers[0]
            best_bid = float(ticker.get("bid1Price", 0))
            best_ask = float(ticker.get("ask1Price", 0))

            if best_bid <= 0 or best_ask <= 0:
                return False, f"Invalid order book for {symbol}"

            return True, (best_bid, best_ask)

        except Exception as e:
            logger.exception(f"Exception getting order book {symbol}")
            return False, str(e)

    def set_trading_stop(
        self,
        symbol: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Sets TP/SL for an open position.
        
        Args:
            symbol: trading pair
            stop_loss: stop loss price
            take_profit: take profit price
        
        Returns:
            (success, result or error)
        """
        try:
            params = {
                "category": "linear",
                "symbol": symbol,
                "positionIdx": 0,  # one-way mode
            }
            
            if stop_loss is not None:
                params["stopLoss"] = str(stop_loss)
            
            if take_profit is not None:
                params["takeProfit"] = str(take_profit)
            
            response = self.session.set_trading_stop(**params)
            
            if response.get("retCode") != 0:
                error = response.get("retMsg", "Unknown error")
                logger.error(f"Error setting TP/SL {symbol}: {error}")
                return False, error
            
            logger.info(f"TP/SL set for {symbol}: SL={stop_loss}, TP={take_profit}")
            return True, "ok"
        
        except Exception as e:
            err_text = str(e)
            # ErrCode 34040 = "not modified": the exchange reports that SL/TP
            # already have exactly these values. This is NOT an error —
            # the goal is achieved. pybit throws InvalidRequestError with
            # this code in the text.
            if "34040" in err_text or "not modified" in err_text.lower():
                logger.info(
                    f"TP/SL for {symbol} already set to specified values "
                    f"(SL={stop_loss}, TP={take_profit}) — skipping."
                )
                return True, "not_modified"
            logger.exception(f"Exception setting TP/SL {symbol}")
            return False, str(e)

    def get_order_info(
        self,
        symbol: str,
        order_id: str,
        retries: int = 3,
        retry_delay: float = 0.5,
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Returns the actual order execution: cumExecQty, avgPrice.

        A market order executes instantly and does NOT appear in get_open_orders,
        so we request order history. History updates with a ~0.5-1 sec delay,
        so we make several attempts.
        """
        for attempt in range(retries):
            try:
                response = self.session.get_order_history(
                    category="linear",
                    symbol=symbol,
                    orderId=order_id,
                )
                if response.get("retCode") != 0:
                    return False, response.get("retMsg")

                orders = response.get("result", {}).get("list", [])
                if orders:
                    return True, orders[0]

            except Exception as e:
                logger.exception(f"Exception getting order {order_id}")
                return False, str(e)

            if attempt < retries - 1:
                time.sleep(retry_delay)

        return True, None

    def get_closed_pnl(self, symbol: str, limit: int = 20) -> Tuple[bool, Optional[List[Dict]]]:
        """Exact realized PnL of closed trades (avg entry/exit prices, commissions)."""
        try:
            response = self.session.get_closed_pnl(
                category="linear", symbol=symbol, limit=limit
            )
            if response.get("retCode") != 0:
                return False, response.get("retMsg")
            return True, response.get("result", {}).get("list", [])
        except Exception as e:
            return False, str(e)
    
    def cancel_all_orders(self, symbol: str) -> Tuple[bool, Optional[str]]:
        """
        Cancels all active orders for the symbol.
        
        Returns:
            (success, result or error)
        """
        try:
            response = self.session.cancel_all_orders(
                category="linear",
                symbol=symbol,
            )
            
            if response.get("retCode") != 0:
                error = response.get("retMsg", "Unknown error")
                logger.warning(f"Partial error cancelling all orders {symbol}: {error}")
            
            logger.info(f"All orders cancelled for {symbol}")
            return True, "ok"
        
        except Exception as e:
            logger.exception(f"Exception cancelling all orders {symbol}")
            return False, str(e)
    
    # ----------------------------------------------------------------
    # Position size calculation
    # ----------------------------------------------------------------
    
    def calculate_position_size(
        self,
        symbol: str,
        capital_usdt: float,
        leverage: int,
        entry_price: float,
        risk_pct: float = 1.0,
        stop_loss_pct: Optional[float] = None,
    ) -> Tuple[bool, Optional[float]]:
        """
        Calculates position size in contracts.
        
        Two modes:
        1. Without risk: qty = (capital × leverage) / entry_price
        2. With risk: qty is calculated so that the loss on SL = risk_pct of capital
        
        Args:
            symbol: trading pair
            capital_usdt: available capital in USDT
            leverage: leverage
            entry_price: entry price
            risk_pct: percentage of capital to risk (for risk mode)
            stop_loss_pct: distance to SL as % of entry price (for risk mode)
        
        Returns:
            (success, qty or error)
        """
        success, info = self.get_instrument_info(symbol)
        if not success:
            return False, None
        
        min_order_qty = info["minOrderQty"]
        qty_step = info["qtyStep"]
        
        # Mode 1: simple calculation by capital
        if stop_loss_pct is None:
            qty_raw = (capital_usdt * leverage) / entry_price
        
        # Mode 2: calculation by risk
        else:
            # Maximum loss in USDT when SL triggers
            max_loss_usdt = capital_usdt * (risk_pct / 100.0)
            
            # Loss per 1 contract = entry_price × stop_loss_pct / 100
            loss_per_contract = entry_price * (stop_loss_pct / 100.0)
            
            # Number of contracts = max_loss_usdt / loss_per_contract
            qty_raw = max_loss_usdt / loss_per_contract
        
        # Round to the allowed step
        qty = self.round_quantity(qty_raw, qty_step)

        min_notional = info.get("minNotionalValue", 5.0)

        # Check minimum size in CONTRACTS.
        # If the calculation gave less than the minimum — raise to minOrderQty,
        # but only when this does not break the notional check (see below).
        if qty < min_order_qty:
            logger.warning(
                f"Calculated position size {qty} is less than minimum "
                f"{min_order_qty}. Using minimum."
            )
            qty = min_order_qty

        # Check minimum order VALUE (Bybit ErrCode 110094).
        # If qty * entry < minNotional — the order will be rejected by the exchange.
        # Return an error instead of waiting for rejection.
        notional = qty * entry_price
        if notional < min_notional:
            msg = (
                f"Position size {qty} contracts ({notional:.2f} USDT) "
                f"is less than minimum order value {min_notional:.2f} USDT "
                f"for {symbol}. Increase allocated capital, position size "
                f"or leverage."
            )
            logger.warning(msg)
            return False, msg

        logger.info(
            f"Position size for {symbol}: {qty} contracts "
            f"(capital={capital_usdt} USDT, leverage={leverage}x, entry={entry_price}, "
            f"notional={notional:.2f} USDT)"
        )

        return True, qty

    def validate_symbol_settings(
        self,
        symbol: str,
        capital_usdt: float,
        leverage: int,
        position_size_pct: float,
        entry_price: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Checks whether it is at all possible to open a position on this coin
        with the given settings.

        Called BEFORE starting trading to warn the user, rather than hitting
        exchange rejection on the first signal.

        Returns:
            (ok: bool, warning: str | None)
            ok=True, warning=None — settings are valid.
            ok=False, warning=<text> — with these settings the order will not pass.
        """
        success, info = self.get_instrument_info(symbol)
        if not success:
            return False, f"Failed to get parameters for {symbol}: {info}"

        min_notional = info.get("minNotionalValue", 5.0)
        min_qty = info.get("minOrderQty", 0.0)
        qty_step = info.get("qtyStep", 0.0)

        # Expected position value with these settings.
        # position_size_pct — share of allocated_capital going into the position.
        position_value_usdt = capital_usdt * (position_size_pct / 100.0)
        notional_with_leverage = position_value_usdt * leverage

        if notional_with_leverage < min_notional:
            return (
                False,
                f"For {symbol} with current settings the order will not pass: "
                f"position value {notional_with_leverage:.2f} USDT "
                f"is less than minimum {min_notional:.2f} USDT. "
                f"Increase capital share per coin, position size "
                f"or leverage (current leverage {leverage}x).",
            )

        # Check minimum number of contracts.
        qty_raw = notional_with_leverage / entry_price if entry_price > 0 else 0
        qty = self.round_quantity(qty_raw, qty_step) if qty_step > 0 else qty_raw

        if qty <= 0 or qty < min_qty:
            return (
                False,
                f"For {symbol} the calculated position size ({qty}) is less "
                f"than minimum ({min_qty}). Increase position size.",
            )

        return True, None

    def get_recent_executions(self, symbol: str, limit: int = 50) -> Tuple[bool, Optional[List[Dict]]]:
        """Execution history: GET /v5/execution/list (pybit: get_executions)."""
        try:
            response = self.session.get_executions(
                category="linear",
                symbol=symbol,
                limit=limit,
            )
            if response.get("retCode") != 0:
                error = response.get("retMsg", "Unknown error")
                logger.error(f"Error getting execution history {symbol}: {error}")
                return False, error
            return True, response.get("result", {}).get("list", [])
        except Exception as e:
            logger.exception(f"Exception getting execution history {symbol}")
            return False, str(e)
    
    def get_ticker_price(self, symbol: str) -> Tuple[bool, Optional[float]]:
        """
        Gets the current instrument price.
        
        Returns:
            (success, price or error)
        """
        try:
            response = self.session.get_tickers(
                category="linear",
                symbol=symbol,
            )
            
            if response.get("retCode") != 0:
                error = response.get("retMsg", "Unknown error")
                logger.error(f"Error getting price {symbol}: {error}")
                return False, error
            
            tickers = response.get("result", {}).get("list", [])
            if not tickers:
                return False, f"No ticker data for {symbol}"
            
            price = float(tickers[0].get("lastPrice", 0))
            return True, price
        
        except Exception as e:
            logger.exception(f"Exception getting price {symbol}")
            return False, str(e)

    def get_fee_rates(self, symbol: str) -> Tuple[bool, Optional[Dict]]:
        """
        Returns taker/maker fee rates for the instrument.

        Endpoint: GET /v5/account/fee-rate
        Bybit returns rates as strings in fractions: "0.00055" = 0.055%.
        The method returns them as percentages (0.055) — this is more convenient
        for comparison with commission_pct in strategy_spec / backtest.
        """
        try:
            response = self.session.get_fee_rates(
                category="linear",
                symbol=symbol,
            )
            if response.get("retCode") != 0:
                error = response.get("retMsg", "Unknown error")
                logger.error(f"Error getting rates {symbol}: {error}")
                return False, error

            rates = response.get("result", {}).get("list", [])
            if not rates:
                return False, f"Empty fee-rate response for {symbol}"

            item = rates[0]
            taker_pct = float(item.get("takerFeeRate", "0.00055")) * 100.0
            maker_pct = float(item.get("makerFeeRate", "0.0002")) * 100.0

            logger.info(
                f"Rates {symbol}: taker={taker_pct:.4f}%, "
                f"maker={maker_pct:.4f}%"
            )
            return True, {"taker_pct": taker_pct, "maker_pct": maker_pct}
        except Exception as e:
            logger.exception(f"Exception getting rates {symbol}")
            return False, str(e)
    
    # ----------------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------------
    
    @staticmethod
    def round_quantity(qty: float, qty_step: float) -> float:
        """
        Rounds quantity to the allowed step (down).
        
        Args:
            qty: original quantity
            qty_step: minimum step (e.g. 0.001)
        
        Returns:
            rounded quantity
        """
        if qty_step == 0:
            return qty
        
        decimal_qty = Decimal(str(qty))
        decimal_step = Decimal(str(qty_step))
        
        rounded = (decimal_qty / decimal_step).quantize(Decimal('1'), rounding=ROUND_DOWN) * decimal_step
        return float(rounded)
    
    @staticmethod
    def round_price(price: float, tick_size: float) -> float:
        """
        Rounds price to the allowed step (down for buy, up for sell).
        
        Args:
            price: original price
            tick_size: minimum price step (e.g. 0.01)
        
        Returns:
            rounded price
        """
        if tick_size == 0:
            return price
        
        decimal_price = Decimal(str(price))
        decimal_tick = Decimal(str(tick_size))
        
        rounded = (decimal_price / decimal_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * decimal_tick
        return float(rounded)