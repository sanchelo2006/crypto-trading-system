"""
Исполнение ордеров через Bybit Unified Trading API для фьючерсной торговли.

Особенности linear perpetuals (USDT-маржа):
- Категория: linear
- Leverage (плечо) устанавливается отдельно
- Margin mode: isolated (по умолчанию) или cross
- Position mode: one-way (позиция в одну сторону)
- Размер позиции указывается в контрактах (qty), а не в USDT
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from decimal import Decimal, ROUND_DOWN, ROUND_UP

from pybit.unified_trading import HTTP

import config

logger = logging.getLogger(__name__)

class BybitError(Exception):
    """Исключение для ошибок API Bybit."""
    pass


class BybitExecutor:
    """
    Низкоуровневый адаптер для работы с Bybit фьючерсами (linear perpetual).
    
    Все методы возвращают (success: bool, result/error).
    """
    
    def __init__(
        self,
        testnet: bool = False,
        api_key: str = None,
        api_secret: str = None,
        default_leverage: int = 10,
        margin_mode: str = "ISOLATED_MARGIN",  # или "REGULAR_MARGIN" (cross)
    ):
        self.testnet = testnet
        self.api_key = api_key or config.BYBIT_API_KEY
        self.api_secret = api_secret or config.BYBIT_API_SECRET
        self.default_leverage = default_leverage
        self.margin_mode = margin_mode
        
        if not self.api_key or not self.api_secret:
            raise ValueError("BYBIT_API_KEY и BYBIT_API_SECRET должны быть заданы")
        
        self.session = HTTP(
            testnet=testnet,
            api_key=self.api_key,
            api_secret=self.api_secret,
        )
        
        # Кэш параметров инструментов (tickSize, qtyStep, minOrderQty)
        self._instruments_cache: Dict[str, Dict] = {}
        
        logger.info(
            f"BybitExecutor инициализирован ({'testnet' if testnet else 'mainnet'}), "
            f"leverage={default_leverage}, margin_mode={margin_mode}"
        )
    
    # ----------------------------------------------------------------
    # Информация об инструментах
    # ----------------------------------------------------------------
    
    def get_instrument_info(self, symbol: str) -> Tuple[bool, Optional[Dict]]:
        """
        Получает параметры торгового инструмента.

        Returns:
            (success, info_dict или error)

        info_dict содержит:
            - tickSize: минимальный шаг цены
            - qtyStep: минимальный шаг количества
            - minOrderQty: минимальный размер ордера
            - maxOrderQty: максимальный размер ордера
            - minNotionalValue: минимальная стоимость ордера в USDT

        Логика отказоустойчивости:
        1. Если в кэше уже есть данные — отдаём сразу, в сеть не идём.
        2. При промахе — retry с экспоненциальной задержкой.
           Bybit testnet периодически отдаёт ErrCode 10016 (Internal System Error)
           и 10000 (Server Timeout) — транзиентные ошибки, обычно проходят
           со 2-3 попытки.
        3. Если все попытки провалились, но в кэше ЕСТЬ данные (например,
           получены раньше, а сейчас сеть опять легла) — возвращаем их.
           tickSize/qtyStep для инструмента практически не меняются,
           так что старые данные абсолютно валидны.
        4. Если совсем ничего нет — возвращаем (False, error).
        """
        # 1. Быстрый путь — данные в кэше
        if symbol in self._instruments_cache:
            return True, self._instruments_cache[symbol]

        last_error = "Unknown error"

        # 2. Retry с экспоненциальной задержкой
        for attempt in range(3):
            try:
                response = self.session.get_instruments_info(
                    category="linear",
                    symbol=symbol,
                )

                if response.get("retCode") != 0:
                    last_error = response.get("retMsg", "Unknown error")
                    logger.warning(
                        f"Информация о {symbol}: retCode={response.get('retCode')}, "
                        f"{last_error} (попытка {attempt + 1}/3)"
                    )
                    if attempt < 2:
                        time.sleep(1.0 * (2 ** attempt))
                    continue

                instruments = response.get("result", {}).get("list", [])
                if not instruments:
                    last_error = f"Инструмент {symbol} не найден"
                    logger.warning(
                        f"Информация о {symbol}: пустой список инструментов"
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
                    f"Информация о {symbol}: tickSize={instrument_data['tickSize']}, "
                    f"qtyStep={instrument_data['qtyStep']}, "
                    f"minOrderQty={instrument_data['minOrderQty']}"
                )

                return True, instrument_data

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Информация о {symbol}: исключение {type(e).__name__} "
                    f"({last_error}), попытка {attempt + 1}/3"
                )
                if attempt < 2:
                    time.sleep(1.0 * (2 ** attempt))

        # 3. Все попытки провалились.
        # Если в кэше есть старые данные — отдаём их.
        if symbol in self._instruments_cache:
            logger.warning(
                f"Информация о {symbol}: не удалось получить свежие данные "
                f"({last_error}). Использую кэш."
            )
            return True, self._instruments_cache[symbol]

        # 4. Совсем ничего нет
        logger.error(
            f"Информация о {symbol}: окончательный провал после 3 попыток. "
            f"Причина: {last_error}"
        )
        return False, last_error
    
    # ----------------------------------------------------------------
    # Баланс и позиции
    # ----------------------------------------------------------------
    
    def get_wallet_balance(self, account_type: str = "UNIFIED") -> Tuple[bool, Optional[Dict]]:
        """
        Получает баланс кошелька.
        
        Returns:
            (success, balance_dict или error_message)
        """
        try:
            response = self.session.get_wallet_balance(accountType=account_type)
            
            if response.get("retCode") != 0:
                error = response.get("retMsg", "Unknown error")
                logger.error(f"Ошибка получения баланса: {error}")
                return False, error
            
            result = response.get("result", {})
            return True, result
        
        except Exception as e:
            logger.exception("Исключение при получении баланса")
            return False, str(e)
    
    def get_usdt_balance(self) -> Tuple[bool, Optional[float]]:
        """
        Возвращает USDT-баланс кошелька.
        availableToWithdraw deprecated для UTA 2.0 (всегда ""), поэтому
        берём первое непустое из availableToWithdraw → walletBalance → equity.
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
                        #раньше здесь было логирование в терминал - но это засоряет
                        return True, value
                    except (TypeError, ValueError):
                        continue
                return True, 0.0
            return True, 0.0
        except Exception:
            logger.exception("Ошибка парсинга баланса USDT")
            return False, None
    
    def get_position_info(self, symbol: str) -> Tuple[bool, Optional[List[Dict]]]:
        """
        Получает информацию об открытых позициях по символу.
        
        Returns:
            (success, list_of_positions или error)
        """
        try:
            response = self.session.get_positions(
                category="linear",
                symbol=symbol,
            )
            
            if response.get("retCode") != 0:
                error = response.get("retMsg", "Unknown error")
                logger.error(f"Ошибка получения позиций {symbol}: {error}")
                return False, error
            
            positions = response.get("result", {}).get("list", [])
            return True, positions
        
        except Exception as e:
            logger.exception(f"Исключение при получении позиций {symbol}")
            return False, str(e)
    
    # ----------------------------------------------------------------
    # Настройка плеча и маржи
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
                logger.info(f"Плечо для {symbol}: {leverage}x")
                return True, "ok"
            error = response.get("retMsg", "Unknown error")
            logger.error(f"Ошибка установки плеча {symbol}: {error}")
            return False, error
        except Exception as e:
            # pybit бросает InvalidRequestError; 110043 = плечо уже установлено
            code = getattr(e, "status_code", None)
            if code == 110043 or "leverage not modified" in str(e).lower():
                logger.info(f"Плечо для {symbol} уже установлено: {leverage}x")
                return True, "ok"
            logger.exception(f"Исключение при установке плеча {symbol}")
            return False, str(e)
    
    def switch_margin_mode(self, symbol: str, margin_mode: str) -> Tuple[bool, Optional[str]]:
        """
        Переключает режим маржи.
        
        Args:
            symbol: торговая пара
            margin_mode: "ISOLATED_MARGIN" или "REGULAR_MARGIN" (cross)
        
        Returns:
            (success, result или error)
        """
        try:
            # Важно: переключение возможно только при отсутствии открытых позиций
            response = self.session.switch_margin_mode(
                category="linear",
                symbol=symbol,
                tradeMode=0 if margin_mode == "REGULAR_MARGIN" else 1,
                buyLeverage=str(self.default_leverage),
                sellLeverage=str(self.default_leverage),
            )
            
            if response.get("retCode") != 0:
                error = response.get("retMsg", "Unknown error")
                # Если ошибка "leverage not modified" — это нормально (уже установлено)
                if "leverage not modified" in error.lower():
                    logger.info(f"Режим маржи для {symbol} уже установлен: {margin_mode}")
                    return True, "ok"
                logger.error(f"Ошибка переключения режима маржи {symbol}: {error}")
                return False, error
            
            logger.info(f"Режим маржи установлен для {symbol}: {margin_mode}")
            return True, "ok"
        
        except Exception as e:
            logger.exception(f"Исключение при переключении режима маржи {symbol}")
            return False, str(e)
    
    # ----------------------------------------------------------------
    # Открытие и закрытие позиций
    # ----------------------------------------------------------------
    
    def place_market_order(
        self,
        symbol: str,
        side: str,  # "Buy" или "Sell"
        qty: float,
        reduce_only: bool = False,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Размещает рыночный ордер на фьючерсном рынке.
        
        Args:
            symbol: торговая пара
            side: "Buy" (лонг) или "Sell" (шорт/закрытие лонга)
            qty: количество контрактов
            reduce_only: True для закрытия позиции
            stop_loss: цена стоп-лосса (опционально)
            take_profit: цена тейк-профита (опционально)
        
        Returns:
            (success, order_result или error)
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
            
            # Добавляем TP/SL к ордеру (если указаны)
            if stop_loss is not None:
                params["stopLoss"] = str(stop_loss)
            
            if take_profit is not None:
                params["takeProfit"] = str(take_profit)
            
            response = self.session.place_order(**params)
            
            if response.get("retCode") != 0:
                error = response.get("retMsg", "Unknown error")
                logger.error(f"Ошибка размещения market ордера {symbol}: {error}")
                return False, error
            
            result = response.get("result", {})
            logger.info(
                f"Market ордер размещён: {symbol} {side} {qty} "
                f"(orderId={result.get('orderId')})"
            )
            return True, result
        
        except Exception as e:
            logger.exception(f"Исключение при размещении market ордера {symbol}")
            return False, str(e)

    def place_limit_order(
        self,
        symbol: str,
        side: str,  # "Buy" или "Sell"
        qty: float,
        price: float,
        reduce_only: bool = False,
        post_only: bool = True,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Размещает лимитный ордер.

        При post_only=True ордер гарантированно исполняется как мейкер.
        Если бы он исполнился как тейкер — биржа его отклонит (код 110016).

        Args:
            symbol: торговая пара
            side: "Buy" (лонг) или "Sell" (шорт)
            qty: количество контрактов
            price: цена лимитного ордера
            reduce_only: True для закрытия позиции
            post_only: True = только мейкер (отклоняется если станет тейкером)
            stop_loss: цена стоп-лосса (опционально)
            take_profit: цена тейк-профита (опционально)

        Returns:
            (success, order_result или error)
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

                # 110016 = ордер был бы исполнен как тейкер → отклонён
                if ret_code == 110016:
                    logger.warning(
                        f"Limit ордер {symbol} отклонён (стал бы тейкером): {error}"
                    )
                    return False, {"error": "post_only_rejected", "detail": error}

                logger.error(
                    f"Ошибка размещения limit ордера {symbol}: {error}"
                )
                return False, error

            result = response.get("result", {})
            logger.info(
                f"Limit ордер размещён: {symbol} {side} {qty} @ {price} "
                f"(orderId={result.get('orderId')}, postOnly={post_only})"
            )
            return True, result

        except Exception as e:
            logger.exception(f"Исключение при размещении limit ордера {symbol}")
            return False, str(e)
    
    def cancel_order(
        self,
        symbol: str,
        order_id: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Отменяет открытый ордер по orderId.

        Returns:
            (success, error или None)
        """
        try:
            response = self.session.cancel_order(
                category="linear",
                symbol=symbol,
                orderId=order_id,
            )
            if response.get("retCode") != 0:
                error = response.get("retMsg", "Unknown error")
                logger.warning(f"Не удалось отменить ордер {order_id}: {error}")
                return False, error
            logger.info(f"Ордер {order_id} отменён для {symbol}")
            return True, None
        except Exception as e:
            logger.exception(f"Исключение при отмене ордера {order_id}")
            return False, str(e)

    def get_open_orders(
        self,
        symbol: str,
    ) -> Tuple[bool, Optional[List[Dict]]]:
        """
        Возвращает список открытых ордеров по символу.

        Returns:
            (success, list_of_orders или error)
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
            logger.exception(f"Исключение при получении ордеров {symbol}")
            return False, str(e)

    def get_best_bid_ask(
        self,
        symbol: str,
    ) -> Tuple[bool, Optional[Tuple[float, float]]]:
        """
        Возвращает лучшие цены покупки и продажи из стакана.

        Returns:
            (success, (best_bid, best_ask) или error)
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
                return False, f"Нет данных тикера для {symbol}"

            ticker = tickers[0]
            best_bid = float(ticker.get("bid1Price", 0))
            best_ask = float(ticker.get("ask1Price", 0))

            if best_bid <= 0 or best_ask <= 0:
                return False, f"Некорректный стакан для {symbol}"

            return True, (best_bid, best_ask)

        except Exception as e:
            logger.exception(f"Исключение при получении стакана {symbol}")
            return False, str(e)

    def set_trading_stop(
        self,
        symbol: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Устанавливает TP/SL для открытой позиции.
        
        Args:
            symbol: торговая пара
            stop_loss: цена стоп-лосса
            take_profit: цена тейк-профита
        
        Returns:
            (success, result или error)
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
                logger.error(f"Ошибка установки TP/SL {symbol}: {error}")
                return False, error
            
            logger.info(f"TP/SL установлены для {symbol}: SL={stop_loss}, TP={take_profit}")
            return True, "ok"
        
        except Exception as e:
            err_text = str(e)
            # ErrCode 34040 = "not modified": биржа сообщает, что SL/TP уже
            # имеют ровно такие же значения. Это НЕ ошибка — цель достигнута.
            # pybit кидает InvalidRequestError с этим кодом в тексте.
            if "34040" in err_text or "not modified" in err_text.lower():
                logger.info(
                    f"TP/SL для {symbol} уже установлены в указанные значения "
                    f"(SL={stop_loss}, TP={take_profit}) — пропуск."
                )
                return True, "not_modified"
            logger.exception(f"Исключение при установке TP/SL {symbol}")
            return False, str(e)

    def get_order_info(
        self,
        symbol: str,
        order_id: str,
        retries: int = 3,
        retry_delay: float = 0.5,
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Возвращает фактическое исполнение ордера: cumExecQty, avgPrice.

        Market-ордер исполняется мгновенно и НЕ появляется в get_open_orders,
        поэтому запрашиваем историю ордеров. История обновляется с задержкой
        ~0.5-1 сек, поэтому делаем несколько попыток.
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
                logger.exception(f"Исключение при получении ордера {order_id}")
                return False, str(e)

            if attempt < retries - 1:
                time.sleep(retry_delay)

        return True, None

    def get_closed_pnl(self, symbol: str, limit: int = 20) -> Tuple[bool, Optional[List[Dict]]]:
        """Точный реализованный PnL закрытых сделок (ср. цены входа/выхода, комиссии)."""
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
        Отменяет все активные ордера по символу.
        
        Returns:
            (success, result или error)
        """
        try:
            response = self.session.cancel_all_orders(
                category="linear",
                symbol=symbol,
            )
            
            if response.get("retCode") != 0:
                error = response.get("retMsg", "Unknown error")
                logger.warning(f"Частичная ошибка отмены всех ордеров {symbol}: {error}")
            
            logger.info(f"Все ордера отменены для {symbol}")
            return True, "ok"
        
        except Exception as e:
            logger.exception(f"Исключение при отмене всех ордеров {symbol}")
            return False, str(e)
    
    # ----------------------------------------------------------------
    # Расчёт размера позиции
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
        Вычисляет размер позиции в контрактах.
        
        Два режима:
        1. Без учёта риска: qty = (capital × leverage) / entry_price
        2. С учётом риска: qty рассчитывается так, чтобы потеря при SL = risk_pct от капитала
        
        Args:
            symbol: торговая пара
            capital_usdt: доступный капитал в USDT
            leverage: плечо
            entry_price: цена входа
            risk_pct: процент капитала, который рискуем (для режима с риском)
            stop_loss_pct: расстояние до SL в % от цены входа (для режима с риском)
        
        Returns:
            (success, qty или error)
        """
        success, info = self.get_instrument_info(symbol)
        if not success:
            return False, None
        
        min_order_qty = info["minOrderQty"]
        qty_step = info["qtyStep"]
        
        # Режим 1: простой расчёт по капиталу
        if stop_loss_pct is None:
            qty_raw = (capital_usdt * leverage) / entry_price
        
        # Режим 2: расчёт по риску
        else:
            # Максимальная потеря в USDT при срабатывании SL
            max_loss_usdt = capital_usdt * (risk_pct / 100.0)
            
            # Потеря на 1 контракт = entry_price × stop_loss_pct / 100
            loss_per_contract = entry_price * (stop_loss_pct / 100.0)
            
            # Количество контрактов = max_loss_usdt / loss_per_contract
            qty_raw = max_loss_usdt / loss_per_contract
        
        # Округляем до допустимого шага
        qty = self.round_quantity(qty_raw, qty_step)

        min_notional = info.get("minNotionalValue", 5.0)

        # Проверяем минимальный размер в КОНТРАКТАХ.
        # Если расчёт дал меньше минимума — поднимаем до minOrderQty, но
        # только когда это не ломает проверку notional (см. ниже).
        if qty < min_order_qty:
            logger.warning(
                f"Вычисленный размер позиции {qty} меньше минимального "
                f"{min_order_qty}. Используем минимальный."
            )
            qty = min_order_qty

        # Проверяем минимальную СТОИМОСТЬ ордера (Bybit ErrCode 110094).
        # Если qty * entry < minNotional — ордер будет отклонён биржей.
        # Возвращаем ошибку вместо того, чтобы ждать отказа.
        notional = qty * entry_price
        if notional < min_notional:
            msg = (
                f"Размер позиции {qty} контрактов ({notional:.2f} USDT) "
                f"меньше минимальной стоимости ордера {min_notional:.2f} USDT "
                f"для {symbol}. Увеличьте выделенный капитал, размер позиции "
                f"или плечо."
            )
            logger.warning(msg)
            return False, msg

        logger.info(
            f"Размер позиции для {symbol}: {qty} контрактов "
            f"(капитал={capital_usdt} USDT, плечо={leverage}x, вход={entry_price}, "
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
        Проверяет, можно ли в принципе открыть позицию по этой монете
        при заданных настройках.

        Вызывается ДО начала торговли, чтобы предупредить пользователя,
        а не натыкаться на отказ биржи при первом сигнале.

        Returns:
            (ok: bool, warning: str | None)
            ok=True, warning=None — настройки валидны.
            ok=False, warning=<текст> — при таких настройках ордер не пройдёт.
        """
        success, info = self.get_instrument_info(symbol)
        if not success:
            return False, f"Не удалось получить параметры {symbol}: {info}"

        min_notional = info.get("minNotionalValue", 5.0)
        min_qty = info.get("minOrderQty", 0.0)
        qty_step = info.get("qtyStep", 0.0)

        # Ожидаемая стоимость позиции при этих настройках.
        # position_size_pct — доля allocated_capital, идущая в позицию.
        position_value_usdt = capital_usdt * (position_size_pct / 100.0)
        notional_with_leverage = position_value_usdt * leverage

        if notional_with_leverage < min_notional:
            return (
                False,
                f"Для {symbol} при текущих настройках ордер не пройдёт: "
                f"стоимость позиции {notional_with_leverage:.2f} USDT "
                f"меньше минимальной {min_notional:.2f} USDT. "
                f"Увеличьте долю капитала на монету, размер позиции "
                f"или плечо (сейчас плечо {leverage}x).",
            )

        # Проверка минимального количества контрактов.
        qty_raw = notional_with_leverage / entry_price if entry_price > 0 else 0
        qty = self.round_quantity(qty_raw, qty_step) if qty_step > 0 else qty_raw

        if qty <= 0 or qty < min_qty:
            return (
                False,
                f"Для {symbol} вычисленный размер позиции ({qty}) меньше "
                f"минимального ({min_qty}). Увеличьте размер позиции.",
            )

        return True, None

    def get_recent_executions(self, symbol: str, limit: int = 50) -> Tuple[bool, Optional[List[Dict]]]:
        """История исполнений: GET /v5/execution/list (pybit: get_executions)."""
        try:
            response = self.session.get_executions(
                category="linear",
                symbol=symbol,
                limit=limit,
            )
            if response.get("retCode") != 0:
                error = response.get("retMsg", "Unknown error")
                logger.error(f"Ошибка получения истории сделок {symbol}: {error}")
                return False, error
            return True, response.get("result", {}).get("list", [])
        except Exception as e:
            logger.exception(f"Исключение при получении истории сделок {symbol}")
            return False, str(e)
    
    def get_ticker_price(self, symbol: str) -> Tuple[bool, Optional[float]]:
        """
        Получает текущую цену инструмента.
        
        Returns:
            (success, цена или error)
        """
        try:
            response = self.session.get_tickers(
                category="linear",
                symbol=symbol,
            )
            
            if response.get("retCode") != 0:
                error = response.get("retMsg", "Unknown error")
                logger.error(f"Ошибка получения цены {symbol}: {error}")
                return False, error
            
            tickers = response.get("result", {}).get("list", [])
            if not tickers:
                return False, f"Нет данных тикера для {symbol}"
            
            price = float(tickers[0].get("lastPrice", 0))
            return True, price
        
        except Exception as e:
            logger.exception(f"Исключение при получении цены {symbol}")
            return False, str(e)

    def get_fee_rates(self, symbol: str) -> Tuple[bool, Optional[Dict]]:
        """
        Возвращает taker/maker ставки комиссии для инструмента.

        Endpoint: GET /v5/account/fee-rate
        Bybit отдаёт ставки строками в долях: "0.00055" = 0.055%.
        Метод возвращает их как проценты (0.055) — так удобнее
        сравнивать с commission_pct в strategy_spec / бэктесте.
        """
        try:
            response = self.session.get_fee_rates(
                category="linear",
                symbol=symbol,
            )
            if response.get("retCode") != 0:
                error = response.get("retMsg", "Unknown error")
                logger.error(f"Ошибка получения ставок {symbol}: {error}")
                return False, error

            rates = response.get("result", {}).get("list", [])
            if not rates:
                return False, f"Пустой ответ fee-rate для {symbol}"

            item = rates[0]
            taker_pct = float(item.get("takerFeeRate", "0.00055")) * 100.0
            maker_pct = float(item.get("makerFeeRate", "0.0002")) * 100.0

            logger.info(
                f"Ставки {symbol}: taker={taker_pct:.4f}%, "
                f"maker={maker_pct:.4f}%"
            )
            return True, {"taker_pct": taker_pct, "maker_pct": maker_pct}
        except Exception as e:
            logger.exception(f"Исключение при получении ставок {symbol}")
            return False, str(e)
    
    # ----------------------------------------------------------------
    # Вспомогательные методы
    # ----------------------------------------------------------------
    
    @staticmethod
    def round_quantity(qty: float, qty_step: float) -> float:
        """
        Округляет количество до допустимого шага (вниз).
        
        Args:
            qty: исходное количество
            qty_step: минимальный шаг (например, 0.001)
        
        Returns:
            округлённое количество
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
        Округляет цену до допустимого шага (вниз для покупки, вверх для продажи).
        
        Args:
            price: исходная цена
            tick_size: минимальный шаг цены (например, 0.01)
        
        Returns:
            округлённая цена
        """
        if tick_size == 0:
            return price
        
        decimal_price = Decimal(str(price))
        decimal_tick = Decimal(str(tick_size))
        
        rounded = (decimal_price / decimal_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * decimal_tick
        return float(rounded)