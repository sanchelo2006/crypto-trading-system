"""
Управление позициями: открытие, закрытие, мониторинг, выход по TP/SL.

Хранит состояние открытых позиций для каждой монеты и синхронизирует
его с реальными позициями на бирже.
"""

import logging
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from trading.executor import BybitExecutor

logger = logging.getLogger(__name__)

# Снимок индикаторов при входе (для анализа)
SNAPSHOT_KEYS = [
    "close", "rsi_14", "adx_14", "volume", "avg_volume_96", "avg_volume_20",
    "sma_20", "sma_50", "macd", "macd_signal", "macd_hist", "atr_14", "vwap",
    "pivot", "resistance_1", "support_1",
]

def _to_dt(ts):
    """Приводит timestamp (str / datetime / pd.Timestamp) к tz-aware datetime UTC."""
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
    Считает bars_held для живой торговли из entry_candle_ts.
    Возвращает None, если данных не хватает (позиция без entry_candle_ts
    или неизвестный timestamp свечи).
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
    Управление позициями для одной монеты.
    
    Отвечает за:
    - Открытие позиций (market ордер + установка TP/SL)
    - Закрытие позиций (по причине: SL, TP, ручное, stop-out)
    - Мониторинг (получение текущего состояния с биржи)
    - Проверку выходов (по management-правилам из strategy_spec)
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
        # Taker-ставка для вычитания комиссий при закрытии.
        # Значение перезаписывается AutoTraderSymbol после запроса
        # get_fee_rates() на старте.
        self.taker_fee_pct = float(taker_fee_pct)
        
        # Локальное состояние: список открытых позиций
        # Каждая позиция — dict с полями:
        # {
        #   "position_id": str (генерируется локально),
        #   "direction": "long" | "short",
        #   "entry_price": float,
        #   "qty": float,
        #   "stop_loss": float,
        #   "take_profit": float,
        #   "entry_time": datetime,
        #   "entry_reasons": List[str],
        #   "entry_snapshot": Dict,
        #   "management": Dict,
        #   "bybit_position_idx": int (для hedge mode),
        # }
        self.positions: List[Dict] = []

    def set_taker_fee_pct(self, value: float) -> None:
        """Обновляет taker-ставку (вызывается из AutoTraderSymbol после get_fee_rates)."""
        try:
            v = float(value)
            if v >= 0:
                self.taker_fee_pct = v
        except (TypeError, ValueError):
            pass
    
    def sync_with_exchange(self) -> Tuple[bool, Optional[str]]:
        """
        Синхронизирует локальное состояние с биржей.

        Единственная задача этого метода — ДОБАВИТЬ orphan-позиции
        (то, что есть на бирже, но не отслеживается локально, например
        после рестарта трейдера). УДАЛЕНИЕ локальных позиций здесь
        НЕ делается — этим занимается auto_trader._detect_closed_positions,
        который умеет правильно логировать CLOSE-событие и подтягивать
        Closed-PnL.

        Returns:
            (success, error_message или None)
        """
        success, result = self.executor.get_position_info(self.symbol)

        if not success:
            return False, result

        # Собираем ненулевые позиции с биржи: side -> {size, avgPrice}
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

        # Добавляем orphan-позиции: то, что есть на бирже, но не отслеживается
        # локально. Если позиция восстановлена из файла состояния —
        # она уже в self.positions и НЕ будет помечена как orphan.
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
                "stop_loss": 0.0,        # неизвестно, не трогаем
                "take_profit": 0.0,      # неизвестно, не трогаем
                "entry_time": datetime.now(timezone.utc),
                "entry_reasons": [],
                "entry_snapshot": {},
                "management": {},
                "is_orphan": True,
                "tpsl_error": "external_position_detected",
            }
            logger.warning(
                f"[{self.symbol}] Обнаружена внешняя позиция на бирже "
                f"(side={side_key}, size={info['size']}). Добавлена как orphan."
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
        Открывает новую позицию.
        
        Шаги:
        1. Размещает market ордер
        2. Вычисляет TP/SL
        3. Устанавливает TP/SL через set_trading_stop
        4. Сохраняет позицию в локальное состояние
        
        Returns:
            (success, error_message или None)
        """
        # 0. Страховка: если на бирже УЖЕ есть позиция по этому символу —
        #    новую не открываем. Защищает от двойных входов при рассинхроне
        #    локального состояния с биржей (например, после рестарта трейдера
        #    или после исключения в середине open_position).
        success_check, exchange_positions = self.executor.get_position_info(self.symbol)
        if success_check and exchange_positions:
            for p in exchange_positions:
                try:
                    size = float(p.get("size", 0) or 0)
                except (TypeError, ValueError):
                    size = 0
                if size > 0:
                    logger.warning(
                        f"[{self.symbol}] Отказ в открытии: на бирже уже есть "
                        f"позиция (side={p.get('side')}, size={size})."
                    )
                    return False, "position_already_exists_on_exchange"

        # Направление ордера для открытия позиции
        side = "Buy" if direction == "long" else "Sell"
        
        # 1. Открываем позицию market ордером
        success, result = self.executor.place_market_order(
            symbol=self.symbol,
            side=side,
            qty=qty,
            reduce_only=False,
        )
        
        if not success:
            logger.error(f"[{self.symbol}] Не удалось открыть позицию: {result}")
            return False, result
        
        order_id = result.get("orderId")

        # Узнаём ФАКТИЧЕСКОЕ исполнение (частичный филл + проскальзывание)
        success_o, order_info = self.executor.get_order_info(self.symbol, order_id)
        if success_o and order_info:
            filled_qty = float(order_info.get("cumExecQty", 0) or 0)
            avg_price = float(order_info.get("avgPrice", 0) or 0)
            if filled_qty > 0:
                qty = filled_qty          # реальный размер, а не рассчитанный
            if avg_price > 0:
                current_price = avg_price  # реальная средняя цена входа

        # если не исполнилось ничего — позицию не открываем
        if qty <= 0:
            logger.error(f"[{self.symbol}] Ордер не исполнился")
            return False, "order_not_filled"
        
        # 2. Вычисляем TP/SL
        stop_loss, take_profit = self._compute_tp_sl(
            direction, current_price, side_spec, indicators
        )
        
        # 3. Устанавливаем TP/SL
        tpsl_error = None
        success_sl, error_sl = self.executor.set_trading_stop(
            symbol=self.symbol,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        if not success_sl:
            tpsl_error = error_sl
            logger.warning(
                f"[{self.symbol}] Не удалось установить TP/SL: {error_sl}. "
                "Позиция открыта, но без защиты."
            )
        
        # 4. Сохраняем в локальное состояние
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
            # Запоминаем ИЗНАЧАЛЬНЫЕ уровни — они не меняются,
            # даже когда trailing/breakeven двигают stop_loss.
            "initial_stop_loss": stop_loss,
            "initial_take_profit": take_profit,
            "entry_time": datetime.now(timezone.utc),
            # Timestamp свечи, на которой сработал сигнал. Нужен для
            # расчёта bars_held в живой торговле (time-stop).
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
            f"[{self.symbol}] Позиция открыта: {direction.upper()} {qty} @ {current_price:.4f}, "
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
        Открывает позицию лимитным ордером (мейкер).

        Алгоритм:
        1. Получает best bid/ask из стакана.
        2. Вычисляет лимитную цену:
           - Лонг: best_bid + offset (покупаем по биду или чуть выше)
           - Шорт: best_ask - offset (продаём по аску или чуть ниже)
        3. Размещает Post-Only лимитный ордер.
        4. Возвращает order_id для мониторинга.

        Позиция НЕ добавляется в self.positions сразу.
        Вызывающий код (auto_trader) должен мониторить филл и
        вызвать _finalize_limit_entry() после исполнения.

        Args:
            direction: "long" или "short"
            qty: размер в контрактах
            side_spec: спека стороны (для расчёта TP/SL)
            entry_reasons: причины входа
            indicators: текущие индикаторы
            entry_candle_ts: timestamp свечи сигнала
            price_offset_ticks: сдвиг цены в тиках от лучшей цены
            timeout_seconds: таймаут ожидания филла

        Returns:
            (success, order_id или error, limit_price)
        """
        # 0. Проверяем, нет ли уже позиции на бирже
        success_check, exchange_positions = self.executor.get_position_info(self.symbol)
        if success_check and exchange_positions:
            for p in exchange_positions:
                try:
                    size = float(p.get("size", 0) or 0)
                except (TypeError, ValueError):
                    size = 0
                if size > 0:
                    return False, "position_already_exists_on_exchange", None

        # 1. Получаем best bid/ask
        success, bid_ask = self.executor.get_best_bid_ask(self.symbol)
        if not success:
            return False, f"Нет стакана: {bid_ask}", None

        best_bid, best_ask = bid_ask

        # 2. Получаем tickSize для округления
        success_info, info = self.executor.get_instrument_info(self.symbol)
        if not success_info or not info:
            return False, f"Нет параметров инструмента: {info}", None

        tick_size = info.get("tickSize", 0.01)

        # 3. Вычисляем лимитную цену
        if direction == "long":
            # Покупаем по биду (мейкер). Если цена вырастет — не исполнится.
            raw_price = best_bid + (price_offset_ticks * tick_size)
            limit_price = self.executor.round_price(raw_price, tick_size)
            side = "Buy"
        else:
            # Продаём по аску (мейкер). Если цена упадёт — не исполнится.
            raw_price = best_ask - (price_offset_ticks * tick_size)
            limit_price = self.executor.round_price(raw_price, tick_size)
            side = "Sell"

        # 4. Размещаем Post-Only лимитный ордер
        success, result = self.executor.place_limit_order(
            symbol=self.symbol,
            side=side,
            qty=qty,
            price=limit_price,
            post_only=True,
        )

        if not success:
            if isinstance(result, dict) and result.get("error") == "post_only_rejected":
                # Ордер стал бы тейкером. Попробуем с пассивной ценой.
                logger.warning(
                    f"[{self.symbol}] Post-Only отклонён. Пробуем пассивнее."
                )
                # Отступаем на 1 тик вглубь стакана
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
                    return False, f"Повторный Post-Only отклонён: {result}", None
            else:
                return False, str(result), None

        order_id = result.get("orderId")
        if not order_id:
            return False, "Нет orderId в ответе", None

        logger.info(
            f"[{self.symbol}] Лимитный ордер размещён: {direction} {qty} @ {limit_price} "
            f"(orderId={order_id}, timeout={timeout_seconds}с)"
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
        Завершает открытие позиции после исполнения лимитного ордера.
        Вызывается из auto_trader после подтверждения филла.

        Устанавливает TP/SL и добавляет позицию в локальный список.

        Returns:
            (success, error_message или None)
        """
        # Вычисляем TP/SL
        stop_loss, take_profit = self._compute_tp_sl(
            direction, entry_price, side_spec, indicators
        )

        # Устанавливаем TP/SL на бирже
        tpsl_error = None
        success_sl, error_sl = self.executor.set_trading_stop(
            symbol=self.symbol,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        if not success_sl:
            tpsl_error = error_sl
            logger.warning(
                f"[{self.symbol}] Не удалось установить TP/SL: {error_sl}. "
                "Позиция открыта без защиты."
            )

        # Формируем снапшот индикаторов
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
            f"[{self.symbol}] Лимитная позиция открыта: {direction.upper()} "
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
        Закрывает позицию.

        Закрывает РЕАЛЬНЫЙ размер позиции с биржи, а не локально
        записанный (market-ордер мог исполниться частично).
        """
        direction = position["direction"]
        qty = position["qty"]
        expected_side = "buy" if direction == "long" else "sell"

        # 1. Узнаём фактический размер позиции на бирже
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
                # На бирже позиции уже нет — второй ордер отправлять нельзя.
                logger.warning(
                    f"[{self.symbol}] Позиция {position['position_id']} на бирже "
                    "отсутствует, reduce-only ордер не отправляю."
                )
                self.positions = [
                    p for p in self.positions
                    if p["position_id"] != position["position_id"]
                ]
                return False, "already_closed_on_exchange"
            qty = exchange_qty  # закрываем ровно столько, сколько реально есть

        # 2. Цена выхода — берём прямо перед отправкой ордера, если не передана
        if current_price is None:
            ok, price = self.executor.get_ticker_price(self.symbol)
            if ok and price:
                current_price = price

        # 3. Отправляем reduce-only market-ордер
        side = "Sell" if direction == "long" else "Buy"
        success, result = self.executor.place_market_order(
            symbol=self.symbol,
            side=side,
            qty=qty,
            reduce_only=True,
        )

        if not success:
            logger.error(f"[{self.symbol}] Не удалось закрыть позицию: {result}")
            return False, result

        # 4. Вычисляем PnL
        entry_price = position["entry_price"]
        exit_price = current_price if current_price else entry_price

        if direction == "long":
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100

        # Комиссии: taker на входе и на выходе. То же самое делает
        # backtest_engine._close_position — так результат совпадает.
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

        # Удаляем из локального состояния
        self.positions = [p for p in self.positions if p["position_id"] != position["position_id"]]

        logger.info(
            f"[{self.symbol}] Позиция закрыта: {direction.upper()} {qty} @ {exit_price:.4f}, "
            f"PnL={pnl_pct:+.2f}%, причина={reason}"
        )

        return True, trade
    
    def close_all_positions(self, reason: str = "manual_stop") -> List[Dict]:
        """
        Закрывает все открытые позиции.
        
        Returns:
            Список закрытых сделок (trade_dict)
        """
        trades = []
        
        for position in list(self.positions):
            success, result = self.close_position(position, reason)
            if success:
                trades.append(result)
            else:
                logger.error(
                    f"[{self.symbol}] Не удалось закрыть позицию {position['position_id']}: {result}"
                )
        
        return trades
    
    def check_exits(
        self,
        current_indicators: Dict,
    ) -> List[Tuple[Dict, str, float]]:
        """
        Проверяет условия выхода для всех открытых позиций.
        
        Returns:
            Список [(position, exit_reason, exit_price), ...]
        """
        exits = []
        
        for position in self.positions:
            exit_price, reason = self._check_exit(position, current_indicators)
            if exit_price is not None:
                exits.append((position, reason, exit_price))
        
        return exits
    
    @staticmethod
    def _check_hard_exit(pos: Dict, row: Dict) -> Tuple[Optional[float], str]:
        """Жёсткие выходы (SL/TP). Используется только в бэктесте —
        в живой торговле SL/TP выставляет биржа."""
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
        Мягкие выходы по management: time_stop, exit_on_level_failure,
        exit_on_trend_loss, exit_on_macd_flip.

        Логика идентична backtest_engine._check_exit. Для time_stop
        используется:
          - pos["bars_held"], если он есть (путь бэктеста);
          - иначе bars_held считается из entry_candle_ts и row["timestamp"]
            (путь живой торговли).
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
        """Полная проверка (используется в бэктесте и тестах)."""
        price, reason = self._check_hard_exit(pos, row)
        if price is not None:
            return price, reason
        return self._check_management_exits(pos, row)

    def check_soft_exit(self, position: Dict, candle: Dict) -> Tuple[Optional[float], str]:
        """
        Публичный метод для живого трейдера: только мягкие выходы.
        SL/TP обрабатывает биржа, повторно их проверять нельзя.
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
        Вычисляет TP и SL для новой позиции.

        Приоритет — ATR-стопы (как в бэктесте).
        Цены округляются под tickSize инструмента: иначе Bybit отклонит
        set_trading_stop, и позиция останется без защиты.
        """
        # Используем 4H ATR если он есть (свинг-стратегии),
        # иначе стандартный 15m ATR
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

        # Fallback — фиксированные проценты
        if stop_dist is None:
            sl_pct = float(side_spec.get("stop_loss_pct", 2.0)) / 100.0
            stop_dist = entry * sl_pct

        # Fallback — если ATR не дал значения, берём проценты из spec.
        if take_dist is None:
            tp_list = side_spec.get("take_profit_pct", [3.0]) or [3.0]
            tp_pct = float(tp_list[0]) / 100.0
            take_dist = entry * tp_pct

        # Расчёт stop/take идёт ВСЕГДА — независимо от того,
        # пришёл take_dist из ATR или из процентов.
        if direction == "long":
            stop = entry - stop_dist
            take = entry + take_dist
        else:
            stop = entry + stop_dist
            take = entry - take_dist

        # ЗАЩИТА: ограничиваем SL и TP разумными процентами от entry.
        # ATR может быть аномально завышен из-за выбросов в OHLCV (особенно на testnet),
        # и тогда SL/TP улетают в 10-20 раз дальше нормы. Лучше сделать
        # консервативный стоп, чем получить -16% на одной сделке.
        MAX_STOP_PCT = 2.5    # стоп не дальше 2.5% от входа
        MAX_TAKE_PCT = 10.0   # тейк не дальше 10% от входа

        if direction == "long":
            stop = max(stop, entry * (1 - MAX_STOP_PCT / 100.0))
            take = min(take, entry * (1 + MAX_TAKE_PCT / 100.0))
        else:
            stop = min(stop, entry * (1 + MAX_STOP_PCT / 100.0))
            take = max(take, entry * (1 - MAX_TAKE_PCT / 100.0))

        # Округляем под tickSize инструмента.
        success, info = self.executor.get_instrument_info(self.symbol)
        if success and info:
            tick = info.get("tickSize")
            if tick:
                stop = self.executor.round_price(stop, tick)
                take = self.executor.round_price(take, tick)

        return stop, take