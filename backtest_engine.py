"""
Механический бэктест-движок.

Исполняет машиночитаемую спецификацию стратегии (strategy_spec), которую AI
генерирует при инициализации монеты. Индикаторы рассчитываются внутри движка
с корректным выравниванием по индексу свечей, pivot-уровни пересчитываются
на каждый день (по данным предыдущего дня).
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import math
import config

try:
    import talib
except ImportError:
    talib = None

from data_fetcher import BybitDataFetcher
from indicators import (
    resample_15m_to_1h,
    compute_1h_indicators,
    merge_1h_features_to_15m,
    get_1h_feature_columns,
)

from strategy_spec import (
    evaluate_conditions,
    validate_strategy_spec,
    explain_conditions,
    default_strategy_spec,
)

logger = logging.getLogger(__name__)

# Индикаторы, которые сохраняем на моменте входа в сделку — для последующего анализа
SNAPSHOT_KEYS = [
    "close", "rsi_14", "adx_14", "volume", "avg_volume_96", "avg_volume_20",
    "sma_20", "sma_50", "macd", "macd_signal", "macd_hist", "atr_14", "vwap",
    "pivot", "resistance_1", "support_1",
]


class BacktestEngine:
    def __init__(self, fetcher: BybitDataFetcher):
        self.fetcher = fetcher
        self.last_simulation_meta = {}
        # Проскальзывание по умолчанию — из config, можно переопределить
        # параметром run()/run_on_dataframe().
        self.slippage_pct = float(getattr(config, "BACKTEST_SLIPPAGE_PCT", 0.0))
    # ------------------------------------------------------------------ #
    # Публичный метод
    # ------------------------------------------------------------------ #
    def run(
        self,
        symbol: str,
        category: str,
        strategy_spec: Optional[Dict],
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 1000.0,
        commission_pct: float = 0.055,   # taker Bybit, совпадает с live-дефолтом
        rules_text: str = "",
        slippage_pct: Optional[float] = None,
    ) -> Dict:
        """
        Выполняет бэктест по strategy_spec на данных, загруженных из fetcher.
        """
        logger.info(f"[{symbol}] Загрузка свечей {start_date} → {end_date}")
        df = self.fetcher.fetch_ohlcv(symbol, "15", start_date, end_date, category)
        if df is None or df.empty:
            return {"error": f"Не удалось загрузить 15m свечи для {symbol}"}

        return self.run_on_dataframe(
            df=df,
            symbol=symbol,
            strategy_spec=strategy_spec,
            initial_capital=initial_capital,
            commission_pct=commission_pct,
            rules_text=rules_text,
            start_date=start_date,
            end_date=end_date,
            slippage_pct=slippage_pct,
        )

    def run_on_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str,
        strategy_spec: Optional[Dict],
        initial_capital: float = 1000.0,
        commission_pct: float = 0.055,   # taker Bybit, совпадает с live-дефолтом
        rules_text: str = "",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        slippage_pct: Optional[float] = None,
    ) -> Dict:
        """
        Выполняет бэктест по уже имеющемуся DataFrame.
        Если индикаторы ещё не рассчитаны — рассчитывает их.
        Если они уже есть — использует как есть.
        """
        spec_source = "ai"
        if strategy_spec is None or validate_strategy_spec(strategy_spec) is not None:
            reason = validate_strategy_spec(strategy_spec) if strategy_spec else "spec отсутствует"
            logger.warning(f"[{symbol}] strategy_spec невалиден ({reason}). Использую default.")
            strategy_spec = default_strategy_spec()
            spec_source = "default"

        if slippage_pct is None:
            slippage_pct = self.slippage_pct

        if df is None or df.empty:
            return {"error": f"Пустой DataFrame для {symbol}"}

        if len(df) < 60:
            return {"error": f"Недостаточно свечей ({len(df)}), нужно минимум 60"}

        try:
            if not self._has_precomputed_indicators(df):
                df = self._build_indicator_frame(df)
            else:
                df = df.copy().reset_index(drop=True)
        except Exception as e:
            logger.exception("Ошибка подготовки индикаторов")
            return {"error": f"Ошибка подготовки индикаторов: {e}"}

        trades, equity, capital = self._simulate(
            df, strategy_spec, initial_capital, commission_pct, slippage_pct
        )
        stats = self._compute_stats(trades, equity, capital, initial_capital)

        stats["risk_controls_enabled"] = bool(
            self.last_simulation_meta.get("risk_controls_enabled", False)
        )
        stats["cooldown_activations"] = int(
            self.last_simulation_meta.get("cooldown_activations", 0)
        )
        stats["skipped_candles_by_cooldown"] = int(
            self.last_simulation_meta.get("skipped_candles_by_cooldown", 0)
        )

        return {
            "trades": trades,
            "equity_curve": equity,
            "stats": stats,
            "strategy_spec_used": strategy_spec,
            "spec_source": spec_source,
            "start_date": start_date,
            "end_date": end_date,
        }

    def _has_precomputed_indicators(self, df: pd.DataFrame) -> bool:
        required = [
            "sma_20", "sma_50", "rsi_14", "atr_14",
            "bb_upper", "bb_middle", "bb_lower",
            "macd", "macd_signal", "macd_hist",
            "adx_14", "vwap",
            "pivot", "resistance_1", "resistance_2", "resistance_3",
            "support_1", "support_2", "support_3",
            "low_32", "high_32", "close_position",
        ] + get_1h_feature_columns()

        return all(col in df.columns for col in required)

    # ------------------------------------------------------------------ #
    # Индикаторы (выровнены по индексу, NaN сохраняются)
    # ------------------------------------------------------------------ #
    def _build_indicator_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if talib is None:
            raise RuntimeError("TA-Lib не установлен")

        df = df.copy().reset_index(drop=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)
        close = df["close"].values.astype(np.float64)

        df["sma_20"] = talib.SMA(close, timeperiod=20)
        df["sma_50"] = talib.SMA(close, timeperiod=50)
        df["rsi_14"] = talib.RSI(close, timeperiod=14)
        df["atr_14"] = talib.ATR(high, low, close, timeperiod=14)

        bb_u, bb_m, bb_l = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = bb_u, bb_m, bb_l

        macd, macd_sig, macd_hist = talib.MACD(close, 12, 26, 9)
        df["macd"], df["macd_signal"], df["macd_hist"] = macd, macd_sig, macd_hist

        df["adx_14"] = talib.ADX(high, low, close, timeperiod=14)

        
        # === НОВЫЕ ИНДИКАТОРЫ ДЛЯ ПРОБОЙНЫХ СТРАТЕГИЙ (Итерация 3) ===
        # 32-свечной минимум/максимум (shift(1) гарантирует отсутствие lookahead)
        df["low_32"] = df["low"].shift(1).rolling(32, min_periods=32).min()
        df["high_32"] = df["high"].shift(1).rolling(32, min_periods=32).max()
        # === A6: моментум-шорт — падение за 2 свечи в единицах ATR ===
        # То же, что vel_2 в impulse_lab. < -2.0 = падение >2 ATR за 2 свечи.
        # shift не нужен: close.diff(2) = close[t] - close[t-2], lookahead нет.
        df["drop_2_atr"] = df["close"].diff(2) / df["atr_14"]

        
        # Внутрибаровая сила: где закрылась свеча относительно своего диапазона (0.0 = лоу, 1.0 = хай)
        df["candle_range"] = (df["high"] - df["low"]).replace(0, np.nan)
        df["close_position"] = (df["close"] - df["low"]) / df["candle_range"]
        # ================================================================

        df["avg_volume_20"] = df["volume"].rolling(20).mean()
        df["avg_volume_96"] = df["volume"].rolling(96).mean()

        # === Портфель: аномальная активность (объём × волатильность) ===
        # Используется в сигнале act_short. Оба компонента нормализованы
        # по своим средним (96 свечей).
        df["vol_ratio"] = df["volume"] / df["avg_volume_96"]
        df["atr_ratio"] = df["atr_14"] / df["atr_14"].rolling(96, min_periods=24).mean()
        df["activity"] = df["vol_ratio"] * df["atr_ratio"]

        df["vwap"] = self._compute_vwap(df)
        self._add_daily_pivots(df)

        # 1h признаки из indicators.py
        df_1h = resample_15m_to_1h(df)
        if df_1h.empty or len(df_1h) < 60:
            for col in get_1h_feature_columns():
                if col not in df.columns:
                    df[col] = np.nan
        else:
            df_1h = compute_1h_indicators(df_1h, talib=talib)
            df = merge_1h_features_to_15m(df, df_1h)


        # === 4h-признаки: ресэмплинг 15m → 4h ===
        df_4h = (
            df.set_index("timestamp")
            .resample("4h")
            .agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            })
            .dropna()
        )

        if len(df_4h) > 2:
            # ATR на 4H
            high_4h, low_4h, close_4h = df_4h["high"], df_4h["low"], df_4h["close"]
            prev_close_4h = close_4h.shift(1)
            tr_4h = pd.concat([
                high_4h - low_4h,
                (high_4h - prev_close_4h).abs(),
                (low_4h - prev_close_4h).abs(),
            ], axis=1).max(axis=1)
            df_4h["atr_14"] = tr_4h.rolling(14, min_periods=14).mean()

            # Средний объём за 20 баров 4H
            df_4h["avg_volume_20"] = df_4h["volume"].rolling(20, min_periods=20).mean()

            # Скорость движения за 1 бар 4H (в единицах ATR)
            df_4h["vel_1"] = close_4h.diff(1) / df_4h["atr_14"].replace(0, np.nan)

            # Позиция закрытия в свече 4H
            candle_range_4h = (high_4h - low_4h).replace(0, np.nan)
            df_4h["close_position"] = (close_4h - low_4h) / candle_range_4h

            # Forward fill на 15m таймлайн
            idx_15m = df["timestamp"]
            df["volume_4h"] = df_4h["volume"].reindex(idx_15m, method="ffill").values
            df["avg_volume_20_4h"] = df_4h["avg_volume_20"].reindex(idx_15m, method="ffill").values
            df["vel_1_4h"] = df_4h["vel_1"].reindex(idx_15m, method="ffill").values
            df["close_position_4h"] = df_4h["close_position"].reindex(idx_15m, method="ffill").values
            df["atr_14_4h"] = df_4h["atr_14"].reindex(idx_15m, method="ffill").values

            # Флаг границы 4H свечи: 1 если текущий 15m бар — первый после
            # закрытия 4H свечи (т.е. минута == 0 и час кратен 4)
            hours = df["timestamp"].dt.hour
            minutes = df["timestamp"].dt.minute
            df["is_4h_boundary"] = ((hours % 4 == 0) & (minutes == 0)).astype(int)
        else:
            # Недостаточно данных для 4H — заполняем NaN/0
            for col in ("volume_4h", "avg_volume_20_4h", "vel_1_4h",
                        "close_position_4h", "atr_14_4h"):
                df[col] = np.nan
            df["is_4h_boundary"] = 0

        return df

    @staticmethod
    def _compute_vwap(df: pd.DataFrame) -> pd.Series:
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        day = df["timestamp"].dt.date
        cum_tv = (typical * df["volume"]).groupby(day).cumsum()
        cum_v = df["volume"].groupby(day).cumsum()
        vwap = cum_tv / cum_v.replace(0, np.nan)
        return vwap

    @staticmethod
    def _add_daily_pivots(df: pd.DataFrame) -> None:
        df["_date"] = df["timestamp"].dt.date
        daily = df.groupby("_date").agg(
            high=("high", "max"), low=("low", "min"), close=("close", "last")
        )
        # Pivot текущего дня считается по ПРЕДЫДУЩЕМУ дню
        prev = daily.shift(1)
        piv = (prev["high"] + prev["low"] + prev["close"]) / 3.0
        levels = pd.DataFrame(index=daily.index)
        levels["pivot"] = piv
        levels["resistance_1"] = 2 * piv - prev["low"]
        levels["resistance_2"] = piv + (prev["high"] - prev["low"])
        levels["resistance_3"] = prev["high"] + 2 * (piv - prev["low"])
        levels["support_1"] = 2 * piv - prev["high"]
        levels["support_2"] = piv - (prev["high"] - prev["low"])
        levels["support_3"] = prev["low"] - 2 * (prev["high"] - piv)

        for col in levels.columns:
            df[col] = df["_date"].map(levels[col])
        df.drop(columns=["_date"], inplace=True)


    def _get_risk_controls(self, spec: Dict) -> Dict:
        """
        Возвращает нормализованные настройки защиты от серий убытков.

        Если блока risk_controls нет или enabled=False, стратегия ведёт себя
        строго как раньше: новые входы не ограничиваются.
        """
        raw = spec.get("risk_controls") or {}

        enabled = bool(raw.get("enabled", False))
        if not enabled:
            return {
                "enabled": False,
                "max_consecutive_losses": 0,
                "pause_candles_after_losses": 0,
                "max_losses_in_window": 0,
                "loss_window_trades": 0,
            }

        return {
            "enabled": True,
            "max_consecutive_losses": max(
                1, int(raw.get("max_consecutive_losses", 3))
            ),
            "pause_candles_after_losses": max(
                1, int(raw.get("pause_candles_after_losses", 96))
            ),
            "max_losses_in_window": max(
                1, int(raw.get("max_losses_in_window", 5))
            ),
            "loss_window_trades": max(
                1, int(raw.get("loss_window_trades", 10))
            ),
        }

    def _should_pause_after_loss(
        self,
        trade: Dict,
        consecutive_losses: int,
        recent_loss_flags: List[int],
        controls: Dict,
    ) -> bool:
        """
        Решает, нужно ли ставить паузу после закрытия убыточной сделки.

        Пауза срабатывает:
        1. При серии max_consecutive_losses убыточных сделок.
        2. При max_losses_in_window убыточных сделках
           среди последних loss_window_trades.
        """
        if not controls.get("enabled", False):
            return False

        if trade.get("result_pct", 0) > 0:
            return False

        if consecutive_losses >= controls["max_consecutive_losses"]:
            return True

        window = controls["loss_window_trades"]
        max_losses = controls["max_losses_in_window"]

        last_window = recent_loss_flags[-window:]
        if len(last_window) >= window and sum(last_window) >= max_losses:
            return True

        return False

    # ------------------------------------------------------------------ #
    # Симуляция
    # ------------------------------------------------------------------ #
    def _simulate(self, df: pd.DataFrame, spec: Dict, initial_capital: float, commission_pct: float, slippage_pct: float = 0.0,):
        long_spec = spec.get("long") or {}
        short_spec = spec.get("short") or {}
        filters = spec.get("filters") or []

        trades: List[Dict] = []
        equity = [initial_capital]
        capital = initial_capital
        position = None
        pending_entry = None  # сигнал, который исполнится по open следующей свечи

        controls = self._get_risk_controls(spec)
        consecutive_losses = 0
        recent_loss_flags: List[int] = []
        cooldown_until_index = None
        cooldown_activations = 0
        skipped_candles_by_cooldown = 0

        records = df.to_dict("records")

        # Относительное проскальзывание (0.05% → 0.0005).
        eps = float(slippage_pct or 0.0) / 100.0

        for i in range(1, len(records)):
            row = records[i]
            prev = records[i - 1]

            # --- Исполнение отложенного входа по OPEN текущей свечи ---
            if position is None and pending_entry is not None:
                direction = pending_entry["direction"]
                raw_open = float(row["open"])
                # Слиппедж «против сделки»: long платит дороже open,
                # short продаёт дешевле.
                if direction == "long":
                    entry_fill = raw_open * (1.0 + eps)
                else:
                    entry_fill = raw_open * (1.0 - eps)

                entry_row = dict(row)
                entry_row["close"] = entry_fill
                position = self._open_position(
                    direction, entry_row, pending_entry["side_spec"]
                )
                position["entry_reasons"] = pending_entry["reasons"]
                pending_entry = None

            # --- Управление открытой позицией ---
            if position is not None:
                position["bars_held"] = position.get("bars_held", 0) + 1
                self._apply_position_management(position, row)

                exit_price, reason = self._check_exit(position, row)
                if exit_price is not None:
                    # Слиппедж применяется к SL/TP. Мягкие выходы
                    # (_check_exit вернёт time_stop / level_failure /
                    # trend_loss / macd_flip) идут по close и не смещаются.
                    exit_price = self._apply_slippage_to_exit(
                        exit_price, reason, position["direction"], eps
                    )
                    capital, trade = self._close_position(
                        position, exit_price, reason, capital, commission_pct, row["timestamp"]
                    )
                    trades.append(trade)
                    equity.append(capital)
                    position = None

                    # Учёт серии убытков и настройка паузы (risk_controls).
                    if trade.get("result_pct", 0) <= 0:
                        consecutive_losses += 1
                        recent_loss_flags.append(1)
                    else:
                        consecutive_losses = 0
                        recent_loss_flags.append(0)

                    window = controls.get("loss_window_trades", 0)
                    if window > 0 and len(recent_loss_flags) > window:
                        recent_loss_flags = recent_loss_flags[-window:]

                    if self._should_pause_after_loss(
                        trade, consecutive_losses, recent_loss_flags, controls
                    ):
                        cooldown_until_index = i + controls.get("pause_candles_after_losses", 0)
                        cooldown_activations += 1

                    continue

            # Пока позиция открыта — новые сигналы не ищем.
            if position is not None:
                continue

            # Пауза после серии убытков: не ищем новые входы.
            if cooldown_until_index is not None and i < cooldown_until_index:
                skipped_candles_by_cooldown += 1
                continue

            # --- Глобальные фильтры ---
            if filters and not evaluate_conditions(filters, row, prev):
                continue

            # --- Сигнал входа Long (оценивается по close свечи, исполнится на следующей) ---
            matched_long = self._entry_signal(long_spec, row, prev)
            if matched_long is not None:
                pending_entry = {
                    "direction": "long",
                    "side_spec": long_spec,
                    "reasons": explain_conditions(matched_long, row, prev),
                }
                continue

            # --- Сигнал входа Short ---
            matched_short = self._entry_signal(short_spec, row, prev)
            if matched_short is not None:
                pending_entry = {
                    "direction": "short",
                    "side_spec": short_spec,
                    "reasons": explain_conditions(matched_short, row, prev),
                }
                continue

        # --- Закрытие в конце периода ---
        if position is not None:
            last = records[-1]
            capital, trade = self._close_position(
                position, last["close"], "end_of_period", capital, commission_pct, last["timestamp"]
            )
            trades.append(trade)
            equity.append(capital)

        self.last_simulation_meta = {
            "risk_controls_enabled": bool(controls.get("enabled", False)),
            "cooldown_activations": int(cooldown_activations),
            "skipped_candles_by_cooldown": int(skipped_candles_by_cooldown),
        }

        return trades, equity, capital

    @staticmethod
    def _open_position(direction: str, row: Dict, side_spec: Dict) -> Dict:
        entry = float(row["close"])

        atr = row.get("atr_14")
        atr_valid = (
            atr is not None
            and not (isinstance(atr, float) and math.isnan(atr))
            and atr > 0
        )

        sl_atr_mult = side_spec.get("stop_loss_atr_mult")
        tp_atr_mult = side_spec.get("take_profit_atr_mult")

        stop_dist = None
        take_dist = None

        # Приоритет — ATR-стопы (нормализуют поведение по всему ценовому диапазону).
        if atr_valid and sl_atr_mult is not None:
            try:
                stop_dist = float(sl_atr_mult) * float(atr)
            except Exception:
                stop_dist = None
        if atr_valid and tp_atr_mult is not None:
            try:
                take_dist = float(tp_atr_mult) * float(atr)
            except Exception:
                take_dist = None

        # Fallback — фиксированные проценты.
        if stop_dist is None:
            sl_pct = float(side_spec.get("stop_loss_pct", 2.0)) / 100.0
            stop_dist = entry * sl_pct
        if take_dist is None:
            tp_list = side_spec.get("take_profit_pct", [3.0]) or [3.0]
            tp_pct = float(tp_list[0]) / 100.0
            take_dist = entry * tp_pct

        if direction == "long":
            stop = entry - stop_dist
            take = entry + take_dist
        else:
            stop = entry + stop_dist
            take = entry - take_dist

        snapshot = {}
        for k in SNAPSHOT_KEYS:
            v = row.get(k)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                snapshot[k] = round(float(v), 6)

        return {
            "direction": direction,
            "entry_price": entry,
            "entry_time": row["timestamp"],
            "stop_loss": stop,
            "take_profit": take,
            "initial_stop_loss": stop,
            "initial_take_profit": take,
            "entry_snapshot": snapshot,
            "management": side_spec.get("management", {}),
            "bars_held": 0,
        }

    @staticmethod
    def _entry_signal(side_spec: Dict, row: Dict, prev: Dict):
        """
        Проверяет вход по side_spec.
        Возвращает список условий сработавшей группы (для explain) или None.

        Приоритет: entry_any (список OR-групп). Если его нет — старое поле entry.
        Полная обратная совместимость со старыми spec.
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

    def _apply_position_management(self, pos: Dict, row: Dict) -> None:
        """
        Опциональное сопровождение позиции. Ничего не делает без блока management.
        Поддерживает:
        - move_to_breakeven_at_tp_pct: доля пути к TP, после которой SL → безубыток;
        - breakeven_offset_pct: смещение безубытка в % (напр. 0.1 покрывает комиссии);
        - trailing_atr_mult: ATR-трейлинг стопа (SL подтягивается за ценой).
        """
        mgmt = pos.get("management", {}) or {}

        # --- 1. Перенос SL в безубыток ---
        trigger = mgmt.get("move_to_breakeven_at_tp_pct")
        if trigger is not None:
            try:
                trigger = float(trigger)
            except Exception:
                trigger = None

            if trigger is not None and 0 < trigger <= 1:
                entry = pos["entry_price"]
                take = pos["take_profit"]
                try:
                    offset = float(mgmt.get("breakeven_offset_pct", 0.0)) / 100.0
                except Exception:
                    offset = 0.0

                if pos["direction"] == "long":
                    trigger_price = entry + (take - entry) * trigger
                    be_price = entry * (1 + offset)
                    if row.get("high") is not None and row["high"] >= trigger_price:
                        if pos["stop_loss"] < be_price:
                            pos["stop_loss"] = be_price
                else:
                    trigger_price = entry - (entry - take) * trigger
                    be_price = entry * (1 - offset)
                    if row.get("low") is not None and row["low"] <= trigger_price:
                        if pos["stop_loss"] > be_price:
                            pos["stop_loss"] = be_price

        # --- 2. ATR-трейлинг стопа ---
        trail_mult = mgmt.get("trailing_atr_mult")
        if trail_mult:
            try:
                trail_mult = float(trail_mult)
            except Exception:
                trail_mult = None

            atr = row.get("atr_14")
            close = row.get("close")
            atr_ok = (
                atr is not None
                and not (isinstance(atr, float) and math.isnan(atr))
                and atr > 0
            )

            if trail_mult and trail_mult > 0 and atr_ok and close is not None:
                if pos["direction"] == "long":
                    new_sl = float(close) - trail_mult * float(atr)
                    if new_sl > pos["stop_loss"]:
                        pos["stop_loss"] = new_sl
                else:
                    new_sl = float(close) + trail_mult * float(atr)
                    if new_sl < pos["stop_loss"]:
                        pos["stop_loss"] = new_sl

    @staticmethod
    def _check_exit(pos: Dict, row: Dict):
        """
        Возвращает (exit_price, reason) или (None, '').

        Логика:
        1. Жёсткие выходы по SL/TP.
        2. Мягкие выходы по management. Выход по потере уровня теперь
           срабатывает не при первом же дипе под уровень, а только когда цена
           уходит за уровень дальше буфера (level_failure_buffer_pct).
        """
        # --- Жёсткие выходы: SL / TP ---
        if pos["direction"] == "long":
            if row["low"] <= pos["stop_loss"]:
                return pos["stop_loss"], "stop_loss"
            if row["high"] >= pos["take_profit"]:
                return pos["take_profit"], "take_profit"
        else:
            if row["high"] >= pos["stop_loss"]:
                return pos["stop_loss"], "stop_loss"
            if row["low"] <= pos["take_profit"]:
                return pos["take_profit"], "take_profit"

        # --- Мягкие выходы по management ---
        mgmt = pos.get("management", {}) or {}
        close_price = row.get("close")
        if close_price is None:
            return None, ""

        # --- Time-stop: выход по времени, если сделка "не пошла" ---
        max_hold = mgmt.get("max_hold_candles")
        if max_hold:
            try:
                if pos.get("bars_held", 0) >= int(max_hold):
                    return float(close_price), "time_stop"
            except (TypeError, ValueError):
                pass

        buffer_pct = float(mgmt.get("level_failure_buffer_pct", 1.0)) / 100.0

        def _valid(v):
            return v is not None and not (isinstance(v, float) and math.isnan(v))

        if pos["direction"] == "long":
            if mgmt.get("exit_on_level_failure"):
                lvl = row.get(mgmt.get("level_field", "resistance_1"))
                if _valid(lvl):
                    # выходим только если цена ушла НИЖЕ уровня дальше буфера
                    if close_price < lvl * (1 - buffer_pct):
                        return float(close_price), "level_failure"

            if mgmt.get("exit_on_trend_loss"):
                sma50 = row.get("sma_50")
                if _valid(sma50) and close_price < sma50:
                    return float(close_price), "trend_loss"

            if mgmt.get("exit_on_macd_flip"):
                macd_hist = row.get("macd_hist")
                if _valid(macd_hist) and macd_hist < 0:
                    return float(close_price), "macd_flip"

        else:
            if mgmt.get("exit_on_level_failure"):
                lvl = row.get(mgmt.get("level_field", "support_1"))
                if _valid(lvl):
                    if close_price > lvl * (1 + buffer_pct):
                        return float(close_price), "level_failure"

            if mgmt.get("exit_on_trend_loss"):
                sma50 = row.get("sma_50")
                if _valid(sma50) and close_price > sma50:
                    return float(close_price), "trend_loss"

            if mgmt.get("exit_on_macd_flip"):
                macd_hist = row.get("macd_hist")
                if _valid(macd_hist) and macd_hist > 0:
                    return float(close_price), "macd_flip"

        return None, ""

    @staticmethod
    def _apply_slippage_to_exit(
        price: float,
        reason: str,
        direction: str,
        eps: float,
    ) -> float:
        """
        Смещает цену выхода в сторону убытка для SL/TP.

        Для мягких выходов (time_stop, level_failure, trend_loss,
        macd_flip) и end_of_period возвращает цену без изменения:
        в живом трейдере они исполняются market-ордером по уже
        наблюдаемой close-цене, а не по уровню.
        """
        if eps <= 0 or reason not in ("stop_loss", "take_profit"):
            return price
        if direction == "long":
            return price * (1.0 - eps)
        return price * (1.0 + eps)

    @staticmethod
    def _close_position(pos: Dict, exit_price: float, reason: str,
                        capital: float, commission_pct: float, exit_time):
        if pos["direction"] == "long":
            pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"]
        else:
            pnl_pct = (pos["entry_price"] - exit_price) / pos["entry_price"]
        # Комиссия на вход и выход
        pnl_pct = (pnl_pct - 2 * commission_pct / 100) * 100
        capital += capital * (pnl_pct / 100)
        trade = {
            "entry_time": pos["entry_time"],
            "exit_time": exit_time,
            "direction": pos["direction"],
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "result_pct": round(pnl_pct, 2),
            "exit_reason": reason,
            "capital_after": round(capital, 2),
            "entry_reasons": pos.get("entry_reasons", []),
            "entry_snapshot": pos.get("entry_snapshot", {}),
        }
        return capital, trade

    # ------------------------------------------------------------------ #
    # Статистика
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_stats(trades, equity, capital, initial_capital) -> Dict:
        total = len(trades)
        if total == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl_pct": 0.0,
                "return_pct": 0.0,
                "final_capital": round(initial_capital, 2),
                "max_drawdown_pct": 0.0,
                "profit_factor": 0.0,
                "avg_trade_pct": 0.0,
                "expectancy_pct": 0.0,
                "avg_win_pct": 0.0,
                "avg_loss_pct": 0.0,
                "long_trades": 0,
                "short_trades": 0,
            }

        wins = [t for t in trades if t["result_pct"] > 0]
        losses = [t for t in trades if t["result_pct"] <= 0]

        total_pnl = sum(t["result_pct"] for t in trades)

        peak = equity[0]
        max_dd = 0.0
        for eq in equity:
            peak = max(peak, eq)
            dd = (peak - eq) / peak * 100 if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        gross_profit = sum(t["result_pct"] for t in wins)
        gross_loss = abs(sum(t["result_pct"] for t in losses))
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_trade = total_pnl / total
        avg_win = (sum(t["result_pct"] for t in wins) / len(wins)) if wins else 0.0
        avg_loss = (abs(sum(t["result_pct"] for t in losses)) / len(losses)) if losses else 0.0

        return {
            "total_trades": total,
            "win_rate": round(len(wins) / total * 100, 1),
            "total_pnl_pct": round(total_pnl, 2),
            "return_pct": round((capital - initial_capital) / initial_capital * 100, 2),
            "final_capital": round(capital, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "profit_factor": round(pf, 2) if pf != float("inf") else 999.0,
            "avg_trade_pct": round(avg_trade, 4),
            "expectancy_pct": round(avg_trade, 4),
            "avg_win_pct": round(avg_win, 4),
            "avg_loss_pct": round(avg_loss, 4),
            "long_trades": sum(1 for t in trades if t["direction"] == "long"),
            "short_trades": sum(1 for t in trades if t["direction"] == "short"),
        }