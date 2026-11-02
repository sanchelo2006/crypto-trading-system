"""
Лаборатория импульсных индикаторов (исследование D1).
Итерация 3: честные сигналы (без lookahead) + новый индикатор внутрибаровой силы.
Итерация 6 (выходы): _simulate_fixed поддерживает трейлинг (trail_atr) и
безубыток (be_atr). При trail_atr=None и be_atr=None поведение идентично
прежнему — все существующие вызовы работают без изменений.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from research.positioning import attach_positioning

# Параметры по умолчанию
H_DEFAULT = 8          # горизонт: 8 x 15m = 2 часа
MOVE_ATR = 2.5         # импульс: |смещение| >= 2.5 ATR
ER_MIN = 0.45
SL_ATR = 0.75          # фикс-сделка
TP_ATR = 1.5
FEE_PCT = 0.2
# Стресс-тест исполнения: проскальзывание round-trip в % поверх комиссии.
# В kline Bybit гэпов нет (open[t] == close[t-1] всегда), поэтому стоимость
# входа измеряем не гэпом, а явным допущением о проскальзывании
SLIPPAGE_PCT = 0.08


def add_features(df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """Добавляет все признаки для тестирования сигналов."""
    if symbol is not None:
        df = attach_positioning(df, symbol)

    d = df.copy()

    if symbol is not None:
        d = attach_positioning(d, symbol)

    close, high, low = d["close"], d["high"], d["low"]
    atr, vol = d["atr_14"], d["volume"]

    # --- POS-1: дополнительные ценовые признаки -----------------------
    low_96 = low.rolling(96, min_periods=96).min()
    high_96 = high.rolling(96, min_periods=96).max()
    range_96 = (high_96 - low_96).replace(0.0, np.nan)
    d["pos_in_range"] = (close - low_96) / range_96

    d["ret_4h"] = close / close.shift(16) - 1.0

    abs_change_96 = (close - close.shift(96)).abs()
    path_length_96 = close.diff().abs().rolling(96, min_periods=96).sum()
    d["er_96"] = abs_change_96 / path_length_96.replace(0.0, np.nan)

    d["pos_vol_ratio"] = (
        vol.rolling(4, min_periods=4).mean()
        / vol.rolling(96, min_periods=96).mean().replace(0.0, np.nan)
    )

    # === Базовые индикаторы ===
    # Bollinger Bands
    d["sma_20"] = close.rolling(20).mean()
    d["std_20"] = close.rolling(20).std()
    d["bb_upper"] = d["sma_20"] + 2 * d["std_20"]
    d["bb_lower"] = d["sma_20"] - 2 * d["std_20"]
    d["bb_middle"] = d["sma_20"]

    # ATR ratio
    d["atr_ratio"] = atr / atr.rolling(96, min_periods=24).mean()

    # Volume ratio
    d["avg_volume_96"] = vol.rolling(96, min_periods=24).mean()
    d["vol_ratio"] = vol / d["avg_volume_96"]

    # === Пробой экстремумов 32 свечей (без lookahead) ===
    prev_high = high.shift(1).rolling(32, min_periods=32).max()
    prev_low = low.shift(1).rolling(32, min_periods=32).min()
    d["brk32_long"] = close > prev_high
    d["brk32_short"] = close < prev_low

    # === НОВЫЙ ИНДИКАТОР: Внутрибаровая сила (Intra-bar Strength) ===
    # Показывает, где закрылась свеча относительно своего диапазона.
    # 1.0 = закрытие в самом хае, 0.0 = закрытие в самом лоу.
    d["candle_range"] = (high - low).replace(0, np.nan)
    d["close_position"] = (close - low) / d["candle_range"]

    # Для шорта мы хотим видеть давление продавцов: закрытие в нижней 40% диапазона свечи
    d["strong_bearish_close"] = d["close_position"] < 0.4
    # Для лонга мы хотим видеть давление покупателей: закрытие в верхней 40% диапазона свечи
    d["strong_bullish_close"] = d["close_position"] > 0.6

    # === Комбинации (Честные, без заглядывания в будущее) ===
    # 1. Пробой + Всплеск объема
    d["vol_spike"] = d["vol_ratio"] > 1.5  # Снизил порог с 2.0 до 1.5 для увеличения выборки
    d["brk32_vol_short"] = d["brk32_short"] & d["vol_spike"]

    # 2. Пробой + Внутрибаровая сила (НОВЫЙ СИГНАЛ)
    d["brk32_strong_short"] = d["brk32_short"] & d["strong_bearish_close"]

    # 3. Пробой + Объем + Внутрибаровая сила (МАКСИМАЛЬНАЯ ФИЛЬТРАЦИЯ)
    d["brk32_vol_strong_short"] = d["brk32_short"] & d["vol_spike"] & d["strong_bearish_close"]

    # === Дополнительные признаки (для полноты) ===
    # RSI
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.rolling(14).mean() / down.rolling(14).mean().replace(0, np.nan)
    d["rsi_14"] = 100 - 100 / (1 + rs)

    # Squeeze
    bw = (d["bb_upper"] - d["bb_lower"]) / d["bb_middle"]
    d["bb_width_pct"] = bw.rolling(500, min_periods=100).rank(pct=True)
    kc_up = d["sma_20"] + 1.5 * atr
    kc_lo = d["sma_20"] - 1.5 * atr
    d["keltner_squeeze"] = (d["bb_upper"] < kc_up) & (d["bb_lower"] > kc_lo)
    d["squeeze_recent"] = d["keltner_squeeze"].rolling(12, min_periods=1).max().astype(bool)

    # Streak: 5+ последовательных свечей в одном направлении
    up_bar = (close > d["open"]).astype(int)
    dn_bar = (close < d["open"]).astype(int)
    d["streak_up"] = (up_bar.rolling(5, min_periods=5).sum() >= 5).astype(bool)
    d["streak_dn"] = (dn_bar.rolling(5, min_periods=5).sum() >= 5).astype(bool)

    # Spike: движение >3 ATR за 8 свечей
    d["spike_up"] = (close.diff(8) > 3 * atr).fillna(False)
    d["spike_dn"] = (close.diff(8) < -3 * atr).fillna(False)

    # Расстояние до EMA
    d["ema_8"] = close.ewm(span=8, adjust=False).mean()
    d["dist_ema8_atr"] = (close - d["ema_8"]) / atr

    # === Итерация 4: скорость движения (возврат за K свечей в единицах ATR) ===
    for k in (2, 4, 8):
        d[f"vel_{k}"] = close.diff(k) / atr

    # === Итерация 5: внутрибаровая сила (если её нет из фрейма движка) ===
    if "close_position" not in d.columns:
        _candle_range = (high - low).replace(0, np.nan)
        d["close_position"] = (close - low) / _candle_range

    # === Итерация 6: признаки для регим-независимых сигналов ===
    # Нормализованный диапазон свечи (в единицах ATR).
    # Нужен для сигнала кульминационной свечи.
    d["range_atr"] = (high - low) / atr.replace(0, np.nan)
    # Аномальная активность: объём × волатильность.
    # Оба компонента уже нормализованы по своим средним (96 свечей).
    d["activity"] = d["vol_ratio"] * d["atr_ratio"]

    return d


def oracle_events(d: pd.DataFrame, h: int = H_DEFAULT,
                  move_atr: float = MOVE_ATR, er_min: float = ER_MIN) -> np.ndarray:
    """Оракул: определяет, был ли импульс в следующие h свечей."""
    close = d["close"].to_numpy(float)
    atr = d["atr_14"].to_numpy(float)
    n = len(close)
    imp = np.zeros(n, dtype=int)
    abs_diff = np.abs(np.diff(close, prepend=close[0]))
    cum = np.cumsum(abs_diff)

    for t in range(n - h):
        a = atr[t]
        if not np.isfinite(a) or a <= 0:
            continue
        net = close[t + h] - close[t]
        path = cum[t + h] - cum[t]
        if path <= 0:
            continue
        er = abs(net) / path
        if er >= er_min and abs(net) >= move_atr * a:
            imp[t] = 1 if net > 0 else -1

    return imp


def _simulate_fixed(d: pd.DataFrame, idx: np.ndarray, direction: int,
                    sl_atr: float = SL_ATR, tp_atr: float = TP_ATR,
                    max_hold: int = H_DEFAULT, entry_mode: str = "next_open",
                    trail_atr: float | None = None,
                    be_atr: float | None = None,
                    cost: float | None = None) -> dict:
    """Симулирует фикс-сделку с опциональным трейлингом и безубытком.

    Новые параметры (по умолчанию None = поведение как раньше):
      trail_atr — трейлинг: стоп подтягивается за лучшей ценой на
                  расстоянии trail_atr*ATR. ATR берётся на момент входа.
      be_atr    — безубыток: при достижении прибыли be_atr*ATR стоп
                  переносится на цену входа.
    """
    close = d["close"].to_numpy(float)
    high = d["high"].to_numpy(float)
    low = d["low"].to_numpy(float)
    opn = d["open"].to_numpy(float)
    atr = d["atr_14"].to_numpy(float)
    n = len(close)
    if cost is None:
        cost = FEE_PCT + SLIPPAGE_PCT
    results, helds, exits = [], [], {"sl": 0, "tp": 0, "time": 0}

    for s in idx:
        if s + 1 >= n or s + max_hold >= n:
            continue
        a = atr[s]
        if not np.isfinite(a) or a <= 0:
            continue
        if entry_mode == "signal_close":
            entry = close[s]
        else:
            entry = opn[s + 1]
        sl = entry - direction * sl_atr * a
        tp = entry + direction * tp_atr * a
        best = entry  # лучшая цена с момента входа (для трейлинга)
        res, held, reason = None, max_hold, "time"

        for k in range(s + 1, min(s + 1 + max_hold, n)):
            lk, hk = low[k], high[k]
            if direction == 1:
                # Безубыток: при достижении +be_atr*ATR стоп -> на вход
                if be_atr is not None and hk >= entry + be_atr * a:
                    sl = max(sl, entry)
                # Трейлинг: стоп подтягивается за максимумом
                if trail_atr is not None:
                    best = max(best, hk)
                    sl = max(sl, best - trail_atr * a)
                # Проверка выходов (SL приоритетнее TP — как в исходной логике)
                if lk <= sl:
                    res, held, reason = (sl - entry) / entry * 100 - cost, k - s, "sl"
                    break
                if hk >= tp:
                    res, held, reason = (tp - entry) / entry * 100 - cost, k - s, "tp"
                    break
            else:
                # Безубыток: при достижении -be_atr*ATR стоп -> на вход
                if be_atr is not None and lk <= entry - be_atr * a:
                    sl = min(sl, entry)
                # Трейлинг: стоп подтягивается за минимумом
                if trail_atr is not None:
                    best = min(best, lk)
                    sl = min(sl, best + trail_atr * a)
                # Проверка выходов
                if hk >= sl:
                    res, held, reason = (entry - sl) / entry * 100 - cost, k - s, "sl"
                    break
                if lk <= tp:
                    res, held, reason = (entry - tp) / entry * 100 - cost, k - s, "tp"
                    break

        if res is None:
            exitp = close[min(s + max_hold, n - 1)]
            res = (exitp - entry) / entry * 100 * direction - cost

        results.append(res)
        helds.append(held)
        exits[reason] += 1

    if not results:
        return {"n_traded": 0, "wr": None, "pf": None, "exp": None,
                "med_bars": None, "tp_rate": None, "sl_rate": None}

    r = np.array(results)
    wins, losses = r[r > 0], r[r <= 0]
    pf = (wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else np.inf
    tot = exits["sl"] + exits["tp"] + exits["time"]

    return {
        "n_traded": len(r),
        "wr": round(100 * len(wins) / len(r), 1),
        "pf": round(float(pf), 2) if np.isfinite(pf) else 999.0,
        "exp": round(float(r.mean()), 3),
        "med_bars": float(np.median(helds)),
        "tp_rate": round(exits["tp"] / tot, 2) if tot else None,
        "sl_rate": round(exits["sl"] / tot, 2) if tot else None,
    }


def score_signal(d: pd.DataFrame, imp: np.ndarray, mask: np.ndarray,
                 name: str, direction: int | None,
                 sl_atr: float = SL_ATR, tp_atr: float = TP_ATR,
                 max_hold: int = H_DEFAULT, entry_mode: str = "next_open",
                 trail_atr: float | None = None,
                 be_atr: float | None = None) -> dict:
    """Оценивает сигнал по оракулу и симуляции."""
    idx = np.where(mask)[0]
    n = len(idx)
    per_day = n / max(1.0, len(d) / 96.0)

    if n == 0:
        return {"signal": name, "n": 0, "per_day": 0.0}

    if direction is None:
        prior = float((imp != 0).mean())
        prec = float((imp[idx] != 0).mean())
        return {"signal": name, "n": n, "per_day": round(per_day, 2),
                "precision": round(prec, 3), "prior": round(prior, 3),
                "lift": round(prec / prior, 2) if prior > 0 else None}

    prior = float((imp == direction).mean())
    prec = float((imp[idx] == direction).mean())
    row = {"signal": name, "n": n, "per_day": round(per_day, 2),
           "precision": round(prec, 3), "prior": round(prior, 3),
           "lift": round(prec / prior, 2) if prior > 0 else None}
    row.update(_simulate_fixed(d, idx, direction, sl_atr, tp_atr, max_hold,
                               entry_mode, trail_atr, be_atr))

    return row


def _chase_diagnosis(d: pd.DataFrame, idx: np.ndarray, direction: int,
                     sl_atr: float = SL_ATR, tp_atr: float = TP_ATR,
                     max_hold: int = H_DEFAULT) -> dict:
    """
    Диагностика погони за ценой (chase).
    Сравнивает идеальный вход (на закрытии сигнальной свечи) с реальным
    входом (на открытии следующей свечи). Разница показывает, сколько
    эджа съедает гэп на входе.
    """
    close = d["close"].to_numpy(float)
    opn = d["open"].to_numpy(float)
    n = len(close)

    # Стоимость гэпа на входе:
    # Для шорта (direction=-1): если open[s+1] < close[s] (цена упала),
    #   то гэп плохой для нас, стоимость = (close[s] - open[s+1]) / close[s] * 100
    # Для лонга (direction=1): если open[s+1] > close[s] (цена выросла),
    #   то гэп плохой для нас, стоимость = (open[s+1] - close[s]) / close[s] * 100
    # Формула: стоимость гэпа = (open[s+1] - close[s]) / close[s] * 100 * direction
    gap_costs = []
    valid_idx = []
    for s in idx:
        if s + 1 >= n:
            continue
        gap = (opn[s + 1] - close[s]) / close[s] * 100
        cost = gap * direction  # Положительно, если гэп плохой для нас
        gap_costs.append(cost)
        valid_idx.append(s)

    if not gap_costs:
        return {
            "exp_next_open": None, "exp_signal_close": None,
            "chase_cost": None, "avg_gap": None, "med_gap": None, "p90_gap": None,
        }

    gap_arr = np.array(gap_costs)

    valid_idx_arr = np.array(valid_idx)
    res_next_open = _simulate_fixed(d, valid_idx_arr, direction, sl_atr, tp_atr, max_hold, entry_mode="next_open")
    res_signal_close = _simulate_fixed(d, valid_idx_arr, direction, sl_atr, tp_atr, max_hold, entry_mode="signal_close")

    exp_next_open = res_next_open.get("exp")
    exp_signal_close = res_signal_close.get("exp")

    chase_cost = None
    if exp_next_open is not None and exp_signal_close is not None:
        chase_cost = round(exp_signal_close - exp_next_open, 3)

    return {
        "exp_next_open": exp_next_open,
        "exp_signal_close": exp_signal_close,
        "chase_cost": chase_cost,
        "avg_gap": round(float(np.mean(gap_arr)), 4),
        "med_gap": round(float(np.median(gap_arr)), 4),
        "p90_gap": round(float(np.percentile(gap_arr, 90)), 4),
    }


def iteration_signals(d: pd.DataFrame) -> list:
    n = len(d)

    # Базовые линии
    rng = np.random.default_rng(42)
    rand_idx = rng.random(n) < (3.5 / 96.0)

    sigs = [
        ("random_long", rand_idx, 1),
        ("random_short", rand_idx, -1),
    ]

    # Базовый сигнал — лидер итерации 4, без фильтров (для сравнения)
    base = d["vel_2"].to_numpy() < -2.0
    sigs.append(("mom2_short_t2.0", base, -1))

    # Колонки фильтров (приходят из фрейма движка; сравнения с NaN дают False —
    # прогревочные бары сигналов не дают)
    vol_ratio = (d["volume"] / d["avg_volume_96"]).to_numpy()
    adx = d["adx_14"].to_numpy()
    rsi = d["rsi_14"].to_numpy()
    close_pos = d["close_position"].to_numpy()

    # === Итерация 5: фильтры поверх шорт-моментума ===
    # 1. Объём: обвал с давлением — продолжение вероятнее
    sigs.append(("m2s2_vol1.5", base & (vol_ratio > 1.5), -1))
    sigs.append(("m2s2_vol2.0", base & (vol_ratio > 2.0), -1))

    # 2. ADX: направленность рынка
    sigs.append(("m2s2_adx20", base & (adx > 20), -1))
    sigs.append(("m2s2_adx25", base & (adx > 25), -1))

    # 3. RSI-развилка: запас хода против каскада
    sigs.append(("m2s2_rsi_gt30", base & (rsi > 30), -1))
    sigs.append(("m2s2_rsi_le30", base & (rsi <= 30), -1))

    # 4. Внутрибаровая сила: свеча закрыта у лоя
    sigs.append(("m2s2_cp_lo", base & (close_pos < 0.3), -1))

    # 5. Комбинации
    sigs.append(("m2s2_vol_cp", base & (vol_ratio > 1.5) & (close_pos < 0.3), -1))
    sigs.append(("m2s2_vol_rsi", base & (vol_ratio > 1.5) & (rsi > 30), -1))

    # === Итерация 6: регим-независимые сигналы ===
    vel_2_arr = d["vel_2"].to_numpy()
    activity = d["activity"].to_numpy()
    range_atr = d["range_atr"].to_numpy()

    # 1. Двунаправленный волатильный моментум.
    # Шорт-вариант уже есть (mom2_short_t2.0, m2s2_vol2.0).
    # Добавляем лонг-вариант: в бычий год работает он, в медвежий — шорт.
    sigs.append(("cont_long_t2.0", vel_2_arr > 2.0, 1))
    sigs.append(("cont_long_vol2.0", (vel_2_arr > 2.0) & (vol_ratio > 2.0), 1))

    # 2. Аномальная активность + направление по моментуму.
    # Гипотеза: всплеск активности (объём × волатильность) предшествует
    # движению. Направление определяется знаком краткосрочного моментума.
    sigs.append(("act_long", (activity > 3.0) & (vel_2_arr > 1.0), 1))
    sigs.append(("act_short", (activity > 3.0) & (vel_2_arr < -1.0), -1))

    # 3. Кульминационная свеча → продолжение.
    # Гипотеза: широкий диапазон + объём + закрытие у экстремума —
    # признак кульминации, после которой движение продолжается.
    sigs.append(("climax_bear", (range_atr > 1.5) & (vol_ratio > 2.0) & (close_pos < 0.2), -1))
    sigs.append(("climax_bull", (range_atr > 1.5) & (vol_ratio > 2.0) & (close_pos > 0.8), 1))

    # === POS-1: позиционный перегруз ===
    # Защита: если позиционные признаки не добавлены (нет файла данных
    # или не передан symbol), просто не генерируем POS-сигналы.
    pos_cols = ("fund_z", "ls_z", "oi_d4", "ret_4h",
                "pos_in_range", "er_96", "pos_vol_ratio", "atr_14")
    if all(c in d.columns for c in pos_cols):
        fz = d["fund_z"].to_numpy()
        lz = d["ls_z"].to_numpy()
        od = d["oi_d4"].to_numpy()
        r4 = d["ret_4h"].to_numpy()
        pir = d["pos_in_range"].to_numpy()
        pvr = d["pos_vol_ratio"].to_numpy()
        atr_pct = (d["atr_14"] / d["close"]).to_numpy()

        active = (d["er_96"].to_numpy() >= 0.20)

        crowd_long  = (fz > 1.5) & (lz > 1.0) & (od > 0.01)
        crowd_short = (fz < -1.5) & (lz < -1.0) & (od > 0.01)

        capitulation_mask = (
            (r4 < -2.0 * atr_pct)
            & (pvr > 2.5)
            & (od < -0.01)
            & (fz < 0.0)
            & (pir < 0.25)
        )

        sigs.append(("POS_SHORT", active & crowd_long & (r4 > 0) & (pir > 0.65), -1))
        sigs.append(("POS_LONG",  active & crowd_short & (r4 < 0) & (pir < 0.35), +1))
        sigs.append(("POS_CAPIT", active & capitulation_mask, +1))

    return sigs


def run_lab(d: pd.DataFrame, diagnose_chase: bool = False,
            symbol: str | None = None) -> list:
    d = add_features(d, symbol=symbol)
    imp = oracle_events(d, h=H_DEFAULT)
    rows = []
    for name, mask, direction in iteration_signals(d):
        row = score_signal(d, imp, mask, name, direction,
                           sl_atr=SL_ATR, tp_atr=TP_ATR, max_hold=H_DEFAULT)
        if diagnose_chase and direction is not None:
            idx = np.where(mask)[0]
            if len(idx) > 0:
                chase = _chase_diagnosis(d, idx, direction,
                                         sl_atr=SL_ATR, tp_atr=TP_ATR, max_hold=H_DEFAULT)
                row.update(chase)
        rows.append(row)
    return rows