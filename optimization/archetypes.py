"""
Архетипы Фазы 0: фиксированные торговые идеи с малым числом параметров.
У каждого архетипа:
- build(p) -> strategy_spec (dict);
- grid()   -> список наборов параметров для грубого перебора.
Структура условий зашита в коде: наружу торчат только 2-3 числовые ручки
(стоп, тейк, один фильтр). Это барьер переобучения: идею нельзя
«переписать», можно лишь настроить риск под волатильность.

Две семьи идей:
- A1-A5: импульсные/трендовые (пробои, откаты, моментум);
- R1-R4: mean-reversion (возврат к среднему внутри диапазона).
Каждая группа входа обязана проходить барьер strategy_spec:
объёмный фильтр + трендовый фильтр + привязка к уровню/импульсу.
"""
from __future__ import annotations
from typing import Dict, List


def _mgmt() -> Dict:
    return {
        "move_to_breakeven_at_tp_pct": None,
        "breakeven_offset_pct": 0.0,
        "trailing_atr_mult": None,
        "max_hold_candles": 192,
        "exit_on_level_failure": False,
        "exit_on_trend_loss": False,
        "exit_on_macd_flip": False,
    }


def _side(conds: List[Dict], s: float, t: float) -> Dict:
    return {
        "entry": conds,
        "stop_loss_atr_mult": s,
        "take_profit_atr_mult": t,
        "management": _mgmt(),
    }


def _risk() -> Dict:
    return {
        "enabled": True,
        "max_consecutive_losses": 8,
        "pause_candles_after_losses": 16,
        "max_losses_in_window": 10,
        "loss_window_trades": 20,
    }


# ------------------------------------------------------------------
# A1. Тренд 1h + откат к SMA20 на 15m (свинг)
# ------------------------------------------------------------------
def _build_pullback(p: Dict) -> Dict:
    s, t, vm = p["stop"], p["tp"], p["vol"]
    return {
        "long": _side([
            {"type": "compare", "left": "close", "op": ">", "right": "sma_50_1h"},
            {"type": "compare", "left": "macd_hist_1h", "op": ">", "right": 0},
            {"type": "compare", "left": "low", "op": "<=", "right": "sma_20"},
            {"type": "compare", "left": "close", "op": ">", "right": "sma_20"},
            {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": vm},
        ], s, t),
        "short": _side([
            {"type": "compare", "left": "close", "op": "<", "right": "sma_50_1h"},
            {"type": "compare", "left": "macd_hist_1h", "op": "<", "right": 0},
            {"type": "compare", "left": "high", "op": ">=", "right": "sma_20"},
            {"type": "compare", "left": "close", "op": "<", "right": "sma_20"},
            {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": vm},
        ], s, t),
        "filters": [],
        "risk_controls": _risk(),
    }


def _grid_pullback() -> List[Dict]:
    return [
        {"stop": s, "tp": t, "vol": v}
        for s in (3.0, 4.5, 6.0)
        for t in (6.0, 9.0, 12.0)
        for v in (1.0, 1.3)
    ]


# ------------------------------------------------------------------
# A2. Пробой волатильности (Боллинджер) по тренду 1h
# Объём зафиксирован 1.2: пробой без объёма — ложный пробой.
# ------------------------------------------------------------------
def _build_breakout(p: Dict) -> Dict:
    s, t, ax = p["stop"], p["tp"], p["adx"]
    return {
        "long": _side([
            {"type": "compare", "left": "close", "op": ">", "right": "sma_50_1h"},
            {"type": "compare", "left": "close", "op": ">", "right": "bb_upper"},
            {"type": "compare", "left": "adx_14_1h", "op": ">", "right": ax},
            {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 1.2},
        ], s, t),
        "short": _side([
            {"type": "compare", "left": "close", "op": "<", "right": "sma_50_1h"},
            {"type": "compare", "left": "close", "op": "<", "right": "bb_lower"},
            {"type": "compare", "left": "adx_14_1h", "op": ">", "right": ax},
            {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 1.2},
        ], s, t),
        "filters": [],
        "risk_controls": _risk(),
    }


def _grid_breakout() -> List[Dict]:
    return [
        {"stop": s, "tp": t, "adx": a}
        for s in (3.0, 4.5, 6.0)
        for t in (6.0, 9.0, 12.0)
        for a in (18.0, 22.0)
    ]


# ------------------------------------------------------------------
# A3. Mean-reversion от экстремума RSI ВНУТРИ тренда 1h
# ------------------------------------------------------------------
def _build_meanrev(p: Dict) -> Dict:
    s, t, r = p["stop"], p["tp"], p["rsi"]
    return {
        "long": _side([
            {"type": "compare", "left": "close", "op": ">", "right": "sma_50_1h"},
            {"type": "compare", "left": "rsi_14", "op": "<", "right": r},
            {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 1.0},
        ], s, t),
        "short": _side([
            {"type": "compare", "left": "close", "op": "<", "right": "sma_50_1h"},
            {"type": "compare", "left": "rsi_14", "op": ">", "right": 100.0 - r},
            {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 1.0},
        ], s, t),
        "filters": [],
        "risk_controls": _risk(),
    }


def _grid_meanrev() -> List[Dict]:
    return [
        {"stop": s, "tp": t, "rsi": r}
        for s in (2.5, 3.5, 4.5)
        for t in (4.0, 6.0, 8.0)
        for r in (30.0, 35.0, 40.0)
    ]


# ------------------------------------------------------------------
# A4. Чистый тренд-моментум 1h (максимально просто)
# ------------------------------------------------------------------
def _build_momentum(p: Dict) -> Dict:
    s, t = p["stop"], p["tp"]
    return {
        "long": _side([
            {"type": "compare", "left": "close", "op": ">", "right": "sma_50_1h"},
            {"type": "compare", "left": "macd_hist_1h", "op": ">", "right": 0},
            {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 1.0},
        ], s, t),
        "short": _side([
            {"type": "compare", "left": "close", "op": "<", "right": "sma_50_1h"},
            {"type": "compare", "left": "macd_hist_1h", "op": "<", "right": 0},
            {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 1.0},
        ], s, t),
        "filters": [],
        "risk_controls": _risk(),
    }


def _grid_momentum() -> List[Dict]:
    return [
        {"stop": s, "tp": t}
        for s in (4.0, 6.0, 8.0)
        for t in (8.0, 12.0, 16.0)
    ]


# ------------------------------------------------------------------
# R1. Mean-reversion от полос Боллинджера во флэте
# Вход — возврат цены внутрь канала после выхода за границу.
# ------------------------------------------------------------------
def _build_bb_meanrev(p: Dict) -> Dict:
    s, t, ax = p["stop"], p["tp"], p["adx"]
    return {
        "long": _side([
            {"type": "cross", "a": "close", "b": "bb_lower", "dir": "above"},
            {"type": "range", "value": "rsi_14", "min": 15.0, "max": 45.0},
            {"type": "compare", "left": "adx_14", "op": "<", "right": ax},
            {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 0.8},
        ], s, t),
        "short": _side([
            {"type": "cross", "a": "close", "b": "bb_upper", "dir": "below"},
            {"type": "range", "value": "rsi_14", "min": 55.0, "max": 85.0},
            {"type": "compare", "left": "adx_14", "op": "<", "right": ax},
            {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 0.8},
        ], s, t),
        "filters": [],
        "risk_controls": _risk(),
    }


def _grid_bb_meanrev() -> List[Dict]:
    return [
        {"stop": s, "tp": t, "adx": a}
        for s in (2.5, 3.5, 4.5)
        for t in (4.0, 6.0, 8.0)
        for a in (18.0, 22.0, 26.0)
    ]


# ------------------------------------------------------------------
# R2. RSI-реверсия от экстремума во флэте (цель — vwap)
# ------------------------------------------------------------------
def _build_rsi_reversion(p: Dict) -> Dict:
    s, t, ax, r = p["stop"], p["tp"], p["adx"], p["rsi"]
    return {
        "long": _side([
            {"type": "range", "value": "rsi_14", "min": 10.0, "max": r},
            {"type": "compare", "left": "close", "op": "<", "right": "vwap"},
            {"type": "compare", "left": "adx_14", "op": "<", "right": ax},
            {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 0.8},
        ], s, t),
        "short": _side([
            {"type": "range", "value": "rsi_14", "min": 100.0 - r, "max": 90.0},
            {"type": "compare", "left": "close", "op": ">", "right": "vwap"},
            {"type": "compare", "left": "adx_14", "op": "<", "right": ax},
            {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 0.8},
        ], s, t),
        "filters": [],
        "risk_controls": _risk(),
    }


def _grid_rsi_reversion() -> List[Dict]:
    return [
        {"stop": s, "tp": t, "adx": a, "rsi": r}
        for s in (2.5, 3.5, 4.5)
        for t in (4.0, 6.0, 8.0)
        for a in (18.0, 22.0, 26.0)
        for r in (25.0, 30.0, 35.0)
    ]


# ------------------------------------------------------------------
# R3. Отскок от границ диапазона (support_1 / resistance_1) во флэте
# Вход в узкой зоне у уровня, но НЕ после пробоя уровня.
# ------------------------------------------------------------------
def _build_range_boundaries(p: Dict) -> Dict:
    s, t, ax = p["stop"], p["tp"], p["adx"]
    return {
        "long": _side([
            {"type": "compare", "left": "close", "op": "<=", "right": {"name": "support_1", "mult": 1.005}},
            {"type": "compare", "left": "close", "op": ">=", "right": {"name": "support_1", "mult": 0.995}},
            {"type": "compare", "left": "adx_14", "op": "<", "right": ax},
            {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 0.8},
        ], s, t),
        "short": _side([
            {"type": "compare", "left": "close", "op": ">=", "right": {"name": "resistance_1", "mult": 0.995}},
            {"type": "compare", "left": "close", "op": "<=", "right": {"name": "resistance_1", "mult": 1.005}},
            {"type": "compare", "left": "adx_14", "op": "<", "right": ax},
            {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 0.8},
        ], s, t),
        "filters": [],
        "risk_controls": _risk(),
    }


def _grid_range_boundaries() -> List[Dict]:
    return [
        {"stop": s, "tp": t, "adx": a}
        for s in (2.5, 3.5, 4.5)
        for t in (4.0, 6.0, 8.0)
        for a in (18.0, 22.0, 26.0)
    ]


# ------------------------------------------------------------------
# R4. Реверсия к VWAP после растяжения во флэте
# ------------------------------------------------------------------
def _build_vwap_reversion(p: Dict) -> Dict:
    s, t, ax = p["stop"], p["tp"], p["adx"]
    return {
        "long": _side([
            {"type": "compare", "left": "close", "op": "<=", "right": {"name": "vwap", "mult": 0.990}},
            {"type": "compare", "left": "close", "op": ">=", "right": {"name": "vwap", "mult": 0.975}},
            {"type": "compare", "left": "adx_14", "op": "<", "right": ax},
            {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 0.8},
        ], s, t),
        "short": _side([
            {"type": "compare", "left": "close", "op": ">=", "right": {"name": "vwap", "mult": 1.010}},
            {"type": "compare", "left": "close", "op": "<=", "right": {"name": "vwap", "mult": 1.025}},
            {"type": "compare", "left": "adx_14", "op": "<", "right": ax},
            {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 0.8},
        ], s, t),
        "filters": [],
        "risk_controls": _risk(),
    }


def _grid_vwap_reversion() -> List[Dict]:
    return [
        {"stop": s, "tp": t, "adx": a}
        for s in (2.5, 3.5, 4.5)
        for t in (4.0, 6.0, 8.0)
        for a in (18.0, 22.0, 26.0)
    ]

# ------------------------------------------------------------------
# A5. Пробой 32-свечного экстремума с всплеском объема и сильным закрытием
# (Основан на успешных результатах impulse_lab итерации 3, без lookahead)
# ------------------------------------------------------------------
def _build_strong_vol_breakout(p: Dict) -> Dict:
    s, t, vm, h = p["stop"], p["tp"], p["vol"], p["hold"]

    long_side = _side([
        {"type": "compare", "left": "close", "op": ">", "right": "high_32"},
        {"type": "compare", "left": "close_position", "op": ">", "right": 0.6},
        {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": vm},
    ], s, t)
    long_side["management"]["max_hold_candles"] = h

    short_side = _side([
        {"type": "compare", "left": "close", "op": "<", "right": "low_32"},
        {"type": "compare", "left": "close_position", "op": "<", "right": 0.4},
        {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": vm},
    ], s, t)
    short_side["management"]["max_hold_candles"] = h

    return {
        "long": long_side,
        "short": short_side,
        "filters": [],
        "risk_controls": _risk(),
    }


def _grid_strong_vol_breakout() -> List[Dict]:
    # Согласовано с impulse_lab (итерация 3): узкие ATR-стопы,
    # короткий горизонт. 3*3*2*2 = 36 комбинаций.
    return [
        {"stop": s, "tp": t, "vol": v, "hold": h}
        for s in (0.75, 1.0, 1.5)
        for t in (1.5, 2.25, 3.0)
        for v in (1.2, 1.5, 2.0)
        for h in (8, 16)
    ]


# ------------------------------------------------------------------
# A6. Моментум-шорт: продолжение ВНИЗ после обвала с объёмом.
# Гипотеза impulse_lab iter 5 (m2s2_vol2.0): падение >2 ATR за 2 свечи
# + volume > 2.0*avg_volume_96. Лонг-пробои и лонг-моментум на 15m
# убыточны всегда и везде (эмпирический вывод итерации 4), поэтому
# archетип SHORT-ONLY — ключ "long" НЕ включается в спеку.
# Это не режим-гейт: валидатор strategy_spec допускает отсутствие
# одной из сторон, а движок просто не генерирует сигналы по пустому блоку.
# ------------------------------------------------------------------
def _build_a6_short_momentum(p: Dict) -> Dict:
    s, t, vm, h = p["stop"], p["tp"], p["vol"], p["hold"]
    short_side = _side([
        {"type": "compare", "left": "drop_2_atr", "op": "<", "right": -2.0},
        {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": vm},
    ], s, t)
    short_side["management"]["max_hold_candles"] = h
    # Лаборатория выходов: трейлинг и безубыток ВРЕДНЫ для этого сигнала
    # (выбивают на отскоке после обвала). Фиксируем на null.
    short_side["management"]["trailing_atr_mult"] = None
    short_side["management"]["move_to_breakeven_at_tp_pct"] = None
    return {
        "short": short_side,
        "filters": [],
        "risk_controls": _risk(),
        "frozen_entries": True,
        # Коридор выходов вокруг лабораторного оптимума (Вариант 2).
        # Оптимизатор может двигать параметры ТОЛЬКО в этих пределах.
        "exit_corridor": {
            "stop_loss_atr_mult": [1.8, 2.2],
            "take_profit_atr_mult": [1.3, 1.7],
            "max_hold_candles": [6, 12],
        },
    }

def _grid_a6_short_momentum() -> List[Dict]:
    # Выходы зафиксированы лабораторией выходов (34 датасета + стресс-тест):
    #   stop=2.0 — широкий стоп переживает отскок после обвала;
    #   tp=1.5   — быстрый тейк фиксирует продолжение до разворота;
    #   hold=8   — интрадей (2 часа), без переноса на ночь.
    # Лабораторный оптимум: exp=+0.19% при slip=0.05%, frac_pos=0.91.
    # Узкие стопы (0.75-1.0) выбиваются отскоком — НЕ использовать.
    return [{"stop": 2.0, "tp": 1.5, "vol": 2.0, "hold": 8}]

# ------------------------------------------------------------------
# Реестр архетипов. Каждый билдер возвращает чистую спеку без
# служебных обёрток: структура идеи зашита в коде, наружу торчат
# только числовые ручки из grid().
# ------------------------------------------------------------------
ARCHETYPES = [
    {"name": "A1_TREND_PULLBACK",      "build": _build_pullback,             "grid": _grid_pullback},
    {"name": "A2_VOL_BREAKOUT",        "build": _build_breakout,             "grid": _grid_breakout},
    {"name": "A3_MEAN_REVERSION",      "build": _build_meanrev,              "grid": _grid_meanrev},
    {"name": "A4_TREND_MOMENTUM",      "build": _build_momentum,             "grid": _grid_momentum},
    {"name": "A5_STRONG_VOL_BREAKOUT", "build": _build_strong_vol_breakout,  "grid": _grid_strong_vol_breakout},
    {"name": "A6_SHORT_MOMENTUM",      "build": _build_a6_short_momentum,    "grid": _grid_a6_short_momentum},
    {"name": "R1_BB_MEAN_REVERSION",   "build": _build_bb_meanrev,           "grid": _grid_bb_meanrev},
    {"name": "R2_RSI_REVERSION",       "build": _build_rsi_reversion,        "grid": _grid_rsi_reversion},
    {"name": "R3_RANGE_BOUNDARIES",    "build": _build_range_boundaries,     "grid": _grid_range_boundaries},
    {"name": "R4_VWAP_REVERSION",      "build": _build_vwap_reversion,       "grid": _grid_vwap_reversion},
]