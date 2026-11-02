"""
Машиночитаемая спецификация торговой стратегии и движок её исполнения.

strategy_spec — строгий JSON, который AI генерирует при инициализации монеты
параллельно с человекочитаемыми правилами. Бэктест исполняет именно его.

Формат strategy_spec:
{
  "long":  {"entry": [<condition>, ...], "stop_loss_pct": 2.0, "take_profit_pct": [3.0]},
  "short": {"entry": [<condition>, ...], "stop_loss_pct": 2.0, "take_profit_pct": [3.0]},
  "filters": [<condition>, ...]   # глобальные фильтры (например ADX), проверяются для обоих направлений
}

Все условия в списке "entry"/"filters" объединяются по И (AND).

Типы условий (<condition>):
  compare : {"type":"compare","left":<operand>,"op":">|>=|<|<=","right":<operand>}
  cross   : {"type":"cross","a":<name>,"b":<name>,"dir":"above|below"}
  range   : {"type":"range","value":<name>,"min":<num>,"max":<num>}
  volume  : {"type":"volume","op":">|>=|<|<=","ref":<name>,"mult":<num>}
  slope   : {"type":"slope","value":<name>,"dir":"up|down","lookback":<int>}

<operand> — либо число, либо имя серии (см. AVAILABLE_SERIES),
            либо объект {"name":<str>,"mult":<num>,"offset":<num>} для right/left.

# Доступные имена серий (AVAILABLE_SERIES) на каждой свече:
# 15m (оригинальные):
#   open, high, low, close, volume,
#   sma_20, sma_50, rsi_14, atr_14,
#   bb_upper, bb_middle, bb_lower,
#   macd, macd_signal, macd_hist, adx_14, vwap,
#   pivot, resistance_1..3, support_1..3,
#   avg_volume_20, avg_volume_96   (скользящее среднее объёма)

# 1h (сгенерированные из 15m):
#   sma_20_1h, sma_50_1h, rsi_14_1h, atr_14_1h,
#   bb_upper_1h, bb_middle_1h, bb_lower_1h,
#   macd_1h, macd_signal_1h, macd_hist_1h, adx_14_1h, vwap_1h,
#   pivot_1h, resistance_1_1h, resistance_2_1h, resistance_3_1h,
#   support_1_1h, support_2_1h, support_3_1h,
#   avg_volume_20_1h, avg_volume_96_1h
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional
import json

logger = logging.getLogger(__name__)

AVAILABLE_SERIES = {
    # 15m (оригинальные)
    "open", "high", "low", "close", "volume",
    "sma_20", "sma_50", "rsi_14", "atr_14",
    "bb_upper", "bb_middle", "bb_lower",
    "macd", "macd_signal", "macd_hist", "adx_14", "vwap",
    "pivot", "resistance_1", "resistance_2", "resistance_3",
    "support_1", "support_2", "support_3",
    "avg_volume_20", "avg_volume_96",
    "low_32", "high_32", "close_position",
    
    # Новые индикаторы для пробойных стратегий (Итерация 3, без lookahead)
    "low_32", "high_32", "candle_range", "close_position",
    # A6: импульс-шорт (моментум после обвала)
    "drop_2_atr",
    # Портфель: аномальная активность (объём × волатильность)
    "vol_ratio", "atr_ratio", "activity",

    # 1h (сгенерированные из 15m)
    "sma_20_1h", "sma_50_1h", "rsi_14_1h", "atr_14_1h",
    "bb_upper_1h", "bb_middle_1h", "bb_lower_1h",
    "macd_1h", "macd_signal_1h", "macd_hist_1h", "adx_14_1h", "vwap_1h",
    "pivot_1h", "resistance_1_1h", "resistance_2_1h", "resistance_3_1h",
    "support_1_1h", "support_2_1h", "support_3_1h",
    "avg_volume_20_1h", "avg_volume_96_1h",
    "vol_ratio", "atr_ratio", "activity",
    
}

_OPS = {
    ">":  lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<":  lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
}


def _is_nan(x: Any) -> bool:
    return isinstance(x, float) and math.isnan(x)


def _resolve(operand: Any, ctx: Dict[str, float]) -> Optional[float]:
    """Возвращает числовое значение операнда или None, если недоступно/NaN."""
    if operand is None:
        return None
    # Число
    if isinstance(operand, (int, float)) and not isinstance(operand, bool):
        return float(operand)
    # Имя серии
    if isinstance(operand, str):
        val = ctx.get(operand)
        return None if (val is None or _is_nan(val)) else float(val)
    # Объект {"name":..., "mult":..., "offset":...}
    if isinstance(operand, dict):
        base = _resolve(operand.get("name"), ctx)
        if base is None:
            return None
        return base * float(operand.get("mult", 1.0)) + float(operand.get("offset", 0.0))
    return None


def _eval_condition(cond: Dict, ctx: Dict[str, float], prev_ctx: Optional[Dict[str, float]]) -> bool:
    """
    Оценивает одно условие.
    При недостатке данных возвращает False.
    """
    if not isinstance(cond, dict):
        return False

    ctype = cond.get("type")

    if ctype == "compare":
        left = _resolve(cond.get("left"), ctx)
        right = _resolve(cond.get("right"), ctx)
        op = _OPS.get(cond.get("op"))
        if left is None or right is None or op is None:
            return False
        return op(left, right)

    if ctype == "range":
        val = _resolve(cond.get("value"), ctx)
        if val is None:
            return False
        lo = float(cond.get("min", -math.inf))
        hi = float(cond.get("max", math.inf))
        return lo <= val <= hi

    if ctype == "volume":
        vol = ctx.get("volume")
        ref = _resolve(cond.get("ref"), ctx)
        op = _OPS.get(cond.get("op", ">"))
        if vol is None or ref is None or op is None or _is_nan(vol):
            return False
        return op(float(vol), ref * float(cond.get("mult", 1.0)))

    if ctype == "cross":
        if prev_ctx is None:
            return False
        a_now = _resolve(cond.get("a"), ctx)
        b_now = _resolve(cond.get("b"), ctx)
        a_prev = _resolve(cond.get("a"), prev_ctx)
        b_prev = _resolve(cond.get("b"), prev_ctx)
        if None in (a_now, b_now, a_prev, b_prev):
            return False

        if cond.get("dir") == "above":
            return a_prev <= b_prev and a_now > b_now
        else:
            return a_prev >= b_prev and a_now < b_now

    if ctype == "slope":
        if prev_ctx is None:
            return False
        now = _resolve(cond.get("value"), ctx)
        before = _resolve(cond.get("value"), prev_ctx)
        if now is None or before is None:
            return False
        return now > before if cond.get("dir") == "up" else now < before

    if ctype == "distance":
        a = _resolve(cond.get("a"), ctx)
        b = _resolve(cond.get("b"), ctx)
        if a is None or b is None or b == 0:
            return False

        dist_pct = abs(a - b) / abs(b) * 100.0
        max_pct = cond.get("max_pct")
        min_pct = cond.get("min_pct")

        if max_pct is not None and dist_pct > float(max_pct):
            return False
        if min_pct is not None and dist_pct < float(min_pct):
            return False
        return True

    logger.warning(f"Неизвестный тип условия в strategy_spec: {ctype}")
    return False


def evaluate_conditions(conditions: List[Dict], ctx: Dict[str, float],
                        prev_ctx: Optional[Dict[str, float]]) -> bool:
    """Все условия объединяются по И (AND). Пустой список → False (нет сигнала)."""
    if not conditions:
        return False
    return all(_eval_condition(c, ctx, prev_ctx) for c in conditions)


def validate_strategy_spec(spec: Any) -> Optional[str]:
    """
    Возвращает текст ошибки или None, если спецификация валидна.
    Валидация мягкая: не ломаем старые spec, но отсекаем явный мусор.
    """
    if not isinstance(spec, dict):
        return "strategy_spec не является объектом"

    if not spec.get("long") and not spec.get("short"):
        return "strategy_spec не содержит ни long, ни short"

    allowed_types = {"compare", "cross", "range", "volume", "slope", "distance"}

    def _validate_condition(c: Any) -> Optional[str]:
        if not isinstance(c, dict):
            return "условие должно быть объектом"
        ctype = c.get("type")
        if ctype not in allowed_types:
            return f"неизвестный тип условия: {ctype}"
        

        # Валидация числовых полей в range
        if ctype == "range":
            for key in ("min", "max"):
                val = c.get(key)
                if val is not None and val != "":
                    try:
                        float(val)
                    except (ValueError, TypeError):
                        return f"range.{key} должен быть числом, получено: '{val}'"
        
        # Валидация mult в volume
        if ctype == "volume":
            mult = c.get("mult")
            if mult is not None:
                try:
                    float(mult)
                except (ValueError, TypeError):
                    return f"volume.mult должен быть числом, получено: '{mult}'"
        
        return None

    for side in ("long", "short"):
        block = spec.get(side)
        if block is None:
            continue

        if not isinstance(block, dict):
            return f"Блок {side} должен быть объектом"

        entry = block.get("entry", [])
        if not isinstance(entry, list):
            return f"{side}.entry должен быть списком"

        for idx, cond in enumerate(entry):
            err = _validate_condition(cond)
            if err:
                return f"{side}.entry[{idx}]: {err}"

        # OR-группы входов: entry_any = [[cond, ...], [cond, ...]]
        entry_any = block.get("entry_any")
        if entry_any is not None:
            if not isinstance(entry_any, list):
                return f"{side}.entry_any должен быть списком групп условий"
            for gidx, group in enumerate(entry_any):
                if not isinstance(group, list) or not group:
                    return f"{side}.entry_any[{gidx}] должен быть непустым списком условий"
                for idx, cond in enumerate(group):
                    err = _validate_condition(cond)
                    if err:
                        return f"{side}.entry_any[{gidx}][{idx}]: {err}"

        if "stop_loss_pct" in block:
            try:
                float(block["stop_loss_pct"])
            except Exception:
                return f"{side}.stop_loss_pct должен быть числом"

        if "take_profit_pct" in block:
            tp = block["take_profit_pct"]
            if not isinstance(tp, list) or len(tp) != 1:
                return f"{side}.take_profit_pct должен быть списком из одного числа"
            try:
                float(tp[0])
            except Exception:
                return f"{side}.take_profit_pct[0] должен быть числом"

        for mult_key in ("stop_loss_atr_mult", "take_profit_atr_mult"):
            if mult_key in block:
                try:
                    v = float(block[mult_key])
                    if v <= 0:
                        return f"{side}.{mult_key} должен быть положительным числом"
                except Exception:
                    return f"{side}.{mult_key} должен быть числом"

        if "management" in block and not isinstance(block["management"], dict):
            return f"{side}.management должен быть объектом"

    filters = spec.get("filters", [])
    if filters is not None:
        if not isinstance(filters, list):
            return "filters должен быть списком"
        for idx, cond in enumerate(filters):
            err = _validate_condition(cond)
            if err:
                return f"filters[{idx}]: {err}"

    if "risk_controls" in spec and not isinstance(spec["risk_controls"], dict):
        return "risk_controls должен быть объектом"

    # --------------------------------------------------
    # Дополнительная проверка качества стратегии
    # --------------------------------------------------
    for side in ("long", "short"):
        block = spec.get(side)
        if not block:
            continue

        groups = block.get("entry_any")
        if not groups:
            entry = block.get("entry") or []
            groups = [entry]

        for group_index, group in enumerate(groups):
            if not group:
                continue

            has_trend_filter = False
            has_volume_filter = False
            has_level_or_momentum = False

            for cond in group:
                ctype = cond.get("type")

                # Собираем ВСЕ имена серий, на которые ссылается условие
                # (включая cross: a/b, volume: ref, объекты {name, mult})
                names = set()
                for key in ("left", "right", "value", "a", "b", "ref"):
                    v = cond.get(key)
                    if isinstance(v, str):
                        names.add(v)
                    elif isinstance(v, dict) and isinstance(v.get("name"), str):
                        names.add(v["name"])

                if ctype == "volume" or names & {
                    "volume",
                    "avg_volume_96",
                    "avg_volume_20",
                    "activity",  # activity = vol_ratio × atr_ratio, содержит объём
                }:
                    has_volume_filter = True

                if names & {
                    "sma_50",
                    "sma_50_1h",
                    "sma_20_1h",
                    "adx_14",
                    "adx_14_1h",
                    "macd_hist",
                    "macd_hist_1h",
                    "macd",
                    "macd_signal",
                    "macd_1h",
                    "macd_signal_1h",
                    # Пробой 32-свечного экстремума — импульсное условие
                    # (результат impulse_lab, итерация 3)
                    "low_32",
                    "high_32",
                    # A6: моментум-шорт (падение >2 ATR за 2 свечи, impulse_lab iter 5)
                    "drop_2_atr",
                }:
                    has_trend_filter = True

                if names & {
                    "resistance_1",
                    "support_1",
                    "pivot",
                    "close",
                    "rsi_14",
                    "macd_hist",
                    # A6: импульс-шорт (падение >2 ATR за 2 свечи, impulse_lab iter 5)
                    "drop_2_atr",
                }:
                    has_level_or_momentum = True

            if not has_volume_filter:
                return f"{side}[{group_index}]: нет объёмного фильтра (обязателен)"

            if not has_trend_filter:
                return f"{side}[{group_index}]: нет трендового фильтра (обязателен)"

            if not has_level_or_momentum:
                return f"{side}[{group_index}]: нет уровня или импульсного подтверждения (обязательно)"

        if not block.get("stop_loss_atr_mult") and not block.get("stop_loss_pct"):
            return f"{side}: нет стоп-лосса"

        if not block.get("take_profit_atr_mult") and not block.get("take_profit_pct"):
            return f"{side}: нет тейк-профита"

    return None


def default_strategy_spec() -> Dict:
    """
    Резервная спецификация.
    Использует ATR-стопы (нормализуют риск по ценовому диапазону).
    exit_on_level_failure ВЫКЛЮЧЕН по умолчанию — он резал сделки в убыток
    у самого уровня входа, не давая тезису реализоваться.
    """
    return {
        "long": {
            "entry": [
                {"type": "compare", "left": "close", "op": ">", "right": "sma_50"},
                {"type": "compare", "left": "close", "op": ">", "right": "resistance_1"},
                {"type": "compare", "left": "adx_14", "op": ">", "right": 20},
                {"type": "compare", "left": "macd_hist", "op": ">", "right": 0},
                {"type": "range", "value": "rsi_14", "min": 45, "max": 65},
                {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 1.2},
            ],
            "stop_loss_atr_mult": 1.5,
            "take_profit_atr_mult": 3.5,
            "stop_loss_pct": 2.0,
            "take_profit_pct": [4.0],
            "management": {
                "move_to_breakeven_at_tp_pct": 0.5,
                "breakeven_offset_pct": 0.1,
                "trailing_atr_mult": None,
                "max_hold_candles": 120,
                "exit_on_level_failure": False,
                "level_field": "resistance_1",
                "level_failure_buffer_pct": 1.0,
                "exit_on_trend_loss": False,
                "exit_on_macd_flip": False,
            },
        },
        "short": {
            "entry": [
                {"type": "compare", "left": "close", "op": "<", "right": "sma_50"},
                {"type": "compare", "left": "close", "op": "<", "right": "support_1"},
                {"type": "compare", "left": "adx_14", "op": ">", "right": 20},
                {"type": "compare", "left": "macd_hist", "op": "<", "right": 0},
                {"type": "range", "value": "rsi_14", "min": 35, "max": 55},
                {"type": "volume", "op": ">", "ref": "avg_volume_96", "mult": 1.2},
            ],
            "stop_loss_atr_mult": 1.5,
            "take_profit_atr_mult": 3.5,
            "stop_loss_pct": 2.0,
            "take_profit_pct": [4.0],
            "management": {
                "move_to_breakeven_at_tp_pct": 0.5,
                "breakeven_offset_pct": 0.1,
                "trailing_atr_mult": None,
                "max_hold_candles": 120,
                "exit_on_level_failure": False,
                "level_field": "support_1",
                "level_failure_buffer_pct": 1.0,
                "exit_on_trend_loss": False,
                "exit_on_macd_flip": False,
            },
        },
        "filters": [
            {"type": "compare", "left": "adx_14", "op": ">", "right": 20},
        ],
    }
def ensure_engine_defaults(spec: Dict) -> Dict:
    """
    Достраивает служебные блоки движка, которые AI часто не возвращает:
      - ATR-множители стопов (если не заданы ни ATR-мульты, ни проценты);
      - management (breakeven, выходы);
      - risk_controls.

    Не перезаписывает уже заданные моделью значения — только заполняет пропуски.
    Возвращает НОВЫЙ dict.
    """
    if not isinstance(spec, dict):
        return spec

    merged = json.loads(json.dumps(spec, ensure_ascii=False))
    reference = default_strategy_spec()

    for side in ("long", "short"):
        block = merged.get(side)
        if not isinstance(block, dict):
            continue

        ref_side = reference.get(side, {})

        # ATR-стопы: добавляем, только если модель не задала НИ ATR, НИ проценты.
        if "stop_loss_atr_mult" not in block and "stop_loss_pct" not in block:
            block["stop_loss_atr_mult"] = ref_side.get("stop_loss_atr_mult", 1.5)
        if "take_profit_atr_mult" not in block and "take_profit_pct" not in block:
            block["take_profit_atr_mult"] = ref_side.get("take_profit_atr_mult", 4.0)

        if not isinstance(block.get("management"), dict):
            block["management"] = {}

        mgmt = block["management"]
        ref_mgmt = ref_side.get("management", {}) or {}
        for key, default_value in ref_mgmt.items():
            if mgmt.get(key) is None:
                mgmt[key] = default_value

    if not isinstance(merged.get("risk_controls"), dict):
        merged["risk_controls"] = {
            "enabled": True,
            "max_consecutive_losses": 4,
            "pause_candles_after_losses": 32,
            "max_losses_in_window": 6,
            "loss_window_trades": 10,
        }

    return merged

def _describe_operand(operand: Any) -> str:
    if isinstance(operand, dict):
        base = operand.get("name", "?")
        mult = operand.get("mult", 1.0)
        offset = operand.get("offset", 0.0)
        s = base
        if mult != 1.0:
            s += f"*{mult}"
        if offset:
            s += f"+{offset}"
        return s
    return str(operand)


def explain_conditions(conditions: List[Dict], ctx: Dict[str, float], prev_ctx: Optional[Dict[str, float]]) -> List[Dict]:
    """
    Возвращает список пояснений по каждому условию:
    {"text": "<условие>", "passed": bool, "detail": "<фактические значения>"}
    """
    out = []

    for cond in conditions:
        passed = _eval_condition(cond, ctx, prev_ctx)
        ctype = cond.get("type")
        text = ""
        detail = ""

        if ctype == "compare":
            left = _describe_operand(cond.get("left"))
            right = _describe_operand(cond.get("right"))
            text = f"{left} {cond.get('op')} {right}"
            lv = _resolve(cond.get("left"), ctx)
            rv = _resolve(cond.get("right"), ctx)
            detail = f"{left}={_fmt(lv)}, {right}={_fmt(rv)}"

        elif ctype == "range":
            v = cond.get("value")
            text = f"{v} в [{cond.get('min')}, {cond.get('max')}]"
            detail = f"{v}={_fmt(_resolve(v, ctx))}"

        elif ctype == "volume":
            ref = cond.get("ref")
            mult = cond.get("mult", 1.0)
            text = f"volume {cond.get('op')} {ref}*{mult}"
            vol = ctx.get("volume")
            refv = _resolve(ref, ctx)
            thr = refv * float(mult) if refv is not None else None
            detail = f"volume={_fmt(vol)}, порог={_fmt(thr)}"

        elif ctype == "cross":
            text = f"{cond.get('a')} пересекает {cond.get('b')} {cond.get('dir')}"
            detail = (
                f"сейчас {cond.get('a')}={_fmt(_resolve(cond.get('a'), ctx))} "
                f"vs {cond.get('b')}={_fmt(_resolve(cond.get('b'), ctx))}"
            )

        elif ctype == "slope":
            text = f"{cond.get('value')} наклон {cond.get('dir')}"
            detail = f"сейчас={_fmt(_resolve(cond.get('value'), ctx))}"

        elif ctype == "distance":
            a_name = _describe_operand(cond.get("a"))
            b_name = _describe_operand(cond.get("b"))
            text = f"distance({a_name}, {b_name})"
            a = _resolve(cond.get("a"), ctx)
            b = _resolve(cond.get("b"), ctx)
            if a is not None and b not in (None, 0):
                dist_pct = abs(a - b) / abs(b) * 100.0
                detail = f"{a_name}={_fmt(a)}, {b_name}={_fmt(b)}, dist={dist_pct:.4f}%"
            else:
                detail = f"{a_name}={_fmt(a)}, {b_name}={_fmt(b)}"

        else:
            text = f"неизвестный тип: {ctype}"

        out.append({
            "text": text,
            "passed": passed,
            "detail": detail
        })

    return out
  


def _fmt(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:.4f}"
