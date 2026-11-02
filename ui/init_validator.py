"""
Санити-бэктест стратегии при инициализации монеты.

Замыкает петлю обратной связи: модель генерирует strategy_spec,
система прогоняет её на истории и возвращает модели числовую диагностику.
Модель пересобирает стратегию сама — ручные правки state.json не нужны.
"""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Callable, Dict, Optional

from strategy_spec import validate_strategy_spec, ensure_engine_defaults

logger = logging.getLogger(__name__)

SANITY_DAYS = 60
MIN_TRADES_PER_WEEK = 1.5
MAX_QUICK_LOSS_SHARE = 0.5
QUICK_LOSS_CANDLES = 3


def build_diagnostics(stats: Dict, trades, df) -> Dict:
    """Числовая диагностика санити-бэктеста + текст для модели."""
    total = int(stats.get("total_trades", 0) or 0)
    if "timestamp" in df.columns:
        days = max(
            1.0,
            (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 86400,
        )
    else:
        days = float(SANITY_DAYS)

    tpw = total / max(1.0, days / 7.0)
    wr = float(stats.get("win_rate", 0.0) or 0.0)
    pf = float(stats.get("profit_factor", 0.0) or 0.0)
    exp = float(stats.get("expectancy_pct", 0.0) or 0.0)
    avg_w = float(stats.get("avg_win_pct", 0.0) or 0.0)
    avg_l = float(stats.get("avg_loss_pct", 0.0) or 0.0)
    payoff = (avg_w / avg_l) if avg_l > 0 else 0.0
    be_wr = 100.0 / (1.0 + payoff) if payoff > 0 else 100.0

    exits: Dict[str, int] = {}
    quick = 0
    losses = 0
    for t in trades:
        key = str(t.get("exit_reason", "?"))
        exits[key] = exits.get(key, 0) + 1
        if float(t.get("result_pct", 0.0)) <= 0:
            losses += 1
            try:
                held = (t["exit_time"] - t["entry_time"]) / timedelta(minutes=15)
            except Exception:
                held = 999
            if held <= QUICK_LOSS_CANDLES:
                quick += 1
    quick_share = (quick / losses) if losses else 0.0

    close = df["close"]
    bh = (float(close.iloc[-1]) / float(close.iloc[0]) - 1.0) * 100.0

    defects = []
    if tpw < MIN_TRADES_PER_WEEK:
        defects.append(
            f"частота {tpw:.1f} сделок/неделю при норме >= {MIN_TRADES_PER_WEEK} "
            f"(фильтры входа слишком жёсткие)"
        )
    if pf < 1.0 and exp < 0:
        defects.append(f"нет эджа: PF={pf:.2f}, expectancy={exp:+.3f}%")
    if quick_share > MAX_QUICK_LOSS_SHARE:
        defects.append(
            f"{quick_share:.0%} убытков закрываются за <={QUICK_LOSS_CANDLES} свечи: "
            f"входы слишком поздние или стоп внутри шума"
        )

    lines = [
        f"=== САНИТИ-БЭКТЕСТ СТРАТЕГИИ (последние {int(days)} дней) ===",
        f"Сделок: {total} ({tpw:.1f}/неделю). WR={wr:.1f}%, payoff={payoff:.2f} "
        f"(безубыток при WR >= {be_wr:.1f}%).",
        f"PF={pf:.2f}, expectancy={exp:+.3f}%, доходность={stats.get('return_pct', 0):+.2f}%, "
        f"DD={stats.get('max_drawdown_pct', 0):.1f}%",
        "Выходы: " + ", ".join(f"{k}={v}" for k, v in sorted(exits.items())),
        f"Быстрые убытки (<= {QUICK_LOSS_CANDLES} свечи): {quick_share:.0%} от всех убытков",
        f"Buy&Hold за тот же период: {bh:+.1f}%",
    ]
    if defects:
        lines.append("ВЫЯВЛЕННЫЕ ДЕФЕКТЫ:")
        lines += [f"- {d}" for d in defects]
        lines += [
            "ТРЕБОВАНИЯ К ПЕРЕРАБОТКЕ:",
            "- Частота: ослабь объёмный/RSI-фильтры и/или добавь вторую OR-группу входа "
            "(ретест/откат); цель 2-5 сделок в неделю.",
            "- Тайминг и стопы: при доле быстрых убытков > 50% входи раньше (ретест, а не догон) "
            "и/или расширь стоп до 2.5-4.0 x ATR15m, чтобы он был вне шума.",
            "- Payoff: если WR ниже безубыточного порока, подними take_profit_atr_mult до 3-5 "
            "или убери входы с малым потенциалом движения.",
            "- Меняй только диагностированные дефекты, не ломай остальные блоки.",
        ]
    else:
        lines.append("Дефектов не выявлено, стратегия принята как база для оптимизации.")

    return {
        "ok": not defects,
        "defects": defects,
        "pf": pf,
        "trades_per_week": tpw,
        "text": "\n".join(lines),
    }


def run_init_with_sanity(
    *,
    ai_client,
    engine,
    symbol: str,
    category: str,
    init_call: Callable[[], Optional[Dict]],
    max_revisions: int = 2,
    fetcher=None,
) -> Optional[Dict]:
    """
    Обёртка над инициализацией: первый ответ модели + до max_revisions
    переработок по результатам санити-бэктеста. Возвращает итоговый response
    в том же формате, что и initialize_coin — сохранение в память не меняется.
    """
    response = init_call()
    if not response:
        return None

    # Пробуем 3 варианта получения fetcher
    if fetcher is None:
        fetcher = getattr(engine, "fetcher", None)
    if fetcher is None:
        try:
            from ui import services as _services
            fetcher = getattr(_services, "data_fetcher", None)
        except Exception:
            pass
    if fetcher is None:
        logger.warning("[%s] init_validator: fetcher не найден, санити-бэктест пропущен", symbol)
        return response

    # Если глобальный движок не инициализирован — создаём локальный экземпляр
    if engine is None:
        try:
            from backtest_engine import BacktestEngine
            engine = BacktestEngine(fetcher=fetcher)
            logger.info("[%s] init_validator: движок создан локально", symbol)
        except Exception as exc:
            logger.warning("[%s] init_validator: не удалось создать движок (%s), санити-бэктест пропущен", symbol, exc)
            return response

    from datetime import datetime, timezone

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=SANITY_DAYS)
    try:
        df = fetcher.fetch_ohlcv(symbol, "15", start, end, category)
    except Exception as exc:
        logger.warning("[%s] init_validator: ошибка загрузки истории (%s), санити-бэктест пропущен", symbol, exc)
        return response

    if df is None or df.empty or len(df) < 500:
        logger.warning("[%s] init_validator: мало истории для санити-бэктеста (%d строк)", symbol, 0 if df is None else len(df))
        return response

    best_spec: Optional[Dict] = None
    best_payload: Optional[Dict] = None
    best_pf = -1e9
    feedback: Optional[str] = None
    last_spec: Optional[Dict] = None
    last_rules = ""

    for attempt in range(max_revisions + 1):
        if feedback is not None:
            revised = ai_client.revise_init_strategy(
                symbol=symbol,
                current_spec=last_spec,
                current_rules=last_rules,
                diagnostics_text=feedback,
                research=response.get("research"),
            )
            logger.info("[%s] revise_init_strategy вернул: %s", symbol, "OK" if revised else "None")
            if revised and revised.get("strategy_spec"):
                for key in (
                    "strategy_spec",
                    "new_rules",
                    "trading_rules_list",
                    "risk_management_rules_list",
                ):
                    if key in revised:
                        response[key] = revised[key]
                logger.info("[%s] INIT: модель пересобрала спеку (раунд %d)", symbol, attempt)
            else:
                logger.warning("[%s] INIT: переработка не дала спеку, раунд %d", symbol, attempt)

        spec = response.get("strategy_spec")
        err = validate_strategy_spec(spec) if spec else "strategy_spec отсутствует"
        last_rules = response.get("new_rules", "") or ""
        if err is not None:
            logger.warning("[%s] INIT: спека невалидна (%s)", symbol, err)
            feedback = f"strategy_spec невалиден: {err}. Верни исправленный строгий JSON."
            last_spec = spec
            continue

        spec = ensure_engine_defaults(spec)
        res = engine.run_on_dataframe(df=df, symbol=symbol, strategy_spec=spec)
        if "error" in res:
            logger.warning("[%s] INIT: санити-бэктест ошибка: %s", symbol, res["error"])
            break

        diag = build_diagnostics(res["stats"], res["trades"], df)
        for line in diag["text"].splitlines():
            logger.info("[%s] %s", symbol, line)

        if diag["pf"] > best_pf:
            best_pf = diag["pf"]
            best_spec = spec
            best_payload = {
                k: response[k]
                for k in (
                    "strategy_spec",
                    "new_rules",
                    "trading_rules_list",
                    "risk_management_rules_list",
                )
                if k in response
            }
            if best_payload.get("strategy_spec") is not spec:
                best_payload["strategy_spec"] = spec

        if diag["ok"]:
            break

        feedback = diag["text"]
        last_spec = spec

    if best_payload is not None:
        response.update(best_payload)
    response["sanity_best_pf"] = round(best_pf, 3) if best_pf > -1e9 else None
    return response