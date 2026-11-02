"""
Экспорт результата одиночного бэктеста в XLSX.

Файл содержит два листа:
  - «Статистика» — шапка (symbol, источник спеки, даты) + все метрики из stats;
  - «Сделки»     — таблица сделок с колонкой entry_reasons.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _fmt_date(value) -> str:
    """Приводит datetime/строку к YYYY-MM-DD. None → 'unknown'."""
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    if value is None:
        return "unknown"
    return str(value)[:10]


def _fmt_dt(value) -> str:
    """Дата+время для колонок сделок."""
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value) if value is not None else ""


def _format_entry_reasons(reasons) -> str:
    """Склеивает список условий входа в одну строку для ячейки Excel."""
    if not reasons:
        return ""
    parts = []
    for r in reasons:
        if not isinstance(r, dict):
            parts.append(str(r))
            continue
        mark = "✓" if r.get("passed") else "✗"
        text = r.get("text", "")
        detail = r.get("detail", "")
        parts.append(f"{mark} {text} → {detail}")
    return " | ".join(parts)


def _build_stats_rows(result: Dict, symbol: str) -> List[Dict]:
    stats = result.get("stats", {}) or {}
    rows = [
        ("Символ", symbol),
        ("Источник стратегии", result.get("spec_source", "")),
        ("Дата начала", _fmt_date(result.get("start_date"))),
        ("Дата конца", _fmt_date(result.get("end_date"))),
        ("", ""),
        ("Всего сделок", stats.get("total_trades", 0)),
        (
            "Long / Short",
            f'{stats.get("long_trades", 0)} / {stats.get("short_trades", 0)}',
        ),
        ("Win Rate (%)", stats.get("win_rate", 0)),
        ("Доходность (%)", stats.get("return_pct", 0)),
        ("Сумма % сделок", stats.get("total_pnl_pct", 0)),
        ("Итоговый капитал ($)", stats.get("final_capital", 0)),
        ("Макс. просадка (%)", stats.get("max_drawdown_pct", 0)),
        ("Profit Factor", stats.get("profit_factor", 0)),
        ("Средняя сделка (%)", stats.get("avg_trade_pct", 0)),
        ("Expectancy (%)", stats.get("expectancy_pct", 0)),
        ("Средний win (%)", stats.get("avg_win_pct", 0)),
        ("Средний loss (%)", stats.get("avg_loss_pct", 0)),
        ("Risk controls", "ON" if stats.get("risk_controls_enabled") else "OFF"),
        ("Cooldown activations", stats.get("cooldown_activations", 0)),
        (
            "Skipped candles by cooldown",
            stats.get("skipped_candles_by_cooldown", 0),
        ),
    ]
    return [{"Метрика": k, "Значение": v} for k, v in rows]


def _build_trades_rows(trades) -> List[Dict]:
    rows = []
    for t in trades or []:
        rows.append(
            {
                "entry_time": _fmt_dt(t.get("entry_time")),
                "exit_time": _fmt_dt(t.get("exit_time")),
                "direction": t.get("direction", ""),
                "entry_price": t.get("entry_price", 0.0),
                "exit_price": t.get("exit_price", 0.0),
                "result_pct": t.get("result_pct", 0.0),
                "exit_reason": t.get("exit_reason", ""),
                "capital_after": t.get("capital_after", 0.0),
                "entry_reasons": _format_entry_reasons(t.get("entry_reasons")),
            }
        )
    return rows


def export_backtest_to_xlsx(
    result: Dict,
    symbol: str,
    output_dir: Path,
    start_date=None,
    end_date=None,
) -> Optional[Path]:
    """
    Сохраняет результат одиночного бэктеста в XLSX.

    Имя файла: {symbol}_backtest_{start}_{end}.xlsx (даты — YYYY-MM-DD).

    Возвращает путь к файлу или None, если сохранить не удалось.
    Ошибки логируются, но не пробрасываются — падение экспорта не должно
    ломать рендер результата в UI.
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        start_str = _fmt_date(
            start_date if start_date is not None else result.get("start_date")
        )
        end_str = _fmt_date(
            end_date if end_date is not None else result.get("end_date")
        )
        filename = f"{symbol}_backtest_{start_str}_{end_str}.xlsx"
        filepath = output_dir / filename

        stats_df = pd.DataFrame(_build_stats_rows(result, symbol))

        trades_rows = _build_trades_rows(result.get("trades"))
        trades_cols = [
            "entry_time",
            "exit_time",
            "direction",
            "entry_price",
            "exit_price",
            "result_pct",
            "exit_reason",
            "capital_after",
            "entry_reasons",
        ]
        trades_df = (
            pd.DataFrame(trades_rows)
            if trades_rows
            else pd.DataFrame(columns=trades_cols)
        )

        with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
            workbook = writer.book
            header_fmt = workbook.add_format(
                {"bold": True, "bg_color": "#DDDDDD", "border": 1}
            )
            text_fmt = workbook.add_format({"valign": "top"})

            stats_df.to_excel(writer, sheet_name="Статистика", index=False)
            ws_stats = writer.sheets["Статистика"]
            ws_stats.set_column(0, 0, 34, text_fmt)
            ws_stats.set_column(1, 1, 44, text_fmt)

            trades_df.to_excel(writer, sheet_name="Сделки", index=False)
            ws_trades = writer.sheets["Сделки"]

            widths = {
                "entry_time": 20,
                "exit_time": 20,
                "direction": 10,
                "entry_price": 14,
                "exit_price": 14,
                "result_pct": 12,
                "exit_reason": 16,
                "capital_after": 16,
                "entry_reasons": 80,
            }
            for idx, col in enumerate(trades_df.columns):
                ws_trades.set_column(idx, idx, widths.get(col, 16), text_fmt)

            # Жирная шапка на обоих листах
            for ws, df in ((ws_stats, stats_df), (ws_trades, trades_df)):
                for col_idx in range(len(df.columns)):
                    ws.write(0, col_idx, str(df.columns[col_idx]), header_fmt)

        logger.info("[%s] Бэктест сохранён: %s", symbol, filepath)
        return filepath

    except Exception as e:
        logger.warning("[%s] Не удалось сохранить XLSX бэктеста: %s", symbol, e)
        return None