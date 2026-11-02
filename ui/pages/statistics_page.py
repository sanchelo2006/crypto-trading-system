"""
Страница статистики: месячный P&L по монетам + капитал на конец месяца.

Данные читаются из trade-логов автоторговли через utils.pnl_history.
Страница read-only: ничего не пишет, торговлю не трогает.
"""
from __future__ import annotations

import logging

from nicegui import ui

from utils.pnl_history import get_monthly_pnl, get_all_symbols_monthly

logger = logging.getLogger(__name__)

ALL_SYMBOLS_LABEL = "Все монеты"


class StatisticsPage:
    """Страница статистики месячного P&L. Зависимость: memory."""

    def __init__(self, memory):
        self.memory = memory
        self.container = ui.column().classes("w-full")

        with self.container:
            self.build_ui()

        self.container.visible = False

    def build_ui(self):
        ui.label("📊 Статистика P&L по месяцам").classes("text-h6 q-mb-md")

        with ui.row().classes("gap-4 items-center q-mb-md"):
            self.symbol_select = ui.select(
                options=[ALL_SYMBOLS_LABEL],
                value=ALL_SYMBOLS_LABEL,
                label="Монета",
                on_change=lambda _: self.refresh(),
            ).classes("w-56")

            ui.button("🔄 Обновить", on_click=self.refresh).props("flat")

        self.chart = ui.echart({
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["P&L (USDT)", "Капитал на конец месяца"]},
            "grid": {
                "left": "70px", "right": "70px",
                "top": "60px", "bottom": "40px",
            },
            "xAxis": {"type": "category", "data": []},
            "yAxis": [
                {"type": "value", "name": "P&L (USDT)"},
                {"type": "value", "name": "Капитал", "position": "right"},
            ],
            "series": [
                {
                    "name": "P&L (USDT)",
                    "type": "bar",
                    "yAxisIndex": 0,
                    "data": [],
                    "itemStyle": {"color": "#4d6bfe"},
                },
                {
                    "name": "Капитал на конец месяца",
                    "type": "line",
                    "yAxisIndex": 1,
                    "data": [],
                    "smooth": True,
                    "symbol": "circle",
                    "lineStyle": {"color": "#28a745"},
                    "itemStyle": {"color": "#28a745"},
                },
            ],
        }).classes("w-full").style("height: 400px;")

        self.table = ui.table(
            columns=[
                {"name": "month", "label": "Месяц", "field": "month", "align": "left"},
                {"name": "pnl", "label": "P&L (USDT)", "field": "pnl", "align": "right"},
                {"name": "trades", "label": "Сделок", "field": "trades", "align": "right"},
                {"name": "capital", "label": "Капитал на конец месяца*",
                 "field": "capital", "align": "right"},
            ],
            rows=[],
            row_key="month",
        ).classes("w-full")

        ui.label(
            "* Капитал восстановлен от текущего значения (risk_state.json) "
            "обратным вычетом P&L. Если были пополнения или выводы — "
            "значение приблизительное."
        ).classes("text-caption text-grey q-mt-sm")

    def _reload_options(self):
        symbols = list(self.memory.get_all_symbols())
        options = [ALL_SYMBOLS_LABEL] + symbols
        self.symbol_select.options = options
        if self.symbol_select.value not in options:
            self.symbol_select.value = ALL_SYMBOLS_LABEL
        self.symbol_select.update()

    def refresh(self):
        """Перечитывает trade-логи и обновляет график + таблицу."""
        try:
            self._reload_options()
        except Exception:
            logger.exception("StatisticsPage: ошибка обновления списка монет")

        selected = self.symbol_select.value or ALL_SYMBOLS_LABEL

        try:
            if selected == ALL_SYMBOLS_LABEL:
                rows = get_all_symbols_monthly(list(self.memory.get_all_symbols()))
            else:
                rows = get_monthly_pnl(selected)
        except Exception:
            logger.exception("StatisticsPage: ошибка чтения истории P&L")
            rows = []

        months = [r["month"] for r in rows]
        pnl_values = [r["pnl_usdt"] for r in rows]
        cap_values = [r["capital_end"] for r in rows]  # None → разрыв на линии

        self.chart.options["xAxis"]["data"] = months
        self.chart.options["series"][0]["data"] = pnl_values
        self.chart.options["series"][1]["data"] = cap_values
        self.chart.update()

        table_rows = []
        for r in rows:
            cap = r["capital_end"]
            table_rows.append({
                "month": r["month"],
                "pnl": f"{r['pnl_usdt']:+.2f}",
                "trades": r["trades"],
                "capital": f"{cap:.2f}" if cap is not None else "—",
            })
        self.table.rows = table_rows
        self.table.update()