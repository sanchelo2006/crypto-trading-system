"""
Statistics page: monthly P&L per coin + capital at the end of the month.

Data is read from the auto-trading trade logs via utils.pnl_history.
The page is read-only: it writes nothing and does not touch trading.
"""
from __future__ import annotations

import logging

from nicegui import ui

from utils.pnl_history import get_monthly_pnl, get_all_symbols_monthly

logger = logging.getLogger(__name__)

ALL_SYMBOLS_LABEL = "All coins"


class StatisticsPage:
    """Monthly P&L statistics page. Dependency: memory."""

    def __init__(self, memory):
        self.memory = memory
        self.container = ui.column().classes("w-full")

        with self.container:
            self.build_ui()

        self.container.visible = False

    def build_ui(self):
        ui.label("📊 Monthly P&L statistics").classes("text-h6 q-mb-md")

        with ui.row().classes("gap-4 items-center q-mb-md"):
            self.symbol_select = ui.select(
                options=[ALL_SYMBOLS_LABEL],
                value=ALL_SYMBOLS_LABEL,
                label="Coin",
                on_change=lambda _: self.refresh(),
            ).classes("w-56")

            ui.button("🔄 Refresh", on_click=self.refresh).props("flat")

        self.chart = ui.echart({
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["P&L (USDT)", "Capital at month end"]},
            "grid": {
                "left": "70px", "right": "70px",
                "top": "60px", "bottom": "40px",
            },
            "xAxis": {"type": "category", "data": []},
            "yAxis": [
                {"type": "value", "name": "P&L (USDT)"},
                {"type": "value", "name": "Capital", "position": "right"},
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
                    "name": "Capital at month end",
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
                {"name": "month", "label": "Month", "field": "month", "align": "left"},
                {"name": "pnl", "label": "P&L (USDT)", "field": "pnl", "align": "right"},
                {"name": "trades", "label": "Trades", "field": "trades", "align": "right"},
                {"name": "capital", "label": "Capital at month end*",
                 "field": "capital", "align": "right"},
            ],
            rows=[],
            row_key="month",
        ).classes("w-full")

        ui.label(
            "* Capital is reconstructed from the current value (risk_state.json) "
            "by subtracting P&L backwards. If there were deposits or withdrawals, "
            "the value is approximate."
        ).classes("text-caption text-grey q-mt-sm")

    def _reload_options(self):
        symbols = list(self.memory.get_all_symbols())
        options = [ALL_SYMBOLS_LABEL] + symbols
        self.symbol_select.options = options
        if self.symbol_select.value not in options:
            self.symbol_select.value = ALL_SYMBOLS_LABEL
        self.symbol_select.update()

    def refresh(self):
        """Re-reads the trade logs and refreshes the chart + table."""
        try:
            self._reload_options()
        except Exception:
            logger.exception("StatisticsPage: failed to refresh the coin list")

        selected = self.symbol_select.value or ALL_SYMBOLS_LABEL

        try:
            if selected == ALL_SYMBOLS_LABEL:
                rows = get_all_symbols_monthly(list(self.memory.get_all_symbols()))
            else:
                rows = get_monthly_pnl(selected)
        except Exception:
            logger.exception("StatisticsPage: failed to read P&L history")
            rows = []

        months = [r["month"] for r in rows]
        pnl_values = [r["pnl_usdt"] for r in rows]
        cap_values = [r["capital_end"] for r in rows]  # None → gap in the line

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