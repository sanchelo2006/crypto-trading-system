"""
Страница общей сводки по монетам.
"""

from __future__ import annotations

from nicegui import ui


class SummaryPage:
    """
    Сводная страница по всем монетам.

    Зависимости передаются явно:
    - memory;
    - функция получения состояния авторежима.
    """

    def __init__(self, memory, auto_mode_get_state):
        self.memory = memory
        self.auto_mode_get_state = auto_mode_get_state

        self.container = ui.column().classes("w-full")

        with self.container:
            self.build_ui()

        self.container.visible = True

    def build_ui(self):
        ui.label("📊 Общая сводка по монетам").classes("text-h6")

        self.summary_table = ui.table(
            columns=[
                {"name": "symbol", "label": "Монета", "field": "symbol", "align": "left"},
                {"name": "status", "label": "Статус", "field": "status", "align": "left"},
                {"name": "setup", "label": "Сетап", "field": "setup", "align": "left"},
                {"name": "signal", "label": "Сигнал", "field": "signal", "align": "left"},
                {"name": "position", "label": "Позиция", "field": "position", "align": "left"},
                {"name": "last_analysis", "label": "Последний анализ", "field": "last_analysis", "align": "left"},
            ],
            rows=[],
            row_key="symbol",
        ).classes("w-full")

        self.refresh()

    def refresh(self):
        if not self.memory:
            return

        rows = []

        for sym in self.memory.get_all_symbols():
            cm = self.memory.get(sym)
            state = cm.get_state()
            analysis = cm.get_last_analysis()

            last_analysis_time = analysis.get("timestamp_utc", "") if analysis else ""

            status = "⚪"
            auto_state = self.auto_mode_get_state(sym)

            if auto_state and auto_state.get("is_running"):
                status = "🟢 Авто"
            elif cm.is_day_closed():
                status = "🔒 Закрыт"
            elif cm.is_initialized():
                status = "✅ Готов"
            else:
                status = "⚠️ Не иниц."

            setup_info = ""
            setup = state.get("last_setup")
            if setup:
                direction = setup.get("direction", "").upper()
                confidence = setup.get("confidence_pct", 0)
                setup_info = f"{direction} ({confidence}%)"

            signal_info = ""
            signal = state.get("last_signal")
            if signal and signal.get("confidence_pct", 0) > 0:
                direction = signal.get("direction", "").upper()
                confidence = signal.get("confidence_pct", 0)
                signal_info = f"🚨 {direction} ({confidence}%)"

            position_info = ""
            active = state.get("active_trade")
            if active:
                direction = active.get("direction", "").upper()
                entry = active.get("entry_price", 0)
                position_info = f"{direction} @ {entry:.4f}"

            rows.append(
                {
                    "symbol": sym,
                    "status": status,
                    "setup": setup_info,
                    "signal": signal_info,
                    "position": position_info,
                    "last_analysis": last_analysis_time,
                }
            )

        self.summary_table.rows = rows
        self.summary_table.update()