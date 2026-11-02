"""
Страница общего журнала сделок.
"""

from __future__ import annotations

from nicegui import ui


class JournalPage:
    """
    Страница журнала сделок по всем монетам.

    Получает зависимости явно:
    - memory;
    - функцию get_all_trades(memory);
    - функцию clear_all_journals(memory).
    """

    def __init__(self, memory, get_all_trades_fn, clear_all_journals_fn):
        self.memory = memory
        self.get_all_trades_fn = get_all_trades_fn
        self.clear_all_journals_fn = clear_all_journals_fn

        self.container = ui.column().classes("w-full")

        with self.container:
            self.build_ui()

        self.container.visible = False

    def build_ui(self):
        ui.label("📋 Журнал сделок (все монеты)").classes("text-h6")

        with ui.row().classes("w-full justify-end"):
            ui.button(
                "🗑 Очистить весь журнал",
                on_click=self.clear_journal,
                color="negative",
            )

        self.journal_table = ui.table(
            columns=[
                {"name": "pair", "label": "Монета", "field": "pair", "align": "left"},
                {"name": "datetime", "label": "Дата/Время", "field": "datetime", "align": "left"},
                {"name": "direction", "label": "Напр.", "field": "direction", "align": "center"},
                {"name": "entry_price", "label": "Вход", "field": "entry_price", "align": "right"},
                {"name": "exit_price", "label": "Выход", "field": "exit_price", "align": "right"},
                {"name": "result_pct", "label": "Результат", "field": "result_pct", "align": "center"},
                {"name": "reason", "label": "Причина", "field": "reason", "align": "left"},
                {"name": "errors", "label": "Ошибки", "field": "errors", "align": "left"},
            ],
            rows=[],
            row_key="index",
        ).classes("w-full")

        self.refresh()

    def refresh(self):
        if not self.memory:
            return

        df = self.get_all_trades_fn(self.memory)
        rows = []

        if not df.empty:
            df_sorted = df.sort_values(["date", "time_utc"], ascending=False)

            for _, row in df_sorted.iterrows():
                rows.append(
                    {
                        "pair": row.get("pair", ""),
                        "datetime": f"{row.get('date', '')} {row.get('time_utc', '')}",
                        "direction": row.get("direction", ""),
                        "entry_price": f"${row.get('entry_price', 0):.4f}",
                        "exit_price": f"${row.get('exit_price', 0):.4f}",
                        "result_pct": f"{row.get('result_pct', 0):+.2f}%",
                        "reason": row.get("reason", ""),
                        "errors": (
                            row.get("errors", "")
                            if row.get("errors") and row.get("errors") != "Не было ошибок"
                            else ""
                        ),
                    }
                )

        self.journal_table.rows = rows
        self.journal_table.update()

    def clear_journal(self):
        with ui.dialog() as dialog, ui.card():
            ui.label("Очистить весь журнал сделок?")
            ui.label("Это действие необратимо!").classes("text-negative")

            with ui.row().classes("justify-end"):
                ui.button("Отмена", on_click=dialog.close)
                ui.button(
                    "Очистить",
                    on_click=lambda: self.do_clear(dialog),
                    color="negative",
                )

            dialog.open()

    def do_clear(self, dialog):
        dialog.close()
        self.clear_all_journals_fn(self.memory)
        self.refresh()
        ui.notify("Журнал очищен.", type="warning")