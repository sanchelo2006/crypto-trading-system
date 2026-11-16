"""
General trade journal page.
"""

from __future__ import annotations

from nicegui import ui


# Marker meaning "no errors" in the journal.
# Older records were written in Russian, so both variants are recognized.
_NO_ERRORS_MARKERS = ("No errors", "Не было ошибок")


class JournalPage:
    """
    Trade journal page across all coins.

    Receives dependencies explicitly:
    - memory;
    - the get_all_trades(memory) function;
    - the clear_all_journals(memory) function.
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
        ui.label("📋 Trade journal (all coins)").classes("text-h6")

        with ui.row().classes("w-full justify-end"):
            ui.button(
                "🗑 Clear entire journal",
                on_click=self.clear_journal,
                color="negative",
            )

        self.journal_table = ui.table(
            columns=[
                {"name": "pair", "label": "Coin", "field": "pair", "align": "left"},
                {"name": "datetime", "label": "Date/Time", "field": "datetime", "align": "left"},
                {"name": "direction", "label": "Dir.", "field": "direction", "align": "center"},
                {"name": "entry_price", "label": "Entry", "field": "entry_price", "align": "right"},
                {"name": "exit_price", "label": "Exit", "field": "exit_price", "align": "right"},
                {"name": "result_pct", "label": "Result", "field": "result_pct", "align": "center"},
                {"name": "reason", "label": "Reason", "field": "reason", "align": "left"},
                {"name": "errors", "label": "Errors", "field": "errors", "align": "left"},
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
                            if row.get("errors") and row.get("errors") not in _NO_ERRORS_MARKERS
                            else ""
                        ),
                    }
                )

        self.journal_table.rows = rows
        self.journal_table.update()

    def clear_journal(self):
        with ui.dialog() as dialog, ui.card():
            ui.label("Clear the entire trade journal?")
            ui.label("This action cannot be undone!").classes("text-negative")

            with ui.row().classes("justify-end"):
                ui.button("Cancel", on_click=dialog.close)
                ui.button(
                    "Clear",
                    on_click=lambda: self.do_clear(dialog),
                    color="negative",
                )

            dialog.open()

    def do_clear(self, dialog):
        dialog.close()
        self.clear_all_journals_fn(self.memory)
        self.refresh()
        ui.notify("Journal cleared.", type="warning")