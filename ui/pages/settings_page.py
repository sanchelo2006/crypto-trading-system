"""
System settings page.
"""

from __future__ import annotations

from nicegui import ui


class SettingsPage:
    """
    Page for editing the application's user settings.

    At this stage the logic is kept as close as possible to the original
    nicegui_app.py: the page shows checkboxes and saves the settings
    through the passed app_settings object.
    """

    def __init__(self, app_settings):
        self.app_settings = app_settings
        self.container = ui.column().classes("w-full")

        with self.container:
            self.build_ui()

        self.container.visible = False

    def build_ui(self):
        ui.label("⚙️ System settings").classes("text-h6 q-mb-md")

        with ui.expansion("📦 Data packages", icon="dataset", value=True).classes("w-full"):
            with ui.column().classes("q-gutter-md w-full"):
                ui.label("Package contents").classes("text-bold")

                with ui.column().classes("gap-2"):
                    self.include_1h = ui.checkbox(
                        "Hourly candles (1h)",
                        value=self.app_settings.get("include_1h"),
                    ).style("width: 260px")

                    # Hint explaining why the 1h series are needed
                    ui.label(
                        "The 1h series are used for the higher-timeframe trend filter, "
                        "improving strategy robustness and reducing the number of noisy signals."
                    ).classes("text-caption text-grey q-mt-xs")

                    self.include_15m = ui.checkbox(
                        "15-minute candles",
                        value=self.app_settings.get("include_15m"),
                    ).style("width: 260px")

                    self.include_fund = ui.checkbox(
                        "Fundamentals (OI, Funding, F&G)",
                        value=self.app_settings.get("include_fundamentals"),
                    ).style("width: 260px")

                    self.include_ob = ui.checkbox(
                        "Order book",
                        value=self.app_settings.get("include_orderbook"),
                    ).style("width: 260px")

                    self.include_trades = ui.checkbox(
                        "Trades",
                        value=self.app_settings.get("include_trades"),
                    ).style("width: 260px")

                    self.include_news = ui.checkbox(
                        "News",
                        value=self.app_settings.get("include_news"),
                    ).style("width: 260px")

                    self.allow_weak_holdout_save = ui.checkbox(
                        "Allow saving the strategy on weak_pass holdout",
                        value=self.app_settings.get("allow_weak_holdout_save", False),
                    ).style("width: 360px")

        with ui.expansion("🧱 Robust window-based selection", icon="insights", value=False).classes("w-full"):
            with ui.column().classes("q-gutter-md w-full"):
                self.robust_min_profitable_windows_ratio = ui.number(
                    "Minimum share of profitable windows",
                    value=float(self.app_settings.get("robust_min_profitable_windows_ratio", 0.67)),
                    format="%.2f",
                ).style("width: 320px")

                self.robust_min_positive_expectancy_windows_ratio = ui.number(
                    "Minimum share of windows with positive expectancy",
                    value=float(self.app_settings.get("robust_min_positive_expectancy_windows_ratio", 0.67)),
                    format="%.2f",
                ).style("width: 320px")

                self.robust_min_pf_pass_windows_ratio = ui.number(
                    "Minimum share of windows with PF > 1",
                    value=float(self.app_settings.get("robust_min_pf_pass_windows_ratio", 0.67)),
                    format="%.2f",
                ).style("width: 320px")

                self.robust_min_median_window_return_pct = ui.number(
                    "Minimum median window return (%)",
                    value=float(self.app_settings.get("robust_min_median_window_return_pct", 0.0)),
                    format="%.2f",
                ).style("width: 320px")

                self.robust_max_worst_window_return_pct = ui.number(
                    "Minimum allowed worst window (%)",
                    value=float(self.app_settings.get("robust_max_worst_window_return_pct", -5.0)),
                    format="%.2f",
                ).style("width: 320px")

                self.robust_min_stability_score = ui.number(
                    "Minimum stability_score",
                    value=float(self.app_settings.get("robust_min_stability_score", 0.5)),
                    format="%.2f",
                ).style("width: 320px")

                self.block_fragile_best_candidate = ui.checkbox(
                    "Block a fragile candidate as best",
                    value=bool(self.app_settings.get("block_fragile_best_candidate", True)),
                ).style("width: 360px")

        ui.button(
            "💾 Save settings",
            on_click=self.save_settings,
        ).classes("bg-primary text-white q-mt-lg")

    def save_settings(self):
        self.app_settings.update({
            "include_1h": bool(self.include_1h.value),
            "include_15m": bool(self.include_15m.value),
            "include_fundamentals": bool(self.include_fund.value),
            "include_orderbook": bool(self.include_ob.value),
            "include_trades": bool(self.include_trades.value),
            "include_news": bool(self.include_news.value),
            "allow_weak_holdout_save": bool(self.allow_weak_holdout_save.value),
            "robust_min_profitable_windows_ratio": float(self.robust_min_profitable_windows_ratio.value),
            "robust_min_positive_expectancy_windows_ratio": float(self.robust_min_positive_expectancy_windows_ratio.value),
            "robust_min_pf_pass_windows_ratio": float(self.robust_min_pf_pass_windows_ratio.value),
            "robust_min_median_window_return_pct": float(self.robust_min_median_window_return_pct.value),
            "robust_max_worst_window_return_pct": float(self.robust_max_worst_window_return_pct.value),
            "robust_min_stability_score": float(self.robust_min_stability_score.value),
            "block_fragile_best_candidate": bool(self.block_fragile_best_candidate.value),
        })

        if self.app_settings.save():
            ui.notify("Settings saved!", type="positive")
        else:
            ui.notify("Failed to save settings", type="negative")