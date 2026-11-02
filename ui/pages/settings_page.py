"""
Страница системных настроек.
"""

from __future__ import annotations

from nicegui import ui


class SettingsPage:
    """
    Страница редактирования пользовательских настроек приложения.

    На этом этапе логика оставлена максимально близкой к исходному
    nicegui_app.py: страница показывает чекбоксы и сохраняет настройки
    через переданный объект app_settings.
    """

    def __init__(self, app_settings):
        self.app_settings = app_settings
        self.container = ui.column().classes("w-full")

        with self.container:
            self.build_ui()

        self.container.visible = False

    def build_ui(self):
        ui.label("⚙️ Настройки системы").classes("text-h6 q-mb-md")

        with ui.expansion("📦 Данные пакетов", icon="dataset", value=True).classes("w-full"):
            with ui.column().classes("q-gutter-md w-full"):
                ui.label("Состав пакетов").classes("text-bold")

                with ui.column().classes("gap-2"):
                    self.include_1h = ui.checkbox(
                        "Часовые свечи (1h)",
                        value=self.app_settings.get("include_1h"),
                    ).style("width: 260px")

                    # Подсказка о том, зачем нужны 1h-серии
                    ui.label(
                        "Серии 1h используются для трендового фильтра старшего таймфрейма, "
                        "повышения устойчивости стратегии и снижения количества шумовых сигналов."
                    ).classes("text-caption text-grey q-mt-xs")

                    self.include_15m = ui.checkbox(
                        "15-мин свечи",
                        value=self.app_settings.get("include_15m"),
                    ).style("width: 260px")

                    self.include_fund = ui.checkbox(
                        "Фундаментал (OI, Funding, F&G)",
                        value=self.app_settings.get("include_fundamentals"),
                    ).style("width: 260px")

                    self.include_ob = ui.checkbox(
                        "Стакан",
                        value=self.app_settings.get("include_orderbook"),
                    ).style("width: 260px")

                    self.include_trades = ui.checkbox(
                        "Трейды",
                        value=self.app_settings.get("include_trades"),
                    ).style("width: 260px")

                    self.include_news = ui.checkbox(
                        "Новости",
                        value=self.app_settings.get("include_news"),
                    ).style("width: 260px")

                    self.allow_weak_holdout_save = ui.checkbox(
                        "Разрешать сохранение стратегии при weak_pass holdout",
                        value=self.app_settings.get("allow_weak_holdout_save", False),
                    ).style("width: 360px")

        with ui.expansion("🧱 Robust-отбор по окнам", icon="insights", value=False).classes("w-full"):
            with ui.column().classes("q-gutter-md w-full"):
                self.robust_min_profitable_windows_ratio = ui.number(
                    "Минимальная доля прибыльных окон",
                    value=float(self.app_settings.get("robust_min_profitable_windows_ratio", 0.67)),
                    format="%.2f",
                ).style("width: 320px")

                self.robust_min_positive_expectancy_windows_ratio = ui.number(
                    "Минимальная доля окон с положительным expectancy",
                    value=float(self.app_settings.get("robust_min_positive_expectancy_windows_ratio", 0.67)),
                    format="%.2f",
                ).style("width: 320px")

                self.robust_min_pf_pass_windows_ratio = ui.number(
                    "Минимальная доля окон с PF > 1",
                    value=float(self.app_settings.get("robust_min_pf_pass_windows_ratio", 0.67)),
                    format="%.2f",
                ).style("width: 320px")

                self.robust_min_median_window_return_pct = ui.number(
                    "Минимальная медианная доходность по окнам (%)",
                    value=float(self.app_settings.get("robust_min_median_window_return_pct", 0.0)),
                    format="%.2f",
                ).style("width: 320px")

                self.robust_max_worst_window_return_pct = ui.number(
                    "Минимально допустимое худшее окно (%)",
                    value=float(self.app_settings.get("robust_max_worst_window_return_pct", -5.0)),
                    format="%.2f",
                ).style("width: 320px")

                self.robust_min_stability_score = ui.number(
                    "Минимальный stability_score",
                    value=float(self.app_settings.get("robust_min_stability_score", 0.5)),
                    format="%.2f",
                ).style("width: 320px")

                self.block_fragile_best_candidate = ui.checkbox(
                    "Блокировать fragile-кандидата как best",
                    value=bool(self.app_settings.get("block_fragile_best_candidate", True)),
                ).style("width: 360px")

        ui.button(
            "💾 Сохранить настройки",
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
            ui.notify("Настройки сохранены!", type="positive")
        else:
            ui.notify("Ошибка сохранения настроек", type="negative")