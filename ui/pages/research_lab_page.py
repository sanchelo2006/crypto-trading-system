"""
Страница исследовательской лаборатории.

Позволяет:
- задать начальный капитал, комиссии, проскальзывание;
- выбрать монеты, распределить доли капитала, задать плечо и размер позиции;
- выбрать стратегии (входы) и параметры выходов;
- запустить портфельную симуляцию;
- увидеть метрики, разбивку по годам и графики доходности.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List

from nicegui import ui

from data_loader import get_available_symbols
from research.portfolio_simulator import (
    STRATEGY_REGISTRY, SimConfig, SymbolConfig,
    run_simulation, save_simulation_report,
)

logger = logging.getLogger(__name__)

# Палитра для графиков (по монете на цвет)
CHART_COLORS = [
    "#5470c6", "#91cc75", "#fac858", "#ee6666", "#73c0de",
    "#3ba272", "#fc8452", "#9a60b4", "#ea7ccc", "#48b8d0",
]


class ResearchLabPage:
    """Страница портфельной лаборатории."""

    def __init__(self, *, backtest_engine):
        self.backtest_engine = backtest_engine
        self.container = ui.column().classes("w-full")
        self._running = False
        self._symbol_inputs: Dict[str, Dict] = {}

        with self.container:
            self.build_ui()
        self.container.visible = False

    # ----------------------------------------------------------------
    # Построение интерфейса
    # ----------------------------------------------------------------
    def build_ui(self):
        ui.label("🔬 Исследовательская лаборатория").classes("text-h5 q-mb-md")
        ui.label(
            "Портфельная симуляция: выберите монеты, стратегии и параметры — "
            "система построит кривые доходности и покажет статистику."
        ).classes("text-caption text-grey q-mb-md")

        self._build_portfolio_settings()
        self._build_exit_settings()
        self._build_strategy_selector()
        self._build_symbol_selector()
        self._build_run_button()

        self.status_label = ui.label("").classes("text-caption text-grey q-mt-sm")
        self.result_container = ui.column().classes("w-full q-mt-md")

    def _build_portfolio_settings(self):
        with ui.card().classes("w-full q-mb-md"):
            ui.label("💰 Параметры портфеля").classes("text-subtitle1")
            with ui.row().classes("gap-4 items-end flex-wrap"):
                self.initial_capital = ui.number(
                    "Начальный капитал (USDT)", value=1000.0, min=10, format="%.2f",
                ).classes("w-48")
                self.fee_pct = ui.number(
                    "Комиссия, % (round-trip)", value=0.11, min=0.0, step=0.01, format="%.2f",
                ).classes("w-44")
                self.slippage_pct = ui.number(
                    "Проскальзывание, %", value=0.05, min=0.0, step=0.01, format="%.2f",
                ).classes("w-44")
                self.reinvest = ui.checkbox("Реинвестировать прибыль", value=True)

    def _build_exit_settings(self):
        with ui.card().classes("w-full q-mb-md"):
            ui.label("🎯 Параметры выходов (общие для всех монет)").classes("text-subtitle1")
            with ui.row().classes("gap-4 items-end flex-wrap"):
                self.sl_atr = ui.number("Стоп-лосс (×ATR)", value=2.0, min=0.5, step=0.1, format="%.1f").classes("w-40")
                self.tp_atr = ui.number("Тейк-профит (×ATR)", value=1.5, min=0.5, step=0.1, format="%.1f").classes("w-40")
                self.max_hold = ui.number("Макс. удержание (свечей)", value=8, min=1, max=192, format="%d").classes("w-44")
            ui.label(
                "Стоп и тейк задаются в множителях ATR(14) — нормализуют риск по волатильности."
            ).classes("text-caption text-grey")

    def _build_strategy_selector(self):
        with ui.card().classes("w-full q-mb-md"):
            ui.label("📈 Стратегии (входы)").classes("text-subtitle1")
            ui.label(
                "Выбранные стратегии объединяются по ИЛИ на каждой монете."
            ).classes("text-caption text-grey q-mb-sm")
            self.strategy_checkboxes = {}
            with ui.grid(columns=2).classes("w-full gap-2"):
                for name, sdef in STRATEGY_REGISTRY.items():
                    cb = ui.checkbox(f"{name}", value=(name == "m2s2_vol2.0"))
                    cb.tooltip(sdef["desc"])
                    self.strategy_checkboxes[name] = cb

    def _build_symbol_selector(self):
        with ui.card().classes("w-full q-mb-md"):
            ui.label("🪙 Монеты, доли капитала и плечо").classes("text-subtitle1")
            ui.label(
                "Сумма долей не обязана быть 100% — свободная часть остаётся в кэше."
            ).classes("text-caption text-grey q-mb-sm")
            symbols = get_available_symbols()
            default_symbols = {"ARBUSDT", "DOGEUSDT", "SOLUSDT"}
            default_share = round(100.0 / max(1, len(default_symbols)), 1)
            for sym in symbols:
                enabled = sym in default_symbols
                with ui.expansion(f"{sym}", icon="currency_bitcoin").classes("w-full"):
                    with ui.row().classes("gap-4 items-center"):
                        enable_cb = ui.checkbox("Торговать", value=enabled).classes("w-28")
                        share = ui.number("Доля капитала (%)", value=default_share if enabled else 0.0,
                                          min=0, max=100, format="%.1f").classes("w-40")
                        pos_size = ui.number("Размер позиции (%)", value=50.0,
                                             min=1, max=100, format="%.1f").classes("w-40")
                        leverage = ui.number("Плечо", value=1.0, min=1, max=20, format="%.1f").classes("w-28")
                        self._symbol_inputs[sym] = {
                            "enabled": enable_cb,
                            "share": share,
                            "position_size": pos_size,
                            "leverage": leverage,
                        }

    def _build_run_button(self):
        with ui.row().classes("w-full q-mt-sm"):
            self.run_btn = ui.button("🚀 Запустить симуляцию", on_click=self.run_simulation_click)
            self.run_btn.props("color=primary size=lg")

    # ----------------------------------------------------------------
    # Сбор конфигурации из полей
    # ----------------------------------------------------------------
    def _collect_config(self) -> SimConfig:
        symbol_cfgs = []
        for sym, inputs in self._symbol_inputs.items():
            if not inputs["enabled"].value:
                continue
            symbol_cfgs.append(SymbolConfig(
                symbol=sym,
                capital_share_pct=float(inputs["share"].value or 0),
                position_size_pct=float(inputs["position_size"].value or 50),
                leverage=float(inputs["leverage"].value or 1),
            ))
        strategies = [n for n, cb in self.strategy_checkboxes.items() if cb.value]
        return SimConfig(
            initial_capital=float(self.initial_capital.value or 1000),
            fee_pct=float(self.fee_pct.value or 0),
            slippage_pct=float(self.slippage_pct.value or 0),
            sl_atr=float(self.sl_atr.value or 2.0),
            tp_atr=float(self.tp_atr.value or 1.5),
            max_hold=int(self.max_hold.value or 8),
            reinvest=bool(self.reinvest.value),
            symbols=symbol_cfgs,
            strategies=strategies,
        )

    # ----------------------------------------------------------------
    # Запуск симуляции
    # ----------------------------------------------------------------
    def run_simulation_click(self):
        if self._running:
            return
        cfg = self._collect_config()
        total_share = sum(s.capital_share_pct for s in cfg.symbols)
        if total_share > 100.001:
            ui.notify(f"Сумма долей = {total_share:.1f}% — не может превышать 100%.",
                      type="negative")
            return
        if not cfg.symbols:
            ui.notify("Выберите хотя бы одну монету.", type="warning")
            return
        if not cfg.strategies:
            ui.notify("Выберите хотя бы одну стратегию.", type="warning")
            return

        self._running = True
        self.run_btn.enabled = False
        self.status_label.set_text("Симуляция запущена…")
        self.result_container.clear()

        def worker():
            try:
                result = run_simulation(self.backtest_engine, cfg)
                report_path = save_simulation_report(result, cfg)
                return result, cfg, None, report_path
            except Exception as e:
                logger.exception("Ошибка симуляции лаборатории")
                return None, cfg, str(e), None

        async def finalize():
            result, cfg, err, report_path = await asyncio.to_thread(worker)
            self._running = False
            self.run_btn.enabled = True
            if err:
                self.status_label.set_text(f"Ошибка: {err}")
                ui.notify(f"Ошибка симуляции: {err}", type="negative")
                return
            self.status_label.set_text("Симуляция завершена")
            self._render_results(result, cfg, report_path=report_path)

        asyncio.create_task(finalize())

    # ----------------------------------------------------------------
    # Отрисовка результатов
    # ----------------------------------------------------------------
    def _render_results(self, result: Dict, cfg: SimConfig, report_path=None):
        self.result_container.clear()
        symbols = result["symbols"]

        if not symbols:
            with self.result_container:
                ui.label("Нет результатов. " + "; ".join(result.get("errors", []))).classes(
                    "text-negative")
            return

        with self.result_container:
            if report_path is not None:
                ui.label(f"📄 Подробный отчёт сохранён: {report_path}").classes(
                    "text-caption text-positive q-mb-sm")
            ui.label("📊 Итоги портфеля").classes("text-h6 q-mt-md")

        total_initial = cfg.initial_capital
        total_final = sum(s["final_capital"] for s in symbols.values()) \
            + result.get("unallocated_cash", 0.0)
        total_pnl = total_final - total_initial
        total_pnl_pct = (total_final / total_initial - 1) * 100 if total_initial else 0

        with self.result_container:
            # --- Верхние карточки метрик ---
            ui.label("📊 Итоги портфеля").classes("text-h6 q-mt-md")
            with ui.row().classes("gap-4 flex-wrap"):
                self._metric_card("Начальный капитал", f"${total_initial:,.2f}")
                self._metric_card("Конечный капитал", f"${total_final:,.2f}",
                                  color="positive" if total_final >= total_initial else "negative")
                self._metric_card("Общий P&L", f"{total_pnl:+,.2f}$ ({total_pnl_pct:+.2f}%)",
                                  color="positive" if total_pnl >= 0 else "negative")
                self._metric_card("Монет / сделок",
                                  f"{len(symbols)} / {sum(s['trades'] for s in symbols.values())}")
                worst_dd = max((s.get("max_drawdown_pct", 0) for s in symbols.values()), default=0)
                self._metric_card("Макс. просадка (монета)", f"{worst_dd:.1f}%",
                                  color="negative" if worst_dd > 20 else "")
                n_bankrupt = sum(1 for s in symbols.values() if s.get("bankrupt"))
                if n_bankrupt:
                    self._metric_card("БАНКРОТСТВО", f"{n_bankrupt} монет", color="negative")

            # --- Пояснение экспектанси ---
            ui.label(
                "ℹ️ exp_mean — средний результат одной сделки в % от цены входа "
                "(до учёта размера позиции и плеча). Показывает качество сигнала самого по себе."
            ).classes("text-caption text-grey q-mt-sm")

            # --- Таблица по монетам ---
            ui.label("🪙 Результаты по монетам").classes("text-h6 q-mt-lg")
            sym_rows = []
            for sym, s in symbols.items():
                sym_rows.append({
                    "symbol": sym,
                    "trades": s["trades"],
                    "win_rate": f"{s['win_rate']:.1f}%",
                    "exp_mean": f"{s['exp_mean']:+.4f}%",
                    "final": f"${s['final_capital']:,.2f}",
                    "pnl": f"{s['pnl']:+,.2f}$",
                    "pnl_pct": f"{s['pnl_pct']:+.2f}%",
                })
            ui.table(
                columns=[
                    {"name": "symbol", "label": "Монета", "field": "symbol", "align": "left"},
                    {"name": "trades", "label": "Сделок", "field": "trades", "align": "center"},
                    {"name": "win_rate", "label": "Win Rate", "field": "win_rate", "align": "center"},
                    {"name": "exp_mean", "label": "exp_mean", "field": "exp_mean", "align": "center"},
                    {"name": "final", "label": "Конечный капитал", "field": "final", "align": "right"},
                    {"name": "pnl", "label": "P&L", "field": "pnl", "align": "right"},
                    {"name": "pnl_pct", "label": "P&L %", "field": "pnl_pct", "align": "right"},
                ],
                rows=sym_rows, row_key="symbol",
            ).classes("w-full")

            # --- Таблица по годам ---
            ui.label("📅 Результаты по годам").classes("text-h6 q-mt-lg")
            year_rows = self._build_year_table(symbols)
            if year_rows:
                ui.table(
                    columns=[
                        {"name": "year", "label": "Год", "field": "year", "align": "center"},
                        {"name": "trades", "label": "Сделок", "field": "trades", "align": "center"},
                        {"name": "exp_mean", "label": "exp_mean", "field": "exp_mean", "align": "center"},
                        {"name": "pnl_pct", "label": "P&L %", "field": "pnl_pct", "align": "center"},
                    ],
                    rows=year_rows, row_key="year",
                ).classes("w-full")

            # --- График доходности ---
            ui.label("📈 Кривые доходности").classes("text-h6 q-mt-lg")
            self._render_equity_chart(result, cfg)

            # --- Ошибки ---
            if result["errors"]:
                ui.label("⚠️ Предупреждения: " + "; ".join(result["errors"])).classes(
                    "text-caption text-warning q-mt-sm")

    def _metric_card(self, title: str, value: str, color: str = ""):
        color_cls = f"text-{color}" if color else ""
        with ui.card().classes("q-pa-md"):
            ui.label(title).classes("text-caption text-grey")
            ui.label(value).classes(f"text-h6 {color_cls}")

    def _build_year_table(self, symbols: Dict) -> List[Dict]:
        """Агрегирует сделки всех монет по годам выхода."""
        by_year: Dict[int, List[float]] = {}
        for s in symbols.values():
            for t in s["trades_detail"]:
                y = t["exit_time"].year
                by_year.setdefault(y, []).append(t["result_pct"])
        rows = []
        for year in sorted(by_year.keys()):
            exps = by_year[year]
            rows.append({
                "year": year,
                "trades": len(exps),
                "exp_mean": f"{sum(exps)/len(exps):+.4f}%",
                "pnl_pct": f"{sum(exps):+.2f}%",
            })
        return rows

    def _render_equity_chart(self, result: Dict, cfg: SimConfig):
        """График: все монеты разными цветами + портфель, на одной сетке."""
        series = []
        legend = []
        color_idx = 0
        for sym, s in result["symbols"].items():
            color = CHART_COLORS[color_idx % len(CHART_COLORS)]
            color_idx += 1
            data = [[int(t.timestamp() * 1000), round(cap, 2)] for (t, cap) in s["curve"]]
            series.append({
                "name": sym, "type": "line", "showSymbol": False,
                "data": data, "lineStyle": {"width": 2}, "itemStyle": {"color": color},
            })
            legend.append(sym)

        # Портфель — жирная чёрная линия
        port_data = [[int(t.timestamp() * 1000), round(cap, 2)]
                     for (t, cap) in result["portfolio_curve"]]
        series.append({
            "name": "Портфель", "type": "line", "showSymbol": False,
            "data": port_data, "lineStyle": {"width": 3, "color": "#222"},
            "itemStyle": {"color": "#222"},
        })
        legend.append("Портфель")

        ui.echart({
            "tooltip": {"trigger": "axis"},
            "legend": {"data": legend},
            "grid": {"left": "4%", "right": "4%", "bottom": "12%", "containLabel": True},
            "xAxis": {"type": "time"},
            "yAxis": {"type": "value", "name": "Капитал, $", "scale": True},
            "dataZoom": [{"type": "slider", "bottom": 0}],
            "series": series,
        }).classes("w-full").style("height: 420px")