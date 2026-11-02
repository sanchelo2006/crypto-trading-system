"""
Логирование сделок, событий и результатов оптимизации в CSV-файлы.

Структура файлов (ротация — раз в неделю, ISO-недели):
    data/auto_trading/logs/{SYMBOL}_trades_YYYY_Www.csv          — сделки
    data/auto_trading/logs/{SYMBOL}_events_YYYY_Www.csv          — события
    data/auto_trading/logs/{SYMBOL}_events_YYYY_Www.html         — события (HTML)
    data/auto_trading/logs/{SYMBOL}_sl_updates_YYYY_Www.csv      — изменения SL
    data/auto_trading/logs/{SYMBOL}_optimization_YYYY_Www.csv    — оптимизация

Www — ISO-неделя, две цифры (01..53). Пример: 2026_W39.

Новый файл создаётся автоматически при наступлении новой недели.
HTML-лог содержит панель поиска по дате (client-side JS).
"""

import csv
import logging
import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import config

logger = logging.getLogger(__name__)

LOGS_DIR = config.DATA_DIR / "auto_trading" / "logs"

# ============================================================================
# HTML-стили для {SYMBOL}_events_YYYY_MM.html.
#
# CSV не поддерживает цвета, поэтому для человеческого чтения ведём
# параллельный HTML-лог. Открывается в браузере, раскрашивает события
# по типу: прибыльные сделки — зелёным, убытки — красным, сигналы —
# фиолетовым, предупреждения — оранжевым.
# ============================================================================

_HTML_HEAD_TEMPLATE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>__SYMBOL__ — события __PERIOD__</title>
<style>
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                 Roboto, "Helvetica Neue", Arial, sans-serif;
    background: #f5f5f7;
    color: #1d1d1f;
    margin: 0;
    padding: 24px;
    line-height: 1.5;
  }
  h1 {
    font-size: 20px;
    font-weight: 600;
    margin: 0 0 20px;
    padding-bottom: 12px;
    border-bottom: 1px solid #e0e0e0;
  }
  .search-bar {
    position: sticky;
    top: 0;
    z-index: 10;
    background: #fff;
    padding: 12px 16px;
    border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06);
    margin-bottom: 20px;
    display: flex;
    gap: 10px;
    align-items: center;
    flex-wrap: wrap;
  }
  .search-bar label {
    font-size: 13px;
    font-weight: 600;
    color: #4d6bfe;
  }
  .search-bar input[type="date"],
  .search-bar select {
    padding: 6px 10px;
    border: 1px solid #d0d0d0;
    border-radius: 6px;
    font-family: inherit;
    font-size: 14px;
    color: #1d1d1f;
    background: #fff;
  }
  .search-bar select { min-width: 160px; }
  .search-bar button {
    padding: 6px 14px;
    border: 1px solid #4d6bfe;
    background: #4d6bfe;
    color: #fff;
    border-radius: 6px;
    font-size: 14px;
    cursor: pointer;
    font-family: inherit;
    font-weight: 500;
  }
  .search-bar button.secondary {
    background: #fff;
    color: #4d6bfe;
  }
  .search-bar button:hover { opacity: 0.9; }
  #filter-status {
    font-size: 13px;
    color: #666;
    margin-left: auto;
    font-family: "SF Mono", Consolas, monospace;
  }
  .time-separator {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 28px 0 10px;
    font-size: 13px;
    font-weight: 600;
    color: #4d6bfe;
    font-family: "SF Mono", Consolas, "Courier New", monospace;
  }
  .time-separator::before {
    content: "";
    flex: 0 0 auto;
    width: 8px;
    height: 8px;
    background: #4d6bfe;
    border-radius: 50%;
  }
  .time-separator::after {
    content: "";
    flex: 1;
    height: 1px;
    background: #e0e0e0;
  }
  .event {
    background: #fff;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    border-left: 4px solid #d0d0d0;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
  }
  .type {
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin: 0 0 4px;
  }
  .msg {
    margin-top: 2px;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .event pre {
    background: #fafafa;
    border: 1px solid #eee;
    border-radius: 4px;
    padding: 10px 12px;
    margin: 10px 0 0;
    font-family: "SF Mono", Consolas, "Courier New", monospace;
    font-size: 13px;
    line-height: 1.55;
    overflow-x: auto;
    white-space: pre;
    color: #333;
  }
  .ev-info       { border-left-color: #4d6bfe; }
  .ev-info .type { color: #4d6bfe; }
  .ev-muted       { border-left-color: #adb5bd; }
  .ev-muted .type { color: #6c757d; }
  .ev-success       { border-left-color: #28a745; background: #f3fbf5; }
  .ev-success .type { color: #28a745; }
  .ev-danger       { border-left-color: #dc3545; background: #fff5f5; }
  .ev-danger .type { color: #dc3545; }
  .ev-warning       { border-left-color: #f0a000; background: #fffaf0; }
  .ev-warning .type { color: #d18000; }
  .ev-signal       { border-left-color: #6f42c1; background: #faf8ff; }
  .ev-signal .type { color: #6f42c1; }
  .ev-management       { border-left-color: #17a2b8; }
  .ev-management .type { color: #17a2b8; }
  .ev-default       { border-left-color: #d0d0d0; }
</style>
<script>
function collectEventTypes() {
  var select = document.getElementById('event-filter');
  var seen = {};
  var types = [];
  document.querySelectorAll('.event .type').forEach(function(el) {
    var t = (el.textContent || '').trim();
    if (t && !seen[t]) { seen[t] = true; types.push(t); }
  });
  types.sort();
  types.forEach(function(t) {
    var opt = document.createElement('option');
    opt.value = t;
    opt.textContent = t;
    select.appendChild(opt);
  });
}

function applyFilters() {
  var dateVal = document.getElementById('date-filter').value;
  var eventVal = document.getElementById('event-filter').value;

  var body = document.body;
  var children = Array.prototype.slice.call(body.children);

  // Группируем: [separator, event1, event2, ...], [separator, event...]
  var groups = [];
  var current = null;
  for (var i = 0; i < children.length; i++) {
    var el = children[i];
    if (!el.classList) continue;
    if (el.classList.contains('time-separator')) {
      current = { sep: el, events: [] };
      groups.push(current);
    } else if (el.classList.contains('event')) {
      if (!current) { current = { sep: null, events: [] }; groups.push(current); }
      current.events.push(el);
    }
  }

  var visibleEvents = 0;
  for (var g = 0; g < groups.length; g++) {
    var group = groups[g];
    var visibleInGroup = 0;
    for (var e = 0; e < group.events.length; e++) {
      var ev = group.events[e];
      var evDate = ev.getAttribute('data-date') || '';
      var typeEl = ev.querySelector('.type');
      var evType = typeEl ? (typeEl.textContent || '').trim() : '';

      var dateOk = !dateVal || evDate === dateVal;
      var typeOk = !eventVal || evType === eventVal;

      if (dateOk && typeOk) {
        ev.style.display = '';
        visibleInGroup++;
      } else {
        ev.style.display = 'none';
      }
    }
    if (group.sep) {
      group.sep.style.display = visibleInGroup > 0 ? '' : 'none';
    }
    visibleEvents += visibleInGroup;
  }

  var status = document.getElementById('filter-status');
  var parts = [];
  if (dateVal) parts.push(dateVal);
  if (eventVal) parts.push(eventVal);
  if (parts.length) {
    status.textContent = 'Показано: ' + visibleEvents + ' (' + parts.join(' + ') + ')';
  } else {
    status.textContent = '';
  }
}

function clearFilters() {
  document.getElementById('date-filter').value = '';
  document.getElementById('event-filter').value = '';
  document.getElementById('filter-status').textContent = '';
  var nodes = document.querySelectorAll('.event, .time-separator');
  for (var i = 0; i < nodes.length; i++) { nodes[i].style.display = ''; }
}

// Backward-compat алиас (использовался раньше в onchange).
function applyDateFilter() { applyFilters(); }
function clearDateFilter() { clearFilters(); }

document.addEventListener('DOMContentLoaded', collectEventTypes);
</script>
</head>
<body>
<h1>__SYMBOL__ — события __PERIOD__</h1>
<div class="search-bar">
  <label for="date-filter">Дата:</label>
  <input type="date" id="date-filter" onchange="applyFilters()" />
  <label for="event-filter">Событие:</label>
  <select id="event-filter" onchange="applyFilters()">
    <option value="">— все —</option>
  </select>
  <button onclick="applyFilters()">Применить</button>
  <button class="secondary" onclick="clearFilters()">Сбросить</button>
  <span id="filter-status"></span>
</div>
"""


def _classify_event(event_type: str, message: str) -> str:
    """
    Возвращает CSS-класс для события. Используется для раскраски
    в HTML-логе. CSV-лог цвета не поддерживает.
    """
    et = (event_type or "").upper()

    if et in ("START", "STATE_LOADED", "OPTIMIZATION_DONE",
              "OPTIMIZATION_START"):
        return "ev-info"
    if et == "STOP":
        return "ev-muted"
    if et == "OPEN":
        return "ev-success"
    if et == "CLOSE_WIN":
        return "ev-success"
    if et in ("CLOSE_LOSS", "ERROR", "TPSL_FAILED", "DAILY_STOP"):
        return "ev-danger"
    if et == "CLOSE":
        # Нейтральное закрытие (например, close_all_positions).
        return "ev-muted"
    if et in ("CONFIG_REJECT", "CONFIG_WARNING", "COOLDOWN",
              "COOLDOWN_ACTIVATED", "SPEC_SANITIZED", "NETWORK"):
        return "ev-warning"
    if et in ("SIGNAL_CHECK", "SIGNAL_DETAIL"):
        return "ev-signal"
    if et == "MGMT":
        return "ev-management"
    return "ev-default"

def _format_entry_reasons(reasons) -> str:
    """
    Преобразует entry_reasons в строку для CSV.

    Поддерживает:
    - список строк: ["RSI < 30", "MACD > 0"] → "RSI < 30; MACD > 0"
    - список dict: [{"text": "...", "passed": True}, ...] → "...; ..."
    """
    if not reasons:
        return ""
    
    result = []
    for r in reasons:
        if isinstance(r, str):
            result.append(r)
        elif isinstance(r, dict):
            # Берём только текст условия
            result.append(r.get("text", str(r)))
        else:
            result.append(str(r))
    
    return "; ".join(result)


class TradeLogger:
    """
    Логгер для одной монеты. Автоматически создаёт новые файлы при смене месяца.
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        # Последний записанный в HTML ключ группы — чтобы вставлять
        # разделитель только при смене логической группы (свечи),
        # а не перед каждым событием. Группа может быть timestamp'ом
        # свечи, что склеивает FILTERS+LONG+SHORT одного цикла в один блок.
        self._last_html_group = None
    
    def _week_suffix(self) -> str:
        """
        Возвращает суффикс текущей ISO-недели: YYYY_Www.

        Пример: "2026_W39".
        Файлы ротируются раз в неделю — CSV/HMTL не разрастаются
        до гигантских размеров, а поиск по дате в HTML становится
        практически мгновенным.
        """
        iso = datetime.now(timezone.utc).isocalendar()
        return f"{iso.year}_W{iso.week:02d}"
    
    def _get_file_path(self, log_type: str) -> Path:
        """
        Возвращает путь к CSV-файлу лога за текущую неделю.

        Args:
            log_type: "trades" | "events" | "optimization" | "sl_updates"
        """
        return LOGS_DIR / f"{self.symbol}_{log_type}_{self._week_suffix()}.csv"

    def _get_html_path(self) -> Path:
        """Путь к HTML-файлу событий за текущую неделю."""
        return LOGS_DIR / f"{self.symbol}_events_{self._week_suffix()}.html"
    
    def _append_row(self, file_path: Path, headers: List[str], row: Dict):
        """Дописывает строку в CSV (создаёт файл с заголовками при необходимости)."""
        try:
            file_exists = file_path.exists()
            
            with open(file_path, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                
                if not file_exists:
                    writer.writeheader()
                
                # Заполняем отсутствующие ключи пустыми строками
                full_row = {h: row.get(h, "") for h in headers}
                writer.writerow(full_row)
        
        except Exception as e:
            logger.error(f"Не удалось записать в лог {file_path}: {e}")

    def _append_html_event(
        self,
        timestamp: str,
        event_type: str,
        message: str,
        details: Optional[str] = None,
        group: Optional[str] = None,
    ):
        """
        Дописывает событие в HTML-лог текущего месяца.

        При первом событии месяца пишется шапка (DOCTYPE + CSS).
        При смене timestamp вставляется визуальный разделитель
        с датой/временем — так соседние итерации не сливаются.

        Ошибки записи HTML-лога не влияют на CSV-лог и торговлю.
        """
        try:
            html_path = self._get_html_path()
            file_exists = html_path.exists()

            css_class = _classify_event(event_type, message or "")
            safe_type = html.escape(str(event_type or ""))
            safe_msg = html.escape(str(message or ""))
            safe_details = html.escape(str(details or "")) if details else ""

            details_block = (
                f"<pre>{safe_details}</pre>" if safe_details else ""
            )

            # Первые 10 символов timestamp — это "YYYY-MM-DD".
            # Используется как data-date для клиентской фильтрации
            # в HTML-панели поиска.
            date_part = timestamp[:10] if timestamp else ""

            # Разделитель: при создании нового файла и при смене
            # логической группы. Если группа не задана — используем
            # timestamp (обратная совместимость).
            group_key = group if group is not None else timestamp
            separator = ""
            if not file_exists or group_key != self._last_html_group:
                separator = (
                    f'<div class="time-separator" '
                    f'data-date="{html.escape(date_part)}">'
                    f'{html.escape(timestamp)}</div>\n'
                )
                self._last_html_group = group_key

            block = (
                f'{separator}'
                f'<div class="event {css_class}" '
                f'data-date="{html.escape(date_part)}">\n'
                f'  <div class="type">{safe_type}</div>\n'
                f'  <div class="msg">{safe_msg}</div>\n'
                f'  {details_block}\n'
                f'</div>\n'
            )

            with open(html_path, "a", encoding="utf-8") as f:
                if not file_exists:
                    head = (
                        _HTML_HEAD_TEMPLATE
                        .replace("__SYMBOL__", html.escape(self.symbol))
                        .replace("__PERIOD__", self._week_suffix())
                    )
                    f.write(head)
                f.write(block)

        except Exception as e:
            logger.warning(
                f"Не удалось записать в HTML-лог событий: {e}"
            )
    
    # ----------------------------------------------------------------
    # Сделки
    # ----------------------------------------------------------------
    
    def log_trade(self, trade: Dict):
        """
        Логирует закрытую сделку.
        
        Ожидаемые поля в trade:
            position_id, entry_time, exit_time, direction, entry_price, exit_price,
            qty, result_pct, exit_reason, position_value_usdt, pnl_usdt
        """
        headers = [
            "timestamp", "position_id", "entry_time", "exit_time", "direction",
            "entry_price", "exit_price", "qty", "position_value_usdt",
            "result_pct", "pnl_usdt", "exit_reason",
            "stop_loss", "take_profit",
            "initial_stop_loss", "initial_take_profit",
            "entry_reasons",
            "entry_type",
        ]

        row = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "position_id": trade.get("position_id", ""),
            "entry_time": str(trade.get("entry_time", "")),
            "exit_time": str(trade.get("exit_time", "")),
            "direction": trade.get("direction", ""),
            "entry_price": trade.get("entry_price", ""),
            "exit_price": trade.get("exit_price", ""),
            "qty": trade.get("qty", ""),
            "position_value_usdt": trade.get("position_value_usdt", ""),
            "result_pct": trade.get("result_pct", ""),
            "pnl_usdt": trade.get("pnl_usdt", ""),
            "exit_reason": trade.get("exit_reason", ""),
            "stop_loss": trade.get("stop_loss", ""),
            "take_profit": trade.get("take_profit", ""),
            "initial_stop_loss": trade.get("initial_stop_loss", ""),
            "initial_take_profit": trade.get("initial_take_profit", ""),
            "entry_reasons": _format_entry_reasons(trade.get("entry_reasons", [])),
            "entry_type": trade.get("entry_type", "market"),
        }
        
        self._append_row(self._get_file_path("trades"), headers, row)

    # ----------------------------------------------------------------
    # Изменения stop-loss (trailing, breakeven)
    # ----------------------------------------------------------------

    def log_sl_update(
        self,
        position_id: str,
        direction: str,
        old_sl: float,
        new_sl: float,
        reason: str = "",
        extra: Optional[Dict] = None,
    ):
        """
        Логирует факт изменения SL для открытой позиции.

        Пишется в отдельный файл {SYMBOL}_sl_updates_YYYY_MM.csv,
        чтобы не ломать схему основной таблицы trades.

        Args:
            position_id: ID позиции, к которой относится изменение.
            direction: "long" | "short".
            old_sl: старое значение SL.
            new_sl: новое значение SL.
            reason: почему SL сдвинулся ("breakeven", "trailing",
                    "breakeven+trailing" или свободный текст).
            extra: произвольные дополнительные поля (например, atr,
                   close_price, свеча). Попадут в колонку "extra" как JSON.
        """
        headers = [
            "timestamp", "position_id", "direction",
            "old_sl", "new_sl", "delta",
            "reason", "extra",
        ]

        import json as _json

        try:
            delta = round(float(new_sl) - float(old_sl), 8)
        except (TypeError, ValueError):
            delta = ""

        row = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "position_id": position_id,
            "direction": direction,
            "old_sl": old_sl,
            "new_sl": new_sl,
            "delta": delta,
            "reason": reason,
            "extra": _json.dumps(extra or {}, ensure_ascii=False),
        }

        self._append_row(self._get_file_path("sl_updates"), headers, row)
    
    # ----------------------------------------------------------------
    # События
    # ----------------------------------------------------------------
    
    def log_event(
        self,
        event_type: str,
        message: str,
        details: Optional[str] = None,
        group: Optional[str] = None,
    ):
        """
        Логирует событие приложения.

        Пишет в два файла одновременно:
          - {SYMBOL}_events_YYYY_MM.csv  — для grep и машинной обработки;
          - {SYMBOL}_events_YYYY_MM.html — для человеческого чтения,
            с цветовой раскраской по типу события.

        Args:
            group: необязательный ключ группы для HTML-разделителя.
                Если задан — разделитель ставится при смене группы,
                а не при смене секунды. Используется, чтобы склеить
                FILTERS + LONG + SHORT + SIGNAL_DETAIL одной свечи
                в один блок. Для CSV игнорируется.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        headers = ["timestamp", "event_type", "message", "details"]

        row = {
            "timestamp": timestamp,
            "event_type": event_type,
            "message": message,
            "details": details or "",
        }

        self._append_row(self._get_file_path("events"), headers, row)
        self._append_html_event(
            timestamp, event_type, message, details, group=group,
        )
    
    # ----------------------------------------------------------------
    # Оптимизация
    # ----------------------------------------------------------------
    
    def log_optimization(self, result: Dict):
        """
        Логирует результаты оптимизации стратегии (только итоговые результаты).
        
        Ожидаемые поля:
            trigger_reason, iterations, test_return_pct, test_expectancy_pct,
            test_profit_factor, test_max_drawdown_pct, holdout_status,
            holdout_return_pct, new_strategy_saved
        """
        headers = [
            "timestamp", "trigger_reason", "iterations",
            "test_return_pct", "test_expectancy_pct", "test_profit_factor",
            "test_max_drawdown_pct", "test_total_trades",
            "holdout_status", "holdout_return_pct", "holdout_profit_factor",
            "new_strategy_saved", "notes",
        ]
        
        best = result.get("best") or {}
        test_stats = best.get("aggregated_test_stats", {}) or {}
        holdout = best.get("holdout_summary") or {}
        
        row = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "trigger_reason": result.get("trigger_reason", ""),
            "iterations": len(result.get("history", [])),
            "test_return_pct": test_stats.get("return_pct", ""),
            "test_expectancy_pct": test_stats.get("expectancy_pct", ""),
            "test_profit_factor": test_stats.get("profit_factor", ""),
            "test_max_drawdown_pct": test_stats.get("max_drawdown_pct", ""),
            "test_total_trades": test_stats.get("total_trades", ""),
            "holdout_status": holdout.get("status", ""),
            "holdout_return_pct": holdout.get("return_pct", ""),
            "holdout_profit_factor": holdout.get("profit_factor", ""),
            "new_strategy_saved": result.get("new_strategy_saved", False),
            "notes": result.get("notes", ""),
        }
        
        self._append_row(self._get_file_path("optimization"), headers, row)