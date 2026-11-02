"""
Менеджер памяти для каждой торгуемой монеты.
Добавлена умная группировка ключевых событий (компрессия) с фильтрацией по времени и дедупликацией.
last_key_events теперь хранится как список словарей.
"""
import os
import shutil
import json
import logging
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple
import pandas as pd

import config

logger = logging.getLogger(__name__)

class CoinMemory:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.data_dir = config.DATA_DIR / symbol
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir = self.data_dir / "archive"
        self.archive_dir.mkdir(exist_ok=True)
        self.journal_path = self.data_dir / "trade_journal.csv"
        self.rules_path = self.data_dir / "trading_rules.md"
        self.state_path = self.data_dir / "state.json"
        self.behavior_path = self.data_dir / "behavior_rules.md"
        self._load()

    def _load(self):
        if self.journal_path.exists():
            try:
                self.journal = pd.read_csv(self.journal_path)
            except Exception:
                self.journal = self._empty_journal()
        else:
            self.journal = self._empty_journal()
        if self.rules_path.exists():
            try:
                with open(self.rules_path, "r", encoding="utf-8") as f:
                    self.rules = f.read()
            except Exception:
                self.rules = ""
        else:
            self.rules = ""
        if self.behavior_path.exists():
            try:
                with open(self.behavior_path, "r", encoding="utf-8") as f:
                    self.behavior_rules = f.read()
            except Exception:
                self.behavior_rules = ""
        else:
            self.behavior_rules = ""
        if self.state_path.exists():
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if raw.get("last_candle_time"):
                    try:
                        raw["last_candle_time"] = datetime.fromisoformat(raw["last_candle_time"])
                    except Exception:
                        raw["last_candle_time"] = None
                # Миграция строки в список
                if isinstance(raw.get("last_key_events"), str):
                    raw["last_key_events"] = []
                self.state = raw
                
            except Exception:
                self.state = self._default_state()
        else:
            self.state = self._default_state()

    def _empty_journal(self) -> pd.DataFrame:
        return pd.DataFrame(columns=[
            "date", "time_utc", "pair", "direction", "entry_price",
            "stop_loss", "take_profit", "exit_price", "result_pct",
            "reason", "errors", "entry_type", "exit_type",
            "signal_snapshot", "market_snapshot"
        ])

    def _default_state(self) -> Dict:
        now = datetime.now(timezone.utc)
        return {
            "is_initialized": False,
            "active_trade": None,
            "last_signal": None,
            "last_setup": None,
            "last_analysis": None,
            "last_update_utc": None,
            "last_morning_update_utc": None,
            "last_candle_time": None,
            "morning_context": "",
            "intraday_context": "",
            "day_closed": False,
            "date": now.strftime("%Y-%m-%d"),
            "trading_rules_list": [],
            "risk_management_rules_list": [],
            "indicators": {},
            "indicator_timestamps": [],
            "last_key_events": [],
        }

    def save(self):
        try:
            self.journal.to_csv(self.journal_path, index=False)
        except Exception as e:
            logger.error(f"Ошибка сохранения журнала {self.symbol}: {e}")
        try:
            with open(self.rules_path, "w", encoding="utf-8") as f:
                f.write(self.rules)
        except Exception as e:
            logger.error(f"Ошибка сохранения правил {self.symbol}: {e}")
        try:
            with open(self.behavior_path, "w", encoding="utf-8") as f:
                f.write(self.behavior_rules)
        except Exception as e:
            logger.error(f"Ошибка сохранения поведенческих правил {self.symbol}: {e}")
        try:
            state_copy = dict(self.state)
            if state_copy.get("last_candle_time") and isinstance(state_copy["last_candle_time"], datetime):
                state_copy["last_candle_time"] = state_copy["last_candle_time"].isoformat()
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(state_copy, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.error(f"Ошибка сохранения состояния {self.symbol}: {e}")

    # ========================
    # Управление ключевыми событиями
    # ========================
    def add_key_events(self, new_events_str: str):
        if not new_events_str:
            return

        # Загружаем существующие события как список словарей
        existing = self.state.get("last_key_events", [])
        if isinstance(existing, str):
            existing = []
        if not isinstance(existing, list):
            existing = []

        # Парсим новые события в структуры
        pattern = (
            r'^• (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) UTC — '
            r'(касание|пробой|аномальный объём|широкий диапазон|консолидация вокруг)'
            r'(?: (\w+))?'
            r'(?: \(([\d.]+)\))?'
            r'(?: (вверх|вниз))?'
            r'(?: \((.*)\))?'
        )

        new_events = []
        for line in new_events_str.splitlines():
            line = line.strip()
            if not line.startswith('•'):
                continue
            match = re.match(pattern, line)
            if not match:
                continue
            time_str = match.group(1)
            event_type = match.group(2)
            level = match.group(3)
            value = float(match.group(4)) if match.group(4) else None
            direction = match.group(5)
            extra = match.group(6)

            new_events.append({
                "time": time_str,
                "type": event_type,
                "level": level,
                "value": value,
                "direction": direction,
                "extra": extra,
            })

        # Дедупликация: ключ — (time, type, level)
        existing_map = {}
        for evt in existing:
            key = (evt.get("time"), evt.get("type"), evt.get("level"))
            existing_map[key] = evt

        for evt in new_events:
            key = (evt["time"], evt["type"], evt["level"])
            if key not in existing_map:
                existing_map[key] = evt
                existing.append(evt)
            else:
                if evt.get("extra"):
                    existing_map[key]["extra"] = evt["extra"]

        # Сортировка по времени
        existing.sort(key=lambda e: e.get("time", ""))

        # --- ФИЛЬТР 1: Дедупликация по уровням (для пробоев/касаний) ---
        filtered = []
        seen_levels = {}
        for evt in existing:
            if evt["type"] in ("касание", "пробой", "консолидация вокруг"):
                level = evt.get("level")
                if not level:
                    filtered.append(evt)
                    continue
                if level in seen_levels:
                    prev = seen_levels[level]
                    prev_type = prev["type"]
                    curr_type = evt["type"]
                    if curr_type == "пробой" and prev_type == "касание":
                        filtered.remove(prev)
                        seen_levels[level] = evt
                        filtered.append(evt)
                    elif curr_type == "касание" and prev_type == "пробой":
                        pass
                    elif evt["time"] > prev["time"]:
                        filtered.remove(prev)
                        seen_levels[level] = evt
                        filtered.append(evt)
                else:
                    seen_levels[level] = evt
                    filtered.append(evt)
            else:
                filtered.append(evt)
        existing = filtered

        # --- ФИЛЬТР 2: Только за последние 24 часа ---
        now = datetime.now(timezone.utc)
        cutoff_24h = now - timedelta(hours=24)
        recent = []
        for evt in existing:
            try:
                dt = datetime.strptime(evt["time"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
                if dt > cutoff_24h:
                    recent.append(evt)
            except ValueError:
                recent.append(evt)
        existing = recent

        # --- ФИЛЬТР 3: Лимит 25 событий ---
        if len(existing) > 25:
            existing = existing[-25:]

        self.state["last_key_events"] = existing

        # Формируем строковое представление для intraday_context
        lines = []
        for evt in existing:
            line = f"• {evt['time']} UTC — {evt['type']}"
            if evt.get("level"):
                line += f" {evt['level']}"
                if evt.get("value") is not None:
                    line += f" ({evt['value']:.4f})"
            if evt.get("direction"):
                line += f" {evt['direction']}"
            if evt.get("extra"):
                line += f" ({evt['extra']})"
            lines.append(line)

        if lines:
            self.state["intraday_context"] = "Ключевые события за период:\n" + "\n".join(lines)
        else:
            self.state["intraday_context"] = ""

        self.save()

    def _compress_events(self, events: List[str]) -> List[str]:
        if not events:
            return []

        pattern = r'^• (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) UTC — (касание|пробой|аномальный объём|широкий диапазон)(?: (\w+))?(?: \(([\d.]+)\))?(?: (вверх|вниз))?(?: \((.*)\))?'

        parsed = []
        for line in events:
            match = re.match(pattern, line)
            if not match:
                parsed.append({'raw': line})
                continue
            dt_str = match.group(1)
            action = match.group(2)
            level_name = match.group(3)
            level_value = match.group(4)
            direction = match.group(5)
            extra = match.group(6)
            try:
                dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            except ValueError:
                parsed.append({'raw': line})
                continue
            parsed.append({
                'dt': dt,
                'action': action,
                'level': level_name,
                'value': float(level_value) if level_value else None,
                'direction': direction,
                'extra': extra,
                'raw': line
            })

        compressed = []
        i = 0
        while i < len(parsed):
            event = parsed[i]
            if event.get('action') in ('аномальный объём', 'широкий диапазон') or 'raw' in event and not event.get('action'):
                compressed.append(event.get('raw', ''))
                i += 1
                continue

            group = [event]
            j = i + 1
            while j < len(parsed):
                next_event = parsed[j]
                if (next_event.get('action') not in ('касание', 'пробой') or
                    next_event.get('level') != event.get('level') or
                    next_event.get('value') != event.get('value')):
                    break
                time_diff = (next_event['dt'] - group[0]['dt']).total_seconds()
                if time_diff > 3600:
                    break
                group.append(next_event)
                j += 1

            if len(group) == 1:
                compressed.append(group[0].get('raw', ''))
            else:
                start_time = group[0]['dt'].strftime('%Y-%m-%d %H:%M')
                end_time = group[-1]['dt'].strftime('%H:%M')
                level_str = f"{event['level']} ({event['value']:.4f})" if event['value'] is not None else event['level']

                touches = sum(1 for e in group if e['action'] == 'касание')
                breakups = sum(1 for e in group if e['action'] == 'пробой' and e['direction'] == 'вверх')
                breakdowns = sum(1 for e in group if e['action'] == 'пробой' and e['direction'] == 'вниз')

                parts = [f"{start_time}–{end_time} UTC — консолидация вокруг {level_str}"]
                if touches:
                    parts.append(f"{touches} касаний")
                if breakups:
                    parts.append(f"{breakups} пробоев вверх")
                if breakdowns:
                    parts.append(f"{breakdowns} пробоев вниз")
                if breakups > breakdowns:
                    parts.append("(давление вверх)")
                elif breakdowns > breakups:
                    parts.append("(давление вниз)")
                else:
                    parts.append("(цена топчется)")

                compressed.append("• " + ", ".join(parts))

            i = j

        return compressed

    def get_last_key_events(self) -> str:
        events = self.state.get("last_key_events", [])
        if isinstance(events, list):
            lines = []
            for evt in events:
                line = f"• {evt.get('time', '')} UTC — {evt.get('type', '')}"
                if evt.get("level"):
                    line += f" {evt['level']}"
                    if evt.get("value") is not None:
                        line += f" ({evt['value']:.4f})"
                if evt.get("direction"):
                    line += f" {evt['direction']}"
                if evt.get("extra"):
                    line += f" ({evt['extra']})"
                lines.append(line)
            return "Ключевые события за период:\n" + "\n".join(lines) if lines else ""
        return str(events)

    def set_last_key_events(self, events):
        self.state["last_key_events"] = events
        self.save()

    # ========================
    # Архивация дня
    # ========================
    def archive_day(self) -> bool:
        date_str = self.state.get("date")
        if not date_str:
            logger.warning(f"[{self.symbol}] Не удалось архивировать: дата не задана в state")
            return False

        day_archive_dir = self.archive_dir / date_str
        if day_archive_dir.exists():
            logger.info(f"[{self.symbol}] День {date_str} уже архивирован")
            return False

        day_archive_dir.mkdir(parents=True, exist_ok=True)

        state_copy = dict(self.state)
        if state_copy.get("last_candle_time") and isinstance(state_copy["last_candle_time"], datetime):
            state_copy["last_candle_time"] = state_copy["last_candle_time"].isoformat()
        with open(day_archive_dir / "state.json", "w", encoding="utf-8") as f:
            json.dump(state_copy, f, ensure_ascii=False, indent=2, default=str)

        (day_archive_dir / "morning_context.md").write_text(self.state.get("morning_context", ""), encoding="utf-8")
        (day_archive_dir / "intraday_context.md").write_text(self.state.get("intraday_context", ""), encoding="utf-8")
        (day_archive_dir / "key_events.md").write_text(self.get_last_key_events(), encoding="utf-8")

        if not self.journal.empty:
            day_trades = self.journal[self.journal["date"] == date_str]
            if not day_trades.empty:
                day_trades.to_csv(day_archive_dir / "trades.csv", index=False)

        logger.info(f"[{self.symbol}] День {date_str} архивирован в {day_archive_dir}")
        return True

    def get_archive_dates(self) -> List[str]:
        if not self.archive_dir.exists():
            return []
        dates = []
        for d in self.archive_dir.iterdir():
            if d.is_dir() and (d / "state.json").exists():
                dates.append(d.name)
        dates.sort(reverse=True)
        return dates

    def load_archive_day(self, date_str: str) -> Optional[Dict]:
        path = self.archive_dir / date_str / "state.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        trades_path = self.archive_dir / date_str / "trades.csv"
        if trades_path.exists():
            data["day_trades"] = pd.read_csv(trades_path).to_dict(orient="records")
        else:
            data["day_trades"] = []
        for file_key, field in [("morning_context.md", "morning_context"),
                                ("intraday_context.md", "intraday_context"),
                                ("key_events.md", "key_events")]:
            fpath = self.archive_dir / date_str / file_key
            if fpath.exists():
                data[field] = fpath.read_text(encoding="utf-8")
        return data

    # --- Управление днём ---
    def close_day(self):
        if self.is_day_closed():
            logger.info(f"[{self.symbol}] День уже закрыт")
            return
        self.archive_day()
        self.state["day_closed"] = True
        self.save()

    def open_new_day(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.state.get("date") == today and not self.is_day_closed():
            logger.info(f"[{self.symbol}] День {today} уже активен")
            return
        if self.state.get("date") and self.state["date"] != today:
            if not self.is_day_closed():
                self.archive_day()
        self.state.update({
            "date": today,
            "day_closed": False,
            "active_trade": None,
            "last_signal": None,
            "last_setup": None,
            "last_analysis": None,
            "last_update_utc": None,
            "last_morning_update_utc": None,
            "last_candle_time": None,
            "morning_context": "",
            "intraday_context": "",
            "last_key_events": [],
            "indicators": {},
            "indicator_timestamps": [],
        })
        self.save()

    def is_day_closed(self) -> bool:
        return self.state.get("day_closed", False)

    # --- Остальные методы ---
    def add_trade_entry(self, entry: Dict):
        now = datetime.now(timezone.utc)
        row = {
            "date": now.strftime("%Y-%m-%d"), "time_utc": now.strftime("%H:%M:%S"),
            "pair": self.symbol, "direction": entry.get("direction", ""),
            "entry_price": entry.get("entry_price", 0), "stop_loss": entry.get("stop_loss", 0),
            "take_profit": entry.get("take_profit", 0), "exit_price": entry.get("exit_price", 0),
            "result_pct": entry.get("result_pct", 0), "reason": entry.get("reason", ""),
            "errors": entry.get("errors", ""),
            "entry_type": entry.get("entry_type", "manual"),
            "exit_type": entry.get("exit_type", "manual"),
            "signal_snapshot": json.dumps(entry.get("signal_snapshot", {}), ensure_ascii=False) if entry.get("signal_snapshot") else "",
            "market_snapshot": json.dumps(entry.get("market_snapshot", {}), ensure_ascii=False) if entry.get("market_snapshot") else "",
        }
        self.journal = pd.concat([self.journal, pd.DataFrame([row])], ignore_index=True)
        self.save()

    def get_journal_df(self) -> pd.DataFrame:
        return self.journal.copy()

    def get_trade_stats(self) -> Dict:
        if self.journal.empty:
            return {"total_trades": 0, "win_rate": 0, "total_pnl_pct": 0, "avg_win_pct": 0, "avg_loss_pct": 0}
        total = len(self.journal)
        wins = len(self.journal[self.journal["result_pct"] > 0])
        win_rate = wins / total * 100 if total > 0 else 0
        total_pnl = self.journal["result_pct"].sum()
        return {
            "total_trades": total, "win_rate": round(win_rate, 1),
            "total_pnl_pct": round(total_pnl, 2),
            "avg_win_pct": round(self.journal[self.journal["result_pct"] > 0]["result_pct"].mean(), 2) if wins > 0 else 0,
            "avg_loss_pct": round(self.journal[self.journal["result_pct"] < 0]["result_pct"].mean(), 2) if total - wins > 0 else 0,
        }

    def get_day_pnl(self, date_str: str = None) -> float:
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.journal.empty:
            return 0.0
        day_trades = self.journal[self.journal["date"] == date_str]
        return day_trades["result_pct"].sum() if not day_trades.empty else 0.0

    def get_rules(self) -> str: return self.rules
    def update_rules(self, new_rules: str): self.rules = new_rules; self.save()
    def append_rule(self, new_rule: str):
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        self.rules += f"\n\n### Правило добавлено {timestamp}\n{new_rule}"
        self.save()

    def get_behavior_rules(self) -> str: return self.behavior_rules
    def update_behavior_rules(self, text: str): self.behavior_rules = text; self.save()
    def append_behavior_rule(self, text: str):
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        self.behavior_rules += f"\n\n### Записано {timestamp}\n{text}"
        self.save()

    def set_morning_context(self, context: str): self.state["morning_context"] = context; self.save()
    def get_morning_context(self) -> str: return self.state.get("morning_context", "")
    def set_intraday_context(self, context: str): self.state["intraday_context"] = context; self.save()
    def get_intraday_context(self) -> str: return self.state.get("intraday_context", "")
    def set_last_analysis(self, analysis: Optional[Dict]): self.state["last_analysis"] = analysis; self.save()
    def get_last_analysis(self) -> Optional[Dict]: return self.state.get("last_analysis")
    def set_last_candle_time(self, dt: Optional[datetime]): self.state["last_candle_time"] = dt; self.save()
    def get_last_candle_time(self) -> Optional[datetime]:
        val = self.state.get("last_candle_time")
        if val is None: return None
        if isinstance(val, datetime): return val
        if isinstance(val, str):
            try: return datetime.fromisoformat(val)
            except: return None
        return None

    def get_state(self) -> Dict: return self.state.copy()
    def update_state(self, updates: Dict): self.state.update(updates); self.save()
    def set_active_trade(self, trade: Optional[Dict]): self.state["active_trade"] = trade; self.save()
    def get_active_trade(self) -> Optional[Dict]: return self.state.get("active_trade")
    def is_initialized(self) -> bool: return self.state.get("is_initialized", False)
    def mark_initialized(self): self.state["is_initialized"] = True; self.save()


    def reload(self):
        self._load()

class MemoryManager:
    def __init__(self):
        self.memories: Dict[str, CoinMemory] = {}
        self._load_all()

    def _load_all(self):
        for pair_cfg in config.TRADING_PAIRS:
            self.memories[pair_cfg["symbol"]] = CoinMemory(pair_cfg["symbol"])

    def get(self, symbol: str) -> CoinMemory:
        if symbol not in self.memories:
            self.memories[symbol] = CoinMemory(symbol)
        return self.memories[symbol]

    def get_all_symbols(self) -> List[str]:
        return list(self.memories.keys())

    def add_symbol(self, symbol: str):
        if symbol not in self.memories:
            self.memories[symbol] = CoinMemory(symbol)

    def remove_symbol(self, symbol: str):
        if symbol in self.memories:
            del self.memories[symbol]
        config.TRADING_PAIRS[:] = [p for p in config.TRADING_PAIRS if p["symbol"] != symbol]
