"""
Подробный текстовый отчёт одной оптимизации.

Пишется потоково (append):
  - не теряет данные при падении или отмене оптимизации;
  - не зависит от UI;
  - не влияет на размер промпта и расход токенов.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class OptimizationReport:

    # Поля с большими массивами данных. В текстовом отчёте заменяются
    # на строку-сводку «<N элементов>». Полные данные сохраняются
    # в CSV-файлах через save_round().
    _SKIP_LIST_FIELDS = frozenset({
        "trades", "train_trades", "test_trades",
        "fold_results",
        "rejected_fingerprints", "rejected_specs",
    })

    # Поля с длинным текстом, которые обрезаются до _MAX_TEXT_LEN.
    _TRUNCATE_FIELDS = frozenset({
        "reasoning",
    })

    _MAX_TEXT_LEN = 500

    def __init__(self, root_dir: Path, symbol: str):
        self.dir = Path(root_dir) / "optimization_reports"
        self.dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_symbol = "".join(
            ch if ch.isalnum() or ch in "_-" else "_" for ch in symbol
        )
        self.path = self.dir / f"{safe_symbol}_{stamp}.txt"

        self._append(
            "=" * 100 + "\n"
            f"ОПТИМИЗАЦИЯ {symbol}\n"
            f"UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
            f"Файл отчёта: {self.path}\n"
            + "=" * 100 + "\n\n"
        )

    def _append(self, text: str) -> None:
        # Ошибка записи отчёта НИКОГДА не должна ронять оптимизацию.
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            logger.exception("Не удалось записать отчёт: %s", self.path)

    def line(self, message: str) -> None:
        """Одна строка прогресса (то, что сейчас уходит в лог)."""
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self._append(f"[{stamp}] {message}\n")

    def _sanitize(self, data: Any) -> Any:
        """Рекурсивно удаляет тяжёлые поля из данных для отчёта.

        - Списки сделок и результатов окон заменяются на счётчик.
        - Длинные тексты (reasoning) обрезаются.
        - Остальные данные проходят без изменений.
        """
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if key in self._SKIP_LIST_FIELDS and isinstance(value, list):
                    result[key] = f"<{len(value)} элементов — полные данные в CSV>"
                elif (key in self._TRUNCATE_FIELDS
                      and isinstance(value, str)
                      and len(value) > self._MAX_TEXT_LEN):
                    result[key] = (value[:self._MAX_TEXT_LEN]
                                   + f"... (обрезано, всего {len(value)} символов)")
                else:
                    result[key] = self._sanitize(value)
            return result
        elif isinstance(data, list):
            return [self._sanitize(item) for item in data]
        return data


    def section(self, title: str, data: Any) -> None:
        """Структурированный блок: spec, метрики, ответ AI и т.д.

        Данные проходят через _sanitize(): тяжёлые массивы (сделки,
        результаты окон) заменяются на счётчики, длинные тексты обрезаются.
        """
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        try:
            clean = self._sanitize(data)
            body = json.dumps(clean, ensure_ascii=False, indent=2, default=str)
        except Exception as exc:
            body = f"<не удалось сериализовать: {exc}>\n{data!r}"

        self._append(
            f"\n{'-' * 100}\n"
            f"[{stamp}] {title}\n"
            f"{'-' * 100}\n"
            f"{body}\n"
        )