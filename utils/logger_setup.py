"""
Настройка логирования с устойчивой к Windows и OneDrive ротацией.

Проблема: TimedRotatingFileHandler на Windows падает с WinError 32,
если лог-файл занят другим процессом (синхронизация OneDrive или повторный
запуск скрипта NiceGUI через runpy при обработке 404).

Решение:
  - SafeTimedRotatingFileHandler глушит ошибки ротации, не роняя приложение
    и не засоряя вывод повторяющимися трейсбэками;
  - опциональный вынос логов из синхронизируемой папки OneDrive
    (через переменную окружения LOGS_DIR или автоматически в LOCALAPPDATA).

setup_logging() идемпотентна (защита от повторной инициализации).
"""

import os
import logging
import sys
import time
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler

import config

_logging_initialized = False

def _make_utc_formatter(fmt: str) -> logging.Formatter:
    """
    Форматтер, который пишет asctime в UTC.

    По умолчанию logging.Formatter.formatTime использует
    time.localtime (локальный пояс сервера). Переопределяем converter
    на time.gmtime — получаем UTC-время без tzinfo, а суффикс 'UTC'
    добавляем прямо в fmt.
    """
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
    formatter.converter = time.gmtime
    return formatter


class SafeTimedRotatingFileHandler(TimedRotatingFileHandler):
    """
    Ротация логов, устойчивая к WinError 32 (файл занят другим процессом).

    Если переименование файла не удалось, приложение не падает:
    продолжаем писать в текущий файл и переносим попытку ротации
    на следующий интервал.
    """

    _warned_once = False

    def doRollover(self):
        try:
            super().doRollover()
        except (PermissionError, OSError) as e:
            # Файл занят. Сдвигаем время следующей ротации, чтобы не пытаться
            # ротировать на каждой записи.
            try:
                current_time = int(self.rolloverAt)
                self.rolloverAt = self.computeRollover(current_time)
            except Exception:
                pass

            # Гарантируем, что поток открыт для дальнейшей записи.
            if self.stream is None:
                try:
                    self.stream = self._open()
                except Exception:
                    pass

            # Предупреждаем один раз, без спама в консоль.
            if not SafeTimedRotatingFileHandler._warned_once:
                SafeTimedRotatingFileHandler._warned_once = True
                sys.stderr.write(
                    "[logger_setup] Rotation of log file postponed - file is busy "
                    "(probably OneDrive). Reason: " + str(e) + "\n"
                )
                sys.stderr.write(
                    "[logger_setup] Tip: move the logs folder out of OneDrive "
                    "using the LOGS_DIR environment variable.\n"
                )


def _resolve_logs_dir() -> Path:
    """
    Определяет папку для логов.

    Приоритет:
      1. переменная окружения LOGS_DIR, если задана явно;
      2. если config.LOGS_DIR находится внутри OneDrive, переносим логи
         в подпапку внутри LOCALAPPDATA (вне синхронизации);
      3. иначе используем config.LOGS_DIR как есть.
    """
    # 1. Явно заданная переменная окружения
    env_dir = os.getenv("LOGS_DIR")
    if env_dir:
        p = Path(env_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    base = Path(config.LOGS_DIR)

    # 2. Проверяем, не лежит ли папка внутри OneDrive
    try:
        in_onedrive = "onedrive" in str(base).lower()
    except Exception:
        in_onedrive = False

    if in_onedrive:
        local_appdata = os.getenv("LOCALAPPDATA")
        if local_appdata:
            alt = Path(local_appdata) / "TradingAgent" / "logs"
            try:
                alt.mkdir(parents=True, exist_ok=True)
                sys.stderr.write(
                    "[logger_setup] Logs folder is inside OneDrive. "
                    "Logs redirected to: " + str(alt) + "\n"
                )
                return alt
            except Exception as e:
                sys.stderr.write(
                    "[logger_setup] Could not create folder outside OneDrive ("
                    + str(e) + "), using original: " + str(base) + "\n"
                )

    # 3. Fallback - исходная папка
    base.mkdir(parents=True, exist_ok=True)
    return base


def setup_logging():
    """Настройка корневого логгера с безопасной ротацией. Вызывать ОДИН раз."""
    global _logging_initialized
    if _logging_initialized:
        return
    _logging_initialized = True

    logs_dir = _resolve_logs_dir()

    # Подавляем шум от библиотек. Логи httpx, httpcore, openai при закрытии
    # HTTP-сессий провоцировали трейсбэки ротации, поэтому поднимаем их уровень.
    noisy_loggers = [
        "streamlit.watcher.local_sources_watcher",
        "transformers",
        "transformers.models",
        "PIL",
        "httpx",
        "httpcore",
        "openai",
        "pybit",
        "urllib3",
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    detailed_formatter = _make_utc_formatter(
        "%(asctime)s UTC [%(levelname)s] %(name)s: %(message)s"
    )
    simple_formatter = _make_utc_formatter(
        "%(asctime)s UTC [%(levelname)s] %(message)s"
    )

    root.handlers.clear()

    # Консоль
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
    console_handler.setFormatter(simple_formatter)
    root.addHandler(console_handler)

    # Общий файл app.log - ротация каждый день, храним 7 дней
    app_handler = SafeTimedRotatingFileHandler(
        logs_dir / "app.log",
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
        delay=True,
    )
    app_handler.setLevel(logging.DEBUG)
    app_handler.setFormatter(detailed_formatter)
    root.addHandler(app_handler)

    # Файл ошибок errors.log - ротация каждый день, храним 30 дней
    error_handler = SafeTimedRotatingFileHandler(
        logs_dir / "errors.log",
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
        delay=True,
    )
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(detailed_formatter)
    root.addHandler(error_handler)

    # Файл пользовательских событий user.log
    user_handler = SafeTimedRotatingFileHandler(
        logs_dir / "user.log",
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
        delay=True,
    )
    user_handler.setLevel(logging.INFO)
    user_handler.setFormatter(detailed_formatter)
    user_handler.addFilter(UserLogFilter())
    root.addHandler(user_handler)

    # Файл message.log - только сообщения модели
    message_logger = logging.getLogger("ai_client.messages")
    message_logger.setLevel(logging.INFO)
    message_logger.propagate = False

    message_handler = SafeTimedRotatingFileHandler(
        logs_dir / "message.log",
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
        delay=True,
    )
    message_handler.setLevel(logging.INFO)
    message_handler.setFormatter(simple_formatter)
    message_logger.addHandler(message_handler)

    # Файл key.log - ключевые события
    key_logger = logging.getLogger("key_events")
    key_logger.setLevel(logging.INFO)
    key_logger.propagate = False

    key_handler = SafeTimedRotatingFileHandler(
        logs_dir / "key.log",
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
        delay=True,
    )
    key_handler.setLevel(logging.INFO)
    key_handler.setFormatter(
        _make_utc_formatter("%(asctime)s UTC %(message)s")
    )
    key_logger.addHandler(key_handler)


class UserLogFilter(logging.Filter):
    """Фильтр: пропускает только записи от определённых логгеров."""

    def filter(self, record):
        return record.name in ("ai_client", "ai_client.token", "data_fetcher", "app")