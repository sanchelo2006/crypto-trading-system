"""
Logging setup with rotation that is robust on Windows and OneDrive.

Problem: TimedRotatingFileHandler on Windows fails with WinError 32
if the log file is locked by another process (OneDrive sync or a repeated
NiceGUI script launch via runpy while handling 404).

Solution:
- SafeTimedRotatingFileHandler suppresses rotation errors without crashing
  the app and without spamming repeated tracebacks;
- optional relocation of logs out of the OneDrive-synced folder
  (via the LOGS_DIR environment variable or automatically into LOCALAPPDATA).

setup_logging() is idempotent (protected against repeated initialization).
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
    Formatter that writes asctime in UTC.
    By default logging.Formatter.formatTime uses time.localtime
    (the server's local timezone). We override converter with time.gmtime
    to get UTC time without tzinfo, and add the 'UTC' suffix directly in fmt.
    """
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
    formatter.converter = time.gmtime
    return formatter


class SafeTimedRotatingFileHandler(TimedRotatingFileHandler):
    """
    Log rotation robust to WinError 32 (file locked by another process).
    If renaming the file fails, the app does not crash: we keep writing
    to the current file and postpone the rotation to the next interval.
    """

    _warned_once = False

    def doRollover(self):
        try:
            super().doRollover()
        except (PermissionError, OSError) as e:
            # File is busy. Shift the next rotation time so we do not retry
            # rotation on every record.
            try:
                current_time = int(self.rolloverAt)
                self.rolloverAt = self.computeRollover(current_time)
            except Exception:
                pass

            # Make sure the stream is open for further writing.
            if self.stream is None:
                try:
                    self.stream = self._open()
                except Exception:
                    pass

            # Warn once, without spamming the console.
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
    Determines the logs folder.
    Priority:
    1. the LOGS_DIR environment variable, if set explicitly;
    2. if config.LOGS_DIR is inside OneDrive, move logs to a subfolder
       inside LOCALAPPDATA (outside sync);
    3. otherwise use config.LOGS_DIR as is.
    """
    # 1. Explicitly set environment variable
    env_dir = os.getenv("LOGS_DIR")
    if env_dir:
        p = Path(env_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    base = Path(config.LOGS_DIR)

    # 2. Check whether the folder is inside OneDrive
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

    # 3. Fallback - original folder
    base.mkdir(parents=True, exist_ok=True)
    return base


def setup_logging():
    """Set up the root logger with safe rotation. Call ONCE."""
    global _logging_initialized
    if _logging_initialized:
        return
    _logging_initialized = True

    logs_dir = _resolve_logs_dir()

    # Suppress library noise. httpx, httpcore and openai logs triggered
    # rotation tracebacks when closing HTTP sessions, so we raise their level.
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

    	
    # Console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
    console_handler.setFormatter(simple_formatter)
    root.addHandler(console_handler)

    # Shared app.log - daily rotation, keep 7 days
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

    	
    # Error file errors.log - daily rotation, keep 30 days
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

    # User events file user.log
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

    # message.log - model messages only
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

    # key.log - key events
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
    """Filter: passes only records from specific loggers."""

    def filter(self, record):
        return record.name in ("ai_client", "ai_client.token", "data_fetcher", "app")