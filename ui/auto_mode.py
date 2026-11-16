"""
Auto mode of the trading system.

The module manages per-coin background threads:
- starting / stopping auto mode;
- the morning package when needed;
- catching up missed candles;
- periodic intraday checks;
- updating auto mode statuses and logs.

No UI components are created directly: interaction with the interface
goes through the passed callback functions and the event queue.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Callable, Dict, Optional
import config


logger = logging.getLogger(__name__)


_auto_threads: Dict[str, dict] = {}
_auto_threads_lock = threading.Lock()


class AutoModeState:
    """
    Auto mode state manager.

    Dependencies are passed once via configure(), so that the module stays
    decoupled from the global variables of nicegui_app.py.
    """

    _memory = None
    _fetcher = None
    _ai_client = None
    _app_settings = None

    _process_ai_response: Optional[Callable] = None
    _catch_up_historical_data: Optional[Callable] = None
    _add_notification: Optional[Callable[[str, str], None]] = None
    _schedule_refresh: Optional[Callable[[str], None]] = None

    @classmethod
    def configure(
        cls,
        *,
        memory,
        fetcher,
        ai_client,
        app_settings,
        process_ai_response: Callable,
        catch_up_historical_data: Callable,
        add_notification: Callable[[str, str], None],
        schedule_refresh: Callable[[str], None],
    ) -> None:
        """
        Configures the auto mode dependencies.

        Called from nicegui_app.py after the services are successfully initialized.
        """
        cls._memory = memory
        cls._fetcher = fetcher
        cls._ai_client = ai_client
        cls._app_settings = app_settings
        cls._process_ai_response = process_ai_response
        cls._catch_up_historical_data = catch_up_historical_data
        cls._add_notification = add_notification
        cls._schedule_refresh = schedule_refresh

        logger.info("AutoModeState configured")

    @classmethod
    def _is_configured(cls) -> bool:
        return all(
            [
                cls._memory is not None,
                cls._fetcher is not None,
                cls._ai_client is not None,
                cls._app_settings is not None,
                cls._process_ai_response is not None,
                cls._catch_up_historical_data is not None,
                cls._add_notification is not None,
                cls._schedule_refresh is not None,
            ]
        )

    @classmethod
    def start(cls, symbol, category, update_callback=None, initial_mode: str = "full"):
        """
        Starts auto mode for a specific coin.

        Start modes:
        - full: regular start (catch-up, morning package if needed, then the loop)
        - position_only: start after a position was opened manually
        -> does NOT run the initial catch-up and does NOT run the morning package,
            to avoid duplicating the position analysis by a separate worker from CoinPage.
        """
        if not cls._is_configured():
            logger.warning(
                "AutoModeState.start called before configure(); start skipped"
            )
            return

        cm = cls._memory.get(symbol)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if cm.get_state().get("date") != today:
            if not cm.is_day_closed():
                cm.archive_day()
            cm.open_new_day()

        if cm.is_day_closed() and cm.get_state().get("date") == today:
            cls._add_notification(
                f"[{symbol}] Day is closed. Open a new day to trade.",
                "warning",
            )
            if update_callback:
                update_callback()
            return

        with _auto_threads_lock:
            if symbol in _auto_threads and _auto_threads[symbol].get("is_running"):
                return

            _auto_threads[symbol] = {
                "is_running": True,
                "thread": None,
                "log": [],
                "current_action": (
                    "Starting position management..."
                    if initial_mode == "position_only"
                    else "Catching up data..."
                ),
                "needs_update": False,
            }
            state = _auto_threads[symbol]

        def auto_loop():
            if cm.is_day_closed():
                state["log"].append("❌ Day is closed")
                state["is_running"] = False
                if update_callback:
                    update_callback()
                return

            now = datetime.now(timezone.utc)
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

            last_morning_str = cm.get_state().get("last_morning_update_utc")
            last_morning_dt = None
            if last_morning_str:
                try:
                    last_morning_dt = datetime.fromisoformat(last_morning_str)
                except Exception:
                    pass

            need_morning = not (last_morning_dt and last_morning_dt >= today_start)

            # Regular auto mode start:
            # - catch up missed candles
            # - morning package if needed
            #
            # Special start after a position was opened manually:
            # - do NOT run the initial catch-up
            # - do NOT run the morning package
            # - do NOT duplicate the position analysis, because it is already
            #   running in a separate worker in CoinPage.open_trade(...)
            if initial_mode != "position_only":
                # === Catch-up of missed candles ===
                state["current_action"] = "Catching up data..."
                pkg, new_time = cls._catch_up_historical_data(
                    fetcher=cls._fetcher,
                    symbol=symbol,
                    category=category,
                    coin_memory=cm,
                )

                if pkg is not None and new_time is not None:
                    cm.set_last_candle_time(new_time)
                    data = pkg.get("data", {})

                    indicators_str = data.get("indicators", "{}")
                    if indicators_str:
                        try:
                            indicators = (
                                json.loads(indicators_str)
                                if isinstance(indicators_str, str)
                                else indicators_str
                            )
                            cm.update_state(
                                {
                                    "indicators": indicators,
                                    "indicator_timestamps": data.get(
                                        "indicators_timestamps", []
                                    ),
                                }
                            )
                        except Exception:
                            pass

                    market_stats_raw = data.get("market_stats", "{}")
                    if market_stats_raw and market_stats_raw != "{}":
                        try:
                            stats = (
                                json.loads(market_stats_raw)
                                if isinstance(market_stats_raw, str)
                                else market_stats_raw
                            )
                            cm.update_state({"market_stats": stats})
                        except Exception:
                            pass

                    state["current_action"] = "Analysis after catch-up..."
                    active = cm.get_active_trade()
                    response = cls._ai_client.analyze(
                        pkg,
                        cm.get_rules(),
                        active,
                        morning_context=cm.get_morning_context(),
                        intraday_context=cm.get_intraday_context(),
                        behavior_rules=cm.get_behavior_rules(),
                    )
                    if response:
                        cls._process_ai_response(
                            response,
                            cm,
                            symbol,
                            has_active_trade=bool(active),
                            data_package=pkg,
                        )
                    state["needs_update"] = True
                    cls._schedule_refresh(symbol)

                # === Morning package (if needed) ===
                if need_morning:
                    try:
                        state["current_action"] = "Morning package..."
                        pkg, new_time = cls._fetcher.get_morning_package(
                            symbol,
                            category,
                            settings=cls._app_settings.data,
                        )
                        if new_time:
                            cm.set_last_candle_time(new_time)

                        active = cm.get_active_trade()
                        response = cls._ai_client.analyze(
                            pkg,
                            cm.get_rules(),
                            active,
                            is_initial=cm.is_day_closed(),
                            behavior_rules=cm.get_behavior_rules(),
                        )
                        if response:
                            cls._process_ai_response(
                                response,
                                cm,
                                symbol,
                                has_active_trade=bool(active),
                                data_package=pkg,
                            )

                        cm.open_new_day()
                        cm.update_state(
                            {"last_morning_update_utc": datetime.now(timezone.utc).isoformat()}
                        )
                        state["needs_update"] = True
                        state["current_action"] = "Morning package completed"
                        cls._add_notification(
                            f"Morning package {symbol} completed",
                            "positive",
                        )
                        cls._schedule_refresh(symbol)

                    except Exception as e:
                        state["log"].append(f"❌ Morning package: {e}")
                        state["current_action"] = f"Error: {e}"
                        cls._add_notification(
                            f"Morning package error {symbol}: {e}",
                            "negative",
                        )
                else:
                    cm.open_new_day()
                    state["needs_update"] = True
                    state["current_action"] = "Day opened"
            else:
                state["needs_update"] = True
                state["current_action"] = "Auto mode started for position management"

            # === Main loop ===
            while True:
                with _auto_threads_lock:
                    thread_state = _auto_threads.get(symbol)
                    if not thread_state or not thread_state.get("is_running"):
                        break

                try:
                    if cm.is_day_closed():
                        state["log"].append("❌ Day is closed")
                        state["is_running"] = False
                        if update_callback:
                            update_callback()
                        return

                    now = datetime.now(timezone.utc)
                    nm = ((now.minute // 15) + 1) * 15
                    nr = now.replace(
                        minute=0, second=config.SCHEDULE_SECOND, microsecond=0,
                    )
                    nr = nr.replace(minute=nm) if nm < 60 else nr + timedelta(hours=1)
                    wait = (nr - now).total_seconds()

                    if 0 < wait < 900:
                        state["current_action"] = f"Waiting for {nr.strftime('%H:%M UTC')}..."
                        for _ in range(int(wait)):
                            with _auto_threads_lock:
                                thread_state = _auto_threads.get(symbol)
                                if not thread_state or not thread_state.get("is_running"):
                                    if update_callback:
                                        update_callback()
                                    return
                            time.sleep(1)
                        continue

                    active = cm.get_active_trade()
                    state["current_action"] = (
                        f"Analysis ({'position' if active else 'entry search'})..."
                    )

                    lt = cm.get_last_candle_time()
                    has_setup = cm.get_state().get("last_setup") is not None and not active

                    if has_setup or active:
                        pkg, new_time = cls._fetcher.get_intraday_light_package(
                            symbol,
                            category,
                            from_time=lt,
                            settings=cls._app_settings.data,
                        )
                    else:
                        pkg, new_time = cls._fetcher.get_intraday_package(
                            symbol,
                            category,
                            from_time=lt,
                            settings=cls._app_settings.data,
                        )

                    if new_time:
                        cm.set_last_candle_time(new_time)

                    response = cls._ai_client.analyze(
                        pkg,
                        cm.get_rules(),
                        active,
                        morning_context=cm.get_morning_context(),
                        intraday_context=cm.get_intraday_context(),
                        behavior_rules=cm.get_behavior_rules(),
                    )
                    if response:
                        cls._process_ai_response(
                            response,
                            cm,
                            symbol,
                            has_active_trade=bool(active),
                            data_package=pkg,
                        )

                    pu = response.get("position_update", {}) if response else {}

                    if active:
                        state["log"].append(
                            f"[{datetime.now(timezone.utc).strftime('%H:%M')}] "
                            f"Position: {pu.get('advice', 'OK')}"
                        )
                    else:
                        if (
                            response
                            and response.get("signal")
                            and response["signal"].get("confidence_pct", 0) > 0
                        ):
                            state["log"].append(
                                f"[{datetime.now(timezone.utc).strftime('%H:%M')}] "
                                f"SIGNAL: {response['signal'].get('direction', '?')}"
                            )

                    state["needs_update"] = True
                    state["current_action"] = "Done"
                    cls._schedule_refresh(symbol)

                except Exception as e:
                    logger.error("Auto loop %s: %s", symbol, e, exc_info=True)
                    state["log"].append(f"❌ {e}")
                    state["current_action"] = f"Error: {e}"
                    time.sleep(60)

                if update_callback:
                    update_callback()

        thread = threading.Thread(target=auto_loop, daemon=True)
        state["thread"] = thread
        thread.start()
        state["log"].append(
            f"🟢 Auto mode {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
        )
        if update_callback:
            update_callback()

    @classmethod
    def stop(cls, symbol):
        """
        Stops auto mode for a specific coin.
        """
        with _auto_threads_lock:
            if symbol in _auto_threads:
                _auto_threads[symbol]["is_running"] = False
                _auto_threads[symbol]["log"].append(
                    f"🔴 Stopped {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
                )

        if cls._schedule_refresh:
            cls._schedule_refresh(symbol)

    @classmethod
    def stop_all(cls):
        """
        Stops all auto modes.
        """
        with _auto_threads_lock:
            symbols = list(_auto_threads.keys())
            for symbol in symbols:
                _auto_threads[symbol]["is_running"] = False

        if cls._schedule_refresh:
            for symbol in symbols:
                cls._schedule_refresh(symbol)

    @classmethod
    def get_state(cls, symbol):
        """
        Returns the auto mode state for a coin.
        """
        with _auto_threads_lock:
            return _auto_threads.get(symbol)