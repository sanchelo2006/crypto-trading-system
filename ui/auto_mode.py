"""
Авторежим торговой системы.

Модуль управляет фоновыми потоками по монетам:
- запуск / остановка авторежима;
- утренний пакет при необходимости;
- докачка пропущенных свечей;
- периодические intraday-проверки;
- обновление статусов и логов авторежима.

UI-компоненты напрямую не создаются: взаимодействие с интерфейсом
идёт через переданные callback-функции и очередь событий.
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
    Менеджер состояния авторежима.

    Зависимости передаются один раз через configure(), чтобы модуль был
    отделён от глобальных переменных nicegui_app.py.
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
        Конфигурирует зависимости авторежима.

        Вызывается из nicegui_app.py после успешной инициализации сервисов.
        """
        cls._memory = memory
        cls._fetcher = fetcher
        cls._ai_client = ai_client
        cls._app_settings = app_settings
        cls._process_ai_response = process_ai_response
        cls._catch_up_historical_data = catch_up_historical_data
        cls._add_notification = add_notification
        cls._schedule_refresh = schedule_refresh

        logger.info("AutoModeState сконфигурирован")

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
        Запускает авторежим для конкретной монеты.

        Режимы запуска:
        - full: обычный запуск (докачка, при необходимости утренний пакет, затем цикл)
        - position_only: запуск после ручного открытия позиции
        -> НЕ выполняет стартовую докачку и НЕ запускает утренний пакет,
            чтобы не дублировать анализ позиции отдельным worker'ом из CoinPage.
        """
        if not cls._is_configured():
            logger.warning(
                "AutoModeState.start вызван до configure(); запуск пропущен"
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
                f"[{symbol}] День закрыт. Откройте новый день для торговли.",
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
                    "Запуск сопровождения позиции..."
                    if initial_mode == "position_only"
                    else "Докачка данных..."
                ),
                "needs_update": False,
            }
            state = _auto_threads[symbol]

        def auto_loop():
            if cm.is_day_closed():
                state["log"].append("❌ День закрыт")
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

            # Обычный старт авторежима:
            # - докачка пропущенных свечей
            # - при необходимости утренний пакет
            #
            # Специальный запуск после ручного открытия позиции:
            # - НЕ делаем стартовую докачку
            # - НЕ делаем утренний пакет
            # - НЕ дублируем анализ позиции, потому что он уже выполняется
            #   отдельным worker'ом в CoinPage.open_trade(...)
            if initial_mode != "position_only":
                # === Докачка пропущенных свечей ===
                state["current_action"] = "Докачка данных..."
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

                    state["current_action"] = "Анализ после докачки..."
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

                # === Утренний пакет (если нужен) ===
                if need_morning:
                    try:
                        state["current_action"] = "Утренний пакет..."
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
                        state["current_action"] = "Утренний пакет выполнен"
                        cls._add_notification(
                            f"Утренний пакет {symbol} выполнен",
                            "positive",
                        )
                        cls._schedule_refresh(symbol)

                    except Exception as e:
                        state["log"].append(f"❌ Утренний пакет: {e}")
                        state["current_action"] = f"Ошибка: {e}"
                        cls._add_notification(
                            f"Ошибка утреннего пакета {symbol}: {e}",
                            "negative",
                        )
                else:
                    cm.open_new_day()
                    state["needs_update"] = True
                    state["current_action"] = "День открыт"
            else:
                state["needs_update"] = True
                state["current_action"] = "Авторежим запущен для сопровождения позиции"

            # === Основной цикл ===
            while True:
                with _auto_threads_lock:
                    thread_state = _auto_threads.get(symbol)
                    if not thread_state or not thread_state.get("is_running"):
                        break

                try:
                    if cm.is_day_closed():
                        state["log"].append("❌ День закрыт")
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
                        state["current_action"] = f"Ожидание {nr.strftime('%H:%M UTC')}..."
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
                        f"Анализ ({'позиция' if active else 'поиск входа'})..."
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
                            f"Позиция: {pu.get('advice', 'OK')}"
                        )
                    else:
                        if (
                            response
                            and response.get("signal")
                            and response["signal"].get("confidence_pct", 0) > 0
                        ):
                            state["log"].append(
                                f"[{datetime.now(timezone.utc).strftime('%H:%M')}] "
                                f"СИГНАЛ: {response['signal'].get('direction', '?')}"
                            )

                    state["needs_update"] = True
                    state["current_action"] = "Готово"
                    cls._schedule_refresh(symbol)

                except Exception as e:
                    logger.error("Автоцикл %s: %s", symbol, e, exc_info=True)
                    state["log"].append(f"❌ {e}")
                    state["current_action"] = f"Ошибка: {e}"
                    time.sleep(60)

                if update_callback:
                    update_callback()

        thread = threading.Thread(target=auto_loop, daemon=True)
        state["thread"] = thread
        thread.start()
        state["log"].append(
            f"🟢 Авторежим {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
        )
        if update_callback:
            update_callback()

    @classmethod
    def stop(cls, symbol):
        """
        Останавливает авторежим по конкретной монете.
        """
        with _auto_threads_lock:
            if symbol in _auto_threads:
                _auto_threads[symbol]["is_running"] = False
                _auto_threads[symbol]["log"].append(
                    f"🔴 Остановлен {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
                )

        if cls._schedule_refresh:
            cls._schedule_refresh(symbol)

    @classmethod
    def stop_all(cls):
        """
        Останавливает все авторежимы.
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
        Возвращает состояние авторежима по монете.
        """
        with _auto_threads_lock:
            return _auto_threads.get(symbol)