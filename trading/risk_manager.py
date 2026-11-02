"""
Контроль рисков автоторговли: дневные и месячные убытки по монетам.

Ведёт учёт реализованного и нереализованного ПнЛ. При достижении лимитов
инициирует остановку торговли (дневной) или оптимизацию стратегии (месячный).

Состояние сохраняется на диск, чтобы не теряться при перезапуске приложения.

ВАЖНО: лимиты считаются от ТЕКУЩЕГО капитала монеты (динамически), а не от начального.
"""

import json
import logging
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional

import config

logger = logging.getLogger(__name__)

STATE_FILE = config.DATA_DIR / "auto_trading" / "risk_state.json"

# Сериализует доступ к файлу из потоков нескольких монет.
# Без него потоки могут затирать апдейты друг друга и читать файл
# в момент его усечения другим потоком.
_state_file_lock = threading.Lock()

class RiskManager:
    """
    Контроллер рисков по одной монете.

    Отслеживает:
    - дневной реализованный ПнЛ (по закрытым сделкам за сегодня)
    - месячный реализованный ПnЛ (по закрытым сделкам за текущий месяц)
    - нереализованный ПnЛ открытых позиций (для полного контроля дневного лимита)
    - ТЕКУЩИЙ капитал монеты (пересчитывается после каждой сделки)
    """

    def __init__(
        self,
        symbol: str,
        daily_loss_limit_pct: float,
        monthly_loss_limit_pct: float,
        allocated_capital_usdt: float,
    ):
        """
        Args:
            symbol: торговая пара
            daily_loss_limit_pct: лимит дневного убытка в % от ТЕКУЩЕГО капитала монеты
            monthly_loss_limit_pct: лимит месячного убытка в % от ТЕКУЩЕГО капитала монеты
            allocated_capital_usdt: начальный выделенный капитал монеты в USDT
        """
        self.symbol = symbol
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.monthly_loss_limit_pct = monthly_loss_limit_pct
        
        # ТЕКУЩИЙ капитал (изменяется после каждой сделки)
        self.current_capital = allocated_capital_usdt
        
        # Начальный капитал (только для логирования)
        self.initial_capital = allocated_capital_usdt

        # Реализованный ПнЛ в USDT
        self.daily_realized_pnl = 0.0
        self.monthly_realized_pnl = 0.0

        # Даты учёта
        self.current_date = datetime.now(timezone.utc).date()
        self.current_month = self.current_date.strftime("%Y-%m")

        # Триггеры (чтобы не срабатывали повторно)
        self.daily_stop_triggered = False
        self.monthly_optimization_triggered = False

        # Загружаем сохранённое состояние
        self._load_state()

    # ----------------------------------------------------------------
    # Учёт ПнЛ
    # ----------------------------------------------------------------

    def record_closed_trade(self, pnl_pct: float, position_value_usdt: float):
        """
        Фиксирует закрытую сделку.

        Args:
            pnl_pct: результат сделки в %
            position_value_usdt: стоимость позиции в USDT (для расчёта абсолютного ПнЛ)
        """
        self._check_rollover()

        pnl_usdt = position_value_usdt * (pnl_pct / 100.0)
        self.daily_realized_pnl += pnl_usdt
        self.monthly_realized_pnl += pnl_usdt
        
        # ОБНОВЛЯЕМ ТЕКУЩИЙ КАПИТАЛ
        self.current_capital += pnl_usdt

        logger.info(
            f"[{self.symbol}] Фиксация сделки: {pnl_usdt:+.2f} USDT. "
            f"Текущий капитал: {self.current_capital:.2f} USDT, "
            f"День: {self.daily_realized_pnl:+.2f} USDT, "
            f"Месяц: {self.monthly_realized_pnl:+.2f} USDT"
        )

        self._save_state()

    def update_capital(self, new_capital: float):
        """
        Обновляет текущий капитал монеты (например, после пополнения счёта).
        
        Args:
            new_capital: новый капитал в USDT
        """
        old = self.current_capital
        self.current_capital = new_capital
        logger.info(f"[{self.symbol}] Капитал обновлён: {old:.2f} → {new_capital:.2f} USDT")
        self._save_state()

    def _check_rollover(self):
        """Проверяет, не наступил ли новый день/месяц, и сбрасывает счётчики."""
        now = datetime.now(timezone.utc).date()

        # Новый день
        if now != self.current_date:
            logger.info(f"[{self.symbol}] Новый день. Сброс дневного счётчика.")
            self.current_date = now
            self.daily_realized_pnl = 0.0
            self.daily_stop_triggered = False

        # Новый месяц
        new_month = now.strftime("%Y-%m")
        if new_month != self.current_month:
            logger.info(f"[{self.symbol}] Новый месяц. Сброс месячного счётчика.")
            self.current_month = new_month
            self.monthly_realized_pnl = 0.0
            self.monthly_optimization_triggered = False

        self._save_state()

    # ----------------------------------------------------------------
    # Проверка лимитов
    # ----------------------------------------------------------------

    def get_total_daily_loss_pct(self, unrealized_pnl_usdt: float) -> float:
        """
        Возвращает суммарный дневной убыток в % от ТЕКУЩЕГО капитала
        (реализованный + нереализованный).
        """
        total_usdt = self.daily_realized_pnl + unrealized_pnl_usdt
        if self.current_capital <= 0:
            return 0.0
        return (total_usdt / self.current_capital) * 100.0

    def get_total_monthly_loss_pct(self) -> float:
        """Возвращает месячный реализованный убыток в % от ТЕКУЩЕГО капитала."""
        if self.current_capital <= 0:
            return 0.0
        return (self.monthly_realized_pnl / self.current_capital) * 100.0

    def should_stop_daily(self, unrealized_pnl_usdt: float) -> bool:
        """
        Проверяет, достигнут ли дневной лимит убытка.

        Срабатывает один раз (флаг), чтобы не дублировать остановку.
        """
        if self.daily_stop_triggered:
            return False

        total_loss_pct = self.get_total_daily_loss_pct(unrealized_pnl_usdt)

        # Лимит задан как положительное число (например, 5 = убыток 5%)
        if total_loss_pct <= -abs(self.daily_loss_limit_pct):
            logger.warning(
                f"[{self.symbol}] Достигнут дневной лимит убытка: "
                f"{total_loss_pct:.2f}% от текущего капитала {self.current_capital:.2f} USDT "
                f"(лимит -{self.daily_loss_limit_pct}%)"
            )
            self.daily_stop_triggered = True
            self._save_state()
            return True

        return False

    def should_optimize_monthly(self) -> bool:
        """
        Проверяет, достигнут ли месячный лимит убытка для запуска оптимизации.

        Срабатывает один раз (флаг), чтобы не запускать оптимизацию повторно.
        """
        if self.monthly_optimization_triggered:
            return False

        monthly_loss_pct = self.get_total_monthly_loss_pct()

        if monthly_loss_pct <= -abs(self.monthly_loss_limit_pct):
            logger.warning(
                f"[{self.symbol}] Достигнут месячный лимит убытка: "
                f"{monthly_loss_pct:.2f}% от текущего капитала {self.current_capital:.2f} USDT "
                f"(лимит -{self.monthly_loss_limit_pct}%). "
                "Требуется оптимизация стратегии."
            )
            self.monthly_optimization_triggered = True
            self._save_state()
            return True

        return False

    # ----------------------------------------------------------------
    # Сохранение состояния
    # ----------------------------------------------------------------

    def _save_state(self):
        """
        Сохраняет состояние риск-менеджера на диск.

        Атомарно (temp-файл + replace) и под локом: краш в момент записи
        или параллельная запись другой монеты не оставят пустой файл.
        """
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

            with _state_file_lock:
                state = {}
                if STATE_FILE.exists():
                    try:
                        with open(STATE_FILE, "r", encoding="utf-8") as f:
                            content = f.read().strip()
                        if content:
                            state = json.loads(content)
                    except (json.JSONDecodeError, OSError) as e:
                        logger.warning(
                            f"[{self.symbol}] risk_state.json повреждён ({e}). "
                            "Начинаю с чистого состояния."
                        )
                        state = {}

                state[self.symbol] = {
                    "daily_realized_pnl": self.daily_realized_pnl,
                    "monthly_realized_pnl": self.monthly_realized_pnl,
                    "current_capital": self.current_capital,
                    "initial_capital": self.initial_capital,
                    "allocated_capital_setting": self.initial_capital,
                    "current_date": str(self.current_date),
                    "current_month": self.current_month,
                    "daily_stop_triggered": self.daily_stop_triggered,
                    "monthly_optimization_triggered": self.monthly_optimization_triggered,
                }

                tmp_path = STATE_FILE.with_suffix(".json.tmp")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(state, f, ensure_ascii=False, indent=2)
                tmp_path.replace(STATE_FILE)

        except Exception as e:
            logger.error(f"Не удалось сохранить состояние риск-менеджера: {e}")

    def _log_alloc_change_if_available(self, old_initial: float) -> None:
        """Хук для случая, когда риск-менеджер захочет уведомить UI.
        Сейчас просто пишет в логгер, но оставлен отдельным методом,
        чтобы потом было куда подключить notify_fn без правки _load_state.
        """
        pass

    def _load_state(self):
        """Загружает состояние риск-менеджера с диска."""
        try:
            with _state_file_lock:
                if not STATE_FILE.exists():
                    return

                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                if not content:
                    logger.info(
                        f"[{self.symbol}] risk_state.json пуст — "
                        "использую начальное состояние"
                    )
                    return

                try:
                    state = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"[{self.symbol}] risk_state.json повреждён ({e}). "
                        "Использую начальное состояние."
                    )
                    return

                symbol_state = state.get(self.symbol)
                if not symbol_state:
                    return

                # Восстанавливаем только если даты актуальны
                saved_date = symbol_state.get("current_date")
                saved_month = symbol_state.get("current_month")

                today = datetime.now(timezone.utc).date()
                this_month = today.strftime("%Y-%m")

                # Обнаружение смены настроек.
                # self.initial_capital уже содержит значение из конструктора
                # (новое allocated_capital_usdt). Если в файле лежит другое
                # значение — пользователь поменял долю капитала, размер
                # позиции или иной параметр, влияющий на аллокацию.
                stored_initial = symbol_state.get("initial_capital")
                stored_current = symbol_state.get("current_capital")

                settings_changed = (
                    stored_initial is not None
                    and abs(float(stored_initial) - float(self.initial_capital)) > 1e-6
                )

                if settings_changed and stored_current is not None:
                    # Корректируем current_capital на дельту аллокации.
                    # Пример: было 100 USDT базы, стало 200. Накопленный PnL
                    # (current - initial = +5) сохраняется, новый баланс = 205.
                    delta = float(self.initial_capital) - float(stored_initial)
                    self.current_capital = float(stored_current) + delta
                    logger.warning(
                        f"[{self.symbol}] Настройки изменились: "
                        f"база капитала {stored_initial:.2f} → "
                        f"{self.initial_capital:.2f} USDT (дельта {delta:+.2f}). "
                        f"Капитал пересчитан: {self.current_capital:.2f} USDT."
                    )
                    self._log_alloc_change_if_available(stored_initial)
                elif stored_current is not None:
                    self.current_capital = float(stored_current)

                # initial_capital уже актуален из конструктора.

                if saved_date == str(today):
                    self.daily_realized_pnl = symbol_state.get("daily_realized_pnl", 0.0)
                    self.daily_stop_triggered = symbol_state.get("daily_stop_triggered", False)

                if saved_month == this_month:
                    self.monthly_realized_pnl = symbol_state.get("monthly_realized_pnl", 0.0)
                    self.monthly_optimization_triggered = symbol_state.get(
                        "monthly_optimization_triggered", False
                    )

                logger.info(
                    f"[{self.symbol}] Состояние риск-менеджера загружено: "
                    f"текущий капитал {self.current_capital:.2f} USDT, "
                    f"день {self.daily_realized_pnl:+.2f} USDT, "
                    f"месяц {self.monthly_realized_pnl:+.2f} USDT"
                )

        except Exception as e:
            logger.error(f"Не удалось загрузить состояние риск-менеджера: {e}")

    def reset_monthly_optimization_flag(self):
        """
        Сбрасывает флаг месячной оптимизации после её успешного завершения.
        Позволяет оптимизации сработать снова в этом же месяце, если убыток продолжится.
        """
        self.monthly_optimization_triggered = False
        self._save_state()