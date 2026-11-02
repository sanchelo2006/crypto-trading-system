"""
Главный цикл автоматической торговли.

Управляет фоновыми потоками по каждой монете:
- мониторинг новых свечей (15m)
- детектирование сигналов по strategy_spec
- открытие позиций с контролем лимитов
- мониторинг открытых позиций и детектирование закрытий
- контроль дневных и месячных лимитов убытков
- запуск оптимизации стратегии при месячном лимите
- обработка ошибок сети и баланса

Лимиты:
- Максимум 1 одновременно открытая позиция на монету
- При дневном лимите убытка — закрыть позиции, остановить торговлю, ждать подтверждения
- При месячном лимите убытка — приостановить торговлю, запустить оптимизацию,
  после её завершения возобновить торговлю автоматически
"""
import math
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Callable, Dict, List, Optional

import config
from trading.executor import BybitExecutor
from trading.signal_detector import SignalDetector
from trading.position_manager import PositionManager
from trading.risk_manager import RiskManager
from trading.trade_logger import TradeLogger
from strategy_spec import evaluate_conditions
from ui.auto_mode import AutoModeState

logger = logging.getLogger(__name__)

# Глобальный реестр активных автотрейдеров по монетам
_active_traders: Dict[str, "AutoTraderSymbol"] = {}
_active_traders_lock = threading.Lock()

MAX_POSITIONS_PER_SYMBOL = 1
# Лимит подряд идущих НЕ-сетевых исключений (баги в коде и т.п.),
# после которого трейдер останавливается сам.
# Сетевые сбои (ConnectionError) НЕ учитываются в этом счётчике и
# никогда не останавливают торговлю автоматически.
NETWORK_RETRY_LIMIT = 60

# Дефолты для auto-sanitize strategy_spec.
# Применяются только если поле ОТСУТСТВУЕТ или равно None.
DEFAULT_TRAILING_ATR_MULT = 2.0        # ATR-множитель трейлинг-стопа
DEFAULT_MAX_HOLD_CANDLES = 48          # 48 × 15m = 12 часов

# Максимальный разрыв между сохранением и загрузкой risk_controls.
# Если приложение было остановлено дольше этого времени, состояние
# сбрасывается: cooldown истёк бы сам, а «серия убытков» недельной
# давности — уже не серия. 24 часа = 96 свечей 15m.
RISK_CONTROLS_MAX_STALENESS_HOURS = 24

# Минимальный интервал между полными перезагрузками истории свечей.
# Цикл автоторговли крутится чаще (раз в 15 сек) ради быстрого мониторинга
# позиций, но полная выгрузка 15m+1h истории на каждой итерации упирается
# в rate limit Bybit. Данные обновляются по свечам (15m), поэтому
# перезагружать их чаще, чем раз в несколько минут, смысла нет.
DATA_FETCH_MIN_INTERVAL_SEC = 240  # 4 минуты


def is_auto_trading_running(symbol: Optional[str] = None) -> bool:
    """
    Проверяет, запущена ли автоторговля.
    
    Если передан символ — проверяет по конкретной монете.
    Иначе — запущена ли хотя бы одна.
    
    Используется для защиты от одновременного запуска с авторежимом.
    """
    with _active_traders_lock:
        if symbol is None:
            return any(t.is_running for t in _active_traders.values())
        
        trader = _active_traders.get(symbol)
        return trader is not None and trader.is_running


def get_all_states() -> Dict[str, Dict]:
    """Возвращает состояния всех автотрейдеров (для UI)."""
    with _active_traders_lock:
        return {
            symbol: trader.get_state()
            for symbol, trader in _active_traders.items()
        }

# Файл для автовосстановления трейдеров после перезапуска
_ACTIVE_TRADERS_FILE = config.DATA_DIR / "auto_trading" / "active_traders.json"


def _save_active_trader(symbol: str, cfg: AutoTraderConfig) -> None:
    """Сохраняет конфигурацию запущенного трейдера для автовосстановления."""
    try:
        _ACTIVE_TRADERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if _ACTIVE_TRADERS_FILE.exists():
            try:
                with open(_ACTIVE_TRADERS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = {}
        data[symbol] = {
            "symbol": cfg.symbol,
            "category": cfg.category,
            "capital_share_pct": cfg.capital_share_pct,
            "position_size_pct": cfg.position_size_pct,
            "leverage": cfg.leverage,
            "daily_loss_limit_pct": cfg.daily_loss_limit_pct,
            "monthly_loss_limit_pct": cfg.monthly_loss_limit_pct,
            "optimization_days_back": cfg.optimization_days_back,
            "optimization_iterations": cfg.optimization_iterations,
            "walk_forward_folds": cfg.walk_forward_folds,
            "initial_capital": cfg.initial_capital,
            "commission_pct": cfg.commission_pct,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        tmp = _ACTIVE_TRADERS_FILE.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, _ACTIVE_TRADERS_FILE)
    except Exception:
        logger.exception(f"[{symbol}] Не удалось сохранить активного трейдера")


def _remove_active_trader(symbol: str) -> None:
    """Удаляет трейдера из файла автовосстановления."""
    try:
        if not _ACTIVE_TRADERS_FILE.exists():
            return
        with open(_ACTIVE_TRADERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if symbol not in data:
            return
        del data[symbol]
        tmp = _ACTIVE_TRADERS_FILE.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, _ACTIVE_TRADERS_FILE)
    except Exception:
        logger.exception(f"[{symbol}] Не удалось удалить трейдера из файла")


def _load_active_traders() -> Dict[str, dict]:
    """Загружает список трейдеров для автовосстановления."""
    if not _ACTIVE_TRADERS_FILE.exists():
        return {}
    try:
        with open(_ACTIVE_TRADERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning("Не удалось прочитать файл активных трейдеров")
        return {}


class AutoTraderConfig:
    """Конфигурация автоторговли для одной монеты."""
    
    def __init__(
        self,
        symbol: str,
        category: str = "linear",
        capital_share_pct: float = 30.0,
        position_size_pct: float = 50.0,
        leverage: int = 10,
        daily_loss_limit_pct: float = 1.0,
        monthly_loss_limit_pct: float = 15.0,
        optimization_days_back: int = 90,
        optimization_iterations: int = 10,
        walk_forward_folds: int = 3,
        initial_capital: float = 1000.0,
        commission_pct: float = 0.05,
    ):
        self.symbol = symbol
        self.category = category
        self.capital_share_pct = capital_share_pct
        self.position_size_pct = position_size_pct
        self.leverage = leverage
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.monthly_loss_limit_pct = monthly_loss_limit_pct
        self.optimization_days_back = optimization_days_back
        self.optimization_iterations = optimization_iterations
        self.walk_forward_folds = walk_forward_folds
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct


class AutoTraderSymbol:
    """
    Автотрейдер для одной монеты. Работает в собственном фоновом потоке.
    """
    
    def __init__(
        self,
        cfg: AutoTraderConfig,
        *,
        executor: BybitExecutor,
        fetcher,
        memory,
        backtest_engine,
        backtest_optimizer,
        ai_client,
        notify_fn: Callable[[str, str], None],
        total_capital_usdt: float,
    ):
        self.cfg = cfg
        self.executor = executor
        self.fetcher = fetcher
        self.memory = memory
        self.backtest_engine = backtest_engine
        self.backtest_optimizer = backtest_optimizer
        self.ai_client = ai_client
        self.notify_fn = notify_fn
        self.total_capital = total_capital_usdt

        self.symbol = cfg.symbol

        # Выделенный капитал монеты
        self.allocated_capital = total_capital_usdt * (cfg.capital_share_pct / 100.0)

        # Taker-ставка (в процентах). Дефолт — стандартная для Bybit
        # linear perpetuals, перезаписывается после get_fee_rates().
        self._taker_fee_pct: float = 0.055

        # Компоненты
        self.signal_detector = SignalDetector()
        self.position_manager = PositionManager(
            executor, self.symbol, cfg.category,
            taker_fee_pct=self._taker_fee_pct,
        )
        self.risk_manager = RiskManager(
            symbol=self.symbol,
            daily_loss_limit_pct=cfg.daily_loss_limit_pct,
            monthly_loss_limit_pct=cfg.monthly_loss_limit_pct,
            allocated_capital_usdt=self.allocated_capital,
        )
        self.trade_logger = TradeLogger(self.symbol)

        # Состояние
        self.is_running = False
        self.is_optimizing = False
        self.consecutive_network_errors = 0
        self._non_network_errors = 0
        self.last_processed_candle_time = None
        self.status_message = "Инициализация..."
        self.log_lines: List[str] = []
        self._thread: Optional[threading.Thread] = None

        # Идентификаторы уже зачтённых записей Closed-PnL
        self._consumed_closed_pnl_ids: set = set()

        # --- risk_controls (пауза после серий убытков) ---
        self._consecutive_losses = 0
        self._recent_loss_flags: List[int] = []
        self._cooldown_remaining = 0  # сколько свечей ещё пауза
        
        # --- HEARTBEAT для UI ---
        self.last_iteration_utc: Optional[datetime] = None
        self.last_candle_time = None
        self.last_price: Optional[float] = None
        self.last_indicators_summary = ""
        self.candles_processed = 0

        # Кэш последних успешно загруженных индикаторов
        self._cached_indicators: Optional[Dict] = None
        self._cached_prev_indicators: Optional[Dict] = None
        self._cached_candle_time = None
        self._cached_at_utc: Optional[datetime] = None

        # Время последней сетевой выгрузки истории (для throttle)
        self._last_data_fetch_utc: Optional[datetime] = None

        # --- Персистентное состояние позиций ---
        # Файл переживает перезапуск приложения: на старте мы восстанавливаем
        # список открытых позиций, чтобы отличить «наши» позиции от внешних
        # (orphan), а также чтобы после сбоя связи догнать закрытия.
        self._state_dir = config.DATA_DIR / "auto_trading" / "positions"
        self._state_path = self._state_dir / f"{self.symbol}.json"

        # --- Отслеживание лимитных ордеров ---
        # Храним активные лимитные ордера для мониторинга филла.
        # Формат: [{ "order_id", "direction", "qty", "price", "side_spec",
        #            "reasons", "indicators", "candle_ts", "placed_at" }]
        self._pending_limit_orders: List[Dict] = []

        # Загружаем сохранённое состояние ДО первого _loop_iteration.
        self._load_state()
    
    # ----------------------------------------------------------------
    # Управление жизненным циклом
    # ----------------------------------------------------------------
    
    def start(self):
        """
        Запускает фоновый поток торговли.

        Returns:
            True — трейдер реально запущен.
            False — старт отклонён (уже запущен / авторежим / настройки
                    не позволяют открыть позицию).
        """
        if self.is_running:
            return False

        # Защита от одновременного запуска с авторежимом
        auto_state = AutoModeState.get_state(self.symbol)
        if auto_state and auto_state.get("is_running"):
            self.notify_fn(
                f"[{self.symbol}] Невозможно запустить автоторговлю: "
                "уже включён авторежим.",
                "warning",
            )
            return False

        # Санитизация strategy_spec: заполняем недостающие поля
        # в management безопасными дефолтами (max_hold_candles, а при
        # отсутствии любого сопровождения — и trailing_atr_mult).
        # Делается ДО запуска цикла, чтобы живая торговля сразу
        # работала с полной спекой.
        try:
            self._sanitize_strategy_spec()
        except Exception:
            logger.exception(
                f"[{self.symbol}] Ошибка санитизации strategy_spec"
            )
            # Не блокируем запуск из-за неудачи санитизации.
        
        # Пред-проверка настроек: пробуем ли мы вообще открыть сделку,
        # которая пройдёт по минимуму биржи. Делаем это ДО запуска цикла.
        # Если настройки не позволяют открыть позицию — отказываемся
        # стартовать, а не крутим цикл впустую (см. SOLUSDT в логах).
        try:
            ok, price = self.executor.get_ticker_price(self.symbol)
            if ok and price:
                valid, warning = self.executor.validate_symbol_settings(
                    symbol=self.symbol,
                    capital_usdt=self.allocated_capital,
                    leverage=self.cfg.leverage,
                    position_size_pct=self.cfg.position_size_pct,
                    entry_price=price,
                )
                if not valid:
                    msg = warning or "Настройки не позволяют открыть позицию."
                    self._log_event("CONFIG_REJECT", msg)
                    self._notify(
                        f"[{self.symbol}] Автоторговля НЕ запущена: {msg}",
                        "negative",
                    )
                    self.status_message = f"Отказ старта: {msg}"
                    return False
        except Exception:
            logger.exception(f"[{self.symbol}] Ошибка пред-проверки настроек")

        self.is_running = True
        # Сохраняем для автовосстановления после перезапуска
        _save_active_trader(self.symbol, self.cfg)
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        self._log_event("START", f"Автоторговля запущена для {self.symbol}")
        self._notify(f"Автоторговля {self.symbol} запущена", "positive")
        return True
    
    def stop(self, close_positions: bool = False, reason: str = "manual"):
        """Останавливает торговлю."""
        self.is_running = False
        self.status_message = f"Остановлено ({reason})"
        
        if close_positions:
            self._close_all_positions(reason=f"stop_{reason}")
        
        self._log_event("STOP", f"Автоторговля остановлена: {reason}")
        self._notify(f"Автоторговля {self.symbol} остановлена", "warning")
        
        with _active_traders_lock:
            _active_traders.pop(self.symbol, None)
        
        # Удаляем из файла автовосстановления
        _remove_active_trader(self.symbol)
    
    # ----------------------------------------------------------------
    # Главный цикл
    # ----------------------------------------------------------------
    
    def _run_loop(self):
        """Основной фоновый цикл торговли."""
        logger.info(f"[{self.symbol}] Поток автоторговли запущен")

        # Инициализация плеча и режима маржи
        if not self._setup_leverage():
            self.stop(reason="leverage_setup_failed")
            return

        while self.is_running:
            try:
                self._loop_iteration()

            except ConnectionError as e:
                # СЕТЕВОЙ СБОЙ. НИКОГДА не останавливаем торговлю
                # автоматически — только сам пользователь может это сделать
                # кнопкой. Ждём восстановления связи и пробуем снова.
                #
                # Экспоненциальный backoff с потолком 300 сек:
                # 15 → 30 → 60 → 120 → 240 → 300 → 300 → ...
                # Так не бомбим биржу запросами при долгом сбое.
                self.consecutive_network_errors += 1
                # Сетевой сбой — не признак бага в коде. Сбрасываем
                # счётчик «прочих» ошибок, чтобы трейдер не остановился
                # из-за чередования: одна кодовая ошибка → пропала сеть →
                # ещё ошибка → ...
                self._non_network_errors = 0
                self._log_event(
                    "NETWORK",
                    f"Ошибка сети #{self.consecutive_network_errors}: {e}. "
                    "Торговля продолжается, ждём восстановления связи.",
                )
                backoff = min(
                    15 * (2 ** min(self.consecutive_network_errors - 1, 4)),
                    300,
                )
                time.sleep(backoff)
                continue

            except Exception as e:
                # ПРОЧИЕ ОШИБКИ (баги в коде, неожиданные состояния).
                # Считаем отдельным счётчиком и останавливаемся только
                # при исчерпании лимита, чтобы не зациклиться на битом коде.
                logger.exception(f"[{self.symbol}] Ошибка в цикле автоторговли")
                self._non_network_errors += 1
                self._log_event(
                    "ERROR",
                    f"Ошибка цикла ({self._non_network_errors}/"
                    f"{NETWORK_RETRY_LIMIT}): {e}",
                )

                if self._non_network_errors >= NETWORK_RETRY_LIMIT:
                    self._notify(
                        f"[{self.symbol}] {NETWORK_RETRY_LIMIT} не-сетевых "
                        "ошибок подряд. Торговля остановлена.",
                        "negative",
                    )
                    self.stop(close_positions=False, reason="repeated_errors")
                    return

                time.sleep(60)
                continue

            else:
                # Успешная итерация — сбрасываем оба счётчика. После
                # длительного сбоя связи следующая успешная итерация
                # вызовет _monitor_positions → _detect_closed_positions,
                # и все сделки, закрытые пока не было сети, будут
                # залогированы (Closed-PnL хранится на бирже до 7 дней).
                self.consecutive_network_errors = 0
                self._non_network_errors = 0

            time.sleep(15)  # Пауза между итерациями

        logger.info(f"[{self.symbol}] Поток автоторговли завершён")
    
    def _setup_leverage(self) -> bool:
        """Устанавливает плечо и режим маржи для монеты."""
        self.status_message = "Настройка плеча..."
        
        success, err = self.executor.set_leverage(self.symbol, self.cfg.leverage)
        if not success:
            self._log_event("ERROR", f"Не удалось установить плечо: {err}")
            self._notify(f"[{self.symbol}] Не удалось установить плечо", "negative")
            return False

        # Taker-ставка комиссии — нужна и для вычитания при закрытии,
        # и для оптимизации (backtest должен считать с той же ставкой).
        try:
            ok, rates = self.executor.get_fee_rates(self.symbol)
            if ok and rates:
                self._taker_fee_pct = float(rates["taker_pct"])
                self.position_manager.set_taker_fee_pct(self._taker_fee_pct)
                self._log_event(
                    "FEE_RATE",
                    f"Taker-комиссия {self.symbol}: "
                    f"{self._taker_fee_pct:.4f}%",
                )
            else:
                self._log_event(
                    "FEE_RATE",
                    f"Не удалось получить taker-ставку ({rates}). "
                    f"Использую дефолт {self._taker_fee_pct:.4f}%",
                )
        except Exception:
            logger.exception(
                f"[{self.symbol}] Ошибка получения taker-ставки, "
                f"использую {self._taker_fee_pct:.4f}%"
            )

        return True

    def _sanitize_strategy_spec(self) -> bool:
        """
        Дополняет strategy_spec безопасными дефолтами перед запуском торговли.

        Что чиним (только если поле ОТСУТСТВУЕТ или None):
          - management.max_hold_candles → DEFAULT_MAX_HOLD_CANDLES
            Защита от зависших сделок. Не зависит от монеты,
            зависит от таймфрейма (15m × 48 = 12 часов).

          - management.trailing_atr_mult → DEFAULT_TRAILING_ATR_MULT
            ТОЛЬКО если в management нет ни одного другого механизма
            сопровождения (breakeven / exit_on_*). Если модель осознанно
            выбрала move_to_breakeven_at_tp_pct (случай HYPEUSDT) —
            trailing не навязываем, это её решение.

        НЕ перезаписывает существующие значения. НЕ трогает другие поля.

        Returns:
            True, если spec был изменён и сохранён.
            False, если изменений не потребовалось.
        """
        try:
            cm = self.memory.get(self.symbol)
            state = cm.get_state() or {}
            spec = state.get("strategy_spec")

            if not isinstance(spec, dict):
                return False

            changed_fields: List[str] = []

            for side_key in ("long", "short"):
                side = spec.get(side_key)
                if not isinstance(side, dict):
                    continue

                mgmt = side.get("management")
                if not isinstance(mgmt, dict):
                    mgmt = {}
                    side["management"] = mgmt

                # --- 1. max_hold_candles: добавляем ВСЕГДА, если не задано ---
                # Это защита от зависших сделок, не стратегия.
                # Живёт вне зависимости от монеты.
                #
                # ВАЖНО: запоминаем, был ли тайм-стоп задан ОСОЗНАННО
                # (до санитизации). Если стратегия сама установила
                # max_hold_candles — это уже механизм выхода, и навязывать
                # трейлинг поверх него нельзя (сломает свинг-стратегии,
                # где трейлинг убивает сделки на откатах).
                had_explicit_max_hold = mgmt.get("max_hold_candles") is not None

                if mgmt.get("max_hold_candles") is None:
                    mgmt["max_hold_candles"] = DEFAULT_MAX_HOLD_CANDLES
                    changed_fields.append(
                        f"{side_key}.management.max_hold_candles"
                    )

                # --- 2. trailing_atr_mult: только если нет другого ---
                #     механизма сопровождения ---
                # Осознанно заданный max_hold_candles считается механизмом
                # выхода — в этом случае трейлинг не навязываем.
                has_any_management = had_explicit_max_hold or any(
                    mgmt.get(key) not in (None, False, 0)
                    for key in (
                        "move_to_breakeven_at_tp_pct",
                        "trailing_atr_mult",
                        "breakeven_offset_pct",
                        "exit_on_level_failure",
                        "exit_on_trend_loss",
                        "exit_on_macd_flip",
                    )
                )

                if (
                    not has_any_management
                    and mgmt.get("trailing_atr_mult") is None
                ):
                    mgmt["trailing_atr_mult"] = DEFAULT_TRAILING_ATR_MULT
                    changed_fields.append(
                        f"{side_key}.management.trailing_atr_mult"
                    )

                if (
                    not has_any_management
                    and mgmt.get("trailing_atr_mult") is None
                ):
                    mgmt["trailing_atr_mult"] = DEFAULT_TRAILING_ATR_MULT
                    changed_fields.append(
                        f"{side_key}.management.trailing_atr_mult"
                    )

            if not changed_fields:
                return False

            cm.update_state({"strategy_spec": spec})
            cm.save()

            logger.info(
                f"[{self.symbol}] strategy_spec дополнен: "
                f"{', '.join(changed_fields)}"
            )
            self._log_event(
                "SPEC_SANITIZED",
                f"strategy_spec дополнен дефолтами: "
                f"{', '.join(changed_fields)}",
            )
            self._notify(
                f"[{self.symbol}] В strategy_spec добавлены дефолты: "
                f"{', '.join(changed_fields)}",
                "warning",
            )
            return True

        except Exception:
            logger.exception(
                f"[{self.symbol}] Ошибка санитизации strategy_spec"
            )
            return False
    
    def _loop_iteration(self):
        """Одна итерация основного цикла."""
        if self.is_optimizing:
            self.status_message = "Оптимизация стратегии..."
            self._monitor_positions()
            return

        self.status_message = "Мониторинг рынка..."

        # 1. Загружаем свежие свечи и индикаторы — НЕ ЧАЩЕ, ЧЕМ РАЗ В N СЕКУНД.
        #    Между загрузками используем кэш. Это критично: полная выгрузка
        #    истории = ~10-12 запросов, и если делать её каждые 15 сек,
        #    Bybit отвечает 10006 (rate limit).
        now_utc = datetime.now(timezone.utc)
        should_fetch = (
            self._last_data_fetch_utc is None
            or (now_utc - self._last_data_fetch_utc).total_seconds()
               >= DATA_FETCH_MIN_INTERVAL_SEC
        )

        if should_fetch:
            indicators, candle_time, prev_indicators = self._fetch_indicators()
            self._last_data_fetch_utc = now_utc

            if indicators is None:
                # Пытаемся обойтись кэшем — трейдер продолжает мониторить
                # уже открытые позиции, но новые сигналы не генерирует.
                cache_age = None
                if self._cached_at_utc is not None:
                    cache_age = (now_utc - self._cached_at_utc).total_seconds()

                if self._cached_indicators is None or cache_age is None or cache_age > 1800:
                    self.consecutive_network_errors += 1
                    self._log_event(
                        "NETWORK",
                        f"Индикаторы недоступны, кэш отсутствует или устарел. "
                        f"Ошибка {self.consecutive_network_errors}/{NETWORK_RETRY_LIMIT}",
                    )
                    raise ConnectionError("Не удалось загрузить индикаторы")

                self.consecutive_network_errors += 1
                self._log_event(
                    "NETWORK",
                    f"Индикаторы недоступны, использую кэш ({int(cache_age)}с). "
                    f"Ошибка {self.consecutive_network_errors}/{NETWORK_RETRY_LIMIT}",
                )
                indicators = self._cached_indicators
                prev_indicators = self._cached_prev_indicators
                candle_time = self._cached_candle_time
            else:
                # Сброс счётчика при успешной загрузке
                self.consecutive_network_errors = 0
                self._cached_indicators = indicators
                self._cached_prev_indicators = prev_indicators
                self._cached_candle_time = candle_time
                self._cached_at_utc = now_utc
        else:
            # Отдаём кэш без сетевого запроса.
            indicators = self._cached_indicators
            prev_indicators = self._cached_prev_indicators
            candle_time = self._cached_candle_time

            if indicators is None:
                # Первой загрузки ещё не было — пропускаем итерацию.
                # Это может быть только на самом старте.
                time.sleep(5)
                return
        
        # --- HEARTBEAT ---
        self.last_iteration_utc = datetime.now(timezone.utc)
        self.last_candle_time = candle_time
        self.last_price = indicators.get("close")
        
        parts = []
        for name in ("rsi_14", "adx_14", "atr_14"):
            v = indicators.get(name)
            if v is not None:
                try:
                    parts.append(f"{name}={float(v):.2f}")
                except (TypeError, ValueError):
                    pass
        self.last_indicators_summary = ", ".join(parts)

        # 2. Мониторим открытые позиции (сопровождение + детект закрытий)
        self._monitor_positions(indicators=indicators)

        # 2.5. Мониторим лимитные ордера
        self._monitor_pending_limit_orders()    

        # 3. Проверяем, не наступил ли новый день/месяц
        self.risk_manager._check_rollover()
        
        # 4. ПРОВЕРЯЕМ РИСК-ЛИМИТЫ ВСЕГДА (даже если позиций нет)
        unrealized = (
            self._get_unrealized_pnl()
            if self.position_manager.positions else 0.0
        )
        
        if self.risk_manager.should_stop_daily(unrealized):
            self._handle_daily_stop()
            return

        # 5. Проверяем месячный лимит (запуск оптимизации)
        if self.risk_manager.should_optimize_monthly():
            self._trigger_optimization()
            return

        # 6. Обрабатываем сигналы только на новой закрытой свече.
        #    CANDLE-события больше не пишем в events: они дублировали
        #    SIGNAL_CHECK на каждой свече и засоряли лог.
        if candle_time != self.last_processed_candle_time:
            self.last_processed_candle_time = candle_time
            self.candles_processed += 1
            self._process_signals(indicators, prev_indicators)
    
    # ----------------------------------------------------------------
    # Индикаторы
    # ----------------------------------------------------------------
    
    def _fetch_indicators(self):
        """
        Загружает свежие свечи и рассчитывает индикаторы.

        Возвращает:
            (текущие индикаторы, время последней свечи, предыдущие индикаторы)
            или (None, None, None) при ошибке

        Грузим только 15m-историю за 5 дней:
          - индикаторам нужно максимум ~100 свечей (SMA50, avg_volume_96);
          - daily-пивоты требуют минимум 2-3 полных дня;
          - 1h-признаки считаются ресэмплингом внутри _build_indicator_frame,
            отдельная выгрузка 1h НЕ НУЖНА и только съедает rate limit.
        """
        try:
            now = datetime.now(timezone.utc)
            start_date = now - timedelta(days=14)

            prices_15m = self.fetcher.fetch_ohlcv(
                symbol=self.symbol,
                interval="15",
                start_date=start_date,
                end_date=now,
                category=self.cfg.category,
            )

            if prices_15m is None or prices_15m.empty:
                return None, None, None

            df = self.backtest_engine._build_indicator_frame(prices_15m)


            if len(df) < 2:
                return None, None, None

            # Последняя свеча может быть незакрытой — используем предпоследнюю
            records = df.to_dict("records")
            current = records[-2]
            prev = records[-3] if len(records) >= 3 else {}

            candle_time = current.get("timestamp")

            return current, candle_time, prev

        except Exception as e:
            logger.exception(f"[{self.symbol}] Ошибка загрузки индикаторов")
            raise ConnectionError(f"Ошибка загрузки индикаторов: {e}")
    
    # ----------------------------------------------------------------
    # Сигналы и открытие позиций
    # ----------------------------------------------------------------
    
    def _process_signals(self, indicators: Dict, prev_indicators: Dict):
        """Обрабатывает торговые сигналы."""
        cm = self.memory.get(self.symbol)
        strategy_spec = cm.get_state().get("strategy_spec")

        if not strategy_spec:
            self._log_event(
                "ERROR",
                "Нет strategy_spec для монеты. Торговля невозможна.",
            )
            self.stop(reason="no_strategy")
            return

        # --- Проверка cooldown после серии убытков ---
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            self._log_event(
                "COOLDOWN",
                f"Пауза после серии убытков. Осталось свечей: {self._cooldown_remaining}",
            )
            # Персистим уменьшенное значение, чтобы рестарт
            # не откатил cooldown назад.
            self._save_state()
            return


        # --- ДИАГНОСТИКА УСЛОВИЙ ВХОДА ---
        # Логируем всё одной группой (привязанной к свече): глобальные
        # фильтры + LONG + SHORT. Это склеивает все сигнальные события
        # одной итерации в один блок в HTML-логе.
        candle_group = (
            str(self.last_candle_time)
            if self.last_candle_time is not None
            else None
        )
        self._log_signal_diagnostics(
            strategy_spec, indicators, prev_indicators, group=candle_group,
        )

        signals = self.signal_detector.detect_signals(
            strategy_spec=strategy_spec,
            current_indicators=indicators,
            prev_indicators=prev_indicators,
        )


        
        if not signals:
            return
        
        # Проверяем лимит открытых позиций
        open_count = len(self.position_manager.positions)
        available_slots = MAX_POSITIONS_PER_SYMBOL - open_count
        
        if available_slots <= 0:
            self._log_event(
                "OPEN",
                f"Сигнал есть, но лимит позиций ({MAX_POSITIONS_PER_SYMBOL}) достигнут. Пропуск.",
            )
            return
        
        for signal in signals[:available_slots]:
            self._open_position_from_signal(signal, indicators)

    def _get_risk_controls(self) -> Dict:
        """
        Читает risk_controls из strategy_spec.
        Нормализует значения так же, как backtest_engine._get_risk_controls.
        """
        try:
            cm = self.memory.get(self.symbol)
            spec = cm.get_state().get("strategy_spec") or {}
        except Exception:
            return {"enabled": False}

        raw = spec.get("risk_controls") or {}
        if not bool(raw.get("enabled", False)):
            return {"enabled": False}

        return {
            "enabled": True,
            "max_consecutive_losses": max(1, int(raw.get("max_consecutive_losses", 3))),
            "pause_candles_after_losses": max(1, int(raw.get("pause_candles_after_losses", 96))),
            "max_losses_in_window": max(1, int(raw.get("max_losses_in_window", 5))),
            "loss_window_trades": max(1, int(raw.get("loss_window_trades", 10))),
        }


    def _should_pause_after_loss(self, result_pct: float, controls: Dict) -> bool:
        """
        Решает, нужно ли активировать паузу после закрытия убыточной сделки.
        Логика идентична backtest_engine._should_pause_after_loss.
        """
        if not controls.get("enabled"):
            return False
        if result_pct > 0:
            return False

        if self._consecutive_losses >= controls["max_consecutive_losses"]:
            return True

        window = controls["loss_window_trades"]
        max_losses = controls["max_losses_in_window"]
        last_window = self._recent_loss_flags[-window:]
        if len(last_window) >= window and sum(last_window) >= max_losses:
            return True

        return False

    def _log_signal_diagnostics(
        self,
        strategy_spec: Dict,
        indicators: Dict,
        prev: Dict,
        group: Optional[str] = None,
    ):
        """
        Логирует проверку условий входа для LONG и SHORT.

        Поддерживает обе формы спецификации:
          - "entry": [cond, ...] — одна группа, все условия по AND;
          - "entry_any": [[cond, ...], [cond, ...]] — OR-группы.

        Args:
            group: ключ группы для HTML-разделителя. Обычно — timestamp
                свечи, чтобы FILTERS + LONG + SHORT склеились в один
                визуальный блок.
        """
        from strategy_spec import evaluate_conditions

        def _check_and_format(conditions: List, label: str) -> str:
            if not conditions:
                return f"{label}: нет условий"

            passed = []
            failed = []

            for cond in conditions:
                if not isinstance(cond, dict):
                    continue

                result = evaluate_conditions([cond], indicators, prev)
                cond_type = cond.get("type", "unknown")

                if cond_type == "compare":
                    left_key = cond.get("left")
                    op = cond.get("op", "?")
                    right = cond.get("right")
                    left_val = indicators.get(left_key, "N/A")

                    if isinstance(right, dict):
                        right_key = right.get("name")
                        mult = right.get("mult", 1.0)
                        right_val = indicators.get(right_key, "N/A")
                        if right_val != "N/A":
                            try:
                                right_val = float(right_val) * float(mult)
                            except Exception:
                                pass
                        desc = (
                            f"{left_key}({left_val}) {op} "
                            f"{right_key}*{mult}({right_val})"
                        )
                    else:
                        desc = f"{left_key}({left_val}) {op} {right}"

                    (passed if result else failed).append(
                        ("✓ " if result else "✗ ") + desc
                    )

                elif cond_type == "range":
                    value_key = cond.get("value")
                    min_val = cond.get("min")
                    max_val = cond.get("max")
                    actual = indicators.get(value_key, "N/A")
                    desc = f"{value_key}({actual}) in [{min_val}, {max_val}]"
                    (passed if result else failed).append(
                        ("✓ " if result else "✗ ") + desc
                    )

                elif cond_type == "volume":
                    ref = cond.get("ref")
                    mult = cond.get("mult", 1.0)
                    vol = indicators.get("volume", "N/A")
                    ref_vol = indicators.get(ref, "N/A")
                    desc = f"volume({vol}) > {ref}({ref_vol})*{mult}"
                    (passed if result else failed).append(
                        ("✓ " if result else "✗ ") + desc
                    )

                elif cond_type == "slope":
                    value_key = cond.get("value")
                    direction = cond.get("dir")
                    desc = f"slope({value_key}) → {direction}"
                    (passed if result else failed).append(
                        ("✓ " if result else "✗ ") + desc
                    )

                else:
                    desc = f"{cond_type}"
                    (passed if result else failed).append(
                        ("✓ " if result else "✗ ") + desc
                    )

            summary = f"{label}: {len(passed)}/{len(conditions)} условий"
            if passed:
                summary += f"\n  PASSED: {'; '.join(passed[:3])}"
                if len(passed) > 3:
                    summary += f" ... и ещё {len(passed) - 3}"
            if failed:
                summary += f"\n  FAILED: {'; '.join(failed[:3])}"
                if len(failed) > 3:
                    summary += f" ... и ещё {len(failed) - 3}"

            return summary

        # --- Глобальные фильтры ---
        filters = strategy_spec.get("filters") or []
        if filters:
            filters_result = evaluate_conditions(filters, indicators, prev)
            filters_summary = _check_and_format(filters, "FILTERS")
            self._log_event(
                "SIGNAL_CHECK",
                f"Глобальные фильтры: "
                f"{'PASSED' if filters_result else 'FAILED'}\n"
                f"{filters_summary}",
                group=group,
            )
            if not filters_result:
                return

        # --- LONG и SHORT ---
        for side in ("long", "short"):
            side_spec = strategy_spec.get(side) or {}
            label = side.upper()

            groups = side_spec.get("entry_any")
            if isinstance(groups, list) and groups:
                for gidx, grp in enumerate(groups, start=1):
                    if not isinstance(grp, list) or not grp:
                        continue
                    grp_label = f"{label} entry_any [{gidx}/{len(groups)}]"
                    grp_summary = _check_and_format(grp, grp_label)
                    grp_passed = evaluate_conditions(grp, indicators, prev)
                    self._log_event(
                        "SIGNAL_CHECK",
                        f"{label}: {'SIGNAL' if grp_passed else 'NO SIGNAL'}\n"
                        f"{grp_summary}",
                        group=group,
                    )
                continue

            entry = side_spec.get("entry") or []
            if entry:
                entry_summary = _check_and_format(entry, f"{label} ENTRY")
                entry_passed = evaluate_conditions(entry, indicators, prev)
                self._log_event(
                    "SIGNAL_CHECK",
                    f"{label}: {'SIGNAL' if entry_passed else 'NO SIGNAL'}\n"
                    f"{entry_summary}",
                    group=group,
                )
    
    def _open_position_from_signal(self, signal: Dict, indicators: Dict):
        """Открывает позицию по сигналу."""
        direction = signal["direction"]
        side_spec = signal["side_spec"]
        reasons = signal["reasons"]
        
        # Получаем текущую цену
        success, current_price = self.executor.get_ticker_price(self.symbol)
        if not success or current_price is None:
            self._log_event("ERROR", f"Не удалось получить цену для открытия: {current_price}")
            return
        
        # Рассчитываем размер позиции
        position_value = self.allocated_capital * (self.cfg.position_size_pct / 100.0)
        
        success, qty = self.executor.calculate_position_size(
            symbol=self.symbol,
            capital_usdt=position_value,
            leverage=self.cfg.leverage,
            entry_price=current_price,
        )

        if not success or qty is None or qty <= 0:
            # qty-ошибка от calculate_position_size может содержать конкретную
            # причину (например, notional < min). Пробрасываем её в лог и UI.
            detail = qty if isinstance(qty, str) else "Не удалось рассчитать размер позиции"
            self._log_event("ORDER_TOO_SMALL", detail)
            self._notify(
                f"[{self.symbol}] Ордер не пройдёт: {detail}",
                "warning",
            )
            return
        
        # Определяем режим входа: лимитный (мейкер) или рыночный (тейкер)
        use_limit_entry = self._should_use_limit_entry(indicators)

        if use_limit_entry:
            # ЛИМИТНЫЙ ВХОД (мейкер, низкая комиссия)
            success, order_id_or_err, limit_price = (
                self.position_manager.open_position_limit(
                    direction=direction,
                    qty=qty,
                    side_spec=side_spec,
                    entry_reasons=reasons,
                    indicators=indicators,
                    entry_candle_ts=self.last_candle_time,
                    price_offset_ticks=0,
                    timeout_seconds=900,  # 15 минут
                )
            )
            if success:
                # Сохраняем ордер для мониторинга
                self._pending_limit_orders.append({
                    "order_id": order_id_or_err,
                    "direction": direction,
                    "qty": qty,
                    "price": float(limit_price),
                    "side_spec": side_spec,
                    "reasons": reasons,
                    "indicators": dict(indicators),
                    "candle_ts": self.last_candle_time,
                    "placed_at": datetime.now(timezone.utc),
                })
                candle_group = (
                    str(self.last_candle_time)
                    if self.last_candle_time is not None
                    else None
                )
                self._log_event(
                    "OPEN",
                    f"Лимитный ордер размещён: {direction.upper()} {qty} "
                    f"@ {limit_price} (Post-Only)",
                    group=candle_group,
                )
                self._save_state()
                return
            else:
                # Лимитный вход не удался — фоллбэк на рыночный
                self._log_event(
                    "WARNING",
                    f"Лимитный вход не удался ({order_id_or_err}). "
                    f"Переключаюсь на рыночный.",
                )
                success, current_price = self.executor.get_ticker_price(self.symbol)
                if not success or current_price is None:
                    self._log_event("ERROR", f"Не удалось получить цену: {current_price}")
                    return

        # РЫНОЧНЫЙ ВХОД (тейкер, как раньше)
        success, err = self.position_manager.open_position(
            direction=direction,
            qty=qty,
            current_price=current_price,
            side_spec=side_spec,
            entry_reasons=reasons,
            indicators=indicators,
            entry_candle_ts=self.last_candle_time,
        )
        
        if success:
            # Сохраняем состояние сразу после успешного открытия — чтобы
            # при рестарте позиция была в файле и не считалась orphan.
            self._save_state()

            candle_group = (
                str(self.last_candle_time)
                if self.last_candle_time is not None
                else None
            )
            self._log_event(
                "OPEN",
                f"Позиция открыта: {direction.upper()} {qty} @ {current_price:.4f}",
                group=candle_group,
            )

            # Проверяем, удалось ли выставить защиту.
            # PositionManager кладёт ошибку в поле tpsl_error последней позиции.
            last_pos = (
                self.position_manager.positions[-1]
                if self.position_manager.positions else None
            )
            tpsl_error = last_pos.get("tpsl_error") if last_pos else None

            if tpsl_error:
                self._log_event(
                    "TPSL_FAILED",
                    f"TP/SL НЕ установлены для {direction.upper()} позиции: {tpsl_error}. "
                    f"Позиция открыта БЕЗ защиты.",
                )
                self._notify(
                    f"[{self.symbol}] ВНИМАНИЕ: {direction.upper()} открыта без TP/SL. "
                    f"Проверьте позицию вручную.",
                    "negative",
                )
            else:
                self._notify(
                    f"[{self.symbol}] Открыта {direction.upper()} позиция @ {current_price:.4f}",
                    "positive",
                )
        else:
            self._log_event("ERROR", f"Не удалось открыть позицию: {err}")
            
            # Если причина — недостаток средств, останавливаем торговлю
            if err and ("insufficient" in str(err).lower() or "balance" in str(err).lower()):
                self._notify(
                    f"[{self.symbol}] Недостаточно средств. Торговля остановлена.",
                    "negative",
                )
                self._log_event("INSUFFICIENT_BALANCE", "Недостаточно средств для открытия позиции")
                self.stop(close_positions=False, reason="insufficient_balance")

    def _should_use_limit_entry(self, indicators: Dict) -> bool:
        """
        Определяет, использовать ли лимитный вход (мейкер).

        Лимитный вход используется когда:
        1. Стратегия не требует немедленного входа (свинг, не скальпинг).
        2. Волатильность не экстремальная (иначе ордер не исполнится).

        Для свинг-стратегий на 4H — всегда лимитный.
        Для скальпинга на 15m — рыночный.
        """
        # Если в спеке указан таймфрейм 4h или выше — лимитный вход
        # Определяем по величине ATR: на 4H ATR значительно больше
        atr = indicators.get("atr_14")
        close = indicators.get("close")
        if atr and close and close > 0:
            atr_pct = (atr / close) * 100
            # На 15m ATR обычно < 0.5%, на 4H > 1%
            if atr_pct > 0.8:
                return True

        # По умолчанию — рыночный вход (безопаснее)
        return False
    
    # ----------------------------------------------------------------
    # Мониторинг позиций
    # ----------------------------------------------------------------
    
    def _monitor_positions(self, indicators: Optional[Dict] = None):
        """
        Мониторит открытые позиции:
        0) Синхронизация с биржей (подхватывает orphan-позиции после рестарта)
        1) Сопровождение (breakeven, trailing) — обновляет SL на бирже
        2) Детект закрытий
        """
        # 0) Всегда синхронизируем локальное состояние с биржей.
        #    Это нужно, чтобы UI показывал реальное число позиций даже
        #    после рестарта трейдера (когда локальный список пуст,
        #    но позиции на бирже остались).
        try:
            self.position_manager.sync_with_exchange()
        except Exception:
            logger.exception(f"[{self.symbol}] Ошибка sync_with_exchange")

        if not self.position_manager.positions:
            return

        # 1) Сопровождение открытых позиций по management-правилам
        if indicators is not None:
            self._apply_live_management(indicators)

        # 2) Синхронизация с биржей и детект закрытий
        self._detect_closed_positions()

    def _monitor_pending_limit_orders(self):
        """
        Проверяет статус лимитных ордеров:
        - Исполнен → завершаем открытие позиции
        - Таймаут → отменяем ордер, пропускаем сигнал
        """
        if not self._pending_limit_orders:
            return

        now = datetime.now(timezone.utc)
        still_pending = []

        for order_info in self._pending_limit_orders:
            order_id = order_info["order_id"]
            placed_at = order_info["placed_at"]
            timeout = 900  # 15 минут

            # Проверяем таймаут
            elapsed = (now - placed_at).total_seconds()
            if elapsed > timeout:
                self.executor.cancel_order(self.symbol, order_id)
                self._log_event(
                    "ORDER_TIMEOUT",
                    f"Лимитный ордер {order_id} не исполнен за {timeout}с. Отменён.",
                )
                continue

            # Проверяем статус ордера
            success, order_data = self.executor.get_order_info(self.symbol, order_id)
            if not success or order_data is None:
                still_pending.append(order_info)
                continue

            status = order_data.get("orderStatus", "")

            if status == "Filled":
                # Ордер исполнен → завершаем открытие позиции
                filled_qty = float(order_data.get("cumExecQty", 0) or 0)
                avg_price = float(order_data.get("avgPrice", 0) or 0)

                if filled_qty > 0 and avg_price > 0:
                    success, err = self.position_manager.finalize_limit_entry(
                        direction=order_info["direction"],
                        qty=filled_qty,
                        entry_price=avg_price,
                        side_spec=order_info["side_spec"],
                        entry_reasons=order_info["reasons"],
                        indicators=order_info["indicators"],
                        entry_candle_ts=order_info["candle_ts"],
                        order_id=order_id,
                    )
                    if success:
                        self._save_state()
                        self._log_event(
                            "OPEN",
                            f"Лимитная позиция открыта: "
                            f"{order_info['direction'].upper()} {filled_qty} "
                            f"@ {avg_price:.4f}",
                        )
                        self._notify(
                            f"[{self.symbol}] Лимитный вход исполнен "
                            f"@ {avg_price:.4f}",
                            "positive",
                        )
                    else:
                        self._log_event(
                            "ERROR",
                            f"Ошибка завершения лимитного входа: {err}",
                        )
                # Не добавляем в still_pending — ордер обработан

            elif status == "Cancelled":
                self._log_event(
                    "ORDER_CANCELLED",
                    f"Лимитный ордер {order_id} был отменён биржей.",
                )
                # Не добавляем в still_pending

            elif status in ("New", "PartiallyFilled"):
                still_pending.append(order_info)

            else:
                # Неизвестный статус — оставляем в ожидании
                still_pending.append(order_info)

        self._pending_limit_orders = still_pending
        
    def _apply_live_management(self, candle: Dict):
        """
        Применяет management-правила к открытым позициям:
        1) Сопровождение SL (breakeven, trailing) — set_trading_stop на бирже.
        2) Мягкие выходы (time_stop, exit_on_*) — при срабатывании позиция
           закрывается market-ордером.

        Оба действия выполняются один раз на свечу.
        """
        candle_ts = candle.get("timestamp")
        if candle_ts is None:
            return

        for position in list(self.position_manager.positions):
            # Orphan-позиции не сопровождаем — у них нет реальных SL/TP.
            if position.get("is_orphan"):
                continue

            last_ts = position.get("_last_managed_candle_ts")
            if last_ts is not None and candle_ts <= last_ts:
                continue  # эту свечу для этой позиции уже обработали

            try:
                self._apply_management_to_position(position, candle)
            except Exception:
                logger.exception(
                    f"[{self.symbol}] Ошибка сопровождения позиции "
                    f"{position.get('position_id')}"
                )

            # --- Мягкие выходы (time_stop, exit_on_*) ---
            try:
                exit_price, reason = self.position_manager.check_soft_exit(
                    position, candle
                )
                if exit_price is not None and reason:
                    self._close_by_soft_exit(position, reason, exit_price)
            except Exception:
                logger.exception(
                    f"[{self.symbol}] Ошибка проверки soft exit для "
                    f"{position.get('position_id')}"
                )

            position["_last_managed_candle_ts"] = candle_ts

    def _apply_management_to_position(self, position: Dict, candle: Dict):
        """
        Пересчитывает SL для одной позиции и, если он изменился,
        отправляет обновление на биржу через set_trading_stop.
        TP не трогаем — оставляем как было при открытии.
        """
        mgmt = position.get("management", {}) or {}
        if not mgmt:
            return

        direction = position["direction"]
        entry = position["entry_price"]
        take = position["take_profit"]
        current_sl = position["stop_loss"]

        new_sl: Optional[float] = None
        breakeven_applied = False
        trailing_applied = False

        # --- 1. Breakeven ---
        trigger = mgmt.get("move_to_breakeven_at_tp_pct")
        if trigger is not None:
            try:
                trigger = float(trigger)
            except (TypeError, ValueError):
                trigger = None

            if trigger is not None and 0 < trigger <= 1:
                try:
                    offset = float(mgmt.get("breakeven_offset_pct", 0.0)) / 100.0
                except (TypeError, ValueError):
                    offset = 0.0

                if direction == "long":
                    trigger_price = entry + (take - entry) * trigger
                    be_price = entry * (1 + offset)
                    high = candle.get("high")
                    if high is not None and high >= trigger_price:
                        if current_sl < be_price:
                            new_sl = be_price
                            breakeven_applied = True
                else:
                    trigger_price = entry - (entry - take) * trigger
                    be_price = entry * (1 - offset)
                    low = candle.get("low")
                    if low is not None and low <= trigger_price:
                        if current_sl > be_price:
                            new_sl = be_price
                            breakeven_applied = True

        # --- 2. Trailing ATR ---
        trail_mult = mgmt.get("trailing_atr_mult")
        if trail_mult:
            try:
                trail_mult = float(trail_mult)
            except (TypeError, ValueError):
                trail_mult = None

            atr = candle.get("atr_14")
            close = candle.get("close")
            atr_ok = (
                atr is not None
                and not (isinstance(atr, float) and math.isnan(atr))
                and atr > 0
            )

            if trail_mult and trail_mult > 0 and atr_ok and close is not None:
                if direction == "long":
                    trail_sl = float(close) - trail_mult * float(atr)
                    if trail_sl > current_sl:
                        if new_sl is None or trail_sl > new_sl:
                            new_sl = trail_sl
                            trailing_applied = True
                else:
                    trail_sl = float(close) + trail_mult * float(atr)
                    if trail_sl < current_sl:
                        if new_sl is None or trail_sl < new_sl:
                            new_sl = trail_sl
                            trailing_applied = True

        # --- 3. Проверки перед отправкой на биржу ---
        if new_sl is None:
            return

        # Получаем tickSize. Если недоступен — НЕ отправляем set_trading_stop
        # с неокруглённым SL: биржа может отклонить такой запрос,
        # и SL молча останется старым. Лучше пропустить свечу и попробовать
        # на следующей, чем создать видимость обновления.
        success_info, info = self.executor.get_instrument_info(self.symbol)

        if not success_info or not info or not info.get("tickSize"):
            self._log_event(
                "ERROR",
                f"SL не обновлён ({direction.upper()}): tickSize недоступен "
                f"({info}). Пропуск до следующей свечи.",
            )
            return

        new_sl = self.executor.round_price(new_sl, info["tickSize"])

        # ВАЖНО: сравниваем УЖЕ ОКРУГЛЁННЫЙ new_sl со старым SL.
        # Раньше сравнение шло до округления: "сырое" 88.8798 != 88.88
        # пропускало запрос дальше, после округления биржа видела то же
        # самое 88.88 и отвечала 34040 — и так каждую свечу.
        if new_sl == current_sl:
            return

        # Проверка стороны SL относительно текущей цены.
        # Биржа отклонит SL, если он с неверной стороны от LastPrice
        # (ErrCode 10001): для long SL должен быть НИЖЕ цены, для short — ВЫШЕ.
        # Берём close последней закрытой свечи как прокси текущей цены —
        # отдельный запрос ticker избыточен.
        close_price = candle.get("close")
        if close_price is not None:
            if direction == "long" and new_sl >= close_price:
                self._log_event(
                    "MGMT",
                    f"SL пропущен ({direction.upper()}): {new_sl:.4f} >= "
                    f"close {close_price:.4f}. Ждём следующую свечу.",
                )
                return
            if direction == "short" and new_sl <= close_price:
                self._log_event(
                    "MGMT",
                    f"SL пропущен ({direction.upper()}): {new_sl:.4f} <= "
                    f"close {close_price:.4f}. Ждём следующую свечу.",
                )
                return

        success, err = self.executor.set_trading_stop(
            symbol=self.symbol,
            stop_loss=new_sl,
            take_profit=take,
        )

        if success:
            old = current_sl
            position["stop_loss"] = new_sl
            self._save_state()

            # Определяем причину сдвига для отдельного лога sl_updates.
            if breakeven_applied and trailing_applied:
                reason = "breakeven+trailing"
            elif breakeven_applied:
                reason = "breakeven"
            elif trailing_applied:
                reason = "trailing"
            else:
                reason = ""

            # Отдельная запись в sl_updates с position_id — чтобы по логу
            # можно было однозначно связать движение SL с конкретной сделкой.
            try:
                self.trade_logger.log_sl_update(
                    position_id=position.get("position_id", ""),
                    direction=direction,
                    old_sl=old,
                    new_sl=new_sl,
                    reason=reason,
                    extra={
                        "atr_14": candle.get("atr_14"),
                        "close": candle.get("close"),
                        "candle_ts": str(candle.get("timestamp") or ""),
                    },
                )
            except Exception:
                logger.exception(
                    f"[{self.symbol}] Не удалось записать sl_update "
                    f"для {position.get('position_id')}"
                )

            self._log_event(
                "MGMT",
                f"SL обновлён ({direction.upper()}): {old:.4f} → {new_sl:.4f} "
                f"[{reason or 'n/a'}]",
            )
        else:
            self._log_event(
                "ERROR",
                f"Не удалось обновить SL ({direction.upper()}): {err}",
            )

    def _close_by_soft_exit(self, position: Dict, reason: str, exit_price: float):
        """
        Закрывает позицию по мягкому выходу (time_stop / exit_on_*).

        Отправляет reduce-only market-ордер через PositionManager.close_position,
        затем сам регистрирует сделку в risk_manager и trade_logger,
        как это делает _close_all_positions.
        """
        self._log_event(
            "SOFT_EXIT",
            f"Мягкий выход ({position['direction'].upper()}): "
            f"{reason} @ ~{exit_price:.4f}",
        )

        success, result = self.position_manager.close_position(
            position, reason, current_price=exit_price
        )
        if not success:
            self._log_event(
                "ERROR",
                f"Не удалось закрыть по soft exit ({reason}): {result}",
            )
            return

        trade = result
        position_value = trade["qty"] * trade["entry_price"]
        self.risk_manager.record_closed_trade(
            trade["result_pct"], position_value
        )
        trade["position_value_usdt"] = round(position_value, 2)
        trade["pnl_usdt"] = round(
            position_value * (trade["result_pct"] / 100.0), 2
        )
        self.trade_logger.log_trade(trade)

        # Учёт risk_controls (как в _register_closed_position)
        controls = self._get_risk_controls()
        pnl_pct = trade["result_pct"]
        if pnl_pct <= 0:
            self._consecutive_losses += 1
            self._recent_loss_flags.append(1)
        else:
            self._consecutive_losses = 0
            self._recent_loss_flags.append(0)
        window = controls.get("loss_window_trades", 10)
        if window > 0 and len(self._recent_loss_flags) > window:
            self._recent_loss_flags = self._recent_loss_flags[-window:]
        if self._should_pause_after_loss(pnl_pct, controls):
            self._cooldown_remaining = controls["pause_candles_after_losses"]
            self._log_event(
                "COOLDOWN_ACTIVATED",
                f"Активирована пауза на {self._cooldown_remaining} свечей",
            )

        self._save_state()

        close_event_type = "CLOSE_WIN" if pnl_pct > 0 else "CLOSE_LOSS"
        self._log_event(
            close_event_type,
            f"Позиция закрыта ({reason}): "
            f"{trade['direction'].upper()} @ {trade['exit_price']:.4f}, "
            f"PnL={pnl_pct:+.2f}% "
            f"({trade['pnl_usdt']:+.2f} USDT)",
        )
        
    
    def _detect_closed_positions(self):
        """
        Детектирует позиции, закрытые на бирже (по исполнению ордеров).
        
        Если размер позиции на бирже стал 0 — логируем закрытие.
        """
        success, exchange_positions = self.executor.get_position_info(self.symbol)
        if not success:
            return
        
        # Строим карту: размер на бирже по направлению
        exchange_size_by_side = {}
        for pos in exchange_positions or []:
            side = pos.get("side", "").lower()
            size = float(pos.get("size", 0))
            if side in ("buy", "sell"):
                exchange_size_by_side[side] = size
        
        for position in list(self.position_manager.positions):

            # ГРЕЙС-ПЕРИОД: биржа регистрирует позицию с задержкой.
            # Не проверяем закрытие первые 2 минуты после открытия.
            # Для orphan'ов entry_time = время рестарта трекера, грейс
            # тоже уместен — биржа могла не успеть отдать свежие данные.
            opened_at = position.get("entry_time")
            if opened_at is not None:
                try:
                    if isinstance(opened_at, str):
                        opened_at = datetime.fromisoformat(opened_at)
                    if (datetime.now(timezone.utc) - opened_at).total_seconds() < 120:
                        continue
                except Exception:
                    pass

            direction = position["direction"]
            side_key = "buy" if direction == "long" else "sell"
            exchange_size = exchange_size_by_side.get(side_key, 0)

            # Если на бирже позиции этого направления нет (или она
            # радикально меньше локально записанной) — считаем закрытой.
            # Это работает и для orphan'ов: у них qty = размер с биржи,
            # так что 0 < qty*0.5 → True сразу после исчезновения позиции.
            if exchange_size < position["qty"] * 0.5:
                self._register_closed_position(position)
    
    def _register_closed_position(self, position: Dict):
        """
        Регистрирует закрытие позиции.

        Точные данные берём из Closed-PnL API биржи. Запись подбираем по:
        - стороне (Buy/Sell),
        - совпадению размера (±50%),
        - времени (не раньше времени входа минус 60 сек),
        - записи, которые уже были использованы ранее, отбрасываем.

        В risk_manager регистрируем РОВНО ОДИН раз на реальное закрытие.
        """
        direction = position["direction"]
        entry_price = position["entry_price"]
        qty = position["qty"]

        pnl_usdt = None
        exit_price = None
        order_id = None

        # 1) Подбираем подходящую запись в Closed-PnL
        success, closed = self.executor.get_closed_pnl(self.symbol, limit=20)
        if success and closed:
            expected_side = "Buy" if direction == "long" else "Sell"

            entry_ts = None
            if isinstance(position.get("entry_time"), datetime):
                entry_ts = position["entry_time"].timestamp()
            elif isinstance(position.get("entry_time"), str):
                try:
                    entry_ts = datetime.fromisoformat(position["entry_time"]).timestamp()
                except Exception:
                    entry_ts = None

            best = None  # (rec_ts, rec)
            for rec in closed:
                try:
                    rec_id = rec.get("orderId")
                    if rec_id and rec_id in self._consumed_closed_pnl_ids:
                        continue

                    rec_side = str(rec.get("side", "")).capitalize()
                    if rec_side != expected_side:
                        continue

                    rec_qty = float(rec.get("qty", 0) or 0)
                    if qty > 0 and rec_qty > 0:
                        ratio = rec_qty / qty
                        if ratio < 0.5 or ratio > 1.5:
                            continue

                    rec_ts = int(rec.get("updatedTime", 0) or 0) / 1000.0
                    if entry_ts is not None and rec_ts < entry_ts - 60:
                        continue

                    if best is None or rec_ts > best[0]:
                        best = (rec_ts, rec)
                except (TypeError, ValueError):
                    continue

            if best is not None:
                latest = best[1]
                try:
                    pnl_usdt = float(latest.get("closedPnl", 0) or 0)
                    exit_price = float(latest.get("avgExitPrice", 0) or 0) or None
                    closed_qty = float(latest.get("qty", 0) or 0)
                    if closed_qty > 0:
                        qty = closed_qty
                    avg_entry = float(latest.get("avgEntryPrice", 0) or 0)
                    if avg_entry > 0:
                        entry_price = avg_entry

                    order_id = latest.get("orderId")
                    if order_id:
                        self._consumed_closed_pnl_ids.add(order_id)
                except (TypeError, ValueError):
                    pnl_usdt = None
                    exit_price = None

        # 2) Fallback — тикер (только если биржа не отдала данные)
        if exit_price is None:
            ok, current_price = self.executor.get_ticker_price(self.symbol)
            exit_price = current_price if ok and current_price else entry_price

        position_value = qty * entry_price

        # 3) Считаем процент
        if pnl_usdt is not None and position_value > 0:
            pnl_pct = pnl_usdt / position_value * 100.0
        else:
            if direction == "long":
                pnl_pct = (exit_price - entry_price) / entry_price * 100.0
            else:
                pnl_pct = (entry_price - exit_price) / entry_price * 100.0

        # Комиссии: taker на входе и выходе. Bybit closedPnl торговые
        # комиссии НЕ включает — вычитаем вручную, чтобы результат
        # совпадал с backtest_engine._close_position.
        pnl_pct -= 2.0 * self._taker_fee_pct
        pnl_usdt = position_value * (pnl_pct / 100.0)

        # 4) Определяем exit_reason
        exit_reason = self._infer_exit_reason(position, exit_price)

        # 5) Фиксируем ОДИН раз в риск-менеджере
        self.risk_manager.record_closed_trade(pnl_pct, position_value)

        # 6) Логируем сделку
        trade = {
            "position_id": position["position_id"],
            "entry_time": position["entry_time"],
            "exit_time": datetime.now(timezone.utc),
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "qty": qty,
            "position_value_usdt": round(position_value, 2),
            "result_pct": round(pnl_pct, 2),
            "pnl_usdt": round(pnl_usdt, 2),
            "exit_reason": exit_reason,
            "stop_loss": position.get("stop_loss"),
            "take_profit": position.get("take_profit"),
            "initial_stop_loss": position.get(
                "initial_stop_loss", position.get("stop_loss")
            ),
            "initial_take_profit": position.get(
                "initial_take_profit", position.get("take_profit")
            ),
            "entry_reasons": position.get("entry_reasons", []),
        }
        self.trade_logger.log_trade(trade)

        # 7) Учёт для risk_controls (см. Fix 4)
        controls = self._get_risk_controls()
        if pnl_pct <= 0:
            self._consecutive_losses += 1
            self._recent_loss_flags.append(1)
        else:
            self._consecutive_losses = 0
            self._recent_loss_flags.append(0)

        window = controls.get("loss_window_trades", 10)
        if window > 0 and len(self._recent_loss_flags) > window:
            self._recent_loss_flags = self._recent_loss_flags[-window:]

        if self._should_pause_after_loss(pnl_pct, controls):
            self._cooldown_remaining = controls["pause_candles_after_losses"]
            self._log_event(
                "COOLDOWN_ACTIVATED",
                f"Активирована пауза на {self._cooldown_remaining} свечей",
            )

        # 8) Удаляем из локального состояния и синхронно сохраняем на диск.
        #    Важно: сначала персистим УДАЛЕНИЕ, потом логируем CLOSE.
        #    Если процесс упадёт между этими шагами, при следующем
        #    старте позиции в файле уже не будет и дубля CLOSE не возникнет.
        self.position_manager.positions = [
            p for p in self.position_manager.positions
            if p["position_id"] != position["position_id"]
        ]
        self._save_state()

        # Разделяем на два event_type: HTML-лог раскрасит прибыльные
        # сделки зелёным, убыточные — красным.
        close_event_type = "CLOSE_WIN" if pnl_pct > 0 else "CLOSE_LOSS"

        self._log_event(
            close_event_type,
            f"Позиция закрыта: {direction.upper()} @ {exit_price:.4f}, "
            f"PnL={pnl_pct:+.2f}% ({pnl_usdt:+.2f} USDT), reason={exit_reason}",
        )

    def _infer_exit_reason(self, position: Dict, exit_price: float) -> str:
        """
        Определяет причину выхода по сравнением цены выхода с известными
        уровнями позиции. Точная причина в Closed-PnL не приходит,
        но этого достаточно для диагностики.
        """
        if exit_price is None:
            return "unknown"

        def _close(a, b, tol_pct: float = 0.3) -> bool:
            if not a or not b:
                return False
            return abs(a - b) / a * 100.0 <= tol_pct

        sl = position.get("stop_loss")
        tp = position.get("take_profit")
        entry = position.get("entry_price")

        if _close(exit_price, sl):
            # Если SL ≈ entry (с точностью до 0.2%) — это breakeven
            if _close(exit_price, entry, tol_pct=0.2):
                return "breakeven"
            return "stop_loss"
        if _close(exit_price, tp):
            return "take_profit"
        return "manual_or_other"
    
    def _get_unrealized_pnl(self) -> float:
        """Возвращает нереализованный ПнЛ открытых позиций в USDT."""
        success, exchange_positions = self.executor.get_position_info(self.symbol)
        if not success:
            return 0.0
        
        total_unrealized = 0.0
        for pos in exchange_positions or []:
            try:
                unrealized = float(pos.get("unrealisedPnl", 0))
                total_unrealized += unrealized
            except (TypeError, ValueError):
                continue
        
        return total_unrealized
    
    def _handle_daily_stop(self):
        """Обработка дневного лимита убытка."""
        self._log_event(
            "DAILY_STOP",
            f"Достигнут дневной лимит убытка. Закрытие позиций и остановка торговли.",
        )
        self._notify(
            f"[{self.symbol}] Достигнут дневной лимит убытка. "
            "Позиции закрыты, торговля остановлена. Для возобновления нажмите «Начать торговлю».",
            "negative",
        )
        
        self._close_all_positions(reason="daily_loss_limit")
        self.stop(reason="daily_loss_limit")
    
    # ----------------------------------------------------------------
    # Оптимизация
    # ----------------------------------------------------------------
    
    def _trigger_optimization(self):
        """Запускает оптимизацию стратегии при месячном лимите убытка."""
        if self.is_optimizing:
            return
        
        self.is_optimizing = True
        self.status_message = "Запуск оптимизации..."
        
        self._log_event(
            "OPTIMIZATION_START",
            f"Месячный лимит убытка достигнут. Запуск оптимизации "
            f"за последние {self.cfg.optimization_days_back} дней.",
        )
        self._notify(
            f"[{self.symbol}] Месячный лимит убытка достигнут. "
            "Запущена оптимизация стратегии. Торговля приостановлена.",
            "warning",
        )
        
        # Запускаем оптимизацию в отдельном потоке, чтобы не блокировать мониторинг
        threading.Thread(target=self._run_optimization, daemon=True).start()
    
    def _run_optimization(self):
        """Выполняет оптимизацию стратегии и возобновляет торговлю."""
        try:
            now = datetime.now(timezone.utc)
            start_date = now - timedelta(days=self.cfg.optimization_days_back)
            end_date = now
            
            result = self.backtest_optimizer.run(
                symbol=self.symbol,
                category=self.cfg.category,
                max_iterations=self.cfg.optimization_iterations,
                initial_capital=self.cfg.initial_capital,
                commission_pct=self._taker_fee_pct,
                csv_path=str(config.DATA_DIR / "auto_trading" / "optimization"),
                start_date=start_date,
                end_date=end_date,
                walk_forward_folds=self.cfg.walk_forward_folds,
                save_best_to_memory=True,
            )
            
            # Определяем, была ли сохранена новая стратегия
            best = result.get("best")
            new_strategy_saved = best is not None
            
            # Логируем результат оптимизации
            log_result = {
                "trigger_reason": "monthly_loss_limit",
                "history": result.get("history", []),
                "best": best,
                "new_strategy_saved": new_strategy_saved,
            }
            self.trade_logger.log_optimization(log_result)
            
            if new_strategy_saved:
                self._log_event(
                    "OPTIMIZATION_DONE",
                    "Оптимизация завершена успешно. Новая стратегия сохранена.",
                )
                self._notify(
                    f"[{self.symbol}] Оптимизация завершена. "
                    "Новая стратегия применена, торговля возобновляется.",
                    "positive",
                )
            else:
                self._log_event(
                    "OPTIMIZATION_DONE",
                    "Оптимизация завершена, но подходящая стратегия не найдена.",
                )
                self._notify(
                    f"[{self.symbol}] Оптимизация завершена, но новая стратегия не найдена. "
                    "Торговля продолжается со старой стратегией.",
                    "warning",
                )
            
            # Флаг НЕ сбрасываем — оптимизация запустится снова только
            # при смене месяца (_check_rollover сбросит его автоматически)
            self.is_optimizing = False
            self.status_message = "Торговля возобновлена после оптимизации"
        
        except Exception as e:
            logger.exception(f"[{self.symbol}] Ошибка оптимизации")
            self._log_event("ERROR", f"Ошибка оптимизации: {e}")
            self._notify(f"[{self.symbol}] Ошибка оптимизации: {e}", "negative")
            self.is_optimizing = False
    
    # ----------------------------------------------------------------
    # Закрытие позиций
    # ----------------------------------------------------------------
    
    def _close_all_positions(self, reason: str = "manual"):
        """Закрывает все открытые позиции."""
        if not self.position_manager.positions:
            return
        
        self.status_message = "Закрытие позиций..."
        
        trades = self.position_manager.close_all_positions(reason=reason)
        
        for trade in trades:
            # Регистрируем в риск-менеджере
            position_value = trade["qty"] * trade["entry_price"]
            self.risk_manager.record_closed_trade(trade["result_pct"], position_value)
            
            # Логируем сделку
            trade["position_value_usdt"] = round(position_value, 2)
            trade["pnl_usdt"] = round(position_value * (trade["result_pct"] / 100.0), 2)
            self.trade_logger.log_trade(trade)

        self._save_state()
        self._log_event(
            "CLOSE",
            f"Закрыто {len(trades)} позиций по причине: {reason}",
        )
    
    def close_all_public(self):
        """Публичный метод для закрытия всех позиций (вызывается из UI)."""
        self._close_all_positions(reason="manual_from_ui")
        self._notify(f"[{self.symbol}] Все позиции закрыты вручную", "info")
    
    def close_positions_public(self):
        """Синоним для close_all_public (закрытие по монете)."""
        self.close_all_public()
    
    # ----------------------------------------------------------------
    # Вспомогательные
    # ----------------------------------------------------------------
    
    def _log_event(
        self,
        event_type: str,
        message: str,
        details: Optional[str] = None,
        group: Optional[str] = None,
    ):
        """Логирует событие в файл и в локальный лог."""
        self.trade_logger.log_event(event_type, message, details, group=group)
        self.log_lines.append(
            f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}"
        )
        # Ограничиваем размер локального лога
        if len(self.log_lines) > 100:
            self.log_lines = self.log_lines[-100:]
    
    def _notify(self, message: str, ntype: str):
        """Отправляет уведомление в UI."""
        try:
            self.notify_fn(message, ntype)
        except Exception as e:
            logger.error(f"Не удалось отправить уведомление: {e}")

    # ----------------------------------------------------------------
    # Персистентность позиций
    # ----------------------------------------------------------------

    def _serialize_position(self, pos: Dict) -> Dict:
        """Готовит позицию к JSON-сериализации."""
        out = {}
        for key, value in pos.items():
            # Транзиентные поля (например _last_managed_candle_ts) не сохраняем.
            if key.startswith("_"):
                continue
            if isinstance(value, datetime):
                out[key] = {"__dt__": value.isoformat()}
            else:
                out[key] = value
        return out

    @staticmethod
    def _deserialize_position(raw: Dict) -> Dict:
        """Обратное преобразование из JSON."""
        out = {}
        for key, value in raw.items():
            if isinstance(value, dict) and "__dt__" in value:
                try:
                    out[key] = datetime.fromisoformat(value["__dt__"])
                except (TypeError, ValueError):
                    out[key] = None
            else:
                out[key] = value
        return out

    def _save_state(self):
        """
        Атомарно сохраняет список позиций, consumed Closed-PnL ids
        и состояние risk_controls (серия убытков, окно, cooldown).
        Падение записи не должно ломать торговлю.
        """
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)

            # Сериализуем лимитные ордера
            serialized_orders = []
            for order in self._pending_limit_orders:
                serialized_orders.append({
                    "order_id": order["order_id"],
                    "direction": order["direction"],
                    "qty": order["qty"],
                    "price": order["price"],
                    "placed_at": order["placed_at"].isoformat(),
                    # side_spec, reasons, indicators не сериализуем —
                    # при рестарте ордер будет отменён, позиция не откроется.
                })

            payload = {
                "symbol": self.symbol,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "positions": [
                    self._serialize_position(p)
                    for p in self.position_manager.positions
                ],
                "pending_limit_orders": serialized_orders,
                "consumed_closed_pnl_ids": sorted(self._consumed_closed_pnl_ids),
                "risk_controls_state": {
                    "consecutive_losses": int(self._consecutive_losses),
                    "recent_loss_flags": list(self._recent_loss_flags),
                    "cooldown_remaining": int(self._cooldown_remaining),
                },
            }

            tmp_path = self._state_path.with_suffix(".json.tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            os.replace(tmp_path, self._state_path)
        except Exception:
            logger.exception(
                f"[{self.symbol}] Не удалось сохранить состояние позиций"
            )

    def _load_state(self):
        """
        Загружает сохранённые позиции с диска.
        Восстановленные позиции помечаются как «наши» (is_orphan не ставим),
        потому что у них сохранены bybit_order_id и реальные SL/TP.
        """
        if not self._state_path.exists():
            return

        try:
            with open(self._state_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            positions_raw = payload.get("positions", []) or []
            loaded = [self._deserialize_position(p) for p in positions_raw]

            # Защита от битого формата: у позиции должны быть обязательные поля.
            valid = []
            for pos in loaded:
                if not all(
                    k in pos for k in (
                        "position_id", "direction", "entry_price", "qty",
                        "stop_loss", "take_profit", "entry_time",
                    )
                ):
                    logger.warning(
                        f"[{self.symbol}] Пропущена некорректная запись позиции: {pos}"
                    )
                    continue
                # orphan-флаг у восстановленных с диска не должен быть True:
                # в файл попадают только «наши» позиции.
                pos["is_orphan"] = bool(pos.get("is_orphan", False))
                valid.append(pos)

            self.position_manager.positions = valid

            consumed = payload.get("consumed_closed_pnl_ids", []) or []
            self._consumed_closed_pnl_ids = set(str(x) for x in consumed)

            # Восстанавливаем risk_controls только если файл свежий.
            # Если приложение было выключено дольше
            # RISK_CONTROLS_MAX_STALENESS_HOURS, состояние сбрасывается:
            # cooldown в свечах при выключенном боте не тикает, а серия
            # убытков «выветривается» естественным образом.
            rc = payload.get("risk_controls_state") or {}
            rc_restored = False
            rc_stale = False
            age_hours = None

            if rc:
                saved_at_str = payload.get("saved_at")
                saved_at_dt = None
                if saved_at_str:
                    try:
                        saved_at_dt = datetime.fromisoformat(saved_at_str)
                        if saved_at_dt.tzinfo is None:
                            saved_at_dt = saved_at_dt.replace(tzinfo=timezone.utc)
                    except (TypeError, ValueError):
                        saved_at_dt = None

                if saved_at_dt is not None:
                    now_utc = datetime.now(timezone.utc)
                    age_hours = (now_utc - saved_at_dt).total_seconds() / 3600.0

                if (
                    age_hours is not None
                    and age_hours > RISK_CONTROLS_MAX_STALENESS_HOURS
                ):
                    rc_stale = True
                    logger.info(
                        f"[{self.symbol}] risk_controls_state устарел "
                        f"({age_hours:.1f}ч > "
                        f"{RISK_CONTROLS_MAX_STALENESS_HOURS}ч). "
                        f"Сброс: consecutive_losses, recent_loss_flags, "
                        f"cooldown_remaining."
                    )
                    self._log_event(
                        "RISK_CONTROLS_RESET",
                        f"Состояние risk_controls сброшено "
                        f"(пауза {age_hours:.1f}ч без торговли).",
                    )
                else:
                    # Свежее состояние — восстанавливаем.
                    try:
                        self._consecutive_losses = int(
                            rc.get("consecutive_losses", 0)
                        )
                    except (TypeError, ValueError):
                        self._consecutive_losses = 0
                    flags_raw = rc.get("recent_loss_flags", []) or []
                    self._recent_loss_flags = [
                        int(x) for x in flags_raw
                        if x in (0, 1, "0", "1")
                    ]
                    try:
                        self._cooldown_remaining = int(
                            rc.get("cooldown_remaining", 0)
                        )
                    except (TypeError, ValueError):
                        self._cooldown_remaining = 0
                    rc_restored = True

            # Восстановление лимитных ордеров: при рестарте мы НЕ можем
            # гарантировать, что индикаторы и спека актуальны.
            # Поэтому просто отменяем все незавершённые ордера.
            pending_orders = payload.get("pending_limit_orders", []) or []
            for po in pending_orders:
                oid = po.get("order_id")
                if oid:
                    self.executor.cancel_order(self.symbol, oid)
                    self._log_event(
                        "ORDER_CANCELLED",
                        f"Ордер {oid} отменён при перезапуске.",
                    )
            self._pending_limit_orders = []        

            if valid or consumed or rc_restored:
                logger.info(
                    f"[{self.symbol}] Восстановлено из состояния на диске: "
                    f"позиций={len(valid)}, "
                    f"consumed_closed_pnl_ids={len(consumed)}, "
                    f"consecutive_losses={self._consecutive_losses}, "
                    f"cooldown_remaining={self._cooldown_remaining}"
                )
                detail = (
                    f"Восстановлено из файла: {len(valid)} позиций, "
                    f"{len(consumed)} зачтённых Closed-PnL записей"
                )
                if rc_restored:
                    detail += (
                        f", серия убытков={self._consecutive_losses}, "
                        f"cooldown={self._cooldown_remaining} свечей"
                    )
                elif rc_stale:
                    detail += ", риск-контроль сброшен (устарел)"
                detail += "."
                self._log_event("STATE_LOADED", detail)

            # ОДНОразовая диагностика по orderId: если сохранённый ордер
            # уже отменён/отклонён биржей — предупреждаем пользователя.
            # Делаем это только при загрузке, а не каждую итерацию, чтобы
            # не тратить rate limit.
            for pos in valid:
                order_id = pos.get("bybit_order_id")
                if not order_id:
                    continue
                ok, info = self.executor.get_order_info(self.symbol, order_id)
                if ok and info:
                    status = str(info.get("orderStatus", ""))
                    if status in ("Cancelled", "Rejected", "Deactivated"):
                        logger.warning(
                            f"[{self.symbol}] Сохранённый ордер {order_id} "
                            f"для позиции {pos['position_id']} имеет статус "
                            f"{status} — стоит проверить позицию вручную."
                        )
                        self._log_event(
                            "STATE_WARNING",
                            f"Ордер {order_id} в статусе {status}. "
                            "Проверьте позицию вручную.",
                        )
        except Exception:
            logger.exception(
                f"[{self.symbol}] Не удалось загрузить состояние позиций"
            )
    
    def get_state(self) -> Dict:
        """Возвращает текущее состояние трейдера (для UI)."""
        return {
            "symbol": self.symbol,
            "is_running": self.is_running,
            "is_optimizing": self.is_optimizing,
            "status_message": self.status_message,
            
            # --- HEARTBEAT ---
            "last_iteration_utc": (
                self.last_iteration_utc.strftime("%H:%M:%S") + " UTC"
                if self.last_iteration_utc else None
            ),
            "last_candle_time": (
                str(self.last_candle_time)[:16] if self.last_candle_time else None
            ),
            "last_price": self.last_price,
            "last_indicators_summary": self.last_indicators_summary,
            "candles_processed": self.candles_processed,
            "_last_iteration_ts": (
                self.last_iteration_utc.timestamp() if self.last_iteration_utc else None
            ),
            
            "open_positions": len(self.position_manager.positions),
            "allocated_capital": round(self.allocated_capital, 2),
            "current_capital": round(self.risk_manager.current_capital, 2),
            "daily_pnl": round(self.risk_manager.daily_realized_pnl, 2),
            "monthly_pnl": round(self.risk_manager.monthly_realized_pnl, 2),
            "daily_pnl_pct": round(self.risk_manager.get_total_daily_loss_pct(0.0), 2),
            "monthly_pnl_pct": round(self.risk_manager.get_total_monthly_loss_pct(), 2),
            "daily_loss_limit": self.cfg.daily_loss_limit_pct,
            "monthly_loss_limit": self.cfg.monthly_loss_limit_pct,
            "log_lines": self.log_lines[-20:],
        }

    @staticmethod
    def _format_condition(cond: Dict, indicators: Dict) -> str:
        """Форматирует условие для лога."""
        try:
            ctype = cond.get("type")
            if ctype == "compare":
                left = cond.get("left")
                op = cond.get("op", "?")
                right = cond.get("right")
                left_val = indicators.get(left, "?")
                if isinstance(right, str):
                    right_val = indicators.get(right, "?")
                    return f"{left}({left_val}) {op} {right}({right_val})"
                elif isinstance(right, dict):
                    ref = right.get("name")
                    mult = right.get("mult", 1.0)
                    ref_val = indicators.get(ref, 0) * mult
                    return f"{left}({left_val}) {op} {ref}*{mult}({ref_val:.4f})"
                else:
                    return f"{left}({left_val}) {op} {right}"
            elif ctype == "range":
                val = cond.get("value")
                vmin = cond.get("min", 0)
                vmax = cond.get("max", 100)
                val_v = indicators.get(val, "?")
                return f"{val}({val_v}) in [{vmin}, {vmax}]"
            elif ctype == "volume":
                ref = cond.get("ref")
                mult = cond.get("mult", 1.0)
                vol = indicators.get("volume", 0)
                ref_v = indicators.get(ref, 0) * mult
                return f"volume({vol}) > {ref}*{mult}({ref_v:.0f})"
            elif ctype == "slope":
                val = cond.get("value")
                direction = cond.get("dir", "?")
                return f"slope({val}) {direction}"
            else:
                return f"[{ctype}]"
        except Exception:
            return "[format_error]"


# ----------------------------------------------------------------
# Глобальные функции управления
# ----------------------------------------------------------------

def start_auto_trading(
    cfg: AutoTraderConfig,
    *,
    executor,
    fetcher,
    memory,
    backtest_engine,
    backtest_optimizer,
    ai_client,
    notify_fn,
    total_capital_usdt: float,
) -> bool:
    """
    Запускает автоторговлю для монеты.

    Возвращает True, если трейдер реально стартовал.
    False — если запуск отклонён (уже запущен / авторежим /
    настройки не позволяют открыть позицию).
    """
    with _active_traders_lock:
        existing = _active_traders.get(cfg.symbol)
        if existing and existing.is_running:
            return False

        trader = AutoTraderSymbol(
            cfg=cfg,
            executor=executor,
            fetcher=fetcher,
            memory=memory,
            backtest_engine=backtest_engine,
            backtest_optimizer=backtest_optimizer,
            ai_client=ai_client,
            notify_fn=notify_fn,
            total_capital_usdt=total_capital_usdt,
        )

        _active_traders[cfg.symbol] = trader

    started = trader.start()

    if not started:
        # Убираем из реестра, чтобы UI не показывал его как запущенный.
        with _active_traders_lock:
            if _active_traders.get(cfg.symbol) is trader:
                _active_traders.pop(cfg.symbol, None)

    return started


def stop_auto_trading(symbol: str, close_positions: bool = False, reason: str = "manual"):
    """Останавливает автоторговлю для монеты."""
    with _active_traders_lock:
        trader = _active_traders.get(symbol)
    
    if trader:
        trader.stop(close_positions=close_positions, reason=reason)


def close_all_positions_for_symbol(symbol: str):
    """Закрывает все позиции по монете (без остановки торговли)."""
    with _active_traders_lock:
        trader = _active_traders.get(symbol)
    
    if trader:
        trader.close_all_public()


def close_all_positions_all_symbols():
    """Закрывает все позиции по всем монетам."""
    with _active_traders_lock:
        traders = list(_active_traders.values())
    
    for trader in traders:
        trader.close_all_public()