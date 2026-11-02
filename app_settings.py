"""
Глобальные динамические настройки приложения.
Хранятся в data/settings.json, загружаются при старте.
"""
import json
import logging
from pathlib import Path

import config

logger = logging.getLogger(__name__)

DEFAULT_SETTINGS = {
    # Пакеты данных
    "morning_days_back": 3,
    "intraday_candles_count": 50,
    "include_1h": True,
    "include_15m": True,
    "include_fundamentals": True,
    "include_orderbook": True,
    "include_trades": True,
    "include_news": True,

    # Параметры наполнения пакетов
    "orderbook_depth": 25,
    "recent_trades_limit": 100,
    "fundamentals_limit": 1000,

    # Авторежим
    "auto_interval_minutes": 15,
    "max_trades_per_day": 5,

    # Риск-менеджмент
    "initial_capital": 0.0,
    "max_stop_loss_pct": 2.0,
    "atr_stop_multiplier": 1.5,

    # Интерфейс
    "timezone_offset_hours": 0,

    # Новости
    "news_sources_enabled": [],
    "news_hours_back_morning": 24,
    "news_hours_back_intraday": 6,
    "use_finbert_sentiment": True,

    # Бэктест / оптимизация
    "allow_weak_holdout_save": False,

    # Robust-отбор по walk-forward окнам
    "robust_min_profitable_windows_ratio": 0.67,
    "robust_min_positive_expectancy_windows_ratio": 0.67,
    "robust_min_pf_pass_windows_ratio": 0.67,
    "robust_min_median_window_return_pct": 0.0,
    "robust_max_worst_window_return_pct": -5.0,
    "robust_min_stability_score": 0.5,
    "block_fragile_best_candidate": True,
}

class AppSettings:
    def __init__(self, path: str = None):
        if path is None:
            path = config.DATA_DIR / "settings.json"
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = DEFAULT_SETTINGS.copy()
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    stored = json.load(f)
                for k, v in stored.items():
                    if k in self.data:
                        self.data[k] = v
                logger.info(f"Настройки загружены из {self.path}")
            except Exception as e:
                logger.error(f"Ошибка загрузки настроек: {e}")

    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            logger.info(f"Настройки сохранены в {self.path}")
            return True
        except Exception as e:
            logger.error(f"Ошибка сохранения настроек: {e}")
            return False

    def get(self, key, default=None):
        return self.data.get(key, default)

    def update(self, updates: dict):
        for k, v in updates.items():
            if k in self.data:
                self.data[k] = v
