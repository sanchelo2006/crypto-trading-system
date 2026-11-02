"""
Детектирование торговых сигналов из strategy_spec.

Читает текущие индикаторы и рыночные данные, проверяет условия входа
по алгоритму, идентичному backtest_engine.py.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

from strategy_spec import evaluate_conditions, explain_conditions

logger = logging.getLogger(__name__)


class SignalDetector:
    """
    Детектор сигналов на основе strategy_spec.
    
    Использует ту же логику, что и движок бэктеста:
    - глобальные фильтры
    - entry / entry_any блоки
    - поддержка Long и Short
    """
    
    def __init__(self):
        pass
    
    def detect_signals(
        self,
        strategy_spec: Dict,
        current_indicators: Dict,
        prev_indicators: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Проверяет условия входа Long и Short.
        
        Args:
            strategy_spec: спецификация стратегии
            current_indicators: текущие индикаторы (последняя свеча)
            prev_indicators: индикаторы предыдущей свечи
        
        Returns:
            Список сигналов: [{"direction": "long"/"short", "reasons": [...]}]
        """
        signals = []
        
        if not strategy_spec:
            return signals
        
        if prev_indicators is None:
            prev_indicators = {}
        
        # Глобальные фильтры
        filters = strategy_spec.get("filters") or []
        if filters and not evaluate_conditions(filters, current_indicators, prev_indicators):
            logger.debug("Глобальные фильтры не пройдены — сигналов нет")
            return signals
        
        # Long
        long_spec = strategy_spec.get("long") or {}
        matched_long = self._check_entry(long_spec, current_indicators, prev_indicators)
        if matched_long is not None:
            signals.append({
                "direction": "long",
                "reasons": explain_conditions(matched_long, current_indicators, prev_indicators),
                "side_spec": long_spec,
            })

            logger.info("LONG сигнал: %s", explain_conditions(matched_long, current_indicators, prev_indicators))
        else:
            # 👇 ДОБАВЬТЕ ЭТО
            logger.debug("LONG не сработал: условия не выполнены. Проверьте entry/entry_any в long spec.")
            # Опционально: распечатайте сами условия для отладки
            entry_conditions = long_spec.get("entry_any") or [long_spec.get("entry")]
            logger.debug("Long entry_any: %s", entry_conditions)
        
        # Short
        short_spec = strategy_spec.get("short") or {}
        matched_short = self._check_entry(short_spec, current_indicators, prev_indicators)
        if matched_short is not None:
            signals.append({
                "direction": "short",
                "reasons": explain_conditions(matched_short, current_indicators, prev_indicators),
                "side_spec": short_spec,
            })

            logger.info("SHORT сигнал: %s", explain_conditions(matched_short, current_indicators, prev_indicators))
        else:
            logger.debug("SHORT не сработал: условия не выполнены.")
            # Опционально: распечатайте сами условия для отладки
            entry_conditions = short_spec.get("entry_any") or [short_spec.get("entry")]
            logger.debug("Short entry_any: %s", entry_conditions)
        
        return signals
    
    @staticmethod
    def _check_entry(side_spec: Dict, row: Dict, prev: Dict) -> Optional[List]:
        """
        Проверяет вход по side_spec (long/short).
        
        Приоритет: entry_any (список OR-групп). Если нет — старое поле entry.
        Полная обратная совместимость со старыми spec.
        
        Returns:
            список условий сработавшей группы (для explain) или None
        """
        groups = side_spec.get("entry_any")
        if isinstance(groups, list) and groups:
            for group in groups:
                if isinstance(group, list) and group and evaluate_conditions(group, row, prev):
                    return group
            return None
        
        entry = side_spec.get("entry")
        if isinstance(entry, list) and entry and evaluate_conditions(entry, row, prev):
            return entry
        
        return None