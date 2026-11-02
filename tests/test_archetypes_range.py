"""
Юнит-тесты семейств архетипов.

Поле "regime" удалено из реестра вместе с детектором режимов рынка.
Семьи определяются по префиксу имени архетипа (см. archetypes.py):
- A*  — трендовая/импульсная семья (пробои, откаты, моментум);
- R*  — флэтовая семья (mean-reversion внутри диапазона).
"""
import pytest
from optimization.archetypes import ARCHETYPES


def _range_archetypes():
    """Флэтовая семья: архетипы с префиксом имени 'R'."""
    return [a for a in ARCHETYPES if a["name"].startswith("R")]


def _trend_archetypes():
    """Трендовая/импульсная семья: архетипы с префиксом имени 'A'."""
    return [a for a in ARCHETYPES if a["name"].startswith("A")]


class TestRangeArchetypes:
    """Тесты для флэтовых архетипов."""
    
    def test_range_archetypes_exist(self):
        """Существуют флэтовые архетипы."""
        range_archetypes = _range_archetypes()
        assert len(range_archetypes) >= 4, f"Ожидалось >= 4 флэтовых архетипов, найдено {len(range_archetypes)}"
    
    def test_range_archetypes_have_required_fields(self):
        """Флэтовые архетипы имеют все необходимые поля."""
        range_archetypes = _range_archetypes()
        
        for archetype in range_archetypes:
            assert "name" in archetype, f"Архетип {archetype} не имеет поля 'name'"
            assert "build" in archetype, f"Архетип {archetype} не имеет поля 'build'"
            assert "grid" in archetype, f"Архетип {archetype} не имеет поля 'grid'"
            assert callable(archetype["build"]), f"build не является функцией для {archetype['name']}"
            assert callable(archetype["grid"]), f"grid не является функцией для {archetype['name']}"
    
    def test_range_archetypes_build_valid_spec(self):
        """Флэтовые архетипы строят валидный strategy_spec."""
        range_archetypes = _range_archetypes()
        
        for archetype in range_archetypes:
            grid = archetype["grid"]()
            assert len(grid) > 0, f"grid() для {archetype['name']} вернул пустой список"
            
            # Берём первый набор параметров
            params = grid[0]
            spec = archetype["build"](params)
            
            # Проверяем структуру
            assert "long" in spec, f"{archetype['name']}: spec не имеет 'long'"
            assert "short" in spec, f"{archetype['name']}: spec не имеет 'short'"
            assert "filters" in spec, f"{archetype['name']}: spec не имеет 'filters'"
            assert "risk_controls" in spec, f"{archetype['name']}: spec не имеет 'risk_controls'"
            
            # Проверяем структуру long/short
            for side in ["long", "short"]:
                assert "entry" in spec[side], f"{archetype['name']}: {side} не имеет 'entry'"
                assert "stop_loss_atr_mult" in spec[side], f"{archetype['name']}: {side} не имеет 'stop_loss_atr_mult'"
                assert "take_profit_atr_mult" in spec[side], f"{archetype['name']}: {side} не имеет 'take_profit_atr_mult'"
                assert "management" in spec[side], f"{archetype['name']}: {side} не имеет 'management'"
    
    def test_range_archetypes_use_adx_filter(self):
        """Флэтовые архетипы используют chop-фильтр: adx_14 (15m) < порог.
        Именно adx_14, а не adx_14_1h: adx_14 входит в разрешённый набор
        «трендовых имён» валидатора (adx_14_1h — нет), поэтому спека
        проходит validate_strategy_spec. Низкий ADX означает флэт, где
        mean-reversion работает лучше всего.
        """
        range_archetypes = _range_archetypes()

        for archetype in range_archetypes:
            grid = archetype["grid"]()
            params = grid[0]
            spec = archetype["build"](params)

            for side in ["long", "short"]:
                conditions = spec[side]["entry"]
                adx_condition = any(
                    c.get("type") == "compare" and c.get("left") == "adx_14" and c.get("op") == "<"
                    for c in conditions
                )
                assert adx_condition, f"{archetype['name']}: {side} не имеет фильтра ADX < threshold"
    
    def test_range_archetypes_capped_take_profits(self):
        """Тейки флэтовой семьи ограничены 8 ATR: цель mean-reversion —
        максимум противоположная граница канала, а не трендовое движение.
        (Порог <=4 ATR снят после скрининга: на реальных данных отскоки
        от экстремумов RSI/BB регулярно проходят 4-8 ATR, оптимальную
        цель выбирает сетка.)
        """
        range_archetypes = _range_archetypes()

        for archetype in range_archetypes:
            for params in archetype["grid"]():
                spec = archetype["build"](params)
                tp_long = spec["long"]["take_profit_atr_mult"]
                tp_short = spec["short"]["take_profit_atr_mult"]
                assert tp_long <= 8.0, f"{archetype['name']}: long TP {tp_long} > 8.0 ATR"
                assert tp_short <= 8.0, f"{archetype['name']}: short TP {tp_short} > 8.0 ATR"

    def test_range_targets_shorter_than_trend(self):
        """Принцип семей: флэтовая семья целится заметно ближе трендовой.
        Mean-reversion снимает прибыль у границы канала, тренд держит движение.
        """
        range_archetypes = _range_archetypes()
        trend_archetypes = _trend_archetypes()

        def max_tp(arch):
            return max(p["tp"] for p in arch["grid"]())

        range_cap = max(max_tp(a) for a in range_archetypes)
        trend_cap = max(max_tp(a) for a in trend_archetypes)
        assert range_cap <= 8.0, f"потолок тейков флэта {range_cap} > 8.0"
        assert trend_cap > range_cap, (
            f"трендовая семья ({trend_cap}) должна целиться дальше флэтовой ({range_cap})"
        )
    
    def test_range_archetypes_small_grid(self):
        """Флэтовые архетипы имеют разумный размер grid (не слишком большой)."""
        range_archetypes = _range_archetypes()
        
        for archetype in range_archetypes:
            grid = archetype["grid"]()
            # Не более 200 комбинаций для фазы 0
            assert len(grid) <= 200, f"{archetype['name']}: grid слишком большой ({len(grid)} комбинаций)"
    
    def test_trend_archetypes_exist(self):
        """Трендовые архетипы (семья A*) присутствуют в реестре."""
        trend_archetypes = _trend_archetypes()
        assert len(trend_archetypes) >= 4, f"Ожидалось >= 4 трендовых архетипов, найдено {len(trend_archetypes)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])