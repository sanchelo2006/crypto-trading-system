"""
Загружает готовую портфельную спеку в память монет.

Использование:
  python scripts/deploy_portfolio.py                    # все 3 монеты
  python scripts/deploy_portfolio.py ARBUSDT DOGEUSDT   # только указанные
  python scripts/deploy_portfolio.py --list             # показать доступные
"""
import json
import sys
from pathlib import Path

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory_manager import MemoryManager
from strategy_spec import validate_strategy_spec, ensure_engine_defaults

# Монеты для портфеля (лучшая тройка из исследования)
PORTFOLIO_SYMBOLS = ["ARBUSDT", "DOGEUSDT", "SOLUSDT"]

def load_portfolio_spec() -> dict:
    """Загружает спеку из JSON файла."""
    spec_path = Path(__file__).resolve().parent.parent / "research" / "portfolio_spec.json"
    if not spec_path.exists():
        print(f"❌ Файл спеки не найден: {spec_path}")
        sys.exit(1)
    
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    
    # Валидация
    err = validate_strategy_spec(spec)
    if err:
        print(f"❌ Спек невалиден: {err}")
        sys.exit(1)
    
    # Дополняем дефолтами движка
    spec = ensure_engine_defaults(spec)
    
    return spec

def deploy_to_symbol(mm: MemoryManager, symbol: str, spec: dict):
    """Записывает спек в память одной монеты."""
    cm = mm.get(symbol)
    
    # Обновляем state
    cm.update_state({
        "strategy_spec": spec,
        "is_initialized": True,
        "last_signal": None,
        "active_trade": None,
    })
    
    # Обновляем правила (человекочитаемые)
    rules_text = f"""# Портфельная стратегия (шорт)

## Вход (OR-логика, 3 группы):
1. **m2s2_vol2.0**: падение >2 ATR за 2 свечи + объём >2x среднего
2. **act_short**: аномальная активность (объём×волатильность >3.0) + падение >1 ATR
3. **mom2_short_t2.0**: падение >2 ATR за 2 свечи

## Выход:
- Stop Loss: 2.0 × ATR(14)
- Take Profit: 1.5 × ATR(14)
- Max Hold: 8 свечей (2 часа)
- Безубыток: отключён
- Трейлинг: отключён

## Risk Controls:
- Max consecutive losses: 6
- Pause after losses: 8 свечей
- Max losses in window: 10 из 20 сделок

## Заморозка:
- Входы заморожены (frozen_entries: true)
- Оптимизатор не может менять условия входа
- Может оптимизировать только выходы и риск-контроли

Исследование: research/portfolio_lab.py (итерация 6)
Результаты: +0.24% exp, 100% прибыльных датасетов, режимно независима
"""
    cm.update_rules(rules_text)
    
    print(f"✅ {symbol}: спек загружен, is_initialized=True")

def main():
    # Парсим аргументы
    symbols = PORTFOLIO_SYMBOLS
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list":
            print("Доступные монеты для портфеля:")
            for s in PORTFOLIO_SYMBOLS:
                print(f"  - {s}")
            return
        symbols = sys.argv[1:]
    
    print(f"📦 Загрузка портфельной спеки в {len(symbols)} монет: {', '.join(symbols)}")
    
    # Загружаем спеку
    spec = load_portfolio_spec()
    print(f"✅ Спек валиден: {len(spec.get('short', {}).get('entry_any', []))} групп входа")
    
    # Инициализируем memory manager
    mm = MemoryManager()
    
    # Деплоим в каждую монету
    for symbol in symbols:
        deploy_to_symbol(mm, symbol, spec)
    
    print(f"\n🎉 Готово! Портфельная стратегия загружена в {len(symbols)} монет.")
    print("Теперь автотрейдер будет использовать её для торговли.")

if __name__ == "__main__":
    main()