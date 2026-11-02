"""
Проверка скрининга архетипов с разбивкой по режимам рынка.
Запуск: python check_screen_regime.py
"""
from data_loader import load_local_data
from data_fetcher import BybitDataFetcher
from backtest_engine import BacktestEngine
from optimization.regime_detector import add_regime_column
from optimization.screen_runner import run_screen

SYMBOL = "XRPUSDT"
YEAR = 2024
ROWS = 15000  # ~5 месяцев 15m-свечей, чтобы проверка не шла слишком долго

# 1. Загрузка данных
df = load_local_data(SYMBOL, YEAR)
print(f"Загружено: {SYMBOL} {YEAR}, строк: {len(df)}")
if len(df) > ROWS:
    df = df.iloc[:ROWS].reset_index(drop=True)
    print(f"Обрезано до первых {ROWS} строк для скорости")

# 2. Движок и однократный расчёт индикаторов
fetcher = BybitDataFetcher(testnet=True)
engine = BacktestEngine(fetcher)
df = engine._build_indicator_frame(df)

# 3. Санити-проверка детектора режима на этих данных
df_regime = add_regime_column(df.copy())
print("\nРаспределение режимов на данных:")
print(df_regime["regime"].value_counts(normalize=True).round(3).to_string())

# 4. Скрининг архетипов с разбивкой по режимам
report, results = run_screen(engine, df, SYMBOL, commission_pct=0.05, initial_capital=1000.0)
print("\n" + report)

# 5. Лучшие архетипы по режимам (структурированный результат Фазы 0)
print("\nКандидаты на сид по именам:")
for name, cand in (results.get("by_name") or {}).items():
    print(f"  {name:24s} regime={cand['regime']:5s} P&L={cand['return_pct']:+7.2f}%  PF={cand['pf']:.2f}")

print("\nЛучшие архетипы по режимам:")
for key in ("overall", "trend", "range"):
    best = results.get(key)
    if best:
        print(f"  {key:8s}: {best['name']:22s} P&L={best['return_pct']:+7.2f}%  PF={best['pf']:.2f}  trades={best['trades']}")
    else:
        print(f"  {key:8s}: нет кандидатов")