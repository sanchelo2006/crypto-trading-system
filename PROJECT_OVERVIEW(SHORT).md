Crypto Trading System — Short Description
A Python system for trading Bybit linear perpetuals built on one principle: the LLM proposes, the engine disposes. DeepSeek studies the market and writes a strategy as strict JSON; a deterministic engine is the only thing that ever decides whether a trade happens.

strategy_spec — the executable contract
strategy_spec.py is the boundary between reasoning and execution. A spec declares, per side (long / short):

entry / entry_any	Lists of AND-groups, OR-combined across groups
stop_loss_atr_mult / take_profit_atr_mult	ATR-based risk, with % fallbacks
management	Breakeven trigger, ATR trailing, max hold, optional soft exits (level failure, trend loss, MACD flip)
filters	Global conditions applied to both sides
risk_controls	Cooldown after a losing streak / N losses in a window
Conditions are built from six primitives — compare, cross, range, volume, slope, distance — evaluated against a fixed indicator frame (SMA 20/50, RSI 14, ATR 14, ADX, MACD, Bollinger, daily-reset VWAP, previous-day pivots, plus 1h and 4h layers).

Three properties make it machine-executable rather than descriptive:

Validation is opinionated. Every entry group must contain a volume filter, a trend filter, and a level-or-momentum confirmation, plus a stop and a take profit. A spec missing them is rejected and replaced by the default — never silently half-applied.

The same evaluator runs everywhere. backtest_engine._entry_signal, trading/signal_detector._check_entry, and the lab simulator share identical semantics. A strategy that backtests with 100 trades produces those same signals live; any drift would invalidate every backtest.

Engine defaults close the gaps. ensure_engine_defaults() fills whatever the model omitted, so a partially-specified spec still executes deterministically rather than crashing.

No LLM is called at execution time. Same spec + same data = same trades, every run.

Scenario 1 — Coin initialization: monitoring and signal generation
Init. ai_client.init_symbol runs up to 5 iterations in which the model may call web_search to research the coin, then is forced to return JSON. The result is not trusted immediately: run_init_with_sanity replays it over the last 60 days of 15m history and checks three diagnostics — trade frequency below 1.5/week, PF < 1.0 with negative expectancy, or over 50% of losses closing within 3 candles. Any trigger sends a specific remediation checklist to revise_init_strategy for up to 2 rounds, keeping the best profit factor. The accepted spec, rules and research are written to data/{SYMBOL}/state.json, is_initialized flips, a new UTC day opens.

Monitoring. data_fetcher assembles three package tiers — morning (wide), intraday (30 candles), intraday-light (2 candles) — plus precomputed market_stats (adaptive ADX threshold, BTC correlation with cache, OI delta, funding, Fear & Greed). The model returns a setup (conditional watch) or a signal (actionable, confidence_pct > 0), each carrying explicit entry and cancellation conditions. AutoMode's loop wakes on every closed 15m candle and uses the light package when a thesis already exists — the token-economy decision.

Execution path. Signals are advisory to the human; real orders come from trading/, gated by two independent flags (BYBIT_TESTNET for orders, BYBIT_DATA_TESTNET for candles). Live risk logic mirrors the backtest line-for-line: same risk_controls, same taker-fee subtraction, same next-candle signal timing.

Scenario 2 — Strategy optimization into coin memory
Two entry points share one engine: backtest_engine.run_on_dataframe() is the objective function, optimization/backtest_optimizer.py is the search.

Path	Trigger	Character
Optimization	♻️ Запустить оптимизацию on the Backtest page	Automated, gated, saves to memory
Laboratory	Lab page + 16 CLI scripts	Manual research, never writes to memory
The optimizer's cycle:

Baseline — buy & hold, so every candidate is measured against simply holding the coin.

Phase 0 (optional) — screens 10 hardcoded archetypes (6 trend/momentum, 4 mean-reversion), each with 2–3 tunable knobs. Structure is fixed deliberately: the idea can't be rewritten, only its risk fitted to volatility.

Data once — full history fetched, indicators computed once, then sliced for every fold of every iteration. Parquet cache is only trusted for closed past periods.

Splits — last 15% reserved as holdout (min 500 candles) that iteration never sees; the rest divided into N walk-forward folds with expanding TRAIN and sequential TEST windows.

Per iteration — evaluate → aggregate → score (test expectancy/PF/return weighted above train, minus drawdown, train-test divergence and profit-concentration penalties, plus a trade-count bonus that punishes fewer than 20 trades with −5.0) → acceptable flag (≥10 test trades, positive expectancy both splits, PF > 1 both) → window robustness (≥2/3 of windows profitable, positive-expectancy and PF-passing; worst ≥ −5%; stability ≥ 0.5) → soft penalty → 30 local numeric mutations → rollback to best if the score regressed → AI proposal (3 attempts, validated, engine controls inherited, frozen_entries and exit_corridor enforced rather than requested, fingerprint-deduplicated) → 20 programmatic mutations as fallback.

Holdout gate — passed needs ≥10 trades, expectancy > 0, PF > 1.05, positive return; 5–9 trades is weak_pass and blocks saving unless the operator explicitly allowed it; failed blocks outright.

Save gate — only a winner that cleared all six gates is written into state.json and trading_rules.md. Otherwise the coin keeps its old spec.

Every iteration is streamed to a text report and per-iteration CSVs, so a crash mid-run leaves usable diagnostics.

Scenario 3 — Laboratory
The research workshop, where nothing reaches production because it looked good on one chart.

UI Lab page — portfolio simulation. Pick strategies from an 8-entry registry (m2s2_vol2.0, act_short, climax_bull, …), assign capital share / position size / leverage per symbol, set shared SL/TP/hold, run. Each symbol simulates independently; the portfolio curve is the sum of per-symbol capital. Outputs per-symbol metrics (exp_mean exposed separately as signal quality before sizing), year-by-year breakdown, an ECharts equity curve, and a full text report with a trade-by-trade log in research/lab_runs/.

CLI labs — 16 scripts sharing impulse_lab._simulate_fixed:

Discovery: impulse_lab (40+ engineered features, an oracle labelling "real" impulses, 20+ candidate signals), swing_lab (1h/4h), positioning (fund_z, oi_d4, ls_z via merge_asof with MAX_AGE staleness checks — no lookahead).

Exit tuning: exit_lab freezes the entry and varies only exits across 145 combinations; swing_exit_lab tests 480 on 4H with a train/test buffer.

Strategy families: grid feasibility → grid simulation with ER hysteresis regime switching; funding arbitrage in three generations (ideal → OU basis risk with Monte Carlo → reload timing around payment); pairs arbitrage with Engle-Granger cointegration and walk-forward z-score trading.

Portfolio & stress: three-signal combination tested at 0.0 → 0.10% slippage, optimal-trio search over 20 combinations of 6 alts, standalone slippage stress with and without exit variation.

The traceability rule. Lab findings are not folklore — they become production constants. m2s2_vol2.0 became the A6_SHORT_MOMENTUM archetype. The exit optimum sl=2.0 / tp=1.5 / hold=8 became A6's exit_corridor ([1.8, 2.2] / [1.3, 1.7] / [6, 12]). The finding that trailing and breakeven hurt that signal is why the optimizer force-nulls them. low_32, high_32, drop_2_atr and activity moved from lab features into the production indicator frame. When a strategy starts failing live, the traceback runs archetype → lab iteration → re-run with new data.

Design invariants across all three
Zero lookahead — signals on the close of bar *i*, fills at the open of *i+1*; every rolling indicator shifted; pivots from yesterday's OHLC; hourly features merged without dragging the 15m close along.

Honest costs — 0.055% commission per side, 0.05% slippage against the trade on entry and on SL/TP fills, deliberately none on soft exits (they execute at an observed close).

Deterministic replay — the engine is pure pandas over a DataFrame it already holds; identical inputs give identical trades.

Diagnosable trades — every trade carries entry_reasons (per-condition pass/fail with actual values) and entry_snapshot (RSI, ADX, ATR, pivots at entry), which is exactly the data the optimizer consumes.

Reasoning never touches money — the chat can write rules and sketches (confidence_pct = 0), the analysis pipeline can raise confidence, and only trading/ can place an order.