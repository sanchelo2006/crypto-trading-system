"""Fix formatting artifacts in PROJECT_OVERVIEW_RUS.md and PROJECT_OVERVIEW_ENG.md.

Run on Windows from the project root:
    python fix_docs.py
    # or, if python is not on PATH:
    py fix_docs.py

The script searches for the .md files in:
    1. Current working directory
    2. Directory containing fix_docs.py
    3. Parent of that directory (in case the script is in scripts/)

So it works whether you run it from the project root or from scripts/.

Backup: each file is copied to <name>.bak before modification.
Idempotent — running it twice is safe.
"""
import re
import shutil
import sys
from pathlib import Path

DIAGRAMS = {
    1: """```mermaid
stateDiagram-v2
    [*] --> Active
    Active --> Active: morning / intraday / trade
    Active --> Closed: close_day()
    Closed --> Archived: archive_day()
    Archived --> Active: open_new_day()<br/>(next day, auto)
    Closed --> Active: open_new_day()
```""",

    6: """```mermaid
flowchart TD
    A["Нажать Инициализировать"] --> B["Fetch morning package"]
    B --> C["ai_client.init_symbol"]
    C --> D{"Success?"}
    D -->|No| E["Notify error, abort"]
    D -->|Yes| F["run_init_with_sanity<br/>60 days backtest"]
    F --> G{"Defects found?"}
    G -->|No| H["Accept spec"]
    G -->|Yes| I["revise_init_strategy<br/>max 2 rounds"]
    I --> J["Rebuild spec"]
    J --> K["Re-backtest, pick best PF"]
    K --> H
    H --> L["process_ai_response"]
    L --> M["mark_initialized"]
    M --> N["open_new_day"]
    N --> O["Schedule UI refresh"]
```""",

    10: """```mermaid
stateDiagram-v2
    [*] --> Uninitialized: fresh coin
    Uninitialized --> Initialized: Инициализировать<br/>(web_search + sanity backtest)
    Initialized --> Active: Утренний пакет
    Active --> Active: Внутридневной / Авторежим
    Active --> HasTrade: Войти / Авторежим signal
    HasTrade --> Active: Закрыть позицию
    Active --> Closed: Закрыть день
    Closed --> Active: Открыть день
    HasTrade --> Closed: Закрыть день
    Active --> Uninitialized: Очистить
    Closed --> Uninitialized: Очистить
    HasTrade --> Uninitialized: Очистить
    Closed --> [*]
```""",

    13: """```mermaid
flowchart TD
    A["User message + context blocks"] --> B["POST to DeepSeek<br/>tools = get_tool_definitions<br/>tool_choice = auto"]
    B --> C{"Model returned tool_calls?"}
    C -->|Yes| D["Append assistant message<br/>with tool_calls"]
    D --> E["Execute each tool<br/>via ToolExecutor.execute"]
    E --> F["Append role=tool<br/>messages with results"]
    F --> G{"iteration < 5?"}
    G -->|Yes| B
    G -->|No| H["Return limit-reached message"]
    C -->|No| I["Return msg.content as final answer"]
    I --> J["Display in chat"]
```""",

    18: """```mermaid
sequenceDiagram
    actor U as User
    participant UI as ChatPage
    participant W as worker thread
    participant Q as UiEventQueue
    participant M as UI timer 2s

    U->>UI: Enter message
    UI->>UI: Disable input + send btn
    UI->>W: asyncio.to_thread(worker)
    W->>W: detect symbol, fetch package
    W->>W: chat_with_tools()
    W->>Q: queue_chat_message(reply)
    W->>Q: queue_chat_loading(False)
    M->>Q: process_chat_queue(chat_page)
    Q->>UI: Add chat bubble
    Q->>UI: Re-enable input + send btn
```""",

    21: """```mermaid
flowchart TD
    A["Load full history"] --> B{"Parquet cache?"}
    B -->|hit| C["Reuse cached df"]
    B -->|miss| D["Fetch + build indicators once"]
    C --> E["Baseline: buy&hold"]
    D --> E
    E --> F{"Phase 0 screen?"}
    F -->|yes| G["Screen 10 archetypes"]
    F -->|no| H["Use current spec from memory"]
    G --> I["Pick best or use seed"]
    I --> H
    H --> J["Split optimization / holdout"]
    J --> K["Split into N walk-forward folds"]
    K --> L["Iteration loop"]
    L --> M["Evaluate spec on TRAIN+TEST per fold"]
    M --> N["Aggregate + score + robust"]
    N --> O["Local search: 30 mutations"]
    O --> P{"Score dropped?"}
    P -->|yes| Q["Rollback to best"]
    P -->|no| R["Ask AI for new spec"]
    Q --> R
    R --> S{"AI gave valid+novel?"}
    S -->|no, 3 attempts| T["Programmatic mutation ×20"]
    S -->|yes| U["Enforce frozen / corridor"]
    T --> U
    U --> L
    L -->|max iters| V["Holdout gate"]
    V --> W{"Save allowed?"}
    W -->|yes| X["Save to memory"]
    W -->|no| Y["Reject"]
```""",

    22: """```mermaid
flowchart LR
    A["impulse_lab<br/>signal discovery"] --> B["portfolio_lab<br/>3-signal combo"]
    B --> C["exit_lab<br/>exit optimum"]
    C --> D["optimization/archetypes.py<br/>A5, A6"]
    D --> E["backtest_optimizer<br/>Phase 0 screen"]
    E --> F["strategy_spec in state.json"]
    F --> G["trading/auto_trader<br/>live execution"]
```""",

    23: """```mermaid
flowchart LR
    AT["auto_trader.py orchestrator"] --> SD["signal_detector"]
    AT --> PM["position_manager"]
    AT --> RM["risk_manager"]
    AT --> TL["trade_logger"]
    SD --> SPEC["strategy_spec evaluate_conditions"]
    PM --> EX["executor"]
    RM --> RS["risk_state.json"]
    TL --> FS["CSV + HTML logs/"]
    AT --> EX
```""",

    27: """```mermaid
sequenceDiagram
    participant C as 15m candle close
    participant AT as auto_trader
    participant SD as signal_detector
    participant EX as executor
    participant PM as position_manager
    participant TL as trade_logger

    C->>AT: new candle
    AT->>AT: _fetch_indicators (if 4min passed)
    AT->>SD: detect_signals(spec, current, prev)
    SD->>SD: filters then long then short
    SD->>AT: signals
    AT->>AT: _should_use_limit_entry
    AT->>EX: get_best_bid_ask / get_ticker_price
    AT->>EX: calculate_position_size
    alt limit entry
        AT->>PM: open_position_limit
        PM->>EX: place_limit_order PostOnly
        PM->>TL: log_event OPEN limit order
        loop each 15s
            AT->>EX: get_order_info
            EX->>AT: status
        end
        EX->>AT: Filled
        AT->>PM: finalize_limit_entry
    else market entry
        PM->>EX: place_market_order
        PM->>EX: get_order_info fill avgPrice
    end
    PM->>PM: _compute_tp_sl with caps
    PM->>EX: set_trading_stop
    PM->>TL: log_event OPEN
    loop Each closed candle
        AT->>PM: check_soft_exit
        alt soft exit
            PM->>EX: reduce_only market
            EX->>PM: exit_price reason
            PM->>AT: record_closed_trade
            AT->>TL: log_trade
            AT->>TL: log_event CLOSE
        else SL or TP hit by exchange
            AT->>EX: get_position_info
            EX->>AT: position gone
            AT->>EX: get_closed_pnl
            EX->>AT: PnL record
            AT->>AT: _infer_exit_reason
            AT->>AT: record_closed_trade
            AT->>TL: log_trade
            AT->>TL: log_event CLOSE
        end
    end
```""",
}

def find_file(name: str) -> Path | None:
    """Search for `name` in cwd, script dir, and script dir's parent."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / name,
        script_dir / name,
        script_dir.parent / name,
    ]
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand
    return None

def fix_file(path: Path) -> None:
    print(f"\n📄 {path}")
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"   💾 Backup: {backup.name}")

    original = path.read_text(encoding="utf-8")
    text = original
    orig_len = len(text)

    # 1. Restore broken diagrams
    def replace_diagram(match: re.Match) -> str:
        svg_id = int(match.group(1))
        return DIAGRAMS.get(svg_id, "")

    text = re.sub(
        r'This diagram is empty\.<svg id="mermaid-svg-(\d+)"[^>]*>.*?</svg>',
        replace_diagram,
        text,
        flags=re.DOTALL,
    )

    # 2. Safety net — remove remaining SVG dumps
    text = re.sub(r'<svg id="mermaid-svg-[^"]*"[^>]*>.*?</svg>', "", text, flags=re.DOTALL)

    # 3. Placeholder text
    text = text.replace("This diagram is empty.", "")
    text = re.sub(r'^Rendering diagram\.\.\.\s*$', "", text, flags=re.MULTILINE)

    # 4. ↕▾ header artifacts
    text = text.replace(" ↕▾ ↕▾", "")
    text = text.replace(" ↕▾", "")

    # 5. − prefix at start of table cells
    text = re.sub(r'^\| +−{1,3}', "| ", text, flags=re.MULTILINE)

    # 6. ⚙ rows and standalone ⚙
    text = re.sub(r'^\|\s*[−-]?⚙\s*\|.*$', "", text, flags=re.MULTILINE)
    text = re.sub(r'^⚙\s*$', "", text, flags=re.MULTILINE)

    # 7. Truncated Section 6 lifecycle flowchart
    text = text.replace(
        "    P --> Q[Detect close\n```",
        "    P --> Q[Detect close via Closed-PnL]\n    Q --> E\n```",
    )

    # 8a. Placeholder JSON (RU)
    text = re.sub(
        r'(\*\*плейсхолдер-сетап\*\* с:)\s*\n\s*```\s*\n\{\s*\n\s*"direction\s*\n```',
        (
            r'\1\n\n```json\n'
            '{\n'
            '  "direction": "long",\n'
            '  "entry_price": 0,\n'
            '  "stop_loss": 0,\n'
            '  "take_profit": [],\n'
            '  "confidence_pct": 0\n'
            '}\n'
            '```'
        ),
        text,
    )
    # 8b. Placeholder JSON (EN)
    text = re.sub(
        r'(\*\*placeholder setup\*\* with:)\s*\n\s*```\s*\n\{\s*\n\s*"direction\s*\n```',
        (
            r'\1\n\n```json\n'
            '{\n'
            '  "direction": "long",\n'
            '  "entry_price": 0,\n'
            '  "stop_loss": 0,\n'
            '  "take_profit": [],\n'
            '  "confidence_pct": 0\n'
            '}\n'
            '```'
        ),
        text,
    )

    # 9. Truncated data tree
    text = re.sub(
        r'(`data/auto_trading/`):\s*\n\s*```\s*\ndata\s*\n```',
        (
            r'\1\n\n```\n'
            'data/\n'
            '├── risk_state.json       # дневные и месячные лимиты убытков\n'
            '├── active_traders.json   # какие символы сейчас торгуют\n'
            '├── positions/            # по одному JSON на символ\n'
            '└── logs/                 # CSV и HTML логи, ротация по неделям\n'
            '```'
        ),
        text,
    )

    # 10. Collapse excessive blank lines
    text = re.sub(r'\n{4,}', "\n\n\n", text)

    # 11. Single trailing newline
    text = text.rstrip() + "\n"

    # 12. Remaining issues
    remaining = []
    if "↕▾" in text:
        remaining.append(f"↕▾ × {text.count('↕▾')}")
    if re.search(r'^⚙\s*$', text, re.MULTILINE):
        remaining.append(f"⚙ × {len(re.findall(r'^⚙\\s*$', text, re.MULTILINE))}")
    if "<svg" in text:
        remaining.append(f"<svg × {text.count('<svg')}")
    if "This diagram is empty." in text:
        remaining.append("This diagram is empty.")
    if "Rendering diagram..." in text:
        remaining.append("Rendering diagram...")

    if text == original:
        print("   ⚠️  No changes made (already clean?)")
        return

    path.write_text(text, encoding="utf-8")
    delta = orig_len - len(text)
    print(f"   ✅ {orig_len} → {len(text)} chars ({delta:+d})")
    if remaining:
        print(f"   ⚠️  Remaining issues: {', '.join(remaining)}")
    else:
        print("   ✅ Clean")

def main() -> int:
    names = ["PROJECT_OVERVIEW_RUS.md", "PROJECT_OVERVIEW_ENG.md"]

    fixed = 0
    missing = []
    for name in names:
        path = find_file(name)
        if path is None:
            missing.append(name)
            print(f"\n⏭️  {name}: not found in cwd, script dir, or parent")
            continue
        fix_file(path)
        fixed += 1

    print(f"\n{'=' * 60}")
    print(f"✅ Fixed: {fixed} | Missing: {len(missing)}")
    if missing:
        print(f"\n❌ Missing files:")
        for m in missing:
            print(f"   - {m}")
        print(f"\n💡 Current directory: {Path.cwd()}")
        print(f"   Files in this directory:")
        try:
            for entry in sorted(Path.cwd().iterdir()):
                if entry.suffix == ".md":
                    print(f"   • {entry.name}")
        except Exception as e:
            print(f"   (error listing: {e})")
    if fixed:
        print("\n💾 Backups saved with .bak suffix — delete when satisfied.")
    print("=" * 60)
    return 0 if not missing else 1

if __name__ == "__main__":
    sys.exit(main())
