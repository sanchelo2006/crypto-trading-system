#!/usr/bin/env python3
"""
Fix PROJECT_OVERVIEW.md — replace broken inline SVG dumps with proper mermaid
code fences, and simplify mermaid blocks that hang certain renderers.

Usage:
    cd C:\\Users\\sanch\\Projects\\TRADING_AGENT_4
    python fix_overview.py                    # fix ./PROJECT_OVERVIEW.md
    python fix_overview.py path/to/file.md    # or pass explicit path

Behavior:
    - Creates a .bak backup before touching anything.
    - Idempotent: running twice is safe.
    - Preserves original line endings (CRLF on Windows).
"""

import re
import sys
from pathlib import Path


# ---------------------------------------------------------------- replacement mermaid code

DAY_LIFECYCLE = """stateDiagram-v2
    [*] --> Active
    Active --> Active: morning / intraday / trade
    Active --> Closed: close_day()
    Closed --> Archived: archive_day()
    Archived --> Active: open_new_day()<br/>(next day, auto)
    Closed --> Active: open_new_day()"""

INITIALIZE = """flowchart TD
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
    N --> O["Schedule UI refresh"]"""

REASONING_STATE = """stateDiagram-v2
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
    Closed --> [*]"""

TOOL_CALLING_LOOP = """flowchart TD
    A["User message + context blocks"] --> B["POST to DeepSeek<br/>tools = get_tool_definitions<br/>tool_choice = auto"]
    B --> C{"Model returned tool_calls?"}
    C -->|Yes| D["Append assistant message<br/>with tool_calls"]
    D --> E["Execute each tool<br/>via ToolExecutor.execute"]
    E --> F["Append role=tool<br/>messages with results"]
    F --> G{"iteration < 5?"}
    G -->|Yes| B
    G -->|No| H["Return limit-reached message"]
    C -->|No| I["Return msg.content as final answer"]
    I --> J["Display in chat"]"""

THREADING_UI = """sequenceDiagram
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
    Q->>UI: Re-enable input + send btn"""

OPTIMIZATION_LOOP = """flowchart TD
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
    W -->|no| Y["Reject"]"""

SIX_MODULES = """flowchart LR
    AT["auto_trader.py orchestrator"] --> SD["signal_detector"]
    AT --> PM["position_manager"]
    AT --> RM["risk_manager"]
    AT --> TL["trade_logger"]
    SD --> SPEC["strategy_spec evaluate_conditions"]
    PM --> EX["executor"]
    RM --> RS["risk_state.json"]
    TL --> FS["CSV + HTML logs/"]
    AT --> EX"""

DAILY_LIFE = """sequenceDiagram
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
    end"""

# Section header text → replacement mermaid code (matched by substring, emoji ignored)
SECTION_MAP = [
    ("Day lifecycle",              DAY_LIFECYCLE),
    ("Initialize",                 INITIALIZE),
    ("reasoning state machine",    REASONING_STATE),
    ("tool-calling loop",          TOOL_CALLING_LOOP),
    ("Threading and UI",           THREADING_UI),
    ("optimization loop",          OPTIMIZATION_LOOP),
    ("six modules",                SIX_MODULES),
    ("daily life of a live trade", DAILY_LIFE),
]

# Simplified replacements for existing (but fragile) mermaid blocks
HOW_IT_WORKS = """flowchart LR
    A["Bybit REST API"] --> B["data_fetcher.py"]
    B --> C["indicators.py<br/>TA-Lib 1h / 4h"]
    B --> D["market_stats<br/>ADX, BTC corr, OI, funding"]
    C --> E["ai_client.py<br/>DeepSeek"]
    D --> E
    E --> F["strategy_spec.json"]
    F --> G["backtest_engine.py"]
    F --> H["trading/<br/>live execution"]
    G --> I["optimization loop"]
    I --> E
    H --> J["NiceGUI dashboard"]

    style A fill:#F7A600,color:#000
    style E fill:#4D6BFE,color:#fff
    style F fill:#2E7D32,color:#fff
    style G fill:#7B1FA2,color:#fff
    style H fill:#C62828,color:#fff"""

PAGE_LAYOUT = """flowchart TB
    subgraph HEADER["Header row"]
        H1["SOLUSDT"]
        H2["ADX порог = 18.4"]
        H3["Инициализировать"]
        H4["Авторежим"]
        H5["Обновить"]
        H6["Очистить"]
    end
    subgraph BATCH["Batch row"]
        B1["Утренний пакет"]
        B2["Внутридневной пакет"]
        B3["Закрыть день"]
        B4["Открыть день"]
    end
    subgraph TABS["Tabs"]
        T1["Анализ"]
        T2["Правила"]
        T3["Архив дня"]
    end
    HEADER --> BATCH
    BATCH --> TABS
    style H1 fill:#7B1FA2,color:#fff
    style H3 fill:#1976D2,color:#fff
    style H4 fill:#388E3C,color:#fff
    style B1 fill:#0288D1,color:#fff
    style B2 fill:#0288D1,color:#fff
    style B3 fill:#F57C00,color:#fff
    style B4 fill:#388E3C,color:#fff"""


# ---------------------------------------------------------------- core fix logic

def replace_svg_dumps(content: str) -> tuple[str, int]:
    """
    For each section header in SECTION_MAP:
      - locate the header (### ... <keyword>)
      - find the first 'This diagram is empty.<svg' AFTER that header
      - replace everything from there up to and including '</svg>'
        with a ```mermaid fence + reconstructed code
    """
    count = 0
    for header_text, mermaid_code in SECTION_MAP:
        header_re = re.compile(rf"###[^\n]*{re.escape(header_text)}")
        header_match = header_re.search(content)
        if not header_match:
            print(f"  [skip] header not found: {header_text}")
            continue

        svg_marker = re.compile(r"This diagram is empty\.<svg")
        marker_match = svg_marker.search(content, header_match.end())
        if not marker_match:
            print(f"  [skip] no SVG dump after: {header_text}")
            continue

        svg_start = marker_match.start()
        svg_close = content.find("</svg>", svg_start)
        if svg_close == -1:
            print(f"  [warn] no </svg> found for: {header_text}")
            continue
        svg_end = svg_close + len("</svg>")

        replacement = f"\n```mermaid\n{mermaid_code.strip()}\n```\n"
        content = content[:svg_start] + replacement + content[svg_end:]
        count += 1
        print(f"  [fixed] {header_text:<32} (removed {svg_end - svg_start:,} chars of SVG)")
    return content, count


def replace_mermaid_block(content: str, header_text: str, new_code: str) -> tuple[str, bool]:
    """Find `### ... <header_text>` section, replace its first ```mermaid fence."""
    header_re = re.compile(rf"###[^\n]*{re.escape(header_text)}[^\n]*\n")
    header_match = header_re.search(content)
    if not header_match:
        return content, False

    block_re = re.compile(r"```mermaid\n.*?\n```", re.DOTALL)
    block_match = block_re.search(content, header_match.end())
    if not block_match:
        return content, False

    replacement = f"```mermaid\n{new_code.strip()}\n```"
    content = content[:block_match.start()] + replacement + content[block_match.end():]
    return content, True


# ---------------------------------------------------------------- entry point

def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = Path(args[0]) if args else Path("PROJECT_OVERVIEW.md")

    if not path.exists():
        print(f"ERROR: file not found: {path.resolve()}")
        print(f"Hint: cd into the project root first, or pass the path explicitly.")
        sys.exit(1)

    print(f"Reading:  {path.resolve()}")

    raw = path.read_bytes()
    line_ending = "\r\n" if b"\r\n" in raw[:4096] else "\n"
    content = raw.decode("utf-8").replace("\r\n", "\n")
    original_len = len(content)
    original_lines = content.count("\n") + 1

    backup = path.with_suffix(path.suffix + ".bak")
    backup.write_bytes(raw)
    print(f"Backup:   {backup.resolve()}")

    # ---- 1) replace the 8 broken SVG dumps ----
    print("\n[1/3] Replacing broken inline SVG dumps with mermaid...")
    content, svg_fixed = replace_svg_dumps(content)
    print(f"      Fixed: {svg_fixed} sections")

    # ---- 2) simplify "How it works" (hanging renderer) ----
    print("\n[2/3] Simplifying 'How it works' mermaid block...")
    content, ok = replace_mermaid_block(content, "How it works", HOW_IT_WORKS)
    print(f"      {'fixed' if ok else 'NOT FOUND — skipped'}")

    # ---- 3) simplify "Page layout" ----
    print("\n[3/3] Simplifying 'Page layout' mermaid block...")
    content, ok = replace_mermaid_block(content, "Page layout", PAGE_LAYOUT)
    print(f"      {'fixed' if ok else 'NOT FOUND — skipped'}")

    # ---- write back with original line endings ----
    out = content.replace("\n", line_ending)
    path.write_bytes(out.encode("utf-8"))

    new_lines = content.count("\n") + 1
    print(f"\nDone.")
    print(f"  {original_len:,} chars / {original_lines:,} lines")
    print(f"   -> {len(content):,} chars / {new_lines:,} lines")
    print(f"  {line_ending!r} line endings preserved")
    print(f"  Original backed up to: {backup.name}")


if __name__ == "__main__":
    main()