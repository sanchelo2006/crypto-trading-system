"""
Temporary translation check script.

Scans the project's .py files for remaining Cyrillic text and prints
"file:line: text" plus a per-file summary. The report is also saved
to check_report.txt.

Run from the project root:
    python check.py

Excluded on purpose:
- tests/ (kept in Russian);
- the research lab and its diagnostic scripts (will be deleted);
- legacy Russian strings kept for backward compatibility.
"""
from __future__ import annotations

import re
from pathlib import Path

# Any Cyrillic character, including Ё/ё.
CYRILLIC = re.compile(r"[\u0400-\u04FF]")

SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules",
    ".idea", ".vscode", ".pytest_cache",
    "research", "tests",
}
SKIP_FILES = {
    "check.py",
    "check_impulse_lab.py",
    "check_regime_quality.py",
    "check_screen_regime.py",
}
SKIP_NAME_PARTS = ("research_lab",)

# Russian text kept on purpose:
# - legacy marker in old trade_journal.csv rows;
# - optional "на " group in the timeframe regex (old Russian rules).
EXPECTED_SUBSTRINGS = ("Не было ошибок", r"на\s+")

ROOT = Path(".")


def is_skipped(path: Path) -> bool:
    if path.name in SKIP_FILES:
        return True
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    if any(part in path.name for part in SKIP_NAME_PARTS):
        return True
    return False


def main() -> None:
    per_file: dict[Path, int] = {}
    details: list[str] = []
    total = 0
    legacy = 0

    for path in sorted(ROOT.rglob("*.py")):
        if is_skipped(path):
            continue

        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError as exc:
            print(f"!! cannot read {path}: {exc}")
            continue

        hits = [
            (number, line.strip())
            for number, line in enumerate(lines, 1)
            if CYRILLIC.search(line)
        ]
        if not hits:
            continue

        count = 0
        details.append(f"\n=== {path} ===")
        for number, line in hits:
            if any(marker in line for marker in EXPECTED_SUBSTRINGS):
                legacy += 1
                details.append(f"[legacy] {path}:{number}: {line}")
            else:
                count += 1
                total += 1
                details.append(f"{path}:{number}: {line}")

        if count:
            per_file[path] = count

    summary = [
        "TRANSLATION CHECK",
        f"files to translate: {len(per_file)}",
        f"lines to translate: {total}",
        f"legacy lines kept on purpose: {legacy}",
        "",
        "per-file counts:",
    ]
    for path, count in sorted(per_file.items(), key=lambda item: -item[1]):
        summary.append(f"  {count:5d}  {path}")

    text = "\n".join(summary + details)
    print(text)
    Path("check_report.txt").write_text(text, encoding="utf-8")
    print("\nSaved to check_report.txt")


if __name__ == "__main__":
    main()