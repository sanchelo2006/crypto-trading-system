"""
Общий conftest для всех тестов.

Гарантирует, что корень проекта в sys.path, чтобы `import config`,
`import trading...` работали при запуске pytest из любого каталога.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))