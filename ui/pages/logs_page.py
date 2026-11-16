"""Logs page — browse, view, download and delete project log files.

Log sources covered:
  1. Application logs        -> get_logs_dir()  (app.log, errors.log, user.log, ...)
  2. Auto-trading logs       -> data/auto_trading/logs/
  3. Optimization reports    -> optimization_reports/

Large files are previewed with only the last N lines; the full file can be
downloaded. Deleting a file that is still open by the running app (e.g.
app.log on Windows) is handled with a friendly warning.
"""
from __future__ import annotations
import html
import os
from datetime import datetime
from pathlib import Path
from typing import List
import json as _json

from nicegui import ui

import config
from utils.logger_setup import get_logs_dir

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
TAIL_LINES = 500                 # how many last lines to show in the preview
PREVIEW_READ_LIMIT = 1_000_000   # max bytes read for the tail preview
MAX_INLINE_PREVIEW = 200_000     # safety cap for characters rendered in UI

# Category definitions: label -> directory resolver
CATEGORIES = [
    {
        'id': 'app_logs',
        'label': 'App Logs',
        'icon': 'description',
        'get_dir': lambda: get_logs_dir(),
    },
    {
        'id': 'auto_trading',
        'label': 'Auto-trading Logs',
        'icon': 'smart_toy',
        'get_dir': lambda: Path(config.DATA_DIR) / 'auto_trading' / 'logs',
    },
    {
        'id': 'optimization',
        'label': 'Optimization Reports',
        'icon': 'science',
        'get_dir': lambda: Path(config.BASE_DIR) / 'optimization_reports',
    },
]


def _human_size(num: float) -> str:
    """Format a byte count as a human readable string."""
    for unit in ('B', 'KB', 'MB', 'GB'):
        if num < 1024.0:
            return f"{num:.1f} {unit}" if unit != 'B' else f"{int(num)} B"
        num /= 1024.0
    return f"{num:.1f} TB"


def _tail_file(path: Path, n_lines: int = TAIL_LINES) -> str:
    """Return the last ``n_lines`` lines of a file (best-effort, no huge reads)."""
    try:
        size = path.stat().st_size
        with path.open('rb') as f:
            # Seek to a window near the end to avoid reading a huge file fully.
            start = max(0, size - PREVIEW_READ_LIMIT)
            f.seek(start)
            data = f.read(PREVIEW_READ_LIMIT)
        text = data.decode('utf-8', errors='replace')
        lines = text.splitlines()
        tail = lines[-n_lines:]
        omitted = len(lines) - len(tail)
        header = ''
        if omitted > 0 or start > 0:
            header = f"... showing last {len(tail)} lines " \
                     f"(full file: {_human_size(size)} — use Download) ...\n\n"
        return header + '\n'.join(tail)
    except Exception as exc:
        return f"[could not read file: {exc}]"


class LogsPage:
    """Page listing all project log files with view/download/delete actions."""

    def __init__(self):
        self.container = ui.column().classes('w-full h-full p-4 gap-4')
        self._file_rows: dict = {}   # category id -> ui.column holding the rows
        self._build()

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------
    def _build(self):
        with self.container:
            ui.label('Logs').classes('text-2xl font-bold')
            ui.label(
                'Browse, preview, download or delete log files. '
                'Large files show only the last lines — download for the full content.'
            ).classes('text-sm text-gray-500')

            ui.tabs().classes('w-full')
            with ui.tabs() as tabs:
                cat_tabs = []
                for cat in CATEGORIES:
                    t = ui.tab(cat['id'], label=cat['label'], icon=cat['icon'])
                    cat_tabs.append((cat, t))

            with ui.tab_panels(tabs, value=cat_tabs[0][1]).classes('w-full'):
                for cat, t in cat_tabs:
                    with ui.tab_panel(t):
                        self._build_category(cat)

    def _build_category(self, cat: dict):
        with ui.row().classes('w-full items-center justify-between mb-2'):
            with ui.row().classes('items-center gap-2'):
                ui.icon(cat['icon']).classes('text-lg')
                ui.label(cat['label']).classes('text-lg font-semibold')
                # directory path shown small
                try:
                    dir_path = cat['get_dir']()
                    ui.label(str(dir_path)).classes('text-xs text-gray-400')
                except Exception:
                    pass
            with ui.row().classes('gap-2'):
                ui.button(icon='refresh', on_click=lambda c=cat: self._refresh_category(c)) \
                    .props('flat round dense').tooltip('Refresh list')
                ui.button(icon='delete_sweep', on_click=lambda c=cat: self._delete_all(c)) \
                    .props('flat round dense color=red').tooltip('Delete ALL files in this category')

        # header row
        with ui.row().classes('w-full text-xs text-gray-400 border-b pb-1'):
            ui.label('Name').classes('w-6/12')
            ui.label('Size').classes('w-2/12 text-right')
            ui.label('Modified').classes('w-2/12 text-right')
            ui.label('Actions').classes('w-2/12 text-right')

        rows_container = ui.column().classes('w-full gap-0')
        self._file_rows[cat['id']] = rows_container
        self._refresh_category(cat)

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------
    def _list_files(self, cat: dict) -> List[Path]:
        d = cat['get_dir']()
        if not d.exists() or not d.is_dir():
            return []
        files = [p for p in d.iterdir() if p.is_file()]
        # newest first
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return files

    def _refresh_category(self, cat: dict):
        rows = self._file_rows.get(cat['id'])
        if rows is None:
            return
        rows.clear()
        with rows:
            files = self._list_files(cat)
            if not files:
                ui.label('No files.').classes('text-gray-400 italic py-2')
                return
            for path in files:
                self._render_file_row(cat, path, rows)

    def _render_file_row(self, cat: dict, path: Path, rows):
        try:
            st = path.stat()
            size_txt = _human_size(st.st_size)
            mtime_txt = datetime.fromtimestamp(st.st_mtime).strftime('%Y-%m-%d %H:%M')
        except Exception:
            size_txt, mtime_txt = '?', '?'

        with ui.row().classes('w-full items-center border-b py-1 hover:bg-gray-50'):
            ui.label(path.name).classes('w-6/12 truncate').tooltip(str(path))
            ui.label(size_txt).classes('w-2/12 text-right')
            ui.label(mtime_txt).classes('w-2/12 text-right')
            with ui.row().classes('w-2/12 justify-end gap-1'):
                ui.button(icon='visibility',
                          on_click=lambda p=path: self._view_file(p)) \
                    .props('flat round dense').tooltip('Preview last lines')
                ui.button(icon='download',
                          on_click=lambda p=path: self._download_file(p)) \
                    .props('flat round dense').tooltip('Download file')
                ui.button(icon='delete',
                          on_click=lambda p=path, c=cat: self._confirm_delete(p, c)) \
                    .props('flat round dense color=red').tooltip('Delete file')

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _view_file(self, path: Path):
        # HTML files: open as rendered HTML in a new browser tab,
        # keep a small dialog with the Download button.
        if path.suffix.lower() == '.html':
            self._open_html_in_new_tab(path)
            return

        # All other files: show a tail preview in a dialog.
        text = _tail_file(path)
        if len(text) > MAX_INLINE_PREVIEW:
            text = text[:MAX_INLINE_PREVIEW] + '\n\n[preview truncated]'
        escaped = html.escape(text)
        with ui.dialog() as dlg, ui.card().classes('w-[80vw]'):
            with ui.row().classes('w-full items-center justify-between'):
                ui.label(path.name).classes('font-semibold')
                ui.button(icon='close', on_click=dlg.close).props('flat round dense')
            ui.html(
                '<pre style="white-space: pre-wrap; word-break: break-word; '
                'overflow: auto; max-height: 65vh; font-size: 12px; '
                'background: #f5f5f5; padding: 8px; border-radius: 4px;">'
                + escaped + '</pre>'
            ).classes('w-full')
            with ui.row().classes('w-full justify-end'):
                ui.button('Download', icon='download',
                          on_click=lambda p=path: self._download_file(p))
        dlg.open()

    def _open_html_in_new_tab(self, path: Path):
        """Opens an HTML log file as rendered HTML in a new browser tab.

        The file content is passed to the browser via run_javascript, where
        it is turned into a Blob URL and opened in a new tab. The Download
        button stays available in the file list row.
        """
        try:
            content = path.read_text(encoding='utf-8')
        except Exception as exc:
            ui.notify(f'Cannot read file: {exc}', type='negative')
            return

        # Escape the content for safe embedding into a JS string literal.
        js_content = _json.dumps(content)
        js_filename = _json.dumps(path.name)

        js_code = f"""
        (function() {{
            var html = {js_content};
            var filename = {js_filename};
            var blob = new Blob([html], {{type: 'text/html; charset=utf-8'}});
            var url = URL.createObjectURL(blob);
            var w = window.open(url, '_blank');
            if (!w) {{
                // Popup-blocker fallback: trigger a download instead.
                var a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            }}
            // Revoke the Blob URL after the new tab has had time to load.
            setTimeout(function() {{ URL.revokeObjectURL(url); }}, 60000);
        }})();
        """
        ui.run_javascript(js_code)

    def _download_file(self, path: Path):
        try:
            ui.download(str(path), filename=path.name)
        except Exception as exc:
            ui.notify(f'Download failed: {exc}', type='negative')

    def _confirm_delete(self, path: Path, cat: dict):
        with ui.dialog() as dlg, ui.card():
            ui.label(f'Delete "{path.name}"?').classes('text-lg font-semibold')
            ui.label('This cannot be undone.').classes('text-sm text-gray-500')
            with ui.row().classes('w-full justify-end gap-2 mt-2'):
                ui.button('Cancel', on_click=dlg.close).props('flat')
                ui.button('Delete', color='red',
                          on_click=lambda: (dlg.close(), self._delete_file(path, cat)))
        dlg.open()

    def _delete_file(self, path: Path, cat: dict):
        try:
            os.remove(path)
            ui.notify(f'Deleted {path.name}', type='positive')
        except PermissionError:
            ui.notify(
                f'"{path.name}" is in use by the running app and cannot be deleted. '
                f'Stop the application and try again.',
                type='warning', multi_line=True,
            )
        except Exception as exc:
            ui.notify(f'Delete failed: {exc}', type='negative')
        self._refresh_category(cat)

    def _delete_all(self, cat: dict):
        files = self._list_files(cat)
        if not files:
            ui.notify('Nothing to delete.', type='info')
            return
        with ui.dialog() as dlg, ui.card():
            ui.label(f'Delete ALL {len(files)} file(s) in "{cat["label"]}"?') \
                .classes('text-lg font-semibold')
            ui.label('This cannot be undone.').classes('text-sm text-gray-500')
            with ui.row().classes('w-full justify-end gap-2 mt-2'):
                ui.button('Cancel', on_click=dlg.close).props('flat')
                ui.button('Delete all', color='red',
                          on_click=lambda: (dlg.close(), self._delete_all_now(files, cat)))
        dlg.open()

    def _delete_all_now(self, files: List[Path], cat: dict):
        deleted, busy = 0, 0
        for p in files:
            try:
                os.remove(p)
                deleted += 1
            except PermissionError:
                busy += 1
            except Exception:
                busy += 1
        msg = f'Deleted {deleted} file(s).'
        if busy:
            msg += f' {busy} file(s) are in use and were skipped.'
            ui.notify(msg, type='warning', multi_line=True)
        else:
            ui.notify(msg, type='positive')
        self._refresh_category(cat)