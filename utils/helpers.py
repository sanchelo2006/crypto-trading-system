"""
Helper functions: formatting data as CSV, date handling, logging.
"""
import pandas as pd
import logging
from datetime import datetime, timezone
from io import StringIO
from typing import Optional, List, Dict, Any
import json
import re
from json_repair import repair_json 

logger = logging.getLogger(__name__)


def df_to_csv_string(df: pd.DataFrame, columns: Optional[List[str]] = None) -> str:
    if df.empty:
        return ""
    if columns:
        df = df[[c for c in columns if c in df.columns]]
    return df.to_csv(index=False)


def csv_string_to_df(csv_str: str) -> pd.DataFrame:
    if not csv_str.strip():
        return pd.DataFrame()
    return pd.read_csv(StringIO(csv_str))


def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Robust JSON parsing of an AI response.
    Order:
    1) direct json.loads
    2) markdown ```json
    3) slice between the first { and the last }
    4) slice by balanced curly braces
    5) attempt to fix strings and strip trailing garbage
    """
    if not text:
        return None

    candidates = []

    # 1. Original text
    candidates.append(text.strip())

    # 2. Markdown block 
    if "```json" in text:
        try:
            content = text.split("```json", 1)[1].split("```", 1)[0].strip()
            if content:
                candidates.append(content)
        except Exception:
            pass

    # 3. Between the first { and the last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start:end + 1].strip())

    	
# 4. By balanced braces
    balanced = _extract_balanced_json_object(text)
    if balanced:
        candidates.append(balanced.strip())

    seen = set()
    unique_candidates = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            unique_candidates.append(c)

    for candidate in unique_candidates:
        # 4.1 direct parsing
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # 4.2_1 repair_json
        try:
            repaired = repair_json(candidate, return_objects=False)
            return json.loads(repaired)
        except Exception:
            pass

        # 4.2 string repair
        try:
            fixed = _fix_json_strings(candidate)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        # 4.3 remove trailing commas
        try:
            fixed = _remove_trailing_commas(_fix_json_strings(candidate))
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        # 4.4 rough normalization of special characters
        try:
            fixed = candidate.replace('\r\n', '\\n').replace('\n', '\\n').replace('\t', '\\t')
            fixed = _remove_trailing_commas(fixed)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    logger.error(
            f"All JSON parsing methods failed. "
            f"First 500 characters of the response: {text[:500]}..."
        )
    return None

def _extract_balanced_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape_next:
            escape_next = False
            continue

        if ch == '\\':
            escape_next = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


def _remove_trailing_commas(text: str) -> str:
    return json.dumps(json.loads(re.sub(r',(\s*[}\]])', r'\1', text)), ensure_ascii=False)


def _fix_json_strings(json_str: str) -> str:
    """
    Fixes unescaped characters inside JSON string values.
    Escapes newlines, carriage returns, tabs and unescaped double quotes.
    """
    result = []
    in_string = False
    escape_next = False
    i = 0
    n = len(json_str)
    
    while i < n:
        ch = json_str[i]
        
        if escape_next:
            result.append(ch)
            escape_next = False
            i += 1
            continue
        
        if ch == '\\':
            result.append(ch)
            escape_next = True
            i += 1
            continue
        
        if ch == '"' and not in_string:
            in_string = True
            result.append(ch)
            i += 1
            continue
        
        if ch == '"' and in_string:
            # Check: is this a closing quote or a quote inside the string?
            lookahead = i + 1
            while lookahead < n and json_str[lookahead] in (' ', '\t', '\r', '\n'):
                lookahead += 1
            if lookahead < n and json_str[lookahead] in (',', ':', '}', ']'):
                in_string = False
                result.append(ch)
                i += 1
                continue
            elif lookahead >= n:
                in_string = False
                result.append(ch)
                i += 1
                continue
            else:
                # Quote inside the string — escape it
                result.append('\\')
                result.append(ch)
                i += 1
                continue
        
        if in_string:
            if ch == '\n':
                result.append('\\n')
            elif ch == '\r':
                if i + 1 < n and json_str[i + 1] == '\n':
                    result.append('\\n')
                    i += 1
                else:
                    result.append('\\n')
            elif ch == '\t':
                result.append('\\t')
            elif ord(ch) < 32:
                result.append(' ')
            else:
                result.append(ch)
        else:
            result.append(ch)
        
        i += 1
    
    return ''.join(result)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ts_to_datetime(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)


def datetime_to_ts_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)
