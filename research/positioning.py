"""POS-1: признаки позиционного перегруза (funding / OI / L/S).

Правило отсутствия утечки: свеча с временем открытия T использует только
записи с timestamp <= T. Каждый источник мерджится ОТДЕЛЬНО на своём
родном таймфрейме (funding 8h, OI/L/S 1h) — это корректно обрабатывает
разную частоту обновлений. Возраст последней записи считается на
naive-UTC numpy-массивах, чтобы избежать ошибок сравнения таймзон.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[1] / "backtest_data"

# Минимум наблюдений родного таймфрейма для расчёта z-score.
MIN_OBS = 96

# Максимальный возраст последней доступной записи. Свежее нет -> NaN.
MAX_AGE = {
    "fund_z": pd.Timedelta(hours=30),
    "oi_d4": pd.Timedelta(hours=6),
    "ls_z": pd.Timedelta(hours=6),
}

FEATURES = ("fund_z", "oi_d4", "ls_z")


def _naive_utc(ts) -> pd.Series:
    """Приводит к naive-UTC: и наивные (считаем UTC), и tz-aware."""
    return pd.to_datetime(ts, utc=True).dt.tz_localize(None)


def _z(s: pd.Series) -> pd.Series:
    """Rolling z-score без заглядывания в будущее."""
    mean = s.rolling(MIN_OBS, min_periods=MIN_OBS).mean()
    std = s.rolling(MIN_OBS, min_periods=MIN_OBS).std()
    return (s - mean) / std.replace(0.0, np.nan)


def _load(symbol: str, subdir: str, col: str) -> pd.DataFrame | None:
    """Загружает один источник: backtest_data/<subdir>/<symbol>.csv."""
    path = DATA_ROOT / subdir / f"{symbol}.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if col not in df.columns:
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["timestamp", col])
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .set_index("timestamp")
    )
    if df.empty:
        return None
    return df[[col]]


def _build_sources(symbol: str) -> dict:
    """feature -> Series, индекс = tz-aware timestamp."""
    sources: dict = {}

    funding = _load(symbol, "funding", "funding_rate")
    if funding is not None:
        sources["fund_z"] = _z(funding["funding_rate"])

    oi = _load(symbol, "open_interest", "open_interest")
    if oi is not None:
        log_oi = np.log(oi["open_interest"].where(oi["open_interest"] > 0))
        sources["oi_d4"] = log_oi.diff(4)

    ls = _load(symbol, "long_short", "long_short_ratio")
    if ls is not None:
        log_ls = np.log(ls["long_short_ratio"].where(ls["long_short_ratio"] > 0))
        sources["ls_z"] = _z(log_ls)

    return sources


def _merge_feature(d: pd.DataFrame, series: pd.Series, col: str,
                   max_age: pd.Timedelta) -> pd.DataFrame:
    """Отдельный backward-merge одного признака + защита от устаревания."""
    s = series.dropna()
    if s.empty:
        d[col] = np.nan
        return d

    s = s.to_frame(name=col).reset_index()
    s = s.rename(columns={s.columns[0]: "timestamp"})
    s["_merge_ts"] = _naive_utc(s["timestamp"])
    s = s.sort_values("_merge_ts")
    s["src_ts"] = s["_merge_ts"]

    m = pd.merge_asof(
        d[["_merge_ts"]],
        s[["_merge_ts", col, "src_ts"]],
        on="_merge_ts",
        direction="backward",
        allow_exact_matches=True,
    )

    d[col] = m[col].to_numpy()

    # Возраст считаем на naive-UTC numpy-массивах — без сравнения таймзон.
    left = d["_merge_ts"].to_numpy()
    right = m["src_ts"].to_numpy()
    age = left - right
    stale = age > max_age.to_timedelta64()
    no_src = pd.isna(m["src_ts"]).to_numpy()
    d.loc[stale | no_src, col] = np.nan

    return d


def attach_positioning(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Присоединяет fund_z / oi_d4 / ls_z к 15m-фрейму без lookahead."""
    d = df.copy()
    d["_merge_ts"] = _naive_utc(d["timestamp"])
    d = d.sort_values("_merge_ts").reset_index(drop=True)

    sources = _build_sources(symbol)

    for col in FEATURES:
        if col not in sources:
            d[col] = np.nan
            continue
        d = _merge_feature(d, sources[col], col, MAX_AGE[col])

    d = d.drop(columns=["_merge_ts"])
    return d