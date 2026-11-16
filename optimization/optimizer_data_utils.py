"""
Data utilities for the optimizer:
- loading and caching candles
- splitting into TRAIN/TEST windows
- splitting into optimization/holdout
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

import config

logger = logging.getLogger(__name__)


class OptimizerDataUtils:
    def __init__(self, engine):
        self.engine = engine

    def load_prepared_dataframe(
        self,
        symbol: str,
        category: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict:
        """
        Downloads the full historical period once and calculates indicators
        once. All optimization iterations then use only slices of this
        DataFrame via engine.run_on_dataframe(), without repeated API requests.
        """
        logger.info(
            f'[{symbol}] One-time download of historical candles for optimization: '
            f'{start_date} → {end_date}'
        )

        cache_dir = config.DATA_DIR / 'ohlcv_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / (
            f"{symbol}_15_{start_date.strftime('%Y%m%d')}_"
            f"{end_date.strftime('%Y%m%d')}.parquet"
        )

        # Cache is used only for fully completed periods.
        cache_usable = end_date.date() < datetime.now(datetime.now().astimezone().tzinfo).date()

        df = None
        if cache_usable and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                logger.info(f'[{symbol}] Candles taken from disk cache: {cache_file.name}')
            except Exception:
                df = None

        if df is None or df.empty:
            df = self.engine.fetcher.fetch_ohlcv(
                symbol=symbol, interval='15',
                start_date=start_date, end_date=end_date, category=category,
            )
            if cache_usable and df is not None and not df.empty:
                try:
                    df.to_parquet(cache_file, index=False)
                except Exception:
                    pass

        if df is None or df.empty:
            return {
                'error': f'Failed to load 15m candles for {symbol}.'
            }

        if len(df) < 150:
            return {
                'error': (
                    f'Not enough candles for optimization: {len(df)}. '
                    'At least 150 candles are needed.'
                )
            }

        try:
            logger.info(
                f'[{symbol}] Candles loaded: {len(df)}. '
                'One-time indicator calculation...'
            )

            prepared_df = self.engine._build_indicator_frame(df)

            logger.info(
                f'[{symbol}] Historical data preparation completed. '
                f'Rows: {len(prepared_df)}. '
                'Further iterations will work without loading candles.'
            )

            return {
                'df': prepared_df,
                'candles_count': len(prepared_df),
            }

        except Exception as e:
            logger.exception(f'[{symbol}] Error preparing indicators')
            return {
                'error': f'Error calculating indicators: {e}'
            }

    @staticmethod
    def make_splits(
        start_date: datetime,
        end_date: datetime,
        train_ratio: float = 0.7,
        walk_forward_folds: int = 1,
    ) -> List[Dict]:
        """
        Forms chronological TRAIN/TEST splits.

        walk_forward_folds = 1:
            Simple split:
            TRAIN = first train_ratio of the period,
            TEST = remainder of the period.

        walk_forward_folds > 1:
            Expanding walk-forward:
            - initial TRAIN occupies train_ratio of the period;
            - the remaining TEST part is divided into N consecutive test windows;
            - before each next TEST, the TRAIN window expands.
        """
        if start_date >= end_date:
            return []

        train_ratio = max(0.5, min(0.9, float(train_ratio)))
        walk_forward_folds = max(1, int(walk_forward_folds))

        total_seconds = (end_date - start_date).total_seconds()
        initial_train_end = start_date + timedelta(
            seconds=total_seconds * train_ratio
        )

        if walk_forward_folds == 1:
            return [{
                'train_start': start_date,
                'train_end': initial_train_end,
                'test_start': initial_train_end,
                'test_end': end_date,
            }]

        remaining_test_seconds = (end_date - initial_train_end).total_seconds()

        if remaining_test_seconds <= 0:
            return [{
                'train_start': start_date,
                'train_end': initial_train_end,
                'test_start': initial_train_end,
                'test_end': end_date,
            }]

        fold_seconds = remaining_test_seconds / walk_forward_folds
        splits = []

        for index in range(walk_forward_folds):
            test_start = initial_train_end + timedelta(
                seconds=fold_seconds * index
            )

            if index == walk_forward_folds - 1:
                test_end = end_date
            else:
                test_end = initial_train_end + timedelta(
                    seconds=fold_seconds * (index + 1)
                )

            if test_start >= test_end:
                continue

            splits.append({
                'train_start': start_date,
                'train_end': test_start,
                'test_start': test_start,
                'test_end': test_end,
            })

        return splits

    @staticmethod
    def split_optimization_and_holdout(
        candles_df,
        holdout_ratio: float = 0.15,
        min_holdout_candles: int = 500,
        min_optimization_candles: int = 3000,
    ):
        """
        Separates the history tail for the final holdout.

        Holdout does NOT participate in optimization and is used only
        for final validation of the best candidate.
        """
        total = len(candles_df)
        if total <= 0:
            return candles_df, candles_df.iloc[0:0].copy(), {
                "enabled": False,
                "reason": "empty_dataset",
                "total_candles": 0,
                "optimization_candles": 0,
                "holdout_candles": 0,
            }

        holdout_count = max(int(total * holdout_ratio), min_holdout_candles)
        optimization_count = total - holdout_count

        if holdout_count < min_holdout_candles or optimization_count < min_optimization_candles:
            return candles_df, candles_df.iloc[0:0].copy(), {
                "enabled": False,
                "reason": "not_enough_data",
                "total_candles": total,
                "optimization_candles": total,
                "holdout_candles": 0,
            }

        optimization_df = candles_df.iloc[:optimization_count].copy()
        holdout_df = candles_df.iloc[optimization_count:].copy()

        return optimization_df, holdout_df, {
            "enabled": True,
            "reason": "ok",
            "total_candles": total,
            "optimization_candles": len(optimization_df),
            "holdout_candles": len(holdout_df),
            "holdout_ratio": holdout_ratio,
        }

    @staticmethod
    def get_frame_boundary_datetimes(
        candles_df,
        fallback_start=None,
        fallback_end=None,
    ):
        """
        Returns the actual datetime boundaries of the DataFrame.
        Priority:
        1) timestamp/datetime/date/open_time/time column
        2) DatetimeIndex
        3) fallback_start / fallback_end
        """
        if candles_df is None or len(candles_df) == 0:
            return fallback_start, fallback_end

        # 1. First try time columns
        for col in ("timestamp", "datetime", "date", "open_time", "time"):
            if col in candles_df.columns and len(candles_df[col]) > 0:
                try:
                    start_dt = pd.to_datetime(candles_df[col].iloc[0], utc=True)
                    end_dt = pd.to_datetime(candles_df[col].iloc[-1], utc=True)

                    if hasattr(start_dt, "to_pydatetime"):
                        start_dt = start_dt.to_pydatetime()
                    if hasattr(end_dt, "to_pydatetime"):
                        end_dt = end_dt.to_pydatetime()

                    return start_dt, end_dt
                except Exception:
                    pass

        # 2. Then datetime index
        try:
            idx = candles_df.index
            if hasattr(idx, "dtype") and str(idx.dtype).startswith("datetime64"):
                start_dt = idx[0]
                end_dt = idx[-1]

                if hasattr(start_dt, "to_pydatetime"):
                    start_dt = start_dt.to_pydatetime()
                if hasattr(end_dt, "to_pydatetime"):
                    end_dt = end_dt.to_pydatetime()

                return start_dt, end_dt
        except Exception:
            pass

        return fallback_start, fallback_end

    @staticmethod
    def describe_frame_period(candles_df) -> Dict:
        """
        Determines the start and end of the period by the time column
        or datetime-index. Numeric index is not considered time.
        """
        result = {"start": None, "end": None}

        if candles_df is None or len(candles_df) == 0:
            return result

        for col in ("timestamp", "datetime", "date", "open_time", "time"):
            if col in candles_df.columns and len(candles_df[col]) > 0:
                try:
                    result["start"] = str(candles_df[col].iloc[0])
                    result["end"] = str(candles_df[col].iloc[-1])
                    return result
                except Exception:
                    pass

        try:
            idx = candles_df.index
            if hasattr(idx, "dtype") and str(idx.dtype).startswith("datetime64"):
                if len(idx) > 0:
                    result["start"] = str(idx[0])
                    result["end"] = str(idx[-1])
                    return result
        except Exception:
            pass

        return result