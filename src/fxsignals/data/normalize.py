"""Normalization and integrity helpers shared by data modules."""

from __future__ import annotations

import pandas as pd

from fxsignals.models import CANDLE_COLUMNS


def normalize_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a raw frame into the canonical Candle contract DataFrame.

    Expects columns ``time, open, high, low, close`` plus optional ``volume``.
    Times are converted to UTC; rows are de-duplicated (keeping the last) and
    sorted ascending. No values are invented and no future data is filled.

    Args:
        df: Raw candles frame with a ``time`` column (or ``time`` index).

    Returns:
        New frame with a UTC DatetimeIndex named ``time`` and exactly the
        columns ``open, high, low, close, volume``.

    Raises:
        KeyError: if required price columns are missing.
    """
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required candle columns: {sorted(missing)}")

    out = df.copy()
    if "time" in out.columns:
        times = out.pop("time")
    elif isinstance(out.index, pd.DatetimeIndex):
        times = pd.Series(out.index)
    else:
        raise KeyError("Candle data must have a 'time' column or a DatetimeIndex")

    idx = pd.to_datetime(times, utc=True)
    out.index = pd.DatetimeIndex(idx, name="time")
    if "volume" not in out.columns:
        out["volume"] = 0.0

    out = out[list(CANDLE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.astype(
        {"open": "float64", "high": "float64", "low": "float64",
         "close": "float64", "volume": "float64"}
    )


def slice_window(
    df: pd.DataFrame,
    start: object | None = None,
    end: object | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Filter a contract frame by inclusive [start, end] window and tail size.

    Args:
        df: Candle-contract DataFrame.
        start: Optional inclusive start time (naive values assumed UTC).
        end: Optional inclusive end time (naive values assumed UTC).
        limit: If given, keep only the last ``limit`` bars of the window.

    Returns:
        A filtered copy of ``df``.

    Raises:
        ValueError: if ``limit`` is non-positive or start is after end.
    """
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    out = df
    if start is not None:
        ts = pd.Timestamp(start)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        out = out[out.index >= ts]
    if end is not None:
        te = pd.Timestamp(end)
        if te.tzinfo is None:
            te = te.tz_localize("UTC")
        out = out[out.index <= te]
    if start is not None and end is not None and pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError("start must not be after end")
    if limit is not None:
        out = out.tail(int(limit))
    return out.copy()
