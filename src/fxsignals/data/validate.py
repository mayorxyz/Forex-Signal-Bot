"""Validation and cleaning of candle DataFrames against the Candle contract."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from fxsignals.models import CANDLE_COLUMNS, Timeframe


@dataclass
class ValidationReport:
    """Outcome of :func:`validate_candles`.

    Attributes:
        ok: True when no problems were found.
        duplicate_timestamps: Count of duplicated index timestamps.
        unsorted: True when the index is not ascending.
        nan_rows: Index labels of rows containing NaN values.
        ohlc_violations: Index labels where high/low wrap logic is broken.
        nonpositive_prices: Index labels where any price is <= 0.
        gap_count: Number of expected-interval gaps (weekends ignored).
        missing_columns: Contract columns absent from the frame.
    """

    ok: bool = True
    duplicate_timestamps: int = 0
    unsorted: bool = False
    nan_rows: list[object] = field(default_factory=list)
    ohlc_violations: list[object] = field(default_factory=list)
    nonpositive_prices: list[object] = field(default_factory=list)
    gap_count: int = 0
    missing_columns: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """One-line human-readable summary of all findings."""
        if self.ok and not self.missing_columns:
            return "OK: candles satisfy the contract"
        parts = [
            f"duplicates={self.duplicate_timestamps}",
            f"unsorted={self.unsorted}",
            f"nan_rows={len(self.nan_rows)}",
            f"ohlc_violations={len(self.ohlc_violations)}",
            f"nonpositive_prices={len(self.nonpositive_prices)}",
            f"gaps={self.gap_count}",
            f"missing_columns={self.missing_columns}",
        ]
        return "INVALID: " + ", ".join(parts)


def _trading_minutes_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
    """Trading minutes between bar opens ``a`` and ``b``, excluding weekends.

    Weekends (Sat/Sun UTC) are treated as non-trading time; Friday-night and
    weekday hours all count, which matches the synthetic/real FX calendar gap
    policy used by this project.
    """
    total = int((b - a).total_seconds() // 60)
    weekend = 0
    day = a.normalize()
    while day < b.normalize():
        if day.weekday() >= 5:  # Saturday / Sunday
            # Overlap of [day, day+1d) with [a, b) in minutes.
            lo = max(a, day)
            hi = min(b, day + pd.Timedelta(days=1))
            if hi > lo:
                weekend += int((hi - lo).total_seconds() // 60)
        day += pd.Timedelta(days=1)
    return total - weekend


def validate_candles(df: pd.DataFrame, timeframe: Timeframe | None = None) -> ValidationReport:
    """Check ``df`` against the Candle contract and return a report.

    Args:
        df: Candidate candles frame (UTC DatetimeIndex named ``time``).
        timeframe: If given, used to detect gaps larger than one bar. Weekend
            gaps are always ignored.

    Returns:
        A populated :class:`ValidationReport` (never raises for bad data).
    """
    rep = ValidationReport()
    rep.missing_columns = [c for c in CANDLE_COLUMNS if c not in df.columns]
    if rep.missing_columns or not isinstance(df.index, pd.DatetimeIndex):
        rep.ok = False
        return rep

    rep.duplicate_timestamps = int(df.index.duplicated(keep="first").sum())
    rep.unsorted = not df.index.is_monotonic_increasing

    nan_mask = df[list(CANDLE_COLUMNS)].isna().any(axis=1)
    rep.nan_rows = [ts for ts, bad in zip(df.index, nan_mask) if bad]

    numeric = df[list(CANDLE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    prices = numeric[["open", "high", "low", "close"]]
    pos_mask = (prices <= 0).any(axis=1) | prices.isna().any(axis=1)
    rep.nonpositive_prices = [ts for ts, bad in zip(df.index, pos_mask) if bad]

    valid = numeric.dropna(subset=["open", "high", "low", "close"])
    ohlc_bad = (
        (valid["high"] < valid[["open", "close"]].max(axis=1))
        | (valid["low"] > valid[["open", "close"]].min(axis=1))
    )
    rep.ohlc_violations = [ts for ts, bad in zip(valid.index, ohlc_bad) if bad]

    if timeframe is not None and len(df.index) >= 2:
        ordered = df.sort_index()
        prev_ts, cur_ts = ordered.index[:-1], ordered.index[1:]
        for a, b in zip(prev_ts, cur_ts):
            # A gap is only flagged when more than one bar is missing after
            # discounting non-trading weekend time.
            if _missing_bars(a, b, timeframe) > 1:
                rep.gap_count += 1

    rep.ok = not (
        rep.duplicate_timestamps
        or rep.unsorted
        or rep.nan_rows
        or rep.ohlc_violations
        or rep.nonpositive_prices
        or rep.gap_count
        or rep.missing_columns
    )
    return rep


def _missing_bars(a: pd.Timestamp, b: pd.Timestamp, timeframe: Timeframe) -> int:
    """Bars expected between consecutive opens ``a`` and ``b`` minus one.

    Weekend time is excluded, so the Friday-close to Sunday-open jump is not
    counted as a gap. Returns 0 when ``b`` is exactly one bar after ``a``.
    """
    minutes = _trading_minutes_between(a, b)
    expected = max(round(minutes / timeframe.minutes), 1)
    return expected - 1


def clean_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy satisfying the Candle contract where possible.

    Drops duplicate timestamps (keeping the last occurrence), sorts ascending,
    removes rows with NaN/non-positive prices or broken OHLC wrap logic. It
    never interpolates or fills values, so no future data can leak in.

    Args:
        df: Candidate candles frame.

    Returns:
        Cleaned DataFrame (may be empty).
    """
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise TypeError("clean_candles requires a DatetimeIndex")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    missing = [c for c in CANDLE_COLUMNS if c not in out.columns]
    if missing:
        raise KeyError(f"Missing required candle columns: {missing}")

    numeric = out[list(CANDLE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    keep = numeric.dropna().index
    prices = numeric.loc[keep, ["open", "high", "low", "close"]]
    keep = keep[(prices > 0).all(axis=1)]
    valid = numeric.loc[keep]
    ok_ohlc = (
        (valid["high"] >= valid[["open", "close"]].max(axis=1))
        & (valid["low"] <= valid[["open", "close"]].min(axis=1))
    )
    keep = keep[ok_ohlc]
    return numeric.loc[keep]
