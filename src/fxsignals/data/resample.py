"""Timeframe resampling and look-ahead-free HTF/LTF alignment.

All timeframe durations come from the central ``TIMEFRAME_DURATIONS`` map in
:mod:`fxsignals.models` so no legacy pandas aliases ("H", "T") or unit-only
Timedelta strings are constructed here (both are rejected by pandas 3).
"""

from __future__ import annotations

import pandas as pd

from fxsignals.models import Timeframe, timeframe_rule, timeframe_timedelta


def _tf_step(tf: Timeframe) -> pd.Timedelta:
    """Return the bar duration for ``tf`` from the central mapping."""
    return timeframe_timedelta(tf.name)


def _infer_tf_minutes(df: pd.DataFrame) -> float | None:
    """Median bar spacing of ``df`` in minutes (None if fewer than two bars).

    Uses index diffs rather :func:`pandas.infer_freq` so weekend gaps and
    non-standard frequencies never produce legacy alias strings.
    """
    if len(df.index) < 2:
        return None
    diffs = df.index.to_series().diff().dropna()
    return float(diffs.median().total_seconds() / 60.0)


def resample_ohlc(
    df: pd.DataFrame,
    target_tf: Timeframe,
    keep_incomplete_last: bool = False,
) -> pd.DataFrame:
    """Aggregate candles to a coarser timeframe.

    Aggregation follows OHLC semantics: open=first, high=max, low=min,
    close=last, volume=sum. Bars are labeled by their interval *start*
    (closed-left), matching the Candle contract convention.

    No look-ahead: unless ``keep_incomplete_last`` is True, the final bar is
    dropped when its full interval has not elapsed relative to the last
    available source bar, i.e. only closed higher-TF bars are returned.

    Args:
        df: Source candles (UTC DatetimeIndex, contract columns).
        target_tf: Coarser destination timeframe.
        keep_incomplete_last: Keep the trailing partial bar if it exists.

    Returns:
        Resampled DataFrame satisfying the Candle contract.

    Raises:
        ValueError: if ``target_tf`` is finer than the source data spacing or
            the frame lacks the required columns.
    """
    cols = {"open", "high", "low", "close", "volume"}
    if not cols.issubset(df.columns):
        raise ValueError(f"resample_ohlc requires columns {sorted(cols)}")
    if df.empty:
        return df.copy()

    src_minutes = _infer_tf_minutes(df)
    if src_minutes is not None and src_minutes > target_tf.minutes:
        raise ValueError(
            f"Cannot resample from ~{int(src_minutes)}min bars to "
            f"{target_tf.name} ({target_tf.minutes}min): target is finer."
        )

    agg = df.resample(
        timeframe_rule(target_tf.name), label="left", closed="left"
    ).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    agg = agg.dropna(subset=["open", "high", "low", "close"])

    if not keep_incomplete_last and len(agg) > 0:
        # The final target bucket is only "closed" once a source bar exists at
        # or after its end time; otherwise it is still forming (look-ahead).
        bucket_end = agg.index[-1] + _tf_step(target_tf)
        if df.index.max() < bucket_end:
            agg = agg.iloc[:-1]

    agg.index = pd.DatetimeIndex(agg.index, name="time")
    return agg


def align_htf_to_ltf(htf_df: pd.DataFrame, ltf_df: pd.DataFrame) -> pd.Series:
    """Map each LTF bar to the open time of the last *closed* HTF bar.

    A higher-TF bar that opens at ``t`` closes at ``t + tf``; it may only be
    used by LTF bars whose open time is >= that close time (equivalently: the
    HTF series is shifted forward by one interval before the backward join),
    so no future HTF information can leak into any LTF row.

    Args:
        htf_df: Candles on the higher timeframe (contract format).
        ltf_df: Candles on the lower timeframe (contract format).

    Returns:
        Series indexed like ``ltf_df`` whose values are the open time of the
        most recently closed HTF bar (NaT until the first HTF bar has closed).
    """
    if htf_df.empty or ltf_df.empty:
        return pd.Series(pd.NaT, index=ltf_df.index, name="htf_open_time")

    step = _tf_step(_tf_from_index(htf_df))
    # Availability time of each HTF bar is its close (open + one interval);
    # the backward join below then guarantees only closed bars are visible.
    avail = pd.DataFrame(
        {"available_at": htf_df.index.to_series() + step},
        index=pd.DatetimeIndex(htf_df.index, name="htf_open"),
    )

    left = (
        ltf_df.index.to_series()
        .rename("ltf_time")
        .to_frame(index=False)
        .sort_values("ltf_time")
    )
    merged = pd.merge_asof(
        left,
        avail.reset_index(),
        left_on="ltf_time",
        right_on="available_at",
        direction="backward",
    )
    out = pd.Series(
        merged["htf_open"].values,
        index=pd.DatetimeIndex(merged["ltf_time"], tz="UTC", name="time"),
        name="htf_open_time",
    )
    return out.sort_index()


def _tf_from_index(df: pd.DataFrame) -> Timeframe:
    """Infer the candle frame's timeframe from its index spacing.

    Uses the median spacing so weekend gaps do not skew the estimate.

    Raises:
        ValueError: if fewer than two bars exist or the spacing does not match
            a supported timeframe.
    """
    if len(df.index) < 2:
        raise ValueError("Need at least two HTF bars to infer the timeframe")
    diffs = df.index.to_series().diff().dropna()
    minutes = int(diffs.median().total_seconds() // 60)
    for tf in Timeframe:
        if tf.minutes == minutes:
            return tf
    raise ValueError(f"HTF index spacing {minutes}min is not a supported timeframe")
