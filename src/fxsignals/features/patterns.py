"""Single-candle pattern detection as boolean Series.

Patterns use body/wick geometry normalized by ATR so thresholds adapt to
volatility. Each bar only inspects itself and the previous bar (causal, no
warm-up beyond the first bar; ATR warm-up NaNs simply make patterns False).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxsignals.features.indicators import atr as atr_indicator


def _parts(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return body size, upper wick, lower wick and full range."""
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise KeyError(f"pattern detection requires a '{col}' column")
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    body = (c - o).abs()
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    rng = (h - l).replace(0.0, np.nan)
    return body, upper, lower, rng


def _atr_ok(atr_series: pd.Series) -> pd.Series:
    """True where ATR is finite and positive."""
    return atr_series.notna() & (atr_series > 0)


def bullish_engulfing(
    df: pd.DataFrame,
    atr_series: pd.Series | None = None,
    min_body_atr: float = 0.5,
) -> pd.Series:
    """Bullish engulfing: prior bearish body fully covered by a larger bullish body.

    Args:
        df: Candle-contract DataFrame.
        atr_series: Optional precomputed ATR (defaults to ``atr(df)``).
        min_body_atr: Minimum current-body size in ATR units.

    Returns:
        Boolean Series aligned to ``df.index``.
    """
    a = atr_series if atr_series is not None else atr_indicator(df)
    o = df["open"].astype(float)
    c = df["close"].astype(float)
    po = o.shift(1)
    pc = c.shift(1)
    body = (c - o).abs()
    prev_body = (pc - po).abs()
    cond = (
        (c > o)
        & (pc < po)
        & (o <= pc)
        & (c >= po)
        & (body > prev_body)
        & (body >= min_body_atr * a)
        & _atr_ok(a)
    )
    return cond.fillna(False).rename("bullish_engulfing")


def bearish_engulfing(
    df: pd.DataFrame,
    atr_series: pd.Series | None = None,
    min_body_atr: float = 0.5,
) -> pd.Series:
    """Bearish engulfing: prior bullish body fully covered by a larger bearish body.

    Same parameters and output shape as :func:`bullish_engulfing`.
    """
    a = atr_series if atr_series is not None else atr_indicator(df)
    o = df["open"].astype(float)
    c = df["close"].astype(float)
    po = o.shift(1)
    pc = c.shift(1)
    body = (c - o).abs()
    prev_body = (pc - po).abs()
    cond = (
        (c < o)
        & (pc > po)
        & (o >= pc)
        & (c <= po)
        & (body > prev_body)
        & (body >= min_body_atr * a)
        & _atr_ok(a)
    )
    return cond.fillna(False).rename("bearish_engulfing")


def pin_bar_bull(
    df: pd.DataFrame,
    atr_series: pd.Series | None = None,
    wick_body_ratio: float = 2.0,
    min_wick_atr: float = 0.5,
    max_close_pos: float = 0.35,
) -> pd.Series:
    """Bullish pin bar: long lower wick rejecting downside.

    Args:
        df: Candle-contract DataFrame.
        atr_series: Optional precomputed ATR.
        wick_body_ratio: Lower wick must exceed this multiple of the body.
        min_wick_atr: Minimum lower-wick length in ATR units.
        max_close_pos: Close must sit in the top ``max_close_pos`` fraction of
            the bar range (rejection happened below).

    Returns:
        Boolean Series aligned to ``df.index``.
    """
    a = atr_series if atr_series is not None else atr_indicator(df)
    body, upper, lower, rng = _parts(df)
    c = df["close"].astype(float)
    l = df["low"].astype(float)
    close_pos = (c - l) / rng
    cond = (
        (lower >= wick_body_ratio * body)
        & (lower >= min_wick_atr * a)
        & (upper <= body)
        & (close_pos >= 1.0 - max_close_pos)
        & _atr_ok(a)
        & rng.notna()
    )
    return cond.fillna(False).rename("pin_bar_bull")


def pin_bar_bear(
    df: pd.DataFrame,
    atr_series: pd.Series | None = None,
    wick_body_ratio: float = 2.0,
    min_wick_atr: float = 0.5,
    max_open_pos: float = 0.35,
) -> pd.Series:
    """Bearish pin bar: long upper wick rejecting upside.

    Mirror of :func:`pin_bar_bull`; ``max_open_pos`` bounds how far below the
    high the close may sit (close near the top means no rejection).
    """
    a = atr_series if atr_series is not None else atr_indicator(df)
    body, upper, lower, rng = _parts(df)
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    close_pos = (h - c) / rng
    cond = (
        (upper >= wick_body_ratio * body)
        & (upper >= min_wick_atr * a)
        & (lower <= body)
        & (close_pos >= 1.0 - max_open_pos)
        & _atr_ok(a)
        & rng.notna()
    )
    return cond.fillna(False).rename("pin_bar_bear")


def inside_bar(df: pd.DataFrame) -> pd.Series:
    """Inside bar: current high/low fully contained in the previous bar's range."""
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    cond = (h <= h.shift(1)) & (l >= l.shift(1))
    return cond.fillna(False).rename("inside_bar")
