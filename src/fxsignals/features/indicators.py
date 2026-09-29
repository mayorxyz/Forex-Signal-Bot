"""Pure pandas/numpy technical indicators over Candle-contract DataFrames.

Every function takes a candle DataFrame (UTC DatetimeIndex; columns
``open, high, low, close, volume``) and returns Series/DataFrames aligned to
the same index. Values are strictly causal: bar ``t`` uses bars ``<= t`` only.
During warm-up the output is NaN and is never back-filled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _close(df: pd.DataFrame) -> pd.Series:
    """Return the close series as float (raises KeyError when absent)."""
    if "close" not in df.columns:
        raise KeyError("indicator requires a 'close' column")
    return df["close"].astype(float)


def _hl(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return high/low float series (raises KeyError when absent)."""
    for col in ("high", "low"):
        if col not in df.columns:
            raise KeyError(f"indicator requires a '{col}' column")
    return df["high"].astype(float), df["low"].astype(float)


def _wilder_mean(x: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing: SMA seed over the first window, then alpha=1/period."""
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    if len(x) < period:
        return pd.Series(np.nan, index=x.index, name=x.name)
    values = x.to_numpy(dtype=float)
    res = np.full(len(values), np.nan)
    res[period - 1] = np.nanmean(values[:period])
    alpha = 1.0 / period
    for i in range(period, len(values)):
        res[i] = res[i - 1] + alpha * (values[i] - res[i - 1])
    return pd.Series(res, index=x.index, name=x.name)


def sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Simple moving average of closes. Min bars: ``period``."""
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    return _close(df).rolling(period, min_periods=period).mean().rename(f"sma_{period}")


def ema(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Exponential moving average of closes (adjust=False, SMA-seeded).

    Min bars: ``period``; earlier values are NaN.
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    return _close(df).ewm(span=period, adjust=False, min_periods=period).mean().rename(
        f"ema_{period}"
    )


def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Relative Strength Index with Wilder smoothing. Min bars: ``period + 1``.

    Output is bounded to [0, 100]; an all-flat sequence yields 50.0.
    """
    close = _close(df)
    delta = close.diff()
    gain = _wilder_mean(delta.clip(lower=0.0), period)
    loss = _wilder_mean(-delta.clip(upper=0.0), period)
    rs = gain / loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(~((loss == 0.0) & (gain > 0.0)), 100.0)
    out = out.where(~((gain == 0.0) & (loss > 0.0)), 0.0)
    out = out.where(~((gain == 0.0) & (loss == 0.0)), 50.0)
    return out.rename(f"rsi_{period}")


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range with Wilder smoothing. Min bars: ``period + 1``.

    The first bar's true range equals its high-low span; results are > 0 for
    valid candles.
    """
    high, low = _hl(df)
    close = _close(df)
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    tr = tr.fillna(high - low)
    return _wilder_mean(tr, period).rename(f"atr_{period}")


def macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD line, signal line and histogram. Min bars: ``slow + signal``.

    Returns:
        DataFrame with columns ``macd``, ``macd_signal``, ``macd_hist``.
    """
    close = _close(df)
    line = (
        close.ewm(span=fast, adjust=False, min_periods=fast).mean()
        - close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    )
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def bollinger(
    df: pd.DataFrame, period: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Bollinger bands. Min bars: ``period``.

    Returns:
        DataFrame with columns ``bb_mid``, ``bb_upper``, ``bb_lower`` and
        ``bb_width`` ((upper - lower) / mid).
    """
    close = _close(df)
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid
    return pd.DataFrame(
        {"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_width": width}
    )


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index with +DI / -DI (Wilder method).

    Min bars: ``2 * period + 1`` (the smoothed DI pair needs one extra bar
    before ADX itself can be averaged).

    Returns:
        DataFrame with columns ``plus_di``, ``minus_di``, ``adx``.
    """
    high, low = _hl(df)
    close = _close(df)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0),
        index=df.index,
    )
    tr = (
        pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        )
        .max(axis=1)
        .fillna(high - low)
    )
    atr_s = _wilder_mean(tr, period)
    plus_di = 100.0 * _wilder_mean(plus_dm, period) / atr_s
    minus_di = 100.0 * _wilder_mean(minus_dm, period) / atr_s
    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx_s = _wilder_mean(dx.dropna(), period).reindex(df.index)
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx_s})


def stochastic(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3, smooth_k: int = 3
) -> pd.DataFrame:
    """Stochastic oscillator %K (smoothed) and %D.

    Min bars: ``k_period + smooth_k + d_period - 2``.

    Returns:
        DataFrame with columns ``stoch_k`` and ``stoch_d``.
    """
    high, low = _hl(df)
    close = _close(df)
    ll = low.rolling(k_period, min_periods=k_period).min()
    hh = high.rolling(k_period, min_periods=k_period).max()
    rng = (hh - ll).replace(0.0, np.nan)
    fast_k = 100.0 * (close - ll) / rng
    fast_k = fast_k.where(hh.notna(), np.nan).fillna(50.0)
    k = fast_k.rolling(smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(d_period, min_periods=d_period).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d})
