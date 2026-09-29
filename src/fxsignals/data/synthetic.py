"""Deterministic synthetic OHLC generator.

Produces realistic-looking candles (trending and ranging regimes, volatility
clustering) so the rest of the bot can run with no external data source.
Generation is fully seeded: same seed + parameters => identical DataFrame.
FX weekend gaps are excluded from the bar timeline.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Iterator

import numpy as np
import pandas as pd

from fxsignals.config import Settings
from fxsignals.data.base import DataProvider, ProviderError, register_provider
from fxsignals.models import CANDLE_COLUMNS, Timeframe

# Anchor every stream at a fixed UTC Monday 00:00 so results never drift in time.
_ANCHOR = pd.Timestamp("2020-01-06T00:00:00Z")

# Approximate base prices used to keep generated series plausible per symbol.
_BASE_PRICES = {
    "EURUSD": 1.10, "GBPUSD": 1.27, "AUDUSD": 0.66, "NZDUSD": 0.61,
    "USDJPY": 150.0, "USDCAD": 1.36, "USDCHF": 0.88, "XAUUSD": 2000.0,
}


@dataclass(frozen=True)
class SyntheticParams:
    """Knobs controlling the generated price process.

    Attributes:
        seed: Master RNG seed; identical seeds give identical output.
        bars: Number of tradable bars produced.
        base_price: Starting mid price (None derives one from the pair).
        vol: Typical per-bar return standard deviation (fraction of price).
        trend_prob: Probability of switching into a trending regime per bar.
        trend_strength: Drift magnitude applied while in a trend regime.
        vol_cluster: Strength of volatility clustering (GARCH-like persistence).
    """

    seed: int = 42
    bars: int = 500
    base_price: float | None = None
    vol: float = 0.0012
    trend_prob: float = 0.02
    trend_strength: float = 0.0009
    vol_cluster: float = 0.85


def _base_price_for(pair: str) -> float:
    """Return a plausible starting price for ``pair`` (falls back to 1.0)."""
    return _BASE_PRICES.get(pair.upper(), 1.0)


def _tradable_slots(n_bars: int, step: timedelta) -> Iterator[pd.Timestamp]:
    """Yield ``n_bars`` consecutive bar-open timestamps skipping weekends."""
    ts = _ANCHOR
    yielded = 0
    while yielded < n_bars:
        # Saturday=5, Sunday=6: skip the whole weekend block.
        if ts.weekday() < 5:
            yield ts
            yielded += 1
        ts = ts + step


def generate_candles(
    pair: str,
    timeframe: Timeframe,
    params: SyntheticParams | None = None,
) -> pd.DataFrame:
    """Generate deterministic synthetic candles satisfying the Candle contract.

    Args:
        pair: Symbol; only used to pick a plausible base price.
        timeframe: Bar interval determining bar spacing.
        params: Optional :class:`SyntheticParams`; defaults used otherwise.

    Returns:
        A contract-compliant DataFrame with ``params.bars`` rows.
    """
    p = params or SyntheticParams()
    if p.bars <= 0:
        raise ValueError(f"bars must be positive, got {p.bars}")

    rng = np.random.default_rng(p.seed)
    base = p.base_price if p.base_price is not None else _base_price_for(pair)
    slots = pd.DatetimeIndex(
        list(_tradable_slots(p.bars, timedelta(minutes=timeframe.minutes)))
    )

    # Regime state machine: 0 = range (noise), +/-1 = trend (directional drift).
    regime = np.zeros(p.bars, dtype=int)
    for i in range(1, p.bars):
        if rng.random() < p.trend_prob:
            regime[i] = 1 if rng.random() < 0.5 else -1
        else:
            regime[i] = regime[i - 1]

    # Volatility clustering: sigma persists across bars (GARCH-style).
    log_sigma = np.full(p.bars, np.log(p.vol))
    shock = rng.standard_normal(p.bars)
    for i in range(1, p.bars):
        log_sigma[i] = (
            np.log(p.vol) * (1 - p.vol_cluster)
            + p.vol_cluster * log_sigma[i - 1]
            + 0.1 * shock[i]
        )
    sigma = np.exp(log_sigma)

    drift = regime * p.trend_strength * np.sign(rng.standard_normal(p.bars))
    rets = drift + sigma * rng.standard_normal(p.bars)

    close = base * np.cumprod(1.0 + rets)
    open_ = np.empty(p.bars)
    open_[0] = base
    open_[1:] = close[:-1]

    # Intrabar excursions are non-negative, so high >= max(o,c), low <= min(o,c).
    wick_hi = np.abs(rng.standard_normal(p.bars)) * sigma * base
    wick_lo = np.abs(rng.standard_normal(p.bars)) * sigma * base
    body = np.abs(close - open_)
    high = np.maximum(open_, close) + wick_hi + 0.05 * body
    low = np.minimum(open_, close) - wick_lo - 0.05 * body

    volume = np.round(np.abs(rng.standard_normal(p.bars)) * 500.0 + 1000.0)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.DatetimeIndex(slots, name="time"),
    )[list(CANDLE_COLUMNS)]
    return df


@register_provider("synthetic")
class SyntheticProvider(DataProvider):
    """DataProvider wrapper around :func:`generate_candles`.

    Each (pair, timeframe) gets its own derived seed so streams are
    independent yet reproducible.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        params: SyntheticParams | None = None,
    ) -> None:
        super().__init__(settings)
        self.params = params or SyntheticParams()

    def get_candles(
        self,
        pair: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Return deterministic synthetic candles, optionally windowed.

        Raises:
            ProviderError: if the requested window lies entirely outside the
                generated history.
        """
        from fxsignals.data.normalize import slice_window

        # Derive a stable per-stream seed (Python's hash() is salted per
        # process, so use a checksum of the symbol instead).
        digest = sum(ord(c) for c in f"{pair.upper()}|{timeframe.name}")
        params = replace(self.params, seed=self.params.seed * 1_000 + digest % 10_000)
        df = generate_candles(pair, timeframe, params)

        if start is not None or end is not None or limit is not None:
            windowed = slice_window(df, start=start, end=end, limit=limit)
            if windowed.empty and (start is not None or end is not None):
                raise ProviderError(
                    f"Synthetic history for {pair}/{timeframe.name} does not "
                    f"overlap window [{start}, {end}]"
                )
            return windowed
        return df
