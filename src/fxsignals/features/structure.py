"""Market structure from confirmed swings: trend state, BOS and CHoCH.

Everything is derived from swings *as they were confirmed* (see
:mod:`fxsignals.features.swings`), so each bar only knows about pivots whose
confirmation bar has already closed. Breaks use **closes**, never wicks.

Definitions used here:
- trend UP: the two most recent confirmed swing highs make a higher high AND
  the two most recent confirmed swing lows make a higher low (HH+HL). DOWN is
  the mirror (LH+LL); otherwise RANGE. Undetermined (fewer than two swings of
  a kind) is also reported as RANGE with ``st_defined=False``.
- BOS up: close breaks above the last confirmed swing high while trend is UP
  (continuation). BOS down mirrors it in a DOWN trend.
- CHoCH: the *first* break against the prevailing structure — a close below
  the last confirmed swing low while trend is UP, or above the last swing
  high while trend is DOWN. A CHoCH flips the internal bias to RANGE until
  new HH/HL (or LH/LL) evidence re-establishes a trend.

BOS and CHoCH are edge-triggered: each fires once, on the first close beyond
each confirmed swing level, not on every later bar that stays past it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fxsignals.features.swings import SwingSet, last_confirmed_swings

TREND_UP = "UP"
TREND_DOWN = "DOWN"
TREND_RANGE = "RANGE"


def _same_level(a: float, b: float) -> bool:
    """Return True if two swing levels denote the same broken level.

    ``np.nan`` matches only ``np.nan`` (no level recorded yet), so a plain
    ``a != b`` comparison would wrongly re-fire on the very first break.
    """
    if np.isnan(a) or np.isnan(b):
        return bool(np.isnan(a) and np.isnan(b))
    return a == b


@dataclass(frozen=True)
class StructureResult:
    """Per-bar market-structure output.

    Attributes:
        frame: DataFrame indexed like the input candles with columns
            ``trend_state`` ('UP'/'DOWN'/'RANGE'), ``st_defined`` (bool: at
            least two swings of each kind known), ``last_swing_high``,
            ``last_swing_low``, ``bos_up``, ``bos_down``, ``choch`` (bool
            Series, edge-triggered: True only on the first bar closing
            beyond each confirmed swing level).
        swings: The :class:`SwingSet` the structure was computed from.
    """

    frame: pd.DataFrame
    swings: SwingSet


def market_structure(
    df: pd.DataFrame, left: int = 3, right: int = 3, n: int = 2
) -> StructureResult:
    """Compute per-bar trend state, BOS and CHoCH flags for ``df``.

    Args:
        df: Candle-contract DataFrame (UTC DatetimeIndex).
        left: Swing pivot left-window size.
        right: Swing confirmation window size.
        n: Number of most-recent swings per kind considered for HH/HL/LH/LL
            comparisons (>= 2 required to classify a trend).

    Returns:
        A :class:`StructureResult`; its ``frame`` is aligned to ``df.index``.

    Raises:
        ValueError: if ``n < 2``.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2 to compare consecutive swings, got {n}")

    swings = last_confirmed_swings(df, n=n, left=left, right=right)
    closes = df["close"].to_numpy(dtype=float)
    events = swings.events

    m = len(df)
    trend = np.full(m, TREND_RANGE, dtype=object)
    defined = np.zeros(m, dtype=bool)
    last_h = np.full(m, np.nan)
    last_l = np.full(m, np.nan)
    bos_up = np.zeros(m, dtype=bool)
    bos_down = np.zeros(m, dtype=bool)
    choch = np.zeros(m, dtype=bool)

    # Edge-trigger state: the last swing level already broken by each flag.
    # A BOS/CHoCH fires only on the first close beyond a *new* level, so
    # later bars that stay past the same level do not re-fire it.
    bos_up_level = bos_down_level = choch_level = np.nan

    # Event cursors keep the per-bar scan O(events + bars), not O(bars*events).
    h_prices: list[float] = []
    l_prices: list[float] = []
    ev_list = list(events.itertuples(index=False)) if not events.empty else []
    cursor = 0
    bias = TREND_RANGE  # internal directional bias, flipped by CHoCH

    for i in range(m):
        while cursor < len(ev_list) and ev_list[cursor].confirmed_pos <= i:
            ev = ev_list[cursor]
            if ev.kind == "high":
                h_prices.append(float(ev.price))
                if len(h_prices) > n:
                    h_prices.pop(0)
            else:
                l_prices.append(float(ev.price))
                if len(l_prices) > n:
                    l_prices.pop(0)
            cursor += 1

        hh = len(h_prices) >= 2 and h_prices[-1] > h_prices[-2]
        hl = len(l_prices) >= 2 and l_prices[-1] > l_prices[-2]
        lh = len(h_prices) >= 2 and h_prices[-1] < h_prices[-2]
        ll = len(l_prices) >= 2 and l_prices[-1] < l_prices[-2]

        if hh and hl:
            state = TREND_UP
        elif lh and ll:
            state = TREND_DOWN
        else:
            state = bias if bias != TREND_RANGE else TREND_RANGE

        last_high = h_prices[-1] if h_prices else np.nan
        last_low = l_prices[-1] if l_prices else np.nan
        close = closes[i]

        broke_high = not np.isnan(last_high) and close > last_high
        broke_low = not np.isnan(last_low) and close < last_low

        if state == TREND_UP and broke_low:
            # CHoCH: first close below the last confirmed swing low in an UP trend.
            if not _same_level(last_low, choch_level):
                choch[i] = True
                choch_level = last_low
            bias = TREND_RANGE
        elif state == TREND_DOWN and broke_high:
            # CHoCH: first close above the last confirmed swing high in a DOWN trend.
            if not _same_level(last_high, choch_level):
                choch[i] = True
                choch_level = last_high
            bias = TREND_RANGE
        elif state == TREND_UP and broke_high:
            # BOS up: continuation break above the last confirmed swing high.
            if not _same_level(last_high, bos_up_level):
                bos_up[i] = True
                bos_up_level = last_high
        elif state == TREND_DOWN and broke_low:
            # BOS down: continuation break below the last confirmed swing low.
            if not _same_level(last_low, bos_down_level):
                bos_down[i] = True
                bos_down_level = last_low

        trend[i] = state
        defined[i] = len(h_prices) >= 2 and len(l_prices) >= 2
        last_h[i] = last_high
        last_l[i] = last_low

    frame = pd.DataFrame(
        {
            "trend_state": trend,
            "st_defined": defined,
            "last_swing_high": last_h,
            "last_swing_low": last_l,
            "bos_up": bos_up,
            "bos_down": bos_down,
            "choch": choch,
        },
        index=df.index,
    )
    return StructureResult(frame=frame, swings=swings)
