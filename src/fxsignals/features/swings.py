"""Swing (pivot) detection with right-side confirmation.

A swing high at pivot bar ``p`` requires ``left`` strictly lower highs before
it and ``right`` strictly lower highs after it. It therefore only becomes
*visible* at the confirmation bar ``c = p + right`` and is stamped there, so a
value at bar ``t`` never depends on bars ``> t``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EVENT_COLUMNS = ("kind", "price", "pivot_time", "confirmed_at", "confirmed_pos")


@dataclass(frozen=True)
class SwingSet:
    """Container for detected pivots and their per-bar causal views.

    Attributes:
        events: One row per confirmed pivot with columns ``kind``
            ('high'/'low'), ``price``, ``pivot_time``, ``confirmed_at`` and
            ``confirmed_pos`` (integer position of the confirmation bar).
        frame: Per-bar DataFrame indexed like the input with columns
            ``swing_high`` / ``swing_low`` (price of the most recent
            *confirmed* pivot of each kind), plus the matching
            ``pivot_time_*`` / ``confirmed_at_*`` stamps; NaN/NaT until the
            first confirmation bar.
        n: How many most-recent swings per kind consumers are expected to
            keep (see :meth:`history`).
    """

    events: pd.DataFrame
    frame: pd.DataFrame
    n: int

    @property
    def highs(self) -> pd.DataFrame:
        """Event rows for swing highs only."""
        return self.events[self.events["kind"] == "high"].reset_index(drop=True)

    @property
    def lows(self) -> pd.DataFrame:
        """Event rows for swing lows only."""
        return self.events[self.events["kind"] == "low"].reset_index(drop=True)

    def history(self, i: int) -> dict[str, list[tuple[float, pd.Timestamp]]]:
        """Last ``n`` confirmed swings of each kind known at bar index ``i``.

        Args:
            i: Index of the bar acting as "now".

        Returns:
            Dict with keys ``'highs'`` and ``'lows'``; each value is a list of
            ``(price, pivot_time)`` tuples ordered oldest to newest, limited
            to the ``n`` most recent confirmations at or before bar ``i``.
        """
        out: dict[str, list[tuple[float, pd.Timestamp]]] = {"highs": [], "lows": []}
        if self.events.empty:
            return out
        visible = self.events[self.events["confirmed_pos"] <= i]
        for kind, key in (("high", "highs"), ("low", "lows")):
            subset = visible[visible["kind"] == kind].tail(self.n)
            out[key] = [(float(r["price"]), r["pivot_time"]) for _, r in subset.iterrows()]
        return out


def _find_pivots(values: np.ndarray, left: int, right: int) -> list[int]:
    """Indices holding the strict maximum of window ``[i-left, i+right]``.

    Such a pivot at ``i`` is confirmed at ``i + right``.
    """
    n = len(values)
    out: list[int] = []
    for i in range(left, n - right):
        window = values[i - left : i + right + 1]
        if values[i] == window.max() and (window == values[i]).sum() == 1:
            out.append(i)
    return out


def swing_events(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    """Return confirmed pivot events with integer confirmation positions.

    Args:
        df: Candle-contract DataFrame.
        left: Bars required strictly beyond the pivot on its left.
        right: Bars required after the pivot for confirmation.

    Returns:
        DataFrame with columns ``kind, price, pivot_time, confirmed_at,
        confirmed_pos`` sorted by confirmation time.

    Raises:
        ValueError: if ``left``/``right`` are negative.
        KeyError: if ``high``/``low`` columns are missing.
    """
    if left < 0 or right < 0:
        raise ValueError(f"left/right must be >= 0, got {left}, {right}")
    for col in ("high", "low"):
        if col not in df.columns:
            raise KeyError(f"swing detection requires a '{col}' column")

    records: list[dict[str, object]] = []
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    # Minima become maxima after negation, so one scan routine serves both.
    for kind, src, search in (("high", highs, highs), ("low", lows, -lows)):
        for i in _find_pivots(search, left, right):
            pos = i + right
            records.append(
                {
                    "kind": kind,
                    "price": float(src[i]),
                    "pivot_time": df.index[i],
                    "confirmed_at": df.index[pos],
                    "confirmed_pos": int(pos),
                }
            )
    events = pd.DataFrame.from_records(records, columns=list(EVENT_COLUMNS))
    if events.empty:
        return events
    return events.sort_values(["confirmed_at", "kind"]).reset_index(drop=True)


def detect_swings(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    """Detect swing highs/lows and return the confirmed-pivot event table.

    Each row describes one pivot: swing-high vs swing-low semantics are
    carried by ``kind``, with ``price`` being the price at the pivot
    (``high`` for swing highs, ``low`` for swing lows), plus ``pivot_time``
    and ``confirmed_at``. A swing is only visible from its ``confirmed_at``
    bar onward — never earlier.

    Args:
        df: Candle-contract DataFrame (UTC DatetimeIndex).
        left: Bars required strictly beyond the pivot on its left.
        right: Bars required after the pivot for confirmation.

    Returns:
        DataFrame with columns ``kind, price, pivot_time, confirmed_at``.

    Raises:
        ValueError: if ``left``/``right`` are negative.
    """
    events = swing_events(df, left=left, right=right)
    return events[[c for c in EVENT_COLUMNS if c != "confirmed_pos"]]


def _event_arrays(
    index: pd.DatetimeIndex,
    events: pd.DataFrame,
    kind: str,
    field: str,
) -> np.ndarray:
    """Build a per-bar object array of ``field`` values stamped at confirmations.

    Args:
        index: The candle index the frame is aligned to.
        events: Event table from :func:`swing_events`.
        kind: ``'high'`` or ``'low'``.
        field: Event column to place ('pivot_time' or 'confirmed_at').

    Returns:
        Object array of length ``len(index)`` holding tz-aware UTC
        :class:`pandas.Timestamp` at each confirmation bar and ``None``
        elsewhere (converted to NaT by :func:`pandas.to_datetime`).
    """
    out: list[object] = [None] * len(index)
    subset = events[events["kind"] == kind]
    for _, ev in subset.iterrows():
        out[int(ev["confirmed_pos"])] = ev[field]
    return np.asarray(out, dtype=object)


def _causal_frame(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Stamp each confirmed pivot at its confirmation bar and ffill forward.

    Datetime columns are constructed via :func:`pandas.to_datetime(...,
    utc=True)` so they carry a tz-aware UTC dtype even when no events exist
    (all-NaT), instead of assigning tz-aware Timestamps into a naive
    ``datetime64`` column.
    """
    frame = pd.DataFrame(index=df.index)
    for col in ("swing_high", "swing_low"):
        frame[col] = np.nan
    for kind in ("high", "low"):
        # Build arrays from event positions/values, then create tz-aware UTC
        # columns; pd.NaT fills bars where no event has been stamped yet.
        pivot_arr = _event_arrays(df.index, events, kind, "pivot_time")
        conf_arr = _event_arrays(df.index, events, kind, "confirmed_at")
        frame[f"pivot_time_{kind}"] = pd.to_datetime(pivot_arr, utc=True)
        frame[f"confirmed_at_{kind}"] = pd.to_datetime(conf_arr, utc=True)
    for _, ev in events.iterrows():
        pos = int(ev["confirmed_pos"])
        suffix = "high" if ev["kind"] == "high" else "low"
        frame.loc[df.index[pos], f"swing_{suffix}"] = float(ev["price"])
    # Forward fill only: past bars never learn about later pivots.
    return frame.ffill()


def last_confirmed_swings(
    df: pd.DataFrame, n: int = 2, left: int = 3, right: int = 3
) -> SwingSet:
    """Return the last ``n`` confirmed swings per kind, known at each bar.

    Args:
        df: Candle-contract DataFrame.
        n: Number of most-recent swings per kind exposed by
            :meth:`SwingSet.history` (>= 1).
        left: Pivot left-window size.
        right: Pivot right-window (confirmation) size.

    Returns:
        A :class:`SwingSet` carrying the raw event table, a per-bar frame of
        the latest confirmed swing prices, and the ``n``-deep history helper.

    Raises:
        ValueError: if ``n < 1``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    events = swing_events(df, left=left, right=right)
    return SwingSet(events=events, frame=_causal_frame(df, events), n=n)
