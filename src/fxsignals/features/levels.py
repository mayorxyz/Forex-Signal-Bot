"""Support/resistance zones clustered from confirmed swing prices.

Zones are built causally: at bar ``t`` only swings confirmed at or before
``t`` contribute, so a zone never contains future information. Zone width is
proportional to ATR (volatility-normalized); pivots within that band of each
other merge into one zone with a touch count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from fxsignals.features.indicators import atr as atr_indicator
from fxsignals.features.swings import last_confirmed_swings


@dataclass(frozen=True)
class SupportResistanceZone:
    """One price zone aggregated from confirmed swing pivots.

    Attributes:
        low: Lower edge of the zone.
        high: Upper edge of the zone.
        center: Mean price of the swings forming the zone.
        touches: Number of confirmed swings clustered into the zone.
        formed_at: Confirmation time of the newest swing in the zone — the
            first moment the zone became fully known.
        kinds: Contributing pivot kinds, e.g. ``{'high', 'low'}``.
    """

    low: float
    high: float
    center: float
    touches: int
    formed_at: pd.Timestamp
    kinds: frozenset[str]

    def contains(self, price: float) -> bool:
        """True when ``price`` lies inside the zone edges (inclusive)."""
        return self.low <= price <= self.high

    def distance_atr(self, price: float, atr_value: float) -> float:
        """Signed distance from ``price`` to the zone, in ATR units.

        Positive when the zone sits above the price, negative below, and 0.0
        when the price is inside the zone. Returns NaN for non-positive ATR.
        """
        if not np.isfinite(atr_value) or atr_value <= 0:
            return float("nan")
        if price < self.low:
            return (self.low - price) / atr_value
        if price > self.high:
            return -(price - self.high) / atr_value
        return 0.0


def _cluster(prices: Sequence[float], tol: float) -> list[list[float]]:
    """Greedy 1-D clustering: merge sorted points within ``tol`` of a member."""
    pts = sorted(float(p) for p in prices)
    if not pts:
        return []
    clusters: list[list[float]] = [[pts[0]]]
    for p in pts[1:]:
        if p - clusters[-1][-1] <= tol:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return clusters


def _build_zones(
    events: pd.DataFrame, tol: float, min_touches: int
) -> list[SupportResistanceZone]:
    """Turn visible swing events into qualifying zones (pure helper)."""
    zones: list[SupportResistanceZone] = []
    order = events.sort_values("price").index
    for cluster in _cluster(events.loc[order, "price"].to_numpy(), tol):
        lo, hi = cluster[0], cluster[-1]
        mask = (events["price"] >= lo) & (events["price"] <= hi)
        members = events[mask]
        if len(members) < min_touches:
            continue
        zones.append(
            SupportResistanceZone(
                low=float(members["price"].min()),
                high=float(members["price"].max()),
                center=float(members["price"].mean()),
                touches=int(len(members)),
                formed_at=pd.Timestamp(members["confirmed_at"].max()),
                kinds=frozenset(members["kind"]),
            )
        )
    return zones


def nearest_zone(
    price: float, zones: Iterable[SupportResistanceZone]
) -> SupportResistanceZone | None:
    """Return the zone closest to ``price`` (edges inclusive), or None.

    Args:
        price: Reference price.
        zones: Any iterable of :class:`SupportResistanceZone`.

    Returns:
        The zone minimizing absolute distance to the zone interval, or
        ``None`` when ``zones`` is empty.
    """
    best: SupportResistanceZone | None = None
    best_d = float("inf")
    for z in zones:
        d = 0.0 if z.contains(price) else min(abs(price - z.low), abs(price - z.high))
        if d < best_d:
            best, best_d = z, d
    return best


def zone_distance_atr(
    price: float, zones: Iterable[SupportResistanceZone], atr_value: float
) -> float:
    """Distance from ``price`` to the nearest zone in ATR units (signed).

    Returns NaN when no zone exists or ATR is not positive.
    """
    z = nearest_zone(price, zones)
    if z is None:
        return float("nan")
    return z.distance_atr(price, atr_value)


def support_resistance_zones(
    df: pd.DataFrame,
    atr_mult: float = 0.5,
    min_touches: int = 2,
    left: int = 3,
    right: int = 3,
    atr_period: int = 14,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster confirmed swing prices into S/R zones known at each bar.

    Args:
        df: Candle-contract DataFrame (UTC DatetimeIndex).
        atr_mult: Cluster tolerance expressed as a multiple of ATR.
        min_touches: Minimum confirmed swings required for a zone to be
            reported (>= 1).
        left: Swing pivot left-window size.
        right: Swing confirmation window size.
        atr_period: ATR look-back used for zone widths.

    Returns:
        Tuple ``(zones, per_bar)`` where:

        - ``zones``: final zone table (one row per zone) with columns
          ``low, high, center, touches, formed_at, kinds``.
        - ``per_bar``: DataFrame aligned to ``df.index`` with causal columns
          ``zone_count`` (zones known at that bar), ``nearest_zone_center``
          and ``nearest_zone_dist_atr`` (nearest known zone relative to that
          bar's close; NaN while no zone exists).

    Raises:
        ValueError: if ``atr_mult <= 0`` or ``min_touches < 1``.
    """
    if atr_mult <= 0:
        raise ValueError(f"atr_mult must be positive, got {atr_mult}")
    if min_touches < 1:
        raise ValueError(f"min_touches must be >= 1, got {min_touches}")

    swings = last_confirmed_swings(df, n=max(min_touches, 2), left=left, right=right)
    atr_series = atr_indicator(df, period=atr_period)
    events = swings.events

    m = len(df)
    zone_count = np.zeros(m, dtype=int)
    near_center = np.full(m, np.nan)
    near_dist = np.full(m, np.nan)

    # Recompute zones only at bars where a new swing gets confirmed.
    event_positions = (
        sorted({int(p) for p in events["confirmed_pos"]}) if not events.empty else []
    )
    snapshots: dict[int, list[SupportResistanceZone]] = {}
    for pos in event_positions:
        visible = events[events["confirmed_pos"] <= pos]
        atr_now = float(atr_series.iloc[pos])
        if not np.isfinite(atr_now) or atr_now <= 0:
            snapshots[pos] = []
            continue
        snapshots[pos] = _build_zones(visible, atr_mult * atr_now, min_touches)

    active: list[SupportResistanceZone] = []
    closes = df["close"].to_numpy(dtype=float)
    for i in range(m):
        if i in snapshots:
            active = snapshots[i]
        zone_count[i] = len(active)
        if active:
            z = nearest_zone(closes[i], active)
            if z is not None:
                near_center[i] = z.center
                near_dist[i] = z.distance_atr(closes[i], float(atr_series.iloc[i]))

    final_zones = snapshots[event_positions[-1]] if event_positions else []
    zones_df = pd.DataFrame.from_records(
        [
            {
                "low": z.low,
                "high": z.high,
                "center": z.center,
                "touches": z.touches,
                "formed_at": z.formed_at,
                "kinds": z.kinds,
            }
            for z in final_zones
        ],
        columns=["low", "high", "center", "touches", "formed_at", "kinds"],
    )
    per_bar = pd.DataFrame(
        {
            "zone_count": zone_count,
            "nearest_zone_center": near_center,
            "nearest_zone_dist_atr": near_dist,
        },
        index=df.index,
    )
    return zones_df, per_bar
