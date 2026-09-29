"""Score-based confluence: weigh feature groups into 0-100 direction scores.

Each factor *group* contributes a weighted partial score (strength 0-1), never
a strict AND-gate. Correlated indicators are counted **once per group**: RSI
and stochastic share the momentum vote; EMA, MACD and ADX share the
trend-strength vote (mirroring ``INDICATOR_GROUPS`` in the feature pipeline).

Inputs are rows of DataFrames produced by ``build_features`` /
``build_mtf_features``, so every value at bar ``t`` already uses bars ``<= t``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from fxsignals.models import Direction


def _finite(x: object) -> bool:
    """True when x is a finite real number."""
    try:
        return math.isfinite(float(x))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp x into [lo, hi]; NaN maps to lo (no evidence => no credit)."""
    if not _finite(x):
        return lo
    return max(lo, min(hi, float(x)))


@dataclass(frozen=True)
class ScoreWeights:
    """Confluence weights per evidence group (defaults sum to exactly 100).

    Attributes:
        htf_bias: Bias-TF trend state plus HTF trend/ADX agreement.
        structure: Entry-TF trend state and BOS/CHoCH direction.
        location: Proximity to support (longs) / resistance (shorts) in ATRs.
        momentum: RSI + stochastic — one combined vote.
        trend_strength: EMA + MACD + ADX — one combined vote.
        trigger: Candlestick pattern in the signal direction.
    """

    htf_bias: float = 30.0
    structure: float = 20.0
    location: float = 20.0
    momentum: float = 10.0
    trend_strength: float = 10.0
    trigger: float = 10.0

    def validated(self) -> "ScoreWeights":
        """Return self after checking non-negativity and total == 100.

        Raises:
            ValueError: on negative/non-numeric weights or a total that does
                not sum to 100 (within 0.01 tolerance).
        """
        values = {
            "htf_bias": self.htf_bias,
            "structure": self.structure,
            "location": self.location,
            "momentum": self.momentum,
            "trend_strength": self.trend_strength,
            "trigger": self.trigger,
        }
        bad = [k for k, v in values.items() if not _finite(v) or v < 0]
        if bad:
            raise ValueError(f"weights must be non-negative numbers, got bad keys: {bad}")
        total = sum(values.values())
        if abs(total - 100.0) > 0.01:
            raise ValueError(f"score weights must sum to 100, got {total}")
        return self

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "ScoreWeights":
        """Build weights from a mapping, keeping defaults for missing keys.

        Raises:
            ValueError: on unknown keys or an invalid total.
        """
        known = {"htf_bias", "structure", "location", "momentum",
                 "trend_strength", "trigger"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown weight keys: {sorted(unknown)}")
        return cls(**{k: float(v) for k, v in data.items()}).validated()


@dataclass(frozen=True)
class GroupScore:
    """One group's contribution to a direction score."""

    name: str
    weight: float
    strength: float  # partial credit in [0, 1]
    points: float    # weight * strength
    reason: str      # human-readable explanation


@dataclass(frozen=True)
class DirectionScore:
    """Total score plus per-group breakdown for one direction."""

    direction: Direction
    score: float
    groups: list[GroupScore] = field(default_factory=list)

    @property
    def reasons(self) -> list[str]:
        """Human-readable strings for every contributing factor."""
        return [g.reason for g in self.groups]


def _num(row: pd.Series, col: str) -> float:
    """Column value as float; NaN when absent or non-numeric."""
    try:
        return float(row.get(col))
    except (TypeError, ValueError):
        return float("nan")


def _flag(row: pd.Series, col: str) -> bool:
    """Boolean column as plain bool (tolerates None/NaN/pd.NA)."""
    val = row.get(col)
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return False
    try:
        return bool(val)
    except (TypeError, ValueError):
        return False


def _zone_side(row: pd.Series, atr: float) -> tuple[float, float, float]:
    """Signed zone offset (center-close)/ATR, raw distance-in-ATR, zone count.

    ``lv_nearest_zone_dist_atr`` from the levels module is signed positive
    when the zone sits ABOVE the close; we re-derive that sign robustly from
    center vs close when only the center is available.
    """
    center = _num(row, "lv_nearest_zone_center")
    dist = _num(row, "lv_nearest_zone_dist_atr")
    count = _num(row, "lv_zone_count")
    close = _num(row, "close")
    if not _finite(dist) and _finite(center) and _finite(close) and _finite(atr) and atr > 0:
        dist = (center - close) / atr
    return dist, dist, count


def score_direction(
    direction: Direction,
    entry_row: pd.Series,
    htf_row: pd.Series | None,
    weights: ScoreWeights | None = None,
) -> DirectionScore:
    """Score one direction from an entry-TF feature row (+ optional HTF row).

    Args:
        direction: LONG or SHORT.
        entry_row: One row of ``build_features(entry_df)`` output (may also
            carry candle columns such as ``close``).
        htf_row: One row of ``build_mtf_features(...)`` output, or None.
        weights: Optional :class:`ScoreWeights` (defaults validated).

    Returns:
        :class:`DirectionScore` with ``score`` in [0, 100]; each group appears
        at most once with partial credit and a reason string.
    """
    w = (weights or ScoreWeights()).validated()
    sign = 1.0 if direction is Direction.LONG else -1.0
    groups: list[GroupScore] = []

    def add(name: str, weight: float, strength: float, reason: str) -> None:
        s = _clamp(strength)
        if s > 0.0:
            groups.append(GroupScore(name, weight, s, weight * s, reason))

    atr = _num(entry_row, "ind_atr")
    close = _num(entry_row, "close")

    # --- HTF bias (30): aligned bias-TF trend state, tempered by HTF ADX ---
    if htf_row is not None:
        htf_state = str(htf_row.get("htf_st_trend_state") or "")
        htf_adx = _num(htf_row, "htf_ind_adx_adx")
        agrees = (htf_state == "UP" and direction is Direction.LONG) or (
            htf_state == "DOWN" and direction is Direction.SHORT
        )
        counter = (htf_state == "UP" and direction is Direction.SHORT) or (
            htf_state == "DOWN" and direction is Direction.LONG
        )
        if counter:
            strength, note = 0.0, f"HTF bias {htf_state} opposes {direction.value}"
        elif agrees:
            adx_credit = _clamp((htf_adx - 20.0) / 10.0) if _finite(htf_adx) else 0.5
            strength = 0.75 + 0.25 * adx_credit
            note = f"HTF bias {htf_state} supports {direction.value}" + (
                f" (ADX {htf_adx:.1f})" if _finite(htf_adx) else ""
            )
        elif htf_state == "RANGE":
            strength, note = 0.25, "HTF range: weak bias credit"
        else:
            strength, note = 0.0, "HTF bias unknown (warm-up)"
        add("htf_bias", w.htf_bias, strength, note)

    # --- Entry-TF structure (20): trend state + BOS / CHoCH in our direction --
    st_state = str(entry_row.get("st_trend_state") or "")
    agrees = (st_state == "UP" and direction is Direction.LONG) or (
        st_state == "DOWN" and direction is Direction.SHORT
    )
    bos = _flag(entry_row, "st_bos_up") if direction is Direction.LONG else _flag(
        entry_row, "st_bos_down"
    )
    choch = _flag(entry_row, "st_choch")
    detail: list[str] = []
    strength = 0.0
    if agrees:
        strength += 0.6
        detail.append(f"structure {st_state}")
    elif st_state == "RANGE":
        strength += 0.1
        detail.append("structure RANGE")
    if bos:
        strength += 0.4
        detail.append("BOS in direction")
    if choch:
        strength += 0.2
        detail.append("CHoCH break")
    add("structure", w.structure, strength,
        "Entry TF: " + (", ".join(detail) if detail else "no structural support"))

    # --- Location (20): near support for longs / resistance for shorts ------
    _, dist, count = _zone_side(entry_row, atr)
    strength = 0.0
    note = "Location: no zone information yet"
    if _finite(dist):
        # dist > 0 means the zone sits above the close (resistance for longs).
        favorable = dist >= 0.0 if direction is Direction.SHORT else dist <= 0.0
        proximity = _clamp(1.0 - abs(dist) / 2.0)
        depth = 0.8 + 0.2 * _clamp(count / 4.0) if _finite(count) else 0.8
        label = "support" if direction is Direction.LONG else "resistance"
        if favorable:
            strength = proximity * depth
            note = f"Location: {label} zone {abs(dist):.2f} ATR away"
        else:
            strength = proximity * 0.25
            note = f"Location: nearest zone lies against {direction.value} ({abs(dist):.2f} ATR)"
    add("location", w.location, strength, note)

    # --- Momentum group (10): RSI + stochastic merged into ONE vote ---------
    votes: list[float] = []
    parts: list[str] = []
    rsi_v = _num(entry_row, "ind_rsi")
    if _finite(rsi_v):
        votes.append(_clamp((35.0 - rsi_v) / 15.0) if direction is Direction.LONG
                     else _clamp((rsi_v - 65.0) / 15.0))
        parts.append(f"RSI {rsi_v:.1f}")
    stoch_k = _num(entry_row, "ind_stoch_k")
    if _finite(stoch_k):
        votes.append(_clamp((30.0 - stoch_k) / 30.0) if direction is Direction.LONG
                     else _clamp((stoch_k - 70.0) / 30.0))
        parts.append(f"stoch %K {stoch_k:.1f}")
    strength = max(votes) if votes else 0.0
    add("momentum", w.momentum, strength,
        f"Momentum [{', '.join(parts) if parts else 'n/a'}] favors {direction.value}")

    # --- Trend-strength group (10): EMA + MACD + ADX merged into ONE vote ---
    members: list[float] = []
    tparts: list[str] = []
    ema20 = _num(entry_row, "ind_ema20")
    ema50 = _num(entry_row, "ind_ema50")
    if _finite(ema20) and _finite(ema50):
        members.append(1.0 if sign * (ema20 - ema50) > 0 else 0.0)
        tparts.append("EMA20/50 aligned")
    if _finite(ema20) and _finite(close) and _finite(atr) and atr > 0:
        members.append(_clamp(sign * (close - ema20) / (2.0 * atr)))
        tparts.append("price vs EMA20")
    macd_line = _num(entry_row, "ind_macd_macd")
    if not _finite(macd_line):  # tolerate a bare ``ind_macd`` column too
        macd_line = _num(entry_row, "ind_macd")
    if _finite(macd_line) and _finite(atr) and atr > 0:
        members.append(_clamp(0.5 + 0.5 * sign * macd_line / atr))
        tparts.append(f"MACD {macd_line:+.4g}")
    adx_v = _num(entry_row, "ind_adx_adx")
    if _finite(adx_v):
        members.append(_clamp((adx_v - 15.0) / 25.0))
        tparts.append(f"ADX {adx_v:.1f}")
    strength = sum(members) / len(members) if members else 0.0
    add("trend_strength", w.trend_strength, strength,
        f"Trend strength [{', '.join(tparts) if tparts else 'n/a'}] supports {direction.value}")

    # --- Trigger (10): candlestick pattern in our direction -----------------
    own = ("pat_bullish_engulfing", "pat_pin_bar_bull") if direction is Direction.LONG \
        else ("pat_bearish_engulfing", "pat_pin_bar_bear")
    fired = [c.replace("pat_", "") for c in own if _flag(entry_row, c)]
    opposing = any(_flag(entry_row, c) for c in (
        ("pat_bearish_engulfing", "pat_pin_bar_bear") if direction is Direction.LONG
        else ("pat_bullish_engulfing", "pat_pin_bar_bull")
    ))
    add("trigger", w.trigger, 1.0 if fired else 0.0,
        f"Trigger: {', '.join(fired) if fired else ('opposing pattern present' if opposing else 'none')}")

    total = min(sum(g.points for g in groups), 100.0)
    return DirectionScore(direction=direction, score=round(total, 2), groups=groups)


def score_both(
    entry_row: pd.Series,
    htf_row: pd.Series | None,
    weights: ScoreWeights | None = None,
) -> tuple[DirectionScore, DirectionScore]:
    """Score LONG and SHORT independently for the same bar.

    Returns:
        ``(long_score, short_score)`` as :class:`DirectionScore` objects.
    """
    w = (weights or ScoreWeights()).validated()
    return (
        score_direction(Direction.LONG, entry_row, htf_row, w),
        score_direction(Direction.SHORT, entry_row, htf_row, w),
    )
