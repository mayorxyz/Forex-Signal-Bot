"""Trade-level construction (entry / stop / target) for SIGNALS only.

This module computes price levels and rejects setups that do not meet the
risk rules. It never sends, modifies or cancels an order — output is analysis
data consumed by :mod:`fxsignals.signals.engine`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from fxsignals.features.levels import SupportResistanceZone, nearest_zone
from fxsignals.models import Direction


class RiskError(ValueError):
    """Raised when a setup cannot produce valid risk levels."""


def pip_size(pair: str) -> float:
    """Return the pip size for ``pair``.

    Convention: 0.0001 for standard quotes, 0.01 for JPY-cross pairs, and
    0.1 for gold (XAUUSD and other XAU quotes). Defaults to 0.0001.

    Args:
        pair: Instrument symbol, e.g. ``'USDJPY'`` or ``'XAUUSD'``.

    Returns:
        Positive float pip size.
    """
    sym = pair.strip().upper()
    if "JPY" in sym:
        return 0.01
    if sym.startswith("XAU") or sym.endswith("XAU"):
        return 0.1
    return 0.0001


@dataclass(frozen=True)
class RiskConfig:
    """Risk-model knobs for level construction.

    Attributes:
        rr_target: Reward:risk ratio used to project take profit.
        min_rr: Minimum acceptable realized R:R (else reject).
        min_sl_atr / max_sl_atr: Allowed stop distance as multiples of ATR.
        sl_atr_buffer: Extra ATR buffer beyond the protecting swing/zone.
    """

    rr_target: float = 2.0
    min_rr: float = 1.5
    min_sl_atr: float = 0.5
    max_sl_atr: float = 3.0
    sl_atr_buffer: float = 0.25

    def validated(self) -> "RiskConfig":
        """Return self after sanity-checking every bound.

        Raises:
            ValueError: on non-positive or contradictory values.
        """
        for name in ("rr_target", "min_rr", "min_sl_atr", "max_sl_atr", "sl_atr_buffer"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be a finite non-negative number, got {v!r}")
        if self.min_sl_atr <= 0 or self.max_sl_atr < self.min_sl_atr:
            raise ValueError(
                f"need 0 < min_sl_atr <= max_sl_atr, got {self.min_sl_atr}, {self.max_sl_atr}"
            )
        if self.rr_target <= 0 or self.min_rr <= 0:
            raise ValueError("rr_target and min_rr must be positive")
        return self


@dataclass(frozen=True)
class TradeLevels:
    """Computed levels for one hypothetical trade (never executed here).

    Attributes:
        entry: Reference entry price (last close; acted on at next bar open).
        stop_loss: Protective stop beyond the opposing swing/zone.
        take_profit: Target from the R:R plan or the next opposing zone.
        rr: Realized reward:risk ratio.
        sl_pips / tp_pips: Distances expressed in pair-aware pips.
        note: Human-readable summary appended to signal reasons.
    """

    entry: float
    stop_loss: float
    take_profit: float
    rr: float
    sl_pips: float
    tp_pips: float
    note: str


def _protecting_level(direction: Direction, row_like: dict[str, float | None]) -> float | None:
    """Nearest confirmed swing guarding the trade direction."""
    key = "st_last_swing_low" if direction is Direction.LONG else "st_last_swing_high"
    val = row_like.get(key)
    if val is None or not math.isfinite(float(val)):
        return None
    return float(val)


def build_trade_levels(
    direction: Direction,
    entry: float,
    atr: float,
    zones: list[SupportResistanceZone] | tuple[SupportResistanceZone, ...],
    cfg: RiskConfig | None = None,
    last_swing_low: float | None = None,
    last_swing_high: float | None = None,
) -> TradeLevels:
    """Build SL/TP for ``direction`` or raise :class:`RiskError`.

    Stop logic: place the stop beyond the *nearest opposing* protecting level
    (swing low for longs / swing high for shorts, plus any S/R zone on that
    side), widened by ``sl_atr_buffer`` ATR, then cap the distance into
    ``[min_sl_atr, max_sl_atr] * ATR``. Take profit uses the R:R target, but
    if the next opposing zone lies closer than the target, TP is pulled just
    short of that zone — provided the resulting R:R stays valid.

    Args:
        direction: LONG or SHORT.
        entry: Reference entry price (> 0).
        atr: Current ATR value (> 0).
        zones: Zones known at this bar (may be empty).
        cfg: Optional :class:`RiskConfig` (defaults validated).
        last_swing_low: Last confirmed swing low, if any.
        last_swing_high: Last confirmed swing high, if any.

    Returns:
        A :class:`TradeLevels` instance.

    Raises:
        RiskError: if inputs are invalid, the SL distance falls outside the
            ATR bounds, or the resulting R:R is below ``min_rr``.
    """
    c = (cfg or RiskConfig()).validated()
    if not all(math.isfinite(x) for x in (entry, atr)) or entry <= 0 or atr <= 0:
        raise RiskError(f"entry and atr must be positive finite numbers, got {entry}, {atr}")

    row = {"st_last_swing_low": last_swing_low, "st_last_swing_high": last_swing_high}
    prot = _protecting_level(direction, row)

    # Zone edges on the protective side count as additional candidate levels.
    side_zones = [
        z for z in zones
        if (direction is Direction.LONG and z.high < entry)
        or (direction is Direction.SHORT and z.low > entry)
    ]
    if side_zones:
        z = nearest_zone(entry, side_zones)
        if z is not None:
            edge = z.high if direction is Direction.LONG else z.low
            prot = edge if prot is None else (min(prot, edge) if direction is Direction.LONG
                                              else max(prot, edge))

    if prot is None:
        raise RiskError(f"no swing/zone reference available for {direction.value} stop placement")

    buffer = c.sl_atr_buffer * atr
    if direction is Direction.LONG:
        raw_dist = entry - (prot - buffer)
    else:
        raw_dist = (prot + buffer) - entry
    dist = min(max(raw_dist, c.min_sl_atr * atr), c.max_sl_atr * atr)
    if raw_dist < c.min_sl_atr * atr - 1e-12 or raw_dist > c.max_sl_atr * atr + 1e-12:
        reason = "too tight" if raw_dist < c.min_sl_atr * atr else "too wide"
        dist_note = f"capped {reason} to {dist / atr:.2f} ATR"
    else:
        dist_note = f"{dist / atr:.2f} ATR"

    stop = entry - dist if direction is Direction.LONG else entry + dist
    reward = c.rr_target * dist

    # Opposing zones may cap TP (take profit just before the wall).
    opp_zones = [
        z for z in zones
        if (direction is Direction.LONG and z.low > stop)
        or (direction is Direction.SHORT and z.high < stop)
    ]
    capped_by_zone = False
    if opp_zones:
        wall = min(z.low for z in opp_zones) if direction is Direction.LONG \
            else max(z.high for z in opp_zones)
        wall_reward = (wall - entry) if direction is Direction.LONG else (entry - wall)
        if 0 < wall_reward < reward:
            reward = wall_reward - 0.25 * atr if wall_reward > 0.5 * atr else wall_reward * 0.9
            capped_by_zone = True

    tp = entry + reward if direction is Direction.LONG else entry - reward
    rr = reward / dist if dist > 0 else float("nan")
    if not math.isfinite(rr) or rr < c.min_rr:
        raise RiskError(
            f"setup rejected: realized RR {rr:.2f} < min_rr {c.min_rr} "
            f"({direction.value}, stop {dist / atr:.2f} ATR)"
        )

    note = (
        f"Risk: SL {dist / atr:.2f} ATR ({dist_note}) beyond "
        f"{'swing low' if direction is Direction.LONG else 'swing high'} {prot:.5g}"
        + (", TP capped by opposing zone" if capped_by_zone else "")
        + f", RR {rr:.2f}"
    )
    return TradeLevels(
        entry=float(entry),
        stop_loss=float(stop),
        take_profit=float(tp),
        rr=float(rr),
        sl_pips=float(dist),   # provisional; engine converts with pair pip size
        tp_pips=float(reward),
        note=note,
    )


def set_pips(levels: TradeLevels, pair: str) -> TradeLevels:
    """Return ``levels`` with ``sl_pips``/``tp_pips`` converted for ``pair``."""
    ps = pip_size(pair)
    return TradeLevels(
        entry=levels.entry,
        stop_loss=levels.stop_loss,
        take_profit=levels.take_profit,
        rr=levels.rr,
        sl_pips=abs(levels.stop_loss - levels.entry) / ps,
        tp_pips=abs(levels.take_profit - levels.entry) / ps,
        note=levels.note,
    )
