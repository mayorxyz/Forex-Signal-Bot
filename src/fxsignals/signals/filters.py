"""Causal pre-signal filters. Each returns ``(passed, reason)``.

Every filter may only look at bars up to and including the evaluated bar —
percentile bands use a trailing window, cooldown uses past signal times, and
session tags come from the (already causal) feature frame. Stateful filters
(cooldown, duplicate suppression) are class-based so live scans and history
backtests share identical semantics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fxsignals.features.sessions import DEFAULT_SESSIONS, SessionConfig, _in_window
from fxsignals.models import Direction


@dataclass(frozen=True)
class FilterConfig:
    """Knobs for the pre-signal filter stack.

    Attributes:
        min_adx: Minimum entry-TF ADX required (skip dead ranges). Use 0 to
            disable.
        allowed_sessions: Session tags that may generate signals; any of
            ``asia/london/newyork/overlap``. Empty tuple disables the filter.
        atr_pct_low / atr_pct_high: ATR must fall inside this percentile band
            of its own trailing history.
        atr_lookback: Trailing window (bars) used for the percentile band.
        atr_min_history: Minimum valid ATR samples before the band applies.
        cooldown_bars: Bars to wait after a pair+direction signal.
        dup_entry_atr: Re-entry suppression radius around a previous signal's
            entry, in ATR units (same pair + direction + zone).
    """

    min_adx: float = 18.0
    allowed_sessions: tuple[str, ...] = ("london", "newyork", "overlap")
    atr_pct_low: float = 10.0
    atr_pct_high: float = 95.0
    atr_lookback: int = 180
    atr_min_history: int = 30
    cooldown_bars: int = 12
    dup_entry_atr: float = 0.75

    def validated(self) -> "FilterConfig":
        """Return self after sanity-checking values.

        Raises:
            ValueError: on out-of-range percentiles or negative windows.
        """
        if not 0 <= self.atr_pct_low < self.atr_pct_high <= 100:
            raise ValueError(
                f"need 0 <= atr_pct_low < atr_pct_high <= 100, got {self.atr_pct_low}, {self.atr_pct_high}"
            )
        if self.atr_lookback < 2 or self.atr_min_history < 1:
            raise ValueError("atr_lookback must be >= 2 and atr_min_history >= 1")
        if self.cooldown_bars < 0 or self.min_adx < 0 or self.dup_entry_atr < 0:
            raise ValueError("cooldown_bars, min_adx and dup_entry_atr must be >= 0")
        valid_tags = {"asia", "london", "newyork", "overlap"}
        unknown = set(self.allowed_sessions) - valid_tags
        if unknown:
            raise ValueError(f"unknown session tags in allowed_sessions: {sorted(unknown)}")
        return self


def adx_filter(row: pd.Series, cfg: FilterConfig) -> tuple[bool, str]:
    """Reject dead ranges: entry-TF ADX must reach ``cfg.min_adx``.

    Args:
        row: Feature row containing ``ind_adx_adx``.
        cfg: Validated filter config.

    Returns:
        ``(passed, reason)``; NaN ADX (warm-up) fails.
    """
    if cfg.min_adx <= 0:
        return True, "adx filter disabled"
    try:
        adx = float(row.get("ind_adx_adx"))
    except (TypeError, ValueError):
        adx = float("nan")
    if not math.isfinite(adx):
        return False, "ADX unavailable (indicator warm-up)"
    if adx < cfg.min_adx:
        return False, f"ADX {adx:.1f} below minimum {cfg.min_adx:.1f} (dead range)"
    return True, f"ADX {adx:.1f} >= {cfg.min_adx:.1f}"


def session_filter(
    timestamp: pd.Timestamp,
    cfg: FilterConfig,
    sessions: SessionConfig | None = None,
) -> tuple[bool, str]:
    """Allow signals only inside configured UTC sessions.

    Args:
        timestamp: Bar open time (tz-aware UTC).
        cfg: Filter config with ``allowed_sessions``.
        sessions: Session windows (defaults applied when None).

    Returns:
        ``(passed, reason)``.
    """
    if not cfg.allowed_sessions:
        return True, "session filter disabled"
    sc = (sessions or DEFAULT_SESSIONS).validated()
    hour = pd.Timestamp(timestamp).tz_convert("UTC").hour
    h = pd.Series([hour])

    tags = {
        "asia": bool(_in_window(h, sc.asia).iloc[0]),
        "london": bool(_in_window(h, sc.london).iloc[0]),
        "newyork": bool(_in_window(h, sc.newyork).iloc[0]),
    }
    tags["overlap"] = tags["london"] and tags["newyork"]
    active = sorted(t for t, on in tags.items() if on)
    if any(tags[t] for t in cfg.allowed_sessions):
        return True, f"session {active or 'off-hours'} allowed"
    return False, f"session {active or 'off-hours'} not in {list(cfg.allowed_sessions)}"


def volatility_filter(atr_series: pd.Series, pos: int, cfg: FilterConfig) -> tuple[bool, str]:
    """Require current ATR inside a trailing percentile band of itself.

    Only values at indices ``<= pos`` are inspected (causal); the band is
    computed over the last ``cfg.atr_lookback`` bars excluding the current
    one. Warm-up (fewer than ``atr_min_history`` samples) passes by default.

    Args:
        atr_series: ATR column aligned to the entry candles.
        pos: Integer position of the evaluated bar.
        cfg: Filter config.

    Returns:
        ``(passed, reason)``.
    """
    cur = float(atr_series.iloc[pos]) if pos < len(atr_series) else float("nan")
    if not math.isfinite(cur) or cur <= 0:
        return False, "ATR unavailable (warm-up)"
    lo_i = max(0, pos - cfg.atr_lookback)
    hist = np.asarray(atr_series.iloc[lo_i:pos], dtype=float)
    hist = hist[np.isfinite(hist) & (hist > 0)]
    if len(hist) < cfg.atr_min_history:
        return True, f"volatility band warming up ({len(hist)} samples)"
    low = float(np.percentile(hist, cfg.atr_pct_low))
    high = float(np.percentile(hist, cfg.atr_pct_high))
    if cur < low:
        return False, f"ATR {cur:.5g} below {cfg.atr_pct_low:.0f}th pct ({low:.5g}): too quiet"
    if cur > high:
        return False, f"ATR {cur:.5g} above {cfg.atr_pct_high:.0f}th pct ({high:.5g}): too volatile"
    return True, f"ATR {cur:.5g} inside [{low:.5g}, {high:.5g}] band"


class CooldownFilter:
    """Suppress a pair+direction for ``cfg.cooldown_bars`` after each signal."""

    def __init__(self, cfg: FilterConfig) -> None:
        self._bars = cfg.validated().cooldown_bars
        self._last: dict[tuple[str, Direction], pd.Timestamp] = {}

    def check(
        self, pair: str, direction: Direction, timestamp: pd.Timestamp, bar_index: pd.Index
    ) -> tuple[bool, str]:
        """Return ``(passed, reason)`` and record the signal when it passes.

        The distance is measured in *bars* using ``bar_index`` positions, so
        missing weekend bars do not distort the cooldown length.
        """
        ts = pd.Timestamp(timestamp)
        key = (pair.upper(), direction)
        prev = self._last.get(key)
        if prev is not None and self._bars > 0:
            try:
                elapsed = int(bar_index.get_loc(ts)) - int(bar_index.get_loc(prev))
            except KeyError:
                elapsed = -1  # previous stamp not in this index: treat as expired
            if 0 <= elapsed < self._bars:
                return False, f"cooldown: {key[1].value} signalled {elapsed} bar(s) ago (<{self._bars})"
        self._last[key] = ts
        return True, "cooldown clear"


class DuplicateFilter:
    """Suppress re-emission of the same pair/direction/entry-zone signal."""

    def __init__(self, cfg: FilterConfig) -> None:
        self._radius_atr = cfg.validated().dup_entry_atr
        self._seen: list[tuple[str, Direction, float]] = []

    def check(
        self, pair: str, direction: Direction, entry: float, atr: float
    ) -> tuple[bool, str]:
        """Return ``(passed, reason)``; passing records the new entry zone."""
        if self._radius_atr <= 0:
            return True, "duplicate suppression disabled"
        radius = self._radius_atr * atr
        for p, d, e in self._seen:
            if p == pair.upper() and d is direction and abs(e - entry) <= radius:
                return False, (
                    f"duplicate: {d.value} near entry {e:.5g} (within {radius:.5g}) already emitted"
                )
        self._seen.append((pair.upper(), direction, float(entry)))
        return True, "new setup"
