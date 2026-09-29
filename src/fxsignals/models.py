"""Core data models for fxsignals.

Defines the timeframe enum, the canonical Candle-DataFrame contract and the
Signal dataclass. All timestamps in this project are timezone-aware UTC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import pandas as pd

CANDLE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


# Central timeframe mapping (single source of truth). Durations are expressed
# as pandas 3 compatible unit strings ("min"/"h"/"D"/"W"); the legacy "H"/"T"
# aliases and unit-only Timedelta strings are no longer accepted by pandas.
TIMEFRAME_DURATIONS: dict[str, str] = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
    "W1": "1W",
}


def timeframe_rule(name: str) -> str:
    """Return the pandas resample rule for a timeframe from the central map.

    This and :func:`timeframe_timedelta` are the only places that read the
    pandas unit strings held in ``TIMEFRAME_DURATIONS``; no enum value is ever
    used as a pandas frequency string.

    Args:
        name: Timeframe code, e.g. ``'H1'`` (case-insensitive).

    Returns:
        A pandas 3 compatible offset string, e.g. ``'4h'``.

    Raises:
        ValueError: if ``name`` is not present in TIMEFRAME_DURATIONS.
    """
    key = name.strip().upper()
    try:
        return TIMEFRAME_DURATIONS[key]
    except KeyError as exc:
        valid = ", ".join(TIMEFRAME_DURATIONS)
        raise ValueError(
            f"Unknown timeframe {name!r}; expected one of: {valid}"
        ) from exc


def timeframe_timedelta(name: str) -> pd.Timedelta:
    """Return the bar duration for a timeframe code from the central mapping.

    Args:
        name: Timeframe code, e.g. ``'H1'`` (case-insensitive).

    Returns:
        The duration as a :class:`pandas.Timedelta`.

    Raises:
        ValueError: if ``name`` is not present in TIMEFRAME_DURATIONS.
    """
    return pd.Timedelta(timeframe_rule(name))


class Timeframe(Enum):
    """Supported bar intervals. Canonical labels: value == name (e.g. 'H4')."""

    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"

    @property
    def minutes(self) -> int:
        """Duration of one bar in minutes (derived from the central map)."""
        return int(timeframe_timedelta(self.name).total_seconds() // 60)

    @property
    def timedelta(self) -> pd.Timedelta:
        """Bar duration as a pandas Timedelta (from the central mapping)."""
        return timeframe_timedelta(self.name)

    @property
    def rule(self) -> str:
        """Pandas resample rule for this timeframe (from the central map)."""
        return timeframe_rule(self.name)

    @classmethod
    def from_str(cls, name: str) -> "Timeframe":
        """Parse a timeframe from its canonical code (``'H1'``, case-insensitive).

        Raises:
            ValueError: if the string does not match any known timeframe.
        """
        key = name.strip().upper()
        for tf in cls:
            if key == tf.name:
                return tf
        valid = ", ".join(tf.name for tf in cls)
        raise ValueError(f"Unknown timeframe {name!r}; expected one of: {valid}")


class Direction(str, Enum):
    """Trade direction of a signal. Signals only - never executed here."""

    LONG = "LONG"
    SHORT = "SHORT"


def empty_candles() -> pd.DataFrame:
    """Return an empty DataFrame that satisfies the Candle contract."""
    idx = pd.DatetimeIndex([], tz="UTC", name="time")
    return pd.DataFrame(
        {col: pd.Series(dtype="float64") for col in CANDLE_COLUMNS}, index=idx
    )


@dataclass(frozen=True)
class Signal:
    """A trade idea produced by the analysis layer.

    Attributes:
        pair: Instrument symbol, e.g. ``'EURUSD'``.
        direction: LONG or SHORT.
        entry: Recommended entry price.
        stop_loss: Protective stop price.
        take_profit: Target price.
        rr: Risk/reward ratio (reward / risk), ``None`` if not computable.
        confidence: Heuristic confidence score, 0-100 inclusive.
        timeframe: Timeframe the signal was generated on.
        reasons: Human-readable explanation strings.
        timestamp: UTC time the signal was created.
    """

    pair: str
    direction: Direction
    entry: float
    stop_loss: float
    take_profit: float
    rr: float | None
    confidence: float
    timeframe: Timeframe
    reasons: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.confidence) <= 100.0:
            raise ValueError(f"confidence must be in [0, 100], got {self.confidence}")
        if self.entry <= 0 or self.stop_loss <= 0 or self.take_profit <= 0:
            raise ValueError("entry, stop_loss and take_profit must be positive prices")
        if self.timestamp.tzinfo is None:
            raise ValueError("Signal.timestamp must be timezone-aware UTC")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the signal to a JSON-friendly plain dictionary."""
        return {
            "pair": self.pair,
            "direction": self.direction.value,
            "entry": self.entry,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "rr": self.rr,
            "confidence": self.confidence,
            "timeframe": self.timeframe.name,
            "reasons": list(self.reasons),
            "timestamp": self.timestamp.isoformat(),
        }
