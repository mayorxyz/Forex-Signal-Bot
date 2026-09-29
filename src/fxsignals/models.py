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


class Timeframe(Enum):
    """Supported bar intervals. Value is the pandas offset alias."""

    M15 = "15min"
    M30 = "30min"
    H1 = "1h"
    H4 = "4h"
    D1 = "1D"

    @property
    def minutes(self) -> int:
        """Duration of one bar in minutes."""
        return _TF_MINUTES[self]

    @property
    def pandas_alias(self) -> str:
        """Offset alias usable with ``DataFrame.resample``."""
        return self.value

    @classmethod
    def from_str(cls, name: str) -> "Timeframe":
        """Parse a timeframe from its code (``'H1'``) or pandas alias.

        Raises:
            ValueError: if the string does not match any known timeframe.
        """
        key = name.strip().upper()
        for tf in cls:
            if key == tf.name or key == tf.value.upper():
                return tf
        valid = ", ".join(tf.name for tf in cls)
        raise ValueError(f"Unknown timeframe {name!r}; expected one of: {valid}")


_TF_MINUTES: dict[Timeframe, int] = {
    Timeframe.M15: 15,
    Timeframe.M30: 30,
    Timeframe.H1: 60,
    Timeframe.H4: 240,
    Timeframe.D1: 1440,
}


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
