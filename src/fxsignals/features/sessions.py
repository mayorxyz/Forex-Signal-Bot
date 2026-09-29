"""Trading-session tagging in UTC with configurable hour ranges.

Sessions are defined by the *bar open time* in UTC hours, half-open interval
``[start, end)``; ranges may wrap past midnight (e.g. Asia 23:00 -> 21:00 is
not needed here because defaults use non-wrapping windows, but wrapping is
supported). All outputs are boolean Series aligned to the candle index — no
warm-up required and no look-ahead (each bar only inspects its own timestamp).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SessionConfig:
    """UTC hour ranges for the major sessions.

    Attributes:
        asia: ``(start_hour, end_hour)`` for the Asian/Tokyo session.
        london: London session window (includes the pre-London Frankfurt move).
        newyork: New York session window.
        london_open_hour: Optional explicit single-hour set treated as
            "London open" business hours for the ``is_london_open_hour`` flag.
    """

    asia: tuple[int, int] = (1, 9)
    london: tuple[int, int] = (7, 16)
    newyork: tuple[int, int] = (12, 21)
    london_open_hour: tuple[int, int] = (8, 17)

    def validated(self) -> "SessionConfig":
        """Return self after checking every window is a valid hour pair.

        Raises:
            ValueError: if any bound is outside 0-23.
        """
        for name, (lo, hi) in (
            ("asia", self.asia),
            ("london", self.london),
            ("newyork", self.newyork),
            ("london_open_hour", self.london_open_hour),
        ):
            if not (0 <= lo <= 23 and 0 <= hi <= 24):
                raise ValueError(f"invalid {name} session window {(lo, hi)}")
        return self


DEFAULT_SESSIONS = SessionConfig()


def _in_window(hours: pd.Series, window: tuple[int, int]) -> pd.Series:
    """Boolean membership of UTC hours in a possibly midnight-wrapping window."""
    lo, hi = window
    if lo <= hi:
        return (hours >= lo) & (hours < hi)
    return (hours >= lo) | (hours < hi)  # wraps past midnight


def session_features(
    df: pd.DataFrame, config: SessionConfig | None = None
) -> pd.DataFrame:
    """Tag each bar with its active UTC session(s).

    Args:
        df: Candle-contract DataFrame with a tz-aware UTC DatetimeIndex.
        config: Optional :class:`SessionConfig`; defaults used otherwise.

    Returns:
        Boolean DataFrame aligned to ``df.index`` with columns ``asia``,
        ``london``, ``newyork``, ``overlap`` (London ∩ New York) and
        ``is_london_open_hour`` (bar open inside configured London business
        hours).

    Raises:
        TypeError: if the index is not a DatetimeIndex or is naive (no tz).
    """
    cfg = (config or DEFAULT_SESSIONS).validated()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("session_features requires a DatetimeIndex")
    if df.index.tz is None:
        raise TypeError("session_features requires a timezone-aware UTC index")

    hours = df.index.tz_convert("UTC").hour
    hours = pd.Series(hours, index=df.index)

    asia = _in_window(hours, cfg.asia)
    london = _in_window(hours, cfg.london)
    newyork = _in_window(hours, cfg.newyork)
    london_open = _in_window(hours, cfg.london_open_hour)

    return pd.DataFrame(
        {
            "asia": asia,
            "london": london,
            "newyork": newyork,
            "overlap": london & newyork,
            "is_london_open_hour": london_open,
        },
        index=df.index,
    )
