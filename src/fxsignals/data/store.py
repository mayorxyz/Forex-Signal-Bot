"""Local candle cache with Parquet backend (CSV fallback if pyarrow is absent)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from fxsignals.config import Settings
from fxsignals.data.base import DataProvider
from fxsignals.models import Timeframe

logger = logging.getLogger(__name__)


def _parquet_available() -> bool:
    """True when pandas can write/read Parquet (pyarrow or fastparquet)."""
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


class CandleStore:
    """Persist provider-fetched candles so repeated runs hit the disk once.

    Files are keyed by pair and timeframe, e.g. ``EURUSD_H1.parquet`` (or
    ``.csv`` when pyarrow is unavailable).
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.use_parquet = _parquet_available()

    @classmethod
    def from_settings(cls, settings: Settings) -> "CandleStore":
        """Build a store using ``paths.cache_dir`` from application settings."""
        return cls(settings.paths.cache_dir)

    def path_for(self, pair: str, timeframe: Timeframe) -> Path:
        """Cache file path for a pair/timeframe (extension follows backend)."""
        ext = "parquet" if self.use_parquet else "csv"
        return self.cache_dir / f"{pair.upper()}_{timeframe.name}.{ext}"

    def save_candles(self, pair: str, timeframe: Timeframe, df: pd.DataFrame) -> Path:
        """Write candles to the cache; returns the written path."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(pair, timeframe)
        out = df.copy()
        out.index = out.index.tz_localize(None) if out.index.tz else out.index
        if self.use_parquet:
            out.to_parquet(path)
        else:
            flat = out.reset_index()
            flat["time"] = flat["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            flat.to_csv(path, index=False)
        logger.debug("Cached %d bars for %s/%s at %s", len(df), pair, timeframe.name, path)
        return path

    def load_candles(self, pair: str, timeframe: Timeframe) -> pd.DataFrame | None:
        """Read cached candles, re-localizing the index to UTC. None if absent."""
        path = self.path_for(pair, timeframe)
        if not path.is_file():
            return None
        if self.use_parquet:
            raw = pd.read_parquet(path)
        else:
            raw = pd.read_csv(path)
            if "time" in raw.columns:
                raw = raw.set_index("time")
        raw.index = pd.DatetimeIndex(raw.index, name="time").tz_localize("UTC")
        return raw.sort_index()

    def get_or_fetch(
        self,
        provider: DataProvider,
        pair: str,
        timeframe: Timeframe,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Return cached candles, fetching (and caching) them on a miss.

        Args:
            provider: Source used when the cache has no entry.
            pair: Instrument symbol.
            timeframe: Bar interval.
            refresh: When True, ignore any existing cache and re-fetch.

        Returns:
            Contract-compliant candles.
        """
        if not refresh:
            cached = self.load_candles(pair, timeframe)
            if cached is not None and not cached.empty:
                return cached
        df = provider.get_candles(pair, timeframe)
        self.save_candles(pair, timeframe, df)
        return df
