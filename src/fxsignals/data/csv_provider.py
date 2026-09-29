"""CSV file data provider.

Reads bars from ``{data_dir}/{PAIR}_{TF}.csv`` where each file has columns
``time,open,high,low,close,volume``. The ``volume`` column may be absent and
is then filled with 0. All times are parsed as UTC.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from fxsignals.config import Settings
from fxsignals.data.base import DataProvider, ProviderError, register_provider
from fxsignals.data.normalize import normalize_candles, slice_window
from fxsignals.models import Timeframe


@register_provider("csv")
class CsvProvider(DataProvider):
    """Filesystem-backed provider reading per-pair/per-timeframe CSV files."""

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        data_dir = getattr(getattr(settings, "paths", None), "data_dir", "./data")
        self.data_dir = Path(data_dir)

    def path_for(self, pair: str, timeframe: Timeframe) -> Path:
        """Return the expected CSV path for a pair/timeframe combination."""
        return self.data_dir / f"{pair.upper()}_{timeframe.name}.csv"

    def get_candles(
        self,
        pair: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Load candles from CSV and return them matching the Candle contract.

        Raises:
            ProviderError: if the file does not exist or cannot be parsed.
        """
        path = self.path_for(pair, timeframe)
        if not path.is_file():
            raise ProviderError(f"No CSV file found at {path}")
        try:
            raw = pd.read_csv(path)
        except (pd.errors.ParserError, UnicodeDecodeError) as exc:
            raise ProviderError(f"Failed to parse {path}: {exc}") from exc

        if raw.empty:
            from fxsignals.models import empty_candles

            return empty_candles()
        try:
            df = normalize_candles(raw)
        except KeyError as exc:
            raise ProviderError(f"CSV file {path} is malformed: {exc}") from exc
        return slice_window(df, start=start, end=end, limit=limit)
