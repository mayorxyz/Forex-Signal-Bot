"""fxsignals: foundation of a forex signal bot (analysis and signals only).

This package provides configuration loading, market-data models, pluggable
data providers (CSV / synthetic), validation, resampling, caching and logging.
It intentionally contains no order-execution code.
"""

__version__ = "0.1.0"

from fxsignals.data import csv_provider as _csv_provider  # registers "csv"
from fxsignals.data import synthetic as _synthetic  # registers "synthetic"
from fxsignals.models import Direction, Signal, Timeframe
from fxsignals.signals.engine import SignalEngine

__all__ = ["Direction", "Signal", "SignalEngine", "Timeframe", "__version__"]
