"""Abstract data provider interface and provider registry.

Future API providers (OANDA, MT5, Twelve Data, ...) plug in by subclassing
:class:`DataProvider`, implementing :meth:`get_candles`, and decorating the
class with ``@register_provider("name")``. No other code changes needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Type

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - avoid circular import at runtime
    from fxsignals.config import Settings

from fxsignals.models import Timeframe


class ProviderError(Exception):
    """Raised when a provider cannot supply the requested data."""


class DataProvider(ABC):
    """Base class for all candle-data sources.

    Subclasses must return DataFrames matching the Candle contract defined in
    :mod:`fxsignals.models`: UTC tz-aware DatetimeIndex named ``time``, columns
    ``open, high, low, close, volume`` in that order, sorted ascending, no
    duplicate timestamps, positive prices and ``high >= max(open, close)``,
    ``low <= min(open, close)``.
    """

    def __init__(self, settings: "Settings | None" = None) -> None:
        self.settings = settings

    @abstractmethod
    def get_candles(
        self,
        pair: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Return candles for ``pair`` on ``timeframe``.

        Args:
            pair: Instrument symbol, e.g. ``'EURUSD'``.
            timeframe: Bar interval.
            start: Optional inclusive UTC start time.
            end: Optional inclusive UTC end time.
            limit: If given, return at most the last ``limit`` bars.

        Returns:
            A DataFrame satisfying the Candle contract.

        Raises:
            ProviderError: if data cannot be produced for the request.
        """


_REGISTRY: dict[str, Type[DataProvider]] = {}


def register_provider(name: str) -> Callable[[Type[DataProvider]], Type[DataProvider]]:
    """Class decorator registering a provider implementation under ``name``.

    Args:
        name: Lookup key used by :func:`get_provider` and by ``provider`` in
            settings.yaml.

    Returns:
        A decorator that registers and returns the class unchanged.

    Raises:
        ValueError: if the name is already taken or the class is not a
            ``DataProvider`` subclass.
    """
    key = name.strip().lower()

    def decorator(cls: Type[DataProvider]) -> Type[DataProvider]:
        if not issubclass(cls, DataProvider):
            raise ValueError(f"{cls.__name__} must subclass DataProvider to be registered")
        if key in _REGISTRY and _REGISTRY[key] is not cls:
            raise ValueError(f"Provider name {key!r} is already registered")
        _REGISTRY[key] = cls
        return cls

    return decorator


def get_provider(name: str, settings: "Settings | None" = None) -> DataProvider:
    """Instantiate the provider registered under ``name``.

    Args:
        name: Registered provider key, e.g. ``'csv'`` or ``'synthetic'``.
        settings: Application settings forwarded to the provider.

    Returns:
        A ready-to-use :class:`DataProvider` instance.

    Raises:
        ProviderError: if the name is unknown.
    """
    key = name.strip().lower()
    if key not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise ProviderError(f"Unknown provider {name!r}; registered providers: {known}")
    return _REGISTRY[key](settings)


def available_providers() -> tuple[str, ...]:
    """Names of all currently registered providers (sorted)."""
    return tuple(sorted(_REGISTRY))
