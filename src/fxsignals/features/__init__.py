"""Feature layer: indicators, swings, structure, levels, sessions, patterns.

All functions are pure pandas/numpy, take Candle-contract DataFrames and are
strictly causal (bar ``t`` only uses bars ``<= t``). Right-side-confirmed
features such as swings are stamped at their confirmation bar.
"""

from fxsignals.features.indicators import (
    adx,
    atr,
    bollinger,
    ema,
    macd,
    rsi,
    sma,
    stochastic,
)
from fxsignals.features.levels import (
    SupportResistanceZone,
    nearest_zone,
    support_resistance_zones,
    zone_distance_atr,
)
from fxsignals.features.patterns import (
    bearish_engulfing,
    bullish_engulfing,
    inside_bar,
    pin_bar_bear,
    pin_bar_bull,
)
from fxsignals.features.pipeline import (
    FEATURE_GROUPS,
    INDICATOR_GROUPS,
    build_features,
    build_mtf_features,
)
from fxsignals.features.sessions import SessionConfig, session_features
from fxsignals.features.structure import StructureResult, market_structure
from fxsignals.features.swings import SwingSet, detect_swings, last_confirmed_swings

__all__ = [
    "FEATURE_GROUPS",
    "INDICATOR_GROUPS",
    "SessionConfig",
    "StructureResult",
    "SupportResistanceZone",
    "SwingSet",
    "adx",
    "atr",
    "bearish_engulfing",
    "bollinger",
    "build_features",
    "build_mtf_features",
    "bullish_engulfing",
    "detect_swings",
    "ema",
    "inside_bar",
    "last_confirmed_swings",
    "macd",
    "market_structure",
    "nearest_zone",
    "pin_bar_bear",
    "pin_bar_bull",
    "rsi",
    "session_features",
    "sma",
    "stochastic",
    "support_resistance_zones",
    "zone_distance_atr",
]
