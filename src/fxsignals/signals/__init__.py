"""Signal engine: features in, trade signals out (analysis only).

Public surface:

- :class:`fxsignals.signals.engine.SignalEngine` — evaluate the latest closed
  bar (:meth:`~fxsignals.signals.engine.SignalEngine.evaluate`) or replay full
  history (:meth:`~fxsignals.signals.engine.SignalEngine.evaluate_history`).
- Scoring: :func:`score_direction`, :func:`score_both`, :class:`ScoreWeights`,
  :class:`DirectionScore`, :class:`GroupScore`.
- Risk levels: :func:`build_trade_levels`, :class:`TradeLevels`,
  :class:`RiskConfig`, :class:`RiskError`, :func:`pip_size`, :func:`set_pips`.
- Causal filters: :func:`adx_filter`, :func:`session_filter`,
  :func:`volatility_filter`, :class:`CooldownFilter`, :class:`DuplicateFilter`,
  :class:`FilterConfig`.

No order execution exists anywhere in this package; signals are advisory data.
"""

from __future__ import annotations

from fxsignals.signals.engine import EngineConfig, SignalEngine
from fxsignals.signals.factory import build_engine_config
from fxsignals.signals.filters import (
    CooldownFilter,
    DuplicateFilter,
    FilterConfig,
    adx_filter,
    session_filter,
    volatility_filter,
)
from fxsignals.signals.risk import (
    RiskConfig,
    RiskError,
    TradeLevels,
    build_trade_levels,
    pip_size,
    set_pips,
)
from fxsignals.signals.scoring import (
    DirectionScore,
    GroupScore,
    ScoreWeights,
    score_both,
    score_direction,
)

__all__ = [
    "CooldownFilter",
    "DirectionScore",
    "DuplicateFilter",
    "EngineConfig",
    "FilterConfig",
    "GroupScore",
    "RiskConfig",
    "RiskError",
    "ScoreWeights",
    "SignalEngine",
    "TradeLevels",
    "adx_filter",
    "build_engine_config",
    "build_trade_levels",
    "pip_size",
    "score_both",
    "score_direction",
    "session_filter",
    "set_pips",
    "volatility_filter",
]
