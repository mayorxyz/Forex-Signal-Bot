"""Build a :class:`~fxsignals.signals.engine.EngineConfig` from settings.

Lives in its own module so :mod:`fxsignals.config` never needs to import the
signal engine (avoids a circular dependency): this file imports both sides and
maps validated :class:`~fxsignals.config.Settings` values onto the engine's
dataclasses.
"""

from __future__ import annotations

from fxsignals.config import Settings
from fxsignals.signals.engine import EngineConfig
from fxsignals.signals.filters import FilterConfig
from fxsignals.signals.risk import RiskConfig
from fxsignals.signals.scoring import ScoreWeights


def build_engine_config(settings: Settings) -> EngineConfig:
    """Translate application settings into an :class:`EngineConfig`.

    Args:
        settings: Fully validated :class:`~fxsignals.config.Settings`; its
            ``signals`` and ``filters`` sections supply every knob.

    Returns:
        A validated :class:`EngineConfig` whose weights sum to 100, with risk
        parameters (rr_target, min_rr, SL bounds/buffer) and filter parameters
        (min_adx, allowed sessions, ATR percentile band/lookback, cooldown)
        mirrored from the YAML sections.

    Raises:
        ValueError: if any mapped value fails sub-config validation.
    """
    s = settings.signals
    f = settings.filters
    return EngineConfig(
        weights=ScoreWeights.from_dict(dict(s.weights)),
        risk=RiskConfig(
            rr_target=s.rr_target,
            min_rr=s.min_rr,
            min_sl_atr=s.min_sl_atr,
            max_sl_atr=s.max_sl_atr,
            sl_atr_buffer=s.sl_atr_buffer,
        ),
        filters=FilterConfig(
            min_adx=f.min_adx,
            allowed_sessions=tuple(f.sessions),
            atr_pct_low=f.atr_percentile_low,
            atr_pct_high=f.atr_percentile_high,
            atr_lookback=f.lookback,
            cooldown_bars=s.cooldown_bars,
        ),
        min_score=s.min_score,
        direction_margin=s.direction_margin,
    ).validated()


__all__ = ["build_engine_config"]
