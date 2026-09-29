"""Feature pipeline: join every feature group into one prefixed DataFrame.

Column prefixes: ``ind_`` indicators, ``sw_`` swings, ``st_`` structure,
``lv_`` levels, ``ses_`` sessions, ``pat_`` candle patterns, ``htf_`` for
bias-timeframe features aligned onto the entry timeframe.

The :data:`FEATURE_GROUPS` / :data:`INDICATOR_GROUPS` maps let a later signal
scorer count each *evidence group* once (correlated indicators such as RSI and
stochastic share the momentum group; MACD and EMAs share the trend group)
instead of double-counting individual oscillators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from fxsignals.data.resample import align_htf_to_ltf
from fxsignals.features import indicators as ind
from fxsignals.features.levels import support_resistance_zones
from fxsignals.features.patterns import (
    bearish_engulfing,
    bullish_engulfing,
    inside_bar,
    pin_bar_bear,
    pin_bar_bull,
)
from fxsignals.features.sessions import SessionConfig, session_features
from fxsignals.features.structure import market_structure
from fxsignals.features.swings import last_confirmed_swings


@dataclass(frozen=True)
class FeatureConfig:
    """Tunable parameters for the whole feature pipeline.

    Attributes:
        ema_periods: EMA lengths computed on closes (trend group).
        sma_period: SMA length (trend group).
        rsi_period / atr_period / adx_period / bb_period / macd_* / stoch_*:
            standard indicator lengths.
        swing_left / swing_right: pivot windows (confirmation is right-sided).
        structure_n: swings per kind considered for HH/HL classification.
        zone_atr_mult / zone_min_touches: S/R clustering knobs.
        engulf_min_body_atr / pin_wick_body_ratio / pin_min_wick_atr:
            pattern thresholds.
        sessions: UTC session windows (None -> defaults).
    """

    ema_periods: tuple[int, ...] = (20, 50)
    sma_period: int = 20
    rsi_period: int = 14
    atr_period: int = 14
    adx_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    stoch_k: int = 14
    stoch_d: int = 3
    swing_left: int = 3
    swing_right: int = 3
    structure_n: int = 2
    zone_atr_mult: float = 0.5
    zone_min_touches: int = 2
    engulf_min_body_atr: float = 0.5
    pin_wick_body_ratio: float = 2.0
    pin_min_wick_atr: float = 0.5
    sessions: SessionConfig | None = None


# Which evidence group each raw indicator belongs to. A scorer should credit
# one vote per GROUP, not per column, because members are correlated.
INDICATOR_GROUPS: dict[str, tuple[str, ...]] = {
    "trend": ("ema", "sma", "macd", "adx"),
    "momentum": ("rsi", "stochastic", "bollinger_width"),
    "volatility": ("atr", "bollinger"),
    "structure": ("swings", "levels", "bos", "choch"),
    "pattern": ("candle_patterns",),
    "time": ("sessions",),
}

# Populated by build_features(): group prefix -> concrete column names.
FEATURE_GROUPS: dict[str, list[str]] = {
    "indicators": [],
    "swings": [],
    "structure": [],
    "levels": [],
    "sessions": [],
    "patterns": [],
    "htf": [],
}


def _prefix(df: pd.DataFrame, pfx: str, group: str) -> pd.DataFrame:
    """Rename columns with ``pfx`` and record them under ``group`` (exact set)."""
    out = df.copy()
    out.columns = [f"{pfx}{c}" for c in out.columns]
    FEATURE_GROUPS[group] = list(out.columns)
    return out


def build_features(
    df: pd.DataFrame, config: FeatureConfig | None = None
) -> pd.DataFrame:
    """Compute all feature groups for ``df`` and join them column-wise.

    Args:
        df: Candle-contract DataFrame (UTC DatetimeIndex).
        config: Optional :class:`FeatureConfig` overriding defaults.

    Returns:
        DataFrame aligned to ``df.index`` whose columns are prefixed by group
        (``ind_``, ``sw_``, ``st_``, ``lv_``, ``ses_``, ``pat_``). Every value
        at bar ``t`` uses bars ``<= t`` only; warm-up values are NaN.

    Raises:
        KeyError: if required candle columns are missing.
    """
    cfg = config or FeatureConfig()
    parts: list[pd.DataFrame] = []

    # --- indicators (ind_) ---
    frames: dict[str, pd.Series | pd.DataFrame] = {}
    for p in cfg.ema_periods:
        frames[f"ema{p}"] = ind.ema(df, period=p)
    frames["sma"] = ind.sma(df, period=cfg.sma_period)
    frames["rsi"] = ind.rsi(df, period=cfg.rsi_period)
    atr_series = ind.atr(df, period=cfg.atr_period)
    frames["atr"] = atr_series
    frames["macd"] = ind.macd(df, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    frames["adx"] = ind.adx(df, period=cfg.adx_period)
    frames["bb"] = ind.bollinger(df, period=cfg.bb_period, num_std=cfg.bb_std)
    frames["stoch"] = ind.stochastic(df, k_period=cfg.stoch_k, d_period=cfg.stoch_d)

    cols: dict[str, pd.Series] = {}
    for name, obj in frames.items():
        if isinstance(obj, pd.Series):
            cols[f"ind_{name}"] = obj
        else:
            for sub in obj.columns:
                cols[f"ind_{name}_{sub}"] = obj[sub]
    FEATURE_GROUPS["indicators"] = sorted(cols)
    parts.append(pd.DataFrame(cols, index=df.index))

    # --- swings (sw_) ---
    swset = last_confirmed_swings(
        df, n=cfg.structure_n, left=cfg.swing_left, right=cfg.swing_right
    )
    parts.append(_prefix(swset.frame, "sw_", "swings"))

    # --- structure (st_) ---
    struct = market_structure(
        df, left=cfg.swing_left, right=cfg.swing_right, n=cfg.structure_n
    )
    st_frame = struct.frame.copy()
    st_frame["trend_up"] = st_frame["trend_state"] == "UP"
    st_frame["trend_down"] = st_frame["trend_state"] == "DOWN"
    parts.append(_prefix(st_frame, "st_", "structure"))

    # --- levels (lv_) ---
    _, lv_per_bar = support_resistance_zones(
        df,
        atr_mult=cfg.zone_atr_mult,
        min_touches=cfg.zone_min_touches,
        left=cfg.swing_left,
        right=cfg.swing_right,
        atr_period=cfg.atr_period,
    )
    parts.append(_prefix(lv_per_bar, "lv_", "levels"))

    # --- sessions (ses_) ---
    parts.append(_prefix(session_features(df, cfg.sessions), "ses_", "sessions"))

    # --- patterns (pat_) ---
    pat = pd.DataFrame(
        {
            "bullish_engulfing": bullish_engulfing(
                df, atr_series=atr_series, min_body_atr=cfg.engulf_min_body_atr
            ),
            "bearish_engulfing": bearish_engulfing(
                df, atr_series=atr_series, min_body_atr=cfg.engulf_min_body_atr
            ),
            "pin_bar_bull": pin_bar_bull(
                df,
                atr_series=atr_series,
                wick_body_ratio=cfg.pin_wick_body_ratio,
                min_wick_atr=cfg.pin_min_wick_atr,
            ),
            "pin_bar_bear": pin_bar_bear(
                df,
                atr_series=atr_series,
                wick_body_ratio=cfg.pin_wick_body_ratio,
                min_wick_atr=cfg.pin_min_wick_atr,
            ),
            "inside_bar": inside_bar(df),
        },
        index=df.index,
    )
    parts.append(_prefix(pat, "pat_", "patterns"))

    result = pd.concat(parts, axis=1)
    result.index = df.index
    return result


def build_mtf_features(
    bias_df: pd.DataFrame,
    entry_df: pd.DataFrame,
    config: FeatureConfig | None = None,
    bias_columns: tuple[str, ...] = (
        "st_trend_state",
        "st_trend_up",
        "st_trend_down",
        "ind_adx_adx",
        "ind_ema20",
        "ind_sma",
    ),
) -> pd.DataFrame:
    """Compute bias-TF features and align them to the entry TF without leakage.

    Bias-side trend state, EMA trend and ADX are computed on ``bias_df``, then
    mapped onto every entry-TF bar through :func:`align_htf_to_ltf`, which
    exposes a bias bar only after it has closed. Resulting columns carry the
    ``htf_`` prefix; rows before the first closed bias bar are NaN/None.

    Args:
        bias_df: Candles on the higher (bias) timeframe.
        entry_df: Candles on the lower (entry) timeframe.
        config: Optional feature configuration shared by both frames.
        bias_columns: Which bias-feature columns to export; defaults cover
            trend state, EMA trend inputs and ADX strength.

    Returns:
        DataFrame aligned to ``entry_df.index`` with ``htf_``-prefixed columns
        plus ``htf_open_time`` (open time of the bias bar being referenced).

    Raises:
        ValueError: if a requested bias column does not exist.
    """
    bias_feats = build_features(bias_df, config)
    missing = [c for c in bias_columns if c not in bias_feats.columns]
    if missing:
        raise ValueError(f"bias feature columns not found: {missing}")

    ref = align_htf_to_ltf(bias_df, entry_df)  # open time of last CLOSED bias bar
    known = ref.dropna()
    positions = pd.Series(-1, index=entry_df.index, dtype=int)
    if not known.empty:
        positions.loc[known.index] = bias_feats.index.get_indexer(known.to_numpy())
    out = pd.DataFrame(index=entry_df.index)
    for col in bias_columns:
        values = bias_feats[col].to_numpy(dtype=object)
        mapped = pd.Series(
            [values[p] if p >= 0 else None for p in positions.to_numpy()],
            index=entry_df.index,
        )
        if col.startswith("ind_") or col == "st_trend_state":
            out[f"htf_{col}"] = mapped  # numeric/NaN or label object column
        else:  # boolean structure flags -> nullable Boolean dtype
            out[f"htf_{col}"] = mapped.astype("boolean")
    out.insert(0, "htf_open_time", ref)
    FEATURE_GROUPS["htf"] = list(out.columns)
    return out


def group_for_column(column: str) -> str | None:
    """Return the feature-group key owning ``column`` (for the signal scorer).

    Looks up :data:`FEATURE_GROUPS`; returns None for unknown columns.
    """
    for group, cols in FEATURE_GROUPS.items():
        if column in cols:
            return group
    return None


def scorer_groups() -> dict[str, Any]:
    """Snapshot of current group -> columns mapping (copy, safe to iterate)."""
    return {g: list(c) for g, c in FEATURE_GROUPS.items()}
