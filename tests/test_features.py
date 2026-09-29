"""Tests for the fxsignals feature layer.

Covers indicator correctness and warm-up, swing confirmation timing,
structure (HH/HL -> UP, BOS on break bar), session tagging, candle patterns,
the no-look-ahead invariant for every pipeline column, and closed-bar-only
HTF alignment in build_mtf_features. Run locally with: pytest tests/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxsignals.data.resample import resample_ohlc
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
    FeatureConfig,
    build_features,
    build_mtf_features,
    group_for_column,
)
from fxsignals.features.sessions import SessionConfig, session_features
from fxsignals.features.structure import market_structure
from fxsignals.features.swings import detect_swings, last_confirmed_swings
from fxsignals.models import Timeframe


def frame_from_rows(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Build a contract frame from (open, high, low, close) tuples at H1 spacing."""
    idx = pd.date_range("2024-06-03", periods=len(rows), freq="1h", tz="UTC")
    arr = np.asarray(rows, dtype=float)
    return pd.DataFrame(
        {
            "open": arr[:, 0],
            "high": arr[:, 1],
            "low": arr[:, 2],
            "close": arr[:, 3],
            "volume": np.full(len(rows), 100.0),
        },
        index=pd.DatetimeIndex(idx, name="time"),
    )


def synthetic_frame(n: int = 240, tf: Timeframe = Timeframe.H1, seed: int = 11) -> pd.DataFrame:
    """Deterministic random-walk frame starting Monday 00:00 UTC (no weekends)."""
    idx = pd.date_range("2024-06-03", periods=n, freq=tf.pandas_alias, tz="UTC")
    rng = np.random.default_rng(seed)
    close = 1.10 + np.cumsum(rng.normal(0.0, 0.0015, n))
    open_ = np.concatenate([[1.10], close[:-1]])
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.0006, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.0006, n))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": np.full(n, 500.0)},
        index=pd.DatetimeIndex(idx, name="time"),
    )


# ---------------------------------------------------------------- indicators

def test_sma_hand_calculated() -> None:
    df = frame_from_rows([(1.0, 1.2, 0.8, 1.1)] * 5)
    out = sma(df, period=3)
    assert out.iloc[:2].isna().all()          # warm-up NaN, never back-filled
    assert out.iloc[2:].notna().all()
    assert out.iloc[2] == pytest.approx(1.1)  # mean of three closes


def test_ema_hand_calculated_and_warmup() -> None:
    closes = [1.0, 2.0, 3.0, 4.0, 5.0]
    df = frame_from_rows([(c, c + 0.1, c - 0.1, c) for c in closes])
    out = ema(df, period=3)
    assert out.iloc[:2].isna().all()
    assert out.iloc[2] == pytest.approx(2.0)  # SMA seed over first 3 closes
    alpha = 2.0 / (3.0 + 1.0)
    assert out.iloc[3] == pytest.approx(out.iloc[2] + alpha * (4.0 - out.iloc[2]))
    assert out.iloc[4] > out.iloc[3]          # rising input lifts the EMA


def test_rsi_bounds_and_direction() -> None:
    up = frame_from_rows([(1.0 + 0.01 * i, 1.01 + 0.01 * i, 0.99 + 0.01 * i,
                           1.0 + 0.01 * i) for i in range(30)])
    down = frame_from_rows([(5.0 - 0.01 * i, 5.01 - 0.01 * i, 4.99 - 0.01 * i,
                             5.0 - 0.01 * i) for i in range(30)])
    r_up, r_down = rsi(up), rsi(down)
    assert r_up.iloc[:14].isna().all()
    vals = pd.concat([r_up, r_down]).dropna()
    assert ((vals >= 0.0) & (vals <= 100.0)).all()
    assert r_up.dropna().iloc[-1] > 70.0
    assert r_down.dropna().iloc[-1] < 30.0
    flat = rsi(frame_from_rows([(2.0, 2.0, 2.0, 2.0)] * 20))
    assert flat.dropna().eq(50.0).all()


def test_atr_positive_and_warmup() -> None:
    df = synthetic_frame(60)
    out = atr(df, period=14)
    assert out.iloc[:14].isna().all()
    tail = out.dropna()
    assert (tail > 0).all()


def test_macd_bollinger_stochastic_adx_shapes() -> None:
    df = synthetic_frame(120)
    m = macd(df)
    assert list(m.columns) == ["macd", "macd_signal", "macd_hist"]
    assert m["macd"].iloc[:26].isna().all()
    assert m.loc[df.index].index.equals(df.index)

    bb = bollinger(df, period=20)
    valid = bb.dropna()
    assert (valid["bb_upper"] >= valid["bb_mid"]).all()
    assert (valid["bb_mid"] >= valid["bb_lower"]).all()
    assert (valid["bb_width"] >= 0).all()

    st = stochastic(df)
    k = st["stoch_k"].dropna()
    assert ((k >= 0) & (k <= 100)).all()

    a = adx(df, period=14)
    assert list(a.columns) == ["plus_di", "minus_di", "adx"]
    av = a.dropna()
    assert ((av["adx"] >= 0) & (av["adx"] <= 100)).all()
    assert a["adx"].iloc[:28].isna().all()  # min bars = 2*period


# -------------------------------------------------------------------- swings


def test_swing_high_appears_only_at_confirmation_bar() -> None:
    closes = [1.0] * 5 + [2.0, 2.2, 2.4, 2.2, 2.0] + [1.5] * 5 + [2.5] * 5 + [2.0] * 5
    df = frame_from_rows([(c - 0.02, c + 0.05, c - 0.05, c) for c in closes])
    events = detect_swings(df, left=3, right=3)
    highs = events[events["kind"] == "high"]
    # Sharp peak at index 7 needs 3 lower highs on each side -> confirmed at 10.
    first = highs.iloc[0]
    assert first["pivot_time"] == df.index[7]
    assert first["confirmed_at"] == df.index[10]
    assert first["price"] == pytest.approx(2.45)

    sw = last_confirmed_swings(df, n=2, left=3, right=3)
    col = sw.frame["swing_high"]
    assert col.iloc[:10].isna().all()   # invisible before confirmation bar
    assert col.iloc[10:].notna().all()  # visible from confirmation onward
    assert col.iloc[10] == pytest.approx(2.45)


def test_swing_events_never_reference_future_data() -> None:
    df = synthetic_frame(200)
    sw = last_confirmed_swings(df, n=2, left=3, right=3)
    if not sw.events.empty:
        assert (sw.events["confirmed_pos"] < len(df)).all()
        assert (sw.events["pivot_time"] <= sw.events["confirmed_at"]).all()


# ----------------------------------------------------------------- structure

def test_uptrend_hh_hl_and_bos_on_break_bar() -> None:
    closes = [1.0] * 5 + [2.0] * 5 + [1.5] * 5 + [2.5] * 5 + [2.0] * 5 + [3.0] * 5
    lows = [c - 0.1 for c in closes]
    highs = [c + 0.1 for c in closes]
    opens = [c - 0.02 for c in closes]
    df = frame_from_rows(list(zip(opens, highs, lows, closes)))
    res = market_structure(df, left=3, right=3)
    f = res.frame
    # Swings: high@9 (2.1) conf@12, low@14 (1.4) conf@17, high@19 (2.6) conf@22,
    # low@24 (1.9) conf@27 -> HH+HL => UP from bar 27 onward.
    assert (f["trend_state"].iloc[27:32] == "UP").all()
    assert f["bos_up"].iloc[:30].sum() == 1
    assert f["bos_up"].iloc[30]           # close 3.0 breaks swing high 2.6
    assert not f["choch"].any()
    assert f["last_swing_high"].iloc[30] == pytest.approx(2.6)
    assert f["last_swing_low"].iloc[30] == pytest.approx(1.9)


def test_downtrend_and_choch_against_structure() -> None:
    closes = [3.0] * 5 + [2.0] * 5 + [2.5] * 5 + [1.5] * 5 + [2.0] * 5 + [1.0] * 5
    df = frame_from_rows([(c - 0.02, c + 0.1, c - 0.1, c) for c in closes])
    res = market_structure(df, left=3, right=3)
    f = res.frame
    assert (f["trend_state"].iloc[27:32] == "DOWN").all()
    assert f["bos_down"].iloc[30]         # close 1.0 breaks swing low 1.4
    # Later rally to 3.0 closes above the last swing high -> CHoCH vs DOWN.
    rallies = closes + [2.2] * 5 + [2.8] * 5 + [3.2] * 5
    df2 = frame_from_rows([(c - 0.02, c + 0.1, c - 0.1, c) for c in rallies])
    f2 = market_structure(df2, left=3, right=3).frame
    assert f2["choch"].any()


# -------------------------------------------------------------------- levels

def test_zones_require_two_touches_and_are_causal() -> None:
    closes = [1.0] * 5 + [2.0] * 5 + [1.5] * 5 + [2.5] * 5 + [2.0] * 5 + [3.0] * 5
    df = frame_from_rows([(c - 0.02, c + 0.1, c - 0.1, c) for c in closes])
    zones, per_bar = support_resistance_zones(
        df, atr_mult=0.5, min_touches=2, atr_period=5
    )
    # Second swing confirms at bar 17; before that no zone can be known.
    assert (per_bar["zone_count"].iloc[:17] == 0).all()
    assert per_bar["zone_count"].iloc[-1] >= 1
    assert (zones["touches"] >= 2).all()
    dists = per_bar["nearest_zone_dist_atr"].dropna()
    assert len(dists) > 0 and (dists.abs() < 50).all()


def test_zone_helpers() -> None:
    z = SupportResistanceZone(
        low=1.0, high=1.1, center=1.05, touches=3,
        formed_at=pd.Timestamp("2024-06-03", tz="UTC"), kinds=frozenset({"high"}),
    )
    assert z.contains(1.05)
    assert z.distance_atr(0.9, 0.1) == pytest.approx(1.0)    # below, positive
    assert z.distance_atr(1.3, 0.1) == pytest.approx(-2.0)   # above, negative
    assert z.distance_atr(1.05, 0.1) == 0.0
    assert np.isnan(z.distance_atr(1.05, 0.0))
    assert nearest_zone(0.5, []) is None
    assert nearest_zone(0.5, [z]) is z
    assert zone_distance_atr(0.5, [z], 0.5) == pytest.approx(1.0)
    assert np.isnan(zone_distance_atr(0.5, [], 0.5))


# ------------------------------------------------------------------ sessions

def test_session_tags_for_known_utc_hours() -> None:
    hours = list(range(24))
    idx = pd.DatetimeIndex(
        [pd.Timestamp(f"2024-06-03T{h:02d}:00", tz="UTC") for h in hours]
    )
    df = pd.DataFrame(
        {c: 1.0 for c in ("open", "high", "low", "close", "volume")}, index=idx
    )
    s = session_features(df)
    assert not s["asia"].iloc[0]                     # window is half-open
    assert s["asia"].iloc[5]                         # 05:00 in asia (1..9)
    assert s["london"].iloc[8] and not s["london"].iloc[17]
    assert s["newyork"].iloc[13] and not s["newyork"].iloc[22]
    assert s["overlap"].iloc[13] and s["overlap"].iloc[15]
    assert not s["overlap"].iloc[5]
    assert s["is_london_open_hour"].iloc[9]
    assert not s["is_london_open_hour"].iloc[17]
    assert s["is_london_open_hour"].iloc[16]
    custom = session_features(df, SessionConfig(london=(0, 2)))
    assert custom["london"].iloc[1] and not custom["london"].iloc[3]
    wrapping = session_features(df, SessionConfig(asia=(22, 3)))
    assert wrapping["asia"].iloc[23] and wrapping["asia"].iloc[1]
    assert not wrapping["asia"].iloc[10]
    with pytest.raises(ValueError):
        SessionConfig(london=(0, 30)).validated()


# ------------------------------------------------------------------ patterns

ATR_WARM = [(1.0, 1.1, 0.9, 1.0)] * 15


def test_bullish_engulfing_detected() -> None:
    rows = ATR_WARM + [(1.05, 1.06, 0.99, 1.00), (0.995, 1.11, 0.99, 1.10)]
    df = frame_from_rows(rows)
    sig = bullish_engulfing(df)
    assert bool(sig.iloc[-1])
    assert not sig.iloc[:-1].any()
    assert not bearish_engulfing(df).any()


def test_bearish_engulfing_detected() -> None:
    rows = ATR_WARM + [(1.00, 1.06, 0.99, 1.05), (1.055, 1.06, 0.985, 0.99)]
    df = frame_from_rows(rows)
    assert bool(bearish_engulfing(df).iloc[-1])
    assert not bullish_engulfing(df).any()


def test_pin_bars_detected() -> None:
    bull = frame_from_rows(ATR_WARM + [(1.00, 1.01, 0.90, 1.005)])
    assert bool(pin_bar_bull(bull).iloc[-1])
    assert not pin_bar_bear(bull).any()
    bear = frame_from_rows(ATR_WARM + [(1.00, 1.10, 0.99, 0.995)])
    assert bool(pin_bar_bear(bear).iloc[-1])
    assert not pin_bar_bull(bear).any()


def test_inside_bar() -> None:
    rows = [(1.0, 1.2, 0.8, 1.1), (1.05, 1.1, 0.9, 1.08), (1.3, 1.5, 1.1, 1.4)]
    sig = inside_bar(frame_from_rows(rows))
    assert [bool(x) for x in sig] == [False, True, False]


# ------------------------------------------------------------------ pipeline

def test_build_features_prefixes_and_groups() -> None:
    df = synthetic_frame(200)
    feats = build_features(df)
    assert feats.index.equals(df.index)
    prefixes = ("ind_", "sw_", "st_", "lv_", "ses_", "pat_")
    assert all(any(c.startswith(p) for c in feats.columns) for p in prefixes)
    # Every column belongs to exactly one tracked feature group.
    assigned = [c for g, cols in FEATURE_GROUPS.items() if g != "htf" for c in cols]
    assert sorted(feats.columns) == sorted(assigned)
    assert len(assigned) == len(set(assigned))
    assert group_for_column("ind_rsi") == "indicators"
    assert group_for_column("pat_inside_bar") == "patterns"
    # Correlated indicators share a group so the scorer counts each group once.
    assert INDICATOR_GROUPS["momentum"] == ("rsi", "stochastic", "bollinger_width")
    assert "macd" in INDICATOR_GROUPS["trend"] and "ema" in INDICATOR_GROUPS["trend"]


def test_no_lookahead_every_feature_column() -> None:
    df = synthetic_frame(200)
    full = build_features(df)
    cutoff = 150
    partial = build_features(df.iloc[:cutoff])
    for col in full.columns:
        a, b = full[col].iloc[:cutoff], partial[col]
        if a.dtype == object or str(a.dtype).startswith("bool"):
            assert (a.fillna("x") == b.fillna("x")).all(), col
        else:
            pd.testing.assert_series_equal(a, b, check_names=False, obj=str(col))


def test_build_mtf_features_closed_bars_only() -> None:
    h1 = synthetic_frame(240)                      # entry TF, Mon 00:00 start
    h4 = resample_ohlc(h1, Timeframe.H4)           # bias TF, closed bars only
    cfg = FeatureConfig(atr_period=5, ema_periods=(10,), sma_period=10,
                        bb_period=10, macd_fast=5, macd_slow=10, macd_signal=4,
                        adx_period=5, stoch_k=7)
    mtf = build_mtf_features(h4, h1, cfg)
    assert all(c.startswith("htf_") for c in mtf.columns)
    ref = mtf["htf_open_time"]
    assert ref.isna().iloc[:3].all()               # first H4 available at bar 3
    assert ref.iloc[3] == h4.index[0]              # once it has CLOSED
    step = pd.Timedelta(hours=4)
    ltf = pd.Series(ref.index, index=ref.index)
    assert ((ref.dropna() + step) <= ltf[ref.notna()]).all()  # never the forming bar

    # Spot-check: value equals the referenced CLOSED H4 feature, and a new H4
    # value only appears from its close time onward.
    bias = build_features(h4, cfg)
    i = ref.last_valid_index()
    assert mtf.loc[i, "htf_ind_ema10"] == bias.loc[ref.loc[i], "ind_ema10"]
    nxt = h4.index[h4.index > ref.loc[i]][0]
    eligible = h1.index[h1.index >= nxt + step]
    if len(eligible):
        assert mtf.loc[eligible[0], "htf_ind_ema10"] == bias.loc[nxt, "ind_ema10"]
        before = mtf["htf_ind_ema10"].loc[mtf.index < nxt + step]
        assert (before != bias.loc[nxt, "ind_ema10"]).all()  # no early leak


def test_mtf_rejects_unknown_bias_columns() -> None:
    h1 = synthetic_frame(60)
    h4 = resample_ohlc(h1, Timeframe.H4)
    with pytest.raises(ValueError):
        build_mtf_features(h4, h1, bias_columns=("does_not_exist",))
