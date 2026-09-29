"""Tests for the fxsignals data layer: validation, resampling, alignment,
synthetic determinism and the CSV provider round-trip."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from fxsignals.config import ConfigError, PathsConfig, default_settings, load_settings
from fxsignals.data import base as base_module
from fxsignals.data.base import (
    DataProvider,
    ProviderError,
    get_provider,
    register_provider,
)
from fxsignals.data.csv_provider import CsvProvider
from fxsignals.data.normalize import normalize_candles
from fxsignals.data.resample import align_htf_to_ltf, resample_ohlc
from fxsignals.data.store import CandleStore
from fxsignals.data.synthetic import SyntheticParams, generate_candles
from fxsignals.data.validate import clean_candles, validate_candles
from fxsignals.models import Direction, Signal, Timeframe


def make_frame(n: int = 48, tf: Timeframe = Timeframe.H1) -> pd.DataFrame:
    """Small deterministic contract-compliant frame for unit tests."""
    idx = pd.date_range("2024-01-01", periods=n, freq=tf.pandas_alias, tz="UTC")
    rng = np.random.default_rng(7)
    close = 1.1 + np.cumsum(rng.normal(0, 0.001, n))
    open_ = np.concatenate([[1.1], close[:-1]])
    high = np.maximum(open_, close) + 0.0005
    low = np.minimum(open_, close) - 0.0005
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": np.full(n, 100.0)},
        index=pd.DatetimeIndex(idx, name="time"),
    )


# ---------------------------------------------------------------- models ---

def test_timeframe_minutes_and_parsing() -> None:
    assert Timeframe.H1.minutes == 60
    assert Timeframe.H4.minutes == 240
    assert Timeframe.from_str("h1") is Timeframe.H1
    with pytest.raises(ValueError):
        Timeframe.from_str("M5")


def test_signal_validation_and_dict() -> None:
    sig = Signal(
        pair="EURUSD",
        direction=Direction.LONG,
        entry=1.10,
        stop_loss=1.09,
        take_profit=1.12,
        rr=2.0,
        confidence=75.0,
        timeframe=Timeframe.H1,
        reasons=["bias up"],
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    d = sig.to_dict()
    assert d["direction"] == "LONG"
    assert d["timeframe"] == "H1"
    assert d["reasons"] == ["bias up"]
    with pytest.raises(ValueError):
        Signal(**{**sig.__dict__, "confidence": 150})
    with pytest.raises(ValueError):
        Signal(**{**sig.__dict__, "timestamp": datetime(2024, 1, 1)})


# ------------------------------------------------------------- validate ----

def test_validate_accepts_good_frame() -> None:
    rep = validate_candles(make_frame(), timeframe=Timeframe.H1)
    assert rep.ok, rep.summary()


def test_validate_detects_problems() -> None:
    df = make_frame(10)
    bad = df.copy()
    # duplicate timestamp
    bad = pd.concat([bad, bad.iloc[[3]]])
    # NaN row
    bad.loc[bad.index[0], "close"] = np.nan
    # OHLC violation on a clean copy
    ohlc_bad = df.copy()
    ohlc_bad.loc[ohlc_bad.index[2], "high"] = 0.5
    # non-positive price
    neg = df.copy()
    neg.loc[neg.index[1], "low"] = -1.0

    assert validate_candles(bad).duplicate_timestamps == 1
    assert len(validate_candles(bad).nan_rows) == 1
    assert len(validate_candles(ohlc_bad).ohlc_violations) == 1
    assert len(validate_candles(neg).nonpositive_prices) == 1


def test_validate_gaps_ignore_weekends() -> None:
    # H1 bars Fri 20:00 -> Mon 20:00: pure weekend closure, not a gap.
    idx = pd.DatetimeIndex(
        ["2024-01-04T20:00Z", "2024-01-07T20:00Z"], tz="UTC", name="time"
    )
    df = pd.DataFrame(1.0, index=idx, columns=["open", "high", "low", "close", "volume"])
    rep = validate_candles(df, timeframe=Timeframe.H1)
    assert rep.gap_count == 0

    # Missing a weekday bar => gap flagged.
    idx2 = pd.DatetimeIndex(
        ["2024-01-08T00:00Z", "2024-01-08T02:00Z"], tz="UTC", name="time"
    )
    df2 = pd.DataFrame(1.0, index=idx2, columns=["open", "high", "low", "close", "volume"])
    assert validate_candles(df2, timeframe=Timeframe.H1).gap_count == 1


def test_clean_candles_drops_bad_rows_only() -> None:
    df = make_frame(10)
    dirty = pd.concat([df, df.iloc[[5]]])  # duplicate
    # NaN row at the last timestamp so dropping it leaves a contiguous frame.
    dirty.loc[dirty.index[-1], "close"] = np.nan
    cleaned = clean_candles(dirty)
    assert cleaned.index.is_monotonic_increasing
    assert not cleaned.index.duplicated().any()
    assert cleaned.index.max() == df.index[-2]  # bad row dropped, nothing invented
    assert len(cleaned) == len(df) - 1
    assert validate_candles(cleaned, timeframe=Timeframe.H1).ok


# ------------------------------------------------------------- resample ----

def test_resample_ohlc_aggregation_correctness() -> None:
    h1 = make_frame(24)  # day 1 + day 2 of hourly bars
    h4 = resample_ohlc(h1, Timeframe.H4, keep_incomplete_last=True)
    first = h4.iloc[0]
    block = h1.iloc[0:4]
    assert first["open"] == block["open"].iloc[0]
    assert first["high"] == block["high"].max()
    assert first["low"] == block["low"].min()
    assert first["close"] == block["close"].iloc[-1]
    assert first["volume"] == block["volume"].sum()
    assert len(h4) == 6


def test_resample_drops_incomplete_last_bar_by_default() -> None:
    h1 = make_frame(10)  # last H4 bucket (08:00-12:00) only has bars to 09:00
    closed = resample_ohlc(h1, Timeframe.H4)
    kept = resample_ohlc(h1, Timeframe.H4, keep_incomplete_last=True)
    assert len(kept) == len(closed) + 1
    assert kept.index[-1] == pd.Timestamp("2024-01-01T08:00", tz="UTC")
    # A higher-TF bar is only available after it closes: the largest bucket
    # kept in `closed` ends at/just past the last source bar.
    assert closed.index.max() + pd.Timedelta(hours=4) <= h1.index.max() + pd.Timedelta(hours=1)


def test_resample_keeps_full_final_bucket() -> None:
    h1 = make_frame(8)  # 00:00..07:00 => buckets 00:00 and 04:00 both complete
    closed = resample_ohlc(h1, Timeframe.H4)
    assert len(closed) == 2


# ----------------------------------------------------- no-look-ahead -------

def test_align_htf_to_ltf_no_future_leakage() -> None:
    ltf = make_frame(24, Timeframe.H1)          # 00:00..23:00
    htf = resample_ohlc(ltf, Timeframe.H4, keep_incomplete_last=True)
    aligned = align_htf_to_ltf(htf, ltf)

    # First three H1 bars cannot see any closed H4 bar yet.
    assert aligned.iloc[0] is pd.NaT or pd.isna(aligned.iloc[0])
    assert pd.isna(aligned.iloc[1]) and pd.isna(aligned.iloc[2])
    # The 04:00 H1 bar sees exactly the 00:00 H4 bar (now closed).
    assert aligned.iloc[4] == htf.index[0]
    # Within the forming 04:00-08:00 H4 bar, alignment stays on the previous one.
    assert all(aligned.iloc[i] == htf.index[0] for i in range(5, 8))
    # Never map an LTF bar to an HTF bar that opens at/after the LTF bar itself.
    for ltf_ts, htf_ts in aligned.dropna().items():
        assert htf_ts + pd.Timedelta(hours=4) <= ltf_ts


# ------------------------------------------------------------ synthetic ----

def test_synthetic_is_deterministic_and_valid() -> None:
    p = SyntheticParams(seed=123, bars=200)
    a = generate_candles("EURUSD", Timeframe.H1, p)
    b = generate_candles("EURUSD", Timeframe.H1, p)
    pd.testing.assert_frame_equal(a, b)

    rep = validate_candles(a, timeframe=Timeframe.H1)
    assert rep.ok, rep.summary()
    # Weekend slots are skipped entirely.
    assert all(ts.weekday() < 5 for ts in a.index)
    # OHLC wrap holds by construction.
    assert (a["high"] >= a[["open", "close"]].max(axis=1)).all()
    assert (a["low"] <= a[["open", "close"]].min(axis=1)).all()
    # Different seed => different path.
    c = generate_candles("EURUSD", Timeframe.H1, SyntheticParams(seed=999, bars=200))
    assert not np.allclose(a["close"].values, c["close"].values)


def test_provider_registry_roundtrip() -> None:
    prov = get_provider("synthetic", default_settings())
    df = prov.get_candles("EURUSD", Timeframe.H1, limit=50)
    assert len(df) == 50
    assert validate_candles(df, timeframe=Timeframe.H1).ok
    with pytest.raises(ProviderError):
        get_provider("does-not-exist")


def test_new_provider_plugs_in_by_subclassing_only() -> None:
    """A future API provider works after subclass + register, no other wiring."""

    class DummyProvider(DataProvider):
        def get_candles(self, pair, timeframe, start=None, end=None, limit=None):
            return make_frame(4, timeframe)

    register_provider("dummy")(DummyProvider)
    try:
        prov = get_provider("dummy", default_settings())
        out = prov.get_candles("EURUSD", Timeframe.M30)
        assert list(out.columns) == ["open", "high", "low", "close", "volume"]
        assert str(out.index.tz) == "UTC"
    finally:
        base_module._REGISTRY.pop("dummy", None)


# ----------------------------------------------------------------- csv -----

def test_csv_provider_roundtrip(tmp_path) -> None:
    df = make_frame(12)
    raw = df.reset_index()
    raw["time"] = raw["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out_dir = tmp_path / "data"
    out_dir.mkdir()
    # Write without volume column to check tolerance (filled with 0).
    raw.drop(columns=["volume"]).to_csv(out_dir / "EURUSD_H1.csv", index=False)

    settings = default_settings()

    # Settings now has defaults for every section, so a minimal override works.
    from dataclasses import replace

    settings = replace(settings, provider="csv", paths=PathsConfig(
        data_dir=out_dir, cache_dir=tmp_path / "cache"
    ))
    prov = CsvProvider(settings)
    loaded = prov.get_candles("EURUSD", Timeframe.H1)
    assert len(loaded) == 12
    assert (loaded["volume"] == 0).all()
    np.testing.assert_allclose(loaded["close"].values, df["close"].values)
    # Windowing works.
    tail = prov.get_candles("EURUSD", Timeframe.H1, limit=3)
    assert len(tail) == 3
    assert tail.index[0] == df.index[-3]
    with pytest.raises(ProviderError):
        prov.get_candles("GBPUSD", Timeframe.H1)


def test_store_get_or_fetch(tmp_path) -> None:
    from fxsignals.data.synthetic import SyntheticProvider

    store = CandleStore(tmp_path / "cache")
    calls = []

    class _CountingProvider(SyntheticProvider):
        """Counts fetches to prove the second call is served from cache."""

        def get_candles(self, pair, timeframe, start=None, end=None, limit=None):
            calls.append(pair)
            return super().get_candles(pair, timeframe, start, end, limit)

    prov = _CountingProvider(default_settings())
    first = store.get_or_fetch(prov, "EURUSD", Timeframe.H1)
    second = store.get_or_fetch(prov, "EURUSD", Timeframe.H1)
    assert len(calls) == 1  # second call served from cache
    pd.testing.assert_frame_equal(first, second)
    assert store.path_for("EURUSD", Timeframe.H1).is_file()


# --------------------------------------------------------------- config ----

def test_load_settings_ok_and_errors(tmp_path) -> None:
    good = tmp_path / "s.yaml"
    good.write_text(
        """
pairs: [EURUSD, GBPUSD]
timeframes: {bias_tf: H4, entry_tf: H1}
provider: synthetic
paths: {data_dir: ./data, cache_dir: ./.cache}
log_level: INFO
""",
        encoding="utf-8",
    )
    s = load_settings(good)
    assert s.provider == "synthetic"
    assert s.timeframes.bias_tf is Timeframe.H4

    bad = tmp_path / "bad.yaml"
    bad.write_text("pairs: [EURUSD]\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(bad)

    worse = tmp_path / "worse.yaml"
    worse.write_text(
        """
pairs: [EURUSD]
timeframes: {bias_tf: H1, entry_tf: H1}
provider: synthetic
paths: {data_dir: ./data, cache_dir: ./.cache}
log_level: INFO
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_settings(worse)


def test_normalize_candles_handles_missing_volume() -> None:
    raw = pd.DataFrame(
        {
            "time": ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00+00:00"],
            "open": [1.0, 1.1], "high": [1.2, 1.3], "low": [0.9, 1.0], "close": [1.1, 1.2],
        }
    )
    df = normalize_candles(raw)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert str(df.index.tz) == "UTC"
    assert (df["volume"] == 0).all()
