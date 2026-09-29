"""Tests for the signal engine: scoring, risk levels, filters, emitter.

Run locally with::

    pytest tests/test_signals.py -q
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from fxsignals.config import ConfigError, default_settings, load_settings
from fxsignals.models import Direction, Signal, Timeframe
from fxsignals.output.emitter import JsonlSink, SignalEmitter
from fxsignals.signals import (
    CooldownFilter,
    DuplicateFilter,
    EngineConfig,
    FilterConfig,
    RiskConfig,
    RiskError,
    ScoreWeights,
    SignalEngine,
    build_trade_levels,
    pip_size,
    score_both,
    score_direction,
    set_pips,
)
from fxsignals.signals.factory import build_engine_config


# --------------------------------------------------------------- helpers ----
def make_row(**overrides) -> pd.Series:
    """A feature row (as produced by build_features) with sane defaults."""
    base = {
        "close": 1.1000,
        "ind_atr": 0.0010,
        "ind_rsi": 30.0,               # oversold -> favors LONG
        "ind_stoch_k": 20.0,           # oversold -> favors LONG
        "ind_ema20": 1.1010,
        "ind_ema50": 1.0990,
        "ind_macd_macd": 0.0005,
        "ind_adx_adx": 30.0,
        "st_trend_state": "UP",
        "st_bos_up": True,
        "st_bos_down": False,
        "st_choch": False,
        "st_last_swing_low": 1.0970,
        "st_last_swing_high": 1.1040,
        "lv_nearest_zone_center": 1.0980,   # support BELOW close (dist < 0)
        "lv_nearest_zone_dist_atr": -0.5,
        "lv_zone_count": 3.0,
        "pat_bullish_engulfing": True,
        "pat_pin_bar_bull": False,
        "pat_bearish_engulfing": False,
        "pat_pin_bar_bear": False,
    }
    base.update(overrides)
    return pd.Series(base)


def make_htf_row(state: str = "UP", adx: float = 30.0) -> pd.Series:
    """An HTF-aligned row as produced by build_mtf_features."""
    return pd.Series({"htf_st_trend_state": state, "htf_ind_adx_adx": adx})


def utc_ts(hour: int = 10, day: int = 6) -> pd.Timestamp:
    """A weekday UTC timestamp in the London session."""
    return pd.Timestamp(datetime(2026, 3, day, hour, 0, tzinfo=timezone.utc))


# ------------------------------------------------------------- scoring ------
class TestScoring:
    def test_weights_default_sum_to_100(self):
        w = ScoreWeights()
        assert w.validated() is w
        assert abs(w.htf_bias + w.structure + w.location
                   + w.momentum + w.trend_strength + w.trigger - 100.0) < 1e-9

    def test_weights_bad_sum_rejected(self):
        with pytest.raises(ValueError):
            ScoreWeights(htf_bias=50.0).validated()

    def test_score_ranges_0_100_both_directions(self):
        long_s, short_s = score_both(make_row(), make_htf_row())
        for s in (long_s, short_s):
            assert 0.0 <= s.score <= 100.0
        assert long_s.score > short_s.score  # row is built long-favorable

    def test_perfect_long_row_near_max(self):
        s = score_direction(Direction.LONG, make_row(), make_htf_row("UP", 35))
        assert s.score >= 90.0
        names = [g.name for g in s.groups]
        assert "htf_bias" in names and "trigger" in names

    def test_correlated_groups_counted_once(self):
        """RSI+stoch share ONE momentum vote; EMA+MACD+ADX share ONE trend vote."""
        only_rsi = make_row(ind_rsi=10.0, ind_stoch_k=math.nan)
        both = make_row(ind_rsi=10.0, ind_stoch_k=5.0)
        m_rsi = next(g for g in score_direction(Direction.LONG, only_rsi, None).groups
                     if g.name == "momentum")
        m_both = next(g for g in score_direction(Direction.LONG, both, None).groups
                      if g.name == "momentum")
        assert m_rsi.points == m_both.points          # max vote, not additive

        adx_only = make_row(ind_ema20=math.nan, ind_ema50=math.nan,
                            ind_macd_macd=math.nan, ind_adx_adx=40.0)
        full_trend = make_row(ind_ema20=1.1010, ind_ema50=1.0990,
                              ind_macd_macd=0.0005, ind_adx_adx=40.0)
        t_adx = next(g for g in score_direction(Direction.LONG, adx_only, None).groups
                     if g.name == "trend_strength")
        t_full = next(g for g in score_direction(Direction.LONG, full_trend, None).groups
                      if g.name == "trend_strength")
        # ADX alone scores the SAME whether or not EMA/MACD also agree: it is
        # averaged inside one trend-strength vote, never added as extra groups.
        assert t_adx.points == pytest.approx(t_full.points)
        assert t_full.points <= t_full.weight          # capped at the group weight
        # one group entry per direction, ever:
        names = [g.name for g in score_direction(Direction.LONG, both, make_htf_row()).groups]
        assert names.count("momentum") == 1 and names.count("trend_strength") == 1

    def test_reasons_are_human_readable(self):
        s = score_direction(Direction.LONG, make_row(), make_htf_row())
        assert s.reasons and all(isinstance(r, str) and r for r in s.reasons)

    def test_missing_htf_row_zeroes_bias_group(self):
        with_htf = score_direction(Direction.LONG, make_row(), make_htf_row())
        without = score_direction(Direction.LONG, make_row(), None)
        assert without.score < with_htf.score
        assert "htf_bias" not in [g.name for g in without.groups]


# ---------------------------------------------------------------- risk ------
class TestRiskLevels:
    def test_long_sl_below_entry_short_above(self):
        longs = build_trade_levels(Direction.LONG, entry=1.1000, atr=0.0010, zones=[],
                                   last_swing_low=1.0970)
        shorts = build_trade_levels(Direction.SHORT, entry=1.1000, atr=0.0010, zones=[],
                                    last_swing_high=1.1030)
        assert longs.stop_loss < longs.entry < longs.take_profit
        assert shorts.stop_loss > shorts.entry > shorts.take_profit
        assert longs.rr == pytest.approx(2.0)

    def test_rr_computation(self):
        lv = build_trade_levels(Direction.LONG, entry=100.0, atr=1.0, zones=[],
                                last_swing_low=98.0)  # dist 2.25 ATR
        assert lv.rr == pytest.approx((lv.take_profit - lv.entry)
                                      / (lv.entry - lv.stop_loss))

    def test_rejection_when_rr_below_min(self):
        from fxsignals.features.levels import SupportResistanceZone

        wall = SupportResistanceZone(
            low=100.6, high=100.8, center=100.7, touches=3,
            formed_at=utc_ts(), kinds=frozenset({"resistance"}),
        )
        with pytest.raises(RiskError):
            build_trade_levels(Direction.LONG, entry=100.0, atr=1.0, zones=[wall],
                               last_swing_low=98.0, cfg=RiskConfig(min_rr=1.5))

    def test_sl_distance_bounds_capped(self):
        tight = build_trade_levels(Direction.LONG, entry=100.0, atr=1.0, zones=[],
                                   last_swing_low=99.95)  # raw 0.30 ATR
        wide = build_trade_levels(Direction.LONG, entry=100.0, atr=1.0, zones=[],
                                  last_swing_low=80.0)    # raw >> 3 ATR
        assert 100.0 - tight.stop_loss == pytest.approx(0.5)   # min_sl_atr cap
        assert 100.0 - wide.stop_loss == pytest.approx(3.0)    # max_sl_atr cap

    def test_pip_sizes_and_conversion(self):
        assert pip_size("EURUSD") == 0.0001
        assert pip_size("USDJPY") == 0.01
        assert pip_size("XAUUSD") == 0.1
        lv = build_trade_levels(Direction.LONG, entry=1.5000, atr=0.0100, zones=[],
                                last_swing_low=1.4850)
        jpy = set_pips(lv, "USDJPY")
        assert jpy.sl_pips == pytest.approx((lv.entry - lv.stop_loss) / 0.01)
        assert jpy.tp_pips == pytest.approx((lv.take_profit - lv.entry) / 0.01)

    def test_no_reference_rejected(self):
        with pytest.raises(RiskError):
            build_trade_levels(Direction.LONG, entry=1.1, atr=0.001, zones=[])


# ------------------------------------------------------------ filters -------
class TestFilters:
    def test_cooldown_blocks_then_expires(self):
        cfg = FilterConfig(cooldown_bars=3)
        idx = pd.date_range("2026-03-02", periods=10, freq=Timeframe.H1.rule, tz="UTC")
        f = CooldownFilter(cfg)
        ok, _ = f.check("EURUSD", Direction.LONG, idx[0], idx)
        assert ok
        ok, reason = f.check("EURUSD", Direction.LONG, idx[2], idx)
        assert not ok and "cooldown" in reason
        assert f.check("EURUSD", Direction.LONG, idx[3], idx)[0]   # 3 bars later

    def test_duplicate_suppression_same_zone(self):
        f = DuplicateFilter(FilterConfig(dup_entry_atr=0.75))
        assert f.check("EURUSD", Direction.LONG, 1.1000, 0.0010)[0]
        ok, reason = f.check("EURUSD", Direction.LONG, 1.1005, 0.0010)
        assert not ok and "duplicate" in reason
        assert f.check("EURUSD", Direction.LONG, 1.1200, 0.0010)[0]  # far zone
        assert f.check("GBPUSD", Direction.LONG, 1.1001, 0.0010)[0]  # other pair

    def test_session_filter_respects_allowed_sessions(self):
        from fxsignals.features.sessions import DEFAULT_SESSIONS, _in_window
        from fxsignals.signals.filters import session_filter

        cfg = FilterConfig(allowed_sessions=("london",))
        sc = DEFAULT_SESSIONS.validated()
        hours = pd.Series([h for h in range(24)])
        tags = {t: set(hours[_in_window(hours, getattr(sc, t))])
                for t in ("asia", "london", "newyork")}
        allowed_hours = sorted(tags["london"] - tags["asia"] - tags["newyork"])
        assert allowed_hours, "expected hours unique to the London session"
        # An hour exclusive to a DISALLOWED session must fail the filter.
        foreign = (tags["asia"] | tags["newyork"]) - tags["london"]
        ts_bad = utc_ts(hour=int(sorted(foreign)[0]))
        passed, reason = session_filter(ts_bad, cfg)
        assert not passed and "not in" in reason


# -------------------------------------------------------------- engine ------
def loose_engine() -> SignalEngine:
    """Engine with gates lowered so hand-built rows can pass end-to-end."""
    cfg = EngineConfig(
        weights=ScoreWeights(),
        risk=RiskConfig(),
        filters=FilterConfig(min_adx=0.0, allowed_sessions=(), cooldown_bars=0),
        min_score=0.0,
        direction_margin=0.0,
        warmup_bars=0,
    )
    return SignalEngine(cfg)


def synthetic_frames(pairs_seed: str = "EURUSD"):
    from fxsignals.data.synthetic import SyntheticProvider

    prov = SyntheticProvider()
    bias = prov.get_candles(pairs_seed, Timeframe.H4, limit=150)
    entry = prov.get_candles(pairs_seed, Timeframe.H1, limit=200)
    return bias, entry


class TestEngine:
    def test_gates_block_weak_signals(self):
        """No signal below min_score or when margin is too small."""
        bias, entry = synthetic_frames()
        strict = SignalEngine(EngineConfig(min_score=100.0, direction_margin=0.0,
                                           filters=FilterConfig(min_adx=0.0,
                                                                allowed_sessions=())))
        assert strict.evaluate("EURUSD", bias, entry) == []
        tie = SignalEngine(EngineConfig(min_score=0.0, direction_margin=100.0,
                                        filters=FilterConfig(min_adx=0.0,
                                                             allowed_sessions=())))
        assert tie.evaluate("EURUSD", bias, entry) == []

    def test_loose_engine_emits_valid_signal(self):
        bias, entry = synthetic_frames()
        sigs = loose_engine().evaluate("EURUSD", bias, entry)
        assert len(sigs) == 1
        s = sigs[0]
        assert isinstance(s, Signal)
        assert s.direction in (Direction.LONG, Direction.SHORT)
        assert 0.0 <= s.confidence <= 100.0
        assert s.rr >= 1.5
        assert s.stop_loss < s.entry < s.take_profit or s.take_profit < s.entry < s.stop_loss
        assert s.timeframe is Timeframe.H1

    def test_evaluate_history_no_look_ahead(self):
        """Signals up to bar n are identical whether computed on df[:n] or full."""
        bias, entry = synthetic_frames()
        cut = 150
        a = loose_engine().evaluate_history("EURUSD", bias, entry.iloc[:cut])
        b = loose_engine().evaluate_history("EURUSD", bias, entry)
        common = b.index[b.index < entry.index[cut - 1]]
        pd.testing.assert_frame_equal(a.loc[a.index.intersection(common)],
                                      b.loc[b.index.intersection(common)])

    def test_loose_history_emits_at_least_one_signal(self):
        """With gates off, the loose engine must fire somewhere in history."""
        bias, entry = synthetic_frames()
        hist = loose_engine().evaluate_history("EURUSD", bias, entry)
        assert not hist.empty

    def test_history_frame_contract(self):
        """evaluate_history returns the documented columns, indexed by UTC time."""
        bias, entry = synthetic_frames()
        hist = loose_engine().evaluate_history("EURUSD", bias, entry)
        cols = ["direction", "score", "entry", "stop_loss", "take_profit", "rr",
                "sl_pips", "tp_pips", "confidence", "reasons"]
        assert list(hist.columns) == cols
        assert hist.index.tz is not None
        if not hist.empty:
            assert hist["direction"].isin(["LONG", "SHORT"]).all()
            longs = hist[hist["direction"] == "LONG"]
            shorts = hist[hist["direction"] == "SHORT"]
            assert (longs["stop_loss"] < longs["entry"]).all()
            assert (shorts["stop_loss"] > shorts["entry"]).all()
            assert (hist["rr"] >= 1.5 - 1e-9).all()

    def test_factory_maps_settings(self):
        settings = default_settings()
        cfg = build_engine_config(settings)
        assert cfg.min_score == settings.signals.min_score == 65.0
        assert cfg.direction_margin == 15.0
        assert cfg.risk.rr_target == 2.0 and cfg.risk.min_rr == 1.5
        assert cfg.filters.allowed_sessions == ("london", "newyork", "overlap")
        assert cfg.filters.cooldown_bars == settings.signals.cooldown_bars == 12
        assert cfg.weights.validated() is cfg.weights


# ------------------------------------------------------------- config -------
BASE_CFG = {
    "pairs": ["EURUSD"],
    "timeframes": {"bias_tf": "H4", "entry_tf": "H1"},
    "provider": "synthetic",
    "paths": {"data_dir": "./d", "cache_dir": "./c"},
    "log_level": "INFO",
}


def write_cfg(tmp_path, name, **sections):
    """Write a minimal valid YAML config plus optional extra sections."""
    import yaml

    data = dict(BASE_CFG)
    data.update(sections)
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


class TestConfigSection:
    def test_defaults_match_settings_yaml_values(self):
        settings = default_settings()
        assert settings.signals.weights["htf_bias"] == 30.0
        assert settings.signals.min_score == 65.0
        assert settings.signals.direction_margin == 15.0
        assert settings.signals.rr_target == 2.0
        assert settings.signals.min_rr == 1.5
        assert settings.signals.min_sl_atr == 0.5
        assert settings.signals.max_sl_atr == 3.0
        assert settings.signals.sl_atr_buffer == 0.25
        assert settings.signals.cooldown_bars == 12
        assert settings.filters.min_adx == 18.0
        assert settings.filters.sessions == ("london", "newyork", "overlap")
        assert settings.filters.atr_percentile_low == 10.0
        assert settings.filters.atr_percentile_high == 95.0
        assert settings.filters.lookback == 180
        assert settings.output.jsonl_path == Path("signals.jsonl")
        assert settings.output.telegram_enabled is False

    def test_project_settings_file_loads(self):
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
        settings = load_settings(path)
        assert settings.signals.cooldown_bars == 12
        assert settings.output.jsonl_path == Path("signals.jsonl")

    def test_missing_sections_fall_back_to_defaults(self, tmp_path):
        settings = load_settings(write_cfg(tmp_path, "bare.yaml"))
        assert settings.signals == default_settings().signals
        assert settings.filters == default_settings().filters
        assert settings.output == default_settings().output

    def test_bad_weights_sum_raises(self, tmp_path):
        p = write_cfg(tmp_path, "bad_w.yaml", signals={
            "weights": {"htf_bias": 50, "structure": 20, "location": 20,
                        "momentum": 10, "trend_strength": 10, "trigger": 5},
            "min_score": 65, "direction_margin": 15, "rr_target": 2.0,
            "min_rr": 1.5, "min_sl_atr": 0.5, "max_sl_atr": 3.0,
            "sl_atr_buffer": 0.25,
        })
        with pytest.raises(ConfigError):
            load_settings(p)

    def test_bad_min_score_and_rr_raise(self, tmp_path):
        common = {"weights": default_settings().signals.weights, "min_score": 65,
                  "direction_margin": 15, "rr_target": 2.0, "min_rr": 1.5,
                  "min_sl_atr": 0.5, "max_sl_atr": 3.0, "sl_atr_buffer": 0.25}
        p1 = write_cfg(tmp_path, "bad_ms.yaml", signals={**common, "min_score": 150})
        p2 = write_cfg(tmp_path, "bad_rr.yaml", signals={**common, "min_rr": 0})
        with pytest.raises(ConfigError):
            load_settings(p1)
        with pytest.raises(ConfigError):
            load_settings(p2)

    def test_bad_session_name_raises(self, tmp_path):
        p = write_cfg(tmp_path, "bad_sess.yaml", filters={
            "min_adx": 18, "sessions": ["tokyo"], "atr_percentile_low": 10,
            "atr_percentile_high": 95, "lookback": 180,
        })
        with pytest.raises(ConfigError):
            load_settings(p)


# ------------------------------------------------------------- emitter ------
class _BrokenSink:
    def send(self, signal: Signal) -> None:
        raise RuntimeError("boom")


def sample_signal() -> Signal:
    return Signal(
        pair="EURUSD", direction=Direction.LONG, entry=1.1000, stop_loss=1.0975,
        take_profit=1.1050, rr=2.0, confidence=78.0, timeframe=Timeframe.H1,
        reasons=["HTF bias UP supports LONG", "Risk: SL 0.75 ATR beyond swing low 1.0970"],
        timestamp=datetime(2026, 3, 6, 10, 0, tzinfo=timezone.utc),
    )


class TestEmitter:
    def test_jsonl_written_and_valid(self, tmp_path):
        path = tmp_path / "out" / "signals.jsonl"
        emitter = SignalEmitter([JsonlSink(path)])
        results = emitter.emit(sample_signal())
        assert results["JsonlSink"] is True
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["pair"] == "EURUSD" and obj["direction"] == "LONG"
        assert obj["confidence"] == 78.0 and len(obj["reasons"]) == 2

    def test_failing_sink_survives(self, tmp_path, caplog):
        path = tmp_path / "signals.jsonl"
        emitter = SignalEmitter([_BrokenSink(), JsonlSink(path)])
        with caplog.at_level(logging.ERROR):
            results = emitter.emit(sample_signal())
        assert results["_BrokenSink"] is False
        assert results["JsonlSink"] is True          # later sinks still ran
        assert path.is_file()                        # scan not crashed
        assert "boom" in caplog.text

    def test_append_order(self, tmp_path):
        path = tmp_path / "signals.jsonl"
        sink = JsonlSink(path)
        sink.send(sample_signal())
        sink.send(sample_signal())
        assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2
