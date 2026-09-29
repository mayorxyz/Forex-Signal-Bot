"""Signal engine: turns feature frames into trade signals (analysis only).

The engine evaluates **closed bars only**: the signal at bar ``t`` uses data
up to ``t`` and is meant to be acted on at the next bar's open. Live
(:meth:`SignalEngine.evaluate`) and historical
(:meth:`SignalEngine.evaluate_history`) paths share one code path so results
match bar-for-bar. No order execution exists anywhere in this package.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import pandas as pd

from fxsignals.features.levels import SupportResistanceZone, support_resistance_zones
from fxsignals.features.pipeline import build_features, build_mtf_features
from fxsignals.models import Signal, Timeframe
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
    build_trade_levels,
    pip_size,
    set_pips,
)
from fxsignals.signals.scoring import ScoreWeights, score_both

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngineConfig:
    """Aggregate configuration for :class:`SignalEngine`.

    Attributes:
        weights: Confluence group weights (must sum to 100).
        risk: Trade-level / R:R rules.
        filters: Pre-signal filter knobs.
        min_score: Minimum direction score to emit (default 65).
        direction_margin: Winning score must exceed the other by this (15).
        warmup_bars: Bars skipped before evaluation begins.
    """

    weights: ScoreWeights = field(default_factory=ScoreWeights)
    risk: RiskConfig = field(default_factory=RiskConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    min_score: float = 65.0
    direction_margin: float = 15.0
    warmup_bars: int = 60

    def validated(self) -> "EngineConfig":
        """Return self after validating every sub-config.

        Raises:
            ValueError: on invalid weights/risk/filters or out-of-range gates.
        """
        self.weights.validated()
        self.risk.validated()
        self.filters.validated()
        if not 0 <= self.min_score <= 100:
            raise ValueError(f"min_score must be within [0, 100], got {self.min_score}")
        if self.direction_margin < 0:
            raise ValueError(f"direction_margin must be >= 0, got {self.direction_margin}")
        if self.warmup_bars < 0:
            raise ValueError(f"warmup_bars must be >= 0, got {self.warmup_bars}")
        return self


def _zones_known_at(zones_df: pd.DataFrame, ts: pd.Timestamp) -> list[SupportResistanceZone]:
    """Zones whose ``formed_at`` is at or before ``ts`` (causal view)."""
    if zones_df.empty:
        return []
    formed = pd.DatetimeIndex(pd.to_datetime(zones_df["formed_at"], utc=True))
    mask = formed <= pd.Timestamp(ts)
    out: list[SupportResistanceZone] = []
    for _, r in zones_df[mask].iterrows():
        kinds = r["kinds"]
        if isinstance(kinds, str):
            kinds = frozenset(x for x in kinds.split(",") if x)
        out.append(
            SupportResistanceZone(
                low=float(r["low"]), high=float(r["high"]), center=float(r["center"]),
                touches=int(r["touches"]), formed_at=pd.Timestamp(r["formed_at"]),
                kinds=frozenset(kinds),
            )
        )
    return out


class SignalEngine:
    """Evaluate features and emit :class:`Signal` objects for closed bars.

    Args:
        config: Aggregate :class:`EngineConfig` (validated on construction).
        entry_timeframe: Timeframe stamped on produced signals (default H1);
            the scanner passes ``settings.timeframes.entry_tf``.
    """

    def __init__(
        self,
        config: EngineConfig | None = None,
        entry_timeframe: Timeframe = Timeframe.H1,
    ) -> None:
        self.config = (config or EngineConfig()).validated()
        self.timeframe = entry_timeframe
        self._cooldown = CooldownFilter(self.config.filters)
        self._duplicates = DuplicateFilter(self.config.filters)

    def reset_state(self) -> None:
        """Clear cooldown/duplicate memory (e.g. between backtest runs)."""
        self._cooldown = CooldownFilter(self.config.filters)
        self._duplicates = DuplicateFilter(self.config.filters)

    # ------------------------------------------------------------------ core
    def _evaluate_bar(
        self,
        pair: str,
        ts: pd.Timestamp,
        entry_row: pd.Series,
        htf_row: pd.Series | None,
        atr_value: float,
        zones: list[SupportResistanceZone],
        atr_trailing: pd.Series | None = None,
    ) -> tuple[list[Signal], list[str]]:
        """Score both directions on one closed bar; apply filters and risk.

        Returns:
            ``(signals, rejection_notes)`` — at most one signal per bar.
        """
        cfg = self.config
        long_s, short_s = score_both(entry_row, htf_row, cfg.weights)
        ranked = sorted([long_s, short_s], key=lambda s: s.score, reverse=True)
        best, other = ranked[0], ranked[1]
        notes: list[str] = []

        if best.score < cfg.min_score:
            notes.append(f"{best.direction.value} score {best.score:.1f} < min {cfg.min_score}")
            return [], notes
        if best.score - other.score < cfg.direction_margin:
            notes.append(
                f"margin {best.score - other.score:.1f} < {cfg.direction_margin} "
                f"(LONG {long_s.score:.1f} vs SHORT {short_s.score:.1f})"
            )
            return [], notes

        passed, reason = adx_filter(entry_row, cfg.filters)
        if not passed:
            notes.append(reason)
            return [], notes
        passed, reason = session_filter(ts, cfg.filters)
        if not passed:
            notes.append(reason)
            return [], notes
        if atr_trailing is not None and len(atr_trailing) > 0:
            passed, reason = volatility_filter(atr_trailing, len(atr_trailing) - 1, cfg.filters)
            if not passed:
                notes.append(reason)
                return [], notes

        close = float(entry_row.get("close"))
        if not math.isfinite(atr_value) or atr_value <= 0 or not math.isfinite(close):
            notes.append("ATR/close unavailable (warm-up)")
            return [], notes
        try:
            levels = build_trade_levels(
                best.direction,
                entry=close,
                atr=atr_value,
                zones=zones,
                cfg=cfg.risk,
                last_swing_low=_opt_float(entry_row.get("st_last_swing_low")),
                last_swing_high=_opt_float(entry_row.get("st_last_swing_high")),
            )
        except RiskError as exc:
            notes.append(str(exc))
            return [], notes
        levels = set_pips(levels, pair)

        passed, reason = self._duplicates.check(pair, best.direction, levels.entry, atr_value)
        if not passed:
            notes.append(reason)
            return [], notes
        # Cooldown measures bar distance on the entry index when available.
        bar_index = entry_row.attrs.get("bar_index") or pd.Index([ts])
        passed, reason = self._cooldown.check(pair, best.direction, ts, bar_index)
        if not passed:
            notes.append(reason)
            return [], notes

        reasons = [g.reason for g in best.groups] + [
            levels.note,
            f"SL {levels.sl_pips:.1f} pips / TP {levels.tp_pips:.1f} pips ({pair})",
            f"Evaluated on closed bar {ts.isoformat()} — act at next bar open.",
        ]
        sig = Signal(
            pair=pair.upper(),
            direction=best.direction,
            entry=round(levels.entry, 8),
            stop_loss=round(levels.stop_loss, 8),
            take_profit=round(levels.take_profit, 8),
            rr=round(levels.rr, 3),
            confidence=round(min(best.score, 100.0), 2),
            timeframe=self.timeframe,
            reasons=reasons,
            timestamp=pd.Timestamp(ts).to_pydatetime(),
        )
        return [sig], notes

    # --------------------------------------------------------------- history
    def evaluate_history(
        self,
        pair: str,
        bias_df: pd.DataFrame,
        entry_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run the engine over every closed entry bar of ``entry_df``.

        Causal guarantees: each bar ``t`` sees features computed from bars
        ``<= t`` (feature layer contract), HTF values only after their bar has
        closed (``align_htf_to_ltf``), zone lists filtered by ``formed_at``,
        the volatility band uses a trailing ATR window, and stateful
        cooldown/duplicate filters replay in time order. The final (possibly
        still forming) bar is never evaluated.

        Args:
            pair: Instrument symbol.
            bias_df: Closed candles on the bias timeframe.
            entry_df: Closed candles on the entry timeframe.

        Returns:
            DataFrame indexed by signal bar time with columns ``direction,
            score, entry, stop_loss, take_profit, rr, sl_pips, tp_pips,
            confidence, reasons`` — rows only where a signal fired (empty
            frame with those columns otherwise).
        """
        cfg = self.config
        feats = build_features(entry_df)
        mtf = build_mtf_features(bias_df, entry_df)
        _, zones_df = support_resistance_zones(entry_df)
        atr_col = feats["ind_atr"]

        self.reset_state()
        records: dict[pd.Timestamp, dict[str, object]] = {}
        n = len(entry_df)
        index = entry_df.index
        for pos in range(cfg.warmup_bars, n - 1):  # last bar may still be forming
            ts = index[pos]
            row = feats.iloc[pos].copy()
            row["close"] = float(entry_df["close"].iloc[pos])
            # Cooldown needs the full entry index to measure bar distance.
            row.attrs["bar_index"] = index
            lo = max(0, pos - cfg.filters.atr_lookback)
            sigs, _notes = self._evaluate_bar(
                pair, ts, row, mtf.iloc[pos], float(atr_col.iloc[pos]),
                _zones_known_at(zones_df, ts),
                atr_trailing=atr_col.iloc[lo: pos + 1],
            )
            for s in sigs:
                records[ts] = {
                    "direction": s.direction.value,
                    "score": s.confidence,
                    "entry": s.entry,
                    "stop_loss": s.stop_loss,
                    "take_profit": s.take_profit,
                    "rr": s.rr,
                    "sl_pips": abs(s.stop_loss - s.entry) / pip_size(s.pair),
                    "tp_pips": abs(s.take_profit - s.entry) / pip_size(s.pair),
                    "confidence": s.confidence,
                    "reasons": "; ".join(s.reasons),
                }
        cols = ["direction", "score", "entry", "stop_loss", "take_profit", "rr",
                "sl_pips", "tp_pips", "confidence", "reasons"]
        if not records:
            return pd.DataFrame(columns=cols)
        out = pd.DataFrame.from_records(
            list(records.values()), index=pd.DatetimeIndex(list(records), tz="UTC", name="time")
        )
        return out[cols]

    # ------------------------------------------------------------------ live
    def evaluate(
        self,
        pair: str,
        bias_df: pd.DataFrame,
        entry_df: pd.DataFrame,
    ) -> list[Signal]:
        """Signals for the latest *closed* entry bar of ``entry_df``.

        Uses exactly the same scoring, filtering and risk logic as
        :meth:`evaluate_history`; returns ``[]`` when no setup qualifies.

        Args:
            pair: Instrument symbol.
            bias_df: Closed bias-timeframe candles (UTC contract).
            entry_df: Closed entry-timeframe candles (UTC contract).

        Returns:
            A list with zero or one :class:`Signal`.
        """
        cfg = self.config
        if len(entry_df) < 2:
            logger.debug("%s: not enough bars to evaluate a closed bar", pair)
            return []
        feats = build_features(entry_df)
        mtf = build_mtf_features(bias_df, entry_df)
        _, zones_df = support_resistance_zones(entry_df)
        pos = len(entry_df) - 2  # newest fully closed bar
        ts = entry_df.index[pos]
        row = feats.iloc[pos].copy()
        row["close"] = float(entry_df["close"].iloc[pos])
        atr_value = float(feats["ind_atr"].iloc[pos])
        if not math.isfinite(atr_value) or atr_value <= 0:
            return []
        # Trailing ATR series for the causal percentile band.
        trailing = feats["ind_atr"].iloc[max(0, pos - cfg.filters.atr_lookback): pos + 1]
        sigs, notes = self._evaluate_bar(
            pair, ts, row, mtf.iloc[pos], atr_value, _zones_known_at(zones_df, ts),
            atr_trailing=trailing,
        )
        for n in notes:
            logger.debug("%s @ %s: %s", pair, ts, n)
        return sigs


def _opt_float(val: object) -> float | None:
    """Best-effort optional float (None for NaN/None/non-numeric)."""
    try:
        f = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None
