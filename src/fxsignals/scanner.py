"""Market scanner: fetch candles, run the signal engine, emit signals.

Analysis and signal output only — this module never places, modifies or
cancels an order. Signals are produced on closed bars and are meant to be
acted on at the next bar's open.

Entry point::

    python -m fxsignals.scanner --once
    python -m fxsignals.scanner --loop [--interval SECONDS]
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta, timezone

from fxsignals.config import Settings, load_settings
from fxsignals.data.base import get_provider
from fxsignals.data.store import CandleStore
from fxsignals.logging_setup import setup_logging
from fxsignals.models import Signal
from fxsignals.output.emitter import SignalEmitter
from fxsignals.signals.engine import SignalEngine
from fxsignals.signals.factory import build_engine_config

logger = logging.getLogger(__name__)


def _make_emitter(settings: Settings) -> SignalEmitter:
    """Build the standard emitter from the ``output`` settings section."""
    return SignalEmitter.default(
        jsonl_path=settings.output.jsonl_path,
        telegram_enabled=settings.output.telegram_enabled,
    )


def run_scan_once(settings: Settings) -> list[Signal]:
    """Scan every configured pair once and emit any signals found.

    For each pair the bias- and entry-timeframe candles are fetched through
    the configured provider via :class:`~fxsignals.data.store.CandleStore`
    (cached on disk), evaluated by :class:`~fxsignals.signals.engine.SignalEngine`
    on the latest *closed* entry bar, and emitted through console + JSONL
    (+ optional Telegram) sinks. A failure on one pair is logged and skipped;
    it never aborts the scan.

    Args:
        settings: Fully validated application settings.

    Returns:
        All signals emitted during this pass, in pair order.
    """
    provider = get_provider(settings.provider, settings)
    store = CandleStore.from_settings(settings)
    engine = SignalEngine(build_engine_config(settings), entry_timeframe=settings.timeframes.entry_tf)
    emitter = _make_emitter(settings)
    bias_tf, entry_tf = settings.timeframes.bias_tf, settings.timeframes.entry_tf

    all_signals: list[Signal] = []
    for pair in settings.pairs:
        try:
            bias_df = store.get_or_fetch(provider, pair, bias_tf)
            entry_df = store.get_or_fetch(provider, pair, entry_tf)
            signals = engine.evaluate(pair, bias_df, entry_df)
            if signals:
                emitter.emit_all(signals)
                s = signals[0]
                logger.info(
                    "%s: %d/%d bars, SIGNAL %s @ %.5g (score %.1f, RR %.2f)",
                    pair, len(bias_df), len(entry_df), s.direction.value,
                    s.entry, s.confidence, s.rr,
                )
            else:
                logger.info(
                    "%s: %d/%d bars, no signal on latest closed bar (%s)",
                    pair, len(bias_df), len(entry_df), entry_tf.name,
                )
            all_signals.extend(signals)
        except Exception as exc:  # noqa: BLE001 - per-pair failures are not fatal
            logger.error("%s: scan failed, skipping pair: %s", pair, exc, exc_info=True)
    return all_signals


def _next_bar_close_delay(entry_minutes: int, grace_seconds: float = 5.0) -> float:
    """Seconds until shortly after the next entry-TF bar close (UTC grid)."""
    now = datetime.now(timezone.utc)
    step = timedelta(minutes=entry_minutes)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elapsed = (now - day_start).total_seconds()
    closes_at = ((int(elapsed // step.total_seconds()) + 1) * step.total_seconds())
    return max(1.0, closes_at - elapsed + grace_seconds)


def run_scan_loop(settings: Settings, interval_seconds: float | None = None) -> None:
    """Run :func:`run_scan_once` repeatedly, just after each entry-TF close.

    Args:
        settings: Fully validated application settings.
        interval_seconds: Fixed sleep between passes. When None, the loop
            sleeps until a few seconds after the next close of an
            ``entry_tf``-length bar aligned to the UTC midnight grid.

    Notes:
        ``KeyboardInterrupt`` exits cleanly (no traceback, code 0 semantics
        handled by the CLI wrapper).
    """
    entry_minutes = settings.timeframes.entry_tf.minutes
    logger.info(
        "scan loop started (provider=%s, entry_tf=%s, pairs=%d)",
        settings.provider, entry_tf_name(settings), len(settings.pairs),
    )
    while True:
        wait = interval_seconds if interval_seconds else _next_bar_close_delay(entry_minutes)
        logger.debug("next scan in %.0f s", wait)
        time.sleep(wait)
        try:
            run_scan_once(settings)
        except Exception as exc:  # noqa: BLE001 - keep the loop alive
            logger.error("scan pass failed: %s", exc, exc_info=True)


def entry_tf_name(settings: Settings) -> str:
    """Name of the configured entry timeframe (logging convenience)."""
    return settings.timeframes.entry_tf.name


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``python -m fxsignals.scanner --once`` / ``--loop``.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success or clean Ctrl-C).
    """
    parser = argparse.ArgumentParser(
        prog="fxsignals.scanner",
        description="Forex SIGNAL scanner (analysis output only; never executes orders).",
    )
    parser.add_argument("--config", default="config/settings.yaml",
                        help="path to settings.yaml (default: %(default)s)")
    parser.add_argument("--once", action="store_true", help="run a single scan pass")
    parser.add_argument("--loop", action="store_true",
                        help="scan continuously, shortly after each entry-TF bar close")
    parser.add_argument("--interval", type=float, default=None,
                        help="fixed seconds between passes in --loop mode")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    setup_logging(level=settings.log_level)
    if not (args.once or args.loop):
        parser.error("choose --once or --loop")
    try:
        if args.once:
            signals = run_scan_once(settings)
            logger.info("scan complete: %d signal(s) emitted", len(signals))
        else:
            run_scan_loop(settings, interval_seconds=args.interval)
    except KeyboardInterrupt:
        logger.info("interrupted; shutting down cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
