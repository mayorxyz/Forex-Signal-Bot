"""Output layer: emit analysis signals to console, JSONL and optional Telegram.

This package only *announces* signals produced by the engine. It contains no
order-execution code of any kind.
"""

from __future__ import annotations

from fxsignals.output.emitter import (
    ConsoleSink,
    JsonlSink,
    SignalEmitter,
    TelegramSink,
)

__all__ = ["ConsoleSink", "JsonlSink", "SignalEmitter", "TelegramSink"]
