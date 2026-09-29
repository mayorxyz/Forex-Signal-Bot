"""Signal emitters: console block, JSONL append file, optional Telegram hook.

Design rules:

- An emitter failure must never crash a scan: every sink is invoked inside a
  try/except; failures are logged and the remaining sinks still run.
- Telegram is **disabled by default**. When explicitly enabled, the bot token
  and chat id come from environment variables (``FXSIGNALS_TELEGRAM_TOKEN`` /
  ``FXSIGNALS_TELEGRAM_CHAT_ID``) — never from settings.yaml — and the request
  uses only :mod:`urllib` (no extra dependencies).
- JSONL lines are exactly ``json.dumps(signal.to_dict())``, one object per
  line, appended in UTC time order of emission.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Protocol

from fxsignals.models import Signal

logger = logging.getLogger(__name__)

TELEGRAM_TOKEN_ENV = "FXSIGNALS_TELEGRAM_TOKEN"
TELEGRAM_CHAT_ID_ENV = "FXSIGNALS_TELEGRAM_CHAT_ID"


class Sink(Protocol):
    """Minimal interface every emitter sink implements."""

    def send(self, signal: Signal) -> None:
        """Deliver one signal; raise on failure (the emitter catches)."""
        ...


class ConsoleSink:
    """Prints a clean, readable multi-line block per signal."""

    def send(self, signal: Signal) -> None:
        """Render ``signal`` as a bordered text block on stdout."""
        d = signal.to_dict()
        bar = "-" * 58
        print(f"\n+{bar}+")
        print(f"| SIGNAL  {d['pair']:<8} {d['direction']:<6} [{d['timeframe']}]")
        print(
            f"| Entry   {d['entry']:<12} SL  {d['stop_loss']:<12} TP  {d['take_profit']:<12}"
        )
        rr = d["rr"]
        print(
            f"| R:R     {'' if rr is None else format(rr, '.2f'):<12} "
            f"Conf  {d['confidence']:.1f}/100"
        )
        print(f"| Bar     {d['timestamp']} (act at next bar open)")
        for reason in d["reasons"]:
            print(f"|  * {reason}")
        print(f"+{bar}+")


class JsonlSink:
    """Appends one JSON object per emitted signal to a ``*.jsonl`` file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def send(self, signal: Signal) -> None:
        """Append ``signal.to_dict()`` as one JSONL line (UTF-8)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(signal.to_dict(), ensure_ascii=False, default=str)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


class TelegramSink:
    """Optional Telegram hook (disabled by default; env-var credentials).

    Args:
        enabled: When False, :meth:`send` is a no-op.
        timeout: HTTP timeout seconds for the API call.
        api_base: Telegram Bot API base URL (overridable for tests).
    """

    def __init__(
        self, enabled: bool = False, timeout: float = 10.0, api_base: str | None = None
    ) -> None:
        self.enabled = enabled
        self.timeout = timeout
        self.api_base = api_base or "https://api.telegram.org"

    def send(self, signal: Signal) -> None:
        """POST the signal as plain text via urllib; raise on API errors.

        Raises:
            RuntimeError: when enabled but env credentials are missing.
            urllib.error.URLError / ValueError: on transport or API failure.
        """
        if not self.enabled:
            return
        token = os.environ.get(TELEGRAM_TOKEN_ENV, "").strip()
        chat_id = os.environ.get(TELEGRAM_CHAT_ID_ENV, "").strip()
        if not token or not chat_id:
            raise RuntimeError(
                "Telegram sink enabled but "
                f"{TELEGRAM_TOKEN_ENV}/{TELEGRAM_CHAT_ID_ENV} are not set"
            )
        d = signal.to_dict()
        reasons = "\n".join(f"- {r}" for r in d["reasons"])
        text = (
            f"fxsignals: {d['pair']} {d['direction']} ({d['timeframe']})\n"
            f"Entry {d['entry']} | SL {d['stop_loss']} | TP {d['take_profit']}\n"
            f"RR {d['rr']} | confidence {d['confidence']}\n"
            f"Bar {d['timestamp']} — analysis only, act at next bar open.\n{reasons}"
        )
        url = f"{self.api_base}/bot{token}/sendMessage"
        data = urllib.parse.urlencode(
            {"chat_id": chat_id, "text": text}
        ).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        if not body.get("ok", False):
            raise ValueError(f"Telegram API error: {body!r}")


class SignalEmitter:
    """Fan out one signal to several sinks; isolate every sink failure.

    Args:
        sinks: Ordered sink objects; each must expose ``send(signal)``.
    """

    def __init__(self, sinks: list[Sink] | None = None) -> None:
        self.sinks: list[Sink] = list(sinks) if sinks is not None else []

    @classmethod
    def default(
        cls,
        jsonl_path: str | Path = "signals.jsonl",
        telegram_enabled: bool = False,
    ) -> "SignalEmitter":
        """Build the standard console + JSONL (+ optional Telegram) emitter.

        Args:
            jsonl_path: Destination of the append-only JSONL signal log.
            telegram_enabled: Turn the Telegram hook on (env creds required).

        Returns:
            A ready-to-use :class:`SignalEmitter`.
        """
        return cls(
            [
                ConsoleSink(),
                JsonlSink(jsonl_path),
                TelegramSink(enabled=telegram_enabled),
            ]
        )

    def emit(self, signal: Signal) -> dict[str, bool]:
        """Send ``signal`` to every sink; never raise.

        Returns:
            Mapping of sink class name -> delivered (True/False). Failures are
            logged at ERROR level with full exception info and swallowed so a
            single bad sink cannot abort a scan.
        """
        results: dict[str, bool] = {}
        for sink in self.sinks:
            name = type(sink).__name__
            try:
                sink.send(signal)
                results[name] = True
            except Exception as exc:  # noqa: BLE001 - sinks must never be fatal
                results[name] = False
                logger.error(
                    "emitter sink %s failed for %s %s: %s",
                    name,
                    signal.pair,
                    signal.direction.value,
                    exc,
                    exc_info=True,
                )
        return results

    def emit_all(self, signals: list[Signal]) -> int:
        """Emit a batch; returns the number of signals processed."""
        for s in signals:
            self.emit(s)
        return len(signals)
