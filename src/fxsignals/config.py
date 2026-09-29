"""Typed configuration loading for fxsignals.

Reads ``config/settings.yaml`` into frozen dataclasses, validating every key
and raising :class:`ConfigError` with a clear message on any problem.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from fxsignals.models import Timeframe

VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
VALID_PROVIDERS = ("csv", "synthetic")
DEFAULT_PAIRS = (
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "USDCHF",
    "NZDUSD",
    "XAUUSD",
)
_SYMBOL_RE = re.compile(r"^[A-Z]{3,6}$")


class ConfigError(Exception):
    """Raised when settings.yaml is missing keys or has invalid values."""


@dataclass(frozen=True)
class TimeframeConfig:
    """Timeframe roles used by the analysis pipeline."""

    bias_tf: Timeframe
    entry_tf: Timeframe


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations for raw data, caches and logs."""

    data_dir: Path
    cache_dir: Path


SCORE_WEIGHT_KEYS = (
    "htf_bias",
    "structure",
    "location",
    "momentum",
    "trend_strength",
    "trigger",
)
VALID_SESSION_TAGS = ("asia", "london", "newyork", "overlap")


@dataclass(frozen=True)
class SignalsConfig:
    """Validated ``signals`` section: scoring gates and risk-level rules.

    Attributes:
        weights: Confluence group weights; must sum to exactly 100.
        min_score: Minimum direction score to emit a signal (0-100).
        direction_margin: Winning direction must beat the other by this much.
        rr_target: Reward:risk target used to project take profit.
        min_rr: Reject setups whose realized R:R is below this (> 0).
        min_sl_atr / max_sl_atr: Allowed stop distance bounds in ATR units.
        sl_atr_buffer: Extra ATR buffer beyond the protecting swing/zone.
        cooldown_bars: No new signal for the same pair+direction within this
            many entry-TF bars (int >= 0; 0 disables).
    """

    weights: dict[str, float]
    min_score: float
    direction_margin: float
    rr_target: float
    min_rr: float
    min_sl_atr: float
    max_sl_atr: float
    sl_atr_buffer: float
    cooldown_bars: int = 12


@dataclass(frozen=True)
class FiltersSectionConfig:
    """Validated ``filters`` section (pre-signal filter knobs).

    Attributes:
        min_adx: Minimum entry-TF ADX (0 disables the regime filter).
        sessions: Allowed session tags (subset of VALID_SESSION_TAGS).
        atr_percentile_low / atr_percentile_high: ATR percentile band bounds.
        lookback: Bars used to compute that percentile band.
    """

    min_adx: float
    sessions: tuple[str, ...]
    atr_percentile_low: float
    atr_percentile_high: float
    lookback: int


@dataclass(frozen=True)
class OutputConfig:
    """Validated ``output`` section (emitter destinations).

    Attributes:
        jsonl_path: Append-only JSONL signal log location.
        telegram_enabled: Telegram hook flag (disabled by default; token and
            chat id come from environment variables, never from YAML).
    """

    jsonl_path: Path
    telegram_enabled: bool


@dataclass(frozen=True)
class Settings:
    """Fully validated application settings."""

    pairs: tuple[str, ...] = DEFAULT_PAIRS
    timeframes: TimeframeConfig = field(
        default_factory=lambda: TimeframeConfig(bias_tf=Timeframe.H4, entry_tf=Timeframe.H1)
    )
    provider: str = "synthetic"
    paths: PathsConfig = field(
        default_factory=lambda: PathsConfig(data_dir=Path("./data"), cache_dir=Path("./.cache"))
    )
    log_level: str = "INFO"
    signals: SignalsConfig = field(default_factory=lambda: default_signals())
    filters: FiltersSectionConfig = field(default_factory=lambda: default_filters())
    output: OutputConfig = field(default_factory=lambda: default_output())


def _require(data: dict[str, Any], key: str, where: str) -> Any:
    if not isinstance(data, dict) or key not in data:
        raise ConfigError(f"Missing required key '{key}' in {where}")
    return data[key]


def _number(data: dict[str, Any], key: str, where: str) -> float:
    """Return ``data[key]`` as a finite float or raise :class:`ConfigError`."""
    raw = _require(data, key, where)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ConfigError(f"{where}.{key} must be a number, got {raw!r}")
    value = float(raw)
    if value != value or value in (float("inf"), float("-inf")):  # NaN / ±inf
        raise ConfigError(f"{where}.{key} must be finite, got {raw!r}")
    return value


def _parse_signals(raw: dict[str, Any] | None) -> SignalsConfig:
    """Validate the ``signals`` mapping into a :class:`SignalsConfig`.

    Args:
        raw: The ``signals`` mapping (None falls back to defaults).

    Raises:
        ConfigError: on unknown keys, non-numeric values, weights not summing
            to 100, min_score outside 0-100, min_rr <= 0 or bad SL bounds.
    """
    where = "signals"
    if raw is None:
        return default_signals()
    if not isinstance(raw, dict):
        raise ConfigError("'signals' must be a mapping")
    expected = {"weights", "min_score", "direction_margin", "rr_target", "min_rr",
                "min_sl_atr", "max_sl_atr", "sl_atr_buffer", "cooldown_bars"}
    unknown = sorted(set(raw) - expected)
    if unknown:
        raise ConfigError(f"Unknown keys in '{where}': {unknown}")

    weights_raw = _require(raw, "weights", where)
    if not isinstance(weights_raw, dict):
        raise ConfigError(f"{where}.weights must be a mapping of group -> number")
    missing_w = [k for k in SCORE_WEIGHT_KEYS if k not in weights_raw]
    extra_w = sorted(set(weights_raw) - set(SCORE_WEIGHT_KEYS))
    if missing_w or extra_w:
        raise ConfigError(
            f"{where}.weights keys mismatch; missing={missing_w}, unknown={extra_w}"
        )
    weights = {k: _number(weights_raw, k, f"{where}.weights") for k in SCORE_WEIGHT_KEYS}
    total = sum(weights.values())
    if abs(total - 100.0) > 0.01:
        raise ConfigError(f"{where}.weights must sum to 100, got {total}")
    negative = sorted(k for k, v in weights.items() if v < 0)
    if negative:
        raise ConfigError(f"{where}.weights must be non-negative, got {negative}")

    min_score = _number(raw, "min_score", where)
    if not 0.0 <= min_score <= 100.0:
        raise ConfigError(f"{where}.min_score must be within [0, 100], got {min_score}")
    direction_margin = _number(raw, "direction_margin", where)
    if direction_margin < 0:
        raise ConfigError(f"{where}.direction_margin must be >= 0, got {direction_margin}")
    min_rr = _number(raw, "min_rr", where)
    if min_rr <= 0:
        raise ConfigError(f"{where}.min_rr must be > 0, got {min_rr}")
    rr_target = _number(raw, "rr_target", where)
    if rr_target <= 0:
        raise ConfigError(f"{where}.rr_target must be > 0, got {rr_target}")
    min_sl_atr = _number(raw, "min_sl_atr", where)
    max_sl_atr = _number(raw, "max_sl_atr", where)
    if min_sl_atr <= 0 or max_sl_atr < min_sl_atr:
        raise ConfigError(
            f"{where} needs 0 < min_sl_atr <= max_sl_atr, got {min_sl_atr}, {max_sl_atr}"
        )
    sl_atr_buffer = _number(raw, "sl_atr_buffer", where)
    if sl_atr_buffer < 0:
        raise ConfigError(f"{where}.sl_atr_buffer must be >= 0, got {sl_atr_buffer}")

    cooldown_raw = raw.get("cooldown_bars", 12)
    if isinstance(cooldown_raw, bool) or not isinstance(cooldown_raw, int):
        if isinstance(cooldown_raw, float) and cooldown_raw.is_integer():
            cooldown_raw = int(cooldown_raw)
        else:
            raise ConfigError(
                f"{where}.cooldown_bars must be an integer >= 0, got {cooldown_raw!r}"
            )
    if cooldown_raw < 0:
        raise ConfigError(f"{where}.cooldown_bars must be >= 0, got {cooldown_raw}")

    return SignalsConfig(
        weights=weights,
        min_score=min_score,
        direction_margin=direction_margin,
        rr_target=rr_target,
        min_rr=min_rr,
        min_sl_atr=min_sl_atr,
        max_sl_atr=max_sl_atr,
        sl_atr_buffer=sl_atr_buffer,
        cooldown_bars=int(cooldown_raw),
    )


def _parse_filters(raw: dict[str, Any]) -> FiltersSectionConfig:
    """Validate the ``filters`` mapping into a :class:`FiltersSectionConfig`.

    Raises:
        ConfigError: on unknown keys, bad numbers or invalid session names.
    """
    where = "filters"
    if not isinstance(raw, dict):
        raise ConfigError("'filters' must be a mapping")
    expected = {"min_adx", "sessions", "atr_percentile_low", "atr_percentile_high",
                "lookback"}
    unknown = sorted(set(raw) - expected)
    if unknown:
        raise ConfigError(f"Unknown keys in '{where}': {unknown}")

    min_adx = _number(raw, "min_adx", where)
    if min_adx < 0:
        raise ConfigError(f"{where}.min_adx must be >= 0, got {min_adx}")
    sessions_raw = _require(raw, "sessions", where)
    if (
        not isinstance(sessions_raw, list)
        or not sessions_raw
        or not all(isinstance(s, str) for s in sessions_raw)
    ):
        raise ConfigError(f"{where}.sessions must be a non-empty list of session names")
    sessions = tuple(s.strip().lower() for s in sessions_raw)
    bad = sorted(set(sessions) - set(VALID_SESSION_TAGS))
    if bad:
        raise ConfigError(
            f"{where}.sessions contains invalid names {bad}; allowed: {VALID_SESSION_TAGS}"
        )
    low = _number(raw, "atr_percentile_low", where)
    high = _number(raw, "atr_percentile_high", where)
    if not 0.0 <= low < high <= 100.0:
        raise ConfigError(
            f"{where} needs 0 <= atr_percentile_low < atr_percentile_high <= 100, "
            f"got {low}, {high}"
        )
    lookback = _number(raw, "lookback", where)
    if lookback < 2 or lookback != int(lookback):
        raise ConfigError(f"{where}.lookback must be an integer >= 2, got {lookback}")

    return FiltersSectionConfig(
        min_adx=min_adx,
        sessions=sessions,
        atr_percentile_low=low,
        atr_percentile_high=high,
        lookback=int(lookback),
    )


def _parse_output(raw: dict[str, Any]) -> OutputConfig:
    """Validate the ``output`` mapping into an :class:`OutputConfig`.

    Raises:
        ConfigError: on unknown keys or wrongly-typed values.
    """
    where = "output"
    if not isinstance(raw, dict):
        raise ConfigError("'output' must be a mapping")
    unknown = sorted(set(raw) - {"jsonl_path", "telegram_enabled"})
    if unknown:
        raise ConfigError(f"Unknown keys in '{where}': {unknown}")
    jsonl_path = Path(str(_require(raw, "jsonl_path", where)))
    telegram_enabled = _require(raw, "telegram_enabled", where)
    if not isinstance(telegram_enabled, bool):
        raise ConfigError(
            f"{where}.telegram_enabled must be true/false, got {telegram_enabled!r}"
        )
    return OutputConfig(jsonl_path=jsonl_path, telegram_enabled=telegram_enabled)


def _parse_timeframe(value: Any, where: str) -> Timeframe:
    if not isinstance(value, str):
        raise ConfigError(f"{where} must be a string timeframe code, got {value!r}")
    try:
        return Timeframe.from_str(value)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def load_settings(path: str | Path = "config/settings.yaml") -> Settings:
    """Load and validate settings from a YAML file.

    Args:
        path: Location of the YAML settings file.

    Returns:
        A frozen :class:`Settings` instance.

    Raises:
        ConfigError: if the file is missing/unreadable or any value is invalid.
    """
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Config file not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {p}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Top level of {p} must be a mapping, got {type(raw).__name__}")

    unknown = set(raw) - {
        "pairs", "timeframes", "provider", "paths", "log_level",
        "signals", "filters", "output",
    }
    if unknown:
        raise ConfigError(f"Unknown top-level config keys: {sorted(unknown)}")

    pairs_raw = _require(raw, "pairs", "settings")
    if not isinstance(pairs_raw, list) or not all(isinstance(x, str) and x for x in pairs_raw):
        raise ConfigError("'pairs' must be a non-empty list of symbol strings")
    pairs = tuple(x.strip().upper() for x in pairs_raw)
    bad_symbols = [x for x in pairs if not _SYMBOL_RE.match(x)]
    if bad_symbols:
        raise ConfigError(f"Invalid symbol(s) in 'pairs': {bad_symbols}")
    if len(set(pairs)) != len(pairs):
        raise ConfigError(f"'pairs' contains duplicates: {list(pairs)}")

    tf_raw = _require(raw, "timeframes", "settings")
    if not isinstance(tf_raw, dict):
        raise ConfigError("'timeframes' must be a mapping with bias_tf and entry_tf")
    bias = _parse_timeframe(_require(tf_raw, "bias_tf", "timeframes"), "timeframes.bias_tf")
    entry = _parse_timeframe(_require(tf_raw, "entry_tf", "timeframes"), "timeframes.entry_tf")
    if bias.minutes <= entry.minutes:
        raise ConfigError(
            f"timeframes.bias_tf ({bias.name}) must be coarser than entry_tf ({entry.name})"
        )

    provider = _require(raw, "provider", "settings")
    if not isinstance(provider, str) or provider.strip().lower() not in VALID_PROVIDERS:
        raise ConfigError(f"'provider' must be one of {VALID_PROVIDERS}, got {provider!r}")
    provider = provider.strip().lower()

    paths_raw = _require(raw, "paths", "settings")
    if not isinstance(paths_raw, dict):
        raise ConfigError("'paths' must be a mapping with data_dir and cache_dir")
    data_dir = Path(str(_require(paths_raw, "data_dir", "paths")))
    cache_dir = Path(str(_require(paths_raw, "cache_dir", "paths")))

    log_level = str(_require(raw, "log_level", "settings")).strip().upper()
    if log_level not in VALID_LOG_LEVELS:
        raise ConfigError(f"'log_level' must be one of {VALID_LOG_LEVELS}, got {log_level!r}")

    # Optional sections: fall back to defaults built from the same values as
    # config/settings.yaml when a section is absent.
    signals = _parse_signals(raw.get("signals")) if "signals" in raw else default_signals()
    filters = _parse_filters(raw.get("filters")) if "filters" in raw else default_filters()
    output = _parse_output(raw.get("output")) if "output" in raw else default_output()

    return Settings(
        pairs=pairs,
        timeframes=TimeframeConfig(bias_tf=bias, entry_tf=entry),
        provider=provider,
        paths=PathsConfig(data_dir=data_dir, cache_dir=cache_dir),
        log_level=log_level,
        signals=signals,
        filters=filters,
        output=output,
    )


def default_signals() -> SignalsConfig:
    """Built-in ``signals`` defaults (mirror config/settings.yaml)."""
    return SignalsConfig(
        weights={
            "htf_bias": 30.0,
            "structure": 20.0,
            "location": 20.0,
            "momentum": 10.0,
            "trend_strength": 10.0,
            "trigger": 10.0,
        },
        min_score=65.0,
        direction_margin=15.0,
        rr_target=2.0,
        min_rr=1.5,
        min_sl_atr=0.5,
        max_sl_atr=3.0,
        sl_atr_buffer=0.25,
        cooldown_bars=12,
    )


def default_filters() -> FiltersSectionConfig:
    """Built-in ``filters`` defaults (mirror config/settings.yaml)."""
    return FiltersSectionConfig(
        min_adx=18.0,
        sessions=("london", "newyork", "overlap"),
        atr_percentile_low=10.0,
        atr_percentile_high=95.0,
        lookback=180,
    )


def default_output() -> OutputConfig:
    """Built-in ``output`` defaults (mirror config/settings.yaml).

    Note: ``Path("./signals.jsonl")`` normalizes to ``Path("signals.jsonl")``.
    """
    return OutputConfig(jsonl_path=Path("./signals.jsonl"), telegram_enabled=False)


def default_settings() -> Settings:
    """Return built-in defaults (useful for tests and quick starts)."""
    return Settings(
        pairs=DEFAULT_PAIRS,
        timeframes=TimeframeConfig(bias_tf=Timeframe.H4, entry_tf=Timeframe.H1),
        provider="synthetic",
        paths=PathsConfig(data_dir=Path("./data"), cache_dir=Path("./.cache")),
        log_level="INFO",
        signals=default_signals(),
        filters=default_filters(),
        output=default_output(),
    )
