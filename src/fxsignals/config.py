"""Typed configuration loading for fxsignals.

Reads ``config/settings.yaml`` into frozen dataclasses, validating every key
and raising :class:`ConfigError` with a clear message on any problem.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Settings:
    """Fully validated application settings."""

    pairs: tuple[str, ...]
    timeframes: TimeframeConfig
    provider: str
    paths: PathsConfig
    log_level: str


def _require(data: dict[str, Any], key: str, where: str) -> Any:
    if not isinstance(data, dict) or key not in data:
        raise ConfigError(f"Missing required key '{key}' in {where}")
    return data[key]


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

    unknown = set(raw) - {"pairs", "timeframes", "provider", "paths", "log_level"}
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

    return Settings(
        pairs=pairs,
        timeframes=TimeframeConfig(bias_tf=bias, entry_tf=entry),
        provider=provider,
        paths=PathsConfig(data_dir=data_dir, cache_dir=cache_dir),
        log_level=log_level,
    )


def default_settings() -> Settings:
    """Return built-in defaults (useful for tests and quick starts)."""
    return Settings(
        pairs=DEFAULT_PAIRS,
        timeframes=TimeframeConfig(bias_tf=Timeframe.H4, entry_tf=Timeframe.H1),
        provider="synthetic",
        paths=PathsConfig(data_dir=Path("./data"), cache_dir=Path("./.cache")),
        log_level="INFO",
    )
