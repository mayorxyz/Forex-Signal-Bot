# fxsignals

Foundation of a forex **signal** bot. This project produces analysis output
only — it contains no order-execution code and never will in its current form.

## Layout

```
config/settings.yaml        # pairs, timeframes, provider, paths, log level
src/fxsignals/config.py     # typed, validated settings loader
src/fxsignals/models.py     # Timeframe enum, Candle contract, Signal dataclass
src/fxsignals/data/         # providers, validation, resampling, cache store
src/fxsignals/logging_setup.py
tests/test_data_layer.py
```

## Setup (Python 3.11+)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .   # or add src/ to PYTHONPATH
```

## Usage

```python
from fxsignals.config import load_settings
from fxsignals.data.base import get_provider
from fxsignals.data.resample import resample_ohlc, align_htf_to_ltf
from fxsignals.data.store import CandleStore
from fxsignals.logging_setup import setup_logging

settings = load_settings("config/settings.yaml")
setup_logging(settings.log_level)

provider = get_provider(settings.provider, settings)
store = CandleStore.from_settings(settings)

h1 = store.get_or_fetch(provider, "EURUSD", settings.timeframes.entry_tf)
h4 = resample_ohlc(h1, settings.timeframes.bias_tf)          # closed bars only
htf_ref = align_htf_to_ltf(h4, h1)                           # last CLOSED H4 per H1 bar
print(h1.tail(), htf_ref.tail())
```

- **Candle contract**: UTC `DatetimeIndex` named `time`, columns
  `open, high, low, close, volume`, sorted ascending, no duplicates, positive
  prices with `high >= max(open, close)` and `low <= min(open, close)`.
- **No look-ahead**: `resample_ohlc` drops the incomplete trailing bar by
  default (`keep_incomplete_last=True` overrides), and `align_htf_to_ltf`
  exposes an HTF bar only after its close time.
- **CSV provider** expects `{data_dir}/{PAIR}_{TF}.csv` with columns
  `time,open,high,low,close,volume` (volume optional, parsed as UTC).
- **Synthetic provider** is deterministic: same seed ⇒ identical candles,
  trending/ranging regimes, volatility clustering, weekend gaps excluded.

## Adding a new data provider

Subclass `DataProvider` and register it — nothing else changes:

```python
from fxsignals.data.base import DataProvider, register_provider
from fxsignals.models import Timeframe

@register_provider("oanda")            # e.g. also "mt5", "twelvedata"
class OandaProvider(DataProvider):
    def get_candles(self, pair, timeframe: Timeframe, start=None, end=None, limit=None):
        # fetch from the API, then return a DataFrame matching the contract
        ...
```

Set `provider: oanda` in `config/settings.yaml` (and extend `VALID_PROVIDERS`
in `config.py` if you want config-time validation for the new name).

## Features

`src/fxsignals/features/` turns candles into analysis-ready frames (pure
pandas/numpy, no TA-Lib). Every output is aligned to the candle index and is
strictly causal: a value at bar *t* uses only bars ≤ *t*; swings are stamped
at their **confirmation** bar; warm-up values stay NaN (never back-filled).

- `indicators.py` — ema, sma, rsi (Wilder), atr (Wilder), macd, adx (+DI/-DI),
  bollinger (mid/upper/lower/width), stochastic (%K/%D); min-bars documented.
- `swings.py` / `structure.py` — pivot detection plus per-bar trend state
  (UP/DOWN/RANGE), last confirmed swing levels, BOS and CHoCH (close breaks).
- `levels.py` — ATR-width S/R zones clustered from confirmed swings, with
  nearest-zone distance in ATR units, visible only after zone formation.
- `sessions.py` — UTC session tags: asia, london, newyork, overlap.
- `patterns.py` — engulfing, pin bar, inside bar as Boolean Series.
- `pipeline.py` — `build_features(df)` joins everything with `ind_ sw_ st_ lv_
  ses_ pat_` prefixes; `build_mtf_features(bias_df, entry_df)` adds closed-bar
  only HTF columns (`htf_`); `FEATURE_GROUPS` / `INDICATOR_GROUPS` mark
  correlated families so downstream scoring counts each group once.

## Signals

`src/fxsignals/signals/` converts features into advisory `Signal` objects on
**closed bars only** (act at next bar open). It never executes orders.

- **Scoring** (`scoring.py`): weighted confluence, 0–100 per direction —
  htf_bias 30, structure 20, location 20, momentum 10 (RSI+stoch counted
  once), trend_strength 10 (EMA+MACD+ADX counted once), trigger 10; partial
  credit and human-readable reasons per group. Weights must sum to 100.
- **Risk** (`risk.py`): SL beyond the protecting swing/zone + ATR buffer,
  capped to [min_sl_atr, max_sl_atr]; TP from rr_target (2.0) or the next
  opposing zone; rejects rr < min_rr (1.5). Pip-aware (JPY 0.01, XAU 0.1).
- **Filters** (`filters.py`, all causal): minimum ADX, allowed sessions
  (default london/newyork/overlap), ATR percentile band, cooldown per
  pair+direction, duplicate-entry suppression.
- **Engine** (`engine.py`): `SignalEngine(cfg).evaluate(pair, bias_df,
  entry_df)` for the latest closed bar; `.evaluate_history(...)` replays every
  bar with identical semantics for backtesting. Emits when score ≥ min_score
  (65) and the winning direction beats the other by direction_margin (15).
  `fxsignals.signals.factory.build_engine_config(settings)` maps the YAML
  `signals:` / `filters:` sections onto an `EngineConfig`.

Running the scanner:

```bash
python -m fxsignals.scanner --once            # single pass over all pairs
python -m fxsignals.scanner --loop            # wake shortly after each H1 close
python -m fxsignals.scanner --loop --interval 3600   # fixed cadence instead
```

Each signal is printed as a readable block and appended to
`output.jsonl_path` (default `./signals.jsonl`) as one JSON object per line
(`Signal.to_dict()`). An optional Telegram hook is **disabled by default**;
enable it with `output.telegram_enabled: true` and provide
`FXSIGNALS_TELEGRAM_TOKEN` / `FXSIGNALS_TELEGRAM_CHAT_ID` environment
variables. Sink failures are logged, never fatal.

## Backtest

`src/fxsignals/backtest/` measures the quality of engine signals **without
executing anything**. It reuses `SignalEngine.evaluate_history`, so live and
backtest logic are identical — no duplicated signal code.

**Conservative simulation rules**

- Entry at the **next bar's open** after the signal bar (never the signal
  bar's close).
- If one bar touches both SL and TP, the **SL is assumed hit first**
  (`sl_first_on_ambiguity: true`, configurable).
- Costs applied at entry and on SL exits; every `r_multiple` in reports is
  **net of costs**.

**Cost model** (defaults, pips per pair)

| Pair class | Spread | Slippage |
|---|---|---|
| Majors (EURUSD, GBPUSD, AUDUSD, USDCAD, USDCHF, NZDUSD) | 1.0 pip | 0.2 pip |
| JPY pairs (USDJPY) | 1.2 pips | 0.2 pip |
| XAUUSD | 25 points-equivalent | 0.2 pip |

**Usage**

```bash
python -m fxsignals.backtest.runner \
    --config config/settings.yaml \
    --pairs EURUSD,GBPUSD \
    --start 2024-01-01 --end 2025-12-31 \
    --split 0.7          # walk-forward: first 70% in-sample, last 30% OOS
```

**Output**: each run writes `reports/backtest_<timestamp>/` containing

- `trades.csv` — every simulated trade (entry/exit, R multiple, pips, reason),
- `metrics.json` — win rate, expectancy, profit factor, max drawdown (R),
  streaks, splits by direction/pair/session/score bucket, factor attribution,
- `summary.md` — plain-text tables,
- `equity_curve.csv` — cumulative R per trade,
- `equity_curve.png` — only if matplotlib is importable (skipped silently).

## Project Status

| Layer | Status |
|-------|--------|
| Data (providers, store, resample) | Built |
| Features (indicators, structure, S/R, sessions, patterns) | Built |
| Signals (scoring, risk, filters, engine, scanner) | Built |
| Backtest (simulator, metrics, report, runner) | Built |
| Real data provider (OANDA/Twelve Data) | Not started |
| Walk-forward optimization | Not started |

## Quick Start

```bash
pip install -r requirements.txt

# single scan pass over all configured pairs (synthetic data by default)
python -m fxsignals.scanner --once

# backtest the engine over available history
python -m fxsignals.backtest.runner --config config/settings.yaml
```

## Data Requirements

- **Default**: the deterministic `synthetic` provider — fine for testing and
  demos, **not** for evaluating real edge.
- **Real data**: switch `provider: csv` in `config/settings.yaml` and place
  files in `data_dir` (default `./data`) named `{PAIR}_{TF}.csv`
  (e.g. `EURUSD_H1.csv`, `EURUSD_H4.csv`).
- **Columns**: `time, open, high, low, close, volume` (timestamps parsed as
  UTC; volume optional).
- **Minimum**: ~2 years of H1 data per pair (plus the H4 bias timeframe) for
  a meaningful backtest.

## Configuration

All runtime behaviour lives in `config/settings.yaml`, loaded and validated by
`src/fxsignals/config.py` (typed dataclasses, clear errors on bad values):

- `pairs`, `timeframes` (bias/entry), `provider`, `paths`, `log_level`
- `signals:` — score gates, weights (must sum to 100), RR bounds, cooldown
- `filters:` — ADX floor, allowed sessions, ATR percentile band, lookback
- `output:` — JSONL path, Telegram toggle (credentials via env vars only)
- `backtest:` — cost model (spread/slippage), SL-first rule, max bars in
  trade, one-trade-per-pair flag, report directory

## Tests

```bash
pytest tests/
```

TA-Lib based indicators are intentionally out of scope for this step.
