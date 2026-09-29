# How to Use fxsignals

fxsignals is a **signal-detection bot for forex** — it analyses candles, scores
confluence and emits trade ideas (pair, direction, entry, stop, target, R:R,
confidence, reasons). It **never executes orders**.

Current state of the codebase:

| Layer | Status |
|-------|--------|
| Data (providers, store, resample, validation) | Built |
| Features (indicators, swings, structure, S/R zones, sessions, patterns) | Built |
| Signals (scoring, risk, filters, engine, scanner, emitters) | Built |
| Backtest (simulator, metrics, report, runner) | **Not built yet** |
| Real data provider (OANDA / MT5 / Twelve Data) | Not started |

Commands below that reference `fxsignals.backtest.*` are forward-looking and
will fail until that layer exists.

## Setup

1. Get the repository and enter it.
2. Python 3.11+ recommended. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Make the package importable (it uses a `src/` layout). Either install in
   editable mode or set `PYTHONPATH`:

   ```bash
   pip install -e .            # if/when a pyproject.toml exists
   # or, no-install route (used by all examples below):
   export PYTHONPATH=src       # Windows: set PYTHONPATH=src
   ```

4. Verify the install:

   ```bash
   python -c "import fxsignals; print('OK')"
   ```

5. Run the test suite (optional sanity check):

   ```bash
   pytest tests/
   ```

## Configuration

Everything lives in `config/settings.yaml`. Pass a different file with
`--config path/to.yaml` on any CLI entry point. Key sections (real key names):

- **`pairs`** — symbols to scan: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD,
  USDCHF, NZDUSD, XAUUSD by default.
- **`timeframes`** — `bias_tf: H4` (directional bias), `entry_tf: H1` (signals).
  Bias TF must be coarser than the entry TF.
- **`provider`** — `synthetic` (default, deterministic demo data) or `csv`.
- **`paths`** — `data_dir: ./data` (CSV input), `cache_dir: ./.cache`
  (Parquet/CSV candle cache).
- **`log_level`** — `DEBUG | INFO | WARNING | ERROR | CRITICAL`.
- **`signals`** — engine gates and confluence weights:
  - `min_score: 65` — emit only when the direction score ≥ this (0–100)
  - `direction_margin: 15` — winning direction must beat the other by this much
  - `rr_target: 2.0`, `min_rr: 1.5` — take-profit target / rejection threshold
  - `min_sl_atr: 0.5`, `max_sl_atr: 3.0`, `sl_atr_buffer: 0.25` — stop placement
  - `cooldown_bars: 12` — no repeat signal per pair+direction within N bars
  - `weights:` — htf_bias 30, structure 20, location 20, momentum 10,
    trend_strength 10, trigger 10 (**must sum to 100**; validated at load time;
    correlated indicators such as RSI+stochastic count once as a group)
- **`filters`** — pre-signal gates:
  - `min_adx: 18.0` — skip dead ranges
  - `sessions: [london, newyork, overlap]` — UTC session tags allowed
  - `atr_percentile_low/high: 10/95` with `lookback: 180` — volatility band
- **`output`** —
  - `jsonl_path: ./signals.jsonl` — append-only signal log
  - `telegram_enabled: false` — Telegram hook; credentials come from
    environment variables only, never YAML

Bad keys/values raise a clear `ConfigError` at startup rather than failing
silently.

## Running the Scanner (live signal detection)

### 1. Prepare data

- **Synthetic (default):** nothing to do — a seeded generator produces
  realistic OHLC so the whole pipeline runs offline. Deterministic: same seed,
  same candles. Good for demos and tests only, never for evaluating edge.
- **Real data:** set `provider: csv` and place files in `data_dir` named
  `{PAIR}_{TF}.csv`, e.g. `data/EURUSD_H1.csv` and `data/EURUSD_H4.csv`.
  Columns: `time,open,high,low,close,volume` (volume may be missing → filled
  with 0). Timestamps are parsed as UTC. Aim for ≥ 2 years of H1 and ≥ 4 years
  of H4 per pair.

Run `fxsignals.data.validate.validate_candles()` on your CSVs first — it
reports duplicates, unsorted rows, NaNs, OHLC-logic violations and gaps.

### 2. Scan once (all configured pairs, latest closed bar)

```bash
python -m fxsignals.scanner --once
```

Useful flags: `--config config/settings.yaml`, `--loop`, `--interval SECONDS`.

### 3. Run continuously (fires shortly after each entry-TF bar close)

```bash
python -m fxsignals.scanner --loop                 # waits out each H1 bar
python -m fxsignals.scanner --loop --interval 3600 # fixed cadence override
```

Stop with Ctrl-C (exits cleanly). Per-pair errors are logged and skipped —
one bad symbol never kills the scan.

### 4. Output

- **Console:** one summary line per pair plus a readable block per signal
  (direction, entry, SL, TP, R:R, confidence, reasons).
- **JSONL:** every signal appended to `output.jsonl_path` as one JSON object
  per line (`Signal.to_dict()`) — easy to pipe into pandas:
  `pd.read_json("signals.jsonl", lines=True)`.
- **Telegram (optional):** disabled by default. Export
  `FXSIGNALS_TELEGRAM_BOT_TOKEN` and `FXSIGNALS_TELEGRAM_CHAT_ID` and set
  `output.telegram_enabled: true`. Emitter failures are logged, never fatal.

A typical signal record contains: `pair`, `direction` (LONG/SHORT), `entry`,
`stop_loss`, `take_profit`, `rr`, `confidence` (0–100 = the confluence score),
`timeframe`, `reasons` (human-readable list of contributing factors),
`timestamp`. Signals are computed on **closed bars only** and are meant to be
acted on at the next bar's open.

## Running Backtests — not available yet

The backtest package (`fxsignals.backtest`) is **not implemented** in this
codebase. When it lands, the intended workflow will be:

```bash
python -m fxsignals.backtest.runner --config config/settings.yaml \
    --pairs EURUSD,GBPUSD --start 2024-01-01 --end 2025-12-31 --split 0.7
```

producing `reports/backtest_<UTC-timestamp>/` with `trades.csv`,
`metrics.json`, `summary.md`, `equity_curve.csv` (+ optional PNG). Rules will
be conservative: entry at next bar's open, SL assumed first when a bar touches
both levels, spread + slippage costs applied. Until then, judge signal quality
by logging `signals.jsonl` over time and reviewing the `reasons` field.

## Interpreting Signals & Troubleshooting

- **No signals generated:** lower `signals.min_score` (try 55) or
  `direction_margin`; check you have enough history (features need warm-up —
  ADX alone needs ~2× period bars; supply several hundred candles); verify the
  bar's session passes `filters.sessions` and ADX ≥ `min_adx`.
- **Provider/config errors at startup:** the message names the offending key;
  fix the YAML (pair names like `EURUSD`, not `EUR/USD`).
- **CSV not loading:** confirm filename pattern `{PAIR}_{TF}.csv` in `data_dir`
  and the required columns; run `validate_candles` to see exact defects.
- **Telegram silent:** env vars unset or `telegram_enabled: false`; check logs.
- **Slow scans:** trim `pairs`, reuse the Parquet cache (`cache_dir`), or drop
  `log_level` to WARNING.

## Extending

- **New data source:** subclass `fxsignals.data.base.DataProvider`, implement
  `get_candles(pair, timeframe, start, end, limit)` returning the candle
  contract (UTC index; open/high/low/close/volume; sorted; no duplicates), and
  decorate with `@register_provider("name")`. Nothing else changes.
- **Tuning:** edit `signals.weights` (must keep summing to 100). Each factor is
  a group, so weight changes rebalance whole families, not single indicators.

## Next Steps (roadmap)

- Add a real-data provider (OANDA / Twelve Data / MT5) for live candles.
- Build the backtester + signal-quality report (simulator, metrics, walk-forward split).
- Walk-forward optimisation of scoring weights.
- Execution adapter as a separate, opt-in layer (deliberately excluded here).
