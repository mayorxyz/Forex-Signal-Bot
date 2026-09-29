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

## Tests

```bash
pytest tests/
```

TA-Lib based indicators are intentionally out of scope for this step.
