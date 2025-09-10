# Epsilab Python SDK

Official Python SDK for accessing Epsilab live portfolio data (signals, weights, trades, equity) using an API key.

## Install

```bash
pip install epsilab
```

If you want HTTP/2 for faster connections, install with:
```bash
pip install "httpx[http2]"
```

### Install from source

```bash
cd /home/matthew/Epsilab/Epsilab-API
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install .
# or for development
pip install -e .
```

## Environment

Set these variables or pass explicitly to the client:

- `EPSILAB_API_BASE` (default `https://www.epsilab.ai/api/ext/v1`)
- `EPSILAB_API_KEY` (a user API key created in Account Settings)

Note:
- API access requires an active Pro subscription. Requests from accounts without Pro will be rejected.
- Live endpoints are rate limited (e.g., ~1 request/second sustained with small bursts). Excessive requests will receive HTTP 429.
- Use `nextEtaMin` from `get_live_status()` to poll intelligently. For example, if `nextEtaMin=80.07`, you can sleep ~80 minutes before the next refresh instead of tight-loop polling. The results will be the same.

## Python Usage

```python
from epsilab import Epsilab

client = Epsilab(api_base="https://www.epsilab.ai/api/ext/v1", api_key="<epsk_live_...>")

# Note: Running the portfolio via API key is disabled. Use the website to initiate runs.

# Fetch live data (typed models)
latest = client.get_live_latest(return_results_if_fresh=True)          # epsilab.models.LiveLatest
status = client.get_live_status()                                      # epsilab.models.LiveStatus
signals = client.get_portfolio_signals(limit=500)                      # List[epsilab.models.PortfolioSignal]
weights = client.get_portfolio_weights(limit=500)                      # List[epsilab.models.PortfolioWeight]
trades = client.get_live_trades(status="PENDING,EXECUTED", include_positions=True)  # List[epsilab.models.PortfolioTrade]
equity = client.get_live_equity(limit=200)                             # List[epsilab.models.EquityPoint]

print(latest)
```

## CLI Usage

```bash
# Set API key in env
export EPSILAB_API_KEY="<epsk_live_...>"
```

## Data models

- `get_live_latest()` → `LiveLatest` with fields:
  - `run_id: Optional[str]`
  - `timeframe: Optional[str]`
  - `signals: List[PortfolioSignal]`
  - `weights: List[PortfolioWeight]`
  - `fresh: Optional[bool]`
  - `recomputed: Optional[bool]`

- `get_live_status()` → `LiveStatus` with fields:
  - `latest_run_id: Optional[str]`
  - `timeframe: Optional[str]`
  - `next_eta_minutes: Optional[float]`
  - `counts: Dict[str, int]`

- `get_portfolio_signals()` → `List[PortfolioSignal]` with fields:
  - `symbol: str`, `signal_type: str`, `strength: Optional[float]`, `timestamp: Optional[str]`, `date: Optional[str]`

- `get_portfolio_weights()` → `List[PortfolioWeight]` with fields:
  - `symbol: str`, `final_weight: Optional[float]`, `raw_weight: Optional[float]`, `timestamp: Optional[str]`, `date: Optional[str]`

- `get_live_trades()` → `List[PortfolioTrade]` with fields:
  - `id: str`, `symbol: str`, `side: str`, `qty: Optional[float]`, `planned_price: Optional[float]`, `planned_at: Optional[str]`, `status: str`, `reasoning: Optional[str]`

- `get_live_members()` → `List[PortfolioMember]` with fields:
  - `strategy_id: str`, `effective_weight: Optional[float]`, `name: Optional[str]`, `visibility: Optional[str]`

- `get_live_equity()` → `List[EquityPoint]` with fields:
  - `date: str`, `value: float`

All models expose `to_dict()`, `to_json()`, and `log()` helpers for convenient logging.
