# Epsilab Python SDK

Python SDK and CLI for the [Epsilab](https://www.epsilab.com) RL Environment Hub.

## What is Epsilab?

Epsilab is an open hub for RL environments. Search, run, and export training data from hosted environments, or publish your own with a single command.

- **Researchers and teams training models** — search and run environments, export GRPO/DPO/SFT/KTO training data, batch evaluation
- **Environment builders** — publish with `epsilab deploy`, immutable content-addressed releases, usage analytics
- **Open by default** — public, shared, unlisted, or private visibility; auto-qualified releases

## Installation

```bash
pip install epsilab                  # SDK + CLI only
pip install epsilab[training]        # + TRL, Together AI, torch
pip install epsilab[tinker]          # + Tinker (custom training loops)
pip install epsilab[fireworks]       # + Fireworks AI
pip install epsilab[all]             # everything
```

## Quick Start

### Deploy an environment

```bash
epsilab login
epsilab env init my-environment
cd my-environment
# implement your logic in server.py, add tasks to tasks.json
epsilab deploy
```

One command builds, uploads, and registers. No registry credentials needed.

### Run an environment

```python
from epsilab import Epsilab

# Credentials are loaded automatically from `epsilab login`
client = Epsilab()

# Browse the hub
listings = client.list_environment_listings(limit=20)
for l in listings:
    print(f"{l.slug:30s}  {l.title}")

# Create a session and interact
session = client.create_environment_session(
    listings[0].deployment_id,
    task_id="bug-hunter-easy-train-001",
    seed=42,
)
result = client.environment_step(
    session.session_id,
    "The bug is a missing null check in the handler.",
    session_token=session.session_token,
)
print(f"Reward: {result.reward}, Done: {result.done}")

# Export training data
export = client.create_environment_export(
    deployment_id=listings[0].deployment_id,
    format="grpo",
)
```

### Post-training with multiple environments

For generalist RL post-training, train across diverse environments
simultaneously — coding, ops, business, etc.:

```bash
# Collect data from 3 envs, format as DPO, train locally
python examples/run_environment.py \
    --envs bug-hunter,refactor,test-writer \
    --algorithm dpo \
    --provider local

# Online GRPO across all available envs
python examples/grpo_training.py --envs all --provider tinker --steps 100
```

The example scripts support any training provider:

| Provider | Install | Description |
|----------|---------|-------------|
| `together` | `pip install epsilab[training]` | Managed fine-tuning via Together AI |
| `fireworks` | `pip install epsilab[fireworks]` | Managed fine-tuning via Fireworks AI |
| `tinker` | `pip install epsilab[tinker]` | Custom training loops on remote GPUs |
| `local` | `pip install epsilab[training]` | TRL on local GPU (SFT/DPO/KTO/GRPO) |

## Creating an Environment

An RL environment is a containerized task server. At minimum it needs:

| File | Purpose |
|------|---------|
| `Dockerfile` | Builds the runtime container image |
| `server.py` | HTTP server implementing the environment protocol |
| `tasks.json` | JSON array defining the task set |

### Environment protocol

Your server must expose two HTTP endpoints:

```
POST /reset   -> {"observation": str}
POST /step    -> {"observation": str, "reward": float, "terminated": bool, "truncated": bool}
```

`/reset` receives `{"task_id": "...", "seed": 42}` and returns the initial observation. `/step` receives `{"action": "..."}` and returns the step result.

### tasks.json format

```json
[
  {
    "task_id": "find-the-bug-001",
    "prompt": "Find and fix the bug in the following code...",
    "difficulty": "easy",
    "split": "train",
    "max_steps": 10
  }
]
```

## Creating an Application Tool

Application tools are reusable plugins (e.g. GitHub, Slack, Calendar) that environments can compose together. A tool needs:

| File | Purpose |
|------|---------|
| `plugin.py` | `AppPlugin` subclass defining the tool's identity and lifecycle |
| `api.py` | API route handlers (FastAPI-style) |
| `state.py` | Deterministic state model for the tool |

```bash
cd my-tool/
epsilab deploy    # auto-detects tool structure
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `epsilab login` | Authenticate with your API key |
| `epsilab logout` | Remove stored credentials |
| `epsilab whoami` | Show current auth and profile status |
| `epsilab deploy` | Build, upload, and register an environment or tool |
| `epsilab env init [slug]` | Scaffold a new environment project |
| `epsilab env list` | List your environment listings |
| `epsilab env search [query]` | Search the hub |
| `epsilab env verify` | Run local preflight checks |
| `epsilab rl sessions` | List your RL sessions |
| `epsilab rl trajectory <id>` | View step-by-step trajectory |
| `epsilab namespace create <slug>` | Create a namespace |

All commands support `--json` for machine-readable output and `-v` for verbose logging.

## Configuration

The SDK resolves credentials in order:
1. Explicit `api_key=` constructor argument
2. `EPSILAB_API_KEY` environment variable
3. `~/.epsilab/credentials.json` (set by `epsilab login`)

Most users just need `epsilab login` — no env vars or code changes required.

| Environment Variable | Constructor Param | Description |
|---|---|---|
| `EPSILAB_API_KEY` | `api_key` | Your API key (overrides stored credentials) |
| `EPSILAB_API_BASE` | `api_base` | API base URL (default: production) |
| `EPSILAB_HTTP_TIMEOUT` | `timeout_seconds` | Request timeout in seconds (default: 120) |
| | `max_retries` | Auto-retry count for 429/5xx (default: 3) |
| | `load_dotenv` | Also read a local `.env` file (default: false) |

## Error Handling

```python
from epsilab import Epsilab, AuthError, InsufficientCreditsError, RateLimitError, ApiError

client = Epsilab()

try:
    session = client.create_environment_session("dep-id", task_id="task-001")
except AuthError:
    print("Invalid API key")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after}s")
except ApiError as e:
    print(f"API error: {e.status_code}")
```

The SDK retries automatically on rate limits (429) and transient server errors (500, 502, 503, 504) with exponential backoff and jitter.

## Examples

| Script | What it does |
|--------|-------------|
| [`examples/example.py`](examples/example.py) | **Start here** — discover envs, run one session, see rewards |
| [`examples/run_environment.py`](examples/run_environment.py) | Collect data across multiple envs, train with SFT/DPO/KTO |
| [`examples/grpo_training.py`](examples/grpo_training.py) | Online GRPO with live environment rewards |
| [`examples/batch_evaluation.py`](examples/batch_evaluation.py) | Benchmark a model across envs with server-side batches |
| [`examples/marketplace_example.py`](examples/marketplace_example.py) | Consumer and publisher hub workflows |

All examples use `argparse` — run with `--help` for options. Key flags:

```bash
python examples/run_environment.py \
    --envs bug-hunter,refactor,test-writer \
    --algorithm dpo \
    --provider local \
    --sessions-per-env 20
```

## Documentation

| Document | Description |
|----------|-------------|
| [API Reference](docs/api-reference.md) | Full method reference for all SDK features |
| [Evaluations (deprecated)](docs/evaluations.md) | Legacy evaluations, voice, routing — will be removed 2026-12-31 |

## License

Apache 2.0 — see [LICENSE](LICENSE).
