# Epsilab Python SDK

Python SDK and CLI for the [Epsilab](https://www.epsilab.com) RL Environment Hub.

## What is Epsilab?

Epsilab is an open hub for RL environments. Search, run, and export training data from hosted environments, or publish your own with a single command.

- **Researchers and teams training models** — search and run environments, export GRPO/DPO/SFT/KTO training data, batch evaluation
- **Environment builders** — publish with `epsilab deploy`, immutable content-addressed releases, usage analytics
- **Open by default** — public, shared, unlisted, or private visibility; auto-qualified releases

## Installation

```bash
pip install epsilab
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

client = Epsilab(api_key="sk-...")

# Find environments
envs = client.search_environments(domain="coding", min_quality_score=0.8)

# Create a session and interact
session = client.create_environment_session("deployment-id", task_id="task-001", seed=42)
result = client.environment_step(
    session.session_id,
    "def fibonacci(n): ...",
    session_token=session.session_token,
)
print(f"Reward: {result.reward}, Done: {result.done}")

# Export training data
export = client.create_environment_export(deployment_id="deployment-id", format="grpo")
```

### TRL GRPO integration

```python
def reward_fn(completions, task_ids, **kwargs):
    rewards = []
    for completion, task_id in zip(completions, task_ids):
        session = client.create_environment_session("deployment-id", task_id=task_id)
        result = client.environment_step(
            session.session_id,
            completion,
            session_token=session.session_token,
        )
        rewards.append(result.reward or 0.0)
    return rewards
```

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

| Environment Variable | Constructor Param | Description |
|---|---|---|
| `EPSILAB_API_KEY` | `api_key` | Your API key |
| `EPSILAB_API_BASE` | `api_base` | API base URL (default: production) |
| `EPSILAB_HTTP_TIMEOUT` | `timeout_seconds` | Request timeout in seconds (default: 120) |
| | `max_retries` | Auto-retry count for 429/5xx (default: 3) |
| | `load_dotenv` | Also read a local `.env` file (default: false) |

## Error Handling

```python
from epsilab import Epsilab, AuthError, InsufficientCreditsError, RateLimitError, ApiError

client = Epsilab(api_key="sk-...")

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

| Script | Description |
|--------|-------------|
| [`examples/run_environment.py`](examples/run_environment.py) | Browse, run sessions, inspect trajectories, export data |
| [`examples/grpo_training.py`](examples/grpo_training.py) | Use environments as live reward functions for TRL GRPO |
| [`examples/batch_evaluation.py`](examples/batch_evaluation.py) | Batch evaluation across tasks with server-side parallelism |
| [`examples/marketplace_example.py`](examples/marketplace_example.py) | Creator and buyer marketplace workflows |

## Documentation

| Document | Description |
|----------|-------------|
| [API Reference](docs/api-reference.md) | Full method reference for all SDK features |
| [Evaluations & More](docs/evaluations.md) | Model evaluations, voice, routing, capability matrix |

## License

Apache 2.0 — see [LICENSE](LICENSE).
