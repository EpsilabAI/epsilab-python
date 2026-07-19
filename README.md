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

~~~bash
epsilab login
epsilab whoami
epsilab init my-environment
cd my-environment
# edit environment.py and tasks.json
epsilab deploy
~~~

The deploy command creates the namespace and listing when needed, builds the
image, uploads it, registers immutable task and verifier releases, and makes
the environment available for hosted sessions. No UUIDs or registry
credentials are required.

### Run an environment

~~~bash
epsilab run epsilab/bug-hunter
epsilab run epsilab/bug-hunter \
  --task bug-hunter-easy-train-001 \
  --action 'BUG_LINE: 6
FIX:
return total / len(numbers)'
~~~

The first form starts an interactive session. The second performs one action
and prints its observation, reward, terminal state, and verification result.

~~~python
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
session = client.wait_for_session(session)
result = client.environment_step(
    session.session_id,
    {
        "action_type": "submit",
        "content": "BUG_LINE: 6\nFIX:\nreturn total / len(numbers)",
    },
    session_token=session.session_token,
)
print(f"Reward: {result.reward}, Done: {result.done}")

# Export training data
export = client.create_environment_export(
    deployment_id=listings[0].deployment_id,
    format="grpo",
)
~~~

### Run a long-horizon coding agent

`run_agent_episode` separates model turns from environment steps. A model may
reason for many turns before calling a tool; every request, reasoning chunk,
assistant message, tool boundary, result, usage record, cancellation, and
error is persisted immediately in the session trace.

```python
from epsilab import AgentToolCall, AgentTurn, AgentUsage, Epsilab

client = Epsilab()

def call_your_model(context):
    # Invoke any provider here. Translate its response into AgentTurn.
    # Do not set a provider token limit if you want an uncapped generation.
    response = your_provider_call(context.history, context.observation)
    return AgentTurn(
        reasoning=response.reasoning,
        message=response.message,
        tool_calls=[
            AgentToolCall(call_id=call.id, name=call.name, arguments=call.arguments)
            for call in response.tool_calls
        ],
        usage=AgentUsage(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
        ),
        provider=response.provider,
        model=response.model,
        provider_request_id=response.request_id,
    )

rollout = client.run_agent_episode(
    deployment_id="deployment-id",
    task_id="task-id",
    model_fn=call_your_model,
    max_turns=500,
    cancel_check=my_cancel_event.is_set,
)
print(rollout.stop_reason, rollout.turns_completed, rollout.session.total_reward)
```

The SDK has no token budget or per-generation cap. `max_turns` is the only
runner limit (maximum 500 model calls); the environment's own immutable task
horizon still governs tool actions. See `examples/run_long_horizon_agent.py`
for a provider-neutral adapter skeleton.

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

### Batch evaluation with your policy

Batch creation provisions reproducible sessions; your code still supplies the
agent actions. `run_batch` handles both parts so sessions cannot be left idle:

```python
batch = client.run_batch(
    deployment_id="deployment-id",
    name="regression sweep",
    task_seed_pairs=[
        {"task_id": "task-001", "seed": 42},
        {"task_id": "task-002", "seed": 42},
    ],
    policy_fn=lambda observation, info: my_agent(observation),
)

for session in batch["sessions"]:
    print(session["task_id"], session["total_reward"])
```

For a batch created in the dashboard or CLI, call
`client.drive_batch(batch_id, policy_fn=my_policy)`.

## Creating an Environment

An RL environment is a containerized task server. At minimum it needs:

| File | Purpose |
|------|---------|
| `environment.py` | Typed, deterministic OpenEnv environment logic |
| `verifier.py` | Independent trajectory replay and reward verification |
| `Dockerfile` | Builds the runtime container image |
| `server.py` | Generated OpenEnv HTTP entry point |
| `tasks.json` | JSON array defining the task set |
| `.epsilab/project.json` | Local project linkage, populated on first deploy |

### Environment protocol

The generated server uses the OpenEnv contract and exposes:

~~~
GET  /health
POST /reset
POST /step
WS   /ws
~~~

`/reset` accepts a task ID and deterministic seed. `/step` accepts a typed
action and returns an observation, optional reward, and terminal state. Most
authors only edit `environment.py`; the generated server and verifier already
implement the transport and replay contracts.

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
| `epsilab init [name]` | Scaffold a working environment project |
| `epsilab deploy` | Build, upload, register, and host an environment or tool |
| `epsilab run <owner>/<name>` | Run an interactive or one-step hosted session |
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
| [`examples/batch_evaluation.py`](examples/batch_evaluation.py) | Benchmark a policy across provisioned environment batches |
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
