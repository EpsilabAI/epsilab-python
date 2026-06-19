# Epsilab Python SDK

Official Python client for the [Epsilab](https://www.epsilab.com) model evaluation and improvement platform.

## What is Epsilab?

Epsilab runs model and harness evaluations on workflow-level tasks, detects recurring capability gaps, and exports evals, trajectories, preference data, SFT examples, and regression tests. Training-data exports anonymize model identities by default using labels such as `target_model` and `reference_A`.

## Installation

```bash
pip install epsilab
```

Or install from source:

```bash
git clone https://github.com/EpsilabAI/epsilab-python.git
cd epsilab-python
pip install -e .
```

## Quick Start

```python
from epsilab import Epsilab

client = Epsilab(api_key="sk-...")

# Compare multiple models in one evaluation (use any OpenRouter model slug)
eval_result = client.create_evaluation(
    ["provider/model-a", "provider/model-b", "provider/model-c"],
    name="Frontier comparison",
    max_tasks=25,
)
print(f"Evaluation started: {eval_result.evaluation_id}")

# Wait for completion
run = client.wait_for_completion(eval_result.runs[0].run_id)
print(f"Completed: {run.task_count} tasks, {run.gap_count} gaps found")

# View capability gaps
for gap in client.get_gaps(run.run_id):
    print(f"  {gap.capability}: alpha={gap.alpha_score:.3f}")

# Export targeted training data (model identities are anonymized by default)
client.export_run(run.run_id, format="dpo", path="output/dpo_pairs.jsonl")
```

## Configuration

| Environment Variable    | Constructor Param  | Description                              |
|-------------------------|--------------------|------------------------------------------|
| `EPSILAB_API_KEY`       | `api_key`          | Your API key                             |
| `EPSILAB_API_BASE`      | `api_base`         | API base URL (default: production)       |
| `EPSILAB_HTTP_TIMEOUT`  | `timeout_seconds`  | Request timeout in seconds (default: 120)|
| —                       | `max_retries`      | Auto-retry count for 429/5xx (default: 3)|
| —                       | `backoff_base`     | Initial retry backoff in seconds (default: 1.0) |
| —                       | `load_dotenv`      | Also read a local `.env` file (default: false) |

The SDK reads process environment variables automatically. To also read a
local `.env` file, opt in explicitly:

```python
client = Epsilab(load_dotenv=True)
```

## Multi-Model Evaluations

Compare multiple models side-by-side on the same task set:

```python
# Simple: just pass model IDs (any OpenRouter-compatible slug)
eval_result = client.create_evaluation(
    ["provider/model-a", "provider/model-b", "provider/model-c"],
    name="Three-way comparison",
)

# Advanced: per-model harness overrides
eval_result = client.create_evaluation(
    [
        {"model_id": "provider/model-a", "harness": "codex"},
        {"model_id": "provider/model-b", "harness": "openhands"},
        "provider/model-c",  # uses default_harness
    ],
    default_harness="codex",
    max_tasks=50,
    domains=["coding", "math"],
)

# Check cost before running
estimate = client.estimate_evaluation_cost(
    ["provider/model-a", "provider/model-b"],
    max_tasks=25,
)
print(f"Cost: {estimate.total_credits} credits (balance: {estimate.balance})")
print(f"Sufficient: {estimate.sufficient}")
for m in estimate.per_model:
    print(f"  {m.model_id}: {m.credits} credits, {m.task_count} tasks")
```

## Bring Your Own Model

Evaluate any OpenAI-compatible endpoint:

```python
run = client.create_run(
    "internal-llm-v3",
    base_url="https://my-company.example.com/v1",
    api_key="sk-model-key",
)
```

Your model credentials are used only during the evaluation and are never stored. Training-data exports anonymize model identities by default using labels such as `target_model` and `reference_A`.

## Client Methods

### Models

| Method | Description |
|--------|-------------|
| `list_models(search, provider, limit)` | Browse available models with live pricing |

### Evaluations

| Method | Description |
|--------|-------------|
| `create_evaluation(models, ...)` | Compare multiple models in one evaluation |
| `estimate_evaluation_cost(models, ...)` | Estimate credit cost before running |
| `suggest_scope(instructions)` | AI-generated scope suggestions from a description |

### Runs

| Method | Description |
|--------|-------------|
| `create_run(model_name, ...)` | Submit a single model for evaluation |
| `get_run(run_id)` | Get run status and summary |
| `list_runs(status, limit, offset)` | List your evaluation runs (single page) |
| `iter_runs(status, page_size)` | Auto-paginating iterator over all runs |
| `wait_for_completion(run_id, ...)` | Block until run completes or fails |
| `cancel_run(run_id)` | Cancel a queued or running evaluation |
| `retry_run(run_id)` | Retry a failed run, reusing completed results |
| `resume_run(run_id, ...)` | Resume a failed run with optional new credentials |
| `delete_run(run_id)` | Delete a run |

### Results & Insights

| Method | Description |
|--------|-------------|
| `get_gaps(run_id)` | Get capability gaps from a completed run |
| `get_artifacts(run_id, ...)` | Get generated artifacts (single page) |
| `iter_artifacts(run_id, ...)` | Auto-paginating iterator over all artifacts |
| `get_insights(run_id)` | Get model rankings, J1/J2/J3 metrics, and analytics |
| `request_review(run_id, gap_ids)` | Request human review for specific gaps |
| `forge(run_id)` | Generate new tasks targeting run gaps |
| `export_run(run_id, format, path)` | Export training data or reports |

### Cross-Run Analytics

| Method | Description |
|--------|-------------|
| `get_leaderboard()` | Cross-run model leaderboard |
| `get_domain_leaderboard()` | Per-domain model scores across runs |
| `get_cost_analysis()` | Cost-efficiency rankings with live pricing |
| `get_precomputed_insights()` | Per-domain best-model recommendations |

### Tasks

| Method | Description |
|--------|-------------|
| `get_task(task_id)` | Get details for a specific task |
| `create_task(task)` | Create a single custom evaluation task |
| `upload_custom_tasks(tasks)` | Batch upload custom evaluation tasks |
| `get_task_upload_limits()` | Get max file size and task count per batch |
| `classify_tasks(tasks)` | Auto-classify tasks by domain and capability |
| `list_tasks(...)` | List available tasks (single page) |
| `iter_tasks(...)` | Auto-paginating iterator over all tasks |
| `delete_task(task_id)` | Delete a custom task |

### API Keys

| Method | Description |
|--------|-------------|
| `list_api_keys()` | List your API keys |
| `create_api_key(label)` | Create a new API key |
| `revoke_api_key(key_id)` | Revoke an API key |

### Billing

| Method | Description |
|--------|-------------|
| `get_credit_balance()` | Get current credit balance |
| `get_credit_ledger(...)` | Get credit transaction history |
| `get_usage(period)` | Get monthly usage summary |

### Voice Evaluations

| Method | Description |
|--------|-------------|
| `register_voice_asset(asset_id, uri, ...)` | Register an audio asset for voice tasks |
| `create_voice_task(task_id, task_type, ...)` | Create a voice evaluation task |
| `create_voice_run(target_model, ...)` | Submit a voice model for evaluation |
| `get_voice_slices(run_id)` | Get per-slice quality metrics |
| `get_voice_timeline(run_id, task_id)` | Get event timeline for replay/debugging |
| `route_voice(prompt, ...)` | Route a voice workload to the best model |

### RL Environments

| Method | Description |
|--------|-------------|
| `create_rl_session(task_id, ...)` | Create an RL environment session and get initial observation |
| `rl_step(session_id, action)` | Take an action and receive observation, reward, done flags |
| `get_rl_trajectory(session_id)` | Get full trajectory for a completed session |
| `verify_rl_trajectory(session_id)` | Replay and verify trajectory integrity |
| `get_rl_curriculum(...)` | Get adaptive curriculum batch biased toward learning frontier |
| `export_rl_sessions(format, ...)` | Export RL sessions as GRPO/DPO/KTO/process supervision data |
| `close_rl_session(session_id)` | Manually close an active session |
| `list_rl_environments(...)` | List available environments (tasks) for RL training |
| `list_rl_sessions(...)` | List your RL sessions with status filters |
| `get_rl_stats(...)` | Get completion rates, reward distribution, difficulty profile |

### Capability Matrix

Cross-run model comparison — aggregate scores, gaps, and training artifacts across all your evaluations.

| Method | Description | Tier |
|--------|-------------|:----:|
| `get_matrix_models(...)` | List all evaluated models with aggregated stats | All |
| `get_matrix_model_gaps(model_id, ...)` | Capability gaps for a specific model | All |
| `get_matrix_model_capabilities(model_id, ...)` | Per-capability breakdown for a model | All |
| `get_matrix_gaps(...)` | Cross-model capability gaps | All |
| `get_matrix_domains(...)` | Per-domain score breakdown | All |
| `get_matrix_artifacts(...)` | Training artifacts from the matrix | All |
| `get_matrix_model_profile(model_id)` | Detailed model profile — coverage, strengths, weaknesses | Enterprise |
| `get_matrix_scores(...)` | Raw score data with pagination | Enterprise |
| `get_matrix_insights(...)` | High-level patterns, rankings, and recommendations | Enterprise |
| `get_matrix_coverage(...)` | Evaluation coverage — tasks per model per domain | Enterprise |

## RL Training Loop

Use RL environments to collect training data through interactive sessions:

```python
from epsilab import Epsilab

client = Epsilab(api_key="sk-...")

# Get adaptive curriculum (tasks at your model's learning frontier)
curriculum = client.get_rl_curriculum(env_type="code_sandbox", batch_size=16)
task_ids = curriculum["curriculum"]["frontier_tasks"] + curriculum["curriculum"]["exploration_tasks"]

for task_id in task_ids:
    # Start a session
    session = client.create_rl_session(
        task_id,
        env_type="code_sandbox",
        reward_mode="partial_credit",
    )
    print(f"Task: {session.observation[:80]}...")

    observation = session.observation
    completed = False
    try:
        for attempt in range(5):
            action = your_model.generate(observation)
            result = client.rl_step(session.session_id, action)
            observation = result.observation

            print(f"  Step {attempt}: reward={result.reward}, done={result.done}")
            if result.done:
                completed = True
                break
    finally:
        if not completed:
            client.close_rl_session(session.session_id, reason="training_interrupted")

# Export collected sessions as GRPO training data
data = client.export_rl_sessions("grpo", env_type="code_sandbox")
print(f"Exported {data['n_records']} GRPO records")
```

### Environment Types

| Type | Description | Multi-step |
|------|-------------|:----------:|
| `single_turn` | Question → answer, scored by the task's configured verifier | No |
| `code_sandbox` | Write code and validate it in an isolated execution environment | Yes |
| `agent_workflow` | Tool-calling agent in a persistent isolated environment | Yes |
| `simulation` | Deterministic state transitions with dense rewards | Yes |

### TRL GRPO Integration

```python
def reward_fn(completions, task_ids, **kwargs):
    """Use Epsilab environment as live reward function."""
    rewards = []
    for completion, task_id in zip(completions, task_ids):
        session = client.create_rl_session(task_id, env_type="code_sandbox")
        completed = False
        try:
            result = client.rl_step(session.session_id, completion)
            completed = result.done
            rewards.append(result.reward or 0.0)
        finally:
            if not completed:
                client.close_rl_session(session.session_id, reason="reward_complete")
    return rewards
```

## Export Formats

| Format | Use Case |
|--------|----------|
| `dpo` | Direct Preference Optimization (chosen/rejected pairs) |
| `quality_dpo` | DPO pairs enriched with quality scores and feedback |
| `sft` | Supervised Fine-Tuning (prompt/completion pairs) |
| `kto` | Kahneman-Tversky Optimization (binary desirability) |
| `grpo` | Group Relative Policy Optimization (grouped completions) |
| `sharegpt` | Multi-turn conversation format |
| `jsonl` | Raw artifacts as NDJSON |
| `report` | Human-readable evaluation report |
| `yaml` | YAML configuration for reproduction |
| `pytest` | Pytest test cases from capability gaps |

Training data exports use anonymized model labels (e.g. `target_model`, `reference_A`) rather than real model identifiers. Chosen/reference answers are verified gold answers, not raw model outputs. Evaluation prompts are included for enterprise accounts; standard accounts receive task ID references.

## Automatic Retries

The SDK automatically retries on rate-limit (429), transient server errors (500, 502, 503, 504), and transient network failures with exponential backoff and jitter. For 429 responses, the `Retry-After` header is respected when valid.

```python
# Default: 3 retries with 1s base backoff
client = Epsilab(api_key="sk-...")

# Customize retry behaviour
client = Epsilab(api_key="sk-...", max_retries=5, backoff_base=2.0)

# Disable retries entirely
client = Epsilab(api_key="sk-...", max_retries=0)
```

## Pagination

List endpoints return a single page by default. Use the `iter_*` methods to auto-paginate:

```python
# Iterate over all runs without manual offset management
for run in client.iter_runs(status="completed"):
    print(run.run_id, run.gap_count)

# Same for artifacts and tasks
for artifact in client.iter_artifacts(run_id):
    print(artifact.artifact_type)

for task in client.iter_tasks(domain="coding"):
    print(task["task_id"])
```

## Error Handling

```python
from epsilab import Epsilab, AuthError, InsufficientCreditsError, RateLimitError, ApiError

client = Epsilab(api_key="sk-...")

try:
    eval_result = client.create_evaluation(["provider/model-a", "provider/model-b"])
except AuthError:
    print("Invalid API key")
except InsufficientCreditsError as e:
    print(f"Not enough credits: {e}")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after}s")
except ApiError as e:
    print(f"API error: {e.status_code}")
```

## Examples

See [`examples/example.py`](examples/example.py) for a complete workflow.

## License

Apache 2.0 — see [LICENSE](LICENSE).
