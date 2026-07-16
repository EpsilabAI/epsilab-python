# Model Evaluations, Voice, Routing, and Capability Matrix (Deprecated)

> **Deprecated**: These features are deprecated as of v0.15.0. The platform
> is focused on the **RL Environment Hub** for post-training workflows.
> These endpoints will remain functional until 2026-12-31 but will not
> receive new features. See [`examples/run_environment.py`](../examples/run_environment.py)
> and [`examples/grpo_training.py`](../examples/grpo_training.py) for
> the recommended approach.

## Model Evaluations

```python
# Compare multiple models side-by-side
eval_result = client.create_evaluation(
    ["provider/model-a", "provider/model-b", "provider/model-c"],
    name="Frontier comparison",
    max_tasks=25,
)

run = client.wait_for_completion(eval_result.runs[0].run_id)
for gap in client.get_gaps(run.run_id):
    print(f"  {gap.capability}: alpha={gap.alpha_score:.3f}")

client.export_run(run.run_id, format="dpo", path="output/dpo_pairs.jsonl")
```

| Method | Description |
|--------|-------------|
| `create_evaluation(models, ...)` | Compare multiple models in one evaluation |
| `estimate_evaluation_cost(models, ...)` | Estimate credit cost before running |
| `create_run(model_name, ...)` | Submit a single model for evaluation |
| `get_run(run_id)` | Get run status and summary |
| `list_runs(status, limit, offset)` | List your evaluation runs |
| `iter_runs(status, page_size)` | Auto-paginating iterator over all runs |
| `wait_for_completion(run_id, ...)` | Block until run completes or fails |
| `cancel_run(run_id)` | Cancel a queued or running evaluation |
| `retry_run(run_id)` / `resume_run(run_id, ...)` | Retry or resume a failed run |
| `delete_run(run_id)` | Delete a run |
| `get_gaps(run_id)` | Get capability gaps from a completed run |
| `get_artifacts(run_id, ...)` / `iter_artifacts(run_id, ...)` | Get generated training artifacts |
| `get_insights(run_id)` | Get model rankings and analytics |
| `export_run(run_id, format, path)` | Export training data or reports |
| `forge(run_id)` | Generate new tasks targeting gaps |

## First-Party RL Environments

Built-in task-based RL environments, separate from marketplace environments:

| Method | Description |
|--------|-------------|
| `create_rl_session(task_id, ...)` | Create an RL session and get initial observation |
| `rl_step(session_id, action)` | Take an action, receive observation and reward |
| `get_rl_trajectory(session_id)` | Get full trajectory for a completed session |
| `verify_rl_trajectory(session_id)` | Replay and verify trajectory integrity |
| `get_rl_curriculum(...)` | Get adaptive curriculum at your model's learning frontier |
| `export_rl_sessions(format, ...)` | Export sessions as GRPO/DPO/KTO training data |
| `close_rl_session(session_id)` | Close an active session |
| `list_rl_environments(...)` | List available environments |
| `list_rl_sessions(...)` | List your sessions |
| `get_rl_stats(...)` | Get completion rates and reward distribution |

## Voice Evaluations

| Method | Description |
|--------|-------------|
| `register_voice_asset(asset_id, uri, ...)` | Register an audio asset |
| `create_voice_task(task_id, task_type, ...)` | Create a voice evaluation task |
| `create_voice_run(target_model, ...)` | Submit a voice model for evaluation |
| `get_voice_slices(run_id)` | Get per-slice quality metrics |
| `get_voice_timeline(run_id, task_id)` | Get event timeline |
| `route_voice(prompt, ...)` | Route a voice workload to the best model |

## Capability Matrix

| Method | Description |
|--------|-------------|
| `get_matrix_models(...)` | List all evaluated models with aggregated stats |
| `get_matrix_model_gaps(model_id, ...)` | Capability gaps for a specific model |
| `get_matrix_model_capabilities(model_id, ...)` | Per-capability breakdown |
| `get_matrix_gaps(...)` | Cross-model capability gaps |
| `get_matrix_domains(...)` | Per-domain score breakdown |
| `get_matrix_artifacts(...)` | Training artifacts from the matrix |
| `get_matrix_model_profile(model_id)` | Detailed model profile |
| `get_matrix_scores(...)` | Raw score data with pagination |
| `get_matrix_insights(...)` | Patterns and recommendations |
| `get_matrix_coverage(...)` | Evaluation coverage per model per domain |

## Cross-Run Analytics

| Method | Description |
|--------|-------------|
| `get_leaderboard()` | Cross-run model leaderboard |
| `get_domain_leaderboard()` | Per-domain model scores |
| `get_cost_analysis()` | Cost-efficiency rankings |
| `get_precomputed_insights()` | Per-domain best-model recommendations |

## Tasks and API Keys

| Method | Description |
|--------|-------------|
| `get_task(task_id)` / `create_task(task)` | Get or create evaluation tasks |
| `upload_custom_tasks(tasks)` | Batch upload custom tasks |
| `classify_tasks(tasks)` | Auto-classify by domain and capability |
| `list_tasks(...)` / `iter_tasks(...)` | List or iterate tasks |
| `list_api_keys()` / `create_api_key(label)` / `revoke_api_key(key_id)` | Manage API keys |
| `get_usage(period)` | Monthly usage summary |
| `get_credit_balance()` / `get_credit_ledger(...)` | Credit balance and history |
