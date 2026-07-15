# Epsilab Python SDK

Python SDK and CLI for the [Epsilab](https://www.epsilab.com) RL Environment Hub & Marketplace.

## What is Epsilab?

Epsilab is a marketplace for verified RL environments. Search, run, and export training data from hosted environments, or publish your own and earn from usage.

Every environment is content-addressed, cryptographically verified, and runs in isolated sandboxes. The platform handles hosting, session management, billing, quality assurance, and training-data export.

**Researchers and teams training models:**
- Search and run hosted RL environments through a single API
- Export training data in GRPO, DPO, SFT, and KTO formats from any environment
- Quality badges, contamination checks, and benchmark results on every release
- Batch evaluation and side-by-side comparison across environments

**Environment builders and engineers:**
- Publish environments with `epsilab env push`
- Content-addressed releases with an automatic qualification pipeline
- Usage analytics, royalty settlement, and access management
- Protocol adapters for Gymnasium, PettingZoo, and custom protocols

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

## CLI

After installing, the `epsilab` command is available in your terminal:

```bash
# Authenticate
epsilab login

# Scaffold a new environment project
epsilab env init my-environment

# Create a namespace and listing
epsilab namespace create my-org
epsilab env create my-env "My Environment" --namespace-id <ns-id>

# Push a release from a manifest
epsilab env push --manifest epsilab.json --listing-id <listing-id>

# Deploy
epsilab env deploy --release-id <rel-id> --listing-id <lst-id>

# Check status and quality
epsilab env status <release-id>

# Grant access to another tenant
epsilab env grant <listing-id> <tenant-id>

# Search the marketplace
epsilab env search "coding environments" --domain coding --min-quality 0.8
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `epsilab login` | Authenticate with your API key |
| `epsilab logout` | Remove stored credentials |
| `epsilab whoami` | Show current auth and profile status |
| `epsilab env init [slug]` | Scaffold a new environment project |
| `epsilab env list` | List your environment listings |
| `epsilab env search [query]` | Search the marketplace |
| `epsilab env create <slug> <title>` | Create a listing |
| `epsilab env push` | Register a new release |
| `epsilab env deploy` | Deploy a release |
| `epsilab env grant <listing> <tenant>` | Grant access to a tenant |
| `epsilab env status <release-id>` | Show release status and quality badges |
| `epsilab env qualify <release-id>` | Start a quality report |
| `epsilab namespace create <slug>` | Create a namespace |
| `epsilab profile show` | Show your creator profile |
| `epsilab profile create <name>` | Create your creator profile |

All commands support `--json` for machine-readable output where applicable. Credentials are stored in `~/.epsilab/credentials.json` (mode 600).

## Quick Start: Running an Environment

```python
from epsilab import Epsilab

client = Epsilab(api_key="sk-...")

# Find high-quality coding environments
envs = client.search_environments(domain="coding", min_quality_score=0.8)
print(f"Found {len(envs)} coding environments")

# Create a session on a deployment
session = client.create_environment_session(
    "deployment-id",
    task_id="task-001",
    seed=42,
)
print(f"Observation: {session.observation}")

# Step through the environment
result = client.environment_step(
    session.session_id,
    "def fibonacci(n): ...",
    session_token=session.session_token,
)
print(f"Reward: {result.reward}, Done: {result.done}")

# Run a full episode with your policy
final = client.run_environment_episode(
    "deployment-id",
    task_id="task-001",
    policy_fn=lambda obs, info: your_model.generate(obs),
    seed=42,
)
print(f"Final reward: {final.reward}")

# Export training data from your sessions
export = client.create_environment_export(deployment_id="deployment-id", format="grpo")
```

## Quick Start: Publishing an Environment

```python
from epsilab import Epsilab

client = Epsilab(api_key="sk-...")

# Create a namespace and listing
ns = client.create_namespace(slug="my-org", display_name="My Org")
listing = client.create_listing(
    namespace_id=ns["namespace_id"],
    slug="code-sandbox-v1",
    title="Code Sandbox v1",
    summary="Sandboxed Python code execution environment",
)

# Register releases (task pack, verifier, environment)
tp = client.create_task_pack_release(
    namespace_id=ns["namespace_id"],
    name="python-tasks",
    release_version="1.0.0",
    artifact_ref="ghcr.io/my-org/tasks:1.0.0",
    artifact_digest="sha256:...",
    usage_policy="open",
    license_id="apache-2.0",
)

ver = client.create_verifier_release(
    namespace_id=ns["namespace_id"],
    name="pytest-verifier",
    release_version="1.0.0",
    runtime_ref="ghcr.io/my-org/verifier:1.0.0",
    runtime_digest="sha256:...",
    source_digest="sha256:...",
    evidence_schema_digest="sha256:...",
    reward_mode="partial_credit",
)

release = client.create_environment_release(
    listing_id=listing.listing_id,
    release_version="1.0.0",
    protocol_version="0.4.1",
    runtime_ref="ghcr.io/my-org/env:1.0.0",
    runtime_digest="sha256:...",
    task_pack_release_id=tp["release_id"],
    verifier_release_id=ver["release_id"],
    action_schema_digest="sha256:...",
    observation_schema_digest="sha256:...",
)

# Deploy and grant access
dep = client.create_deployment(
    listing_id=listing.listing_id,
    alias="prod",
    environment_release_id=release.release_id,
)

client.grant_entitlement(
    grantee_tenant_id="buyer-tenant-id",
    listing_id=listing.listing_id,
    license_id="apache-2.0",
)

# Track usage analytics
stats = client.get_creator_aggregates(release_id=release.release_id)
```

## TRL GRPO Integration

Use any marketplace environment as a live reward function:

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

## Configuration

| Environment Variable    | Constructor Param  | Description                              |
|-------------------------|--------------------|------------------------------------------|
| `EPSILAB_API_KEY`       | `api_key`          | Your API key                             |
| `EPSILAB_API_BASE`      | `api_base`         | API base URL (default: production)       |
| `EPSILAB_HTTP_TIMEOUT`  | `timeout_seconds`  | Request timeout in seconds (default: 120)|
|                         | `max_retries`      | Auto-retry count for 429/5xx (default: 3)|
|                         | `backoff_base`     | Initial retry backoff in seconds (default: 1.0) |
|                         | `load_dotenv`      | Also read a local `.env` file (default: false) |

```python
client = Epsilab(load_dotenv=True)  # reads from .env file
```

## API Reference

### Discovery

| Method | Description |
|--------|-------------|
| `list_environment_listings(...)` | Browse environments you have access to |
| `list_public_listings(query, sort_by, ...)` | Browse the public marketplace catalog |
| `search_environments(query, domain, ...)` | Quality-weighted environment search |
| `get_environment_release(release_id)` | Get release details and status |

### Hosted Sessions

| Method | Description |
|--------|-------------|
| `create_environment_session(deployment_id, ...)` | Create a session and get the initial observation |
| `get_environment_session(session_id)` | Get session state, reward, and step count |
| `environment_step(session_id, action, ...)` | Take an action, receive observation and reward |
| `cancel_environment_session(session_id)` | Cancel a running session |
| `refresh_session_token(session_id)` | Refresh token for long-running sessions |
| `run_environment_episode(deployment_id, ...)` | Run a complete episode with your policy function |

### Exports and Batches

| Method | Description |
|--------|-------------|
| `create_environment_export(deployment_id, format, ...)` | Start an export job for session data |
| `get_environment_export(export_id)` | Get export job status |
| `list_environment_exports(...)` | List export jobs |
| `create_batch(deployment_id, name, task_seed_pairs, ...)` | Run batch evaluation across tasks |
| `get_batch(batch_id)` | Get batch status |
| `list_batches(...)` | List batch jobs |
| `get_batch_sessions(batch_id)` | Get sessions from a batch |
| `cancel_batch(batch_id)` | Cancel a running batch |
| `get_batch_comparison(batch_id)` | Get batch comparison report |

### Quality and Assurance

| Method | Description |
|--------|-------------|
| `list_quality_reports(...)` | List quality reports for releases |
| `get_quality_report(report_id)` | Get quality report details |
| `get_quality_checks(report_id)` | Get individual checks within a report |
| `list_quality_badges(...)` | List quality badges (gold, silver, etc.) |
| `list_contamination_findings(...)` | List contamination findings |
| `list_benchmark_results(...)` | List benchmark results |

### Billing

| Method | Description |
|--------|-------------|
| `list_license_versions(release_id)` | List license versions for a release |
| `get_license_version(license_version_id)` | Get license details |
| `list_session_charges(...)` | List session charges |
| `list_charge_adjustments(...)` | List charge adjustments |
| `list_invoices()` | List environment invoices |
| `get_invoice(invoice_id)` | Get invoice details |
| `get_invoice_line_items(invoice_id)` | Get invoice line items |
| `get_charge_summary(since, until)` | Aggregated charge summary |

### Reviews and Purchases

| Method | Description |
|--------|-------------|
| `create_review(listing_id, rating, title, ...)` | Submit a review |
| `list_reviews(listing_id)` | List reviews for a listing |
| `create_purchase(listing_id, amount_cents, ...)` | Purchase access to a listing |
| `list_purchases(...)` | List your purchases |

### Creator: Registry and Publishing

| Method | Description |
|--------|-------------|
| `create_namespace(slug, display_name)` | Create a namespace |
| `create_listing(namespace_id, slug, title, ...)` | Create a listing |
| `update_listing(listing_id, expected_revision, ...)` | Update listing metadata |
| `create_task_pack_release(...)` | Register a task pack release |
| `create_verifier_release(...)` | Register a verifier release |
| `create_environment_release(...)` | Register an environment release |

### Creator: Deployments and Entitlements

| Method | Description |
|--------|-------------|
| `create_deployment(listing_id, alias, ...)` | Deploy a release for hosted sessions |
| `create_deployment_revision(deployment_id, ...)` | Update a deployment with a new release |
| `grant_entitlement(grantee_tenant_id, listing_id, ...)` | Grant access to a tenant |
| `revoke_entitlement(entitlement_id)` | Revoke an entitlement |
| `list_entitlements(...)` | List entitlements |

### Creator: Analytics and Profile

| Method | Description |
|--------|-------------|
| `get_creator_aggregates(...)` | Usage analytics per release |
| `create_creator_profile(display_name, ...)` | Create your public creator profile |
| `get_creator_profile()` | Get your profile |
| `update_creator_profile(...)` | Update your profile |
| `create_quality_report(release_id, report_type, ...)` | Start a quality report |
| `request_publish(listing_id)` | Submit for moderation review |
| `create_changelog(release_id, version_label, summary, ...)` | Publish a changelog |
| `list_changelogs(release_id)` | List changelogs for a release |

### Creator: Settlement

| Method | Description |
|--------|-------------|
| `get_creator_account()` | Get settlement account balance |
| `list_royalty_rules()` | List royalty rules |
| `list_accruals(status)` | List royalty accruals |
| `list_settlement_adjustments()` | List settlement adjustments |
| `list_payout_batches()` | List payout batches |
| `list_creator_statements()` | List period statements |

### Adapters

| Method | Description |
|--------|-------------|
| `list_adapters(protocol_family, ...)` | List protocol adapters |
| `get_adapter(adapter_id)` | Get adapter details |
| `list_adapter_versions(adapter_id, ...)` | List adapter versions |
| `get_adapter_conformance(adapter_id, ...)` | Conformance test results |
| `check_adapter_equivalence(adapter_id, version_id)` | Behavioral equivalence check |
| `report_adapter_usage(adapter_id, version_id, event_type, ...)` | Report usage telemetry |

### Data Models

| Class | Description |
|-------|-------------|
| `EnvironmentListing` | A marketplace listing with title, visibility, and moderation state |
| `EnvironmentSession` | A hosted session with observation, reward, token, and status |
| `EnvironmentStepResult` | Step result with observation, reward, terminated, and truncated flags |
| `EnvironmentRelease` | An immutable, content-addressed release |

## Export Formats

| Format | Use Case |
|--------|----------|
| `dpo` | Direct Preference Optimization (chosen/rejected pairs) |
| `sft` | Supervised Fine-Tuning (prompt/completion pairs) |
| `grpo` | Group Relative Policy Optimization (grouped completions) |
| `kto` | Kahneman-Tversky Optimization (binary desirability) |
| `quality_dpo` | DPO pairs enriched with quality scores and feedback |
| `sharegpt` | Multi-turn conversation format |
| `jsonl` | Raw session data as NDJSON |
| `process_supervision` | Step-level reward annotations |

## Error Handling

```python
from epsilab import Epsilab, AuthError, InsufficientCreditsError, RateLimitError, ApiError

client = Epsilab(api_key="sk-...")

try:
    session = client.create_environment_session("dep-id", task_id="task-001")
except AuthError:
    print("Invalid API key")
except InsufficientCreditsError as e:
    print(f"Not enough credits: {e}")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after}s")
except ApiError as e:
    print(f"API error: {e.status_code}")
```

The SDK retries automatically on rate limits (429) and transient server errors (500, 502, 503, 504) with exponential backoff and jitter.

```python
client = Epsilab(api_key="sk-...", max_retries=5, backoff_base=2.0)
```

---

<details>
<summary><strong>Additional features: model evaluations, voice, routing, capability matrix</strong></summary>

The SDK also includes methods for model evaluation, voice evaluation, intelligent routing, and capability matrix analysis. These features are fully functional but are not the current focus.

### Model Evaluations

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

### First-Party RL Environments

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

### Voice Evaluations

| Method | Description |
|--------|-------------|
| `register_voice_asset(asset_id, uri, ...)` | Register an audio asset |
| `create_voice_task(task_id, task_type, ...)` | Create a voice evaluation task |
| `create_voice_run(target_model, ...)` | Submit a voice model for evaluation |
| `get_voice_slices(run_id)` | Get per-slice quality metrics |
| `get_voice_timeline(run_id, task_id)` | Get event timeline |
| `route_voice(prompt, ...)` | Route a voice workload to the best model |

### Capability Matrix

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

### Cross-Run Analytics

| Method | Description |
|--------|-------------|
| `get_leaderboard()` | Cross-run model leaderboard |
| `get_domain_leaderboard()` | Per-domain model scores |
| `get_cost_analysis()` | Cost-efficiency rankings |
| `get_precomputed_insights()` | Per-domain best-model recommendations |

### Tasks and API Keys

| Method | Description |
|--------|-------------|
| `get_task(task_id)` / `create_task(task)` | Get or create evaluation tasks |
| `upload_custom_tasks(tasks)` | Batch upload custom tasks |
| `classify_tasks(tasks)` | Auto-classify by domain and capability |
| `list_tasks(...)` / `iter_tasks(...)` | List or iterate tasks |
| `list_api_keys()` / `create_api_key(label)` / `revoke_api_key(key_id)` | Manage API keys |
| `get_usage(period)` | Monthly usage summary |
| `get_credit_balance()` / `get_credit_ledger(...)` | Credit balance and history |

</details>

## Examples

- [`examples/marketplace_example.py`](examples/marketplace_example.py) — Marketplace workflows for both sides
- [`examples/example.py`](examples/example.py) — Evaluation, export, and RL workflow

## License

Apache 2.0 — see [LICENSE](LICENSE).
