# API Reference

Full reference for the `epsilab` Python SDK.

## Discovery

| Method | Description |
|--------|-------------|
| `list_environment_listings(...)` | Browse public, owned, and shared environments; authentication is optional for public discovery |
| `get_environment_listing(listing_id)` | Open a visible listing directly, including unlisted links |
| `list_public_listings(query, sort_by, ...)` | Browse the public marketplace catalog |
| `search_environments(query, domain, ...)` | Quality-weighted environment search |
| `get_environment_release(release_id)` | Get release details and status |

## Application Tools

| Method | Description |
|--------|-------------|
| `list_application_tools(query, plugin, ...)` | Browse public reusable application tools |
| `get_application_tool(tool_id)` | Open a public or unlisted tool directly |
| `get_application_tool_release(release_id)` | Get an immutable tool release |
| `create_application_tool(...)` | Create a public, unlisted, or private tool listing |
| `create_application_tool_release(...)` | Publish an immutable tool release |

## Hosted Sessions

| Method | Description |
|--------|-------------|
| `create_environment_session(deployment_id, ...)` | Create a session and get the initial observation |
| `get_environment_session(session_id)` | Get session state, reward, and step count |
| `environment_step(session_id, action, ...)` | Take an action, receive observation and reward |
| `cancel_environment_session(session_id)` | Cancel a running session |
| `refresh_session_token(session_id)` | Refresh token for long-running sessions |
| `run_environment_episode(deployment_id, ...)` | Run a complete episode with your policy function |

## Exports and Batches

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

## Quality and Assurance

| Method | Description |
|--------|-------------|
| `list_quality_reports(...)` | List quality reports for releases |
| `get_quality_report(report_id)` | Get quality report details |
| `get_quality_checks(report_id)` | Get individual checks within a report |
| `list_quality_badges(...)` | List quality badges (gold, silver, etc.) |
| `list_contamination_findings(...)` | List contamination findings |
| `list_benchmark_results(...)` | List benchmark results |

## Billing

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

## Reviews and Purchases

| Method | Description |
|--------|-------------|
| `create_review(listing_id, rating, title, ...)` | Submit a review |
| `list_reviews(listing_id)` | List reviews for a listing |
| `create_purchase(listing_id, license_version_id, ...)` | Purchase access under a published license offer |
| `list_purchases(...)` | List your purchases |

## Creator: Publishing

| Method | Description |
|--------|-------------|
| `create_namespace(slug, display_name)` | Create a namespace |
| `create_listing(namespace_id, slug, title, ...)` | Create a listing |
| `update_listing(listing_id, expected_revision, ...)` | Update listing metadata |
| `create_task_pack_release(...)` | Register a task pack release |
| `create_verifier_release(...)` | Register a verifier release |
| `create_environment_release(...)` | Register an environment release |

## Creator: Deployments and Entitlements

| Method | Description |
|--------|-------------|
| `create_deployment(listing_id, alias, ...)` | Deploy a release for hosted sessions |
| `create_deployment_revision(deployment_id, ...)` | Update a deployment with a new release |
| `grant_entitlement(grantee_tenant_id, listing_id, ...)` | Grant access to a tenant |
| `revoke_entitlement(entitlement_id)` | Revoke an entitlement |
| `list_entitlements(...)` | List entitlements |

## Creator: Analytics and Profile

| Method | Description |
|--------|-------------|
| `get_creator_aggregates(...)` | Usage analytics per release |
| `create_creator_profile(display_name, ...)` | Create your public creator profile |
| `get_creator_profile()` | Get your profile |
| `update_creator_profile(...)` | Update your profile |
| `create_quality_report(release_id, report_type, ...)` | Start a quality report |
| `request_publish(listing_id)` | Publish listing to the hub |
| `create_changelog(release_id, version_label, summary, ...)` | Publish a changelog |
| `list_changelogs(release_id)` | List changelogs for a release |

## Creator: Settlement

| Method | Description |
|--------|-------------|
| `get_creator_account()` | Get settlement account balance |
| `list_royalty_rules()` | List royalty rules |
| `list_accruals(status)` | List royalty accruals |
| `list_settlement_adjustments()` | List settlement adjustments |
| `list_payout_batches()` | List payout batches |
| `list_creator_statements()` | List period statements |

## Adapters

| Method | Description |
|--------|-------------|
| `list_adapters(protocol_family, ...)` | List protocol adapters |
| `get_adapter(adapter_id)` | Get adapter details |
| `list_adapter_versions(adapter_id, ...)` | List adapter versions |
| `get_adapter_conformance(adapter_id, ...)` | Conformance test results |
| `check_adapter_equivalence(adapter_id, version_id)` | Behavioral equivalence check |
| `report_adapter_usage(adapter_id, version_id, event_type, ...)` | Report usage telemetry |

## Image Upload

| Method | Description |
|--------|-------------|
| `upload_image(tarball_path, tag)` | Upload a Docker image tarball to the platform |
| `get_platform_config()` | Get platform capabilities (upload support, size limits) |

## Data Models

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
