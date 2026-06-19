"""Example: evaluate models and explore results via the Epsilab SDK.

Usage:
    1. Copy ``.env.example`` to ``.env`` and add your API key.
    2. Run ``python examples/example.py``.
    3. Set ``EPSILAB_RUN_RL_EXAMPLE=1`` to also run one interactive RL step.

The example first checks for existing completed runs to avoid
unnecessary charges. If none exist, it creates a small single-model
run using a free model (if available) with only 5 tasks.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from epsilab import Epsilab


def _find_free_model(client: Epsilab) -> Optional[str]:
    """Return the ID of a free model, or None."""
    data = client.list_models(limit=200)
    for m in data.get("models", []):
        prompt = m.get("prompt_cost_per_m") or 0
        completion = m.get("completion_cost_per_m") or 0
        if prompt == 0 and completion == 0:
            return m["model_id"]
    return None


def _run_rl_example(client: Epsilab) -> None:
    """Run one interactive RL step without submitting placeholder data."""
    curriculum = client.get_rl_curriculum(env_type="single_turn", batch_size=1)
    batch = curriculum.get("curriculum", {})
    task_ids = batch.get("frontier_tasks", []) + batch.get("exploration_tasks", [])
    if not task_ids:
        print("No RL curriculum tasks are currently available")
        return

    session = client.create_rl_session(
        task_ids[0],
        env_type="single_turn",
        reward_mode="continuous",
    )
    completed = False
    try:
        print(f"\nTask: {session.observation}")
        action = input("Response (leave blank to cancel): ").strip()
        if not action:
            return

        result = client.rl_step(session.session_id, action)
        completed = result.done
        print(f"Reward: {result.reward}, done={result.done}")
    finally:
        if not completed:
            client.close_rl_session(session.session_id, reason="example_cancelled")


def main() -> None:
    client = Epsilab(load_dotenv=True)

    # ── Find or create a run ──────────────────────────────────────────

    runs = client.list_runs(status="completed", limit=1)

    if runs:
        run = runs[0]
        print(f"Using existing run: {run.run_id}")
    else:
        print("No completed runs found — creating a small run...")

        free_model = _find_free_model(client)
        model_id = free_model or "google/gemini-2.5-flash"
        if free_model:
            print(f"Found free model: {model_id}")
        else:
            print(f"No free models available, using: {model_id}")

        estimate = client.estimate_evaluation_cost(
            [model_id],
            max_tasks=5,
        )
        print(f"Estimated cost: {estimate.total_credits} credits")
        if not estimate.sufficient:
            print(
                f"Insufficient credits (balance: {estimate.balance}). "
                "Please add credits and try again."
            )
            return

        run = client.create_run(model_id, max_tasks=5)
        print(f"Run started: {run.run_id}")

        run = client.wait_for_completion(run.run_id, poll_interval=10)
        print(f"Completed: {run.task_count} tasks, {run.gap_count} gaps")

    run_id = run.run_id

    # ── Inspect results ───────────────────────────────────────────────

    print(f"\nRun {run_id}: {run.task_count} tasks, {run.gap_count} gaps")

    gaps = client.get_gaps(run_id)
    for gap in gaps:
        print(
            f"  Gap: {gap.capability}  alpha={gap.alpha_score:.3f}  "
            f"priority={gap.priority}"
        )

    insights = client.get_insights(run_id)
    for model in insights.get("model_performance", []):
        alias = model.get("alias", model.get("model_id", "?"))
        score = model.get("mean_score", 0)
        print(f"  {alias}: {score:.1%}")

    artifacts = client.get_artifacts(run_id)
    print(f"\n{len(artifacts)} artifacts generated")

    # ── Export training data ──────────────────────────────────────────

    if artifacts:
        out = Path("output")
        out.mkdir(exist_ok=True)
        client.export_run(run_id, format="dpo", path=str(out / "dpo_pairs.jsonl"))
        print("Exported DPO pairs to output/dpo_pairs.jsonl")

        client.export_run(
            run_id,
            format="process_supervision",
            path=str(out / "process_supervision.jsonl"),
        )
        print("Exported process supervision to output/process_supervision.jsonl")

    # ── Refined trajectories ─────────────────────────────────────────

    refined = client.get_refined_trajectories(run_id)
    if refined:
        print(f"\n{len(refined)} refined trajectories available:")
        for t in refined[:3]:
            ratio = t.compression_ratio or 0
            domain = t.content.get("domain", "?")
            orig = t.content.get("original_step_count", "?")
            new = t.content.get("refined_step_count", "?")
            print(f"  {domain}: {orig}→{new} steps ({ratio:.0%} of original)")
    else:
        print("\nNo refined trajectories (requires passing workflow tasks)")

    # ── RL environment loop ─────────────────────────────────────────

    # RL sessions may consume credits and contribute training records, so this
    # interactive example is explicitly opt-in.
    if os.environ.get("EPSILAB_RUN_RL_EXAMPLE") == "1":
        _run_rl_example(client)

    # ── Billing ───────────────────────────────────────────────────────

    balance = client.get_credit_balance()
    print(f"\nCredits remaining: {balance.get('balance', 0)}")

    for usage in client.get_usage():
        print(f"  {usage.period}: {usage.run_count} runs, ${usage.total_cost_usd:.2f}")


if __name__ == "__main__":
    main()
