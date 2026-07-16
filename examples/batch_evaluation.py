"""Example: batch evaluation across tasks and environments.

Run your model against many tasks in parallel and compare performance
across environments. Useful for:
    - Benchmarking a model before/after fine-tuning
    - Comparing multiple environments for training signal quality
    - Generating large-scale training data from diverse tasks

Usage:
    pip install epsilab
    export EPSILAB_API_KEY=sk-...
    python examples/batch_evaluation.py
"""

from __future__ import annotations

from epsilab import Epsilab


def evaluate_model(client: Epsilab, deployment_id: str, task_ids: list[str], model_fn):
    """Run a model against a set of tasks and collect rewards."""
    results = []
    for task_id in task_ids:
        session = client.create_environment_session(
            deployment_id,
            task_id=task_id,
            seed=42,
        )

        action = model_fn(session.observation)

        result = client.environment_step(
            session.session_id,
            action,
            session_token=session.session_token,
        )
        results.append({
            "task_id": task_id,
            "reward": result.reward or 0.0,
            "terminated": result.terminated,
            "observation": str(result.observation)[:200] if result.observation else None,
        })
        print(f"  {task_id:40s}  reward={result.reward or 0:.3f}")

    avg = sum(r["reward"] for r in results) / len(results) if results else 0
    solved = sum(1 for r in results if r["reward"] > 0.5)
    print(f"\n  Average reward: {avg:.3f}")
    print(f"  Solved: {solved}/{len(results)}")
    return results


def main():
    client = Epsilab(load_dotenv=True)

    print("═" * 60)
    print("Batch Evaluation Example")
    print("═" * 60)

    # ── List available environments ──────────────────────────────
    listings = client.list_environment_listings(limit=10)
    print(f"\nAvailable environments ({len(listings)}):")
    for listing in listings[:5]:
        print(f"  {listing.slug:30s}  {listing.title}")

    # ── Example: evaluate a simple policy ────────────────────────
    # Replace with your model's inference function:
    def dummy_model(observation: str) -> str:
        return f"Based on the observation, here is my solution: {observation[:100]}"

    # To run a real batch evaluation:
    #
    #   results = evaluate_model(
    #       client,
    #       deployment_id="<your-deployment-id>",
    #       task_ids=["task-001", "task-002", "task-003"],
    #       model_fn=dummy_model,
    #   )
    #
    # Or use the batch API for server-side parallelism:
    #
    #   batch = client.create_batch(
    #       deployment_id="<deployment-id>",
    #       name="pre-training-baseline",
    #       task_seed_pairs=[
    #           {"task_id": "task-001", "seed": 42},
    #           {"task_id": "task-002", "seed": 42},
    #           {"task_id": "task-003", "seed": 42},
    #       ],
    #   )
    #   print(f"Batch started: {batch.get('batch_id')}")
    #
    #   # Poll until complete
    #   import time
    #   while True:
    #       status = client.get_batch(batch["batch_id"])
    #       if status.get("status") in ("completed", "failed"):
    #           break
    #       time.sleep(5)
    #
    #   # Get results
    #   sessions = client.get_batch_sessions(batch["batch_id"])
    #   comparison = client.get_batch_comparison(batch["batch_id"])

    # ── Export training data ─────────────────────────────────────
    print("\nSupported export formats:")
    formats = ["grpo", "dpo", "sft", "kto", "process_supervision"]
    for fmt in formats:
        print(f"  {fmt:25s}  epsilab export --format {fmt}")

    print("\n✓ Done")
    client.close()


if __name__ == "__main__":
    main()
