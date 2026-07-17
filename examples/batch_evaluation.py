"""Batch evaluation: benchmark a model across environments and tasks.

Run your model against many tasks in parallel across one or more
environments and compare performance. Useful for:
    - Benchmarking before/after fine-tuning
    - Comparing environments for training signal quality
    - Generating large-scale training data from diverse tasks

Usage:
    pip install epsilab
    epsilab login

    # Evaluate on one environment
    python examples/batch_evaluation.py --envs bug-hunter

    # Evaluate across multiple environments
    python examples/batch_evaluation.py --envs bug-hunter,refactor,test-writer

    # Server-side batch (parallel, uses the batch API)
    python examples/batch_evaluation.py --envs bug-hunter --mode batch
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from epsilab import Epsilab


def resolve_environments(client: Epsilab, env_spec: str) -> list[dict]:
    """Resolve --envs into a list of {slug, deployment_id} dicts."""
    listings = client.list_environment_listings(limit=200)
    available = [l for l in listings if l.deployment_id]

    if env_spec == "all":
        return [{"slug": l.slug, "deployment_id": l.deployment_id} for l in available]

    slugs = [s.strip() for s in env_spec.split(",") if s.strip()]
    available_map = {l.slug: l for l in available}
    resolved = []
    for slug in slugs:
        if slug not in available_map:
            avail = sorted(available_map.keys())[:15]
            raise SystemExit(f"Environment '{slug}' not found.\nAvailable: {', '.join(avail)}")
        l = available_map[slug]
        resolved.append({"slug": l.slug, "deployment_id": l.deployment_id})
    return resolved


def evaluate_sequential(
    client: Epsilab,
    environments: list[dict],
    model_fn,
    tasks_per_env: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Run a model sequentially against tasks and collect rewards."""
    results = []
    for env in environments:
        slug = env["slug"]
        dep_id = env["deployment_id"]
        task_ids = [f"{slug}-train-easy-{str(i).zfill(3)}" for i in range(1, tasks_per_env + 1)]
        print(f"\n  [{slug}] evaluating {len(task_ids)} tasks ...")

        for task_id in task_ids:
            try:
                session = client.create_environment_session(dep_id, task_id=task_id, seed=seed)
                session = client.wait_for_session(session)
                action = model_fn(session.observation)
                result = client.environment_step(
                    session.session_id, action, session_token=session.session_token,
                )
                results.append({
                    "env": slug,
                    "task_id": task_id,
                    "reward": result.reward or 0.0,
                    "terminated": result.terminated,
                })
                print(f"    {task_id:45s}  reward={result.reward or 0:.3f}")
            except Exception as e:
                print(f"    {task_id:45s}  skipped: {e}")

    return results


def evaluate_batch(
    client: Epsilab,
    environments: list[dict],
    tasks_per_env: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Run server-side batch evaluation (parallel)."""
    results = []
    for env in environments:
        slug = env["slug"]
        dep_id = env["deployment_id"]
        task_seed_pairs = [
            {"task_id": f"{slug}-train-easy-{str(i).zfill(3)}", "seed": seed}
            for i in range(1, tasks_per_env + 1)
        ]
        print(f"\n  [{slug}] submitting batch of {len(task_seed_pairs)} tasks ...")

        try:
            batch = client.create_batch(
                deployment_id=dep_id,
                name=f"eval-{slug}",
                task_seed_pairs=task_seed_pairs,
            )
            batch_id = batch.get("batch_id")
            print(f"    Batch {batch_id}")

            while True:
                status = client.get_batch(batch_id)
                state = status.get("status", "unknown")
                print(f"    Status: {state}")
                if state in ("completed", "failed", "cancelled"):
                    break
                time.sleep(5)

            if state == "completed":
                sessions = client.get_batch_sessions(batch_id)
                for s in sessions:
                    results.append({
                        "env": slug,
                        "task_id": s.get("task_id", "?"),
                        "reward": s.get("total_reward", 0.0),
                        "terminated": s.get("status") == "completed",
                    })
            else:
                print(f"    Batch {state}")
        except Exception as e:
            print(f"    Batch failed: {e}")

    return results


def print_summary(results: list[dict]) -> None:
    """Print per-environment and overall summary."""
    if not results:
        print("\n  No results to summarize.")
        return

    by_env: dict[str, list[dict]] = {}
    for r in results:
        by_env.setdefault(r["env"], []).append(r)

    print(f"\n{'Environment':<30s} {'Tasks':>5s} {'Solved':>6s} {'Avg Reward':>12s}")
    print("-" * 55)
    for env, records in sorted(by_env.items()):
        avg = sum(r["reward"] for r in records) / len(records)
        solved = sum(1 for r in records if r["reward"] > 0.5)
        print(f"  {env:<28s} {len(records):>5d} {solved:>6d} {avg:>12.3f}")

    total_avg = sum(r["reward"] for r in results) / len(results)
    total_solved = sum(1 for r in results if r["reward"] > 0.5)
    print("-" * 55)
    print(f"  {'OVERALL':<28s} {len(results):>5d} {total_solved:>6d} {total_avg:>12.3f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch evaluation across environments and tasks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--envs", default="bug-hunter",
                    help='Comma-separated slugs or "all" (default: bug-hunter)')
    p.add_argument("--tasks-per-env", type=int, default=5,
                    help="Tasks to evaluate per environment (default: 5)")
    p.add_argument("--seed", type=int, default=42,
                    help="Seed for reproducibility (default: 42)")
    p.add_argument("--mode", choices=["sequential", "batch"], default="sequential",
                    help="Sequential (client-side) or batch (server-side) (default: sequential)")
    p.add_argument("--output", default=None,
                    help="Save results to JSONL file")
    return p.parse_args()


def main():
    args = parse_args()
    client = Epsilab(load_dotenv=True)

    print("=" * 60)
    print("  Batch Evaluation")
    print("=" * 60)

    environments = resolve_environments(client, args.envs)
    print(f"\n  {len(environments)} environment(s): {', '.join(e['slug'] for e in environments)}")
    print(f"  Mode: {args.mode}, {args.tasks_per_env} tasks/env, seed={args.seed}")

    def model_fn(observation: str) -> str:
        return f"Based on the observation, here is my solution: {observation[:100]}"

    if args.mode == "batch":
        results = evaluate_batch(client, environments, args.tasks_per_env, args.seed)
    else:
        results = evaluate_sequential(client, environments, model_fn, args.tasks_per_env, args.seed)

    print_summary(results)

    if args.output:
        out = Path(args.output)
        with open(out, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\n  Results saved to {out}")

    print("\n  Supported export formats: grpo, dpo, sft, kto, process_supervision")
    print("  Use:  epsilab export --format <fmt> --deployment <id>")

    client.close()


if __name__ == "__main__":
    main()
