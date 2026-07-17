"""Post-train a model using Epsilab environments.

End-to-end example: discover public environments, run your model
against them to collect training data, export as GRPO, and fine-tune
with TRL.

Usage:
    pip install epsilab trl transformers datasets
    export EPSILAB_API_KEY=sk-...
    python examples/run_environment.py

Steps:
    1. Search the hub for public environments with deployments
    2. Run episodes — submit your model's outputs, receive rewards
    3. Export session data as GRPO-formatted training records
    4. Fine-tune with TRL's GRPOTrainer using the exported data
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

from epsilab import Epsilab


def find_deployed_environment(
    client: Epsilab,
    domain: str = "coding",
) -> Optional[dict]:
    """Search for a public environment that has an active deployment."""
    listings = client.search_environments(domain=domain, limit=20)
    for env in listings:
        dep_id = env.get("deployment_id") if isinstance(env, dict) else getattr(env, "deployment_id", None)
        if dep_id:
            return {
                "deployment_id": dep_id,
                "slug": env.get("slug") if isinstance(env, dict) else getattr(env, "slug", "?"),
                "title": env.get("title") if isinstance(env, dict) else getattr(env, "title", "?"),
            }
    return None


def collect_episodes(
    client: Epsilab,
    deployment_id: str,
    policy_fn,
    task_ids: list[str],
    seed: int = 42,
    max_steps: int = 10,
) -> list[dict]:
    """Run the policy against each task, returning per-episode stats."""
    episodes = []
    for task_id in task_ids:
        session = client.create_environment_session(
            deployment_id,
            task_id=task_id,
            seed=seed,
        )
        steps = 0
        total_reward = 0.0
        done = False

        while not done and steps < max_steps:
            action = policy_fn(session.observation if steps == 0 else result.observation)
            result = client.environment_step(
                session.session_id,
                action,
                session_token=session.session_token,
            )
            steps += 1
            total_reward += result.reward or 0.0
            done = result.terminated or result.truncated

        episodes.append({
            "session_id": session.session_id,
            "task_id": task_id,
            "steps": steps,
            "total_reward": total_reward,
        })
        print(f"  {task_id:40s}  steps={steps}  reward={total_reward:.3f}")

    return episodes


def export_training_data(
    client: Epsilab,
    deployment_id: str,
    output_dir: Path,
    fmt: str = "grpo",
) -> Optional[dict]:
    """Start an export job and poll until ready."""
    export = client.create_environment_export(
        deployment_id=deployment_id,
        format=fmt,
    )
    export_id = export.get("export_id")
    if not export_id:
        print("  Export creation returned no export_id — skipping")
        return None

    print(f"  Export started: {export_id} (format={fmt})")

    for _ in range(60):
        status = client.get_environment_export(export_id)
        state = status.get("status", "unknown")
        if state == "completed":
            break
        if state == "failed":
            print(f"  Export failed: {status.get('error', 'unknown error')}")
            return None
        time.sleep(2)
    else:
        print("  Export timed out after 120s")
        return None

    records = status.get("records", [])
    if records:
        out_path = output_dir / f"training_{fmt}.jsonl"
        with open(out_path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        print(f"  Saved {len(records)} records to {out_path}")

    return status


def train_with_trl(data_path: Path) -> None:
    """Load exported GRPO data and run a short training loop with TRL.

    Requires: pip install trl transformers datasets torch
    """
    try:
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError:
        print("\n  To run training, install: pip install trl transformers datasets torch")
        print("  Skipping training step.")
        return

    records = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("  No training records found — skipping training")
        return

    prompts = [r.get("prompt", r.get("query", "")) for r in records]
    completions = [r.get("completion", r.get("response", "")) for r in records]
    rewards = [r.get("reward", 0.0) for r in records]
    dataset = Dataset.from_dict({
        "prompt": prompts,
        "completion": completions,
        "reward": rewards,
    })
    print(f"  Loaded {len(dataset)} training examples")

    model_name = "Qwen/Qwen3-0.6B"
    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    config = GRPOConfig(
        output_dir="output/grpo-finetuned",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=1,
        max_steps=20,
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("  Starting GRPO training...")
    trainer.train()
    trainer.save_model("output/grpo-finetuned")
    print(f"  Model saved to output/grpo-finetuned/")


def main():
    client = Epsilab(load_dotenv=True)
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Post-Training with Epsilab Environments")
    print("=" * 60)

    # ── 1. Discover a public environment ─────────────────────────
    print("\n1. Searching for deployed environments...")

    env = find_deployed_environment(client, domain="coding")
    if not env:
        env = find_deployed_environment(client, domain="math")
    if not env:
        listings = client.list_environment_listings(limit=20)
        for listing in listings:
            if listing.deployment_id:
                env = {
                    "deployment_id": listing.deployment_id,
                    "slug": listing.slug,
                    "title": listing.title,
                }
                break

    if not env:
        print("  No deployed environments found.")
        print("  Deploy one first: epsilab env init my-env && cd my-env && epsilab deploy")
        client.close()
        return

    print(f"  Found: {env['title']} ({env['slug']})")
    print(f"  Deployment: {env['deployment_id']}")

    # ── 2. Collect episodes ──────────────────────────────────────
    print("\n2. Running episodes...")

    def policy(observation: str) -> str:
        """Placeholder policy — replace with your model's inference."""
        return f"Solution: {str(observation)[:200]}"

    sessions = client.list_rl_sessions(limit=5)
    past_task_ids = [
        s.get("task_id") for s in sessions
        if s.get("task_id") and s.get("status") == "completed"
    ]

    if past_task_ids:
        print(f"  Found {len(past_task_ids)} completed sessions from prior runs")
    else:
        print("  Running new episodes against the environment...")
        try:
            episodes = collect_episodes(
                client,
                env["deployment_id"],
                policy_fn=policy,
                task_ids=["task-001", "task-002", "task-003"],
                max_steps=5,
            )
            avg_reward = (
                sum(e["total_reward"] for e in episodes) / len(episodes)
                if episodes else 0
            )
            print(f"\n  {len(episodes)} episodes, avg reward: {avg_reward:.3f}")
        except Exception as e:
            print(f"  Could not run episodes: {e}")
            print("  Continuing with export from any existing sessions...")

    # ── 3. Export training data ──────────────────────────────────
    print("\n3. Exporting training data as GRPO...")

    export = export_training_data(client, env["deployment_id"], output_dir, fmt="grpo")
    data_path = output_dir / "training_grpo.jsonl"

    if not export or not data_path.exists():
        print("  No training data exported.")
        print("  Run more sessions first, then re-run this script.")
        client.close()
        return

    # ── 4. Fine-tune ─────────────────────────────────────────────
    print("\n4. Fine-tuning with TRL GRPOTrainer...")
    train_with_trl(data_path)

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Post-training complete.")
    print(f"  Environment:  {env['title']}")
    print(f"  Training data: {data_path}")
    if Path("output/grpo-finetuned").exists():
        print(f"  Fine-tuned model: output/grpo-finetuned/")
    print("=" * 60)

    client.close()


if __name__ == "__main__":
    main()
