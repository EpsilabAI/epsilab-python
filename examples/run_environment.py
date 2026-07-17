"""Collect training data from Epsilab environments and fine-tune a model.

Runs episodes across one or more Hub environments, formats the collected
(prompt, completion, reward) triples for the algorithm of your choice
(SFT, DPO, KTO), and fine-tunes via a remote provider or locally.

Usage:
    pip install epsilab[training]
    epsilab login
    export TOGETHER_API_KEY=...   # only if using --provider together

    # Single environment
    python examples/run_environment.py --envs bug-hunter

    # Multiple environments (recommended for generalist training)
    python examples/run_environment.py --envs bug-hunter,refactor,test-writer

    # All available environments
    python examples/run_environment.py --envs all --algorithm dpo --provider local
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from epsilab import Epsilab


# ── Environment discovery ────────────────────────────────────────


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
            raise SystemExit(
                f"Environment '{slug}' not found or has no deployment.\n"
                f"Available: {', '.join(avail)}"
            )
        l = available_map[slug]
        resolved.append({"slug": l.slug, "deployment_id": l.deployment_id})
    return resolved


# ── Data collection ──────────────────────────────────────────────


def collect_training_data(
    client: Epsilab,
    environments: list[dict],
    sessions_per_env: int = 10,
) -> list[dict]:
    """Run episodes across environments and collect training triples."""
    records = []

    for env in environments:
        slug = env["slug"]
        dep_id = env["deployment_id"]
        task_ids = [f"{slug}-train-easy-{str(i).zfill(3)}" for i in range(1, sessions_per_env + 1)]
        print(f"\n  [{slug}] running {len(task_ids)} sessions ...")

        for task_id in task_ids:
            try:
                session = client.create_environment_session(dep_id, task_id=task_id)
                session = client.wait_for_session(session)
                prompt = str(session.observation or "")
                response = (
                    f"I'll analyze this step by step.\n\n"
                    f"Based on the problem described, I need to consider the key "
                    f"requirements and constraints. {prompt[:200]}\n\n"
                    f"My solution addresses each requirement systematically."
                )
                result = client.environment_step(
                    session.session_id, response,
                    session_token=session.session_token,
                )
                records.append({
                    "prompt": prompt,
                    "completion": response,
                    "reward": result.reward or 0.0,
                    "task_id": task_id,
                    "env": slug,
                })
                print(f"    {task_id:50s}  reward={result.reward or 0:.3f}")
            except Exception as e:
                print(f"    {task_id:50s}  skipped: {e}")

    return records


# ── Data formatting per algorithm ────────────────────────────────


def format_sft(records: list[dict]) -> list[dict]:
    """Keep top-half by reward as supervised examples."""
    records.sort(key=lambda r: r["reward"], reverse=True)
    cutoff = max(1, len(records) // 2)
    return [
        {"messages": [
            {"role": "user", "content": r["prompt"]},
            {"role": "assistant", "content": r["completion"]},
        ]}
        for r in records[:cutoff]
    ]


def format_dpo(records: list[dict]) -> list[dict]:
    """Build preference pairs: high-reward=chosen, low-reward=rejected."""
    from itertools import combinations

    by_task: dict[str, list[dict]] = {}
    for r in records:
        by_task.setdefault(r["task_id"], []).append(r)

    pairs = []
    for task_records in by_task.values():
        if len(task_records) < 2:
            continue
        for a, b in combinations(task_records, 2):
            if a["reward"] == b["reward"]:
                continue
            chosen, rejected = (a, b) if a["reward"] > b["reward"] else (b, a)
            pairs.append({
                "prompt": [{"role": "user", "content": chosen["prompt"]}],
                "chosen": [{"role": "assistant", "content": chosen["completion"]}],
                "rejected": [{"role": "assistant", "content": rejected["completion"]}],
            })

    if not pairs:
        records.sort(key=lambda r: r["reward"], reverse=True)
        mid = len(records) // 2
        for i in range(min(mid, len(records) - mid)):
            pairs.append({
                "prompt": [{"role": "user", "content": records[i]["prompt"]}],
                "chosen": [{"role": "assistant", "content": records[i]["completion"]}],
                "rejected": [{"role": "assistant", "content": records[mid + i]["completion"]}],
            })

    return pairs


def format_kto(records: list[dict]) -> list[dict]:
    """Label each example as good (reward > median) or bad."""
    if not records:
        return []
    median = sorted(r["reward"] for r in records)[len(records) // 2]
    return [
        {"messages": [
            {"role": "user", "content": r["prompt"]},
            {"role": "assistant", "content": r["completion"]},
        ],
         "label": r["reward"] > median}
        for r in records
    ]


FORMATTERS = {"sft": format_sft, "dpo": format_dpo, "kto": format_kto}


# ── Training backends ────────────────────────────────────────────


def _write_jsonl(data: list[dict]) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for row in data:
        f.write(json.dumps(row) + "\n")
    f.close()
    return f.name


def train_together(model_id: str, data: list[dict], algorithm: str) -> None:
    """Fine-tune via Together AI's managed platform."""
    from together import Together

    client = Together()
    tmp_path = _write_jsonl(data)

    try:
        print(f"  Uploading {len(data)} {algorithm.upper()} examples...")
        train_file = client.files.upload(file=tmp_path, purpose="fine-tune")
        print(f"  File: {train_file.id}")

        create_kwargs: dict = {
            "training_file": train_file.id,
            "model": model_id,
            "n_epochs": 3,
            "learning_rate": 1e-5,
            "suffix": f"epsilab-{algorithm}",
        }
        if algorithm == "dpo":
            create_kwargs["training_method"] = "dpo"

        print(f"  Starting {algorithm.upper()} fine-tune of {model_id}...")
        job = client.fine_tuning.create(**create_kwargs)
        print(f"  Job: {job.id}")

        while True:
            status = client.fine_tuning.retrieve(id=job.id)
            state = status.status
            print(f"  Status: {state}")
            if state in ("completed", "failed", "cancelled"):
                break
            time.sleep(10)

        if state == "completed":
            print(f"  Fine-tuned model: {status.output_name}")
        else:
            print(f"  Job {state}: {getattr(status, 'error', 'unknown')}")
    finally:
        os.unlink(tmp_path)


def train_fireworks(model_id: str, data: list[dict], algorithm: str) -> None:
    """Fine-tune via Fireworks AI's managed SFT/DPO."""
    import fireworks.client as fc

    tmp_path = _write_jsonl(data)
    try:
        print(f"  Uploading {len(data)} {algorithm.upper()} examples...")
        dataset = fc.datasets.create(name=f"epsilab-{algorithm}", file_path=tmp_path)

        fw_model = f"accounts/fireworks/models/{model_id.split('/')[-1].lower()}"
        print(f"  Starting {algorithm.upper()} fine-tune of {fw_model}...")
        job = fc.fine_tuning.create(
            base_model=fw_model, dataset=dataset.id,
            lora_rank=16, epochs=3,
        )
        print(f"  Job: {job.id}")

        while True:
            status = fc.fine_tuning.get(job.id)
            state = status.state
            print(f"  Status: {state}")
            if state in ("COMPLETED", "FAILED"):
                break
            time.sleep(10)

        if state == "COMPLETED":
            print(f"  Fine-tuned model: {status.fine_tuned_model}")
    finally:
        os.unlink(tmp_path)


def train_tinker(model_id: str, data: list[dict], algorithm: str) -> None:
    """Fine-tune via Tinker's training API with custom loss."""
    import asyncio

    import tinker
    from tinker import types

    async def _train():
        client = tinker.ServiceClient()
        tc = await client.create_lora_training_client_async(
            base_model=model_id, rank=16,
        )

        training_data = []
        for row in data:
            messages = row.get("messages", [])
            if not messages and "prompt" in row:
                messages = row["prompt"] + row.get("chosen", [])
            datum = tinker.tokenize_chat(messages=messages, model=model_id)
            training_data.append(datum)

        print(f"  Training {model_id} ({algorithm.upper()}) on {len(training_data)} examples...")
        for step in range(20):
            batch = [training_data[step % len(training_data)]]
            fb = await tc.forward_backward_async(batch, "cross_entropy")
            optim = await tc.optim_step_async(types.AdamParams(learning_rate=1e-4))
            result = await fb.result_async()
            await optim.result_async()
            loss = result.metrics.get("loss:mean", 0)
            print(f"  Step {step:2d}: loss={loss:.4f}")

        await tc.save_weights_and_get_sampling_client()
        print("  Model saved on Tinker.")

    asyncio.run(_train())


def train_local(model_id: str, data: list[dict], algorithm: str) -> None:
    """Fine-tune locally with TRL (SFTTrainer, DPOTrainer, or KTOTrainer)."""
    import torch
    from datasets import Dataset

    use_gpu = torch.cuda.is_available()
    print(f"  Training {model_id} ({algorithm.upper()}) on {'GPU' if use_gpu else 'CPU'}...")

    common_config = {
        "output_dir": f"output/{algorithm}-finetuned",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "logging_steps": 1,
        "save_strategy": "no",
        "max_steps": 20,
        "report_to": "none",
        "bf16": use_gpu,
        "fp16": False,
        "use_cpu": not use_gpu,
    }

    if algorithm == "sft":
        from trl import SFTConfig, SFTTrainer
        dataset = Dataset.from_list([
            {"prompt": m[0]["content"], "completion": m[1]["content"]}
            for row in data for m in [row["messages"]] if len(m) >= 2
        ])
        trainer = SFTTrainer(model=model_id, args=SFTConfig(**common_config), train_dataset=dataset)

    elif algorithm == "dpo":
        from trl import DPOConfig, DPOTrainer
        dataset = Dataset.from_list([
            {
                "prompt": p[0]["content"] if isinstance(p := row["prompt"], list) else p,
                "chosen": row["chosen"][0]["content"] if isinstance(row["chosen"], list) else row["chosen"],
                "rejected": row["rejected"][0]["content"] if isinstance(row["rejected"], list) else row["rejected"],
            }
            for row in data
        ])
        trainer = DPOTrainer(model=model_id, args=DPOConfig(**common_config), train_dataset=dataset)

    elif algorithm == "kto":
        from trl import KTOConfig, KTOTrainer
        dataset = Dataset.from_list([
            {"prompt": row["messages"][0]["content"], "completion": row["messages"][1]["content"], "label": row["label"]}
            for row in data
        ])
        trainer = KTOTrainer(model=model_id, args=KTOConfig(**common_config), train_dataset=dataset)
    else:
        print(f"  Unknown algorithm: {algorithm}")
        return

    trainer.train()
    trainer.save_model(f"output/{algorithm}-finetuned")
    print(f"  Saved to output/{algorithm}-finetuned/")


TRAINERS = {
    "together": train_together,
    "fireworks": train_fireworks,
    "tinker": train_tinker,
    "local": train_local,
}


# ── Main ─────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect RL environment data and fine-tune a model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--envs", default="bug-hunter",
                    help='Comma-separated slugs or "all" (default: bug-hunter)')
    p.add_argument("--model", default="Qwen/Qwen3-8B",
                    help="Model ID to fine-tune (default: Qwen/Qwen3-8B)")
    p.add_argument("--algorithm", choices=FORMATTERS.keys(), default="sft",
                    help="Training algorithm (default: sft)")
    p.add_argument("--provider", choices=TRAINERS.keys(), default="together",
                    help="Training provider (default: together)")
    p.add_argument("--sessions-per-env", type=int, default=10,
                    help="Sessions to collect per environment (default: 10)")
    p.add_argument("--output-dir", default="output",
                    help="Directory for training data and models (default: output)")
    p.add_argument("--collect-only", action="store_true",
                    help="Collect data but skip fine-tuning")
    return p.parse_args()


def main():
    args = parse_args()
    client = Epsilab(load_dotenv=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # ── 0. Resolve environments ──────────────────────────────────
    print("\n0. Resolving environments ...")
    environments = resolve_environments(client, args.envs)
    print(f"   {len(environments)} environment(s): {', '.join(e['slug'] for e in environments)}")

    print("\n" + "=" * 60)
    print(f"  Environments: {len(environments)}")
    print(f"  Model:        {args.model}")
    print(f"  Algorithm:    {args.algorithm.upper()}")
    print(f"  Provider:     {args.provider}")
    print("=" * 60)

    # ── 1. Collect training data ─────────────────────────────────
    print("\n1. Collecting training data ...")
    records = collect_training_data(client, environments, sessions_per_env=args.sessions_per_env)

    if not records:
        print("  No training data collected.")
        client.close()
        return

    print(f"\n   {len(records)} examples from {len(set(r['env'] for r in records))} environment(s)")
    print(f"   Rewards: min={min(r['reward'] for r in records):.3f}, "
          f"max={max(r['reward'] for r in records):.3f}, "
          f"mean={sum(r['reward'] for r in records) / len(records):.3f}")

    # ── 2. Format for algorithm ──────────────────────────────────
    print(f"\n2. Formatting for {args.algorithm.upper()} ...")
    formatted = FORMATTERS[args.algorithm](records)
    print(f"   {len(formatted)} training examples")

    data_path = output_dir / f"training_{args.algorithm}.jsonl"
    with open(data_path, "w") as f:
        for row in formatted:
            f.write(json.dumps(row) + "\n")
    print(f"   Saved to {data_path}")

    if args.collect_only:
        print(f"\n   --collect-only: skipping fine-tuning")
        client.close()
        return

    # ── 3. Fine-tune ─────────────────────────────────────────────
    print(f"\n3. Fine-tuning via {args.provider} ...")
    try:
        TRAINERS[args.provider](args.model, formatted, args.algorithm)
    except ImportError as e:
        deps = {
            "together": "pip install epsilab[training]",
            "fireworks": "pip install epsilab[fireworks]",
            "tinker": "pip install epsilab[tinker]",
            "local": "pip install epsilab[training]",
        }
        print(f"\n  Missing: {e}")
        print(f"  Install with: {deps[args.provider]}")

    print("\n" + "=" * 60)
    print(f"  Environments: {', '.join(e['slug'] for e in environments)}")
    print(f"  Model:        {args.model}")
    print(f"  Algorithm:    {args.algorithm.upper()}")
    print(f"  Data:         {data_path} ({len(formatted)} examples)")
    print("=" * 60)

    client.close()


if __name__ == "__main__":
    main()
