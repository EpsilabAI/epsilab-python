"""Online GRPO: train a model with live environment rewards.

The model generates completions, Epsilab environments score them,
and GRPO updates the policy. Uses a remote training provider
(Tinker or Fireworks) so no local GPU is needed.

GRPO requires online interaction (sample -> score -> update), so it
can't use a managed fine-tuning API. For offline algorithms that
work with pre-collected data (SFT, DPO, KTO), see run_environment.py.

Usage:
    pip install epsilab[tinker]
    epsilab login
    export TINKER_API_KEY=...   # only if using --provider tinker

    # Single environment
    python examples/grpo_training.py --envs bug-hunter

    # Multiple environments (cycles through them each step)
    python examples/grpo_training.py --envs bug-hunter,refactor,test-writer

    # All environments
    python examples/grpo_training.py --envs all --provider local --steps 50
"""

from __future__ import annotations

import argparse

from epsilab import Epsilab

from _environment_utils import (
    resolve_environments,
    submission,
    task_ids_for_environment,
    terminal_reward,
)

# ── Scoring ──────────────────────────────────────────────────────


def score_completions(
    client: Epsilab,
    deployment_id: str,
    task_id: str,
    prompts: list[str],
    completions: list[str],
) -> list[float]:
    """Score completions by running them through an Epsilab environment."""
    rewards = []
    pairs = zip(prompts, completions, strict=True)
    for index, (_prompt, completion) in enumerate(pairs, start=1):
        try:
            session = client.create_environment_session(deployment_id, task_id=task_id)
            session = client.wait_for_session(session)
            result = client.environment_step(
                session.session_id,
                submission(completion),
                session_token=session.session_token,
            )
            rewards.append(
                terminal_reward(result, context=f"Completion {index}")
            )
        except Exception as exc:
            raise RuntimeError(
                f"Environment scoring failed for completion {index}; training was stopped"
            ) from exc
    return rewards


# ── Tinker backend ───────────────────────────────────────────────


def train_tinker(
    epsilab_client: Epsilab, model_id: str, prompts: list[str],
    environments: list[dict], n_steps: int,
) -> None:
    """Online GRPO using Tinker's training + sampling API."""
    import asyncio

    import numpy as np
    import tinker
    from tinker import types

    async def _train():
        client = tinker.ServiceClient()
        training_client = await client.create_lora_training_client_async(base_model=model_id, rank=16)
        sampling_client = await training_client.save_weights_and_get_sampling_client()
        num_generations = 4

        for step in range(n_steps):
            env = environments[step % len(environments)]
            prompt = prompts[step % len(prompts)]

            completions = []
            for _ in range(num_generations):
                response = await sampling_client.sample_async(prompt=prompt, max_tokens=256, temperature=0.8)
                completions.append(response.text)

            rewards = score_completions(
                epsilab_client, env["deployment_id"], env["task_id"],
                [prompt] * num_generations, completions,
            )

            mean_reward = np.mean(rewards)
            advantages = [(r - mean_reward) for r in rewards]
            best_idx = int(np.argmax(rewards))

            datum = tinker.tokenize_chat(
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completions[best_idx]},
                ],
                model=model_id,
            )
            weights = np.array(datum.loss_fn_inputs["weights"].data, dtype=np.float32)
            weights *= max(0, advantages[best_idx])
            datum.loss_fn_inputs["weights"] = tinker.NumpyData(weights)

            fb = await training_client.forward_backward_async([datum], "cross_entropy")
            optim = await training_client.optim_step_async(types.AdamParams(learning_rate=5e-5))
            fb_result = await fb.result_async()
            await optim.result_async()

            loss = fb_result.metrics.get("loss:mean", 0)
            print(
                f"  Step {step:2d} [{env['slug']}]: loss={loss:.4f}  "
                f"reward=[{', '.join(f'{r:.3f}' for r in rewards)}]  "
                f"mean={mean_reward:.3f}"
            )

            if (step + 1) % 5 == 0:
                sampling_client = await training_client.save_weights_and_get_sampling_client()

        await training_client.save_weights_and_get_sampling_client()
        print("\n  Training complete. Model checkpoint saved on Tinker.")

    asyncio.run(_train())


# ── Fireworks backend ────────────────────────────────────────────


def train_fireworks(
    epsilab_client: Epsilab, model_id: str, prompts: list[str],
    environments: list[dict], n_steps: int,
) -> None:
    """Online GRPO using Fireworks Training API."""
    import asyncio

    import numpy as np
    from fireworks.training.sdk import FiretitanServiceClient

    async def _train():
        fw_model = f"accounts/fireworks/models/{model_id.split('/')[-1].lower()}"
        service = FiretitanServiceClient.from_firetitan_config(
            api_key=None, base_model=fw_model, tokenizer_model=model_id,
            lora_rank=16, learning_rate=5e-5, cleanup_trainer_on_close=True,
        )
        training_client = service.create_training_client(base_model=fw_model, lora_rank=16)
        sampler = service.create_deployment_sampler()
        num_generations = 4

        for step in range(n_steps):
            env = environments[step % len(environments)]
            prompt = prompts[step % len(prompts)]

            completions = []
            for _ in range(num_generations):
                response = await sampler.sample_async(prompt=prompt, max_tokens=256)
                completions.append(response.text)

            rewards = score_completions(
                epsilab_client, env["deployment_id"], env["task_id"],
                [prompt] * num_generations, completions,
            )
            mean_reward = np.mean(rewards)
            best_idx = int(np.argmax(rewards))
            print(
                f"  Step {step:2d} [{env['slug']}]: "
                f"reward=[{', '.join(f'{r:.3f}' for r in rewards)}]  mean={mean_reward:.3f}"
            )

            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt},
                 {"role": "assistant", "content": completions[best_idx]}],
                tokenize=False,
            )
            import tinker
            datum = tinker.Datum.from_text(text, model=model_id)
            weights = np.array(datum.loss_fn_inputs["weights"].data, dtype=np.float32)
            weights *= max(0, rewards[best_idx] - mean_reward)
            datum.loss_fn_inputs["weights"] = tinker.NumpyData(weights)

            training_client.forward_backward_custom([datum], lambda d, lp: None).result()
            training_client.optim_step(tinker.AdamParams(learning_rate=5e-5)).result()

        print("\n  Training complete.")

    asyncio.run(_train())


# ── TRL local backend ───────────────────────────────────────────


def train_local(
    epsilab_client: Epsilab, model_id: str, prompts: list[str],
    environments: list[dict], n_steps: int,
) -> None:
    """Online GRPO using TRL's GRPOTrainer (requires local GPU)."""
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    dataset = Dataset.from_dict({"prompt": prompts})

    env = environments[0]

    def reward_fn(*, prompts, completions, **kwargs) -> list[float]:
        texts = [c if isinstance(c, str) else str(c) for c in completions]
        return score_completions(epsilab_client, env["deployment_id"], env["task_id"], prompts, texts)

    import torch
    use_gpu = torch.cuda.is_available()
    config = GRPOConfig(
        output_dir="output/grpo-finetuned",
        per_device_train_batch_size=1,
        num_generations=4,
        max_steps=n_steps,
        logging_steps=1,
        report_to="none",
        bf16=use_gpu,
        use_cpu=not use_gpu,
    )

    trainer = GRPOTrainer(model=model_id, args=config, train_dataset=dataset, reward_funcs=reward_fn)
    trainer.train()
    trainer.save_model("output/grpo-finetuned")
    print("  Saved to output/grpo-finetuned/")


# ── Main ─────────────────────────────────────────────────────────

TRAINERS = {
    "tinker": train_tinker,
    "fireworks": train_fireworks,
    "local": train_local,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Online GRPO training with live environment rewards.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--envs", default="bug-hunter",
                    help='Comma-separated slugs or "all" (default: bug-hunter)')
    p.add_argument("--model", default="Qwen/Qwen3-8B",
                    help="Model ID to train (default: Qwen/Qwen3-8B)")
    p.add_argument("--provider", choices=TRAINERS.keys(), default="tinker",
                    help="Training backend (default: tinker)")
    p.add_argument("--steps", type=int, default=20,
                    help="Number of GRPO training steps (default: 20)")
    return p.parse_args()


def main():
    args = parse_args()
    epsilab_client = Epsilab(load_dotenv=True)

    print("\n  Resolving environments ...")
    environments = resolve_environments(epsilab_client, args.envs)
    for environment in environments:
        environment["task_id"] = task_ids_for_environment(
            epsilab_client,
            environment["slug"],
        )[0]
    print(f"  {len(environments)} environment(s): {', '.join(e['slug'] for e in environments)}")

    print("=" * 60)
    print(f"  Environments: {', '.join(e['slug'] for e in environments)}")
    print(f"  Model:        {args.model}")
    print(f"  Provider:     {args.provider}")
    print(f"  Steps:        {args.steps}")
    print("=" * 60)

    prompts = [
        "Debug this function and explain the root cause.",
        "Write a comprehensive test suite for this module.",
        "Review this pull request for correctness and style issues.",
        "Refactor this code to improve readability and maintainability.",
        "Fix the failing CI pipeline and explain what went wrong.",
    ]

    print(f"\n  Training for {args.steps} steps across {len(environments)} env(s)")
    print("  Each step: sample 4 completions, score via environment, update\n")

    try:
        TRAINERS[args.provider](epsilab_client, args.model, prompts, environments, args.steps)
    except ImportError as e:
        deps = {
            "tinker": "pip install tinker tinker-cookbook",
            "fireworks": "pip install 'fireworks-ai[training]'",
            "local": "pip install trl transformers datasets torch",
        }
        print(f"\n  Missing: {e}")
        print(f"  Install with: {deps[args.provider]}")

    print("\n" + "=" * 60)
    epsilab_client.close()


if __name__ == "__main__":
    main()
