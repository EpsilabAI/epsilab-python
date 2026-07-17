"""Use Epsilab environments as live reward functions for GRPO training.

Instead of training a static reward model, environments provide
real-time, verifiable rewards. Each candidate completion is submitted
to a sandboxed environment that returns a scalar reward based on
actual execution (e.g. tests pass, code compiles, answer is correct).

Usage:
    pip install epsilab trl transformers datasets
    export EPSILAB_API_KEY=sk-...
    python examples/grpo_training.py

What this does:
    1. Searches the hub for a deployed environment
    2. Builds a reward function backed by that environment
    3. Wires it into TRL's GRPOTrainer for live training
"""

from __future__ import annotations

from epsilab import Epsilab


def make_reward_fn(client: Epsilab, deployment_id: str):
    """Create a reward function backed by an Epsilab environment.

    The returned function has the signature TRL expects:
        reward_fn(completions, prompts, **kwargs) -> list[float]

    Each completion is submitted to the environment as an action.
    The environment scores it (runs tests, checks correctness, etc.)
    and returns a reward.
    """

    def reward_fn(
        completions: list[str],
        prompts: list[str],
        **kwargs,
    ) -> list[float]:
        task_ids = kwargs.get("task_ids", [None] * len(completions))
        rewards = []
        for completion, task_id in zip(completions, task_ids):
            try:
                session = client.create_environment_session(
                    deployment_id,
                    task_id=task_id or "default",
                    seed=kwargs.get("seed"),
                )
                result = client.environment_step(
                    session.session_id,
                    completion,
                    session_token=session.session_token,
                )
                rewards.append(result.reward or 0.0)
            except Exception:
                rewards.append(0.0)
        return rewards

    return reward_fn


def find_deployment(client: Epsilab, domain: str = "coding") -> str | None:
    """Find a deployed environment for the given domain."""
    listings = client.search_environments(domain=domain, limit=20)
    for env in listings:
        dep_id = env.get("deployment_id") if isinstance(env, dict) else getattr(env, "deployment_id", None)
        if dep_id:
            title = env.get("title") if isinstance(env, dict) else getattr(env, "title", "?")
            slug = env.get("slug") if isinstance(env, dict) else getattr(env, "slug", "?")
            print(f"  Using: {title} ({slug})")
            return dep_id
    return None


def main():
    client = Epsilab(load_dotenv=True)

    print("Searching for a deployed coding environment...")
    deployment_id = find_deployment(client, domain="coding")

    if not deployment_id:
        print("No deployed environments found.")
        print("Deploy one first:  epsilab env init my-env && cd my-env && epsilab deploy")
        print("Or run:  python examples/run_environment.py  for the full workflow")
        client.close()
        return

    reward_fn = make_reward_fn(client, deployment_id)

    print("\nTesting reward function with a sample completion...")
    test_rewards = reward_fn(
        completions=["def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"],
        prompts=["Write a fibonacci function"],
        task_ids=["task-001"],
    )
    print(f"  Reward: {test_rewards[0]:.3f}")

    # ── Wire into TRL ────────────────────────────────────────────
    try:
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError:
        print("\nTo run training, install:  pip install trl transformers datasets torch")
        print("\nOnce installed, the integration is:")
        print("  reward_fn = make_reward_fn(client, deployment_id)")
        print("  trainer = GRPOTrainer(model=model, reward_funcs=[reward_fn], ...)")
        print("  trainer.train()")
        client.close()
        return

    model_name = "Qwen/Qwen3-0.6B"
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompts = [
        "Write a function that reverses a string",
        "Write a function that checks if a number is prime",
        "Write a function that computes factorial",
    ]
    dataset = Dataset.from_dict({"prompt": prompts})

    config = GRPOConfig(
        output_dir="output/grpo-live-reward",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        logging_steps=1,
        max_steps=10,
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
        processing_class=tokenizer,
    )

    print("Starting GRPO training with live environment rewards...")
    trainer.train()
    trainer.save_model("output/grpo-live-reward")
    print(f"\nModel saved to output/grpo-live-reward/")

    client.close()


if __name__ == "__main__":
    main()
