"""Example: use Epsilab environments as live reward functions for GRPO training.

This shows how to integrate any Epsilab environment into a TRL GRPO
training loop. The environment provides real-time rewards instead of
a static reward model.

Usage:
    pip install epsilab trl transformers
    export EPSILAB_API_KEY=sk-...
    python examples/grpo_training.py

What this does:
    1. Connects to an Epsilab environment (e.g. bug-hunter, code-review)
    2. Defines a reward function that submits completions to the environment
    3. Shows how to wire it into TRL's GRPOTrainer
"""

from __future__ import annotations

from epsilab import Epsilab


def make_reward_fn(client: Epsilab, deployment_id: str):
    """Create a reward function backed by an Epsilab environment.

    The returned function has the signature TRL expects:
        reward_fn(completions, task_ids, **kwargs) -> list[float]
    """

    def reward_fn(completions: list[str], task_ids: list[str], **kwargs) -> list[float]:
        rewards = []
        for completion, task_id in zip(completions, task_ids):
            session = client.create_environment_session(
                deployment_id,
                task_id=task_id,
                seed=kwargs.get("seed"),
            )
            result = client.environment_step(
                session.session_id,
                completion,
                session_token=session.session_token,
            )
            rewards.append(result.reward or 0.0)
        return rewards

    return reward_fn


def main():
    client = Epsilab(load_dotenv=True)

    print("Searching for coding environments...")
    results = client.search_environments(domain="coding", limit=5)
    for env in results:
        print(f"  {env.get('slug', '?'):30s}  {env.get('title', '?')}")

    if not results:
        print("No environments found. Deploy one first:")
        print("  cd env-catalog/code/bug-hunter && epsilab deploy")
        return

    # For a real training run, you'd use a deployment ID from `epsilab env list`
    # and wire the reward function into TRL:
    #
    #   from trl import GRPOTrainer, GRPOConfig
    #
    #   reward_fn = make_reward_fn(client, deployment_id="<your-deployment-id>")
    #
    #   trainer = GRPOTrainer(
    #       model=model,
    #       args=GRPOConfig(output_dir="output"),
    #       train_dataset=dataset,
    #       reward_funcs=[reward_fn],
    #   )
    #   trainer.train()

    print("\nTo use in a GRPO training loop:")
    print("  1. Deploy an environment:  epsilab deploy")
    print("  2. Note the deployment ID from the output")
    print("  3. Use make_reward_fn(client, deployment_id) as your reward function")
    print("  4. Pass it to TRL's GRPOTrainer.reward_funcs")

    client.close()


if __name__ == "__main__":
    main()
