"""Example: run an Epsilab environment end-to-end.

Demonstrates the full lifecycle: browse → pick an environment →
create sessions → step through tasks → inspect trajectories →
export training data.

Usage:
    pip install epsilab
    export EPSILAB_API_KEY=sk-...
    python examples/run_environment.py
"""

from __future__ import annotations

from epsilab import Epsilab


def run_single_episode(client: Epsilab, deployment_id: str, task_id: str) -> dict:
    """Run one episode: reset → step until terminal → return trajectory."""
    session = client.create_environment_session(
        deployment_id,
        task_id=task_id,
        seed=42,
    )
    print(f"\n  Session: {session.session_id}")
    print(f"  Task:    {task_id}")
    print(f"  Initial observation: {str(session.observation)[:200]}")

    steps = 0
    total_reward = 0.0
    done = False

    while not done and steps < 20:
        action = f"Attempt to solve step {steps + 1}"
        result = client.environment_step(
            session.session_id,
            action,
            session_token=session.session_token,
        )
        steps += 1
        total_reward += result.reward or 0.0
        done = result.terminated or result.truncated

        print(f"  Step {steps}: reward={result.reward:.3f}  "
              f"terminated={result.terminated}  truncated={result.truncated}")
        if result.observation:
            print(f"           obs: {str(result.observation)[:150]}")

    print(f"  Episode complete: {steps} steps, total reward={total_reward:.3f}")
    return {"session_id": session.session_id, "steps": steps, "reward": total_reward}


def main():
    client = Epsilab(load_dotenv=True)

    # ── 1. Browse environments ───────────────────────────────────
    print("═" * 60)
    print("Browsing the Epsilab Environment Hub")
    print("═" * 60)

    listings = client.list_environment_listings(limit=10)
    print(f"\nFound {len(listings)} environments:")
    for listing in listings[:10]:
        print(f"  {listing.slug:30s}  {listing.title}  ({listing.visibility})")

    # ── 2. Search by domain ──────────────────────────────────────
    print("\n── Searching for coding environments ──")
    coding = client.search_environments(domain="coding", limit=5)
    print(f"Found {len(coding)} coding environments")

    # ── 3. Run episodes ──────────────────────────────────────────
    # Replace with a real deployment ID to run actual episodes:
    #
    #   episodes = []
    #   for task in ["bug-hunter-easy-001", "bug-hunter-medium-001"]:
    #       result = run_single_episode(client, "<deployment-id>", task)
    #       episodes.append(result)
    #
    #   print(f"\nCompleted {len(episodes)} episodes")
    #   avg_reward = sum(e["reward"] for e in episodes) / len(episodes)
    #   print(f"Average reward: {avg_reward:.3f}")

    # ── 4. List past sessions ────────────────────────────────────
    print("\n── Your recent sessions ──")
    sessions = client.list_rl_sessions(limit=5)
    for s in sessions:
        sid = s.get("session_id", "?")
        task = s.get("task_id", "?")
        status = s.get("status", "?")
        reward = s.get("total_reward")
        reward_str = f"{reward:.3f}" if reward is not None else "-"
        print(f"  {sid[:12]}  {task[:35]:35s}  {status:10s}  reward={reward_str}")

    # ── 5. Inspect a trajectory ──────────────────────────────────
    completed = [s for s in sessions if s.get("status") == "completed"]
    if completed:
        sid = completed[0]["session_id"]
        print(f"\n── Trajectory for {sid[:12]} ──")
        trajectory = client.get_rl_trajectory(sid)
        for step in trajectory.get("steps", []):
            r = step.get("reward", 0)
            obs = str(step.get("observation", ""))[:100]
            print(f"  Step {step.get('step_idx', '?')}: reward={r:.3f}  obs={obs}")

    # ── 6. Export training data ──────────────────────────────────
    # Uncomment to export from a deployment:
    #
    #   export = client.create_environment_export(
    #       deployment_id="<deployment-id>",
    #       format="grpo",
    #   )
    #   print(f"\nExport started: {export.get('export_id')}")
    #   print(f"  Format: {export.get('format')}")
    #   print(f"  Status: {export.get('status')}")

    print("\n✓ Done")
    client.close()


if __name__ == "__main__":
    main()
