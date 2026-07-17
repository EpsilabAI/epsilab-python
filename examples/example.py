"""Quick start: interact with the Epsilab RL Environment Hub.

Lists available environments, runs a single session, and shows
the step result. This is the simplest possible example.

Usage:
    pip install epsilab
    epsilab login
    python examples/example.py
"""

from __future__ import annotations

import argparse

from epsilab import Epsilab


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick start: run one environment session.")
    p.add_argument("--env", default=None,
                    help="Environment slug (default: first available)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    client = Epsilab(load_dotenv=True)

    # ── Discover environments ────────────────────────────────────
    print("Discovering environments ...")
    listings = client.list_environment_listings(limit=20)
    deployed = [l for l in listings if l.deployment_id]
    print(f"  {len(deployed)} environments with active deployments:\n")

    for l in deployed[:10]:
        print(f"    {l.slug:30s}  {l.title}")
    if len(deployed) > 10:
        print(f"    ... and {len(deployed) - 10} more")

    if not deployed:
        print("  No environments available.")
        client.close()
        return

    # ── Pick an environment ──────────────────────────────────────
    if args.env:
        match = [l for l in deployed if l.slug == args.env]
        if not match:
            print(f"\n  Environment '{args.env}' not found.")
            client.close()
            return
        listing = match[0]
    else:
        listing = deployed[0]

    dep_id = listing.deployment_id
    task_id = f"{listing.slug}-train-easy-001"
    print(f"\n  Running: {listing.title} ({listing.slug})")
    print(f"  Deployment: {dep_id[:12]}...")

    # ── Create a session ─────────────────────────────────────────
    print("\nCreating session ...")
    session = client.create_environment_session(dep_id, task_id=task_id)
    print(f"  Session: {session.session_id[:12]}...")
    session = client.wait_for_session(session)
    print(f"  Status:  {session.status}")
    print(f"  Observation: {str(session.observation or '')[:200]}")

    if session.is_terminal:
        print(f"\n  Session ended early: {session.status}")
        client.close()
        return

    # ── Take a step ──────────────────────────────────────────────
    print("\nTaking a step ...")
    action = "I'll analyze this step by step and provide a solution."
    result = client.environment_step(
        session.session_id, action, session_token=session.session_token,
    )
    print(f"  Reward:     {result.reward}")
    print(f"  Terminated: {result.terminated}")
    print(f"  Truncated:  {result.truncated}")
    print(f"  Done:       {result.done}")

    # ── Credits ──────────────────────────────────────────────────
    try:
        balance = client.get_credit_balance()
        print(f"\nCredits remaining: {balance.get('balance', '?')}")
    except Exception:
        pass

    client.close()


if __name__ == "__main__":
    main()
