"""Interact with the Epsilab Environment Hub and Marketplace.

Demonstrates both consumer and publisher workflows:

    - **Consumer:** discover environments, run sessions, star favorites,
      view entitlements and purchases
    - **Publisher:** create a namespace, list environments, grant access

Usage:
    pip install epsilab
    epsilab login
    python examples/marketplace_example.py
"""

from __future__ import annotations

import argparse

from epsilab import Epsilab


def consumer_workflow(client: Epsilab) -> None:
    """Browse the hub, run a session, and inspect listings."""

    # ── Browse the catalog ───────────────────────────────────────
    print("\n-- Environment Hub --")
    listings = client.list_environment_listings(limit=10)
    deployed = [l for l in listings if l.deployment_id]
    print(f"  {len(deployed)} deployed environments:")
    for l in deployed[:5]:
        stars = f"  [{l.star_count} stars]" if hasattr(l, "star_count") and l.star_count else ""
        print(f"    {l.slug:30s}  {l.title}{stars}")

    # ── Application tools ────────────────────────────────────────
    print("\n-- Application Tools --")
    tools = client.list_application_tools(limit=5)
    print(f"  {len(tools)} tools:")
    for t in tools[:5]:
        name = t.get("name") or t.get("slug", "?")
        print(f"    {name}")

    # ── Run a quick session ──────────────────────────────────────
    if deployed:
        listing = deployed[0]
        dep_id = listing.deployment_id
        task_id = f"{listing.slug}-train-easy-001"
        print(f"\n-- Quick session: {listing.slug} --")
        try:
            session = client.create_environment_session(dep_id, task_id=task_id)
            session = client.wait_for_session(session)
            result = client.environment_step(
                    session.session_id, "Analyzing the problem...",
                    session_token=session.session_token,
                )
            print(f"  Reward: {result.reward}, Done: {result.done}")
        except Exception as e:
            print(f"  Session skipped: {e}")

    # ── Entitlements ─────────────────────────────────────────────
    print("\n-- Your Entitlements --")
    try:
        entitlements = client.list_entitlements(limit=10)
        print(f"  {len(entitlements)} entitlement(s)")
    except Exception as e:
        print(f"  {e}")

    # ── Billing ──────────────────────────────────────────────────
    print("\n-- Credits --")
    try:
        balance = client.get_credit_balance()
        print(f"  Balance: {balance.get('balance', '?')}")
    except Exception:
        print("  (credit balance not available)")


def publisher_workflow(client: Epsilab) -> None:
    """Show the publisher side: namespaces, listings, analytics."""

    print("\n-- Publisher Profile --")
    try:
        profile = client.get_creator_profile()
        print(f"  Profile: {profile.get('display_name', 'unnamed')}")
    except Exception:
        print("  No creator profile yet")

    print("\n-- Your Namespaces --")
    try:
        namespaces = client.list_namespaces()
        for ns in namespaces[:5]:
            slug = ns.get("slug", "?")
            ns_id = ns.get("namespace_id", "?")
            print(f"    {slug:20s}  {ns_id}")
    except Exception as e:
        print(f"  {e}")

    print("\n-- Your Listings --")
    listings = client.list_environment_listings(limit=10)
    owned = [l for l in listings if l.is_owner]
    print(f"  {len(owned)} owned listing(s)")
    for l in owned[:5]:
        print(f"    {l.slug:30s}  visibility={l.visibility}  moderation={l.moderation_state}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hub and marketplace workflow examples.")
    p.add_argument("--publisher", action="store_true",
                    help="Also run the publisher workflow")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    client = Epsilab(load_dotenv=True)

    print("=" * 60)
    print("  Epsilab Environment Hub & Marketplace")
    print("=" * 60)

    consumer_workflow(client)

    if args.publisher:
        publisher_workflow(client)
    else:
        print("\n  (Run with --publisher to see the publisher workflow)")

    print("\nDone")
    client.close()


if __name__ == "__main__":
    main()
