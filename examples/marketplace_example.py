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

from _environment_utils import submission, task_ids_for_environment


def consumer_workflow(client: Epsilab) -> None:
    """Browse the hub, run a session, and inspect listings."""

    # ── Browse the catalog ───────────────────────────────────────
    print("\n-- Environment Hub --")
    listings = client.list_environment_listings(limit=10)
    deployed = [listing for listing in listings if listing.deployment_id]
    print(f"  {len(deployed)} deployed environments:")
    for listing in deployed[:5]:
        stars = f"  [{listing.star_count} stars]" if listing.star_count else ""
        print(f"    {listing.slug:30s}  {listing.title}{stars}")

    # ── Application tools ────────────────────────────────────────
    print("\n-- Application Tools --")
    tools = client.list_application_tools(limit=5)
    print(f"  {len(tools)} tools:")
    for t in tools[:5]:
        print(f"    {t.slug}")

    # ── Run a quick session ──────────────────────────────────────
    if deployed:
        listing = deployed[0]
        dep_id = listing.deployment_id
        task_id = task_ids_for_environment(client, listing.slug)[0]
        print(f"\n-- Quick session: {listing.slug} --")
        try:
            session = client.create_environment_session(dep_id, task_id=task_id)
            session = client.wait_for_session(session)
            result = client.environment_step(
                session.session_id,
                submission("Analyzing the problem..."),
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

    print("\n-- Your Listings --")
    listings = client.list_environment_listings(limit=10)
    owned = [listing for listing in listings if listing.is_owner]
    print(f"  {len(owned)} owned listing(s)")
    for listing in owned[:5]:
        print(
            f"    {listing.slug:30s}  visibility={listing.visibility}  "
            f"moderation={listing.moderation_state}"
        )


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
