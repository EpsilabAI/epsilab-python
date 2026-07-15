"""Example: interact with the Epsilab Environment Hub & Marketplace.

Usage:
    1. Copy ``.env.example`` to ``.env`` and add your API key.
    2. Run ``python examples/marketplace_example.py``.

    Set ``EPSILAB_RUN_CREATOR_EXAMPLE=1`` to also run the creator workflow.

This example demonstrates both buyer and creator workflows:

    - **Buyer:** discover environments, run a session, inspect quality
    - **Creator:** register a namespace/listing/release, deploy, grant access
"""

from __future__ import annotations

import os

from epsilab import Epsilab


def buyer_workflow(client: Epsilab) -> None:
    """Discover environments and interact with a hosted session."""

    # ── Browse the public catalog ────────────────────────────────
    print("\n── Public Marketplace ──")
    listings = client.list_public_listings(sort_by="popular", limit=5)
    print(f"Found {len(listings)} public listings")
    for item in listings[:3]:
        print(f"  • {item.get('title', 'untitled')} ({item.get('listing_id', '?')})")

    # ── Quality-weighted search ──────────────────────────────────
    print("\n── Search for coding environments ──")
    results = client.search_environments(
        domain="coding",
        min_quality_score=0.7,
        limit=5,
    )
    print(f"Found {len(results)} quality coding environments")

    if not results:
        print("No environments available to demo — skipping session workflow")
        return

    # ── Inspect quality ──────────────────────────────────────────
    print("\n── Quality Information ──")
    badges = client.list_quality_badges(limit=5)
    print(f"Quality badges available: {len(badges)}")

    reports = client.list_quality_reports(report_type="qualification", limit=3)
    print(f"Qualification reports: {len(reports)}")

    # ── Check entitlements ───────────────────────────────────────
    print("\n── Your Entitlements ──")
    entitlements = client.list_entitlements(limit=10)
    print(f"You have {len(entitlements)} entitlements")

    # ── Billing overview ─────────────────────────────────────────
    print("\n── Billing ──")
    charges = client.list_session_charges(billable_only=True)
    print(f"Billable charges: {len(charges)}")

    summary = client.get_charge_summary()
    print(f"Charge summary: {summary}")


def creator_workflow(client: Epsilab) -> None:
    """Register a namespace, listing, and release (dry run)."""

    print("\n── Creator Profile ──")
    try:
        profile = client.get_creator_profile()
        print(f"Profile: {profile.get('display_name', 'unnamed')}")
    except Exception:
        print("No creator profile yet — creating one")
        profile = client.create_creator_profile(
            display_name="SDK Example Org",
            bio="Example creator from the Python SDK",
        )
        print(f"Created profile: {profile.get('display_name')}")

    print("\n── Creator Analytics ──")
    aggregates = client.get_creator_aggregates(limit=5)
    print(f"Releases with analytics: {len(aggregates)}")

    print("\n── Settlement ──")
    try:
        account = client.get_creator_account()
        print(f"Balance: {account.get('balance_cents', 0)} cents")
    except Exception as e:
        print(f"No settlement account: {e}")

    rules = client.list_royalty_rules()
    print(f"Royalty rules: {len(rules)}")

    accruals = client.list_accruals(status="pending")
    print(f"Pending accruals: {len(accruals)}")

    print("\n── Adapters ──")
    adapters = client.list_adapters(limit=5)
    print(f"Available adapters: {len(adapters)}")
    for adapter in adapters[:3]:
        print(f"  • {adapter.get('name', 'unnamed')} ({adapter.get('protocol_family', '?')})")


def main() -> None:
    client = Epsilab(load_dotenv=True)

    print("═" * 60)
    print("Epsilab Environment Hub & Marketplace — SDK Example")
    print("═" * 60)

    buyer_workflow(client)

    if os.environ.get("EPSILAB_RUN_CREATOR_EXAMPLE") == "1":
        creator_workflow(client)
    else:
        print("\n(Set EPSILAB_RUN_CREATOR_EXAMPLE=1 to run the creator workflow)")

    print("\n✓ Done")
    client.close()


if __name__ == "__main__":
    main()
