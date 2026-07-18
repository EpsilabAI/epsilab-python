"""Shared environment discovery helpers for executable examples."""

from __future__ import annotations

import math

from epsilab import Epsilab


def resolve_environments(client: Epsilab, specification: str) -> list[dict[str, str]]:
    """Resolve a comma-separated slug list against active hub deployments."""
    listings = client.list_environment_listings(limit=100)
    available = [listing for listing in listings if listing.deployment_id]
    if specification == "all":
        selected = available
    else:
        requested = [slug.strip() for slug in specification.split(",") if slug.strip()]
        by_slug = {listing.slug: listing for listing in available}
        missing = [slug for slug in requested if slug not in by_slug]
        if missing:
            choices = ", ".join(sorted(by_slug)[:15])
            raise SystemExit(
                f"Environment '{missing[0]}' was not found or is not deployed.\n"
                f"Available: {choices}"
            )
        selected = [by_slug[slug] for slug in requested]
    return [
        {
            "slug": listing.slug,
            "deployment_id": str(listing.deployment_id),
        }
        for listing in selected
    ]


def task_ids_for_environment(
    client: Epsilab,
    slug: str,
    *,
    limit: int = 1,
) -> list[str]:
    """Return real task IDs for an environment, preferring its train split."""
    candidates = {
        str(task["task_id"])
        for task in client.iter_tasks(source="custom", page_size=100)
        if isinstance(task.get("task_id"), str)
        and (
            str(task["task_id"]).startswith(f"{slug}-")
            or task.get("capability") == slug
        )
    }
    ordered = sorted(candidates, key=lambda task_id: ("-train-" not in task_id, task_id))
    if ordered:
        return ordered[: max(1, limit)]
    return [f"{slug}-easy-train-001"]


def submission(content: str) -> dict[str, str]:
    """Build the structured text action used by first-party catalog environments."""
    return {"content": content, "action_type": "submit"}


def terminal_reward(result: object, *, context: str) -> float:
    """Return a finite terminal reward without fabricating failed outcomes."""
    reward = getattr(result, "reward", None)
    if not getattr(result, "done", False) or reward is None:
        raise RuntimeError(f"{context} did not return a terminal reward")

    value = float(reward)
    if not math.isfinite(value):
        raise RuntimeError(f"{context} returned a non-finite reward")
    return value
