from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))

from _environment_utils import task_ids_for_environment, terminal_reward  # noqa: E402
from grpo_training import score_completions  # noqa: E402


class _TaskClient:
    def iter_tasks(self, **_: object):
        yield {"task_id": "form-contact-001", "capability": "form-filler"}
        yield {"task_id": "form-filler-hard-train-002", "capability": "other"}
        yield {"task_id": "unrelated-task", "capability": "other"}


def test_task_discovery_accepts_capability_and_slug_prefix() -> None:
    assert task_ids_for_environment(_TaskClient(), "form-filler", limit=5) == [
        "form-filler-hard-train-002",
        "form-contact-001",
    ]


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (SimpleNamespace(done=False, reward=None), "did not return a terminal reward"),
        (SimpleNamespace(done=True, reward=float("nan")), "returned a non-finite reward"),
    ],
)
def test_terminal_reward_rejects_invalid_training_signals(
    result: SimpleNamespace,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        terminal_reward(result, context="Example")


def test_terminal_reward_preserves_valid_zero() -> None:
    assert terminal_reward(SimpleNamespace(done=True, reward=0.0), context="Example") == 0.0


class _FailingScoringClient:
    def create_environment_session(self, *_: object, **__: object):
        raise ConnectionError("provider unavailable")


def test_grpo_stops_on_scoring_infrastructure_failure() -> None:
    with pytest.raises(RuntimeError, match="training was stopped") as exc_info:
        score_completions(
            _FailingScoringClient(),
            "deployment-id",
            "task-id",
            ["prompt"],
            ["completion"],
        )

    assert isinstance(exc_info.value.__cause__, ConnectionError)


def test_grpo_rejects_mismatched_prompt_and_completion_counts() -> None:
    with pytest.raises(ValueError, match=r"zip\(\) argument 2 is shorter"):
        score_completions(
            _FailingScoringClient(),
            "deployment-id",
            "task-id",
            ["prompt"],
            [],
        )
