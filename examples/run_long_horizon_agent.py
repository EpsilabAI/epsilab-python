"""Provider-neutral skeleton for a traced, long-horizon coding rollout.

Set ``EPSILAB_DEPLOYMENT_ID`` and ``EPSILAB_TASK_ID``, then implement
``call_model`` for the model API you want to use. The runner intentionally
does not choose a provider or impose a token budget.
"""

from __future__ import annotations

import os
import threading

from epsilab import AgentRunContext, AgentTurn, Epsilab


def call_model(context: AgentRunContext) -> AgentTurn:
    """Translate one provider response into an ``AgentTurn``.

    A real adapter should pass ``context.history`` and ``context.observation``
    to the provider, then return reasoning, messages, native tool calls, usage,
    model identity, and the provider request ID. Reasoning-only responses are
    valid: return ``AgentTurn(reasoning=...)`` with no tool calls.
    """
    raise NotImplementedError("Connect your model provider in call_model()")


def main() -> None:
    deployment_id = os.environ["EPSILAB_DEPLOYMENT_ID"]
    task_id = os.environ["EPSILAB_TASK_ID"]
    cancel_event = threading.Event()
    client = Epsilab()
    try:
        result = client.run_agent_episode(
            deployment_id,
            task_id=task_id,
            model_fn=call_model,
            max_turns=500,
            cancel_check=cancel_event.is_set,
            agent_id="custom-long-horizon-runner",
        )
        print(
            f"stop={result.stop_reason} turns={result.turns_completed} "
            f"steps={result.environment_steps} reward={result.session.total_reward}"
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
