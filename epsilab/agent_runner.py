"""Provider-neutral long-horizon agent loop for hosted environments."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from .exceptions import AuthError
from .models import (
    AgentEpisodeResult,
    AgentRunContext,
    AgentToolCall,
    AgentTurn,
    AgentUsage,
    EnvironmentSession,
    EnvironmentStepResult,
)

if TYPE_CHECKING:
    from .client import EpsilabClient

logger = logging.getLogger("epsilab.agent_runner")

_MAX_RUNNER_TURNS = 500
_TRACE_TEXT_CHARS = 50_000
_TRACE_JSON_BYTES = 200_000


class LongHorizonAgentRunner:
    """Drive model turns independently from environment actions."""

    def __init__(self, client: "EpsilabClient") -> None:
        self._client = client
        self._token = ""

    def run(
        self,
        session: EnvironmentSession,
        *,
        model_fn: Callable[[AgentRunContext], AgentTurn],
        max_turns: int = _MAX_RUNNER_TURNS,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> AgentEpisodeResult:
        """Run up to ``max_turns`` model calls with no SDK token budget."""
        if isinstance(max_turns, bool) or not isinstance(max_turns, int) or not 1 <= max_turns <= _MAX_RUNNER_TURNS:
            raise ValueError(f"max_turns must be between 1 and {_MAX_RUNNER_TURNS}")
        session = self._client.wait_for_session(session)
        if session.is_terminal:
            return AgentEpisodeResult(
                session=session,
                stop_reason="session_terminal",
                turns_completed=0,
                environment_steps=session.steps_taken,
            )
        if not session.is_active:
            raise RuntimeError(
                f"Session {session.session_id} cannot be run while status is {session.status!r}"
            )
        self._token = self._resolve_session_token(session)

        observation = session.observation or ""
        history: List[Dict[str, Any]] = []
        environment_steps = session.steps_taken
        turns_completed = 0
        total_input_tokens = 0
        total_output_tokens = 0
        stage = "starting"
        try:
            self._trace(
                session,
                "lifecycle",
                {"state": "started", "max_turns": max_turns, "environment_steps": environment_steps},
            )
            for turn_index in range(max_turns):
                if self._cancelled(cancel_check):
                    return self._cancel_result(
                        session,
                        reason="cancelled",
                        turns_completed=turns_completed,
                        environment_steps=environment_steps,
                        input_tokens=total_input_tokens,
                        output_tokens=total_output_tokens,
                        history=history,
                    )

                stage = "model_call"
                self._trace(
                    session,
                    "model_request",
                    {
                        "turn": turn_index,
                        "environment_steps": environment_steps,
                        "observation_digest": _text_digest(observation),
                        "observation_bytes": len(observation.encode("utf-8")),
                    },
                )
                context = AgentRunContext(
                    session_id=session.session_id,
                    task_id=session.task_id,
                    turn_index=turn_index,
                    environment_steps=environment_steps,
                    observation=observation,
                    history=[dict(item) for item in history],
                )
                turn = model_fn(context)
                _validate_turn(turn)
                turns_completed = turn_index + 1
                if turn.usage is not None:
                    total_input_tokens += turn.usage.input_tokens or 0
                    total_output_tokens += turn.usage.output_tokens or 0

                stage = "trace_model_output"
                common = {
                    "turn": turn_index,
                    "provider": turn.provider,
                    "model": turn.model,
                    "provider_request_id": turn.provider_request_id,
                }
                if turn.reasoning is not None:
                    self._trace_text(session, "reasoning", turn.reasoning, common)
                if turn.message is not None:
                    self._trace_text(session, "assistant_message", turn.message, common)
                if turn.usage is not None:
                    self._trace(session, "usage", {**common, **turn.usage.to_dict()})

                history.append(
                    {
                        "role": "assistant",
                        "turn": turn_index,
                        "reasoning": turn.reasoning,
                        "content": turn.message,
                        "tool_calls": [
                            {
                                "call_id": call.call_id,
                                "name": call.name,
                                "arguments": dict(call.arguments),
                            }
                            for call in turn.tool_calls
                        ],
                        "provider": turn.provider,
                        "model": turn.model,
                        "provider_request_id": turn.provider_request_id,
                        "metadata": dict(turn.metadata),
                    }
                )

                if self._cancelled(cancel_check):
                    return self._cancel_result(
                        session,
                        reason="cancelled",
                        turns_completed=turns_completed,
                        environment_steps=environment_steps,
                        input_tokens=total_input_tokens,
                        output_tokens=total_output_tokens,
                        history=history,
                    )

                for call in turn.tool_calls:
                    if self._cancelled(cancel_check):
                        return self._cancel_result(
                            session,
                            reason="cancelled",
                            turns_completed=turns_completed,
                            environment_steps=environment_steps,
                            input_tokens=total_input_tokens,
                            output_tokens=total_output_tokens,
                            history=history,
                        )
                    stage = "tool_call"
                    action = call.to_environment_action()
                    self._trace(session, "tool_call", _tool_call_trace(turn_index, call, action, turn))
                    result = self._step(session, action)
                    environment_steps += 1
                    observation = result.observation
                    self._trace_tool_result(session, turn_index, call, result)
                    history.append(
                        {
                            "role": "tool",
                            "turn": turn_index,
                            "call_id": call.call_id,
                            "name": call.name,
                            "content": result.observation,
                            "reward": result.reward,
                            "terminated": result.terminated,
                            "truncated": result.truncated,
                            "info": dict(result.info),
                        }
                    )
                    if result.done:
                        self._trace(
                            session,
                            "lifecycle",
                            {
                                "state": "environment_terminal",
                                "turns_completed": turns_completed,
                                "environment_steps": environment_steps,
                            },
                        )
                        final_session = self._client.get_environment_session(session.session_id)
                        return AgentEpisodeResult(
                            session=final_session,
                            stop_reason="environment_terminal",
                            turns_completed=turns_completed,
                            environment_steps=environment_steps,
                            input_tokens=total_input_tokens,
                            output_tokens=total_output_tokens,
                            history=history,
                        )

            return self._cancel_result(
                session,
                reason="max_turns",
                turns_completed=turns_completed,
                environment_steps=environment_steps,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                history=history,
            )
        except Exception as exc:
            try:
                self._trace(
                    session,
                    "error",
                    {"stage": stage, "error_type": type(exc).__name__, "turns_completed": turns_completed},
                )
            except Exception:
                logger.warning("Could not persist agent runner failure trace", exc_info=True)
            self._cleanup_session(session)
            raise

    def _resolve_session_token(self, session: EnvironmentSession) -> str:
        token = session.session_token
        if not token:
            token = self._client.refresh_session_token(session.session_id).get("session_token")
        if not isinstance(token, str) or not token:
            raise RuntimeError(f"Session {session.session_id} did not return a session credential")
        return token

    def _refresh_token(self, session: EnvironmentSession) -> None:
        token = self._client.refresh_session_token(session.session_id).get("session_token")
        if not isinstance(token, str) or not token:
            raise RuntimeError(f"Session {session.session_id} could not refresh its session credential")
        self._token = token

    def _trace(self, session: EnvironmentSession, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        idem_key = self._client._auto_idem_key()
        public_payload = {field: value for field, value in payload.items() if value is not None}
        try:
            return self._client.append_environment_trace_event(
                session.session_id,
                event_type=event_type,
                payload=public_payload,
                session_token=self._token,
                occurred_at=datetime.now(timezone.utc).isoformat(),
                idempotency_key=idem_key,
            )
        except AuthError:
            self._refresh_token(session)
            return self._client.append_environment_trace_event(
                session.session_id,
                event_type=event_type,
                payload=public_payload,
                session_token=self._token,
                occurred_at=datetime.now(timezone.utc).isoformat(),
                idempotency_key=idem_key,
            )

    def _trace_text(
        self,
        session: EnvironmentSession,
        event_type: str,
        content: str,
        common: Dict[str, Any],
    ) -> None:
        chunks = [content[index : index + _TRACE_TEXT_CHARS] for index in range(0, len(content), _TRACE_TEXT_CHARS)]
        if not chunks:
            chunks = [""]
        for index, chunk in enumerate(chunks):
            self._trace(
                session,
                event_type,
                {
                    **common,
                    "content": chunk,
                    "chunk_index": index,
                    "chunk_count": len(chunks),
                },
            )

    def _trace_tool_result(
        self,
        session: EnvironmentSession,
        turn_index: int,
        call: AgentToolCall,
        result: EnvironmentStepResult,
    ) -> None:
        common = {
            "turn": turn_index,
            "call_id": call.call_id,
            "name": call.name,
            "reward": result.reward,
            "terminated": result.terminated,
            "truncated": result.truncated,
        }
        self._trace_text(session, "tool_result", result.observation, common)

    def _step(self, session: EnvironmentSession, action: Dict[str, Any]) -> EnvironmentStepResult:
        key = self._client._auto_idem_key()
        try:
            return self._client.environment_step(
                session.session_id,
                action,
                session_token=self._token,
                idempotency_key=key,
            )
        except AuthError:
            self._refresh_token(session)
            return self._client.environment_step(
                session.session_id,
                action,
                session_token=self._token,
                idempotency_key=key,
            )

    def _cancel_result(
        self,
        session: EnvironmentSession,
        *,
        reason: str,
        turns_completed: int,
        environment_steps: int,
        input_tokens: int,
        output_tokens: int,
        history: List[Dict[str, Any]],
    ) -> AgentEpisodeResult:
        event_type = "cancellation" if reason == "cancelled" else "lifecycle"
        self._trace(
            session,
            event_type,
            {"state": reason, "turns_completed": turns_completed, "environment_steps": environment_steps},
        )
        self._client.cancel_environment_session(session.session_id)
        final_session = self._client.get_environment_session(session.session_id)
        return AgentEpisodeResult(
            session=final_session,
            stop_reason=reason,
            turns_completed=turns_completed,
            environment_steps=environment_steps,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            history=history,
        )

    def _cleanup_session(self, session: EnvironmentSession) -> None:
        try:
            current = self._client.get_environment_session(session.session_id)
            if not current.is_terminal:
                self._client.cancel_environment_session(session.session_id)
        except Exception:
            logger.warning(
                "Could not clean up environment session %s after agent runner failure",
                session.session_id,
                exc_info=True,
            )

    @staticmethod
    def _cancelled(cancel_check: Optional[Callable[[], bool]]) -> bool:
        return bool(cancel_check is not None and cancel_check())


def _validate_turn(turn: object) -> None:
    if not isinstance(turn, AgentTurn):
        raise TypeError("model_fn must return AgentTurn")
    call_ids = set()
    for call in turn.tool_calls:
        if not isinstance(call, AgentToolCall):
            raise TypeError("AgentTurn.tool_calls must contain AgentToolCall values")
        if not call.call_id or not call.name:
            raise ValueError("agent tool calls require non-empty call_id and name")
        if call.call_id in call_ids:
            raise ValueError("agent tool call IDs must be unique within a turn")
        call_ids.add(call.call_id)
        if not isinstance(call.arguments, dict):
            raise TypeError("agent tool call arguments must be a dictionary")
        _json_bytes(call.arguments)
    if turn.usage is not None:
        _validate_usage(turn.usage)
    _json_bytes(turn.metadata)


def _validate_usage(usage: AgentUsage) -> None:
    for name in ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens"):
        value = getattr(usage, name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(f"AgentUsage.{name} must be a non-negative integer")
    if usage.cost_usd is not None and (
        isinstance(usage.cost_usd, bool)
        or not isinstance(usage.cost_usd, (int, float))
        or not math.isfinite(float(usage.cost_usd))
        or usage.cost_usd < 0
    ):
        raise ValueError("AgentUsage.cost_usd must be a non-negative finite number")


def _tool_call_trace(
    turn_index: int,
    call: AgentToolCall,
    action: Dict[str, Any],
    turn: AgentTurn,
) -> Dict[str, Any]:
    encoded = _json_bytes(action)
    payload: Dict[str, Any] = {
        "turn": turn_index,
        "call_id": call.call_id,
        "name": call.name,
        "action_digest": f"sha256:{hashlib.sha256(encoded).hexdigest()}",
        "action_bytes": len(encoded),
        "provider": turn.provider,
        "model": turn.model,
        "provider_request_id": turn.provider_request_id,
    }
    if len(encoded) <= _TRACE_JSON_BYTES and not _contains_sensitive_key(call.arguments):
        payload["arguments"] = dict(call.arguments)
    else:
        payload["arguments_externalized_to_environment_step"] = True
    return payload


def _json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("agent turn metadata and arguments must be finite JSON") from exc


def _text_digest(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _contains_sensitive_key(value: object) -> bool:
    sensitive = {
        "apikey",
        "authorization",
        "accesstoken",
        "refreshtoken",
        "cookie",
        "password",
        "secret",
    }
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = "".join(character for character in str(key).lower() if character.isalnum())
            if normalized in sensitive or _contains_sensitive_key(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_sensitive_key(item) for item in value)
    return False
