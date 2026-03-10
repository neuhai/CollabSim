"""Action schema definitions and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ActionValidationError(Exception):
    """Raised when an action payload fails validation."""

    message: str

    def __str__(self) -> str:
        return self.message


ACTION_TYPES: tuple[str, ...] = (
    "communicate",
    "decide",
    "propose",
    "respond",
    "transfer",
    "do_nothing",
)


def validate_action(action: Mapping[str, Any]) -> None:
    """Validate a single action envelope and payload.

    Args:
        action: Action envelope dictionary.

    Raises:
        ActionValidationError: If validation fails.
    """

    if not isinstance(action, Mapping):
        raise ActionValidationError("Action must be a mapping/object.")

    for key in ("type", "actor_id", "timestamp", "payload"):
        if key not in action:
            raise ActionValidationError(f"Action missing required field: {key}")

    action_type = action["type"]
    if action_type not in ACTION_TYPES:
        raise ActionValidationError(f"Unsupported action type: {action_type}")

    payload = action["payload"]
    if not isinstance(payload, Mapping):
        raise ActionValidationError("Action payload must be a mapping/object.")

    if action_type == "communicate":
        _validate_communicate(payload)
    elif action_type == "decide":
        _validate_decide(payload)
    elif action_type == "propose":
        _validate_propose(payload)
    elif action_type == "respond":
        _validate_respond(payload)
    elif action_type == "transfer":
        _validate_transfer(payload)
    elif action_type == "do_nothing":
        _validate_do_nothing(payload)


def _validate_communicate(payload: Mapping[str, Any]) -> None:
    _require_fields(payload, ("channel", "content", "content_type"))
    channel = payload["channel"]
    if channel not in ("broadcast", "direct"):
        raise ActionValidationError("communicate.channel must be broadcast or direct.")
    if channel == "direct":
        recipients = payload.get("recipients")
        if not isinstance(recipients, list) or not recipients:
            raise ActionValidationError("communicate.recipients must be a non-empty list.")
    if payload["content_type"] not in ("text", "json"):
        raise ActionValidationError("communicate.content_type must be text or json.")


def _validate_decide(payload: Mapping[str, Any]) -> None:
    _require_fields(payload, ("decision_id", "choice", "reveal"))
    if payload["reveal"] not in ("sequential", "aggregated", "simultaneous"):
        raise ActionValidationError("decide.reveal must be sequential, aggregated, or simultaneous.")


def _validate_propose(payload: Mapping[str, Any]) -> None:
    _require_fields(payload, ("proposal_id", "target_ids", "terms"))
    target_ids = payload["target_ids"]
    if not isinstance(target_ids, list) or not target_ids:
        raise ActionValidationError("propose.target_ids must be a non-empty list.")
    if not isinstance(payload["terms"], Mapping):
        raise ActionValidationError("propose.terms must be an object.")


def _validate_respond(payload: Mapping[str, Any]) -> None:
    _require_fields(payload, ("proposal_id", "response"))
    response = payload["response"]
    if response not in ("accept", "reject", "counter"):
        raise ActionValidationError("respond.response must be accept, reject, or counter.")
    if response == "counter":
        if "counter_terms" not in payload or not isinstance(payload["counter_terms"], Mapping):
            raise ActionValidationError("respond.counter_terms required for counter response.")


def _validate_transfer(payload: Mapping[str, Any]) -> None:
    _require_fields(payload, ("resource_id", "amount", "to"))
    amount = payload["amount"]
    if not isinstance(amount, (int, float)) or amount <= 0:
        raise ActionValidationError("transfer.amount must be a positive number.")


def _validate_do_nothing(payload: Mapping[str, Any]) -> None:
    if "reason" in payload and not isinstance(payload.get("reason"), str):
        raise ActionValidationError("do_nothing.reason must be a string when provided.")


def _require_fields(payload: Mapping[str, Any], fields: tuple[str, ...]) -> None:
    for field in fields:
        if field not in payload:
            raise ActionValidationError(f"Payload missing required field: {field}")
