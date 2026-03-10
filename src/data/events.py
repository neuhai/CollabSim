"""Event taxonomy and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class EventValidationError(Exception):
    """Raised when an event fails validation."""

    message: str

    def __str__(self) -> str:
        return self.message


EVENT_TYPES: tuple[str, ...] = (
    "action_submitted",
    "action_validated",
    "action_rejected",
    "action_proposed",
    "observation_built",
    "state_updated",
    "message_delivered",
    "decision_buffered",
    "decision_revealed",
    "decision_timed_out",
    "proposal_created",
    "proposal_responded",
    "proposal_expired",
    "resource_transferred",
    "probe_asked",
    "probe_answered",
)

VISIBILITY_TYPES: tuple[str, ...] = ("public", "private", "system")


def validate_event(event: Mapping[str, Any]) -> None:
    """Validate a single event envelope."""

    if not isinstance(event, Mapping):
        raise EventValidationError("Event must be a mapping/object.")

    for key in ("event_id", "event_type", "timestamp", "actor_id", "visibility", "payload"):
        if key not in event:
            raise EventValidationError(f"Event missing required field: {key}")

    event_type = event["event_type"]
    if event_type not in EVENT_TYPES:
        raise EventValidationError(f"Unsupported event_type: {event_type}")

    visibility = event["visibility"]
    if visibility not in VISIBILITY_TYPES:
        raise EventValidationError("visibility must be public, private, or system.")

    if not isinstance(event["payload"], Mapping):
        raise EventValidationError("payload must be a mapping/object.")
