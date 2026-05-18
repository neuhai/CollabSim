"""Agent interface contract for reproducible experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class AgentMetadata:
    """Metadata describing an agent's role and model configuration."""

    agent_id: str
    role: str
    model_provider: str
    model_name: str
    model_version: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    prompt_version: str | None = None


@dataclass(frozen=True)
class Observation:
    """Observation visible to an agent at a given step."""

    state: dict[str, Any]
    visible_events: list[dict[str, Any]]
    step_index: int
    memory: dict[str, Any] | None = None
    game_status: dict[str, Any] | None = None


@dataclass(frozen=True)
class ActionProposal:
    """Agent-proposed action and optional metadata."""

    action: dict[str, Any]
    rationale: str | None = None
    alternatives: list[dict[str, Any]] | None = None
    prompt_text: str | None = None
    raw_response: str | None = None
    prompt_static: str | None = None
    prompt_update: dict[str, Any] | None = None


class AgentInterface(Protocol):
    """Protocol for agent implementations."""

    metadata: AgentMetadata

    def context_update(self, observation: Observation) -> ActionProposal:
        """Return an action proposal given the current context update."""

    def reset(self, seed: int | None = None) -> None:
        """Reset internal state for reproducible experiments."""

    def serialize(self) -> dict[str, Any]:
        """Return JSON-serializable internal state."""

    def load(self, state: dict[str, Any]) -> None:
        """Restore internal state from serialized data."""
