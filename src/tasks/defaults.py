"""Default task definitions."""

from __future__ import annotations

from typing import Any

from src.tasks.registry import TaskDefinition


def noop_init_state(config: dict[str, Any]) -> dict[str, Any]:
    """Return a minimal task state for placeholder runs."""

    _ = config
    return {"complete": False}


def noop_step(state: dict[str, Any]) -> dict[str, Any]:
    """No-op step that leaves state unchanged."""

    return state


NOOP_TASK = TaskDefinition(
    name="noop",
    init_state=noop_init_state,
    step=noop_step,
)
