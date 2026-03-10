"""Deterministic hidden-profile task scaffolding."""

from __future__ import annotations

from typing import Any

from src.tasks.registry import TaskDefinition


def hidden_profile_init_state(config: dict[str, Any]) -> dict[str, Any]:
    """Initialize a deterministic hidden-profile task state.

    Args:
        config: Experiment configuration containing task settings.

    Returns:
        Task state with shared/private facts and completion tracking.

    Raises:
        ValueError: If task.target_steps is invalid.

    Research rationale:
        Provides a minimal hidden-profile scaffold that preserves the
        information distribution structure without action-specific logic.
    """

    task_cfg = config.get("task", {})
    target_steps = task_cfg.get("target_steps", 3)
    if not isinstance(target_steps, int) or target_steps <= 0:
        raise ValueError("task.target_steps must be a positive integer.")
    shared_facts = task_cfg.get("shared_facts", [])
    if not isinstance(shared_facts, list):
        shared_facts = []
    private_facts = task_cfg.get("private_facts", {})
    if not isinstance(private_facts, dict):
        private_facts = {}
    return {
        "target_steps": target_steps,
        "steps_taken": 0,
        "complete": False,
        "shared_facts": shared_facts,
        "private_facts": private_facts,
    }


def hidden_profile_step(state: dict[str, Any]) -> dict[str, Any]:
    """Advance the hidden-profile task by one step.

    Args:
        state: Mutable task state to update in place.

    Returns:
        Updated task state after incrementing steps_taken.

    Raises:
        ValueError: If task state fields are missing or invalid.
    """

    if state.get("complete") is True:
        return state
    steps_taken = state.get("steps_taken")
    target_steps = state.get("target_steps")
    if not isinstance(steps_taken, int) or not isinstance(target_steps, int):
        raise ValueError("steps_taken and target_steps must be integers.")
    state["steps_taken"] = steps_taken + 1
    if state["steps_taken"] >= target_steps:
        state["complete"] = True
    return state


HIDDEN_PROFILE_TASK = TaskDefinition(
    name="hidden_profile",
    init_state=hidden_profile_init_state,
    step=hidden_profile_step,
)
