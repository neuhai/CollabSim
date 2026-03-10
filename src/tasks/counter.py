"""Deterministic counter task for end-to-end validation."""

from __future__ import annotations

from typing import Any

from src.tasks.registry import TaskDefinition


def counter_init_state(config: dict[str, Any]) -> dict[str, Any]:
    """Initialize a deterministic counter task state.

    Args:
        config: Experiment configuration containing task settings.

    Returns:
        Task state with counters and completion flag.

    Raises:
        ValueError: If task.target_steps is missing or invalid.

    Research rationale:
        Provides a deterministic, low-complexity task to validate the
        controller-to-logging pipeline without agent-dependent variance.
    """

    task_cfg = config.get("task", {})
    target_steps = task_cfg.get("target_steps", 3)
    if not isinstance(target_steps, int) or target_steps <= 0:
        raise ValueError("task.target_steps must be a positive integer.")
    return {
        "target_steps": target_steps,
        "remaining_steps": target_steps,
        "steps_taken": 0,
        "complete": False,
    }


def counter_step(state: dict[str, Any]) -> dict[str, Any]:
    """Advance the counter by one step and mark completion.

    Args:
        state: Mutable task state to update in place.

    Returns:
        Updated task state after decrementing remaining steps.

    Raises:
        ValueError: If state fields are missing or invalid.
    """

    if state.get("complete") is True:
        return state
    remaining = state.get("remaining_steps")
    steps_taken = state.get("steps_taken")
    if not isinstance(remaining, int) or not isinstance(steps_taken, int):
        raise ValueError("remaining_steps and steps_taken must be integers.")
    if remaining <= 0:
        state["remaining_steps"] = 0
        state["complete"] = True
        return state
    state["remaining_steps"] = remaining - 1
    state["steps_taken"] = steps_taken + 1
    if state["remaining_steps"] <= 0:
        state["complete"] = True
    return state


COUNTER_TASK = TaskDefinition(
    name="counter",
    init_state=counter_init_state,
    step=counter_step,
)
