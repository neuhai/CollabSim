"""Deterministic accumulator task for reference runs."""

from __future__ import annotations

from typing import Any

from src.tasks.registry import TaskDefinition


def accumulator_init_state(config: dict[str, Any]) -> dict[str, Any]:
    """Initialize a deterministic accumulator task state.

    Args:
        config: Experiment configuration containing task settings.

    Returns:
        Task state with accumulator fields and completion flag.

    Raises:
        ValueError: If task.target_value or task.increment is invalid.

    Research rationale:
        Provides a deterministic accumulation task with distinct state
        dynamics to support reference runs and ablations.
    """

    task_cfg = config.get("task", {})
    target_value = task_cfg.get("target_value", 10)
    increment = task_cfg.get("increment", 2)
    if not isinstance(target_value, int) or target_value <= 0:
        raise ValueError("task.target_value must be a positive integer.")
    if not isinstance(increment, int) or increment <= 0:
        raise ValueError("task.increment must be a positive integer.")
    return {
        "target_value": target_value,
        "increment": increment,
        "current_value": 0,
        "steps_taken": 0,
        "complete": False,
    }


def accumulator_step(state: dict[str, Any]) -> dict[str, Any]:
    """Advance the accumulator by one step and mark completion.

    Args:
        state: Mutable task state to update in place.

    Returns:
        Updated task state after applying the increment.

    Raises:
        ValueError: If state fields are missing or invalid.
    """

    if state.get("complete") is True:
        return state
    current_value = state.get("current_value")
    increment = state.get("increment")
    steps_taken = state.get("steps_taken")
    target_value = state.get("target_value")
    if not all(isinstance(value, int) for value in (current_value, increment, steps_taken, target_value)):
        raise ValueError("Accumulator state fields must be integers.")
    if increment <= 0 or target_value <= 0:
        raise ValueError("increment and target_value must be positive integers.")
    state["current_value"] = current_value + increment
    state["steps_taken"] = steps_taken + 1
    if state["current_value"] >= target_value:
        state["complete"] = True
    return state


ACCUMULATOR_TASK = TaskDefinition(
    name="accumulator",
    init_state=accumulator_init_state,
    step=accumulator_step,
)
