"""Task registry interface for pluggable experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class TaskDefinition:
    """Task definition for initializing and stepping task state."""

    name: str
    init_state: Callable[[dict[str, Any]], dict[str, Any]]
    step: Callable[[dict[str, Any]], dict[str, Any]]
    apply_action: (
        Callable[
            [dict[str, Any], str, dict[str, Any], Callable[..., dict[str, Any]]],
            bool,
        ]
        | None
    ) = None


class TaskRegistry:
    """Registry for available tasks."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskDefinition] = {}

    def register(self, task: TaskDefinition) -> None:
        self._tasks[task.name] = task

    def get(self, name: str) -> TaskDefinition:
        if name not in self._tasks:
            raise KeyError(f"Task not registered: {name}")
        return self._tasks[name]


def register_task(registry: TaskRegistry, task: TaskDefinition) -> TaskRegistry:
    """Register a task and return the registry for chaining."""

    registry.register(task)
    return registry


def build_default_registry() -> TaskRegistry:
    """Build a registry with default tasks pre-registered."""

    from src.tasks.accumulator import ACCUMULATOR_TASK
    from src.tasks.counter import COUNTER_TASK
    from src.tasks.daytrader import DAYTRADER_TASK
    from src.tasks.defaults import NOOP_TASK
    from src.tasks.hidden_profile import HIDDEN_PROFILE_TASK
    from src.tasks.maptask import MAPTASK_TASK
    from src.tasks.shapefactory import SHAPEFACTORY_TASK

    registry = TaskRegistry()
    register_task(registry, NOOP_TASK)
    register_task(registry, COUNTER_TASK)
    register_task(registry, ACCUMULATOR_TASK)
    register_task(registry, HIDDEN_PROFILE_TASK)
    register_task(registry, SHAPEFACTORY_TASK)
    register_task(registry, DAYTRADER_TASK)
    return register_task(registry, MAPTASK_TASK)
