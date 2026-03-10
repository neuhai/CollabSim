"""Controller factory helpers."""

from __future__ import annotations

from typing import Any

from src.controller.controller import ExperimentController, ExperimentState
from src.data.logging import init_run_dir
from src.tasks.registry import build_default_registry


def build_controller(
    state: ExperimentState,
    config: dict[str, Any] | None = None,
    run_id: str | None = None,
    run_paths: Any | None = None,
    output_dir: str | None = None,
    task_registry: Any | None = None,
    probe_registry: Any | None = None,
) -> ExperimentController:
    """Build a controller with default task registry wiring."""

    registry = task_registry or build_default_registry()
    resolved_paths = run_paths
    if resolved_paths is None and output_dir is not None:
        resolved_paths = init_run_dir(output_dir)
    return ExperimentController(
        state,
        config=config,
        run_id=run_id,
        run_paths=resolved_paths,
        task_registry=registry if config is not None else None,
        probe_registry=probe_registry,
    )
