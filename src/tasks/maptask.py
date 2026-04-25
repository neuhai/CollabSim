"""MapTask runtime with map progress updates."""

# Task audit summary:
# - Initial state: participants with role/map/map_progress and target_steps/steps_taken/complete.
# - Supported actions: task-specific update_map_progress via maptask_apply_action.
# - Stop condition: task marks complete when steps_taken >= target_steps.
# - Probing trigger: no task-local trigger; probing cadence is controlled by controller/probe config.

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from src.tasks.registry import TaskDefinition


def maptask_init_state(config: dict[str, Any]) -> dict[str, Any]:
    """Initialize MapTask state."""

    task_cfg = config.get("task", {})
    target_steps = task_cfg.get("target_steps", 30)
    if not isinstance(target_steps, int) or target_steps <= 0:
        raise ValueError("task.target_steps must be a positive integer for maptask.")
    roles = task_cfg.get("roles", {})
    maps = task_cfg.get("maps", {})
    landmarks_bundle = _load_landmarks_bundle(config, task_cfg)
    if not isinstance(roles, dict):
        roles = {}
    if not isinstance(maps, dict):
        maps = {}
    agents = config.get("agents", [])
    participants: dict[str, dict[str, Any]] = {}
    for idx, agent in enumerate(agents):
        if not isinstance(agent, dict):
            continue
        agent_id = agent.get("id")
        if not isinstance(agent_id, str) or not agent_id:
            continue
        role = roles.get(agent_id)
        if role not in ("guider", "follower"):
            role = "guider" if idx == 0 else "follower"
        map_info = maps.get(agent_id)
        if not isinstance(map_info, dict):
            map_info = {}
        merged_map_info: dict[str, Any] = {}
        if landmarks_bundle:
            merged_map_info.update(_build_role_map_info(landmarks_bundle, role))
        merged_map_info.update(deepcopy(map_info))
        _attach_map_text(config, merged_map_info)
        participants[agent_id] = {
            "role": role,
            "map": merged_map_info,
            "map_progress": {},
        }
    return {
        "task_type": "maptask",
        "target_steps": target_steps,
        "steps_taken": 0,
        "complete": False,
        "participants": participants,
    }


def maptask_step(state: dict[str, Any]) -> dict[str, Any]:
    """Advance MapTask by one step."""

    if state.get("complete") is True:
        return state
    steps_taken = state.get("steps_taken", 0)
    target_steps = state.get("target_steps", 0)
    if not isinstance(steps_taken, int) or not isinstance(target_steps, int):
        raise ValueError("maptask state steps must be integers.")
    state["steps_taken"] = steps_taken + 1
    if state["steps_taken"] >= target_steps:
        state["complete"] = True
    return state


def maptask_apply_action(
    state: Any,
    actor_id: str,
    action: dict[str, Any],
    emit_event: Callable[..., dict[str, Any]],
) -> bool:
    """Apply MapTask-specific actions against task state."""

    if action.get("type") != "update_map_progress":
        return False
    payload = action.get("payload", {})
    if not isinstance(payload, dict):
        return False
    task_state = state.task_state
    if not isinstance(task_state, dict) or task_state.get("task_type") != "maptask":
        return False
    participants = task_state.get("participants", {})
    if not isinstance(participants, dict):
        return False
    me = participants.get(actor_id)
    if not isinstance(me, dict):
        return False
    progress = payload.get("map_progress")
    if not isinstance(progress, dict):
        return False

    role = me.get("role")
    if role != "follower":
        emit_event(
            event_type="action_rejected",
            actor_id=actor_id,
            visibility="system",
            payload={
                "action": {"type": "update_map_progress", "payload": payload},
                "error_message": "Only follower can update map progress.",
            },
        )
        return True

    drawn_points = _extract_drawn_points(payload, progress)
    if drawn_points is None:
        emit_event(
            event_type="action_rejected",
            actor_id=actor_id,
            visibility="system",
            payload={
                "action": {"type": "update_map_progress", "payload": payload},
                "error_message": "update_map_progress requires drawn_points as a list of [row,col].",
            },
        )
        return True

    grid = _ensure_working_grid(me)
    if not grid:
        emit_event(
            event_type="action_rejected",
            actor_id=actor_id,
            visibility="system",
            payload={
                "action": {"type": "update_map_progress", "payload": payload},
                "error_message": "Follower map copy is unavailable.",
            },
        )
        return True

    existing_points = _normalize_point_set(me.get("drawn_route_points", []))
    anchors = _collect_anchor_points(task_state)
    added_points, error_message = _validate_and_apply_drawn_points(
        grid=grid,
        drawn_points=drawn_points,
        existing_points=existing_points,
        anchor_points=anchors,
    )
    if error_message is not None:
        emit_event(
            event_type="action_rejected",
            actor_id=actor_id,
            visibility="system",
            payload={
                "action": {"type": "update_map_progress", "payload": payload},
                "error_message": error_message,
            },
        )
        return True

    current = me.get("map_progress")
    if not isinstance(current, dict):
        current = {}
        me["map_progress"] = current
    current.update(progress)
    current["last_drawn_points"] = [[row, col] for row, col in added_points]
    current["total_drawn_points"] = len(existing_points.union(added_points))

    all_points = existing_points.union(added_points)
    me["drawn_route_points"] = [[row, col] for row, col in sorted(all_points)]
    me["map_working_text"] = "\n".join("".join(row) for row in grid)

    finish_cell = _finish_cell(task_state)
    if isinstance(finish_cell, tuple) and finish_cell in all_points:
        task_state["complete"] = True

    emit_event(
        event_type="map_progress_updated",
        actor_id=actor_id,
        visibility="public",
        payload={
            "map_progress": dict(current),
            "drawn_points_added": [[row, col] for row, col in added_points],
            "drawn_points_total": len(all_points),
        },
    )
    return True


def _load_landmarks_bundle(config: dict[str, Any], task_cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Load optional maptask landmark bundle referenced by task.landmarks_path."""

    landmarks_path = task_cfg.get("landmarks_path")
    if not isinstance(landmarks_path, str) or not landmarks_path:
        return None

    resolved = _resolve_landmarks_path(config, landmarks_path)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ValueError("PyYAML is required to load task.landmarks_path for maptask.") from exc

    try:
        with resolved.open("r", encoding="utf-8") as handle:
            bundle = yaml.safe_load(handle)
    except OSError as exc:
        raise ValueError(f"Failed to read maptask landmarks file '{resolved}': {exc}") from exc
    except yaml.YAMLError as exc:  # type: ignore[attr-defined]
        raise ValueError(f"Failed to parse maptask landmarks YAML '{resolved}': {exc}") from exc

    if not isinstance(bundle, dict):
        raise ValueError(f"task.landmarks_path '{resolved}' must contain a YAML object.")
    return bundle


def _resolve_landmarks_path(config: dict[str, Any], landmarks_path: str) -> Path:
    """Resolve landmark bundle path relative to experiment config file."""

    path = Path(landmarks_path).expanduser()
    if path.is_absolute():
        return path
    config_dir = config.get("__config_dir")
    if isinstance(config_dir, str) and config_dir:
        return (Path(config_dir) / path).resolve()
    return path.resolve()


def _build_role_map_info(bundle: dict[str, Any], role: str) -> dict[str, Any]:
    """Build map metadata for a participant role from a landmark bundle."""

    info: dict[str, Any] = {}
    map_id = bundle.get("map_id")
    if isinstance(map_id, str) and map_id:
        info["map_id"] = map_id

    map_files = bundle.get("map_files")
    if isinstance(map_files, dict):
        role_file = map_files.get(role)
        if isinstance(role_file, str) and role_file:
            info["map_file"] = role_file

    for key in ("grid", "special_points", "landmarks", "minor_markers"):
        value = bundle.get(key)
        if value is not None:
            info[key] = deepcopy(value)

    return info


def _attach_map_text(config: dict[str, Any], map_info: dict[str, Any]) -> None:
    """Attach text-map content and shape metadata when map_file points to a TXT file."""

    map_file = map_info.get("map_file")
    if not isinstance(map_file, str) or not map_file:
        return
    if not map_file.lower().endswith(".txt"):
        return

    map_path = _resolve_task_file_path(config, map_file)
    try:
        text = map_path.read_text(encoding="utf-8")
    except OSError:
        return

    lines = text.splitlines()
    width = max((len(line) for line in lines), default=0)
    map_info["map_format"] = "ascii_txt"
    map_info["map_text"] = text
    map_info["map_rows"] = len(lines)
    map_info["map_cols"] = width
    if "grid" not in map_info:
        map_info["grid"] = {"rows": len(lines), "cols": width}


def _extract_drawn_points(payload: dict[str, Any], progress: dict[str, Any]) -> list[tuple[int, int]] | None:
    raw_points = payload.get("drawn_points")
    if raw_points is None:
        raw_points = progress.get("drawn_points")
    if not isinstance(raw_points, list):
        return None
    points: list[tuple[int, int]] = []
    for item in raw_points:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None
        row, col = item
        if not isinstance(row, int) or not isinstance(col, int):
            return None
        points.append((row, col))
    return points


def _ensure_working_grid(participant: dict[str, Any]) -> list[list[str]]:
    existing = participant.get("map_working_grid")
    if isinstance(existing, list) and existing and all(isinstance(row, list) for row in existing):
        normalized: list[list[str]] = []
        for row in existing:
            if not isinstance(row, list):
                continue
            normalized_row = [cell if isinstance(cell, str) and cell else " " for cell in row]
            normalized.append(normalized_row)
        if normalized:
            participant["map_working_grid"] = normalized
            return normalized

    map_info = participant.get("map")
    if not isinstance(map_info, dict):
        return []
    map_text = map_info.get("map_text")
    if not isinstance(map_text, str) or not map_text:
        return []
    lines = map_text.splitlines()
    width = max((len(line) for line in lines), default=0)
    grid = [list(line.ljust(width)) for line in lines]
    participant["map_working_grid"] = grid
    participant["map_working_text"] = map_text
    return grid


def _normalize_point_set(raw_points: Any) -> set[tuple[int, int]]:
    if not isinstance(raw_points, list):
        return set()
    result: set[tuple[int, int]] = set()
    for item in raw_points:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        row, col = item
        if isinstance(row, int) and isinstance(col, int):
            result.add((row, col))
    return result


def _collect_anchor_points(task_state: dict[str, Any]) -> set[tuple[int, int]]:
    anchors: set[tuple[int, int]] = set()
    participants = task_state.get("participants")
    if not isinstance(participants, dict):
        return anchors
    for participant in participants.values():
        if not isinstance(participant, dict):
            continue
        map_info = participant.get("map")
        if not isinstance(map_info, dict):
            continue
        special_points = map_info.get("special_points")
        if not isinstance(special_points, dict):
            continue
        for key in ("start", "finish"):
            entry = special_points.get(key)
            if not isinstance(entry, dict):
                continue
            cell = entry.get("cell")
            point = _coerce_cell(cell)
            if point is not None:
                anchors.add(point)
    return anchors


def _finish_cell(task_state: dict[str, Any]) -> tuple[int, int] | None:
    participants = task_state.get("participants")
    if not isinstance(participants, dict):
        return None
    for participant in participants.values():
        if not isinstance(participant, dict):
            continue
        map_info = participant.get("map")
        if not isinstance(map_info, dict):
            continue
        special_points = map_info.get("special_points")
        if not isinstance(special_points, dict):
            continue
        finish = special_points.get("finish")
        if not isinstance(finish, dict):
            continue
        point = _coerce_cell(finish.get("cell"))
        if point is not None:
            return point
    return None


def _coerce_cell(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    row, col = value
    if not isinstance(row, int) or not isinstance(col, int):
        return None
    return (row, col)


def _validate_and_apply_drawn_points(
    *,
    grid: list[list[str]],
    drawn_points: list[tuple[int, int]],
    existing_points: set[tuple[int, int]],
    anchor_points: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], str | None]:
    if not drawn_points:
        return set(), "drawn_points cannot be empty."
    if not grid:
        return set(), "Map grid is empty."

    max_row = len(grid)
    max_col = max((len(row) for row in grid), default=0)
    connectivity = set(existing_points).union(anchor_points)
    added_points: set[tuple[int, int]] = set()

    for point in drawn_points:
        row, col = point
        if row < 0 or col < 0 or row >= max_row or col >= max_col:
            return set(), f"Point {point} is out of map bounds."
        if col >= len(grid[row]):
            return set(), f"Point {point} is out of row bounds."
        cell = grid[row][col]
        if cell == "#":
            return set(), f"Point {point} is blocked by obstacle '#'."
        if point in existing_points or point in added_points:
            continue
        if connectivity and point not in connectivity and not _has_neighbor(point, connectivity.union(added_points)):
            return set(), (
                f"Point {point} is not connected to existing route/start/finish "
                "within the 4-neighborhood."
            )
        if cell not in {"S", "F"}:
            grid[row][col] = "."
        added_points.add(point)
    return added_points, None


def _has_neighbor(point: tuple[int, int], candidates: set[tuple[int, int]]) -> bool:
    row, col = point
    for near_row, near_col in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
        if (near_row, near_col) in candidates:
            return True
    return False


def _resolve_task_file_path(config: dict[str, Any], file_path: str) -> Path:
    """Resolve a task-related file path relative to config dir first, then workspace cwd."""

    path = Path(file_path).expanduser()
    if path.is_absolute():
        return path
    config_dir = config.get("__config_dir")
    if isinstance(config_dir, str) and config_dir:
        candidate = (Path(config_dir) / path).resolve()
        if candidate.exists():
            return candidate
    return path.resolve()


MAPTASK_TASK = TaskDefinition(
    name="maptask",
    init_state=maptask_init_state,
    step=maptask_step,
    apply_action=maptask_apply_action,
)
