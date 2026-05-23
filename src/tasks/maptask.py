"""MapTask runtime with map progress updates."""

# Task audit summary:
# - Initial state: participants with role/map/map_progress and target_steps/steps_taken/complete.
# - Supported actions: draw, erase, undo, reset via maptask_apply_action.
# - Stop condition: task marks complete when steps_taken >= target_steps.
# - Probing trigger: no task-local trigger; probing cadence is controlled by controller/probe config.

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from src.tasks.maptask_drawing_accuracy import compute_drawing_accuracy_snapshot, load_score_board_file
from src.tasks.registry import TaskDefinition


def maptask_init_state(config: dict[str, Any]) -> dict[str, Any]:
    """Initialize MapTask state."""

    task_cfg = config.get("task", {})
    target_steps = task_cfg.get("target_steps", 30)
    if not isinstance(target_steps, int) or target_steps <= 0:
        raise ValueError("task.target_steps must be a positive integer for maptask.")
    roles = task_cfg.get("roles", {})
    maps = task_cfg.get("maps", {})
    material = _load_map_material(config, task_cfg)
    landmarks_bundle = None if material is not None else _load_landmarks_bundle(config, task_cfg)
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
        if material is not None:
            _apply_map_material_to_map_info(merged_map_info, role, material)
        else:
            _attach_map_text(config, merged_map_info)
        participants[agent_id] = {
            "role": role,
            "map": merged_map_info,
            "map_progress": {},
        }
    canvas_vis = task_cfg.get("canvas_visibility", True)
    visibility_line = (
        "- Canvas visibility is ON: the guide may observe the follower's drawing updates via shared task state and events."
        if canvas_vis is not False
        else "- Canvas visibility is OFF: the guide does not see the follower's route drawing updates."
    )
    state: dict[str, Any] = {
        "task_type": "maptask",
        "target_steps": target_steps,
        "steps_taken": 0,
        "complete": False,
        "participants": participants,
        "game_rule_canvas_visibility_line": visibility_line,
        "maptask_canvas_visibility": canvas_vis is not False,
    }
    if material is not None:
        state["reference_route_cells"] = [[row, col] for row, col in material["route_cells"]]
        sb_matrix = material.get("score_board_matrix")
        if isinstance(sb_matrix, list) and sb_matrix:
            state["drawing_score_board"] = sb_matrix
    for aid, pdata in participants.items():
        if isinstance(pdata, dict) and pdata.get("role") == "follower":
            _prime_follower_base_grid(pdata)
    for aid, pdata in participants.items():
        if isinstance(pdata, dict) and pdata.get("role") == "follower":
            _sync_follower_live_canvas(state, aid, pdata)
    return state


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

    action_type = action.get("type")
    if action_type not in ("draw", "erase", "undo", "reset"):
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

    if action_type in ("draw", "erase", "undo", "reset"):
        role = me.get("role")
        if role != "follower":
            emit_event(
                event_type="action_rejected",
                actor_id=actor_id,
                visibility="system",
                payload={
                    "action": {"type": action_type, "payload": action.get("payload")},
                    "error_message": "Only the follower can edit the route map.",
                },
            )
            return True

    if action_type == "draw":
        payload = action.get("payload", {})
        if not isinstance(payload, dict):
            emit_event(
                event_type="action_rejected",
                actor_id=actor_id,
                visibility="system",
                payload={
                    "action": {"type": action_type, "payload": action.get("payload")},
                    "error_message": "Malformed payload object.",
                },
            )
            return True
        progress = payload.get("map_progress")
        if not isinstance(progress, dict):
            progress = {}
        cells = payload.get("cells")
        if cells is None:
            cells = payload.get("drawn_points")
        payload = {
            "map_progress": {**progress, "drawn_points": cells},
            "drawn_points": cells,
        }
        return _maptask_apply_follower_draw_payload(state, actor_id, payload, emit_event, log_type=action_type)

    if action_type == "erase":
        payload = action.get("payload", {})
        return _maptask_apply_follower_erase(state, actor_id, payload if isinstance(payload, dict) else {}, emit_event)

    if action_type == "undo":
        return _maptask_apply_follower_undo(state, actor_id, emit_event)

    if action_type == "reset":
        return _maptask_apply_follower_reset(state, actor_id, emit_event)

    return False


def _maptask_apply_follower_draw_payload(
    state: Any,
    actor_id: str,
    payload: dict[str, Any],
    emit_event: Callable[..., dict[str, Any]],
    *,
    log_type: str,
) -> bool:
    task_state = state.task_state
    participants = task_state.get("participants", {})
    me = participants.get(actor_id, {})

    progress = payload.get("map_progress")
    if not isinstance(progress, dict):
        emit_event(
            event_type="action_rejected",
            actor_id=actor_id,
            visibility="system",
            payload={
                "action": {"type": log_type, "payload": payload},
                "error_message": "draw requires map_progress object (may be empty) plus cells or drawn_points.",
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
                "action": {"type": log_type, "payload": payload},
                "error_message": "Route draw requires drawn_points (or draw.cells) as a list of [row,col].",
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
                "action": {"type": log_type, "payload": payload},
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
                "action": {"type": log_type, "payload": payload},
                "error_message": error_message,
            },
        )
        return True

    if not added_points:
        current_pos = me.get("current_position")
        pos_hint = f" Your current_position is {current_pos}. Start your next segment from there." if current_pos else ""
        emit_event(
            event_type="action_rejected",
            actor_id=actor_id,
            visibility="system",
            payload={
                "action": {"type": log_type, "payload": payload},
                "error_message": f"All submitted points are already drawn — no new cells were added.{pos_hint}",
            },
        )
        return True

    current = me.get("map_progress")
    if not isinstance(current, dict):
        current = {}
        me["map_progress"] = current
    merged_progress = {**progress}
    merged_progress["drawn_points"] = [[row, col] for row, col in drawn_points]
    current.update(merged_progress)
    ordered_added = [pt for pt in drawn_points if pt in added_points]
    current["last_drawn_points"] = [[row, col] for row, col in ordered_added]
    current["total_drawn_points"] = len(existing_points.union(added_points))

    all_points = existing_points.union(added_points)
    me["drawn_route_points"] = [[row, col] for row, col in sorted(all_points)]
    me["map_working_text"] = "\n".join("".join(row) for row in grid)
    if drawn_points:
        me["current_position"] = list(drawn_points[-1])

    # Store pre-draw snapshot so undo can restore exactly to this state.
    me.setdefault("map_draw_batches", []).append([[row, col] for row, col in sorted(existing_points)])

    finish_cell = _finish_cell(task_state)
    if isinstance(finish_cell, tuple) and finish_cell in all_points:
        task_state["complete"] = True

    mp_payload: dict[str, Any] = {
        "map_progress": dict(current),
        "drawn_points_added": [[row, col] for row, col in ordered_added],
        "drawn_points_total": len(all_points),
    }
    acc = compute_drawing_accuracy_snapshot(task_state)
    if acc is not None:
        mp_payload["drawing_accuracy"] = acc
    emit_event(
        event_type="map_progress_updated",
        actor_id=actor_id,
        visibility="public",
        payload=mp_payload,
    )
    _sync_follower_live_canvas(task_state, actor_id, me)
    return True


def _maptask_apply_follower_erase(
    state: Any,
    actor_id: str,
    payload: dict[str, Any],
    emit_event: Callable[..., dict[str, Any]],
) -> bool:
    task_state = state.task_state
    participants = task_state.get("participants", {})
    me = participants.get(actor_id, {})
    cells_raw = payload.get("cells")
    if not isinstance(cells_raw, list):
        emit_event(
            event_type="action_rejected",
            actor_id=actor_id,
            visibility="system",
            payload={"action": {"type": "erase", "payload": payload}, "error_message": "erase.cells required."},
        )
        return True
    erase_points: list[tuple[int, int]] = []
    for item in cells_raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            emit_event(
                event_type="action_rejected",
                actor_id=actor_id,
                visibility="system",
                payload={"action": {"type": "erase", "payload": payload}, "error_message": "erase.cells must be [row,col]."},
            )
            return True
        row, col = item
        if not isinstance(row, int) or not isinstance(col, int):
            emit_event(
                event_type="action_rejected",
                actor_id=actor_id,
                visibility="system",
                payload={"action": {"type": "erase", "payload": payload}, "error_message": "erase cells must be integers."},
            )
            return True
        erase_points.append((row, col))

    drawn = _normalize_point_set(me.get("drawn_route_points", []))
    for p in erase_points:
        if p not in drawn:
            emit_event(
                event_type="action_rejected",
                actor_id=actor_id,
                visibility="system",
                payload={
                    "action": {"type": "erase", "payload": payload},
                    "error_message": f"Cannot erase {p}: not part of the current drawn route.",
                },
            )
            return True

    # Store pre-erase snapshot so undo can restore exactly to this state.
    pre_erase_snapshot = [[row, col] for row, col in sorted(drawn)]
    for p in erase_points:
        drawn.discard(p)

    me.setdefault("map_draw_batches", []).append(pre_erase_snapshot)
    _rebuild_follower_grid_from_drawn(me, drawn)
    me["drawn_route_points"] = [[row, col] for row, col in sorted(drawn)]
    me["current_position"] = _follower_cursor_after_mutations(me, drawn)
    current = me.get("map_progress")
    if isinstance(current, dict):
        current["total_drawn_points"] = len(drawn)
        current.pop("last_drawn_points", None)

    _maptask_refresh_route_completion(task_state, drawn)

    mp_payload_er: dict[str, Any] = {
        "map_progress": dict(me["map_progress"]) if isinstance(me.get("map_progress"), dict) else {},
        "drawn_points_erased": [[row, col] for row, col in erase_points],
        "drawn_points_total": len(drawn),
    }
    acc_er = compute_drawing_accuracy_snapshot(task_state)
    if acc_er is not None:
        mp_payload_er["drawing_accuracy"] = acc_er
    emit_event(
        event_type="map_progress_updated",
        actor_id=actor_id,
        visibility="public",
        payload=mp_payload_er,
    )
    _sync_follower_live_canvas(task_state, actor_id, me)
    return True


def _maptask_apply_follower_undo(
    state: Any,
    actor_id: str,
    emit_event: Callable[..., dict[str, Any]],
) -> bool:
    task_state = state.task_state
    participants = task_state.get("participants", {})
    me = participants.get(actor_id, {})
    batches = me.get("map_draw_batches")
    if not isinstance(batches, list) or not batches:
        emit_event(
            event_type="action_rejected",
            actor_id=actor_id,
            visibility="system",
            payload={"action": {"type": "undo", "payload": {}}, "error_message": "Nothing to undo."},
        )
        return True
    previous_snapshot = batches.pop()
    points: set[tuple[int, int]] = set()
    if isinstance(previous_snapshot, list):
        for item in previous_snapshot:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                r, c = item
                if isinstance(r, int) and isinstance(c, int):
                    points.add((r, c))

    _rebuild_follower_grid_from_drawn(me, points)
    me["drawn_route_points"] = [[row, col] for row, col in sorted(points)]
    me["current_position"] = _follower_cursor_after_mutations(me, points)

    current = me.get("map_progress")
    if isinstance(current, dict):
        current["total_drawn_points"] = len(points)
        current.pop("last_drawn_points", None)

    _maptask_refresh_route_completion(task_state, points)

    mp_payload_undo: dict[str, Any] = {
        "map_progress": dict(me["map_progress"]) if isinstance(me.get("map_progress"), dict) else {},
        "undo": True,
        "drawn_points_total": len(points),
    }
    acc_u = compute_drawing_accuracy_snapshot(task_state)
    if acc_u is not None:
        mp_payload_undo["drawing_accuracy"] = acc_u
    emit_event(
        event_type="map_progress_updated",
        actor_id=actor_id,
        visibility="public",
        payload=mp_payload_undo,
    )
    _sync_follower_live_canvas(task_state, actor_id, me)
    return True


def _maptask_apply_follower_reset(
    state: Any,
    actor_id: str,
    emit_event: Callable[..., dict[str, Any]],
) -> bool:
    task_state = state.task_state
    participants = task_state.get("participants", {})
    me = participants.get(actor_id, {})
    me["map_draw_batches"] = []
    me["drawn_route_points"] = []
    me["map_progress"] = {}
    base = me.get("map_base_grid")
    if isinstance(base, list) and base:
        me["map_working_grid"] = deepcopy(base)
        me["map_working_text"] = "\n".join("".join(row) for row in me["map_working_grid"])
    else:
        _ensure_working_grid(me)
    start = _follower_start_cell(me)
    me["current_position"] = list(start) if start is not None else None
    task_state["complete"] = False

    mp_payload_rs: dict[str, Any] = {
        "map_progress": {},
        "reset": True,
        "drawn_points_total": 0,
    }
    acc_r = compute_drawing_accuracy_snapshot(task_state)
    if acc_r is not None:
        mp_payload_rs["drawing_accuracy"] = acc_r
    emit_event(
        event_type="map_progress_updated",
        actor_id=actor_id,
        visibility="public",
        payload=mp_payload_rs,
    )
    _sync_follower_live_canvas(task_state, actor_id, me)
    return True


def _sync_follower_live_canvas(
    task_state: dict[str, Any],
    follower_id: str,
    participant: dict[str, Any],
) -> None:
    """Mirror follower canvas at task_state top-level for guide prompts when canvas visibility is on."""

    if task_state.get("maptask_canvas_visibility") is False:
        task_state.pop("maptask_follower_live_canvas", None)
        return
    wt = participant.get("map_working_text")
    task_state["maptask_follower_live_canvas"] = {
        "follower_id": follower_id,
        "drawn_route_points": participant.get("drawn_route_points", []),
        "current_position": participant.get("current_position"),
        "map_working_text": wt if isinstance(wt, str) else "",
    }


def _ascii_without_legend(text: str) -> str:
    cut = text.find("\n# Legend")
    if cut != -1:
        text = text[:cut]
    return text.rstrip("\n")


def _strip_route_cells(text: str) -> str:
    body = _ascii_without_legend(text)
    return "\n".join(line.replace("*", ".") for line in body.splitlines())


def _load_map_material(config: dict[str, Any], task_cfg: dict[str, Any]) -> dict[str, Any] | None:
    spec = task_cfg.get("map_material")
    if not isinstance(spec, dict):
        return None
    mj = spec.get("map_json")
    rj = spec.get("map_route_json")
    rt = spec.get("map_route_txt")
    if not all(isinstance(x, str) and x.strip() for x in (mj, rj, rt)):
        return None
    map_json_path = _resolve_task_file_path(config, mj.strip())
    route_json_path = _resolve_task_file_path(config, rj.strip())
    route_txt_path = _resolve_task_file_path(config, rt.strip())
    try:
        map_spec = json.loads(map_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"task.map_material.map_json unreadable or invalid JSON: {exc}") from exc
    if not isinstance(map_spec, dict):
        raise ValueError("task.map_material.map_json must contain a JSON object.")
    try:
        route_obj = json.loads(route_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"task.map_material.map_route_json unreadable or invalid JSON: {exc}") from exc
    if not isinstance(route_obj, dict):
        raise ValueError("task.map_material.map_route_json must contain a JSON object.")
    route_cells = route_obj.get("route_cells")
    if not isinstance(route_cells, list) or not route_cells:
        raise ValueError("map_route_json.route_cells must be a non-empty list.")
    normalized_route: list[tuple[int, int]] = []
    for item in route_cells:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            r, c = item
            if isinstance(r, int) and isinstance(c, int):
                normalized_route.append((r, c))
    if len(normalized_route) != len(route_cells):
        raise ValueError("route_cells must be a list of [row, col] integer pairs.")
    try:
        route_txt_full = route_txt_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"task.map_material.map_route_txt unreadable: {exc}") from exc

    ascii_guide = _ascii_without_legend(route_txt_full)
    ascii_follower = _strip_route_cells(route_txt_full)
    start_t = _coerce_cell(map_spec.get("start_cell"))
    finish_t = normalized_route[-1]

    sb_matrix: list[list[int]] | None = None
    sb_txt = spec.get("score_board_txt")
    if isinstance(sb_txt, str) and sb_txt.strip():
        sb_path = _resolve_task_file_path(config, sb_txt.strip())
    else:
        sb_path = map_json_path.parent / "score_board.txt"
    if sb_path.is_file():
        try:
            sb_matrix = load_score_board_file(sb_path)
        except OSError:
            sb_matrix = None

    return {
        "map_spec": map_spec,
        "route_cells": normalized_route,
        "ascii_guide": ascii_guide,
        "ascii_follower": ascii_follower,
        "start_cell": start_t,
        "finish_cell": finish_t,
        "score_board_matrix": sb_matrix,
    }


def _apply_map_material_to_map_info(map_info: dict[str, Any], role: str, material: dict[str, Any]) -> None:
    spec = material["map_spec"]
    af_lines = material["ascii_follower"].splitlines()
    ag_lines = material["ascii_guide"].splitlines()
    rows = max(len(af_lines), len(ag_lines))
    cols = max(max((len(line) for line in af_lines), default=0), max((len(line) for line in ag_lines), default=0))
    landmarks = spec.get("landmarks")
    map_info["grid"] = {"rows": rows, "cols": cols}
    if isinstance(landmarks, dict):
        map_info["landmarks"] = deepcopy(landmarks)
    map_info["map_format"] = "ascii_txt"
    sp: dict[str, Any] = {}
    sc = material["start_cell"]
    if isinstance(sc, tuple):
        sp["start"] = {"cell": [sc[0], sc[1]]}
    fc = material["finish_cell"]
    if isinstance(fc, tuple):
        sp["finish"] = {"cell": [fc[0], fc[1]]}
    map_info["special_points"] = sp
    map_json_txt = json.dumps(spec, ensure_ascii=False, indent=2)
    route_txt = json.dumps({"route_cells": [list(p) for p in material["route_cells"]]}, ensure_ascii=False)
    if role == "guider":
        map_info["map_text"] = material["ascii_guide"]
        map_info["current_map_prompt"] = (
            "Map specification (JSON):\n"
            + map_json_txt
            + "\n\nGround-truth route as ordered [row, col] cells:\n"
            + route_txt
            + "\n\nASCII map (route cells marked '*'; borders/obstacles '#'; start 'S'):\n"
            + material["ascii_guide"]
        )
    else:
        map_info["map_text"] = material["ascii_follower"]
        map_info["current_map_prompt"] = (
            "Map specification (JSON):\n"
            + map_json_txt
            + "\n\nASCII map for the follower (route hidden; '*' replaced with walkable '.'):\n"
            + material["ascii_follower"]
        )
    map_info["map_rows"] = rows
    map_info["map_cols"] = cols


def _prime_follower_base_grid(participant: dict[str, Any]) -> None:
    grid = _ensure_working_grid(participant)
    if grid:
        participant["map_base_grid"] = deepcopy(grid)
    participant["map_draw_batches"] = []


def _rebuild_follower_grid_from_drawn(participant: dict[str, Any], drawn: set[tuple[int, int]]) -> None:
    base = participant.get("map_base_grid")
    if not isinstance(base, list) or not base:
        _ensure_working_grid(participant)
        participant["map_base_grid"] = deepcopy(participant["map_working_grid"])
        base = participant.get("map_base_grid")
    if not isinstance(base, list):
        return
    grid = deepcopy(base)
    for row, col in drawn:
        if row < 0 or col < 0 or row >= len(grid):
            continue
        if col >= len(grid[row]):
            continue
        ch = grid[row][col]
        if ch not in {"S", "F", "#"}:
            grid[row][col] = "."
    participant["map_working_grid"] = grid
    participant["map_working_text"] = "\n".join("".join(r) for r in grid)


def _follower_start_cell(participant: dict[str, Any]) -> tuple[int, int] | None:
    map_info = participant.get("map")
    if not isinstance(map_info, dict):
        return None
    special = map_info.get("special_points")
    if not isinstance(special, dict):
        return None
    start = special.get("start")
    if isinstance(start, dict):
        return _coerce_cell(start.get("cell"))
    return None


def _follower_cursor_after_mutations(participant: dict[str, Any], drawn: set[tuple[int, int]]) -> list[int] | None:
    if not drawn:
        start = _follower_start_cell(participant)
        return list(start) if start is not None else None
    batches = participant.get("map_draw_batches")
    if isinstance(batches, list) and batches:
        last_batch = batches[-1]
        if isinstance(last_batch, list) and last_batch:
            tail = last_batch[-1]
            if isinstance(tail, (list, tuple)) and len(tail) == 2:
                r, c = tail
                if isinstance(r, int) and isinstance(c, int) and (r, c) in drawn:
                    return [r, c]
    ordered = sorted(drawn)
    return list(ordered[-1])


def _maptask_refresh_route_completion(task_state: dict[str, Any], drawn: set[tuple[int, int]]) -> None:
    finish_cell = _finish_cell(task_state)
    if isinstance(finish_cell, tuple) and finish_cell in drawn:
        task_state["complete"] = True
    else:
        task_state["complete"] = False


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
