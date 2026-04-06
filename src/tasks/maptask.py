"""MapTask runtime with map progress updates."""

from __future__ import annotations

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
        participants[agent_id] = {
            "role": role,
            "map": map_info,
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

    current = me.get("map_progress")
    if not isinstance(current, dict):
        current = {}
        me["map_progress"] = current
    current.update(progress)

    emit_event(
        event_type="map_progress_updated",
        actor_id=actor_id,
        visibility="public",
        payload={"map_progress": dict(current)},
    )
    return True


MAPTASK_TASK = TaskDefinition(
    name="maptask",
    init_state=maptask_init_state,
    step=maptask_step,
    apply_action=maptask_apply_action,
)
