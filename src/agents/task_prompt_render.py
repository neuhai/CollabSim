"""Dynamic substitutions for task prompt templates (per turn, per agent)."""

from __future__ import annotations

import json
from typing import Any

from src.agents.interface import Observation


def _stringify_template_values(values: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            out[key] = json.dumps(value, ensure_ascii=False)
        else:
            out[key] = str(value)
    return out


def _render_placeholders(template: str, values: dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{key}}}", value)
        rendered = rendered.replace(f"%%{key}%%", value)
    return rendered


_MAPTASK_MAP_OMITTED = (
    "<Current Map omitted on this incremental turn to reduce context size. "
    "Use the experiment rules above, your prior reasoning, and the Status update block "
    "(peer messages and optional follower canvas snapshot for the guide when enabled).>"
)


def _maptask_template_values(
    agent_id: str, task_state: dict[str, Any], *, compact_map: bool = False
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "CURRENT_MAP": "" if not compact_map else _MAPTASK_MAP_OMITTED,
        "GAME_RULE_CANVAS_VISIBILITY_LINE": "",
    }
    line = task_state.get("game_rule_canvas_visibility_line")
    if isinstance(line, str):
        values["GAME_RULE_CANVAS_VISIBILITY_LINE"] = line
    participants = task_state.get("participants")
    if isinstance(participants, dict):
        me = participants.get(agent_id)
        if isinstance(me, dict):
            map_info = me.get("map")
            if isinstance(map_info, dict) and not compact_map:
                prompt = map_info.get("current_map_prompt")
                if isinstance(prompt, str):
                    values["CURRENT_MAP"] = prompt
    return values


def _shapefactory_template_values(
    agent_id: str,
    task_state: dict[str, Any],
    protocol_context: dict[str, Any] | None,
) -> dict[str, str]:
    rules = task_state.get("rules")
    if not isinstance(rules, dict):
        rules = {}
    participants = task_state.get("participants")
    if not isinstance(participants, dict):
        participants = {}
    me = participants.get(agent_id)
    if not isinstance(me, dict):
        me = {}

    specialty = me.get("specialty", "")
    tasks = me.get("tasks")
    if isinstance(tasks, list):
        current_orders = json.dumps(tasks, ensure_ascii=False)
    else:
        current_orders = "[]"

    order_n = rules.get("shapes_order")
    if not isinstance(order_n, int) or order_n <= 0:
        order_n = len(tasks) if isinstance(tasks, list) else 0

    starting_money = rules.get("starting_money")
    if starting_money is None:
        starting_money = me.get("money", "")

    prod_time = rules.get("production_time")
    if prod_time is None:
        prod_time = protocol_context.get("produce_shape_delay_sec") if protocol_context else None
    if prod_time is None:
        prod_time = 10

    lines: list[str] = []
    for pid in task_state.get("agent_ids") or sorted(participants.keys()):
        if not isinstance(pid, str):
            continue
        row = participants.get(pid)
        if not isinstance(row, dict):
            continue
        spec = row.get("specialty", "?")
        lines.append(f"- {pid}: specialty {spec}")
    participants_list = "\n".join(lines) if lines else "(none)"

    pc = protocol_context or {}
    comm = pc.get("communication_level")
    if not isinstance(comm, str):
        comm = ""

    values: dict[str, Any] = {
        "communication_level": comm,
        "shape_amount_per_order": order_n,
        "incentive_money": rules.get("incentive_money", ""),
        "starting_money": starting_money,
        "specialty_shape": specialty,
        "specialty_cost": rules.get("specialty_cost", ""),
        "regular_cost": rules.get("regular_cost", ""),
        "production_time": prod_time,
        "max_production_num": rules.get("max_production_num", ""),
        "price_min": rules.get("min_trade_price", ""),
        "price_max": rules.get("max_trade_price", ""),
        "current_orders": current_orders,
        "participants_list": participants_list,
    }
    return _stringify_template_values(values)


def render_task_prompt_template(
    template: str,
    agent_id: str,
    observation: Observation,
    protocol_context: dict[str, Any] | None,
    *,
    maptask_prompt_style: str = "full",
) -> str:
    """Apply protocol/task_state placeholders; supports ShapeFactory static task markdown."""

    merged: dict[str, str] = {}
    pc = dict(protocol_context) if isinstance(protocol_context, dict) else {}
    base_keys = (
        "communication_level",
        "communication_mode",
        "allowed_message_channels",
        "produce_shape_delay_sec",
    )
    for key in base_keys:
        if key in pc and key not in merged:
            val = pc[key]
            if isinstance(val, list):
                merged[key] = json.dumps(val, ensure_ascii=False)
            elif val is not None:
                merged[key] = str(val)

    st = observation.state if isinstance(observation.state, dict) else {}
    task_state = st.get("task_state")
    if isinstance(task_state, dict) and task_state.get("task_type") == "shapefactory":
        merged.update(_shapefactory_template_values(agent_id, task_state, pc))

    if isinstance(task_state, dict) and task_state.get("task_type") == "maptask":
        compact_map = maptask_prompt_style == "incremental"
        merged.update(
            _stringify_template_values(_maptask_template_values(agent_id, task_state, compact_map=compact_map))
        )

    rendered = _render_placeholders(template, merged)
    return rendered
