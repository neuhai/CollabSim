"""Shared action/probe prompt assembly (MapTask incremental path avoids huge repeat payloads)."""

from __future__ import annotations

import json
from typing import Any

from src.agents.interface import Observation
from src.agents.status_update_format import format_agent_status_update
from src.agents.task_prompt_render import render_task_prompt_template


def _render_template(template: str, values: dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{key}}}", value)
    return rendered


def maptask_task_state(observation: Observation) -> dict[str, Any] | None:
    st = observation.state if isinstance(observation.state, dict) else {}
    ts = st.get("task_state")
    if isinstance(ts, dict) and ts.get("task_type") == "maptask":
        return ts
    return None


def _continuation_blurb(agent_id: str, communication_limits: str) -> str:
    lim = communication_limits.strip()
    lim_line = f"\nCommunication limits: {lim}" if lim else ""
    return (
        "Continuation turn: long instructions from your first model call are omitted to save context. "
        "Follow the same participant id, task rules, allowed actions, and JSON output format as turn 1."
        f"{lim_line}\n\n"
        f"Your participant id is: {agent_id}"
    )


def _maptask_incremental_intro(observation: Observation, role: str, communication_limits: str) -> str:
    ts = maptask_task_state(observation) or {}
    canvas_on = ts.get("maptask_canvas_visibility") is not False
    r = (role or "").strip().lower()
    role_title = "GUIDE" if r == "guider" else "FOLLOWER" if r == "follower" else role.upper()
    canvas_clause = ""
    if r == "guider" and canvas_on:
        canvas_clause = ", and the follower’s latest canvas snapshot when present,"
    lim = communication_limits.strip()
    lim_line = f"\nCommunication limits: {lim}" if lim else ""
    return (
        "Map Task — incremental update. Full system instructions and full map payloads are not repeated each call "
        f"to avoid context overflow.{lim_line}\n"
        f"You are the {role_title}. The status block lists only new peer signals{canvas_clause} "
        "(not the entire simulation history)."
    )


def _compact_observation_payload(
    agent_id: str,
    observation: Observation,
    status_update: str,
    *,
    maptask_incremental: bool = False,
) -> dict[str, Any]:
    return {
        "agent_id": agent_id,
        "step_index": observation.step_index,
        "status_update_text": status_update,
        "visible_event_count": len(observation.visible_events or []),
        "prompt_continuation_only": True,
        **({"maptask_incremental": True} if maptask_incremental else {}),
    }


def compose_action_prompt(
    observation: Observation,
    *,
    agent_id: str,
    role: str,
    system_prompt: str,
    persona_prompt_template: str,
    persona_profile: str | None,
    task_prompt_template: str,
    protocol_prompt_template: str,
    protocol_context: dict[str, Any] | None,
    communication_limits: str,
    action_space_prompt_template: str,
    return_format_prompt_template: str,
    action_prompt_template: str,
    allowed_actions: list[str],
    decide_reveal: str | None,
    use_full_prompt: bool = True,
) -> tuple[str, str, dict[str, Any]]:
    """Return (full_prompt, static_prefix, observation_payload) for one model call.

    When ``use_full_prompt`` is False, only a short continuation header plus the action
    template (with embedded status update) is sent — matching “turn 2+” behavior.
    """

    allowed_csv = ", ".join(allowed_actions) if allowed_actions else "message, decide"
    dr = decide_reveal or "aggregated"
    status_update = format_agent_status_update(agent_id, observation)
    action_prompt = _render_template(
        action_prompt_template,
        {
            "allowed_actions": allowed_csv,
            "status_update": status_update,
            "observation_json": status_update,
            "decide_reveal": dr,
        },
    )
    is_maptask = maptask_task_state(observation) is not None

    if is_maptask and not use_full_prompt:
        task_rendered = render_task_prompt_template(
            task_prompt_template,
            agent_id,
            observation,
            protocol_context,
            maptask_prompt_style="incremental",
        )
        intro = _maptask_incremental_intro(observation, role, communication_limits)
        action_space_prompt = _render_template(action_space_prompt_template, {"allowed_actions": allowed_csv})
        return_format_prompt = _render_template(return_format_prompt_template, {"decide_reveal": dr})
        static_prefix = (
            f"{intro}\n\n"
            f"Your participant id is: {agent_id}\n\n"
            f"{task_rendered}\n\n"
            f"{action_space_prompt}\n\n"
            f"{return_format_prompt}"
        )
        full_prompt = (
            f"{intro}\n\n"
            f"Your participant id is: {agent_id}\n\n"
            f"{task_rendered}\n\n"
            f"{action_space_prompt}\n\n"
            f"Status update:\n{status_update}\n\n"
            f"{return_format_prompt}\n\n"
            f"{action_prompt}"
        )
        observation_payload = _compact_observation_payload(
            agent_id, observation, status_update, maptask_incremental=True
        )
        return full_prompt, static_prefix, observation_payload

    if not use_full_prompt:
        continuation = _continuation_blurb(agent_id, communication_limits)
        full_prompt = f"{continuation}\n\n{action_prompt}"
        observation_payload = _compact_observation_payload(agent_id, observation, status_update)
        return full_prompt, continuation, observation_payload

    persona_resolved = persona_profile or "Default collaborative persona."
    persona_prompt = _render_template(persona_prompt_template, {"persona_profile": persona_resolved})
    map_style_kw: dict[str, str] = {"maptask_prompt_style": "full"} if is_maptask else {}
    task_prompt_rendered = render_task_prompt_template(
        task_prompt_template,
        agent_id,
        observation,
        protocol_context,
        **map_style_kw,
    )
    protocol_prompt = _render_template(
        protocol_prompt_template,
        {
            "protocol_json": json.dumps(protocol_context or {}, ensure_ascii=False),
            "communication_limits": communication_limits,
        },
    )
    action_space_prompt = _render_template(action_space_prompt_template, {"allowed_actions": allowed_csv})
    return_format_prompt = _render_template(return_format_prompt_template, {"decide_reveal": dr})
    static_prefix = (
        f"{system_prompt}\n\n"
        f"Your participant id is: {agent_id}\n\n"
        f"{persona_prompt}\n\n"
        f"{task_prompt_rendered}\n\n"
        f"{protocol_prompt}\n\n"
        f"{action_space_prompt}\n\n"
        f"{return_format_prompt}"
    )
    full_prompt = (
        f"{system_prompt}\n\n"
        f"Your participant id is: {agent_id}\n\n"
        f"{persona_prompt}\n\n"
        f"{task_prompt_rendered}\n\n"
        f"{protocol_prompt}\n\n"
        f"{action_space_prompt}\n\n"
        f"Status update:\n{status_update}\n\n"
        f"{return_format_prompt}\n\n"
        f"{action_prompt}"
    )
    observation_payload: dict[str, Any] = {
        "agent_id": agent_id,
        "state": observation.state,
        "visible_events": observation.visible_events,
        "game_status": observation.game_status,
    }
    return full_prompt, static_prefix, observation_payload


def compose_probe_prompt(
    observation: Observation,
    *,
    agent_id: str,
    role: str,
    system_prompt: str,
    persona_prompt_template: str,
    persona_profile: str | None,
    task_prompt_template: str,
    protocol_prompt_template: str,
    protocol_context: dict[str, Any] | None,
    communication_limits: str,
    action_space_prompt_template: str,
    allowed_actions: list[str],
    decide_reveal: str | None,
    prompt: str,
    construct: str | None,
    use_full_prompt: bool = True,
) -> str:
    allowed_csv = ", ".join(allowed_actions) if allowed_actions else "message, decide"
    dr = decide_reveal or "aggregated"
    status_update = format_agent_status_update(agent_id, observation)
    probe_context = f"Construct: {construct}\n\n" if construct else ""
    probe_tail = (
        "Instead of selecting an action, answer the probe items below.\n"
        f"{probe_context}"
        f"{prompt}\n\n"
        "Return strict JSON only."
    )
    action_refs = (
        f"Action payload references:\n"
        f"- message: {{\"channel\":\"broadcast|direct\",\"content\":\"...\",\"content_type\":\"text\",\"recipients\":[\"B\"] (required for direct)}}\n"
        f"- decide: {{\"decision_id\":\"plan_selection\",\"choice\":\"...\",\"reveal\":\"{dr}\"}}\n"
        f"- produce_shape: {{\"shape\":\"<choose_from_task_state>\",\"quantity\":1}}\n"
        f"- propose_trade_offer: {{\"offer_type\":\"buy|sell\",\"shape\":\"<shape_from_task_state>\",\"price_per_unit\":20,\"target_id\":\"B\",\"quantity\":1}}\n"
        f"- trade_response: {{\"transaction_id\":\"offer_2_1\",\"response_type\":\"accept|decline\"}}\n"
        f"- cancel_trade_offer: {{\"transaction_id\":\"offer_2_1\"}}\n"
        f"- fulfill_order: {{\"order_indices\":[0,1]}}\n"
        f"- make_individual_investment: {{\"invest_price\":30}}\n"
        f"- make_group_investment: {{\"invest_price\":30}}\n"
        f"- draw: {{\"cells\":[[19,53],[20,53],[21,53]]}}\n"
        f"- erase: {{\"cells\":[[19,53],[20,53]]}}\n"
        f"- undo: {{}}\n"
        f"- reset: {{}}\n"
        f"- do_nothing: {{\"reason\":\"...\"}}\n\n"
    )

    is_maptask = maptask_task_state(observation) is not None

    if not use_full_prompt:
        continuation = _continuation_blurb(agent_id, communication_limits)
        return (
            f"{continuation}\n\n"
            f"Status update:\n{status_update}\n\n"
            f"{action_refs}"
            f"{probe_tail}"
        )

    if is_maptask:
        persona_resolved = persona_profile or "Default collaborative persona."
        persona_prompt = _render_template(persona_prompt_template, {"persona_profile": persona_resolved})
        task_rendered = render_task_prompt_template(
            task_prompt_template,
            agent_id,
            observation,
            protocol_context,
            maptask_prompt_style="full",
        )
        protocol_prompt = _render_template(
            protocol_prompt_template,
            {
                "protocol_json": json.dumps(protocol_context or {}, ensure_ascii=False),
                "communication_limits": communication_limits,
            },
        )
        action_space_prompt = _render_template(action_space_prompt_template, {"allowed_actions": allowed_csv})
        return (
            f"{system_prompt}\n\n"
            f"Your participant id is: {agent_id}\n\n"
            f"{persona_prompt}\n\n"
            f"{task_rendered}\n\n"
            f"{protocol_prompt}\n\n"
            f"{action_space_prompt}\n\n"
            f"Status update:\n{status_update}\n\n"
            f"{action_refs}"
            f"{probe_tail}"
        )

    persona_resolved = persona_profile or "Default collaborative persona."
    persona_prompt = _render_template(persona_prompt_template, {"persona_profile": persona_resolved})
    task_prompt_rendered = render_task_prompt_template(
        task_prompt_template,
        agent_id,
        observation,
        protocol_context,
    )
    protocol_prompt = _render_template(
        protocol_prompt_template,
        {
            "protocol_json": json.dumps(protocol_context or {}, ensure_ascii=False),
            "communication_limits": communication_limits,
        },
    )
    action_space_prompt = _render_template(action_space_prompt_template, {"allowed_actions": allowed_csv})
    return (
        f"{system_prompt}\n\n"
        f"Your participant id is: {agent_id}\n\n"
        f"{persona_prompt}\n\n"
        f"{task_prompt_rendered}\n\n"
        f"{protocol_prompt}\n\n"
        f"{action_space_prompt}\n\n"
        f"Status update:\n{status_update}\n\n"
        f"{action_refs}"
        f"{probe_tail}"
    )
