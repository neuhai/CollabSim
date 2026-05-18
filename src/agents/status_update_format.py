"""Human-readable status update block for agent prompts (replaces raw Observation JSON)."""

from __future__ import annotations

import json
from typing import Any

from src.agents.interface import Observation


def _task_phase_status_lines(observation: Observation) -> list[str]:
    """Human-readable phase reminders for prompts (DayTrader + Hidden Profile)."""

    st = observation.state if isinstance(observation.state, dict) else {}
    ts = st.get("task_state")
    if not isinstance(ts, dict):
        return []
    task_type = ts.get("task_type")
    phase = ts.get("phase")
    if not isinstance(phase, str):
        return []

    if task_type == "daytrader":
        ri = ts.get("round_index")
        ri_txt = str(ri) if isinstance(ri, int) else "?"
        lines = [
            f"DayTrader · round_index={ri_txt} · phase={phase}",
        ]
        if phase == "decision":
            lines.append(
                "Decision phase: submit make_individual_investment, make_group_investment, or do_nothing only. "
                "Messaging is not allowed until the group_chat phase begins."
            )
        elif phase == "group_chat":
            lines.append("Group-chat phase: you may use message and do_nothing only.")
        return lines

    if task_type == "hidden_profile":
        pr = ts.get("phase_rules")
        phase_rules = pr if isinstance(pr, dict) else {}
        initial_id = phase_rules.get("initial_vote_decision_id", "initial_vote")
        final_id = phase_rules.get("final_vote_decision_id", "final_vote")
        if not isinstance(initial_id, str) or not initial_id:
            initial_id = "initial_vote"
        if not isinstance(final_id, str) or not final_id:
            final_id = "final_vote"

        lines = [f"Hidden Profile · phase={phase}"]
        if phase == "initial":
            lines.append(
                f"Initial vote: use decide only, with decision_id exactly \"{initial_id}\" and a non-empty choice."
            )
        elif phase == "final":
            lines.append(
                f"Final vote: use decide only, with decision_id exactly \"{final_id}\" and a non-empty choice."
            )
        elif phase == "discussion":
            disc_raw = phase_rules.get("discussion_action_types")
            if isinstance(disc_raw, list):
                allowed = [str(x) for x in disc_raw if isinstance(x, str) and x]
            else:
                allowed = ["message"]
            if not allowed:
                allowed = ["message"]
            lines.append(
                "Discussion phase: use "
                + ", ".join(allowed)
                + ", or do_nothing. Voting (decide) is not allowed until the final phase."
            )
        else:
            lines.append("Follow task rules for allowed actions in this phase.")
        return lines

    return []


def _maptask_my_role_and_peer(participants: dict[str, Any], agent_id: str) -> tuple[str | None, str | None]:
    me = participants.get(agent_id)
    if not isinstance(me, dict):
        return None, None
    my_role = me.get("role")
    if not isinstance(my_role, str):
        return None, None
    for pid, row in participants.items():
        if pid == agent_id or not isinstance(row, dict):
            continue
        other_role = row.get("role")
        if isinstance(other_role, str) and other_role != my_role:
            return my_role, pid
    return my_role, None


def _format_maptask_incremental_status(agent_id: str, observation: Observation) -> str:
    """MapTask: only peer messages, plus follower canvas for guide when visibility is on."""

    lines: list[str] = ["=== Map task · incremental view ==="]
    st_root = observation.state if isinstance(observation.state, dict) else {}
    ts = st_root.get("task_state")
    if not isinstance(ts, dict) or ts.get("task_type") != "maptask":
        return "\n".join(lines).strip()

    si = observation.step_index
    gs = observation.game_status if isinstance(observation.game_status, dict) else {}
    ms = gs.get("max_steps")
    if isinstance(ms, int) and ms > 0:
        lines.append(f"Step: {si} / {ms}")
    else:
        lines.append(f"Step index: {si}")

    participants = ts.get("participants")
    role: str | None = None
    peer_id: str | None = None
    if isinstance(participants, dict):
        role, peer_id = _maptask_my_role_and_peer(participants, agent_id)
    lines.append(f"Your role: {role or '?'}. Peer id: {peer_id or '?'}.")
    lines.append("")

    lines.append(f"=== Messages from peer ({peer_id or 'unknown'}) ===")
    peer_messages: list[str] = []
    events = observation.visible_events if isinstance(observation.visible_events, list) else []
    cap = 48
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("event_type") != "message_delivered":
            continue
        actor = event.get("actor_id")
        if actor == agent_id:
            continue
        if peer_id is not None and actor != peer_id:
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        channel = payload.get("channel")
        if channel == "direct":
            rec = payload.get("recipients")
            if not isinstance(rec, list) or agent_id not in rec:
                continue
        content = payload.get("content")
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        mid = payload.get("message_id")
        peer_messages.append(f"- [{mid}] {content.strip()}")

    if peer_messages:
        if len(peer_messages) > cap:
            peer_messages = peer_messages[-cap:]
            lines.append(f"(showing last {cap} peer messages)")
        lines.extend(peer_messages)
    else:
        lines.append("(none in current visible event window)")

    canvas_on = ts.get("maptask_canvas_visibility") is not False
    if role == "guider" and canvas_on:
        live = ts.get("maptask_follower_live_canvas")
        lines.append("")
        lines.append("=== Follower canvas snapshot (visibility ON) ===")
        if isinstance(live, dict):
            fid = live.get("follower_id")
            lines.append(f"Follower id: {fid!s}")
            cp = live.get("current_position")
            if cp is not None:
                lines.append(f"Follower current brush cell [row, col]: {cp}")
            pts = live.get("drawn_route_points")
            if isinstance(pts, list):
                n = len(pts)
                if n <= 120:
                    lines.append(f"Drawn route cells (count={n}): {json.dumps(pts, ensure_ascii=False)}")
                else:
                    lines.append(f"Drawn route cells count={n} (coordinates omitted to save context)")
            wt = live.get("map_working_text")
            if isinstance(wt, str) and wt.strip():
                lines.append("Follower working map (ASCII; '.' = ink on route):")
                lines.append(wt)
        else:
            lines.append("(no canvas state yet)")

    return "\n".join(lines).strip()


def format_agent_status_update(agent_id: str, observation: Observation) -> str:
    """Format game status plus visible state, events, and memory for the prompt."""

    st_root0 = observation.state if isinstance(observation.state, dict) else {}
    ts0 = st_root0.get("task_state")
    if isinstance(ts0, dict) and ts0.get("task_type") == "maptask":
        return _format_maptask_incremental_status(agent_id, observation)

    lines: list[str] = []

    gs = observation.game_status
    if isinstance(gs, dict) and gs:
        lines.append("=== Game Status ===")
        lines.append(f"Session: {gs.get('session_status', 'unknown')}")
        si = gs.get("step_index")
        if si is not None:
            ms = gs.get("max_steps")
            if isinstance(ms, int) and ms > 0:
                lines.append(f"Step: {si} / {ms}")
                rs = gs.get("remaining_steps")
                if isinstance(rs, int):
                    lines.append(f"Remaining steps (within max_steps budget): {rs}")
            else:
                lines.append(f"Step index: {si}")
        st = gs.get("sim_time_sec")
        if st is not None:
            lines.append(f"Simulated time: {st}s")
        wall_rem = gs.get("wall_remaining_sec")
        if isinstance(wall_rem, (int, float)) and wall_rem >= 0:
            m = int(wall_rem // 60)
            s = int(wall_rem % 60)
            lines.append(f"Wall-clock time remaining: {m}m {s}s")
        el = gs.get("wall_elapsed_sec")
        if isinstance(el, (int, float)) and el >= 0 and wall_rem is None:
            lines.append(f"Wall-clock elapsed: {round(el, 1)}s")
        dlim = gs.get("duration_limit_sec")
        if isinstance(dlim, (int, float)) and dlim > 0 and wall_rem is None:
            lines.append(f"Configured duration limit: {dlim}s")
        term = gs.get("termination_condition")
        if term:
            lines.append(f"Termination policy: {term}")
        sm = gs.get("step_mode")
        if sm:
            lines.append(f"Step mode: {sm}")
        lines.append("")

    phase_lines = _task_phase_status_lines(observation)
    if phase_lines:
        lines.append("=== Task phase status (allowed actions this turn) ===")
        lines.extend(phase_lines)
        lines.append("")

    lines.append("=== Your visible state (filtered for you) ===")
    try:
        lines.append(json.dumps(observation.state, ensure_ascii=False, indent=2))
    except (TypeError, ValueError):
        lines.append(str(observation.state))

    events = observation.visible_events
    if events:
        lines.append("")
        lines.append("=== Recent visible events ===")
        window = events[-16:]
        for event in window:
            if not isinstance(event, dict):
                continue
            et = event.get("event_type", "?")
            aid = event.get("actor_id", "?")
            summary = _summarize_event_payload(event)
            lines.append(f"- [{et}] actor={aid}{summary}")

    return "\n".join(lines).strip()


def _summarize_event_payload(event: dict[str, Any]) -> str:
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return ""
    keys = ("error_message", "message_id", "channel", "content_type", "recipients")
    parts: list[str] = []
    for key in keys:
        if key in payload:
            val = payload[key]
            if key == "content" and isinstance(val, str) and len(val) > 80:
                val = val[:77] + "..."
            parts.append(f"{key}={val!r}")
    if parts:
        return " " + ", ".join(parts)
    return ""
