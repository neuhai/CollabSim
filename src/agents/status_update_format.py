"""Human-readable status update block for agent prompts (replaces raw Observation JSON)."""

from __future__ import annotations

import json
from typing import Any

from src.agents.interface import Observation


def _task_phase_status_lines(agent_id: str, observation: Observation) -> list[str]:
    """Human-readable phase reminders for prompts (DayTrader + Hidden Profile)."""

    st = observation.state if isinstance(observation.state, dict) else {}
    ts = st.get("task_state")
    if not isinstance(ts, dict):
        return []
    task_type = ts.get("task_type")
    gs = observation.game_status if isinstance(observation.game_status, dict) else {}
    phase = ts.get("phase")
    if not isinstance(phase, str) and isinstance(gs.get("hidden_profile_phase"), str):
        phase = gs.get("hidden_profile_phase")
    if not isinstance(phase, str):
        return []

    if task_type == "daytrader":
        ri = ts.get("round_index")
        ri_txt = str(ri) if isinstance(ri, int) else "?"
        target_rounds = ts.get("target_rounds")
        tr_txt = str(target_rounds) if isinstance(target_rounds, int) else "?"
        rounds_completed = ts.get("rounds_completed")
        rc_txt = str(rounds_completed) if isinstance(rounds_completed, int) else "0"
        phase_rules = ts.get("phase_rules")
        pr = phase_rules if isinstance(phase_rules, dict) else {}
        interval = pr.get("discussion_every_n_rounds")
        if not isinstance(interval, int) or interval <= 0:
            interval = 5
        lines = [
            f"DayTrader · round {ri_txt} of {tr_txt} · phase={phase} · rounds_completed={rc_txt}",
            f"Group discussion runs after every {interval} decision rounds (rounds {interval}, {interval * 2}, …).",
        ]
        if phase == "decision":
            if isinstance(ri, int) and ri > 0 and ri % interval == 0:
                lines.append(
                    "Decision phase (discussion block): submit make_individual_investment, "
                    "make_group_investment, or do_nothing only. Group chat follows when all have acted."
                )
            else:
                next_disc = ((ri - 1) // interval + 1) * interval if isinstance(ri, int) and ri > 0 else interval
                lines.append(
                    f"Decision phase: submit make_individual_investment, make_group_investment, or do_nothing only. "
                    f"Messaging is not allowed until group_chat after round {next_disc}."
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

        phase_labels = {
            "initial": "INITIAL VOTE (pre-discussion) — decide only, no messaging",
            "discussion": "DISCUSSION — message / do_nothing only, no decide",
            "final": "FINAL VOTE (post-discussion) — decide only, no messaging",
        }
        lines = [
            f"Hidden Profile · current phase: {phase} ({phase_labels.get(phase, 'see task rules')})",
        ]
        init_range = gs.get("hidden_profile_initial_vote_step_range")
        disc_range = gs.get("hidden_profile_discussion_step_range")
        final_range = gs.get("hidden_profile_final_vote_step_range")
        if isinstance(init_range, str) and isinstance(final_range, str):
            disc_text = disc_range if isinstance(disc_range, str) else "middle steps"
            lines.append(
                f"Step schedule: initial vote steps {init_range}; discussion steps {disc_text}; "
                f"final vote steps {final_range}."
            )
        step_idx = gs.get("step_index")
        if isinstance(step_idx, int) and step_idx > 0:
            lines.append(f"Current simulation step: {step_idx}.")
        remaining = gs.get("remaining_steps")
        max_steps = gs.get("max_steps") or gs.get("hidden_profile_max_steps")
        if isinstance(remaining, int) and isinstance(max_steps, int) and max_steps > 0:
            lines.append(f"Remaining steps (within max_steps budget): {remaining} of {max_steps}.")
        initial_done = gs.get("hidden_profile_initial_vote_submitted")
        final_done = gs.get("hidden_profile_final_vote_submitted")
        if initial_done is True:
            lines.append("Your initial vote: already submitted.")
        elif initial_done is False and phase != "initial":
            lines.append("Your initial vote: already submitted.")
        if final_done is True:
            lines.append("Your final vote: already submitted.")
        if gs.get("session_complete") is True:
            lines.append("Session is complete. No further actions are required.")
        if gs.get("hidden_profile_forced_final_vote") is True and phase == "final":
            lines.append(
                "The session is ending before the scheduled final step; submit your final vote now."
            )

        if phase == "initial":
            if initial_done is True:
                lines.append(
                    "Wait for other participants to finish initial voting. "
                    "do_nothing is allowed; message and decide are not."
                )
            else:
                lines.append(
                    f"Submit exactly one decide with decision_id \"{initial_id}\" and a non-empty choice. "
                    "message and do_nothing are not allowed in this phase."
                )
        elif phase == "final":
            if final_done is True:
                lines.append(
                    "Wait for other participants to finish final voting (or for the run to end). "
                    "do_nothing is allowed; message and decide are not."
                )
            else:
                lines.append(
                    f"Submit exactly one decide with decision_id \"{final_id}\" and a non-empty choice. "
                    "message and do_nothing are not allowed in this phase."
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
                + ", or do_nothing. decide / voting is not allowed until the final phase."
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


def _maptask_start_cell(participant: dict[str, Any]) -> list[int] | None:
    map_info = participant.get("map")
    if not isinstance(map_info, dict):
        return None
    special = map_info.get("special_points")
    if not isinstance(special, dict):
        return None
    start = special.get("start")
    if not isinstance(start, dict):
        return None
    cell = start.get("cell")
    if isinstance(cell, (list, tuple)) and len(cell) == 2:
        row, col = cell
        if isinstance(row, int) and isinstance(col, int):
            return [row, col]
    return None


def _maptask_is_diagonal_step(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) == 1 and abs(a[1] - b[1]) == 1


def _maptask_diagonal_hint(cells: list[Any]) -> str | None:
    points: list[tuple[int, int]] = []
    for item in cells:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            row, col = item
            if isinstance(row, int) and isinstance(col, int):
                points.append((row, col))
    for prev, nxt in zip(points, points[1:]):
        if _maptask_is_diagonal_step(prev, nxt):
            return (
                f"Hint: {list(nxt)} is diagonal from {list(prev)}; "
                f"use a 4-connected stair-step such as [{prev[0] + 1}, {prev[1]}] "
                f"or [{prev[0]}, {prev[1] + (1 if nxt[1] > prev[1] else -1)}]."
            )
    return None


def _maptask_follower_route_lines(participants: dict[str, Any], agent_id: str) -> list[str]:
    me = participants.get(agent_id)
    if not isinstance(me, dict):
        return []
    lines: list[str] = ["=== Your route state ==="]
    pts = me.get("drawn_route_points")
    cp = me.get("current_position")
    start = _maptask_start_cell(me)
    if isinstance(pts, list) and pts:
        n = len(pts)
        lines.append(f"Drawn route count: {n}")
        tail = pts[-20:] if n > 20 else pts
        if n > 20:
            lines.append(f"Drawn route cells (last {len(tail)} of {n}): {json.dumps(tail, ensure_ascii=False)}")
        else:
            lines.append(f"Drawn route cells: {json.dumps(tail, ensure_ascii=False)}")
        if cp is not None:
            lines.append(f"Current brush / anchor [row, col]: {cp}")
    else:
        lines.append("Drawn route count: 0 (nothing confirmed on canvas yet)")
        anchor = cp if cp is not None else start
        if anchor is not None:
            lines.append(f"Anchor [row, col]: {anchor}")
        if start is not None:
            lines.append(
                f"Start S is at {start}. First draw.cells point must be 4-connected to S or existing route — "
                "diagonal steps are rejected."
            )
    return lines


def _maptask_rejection_lines(events: list[Any], agent_id: str, *, cap: int = 8) -> list[str]:
    rejections: list[str] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("event_type") != "action_rejected":
            continue
        if event.get("actor_id") != agent_id:
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        action = payload.get("action")
        if not isinstance(action, dict):
            continue
        action_type = action.get("type")
        if action_type not in ("draw", "erase", "undo", "reset"):
            continue
        error_message = payload.get("error_message")
        if not isinstance(error_message, str):
            error_message = "unknown error"
        line = f"- {action_type} rejected: {error_message}"
        action_payload = action.get("payload")
        if isinstance(action_payload, dict) and action_type == "draw":
            cells = action_payload.get("cells")
            if cells is None:
                cells = action_payload.get("drawn_points")
            if isinstance(cells, list):
                hint = _maptask_diagonal_hint(cells)
                if hint:
                    line += f"\n  {hint}"
        rejections.append(line)
    if not rejections:
        return []
    if len(rejections) > cap:
        rejections = rejections[-cap:]
    lines = ["=== Recent action rejections (fix before resubmitting) ==="]
    if len(rejections) == cap:
        lines.append(f"(showing last {cap} rejections)")
    lines.extend(rejections)
    lines.append("Do not repeat the same invalid draw/erase payload; adjust cells per the error above.")
    return lines


def _format_maptask_incremental_status(agent_id: str, observation: Observation) -> str:
    """MapTask: peer messages, follower route/rejection feedback, guider canvas when enabled."""

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
        rs = gs.get("remaining_steps")
        if isinstance(rs, int):
            lines.append(f"Remaining steps (within max_steps budget): {rs}")
    else:
        lines.append(f"Step index: {si}")

    participants = ts.get("participants")
    role: str | None = None
    peer_id: str | None = None
    if isinstance(participants, dict):
        role, peer_id = _maptask_my_role_and_peer(participants, agent_id)
    lines.append(f"Your role: {role or '?'}. Peer id: {peer_id or '?'}.")
    lines.append("")

    events = observation.visible_events if isinstance(observation.visible_events, list) else []
    if role == "follower" and isinstance(participants, dict):
        route_lines = _maptask_follower_route_lines(participants, agent_id)
        if route_lines:
            lines.extend(route_lines)
            lines.append("")
        rejection_lines = _maptask_rejection_lines(events, agent_id)
        if rejection_lines:
            lines.extend(rejection_lines)
            lines.append("")

    lines.append(f"=== Messages from peer ({peer_id or 'unknown'}) ===")
    peer_messages: list[str] = []
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

    phase_lines = _task_phase_status_lines(agent_id, observation)
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
    keys = ("error_message", "message_id", "channel", "content", "content_type", "recipients")
    parts: list[str] = []
    for key in keys:
        if key in payload:
            parts.append(f"{key}={payload[key]!r}")
    if parts:
        return " " + ", ".join(parts)
    return ""
