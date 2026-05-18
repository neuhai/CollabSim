"""Deterministic hidden-profile task scaffolding."""

# Task audit summary:
# - Initial state: target_steps/steps_taken/complete plus shared_facts and private_facts.
# - Supported actions: no task-specific apply_action; only controller-level actions are processed.
# - Stop condition: task marks complete when steps_taken >= target_steps.
# - Probing trigger: no task-local trigger; probing cadence is controlled by controller/probe config.

from __future__ import annotations

from typing import Any, Callable

from src.tasks.registry import TaskDefinition


def hidden_profile_init_state(config: dict[str, Any]) -> dict[str, Any]:
    """Initialize a deterministic hidden-profile task state.

    Args:
        config: Experiment configuration containing task settings.

    Returns:
        Task state with shared/private facts and completion tracking.

    Raises:
        ValueError: If task.target_steps is invalid.

    Research rationale:
        Provides a minimal hidden-profile scaffold that preserves the
        information distribution structure without action-specific logic.
    """

    task_cfg = config.get("task", {})
    target_steps = task_cfg.get("target_steps", 3)
    if not isinstance(target_steps, int) or target_steps <= 0:
        raise ValueError("task.target_steps must be a positive integer.")
    shared_facts = task_cfg.get("shared_facts", [])
    if not isinstance(shared_facts, list):
        shared_facts = []
    private_facts = task_cfg.get("private_facts", {})
    if not isinstance(private_facts, dict):
        private_facts = {}
    phase_rules_cfg = task_cfg.get("phase_rules", {})
    if not isinstance(phase_rules_cfg, dict):
        phase_rules_cfg = {}
    initial_vote_decision_id = phase_rules_cfg.get("initial_vote_decision_id", "initial_vote")
    if not isinstance(initial_vote_decision_id, str) or not initial_vote_decision_id:
        initial_vote_decision_id = "initial_vote"
    final_vote_decision_id = phase_rules_cfg.get("final_vote_decision_id", "final_vote")
    if not isinstance(final_vote_decision_id, str) or not final_vote_decision_id:
        final_vote_decision_id = "final_vote"
    discussion_duration_sec = phase_rules_cfg.get("discussion_duration_sec", task_cfg.get("discussion_duration_sec", 60))
    if not isinstance(discussion_duration_sec, (int, float)) or float(discussion_duration_sec) < 0:
        discussion_duration_sec = 60
    participants: dict[str, dict[str, Any]] = {}
    agents = config.get("agents", [])
    if isinstance(agents, list):
        for agent in agents:
            if not isinstance(agent, dict):
                continue
            agent_id = agent.get("id")
            if not isinstance(agent_id, str) or not agent_id:
                continue
            participants[agent_id] = {
                "initial_vote": None,
                "final_vote": None,
            }
    return {
        "task_type": "hidden_profile",
        "target_steps": target_steps,
        "steps_taken": 0,
        "complete": False,
        "shared_facts": shared_facts,
        "private_facts": private_facts,
        "participants": participants,
        "phase": "initial",
        "discussion_started_at_sec": None,
        "discussion_all_do_nothing_streak": 0,
        "discussion_force_final": False,
        "phase_rules": {
            "initial_vote_decision_id": initial_vote_decision_id,
            "final_vote_decision_id": final_vote_decision_id,
            "initial_vote_steps": phase_rules_cfg.get("initial_vote_steps", 3),
            "final_vote_steps": phase_rules_cfg.get("final_vote_steps", 3),
            "discussion_duration_sec": float(discussion_duration_sec),
        },
    }


def hidden_profile_step(state: dict[str, Any]) -> dict[str, Any]:
    """Advance the hidden-profile task by one step.

    Args:
        state: Mutable task state to update in place.

    Returns:
        Updated task state after incrementing steps_taken.

    Raises:
        ValueError: If task state fields are missing or invalid.
    """

    if state.get("complete") is True:
        return state
    steps_taken = state.get("steps_taken")
    target_steps = state.get("target_steps")
    if not isinstance(steps_taken, int) or not isinstance(target_steps, int):
        raise ValueError("steps_taken and target_steps must be integers.")
    state["steps_taken"] = steps_taken + 1
    participants = state.get("participants")
    # Keep step-count completion only for legacy setups without participant voting.
    if (not isinstance(participants, dict) or not participants) and state["steps_taken"] >= target_steps:
        state["complete"] = True
    return state


def hidden_profile_apply_action(
    state: Any,
    actor_id: str,
    action: dict[str, Any],
    emit_event: Callable[..., dict[str, Any]],
) -> bool:
    """Capture hidden-profile vote actions in task state."""

    if action.get("type") != "decide":
        return False
    payload = action.get("payload")
    if not isinstance(payload, dict):
        return False
    decision_id = payload.get("decision_id")
    choice = payload.get("choice")
    if not isinstance(choice, str) or not choice:
        return False
    task_state = state.task_state
    if not isinstance(task_state, dict):
        return False
    phase_rules = task_state.get("phase_rules", {})
    if not isinstance(phase_rules, dict):
        phase_rules = {}
    initial_vote_decision_id = phase_rules.get("initial_vote_decision_id", "initial_vote")
    if not isinstance(initial_vote_decision_id, str) or not initial_vote_decision_id:
        initial_vote_decision_id = "initial_vote"
    final_vote_decision_id = phase_rules.get("final_vote_decision_id", "final_vote")
    if not isinstance(final_vote_decision_id, str) or not final_vote_decision_id:
        final_vote_decision_id = "final_vote"
    if decision_id not in {initial_vote_decision_id, final_vote_decision_id}:
        return False
    participants = task_state.get("participants")
    if not isinstance(participants, dict):
        return False
    entry = participants.get(actor_id)
    if not isinstance(entry, dict):
        return False

    key = "initial_vote" if decision_id == initial_vote_decision_id else "final_vote"
    entry[key] = choice
    emit_event(
        event_type=f"hidden_profile_{key}_submitted",
        actor_id=actor_id,
        visibility="system",
        payload={"decision_id": decision_id, "choice": choice},
    )
    if decision_id == final_vote_decision_id:
        if participants and all(
            isinstance(item, dict) and isinstance(item.get("final_vote"), str) and item.get("final_vote")
            for item in participants.values()
        ):
            task_state["complete"] = True
    return True


HIDDEN_PROFILE_TASK = TaskDefinition(
    name="hidden_profile",
    init_state=hidden_profile_init_state,
    step=hidden_profile_step,
    apply_action=hidden_profile_apply_action,
)
