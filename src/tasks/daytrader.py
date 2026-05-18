"""DayTrader task runtime with investment actions."""

# Task audit summary:
# - Initial state: participants only exposes public metadata in task_state.
# - Private balances/history are stored in state.buffers and never shared in task_state.
# - Rules include min/max trade price and target_rounds/steps_taken/complete.
# - Supported actions: make_individual_investment (2x return) and make_group_investment (pool shared).
# - Group pool is settled at end of each decision phase before group_chat begins.
# - Stop condition: task marks complete when rounds_completed >= target_rounds.

from __future__ import annotations

from typing import Any, Callable

from src.tasks.registry import TaskDefinition

_PRIVATE_KEY = "daytrader_private_participants"
_GROUP_POOL_KEY = "daytrader_group_pool"
_ROUND_START_MONEY_KEY = "daytrader_round_start_money"
_ROUND_BONUS = 90.0


def daytrader_init_state(config: dict[str, Any]) -> dict[str, Any]:
    """Initialize DayTrader task state."""

    task_cfg = config.get("task", {})
    target_rounds = task_cfg.get("target_rounds")
    if target_rounds is None:
        legacy_target_steps = task_cfg.get("target_steps")
        if isinstance(legacy_target_steps, int) and legacy_target_steps > 0:
            target_rounds = max(1, (legacy_target_steps + 1) // 2)
        else:
            target_rounds = 15
    if not isinstance(target_rounds, int) or target_rounds <= 0:
        raise ValueError("task.target_rounds must be a positive integer for daytrader.")
    starting_money = task_cfg.get("starting_money", 200.0)
    min_trade_price = float(task_cfg.get("min_trade_price", 15.0))
    max_trade_price = float(task_cfg.get("max_trade_price", 100.0))
    agents = config.get("agents", [])
    participants: dict[str, dict[str, Any]] = {}
    for agent in agents:
        if not isinstance(agent, dict):
            continue
        agent_id = agent.get("id")
        if not isinstance(agent_id, str) or not agent_id:
            continue
        participants[agent_id] = {}
    participant_ids = sorted(participants.keys())
    phase_rules_cfg = task_cfg.get("phase_rules", {})
    if not isinstance(phase_rules_cfg, dict):
        phase_rules_cfg = {}
    discussion_every = phase_rules_cfg.get("discussion_every_n_rounds", 5)
    if not isinstance(discussion_every, int) or discussion_every <= 0:
        discussion_every = 5
    group_chat_max_turns = phase_rules_cfg.get("group_chat_max_turns", 12)
    if not isinstance(group_chat_max_turns, int) or group_chat_max_turns <= 0:
        group_chat_max_turns = 12
    return {
        "task_type": "daytrader",
        "target_rounds": target_rounds,
        "target_steps": target_rounds * 2,
        "phase_rules": {
            "discussion_every_n_rounds": discussion_every,
            "group_chat_max_turns": group_chat_max_turns,
        },
        "starting_money": float(starting_money),
        "steps_taken": 0,
        "rounds_completed": 0,
        "complete": False,
        "round_index": 1,
        "phase": "decision",
        "phase_step_in_round": 1,
        "decision_pending_agents": participant_ids,
        "group_chat_turns": 0,
        "group_chat_silent_agents": [],
        "participants": participants,
        "rules": {
            "min_trade_price": min_trade_price,
            "max_trade_price": max_trade_price,
        },
    }


def daytrader_step(state: dict[str, Any]) -> dict[str, Any]:
    """No-op: rounds_completed and complete are updated by phase transitions in the controller."""
    return state


def _get_or_init_private_participants(state: Any) -> dict[str, Any]:
    private_participants = state.buffers.get(_PRIVATE_KEY)
    if isinstance(private_participants, dict):
        return private_participants
    task_state = state.task_state
    starting_money = float(task_state.get("starting_money", 200.0)) if isinstance(task_state, dict) else 200.0
    participants = task_state.get("participants", {}) if isinstance(task_state, dict) else {}
    private_participants = {
        agent_id: {"money": starting_money, "investment_history": []}
        for agent_id in participants.keys()
        if isinstance(agent_id, str) and agent_id
    }
    state.buffers[_PRIVATE_KEY] = private_participants
    return private_participants


def daytrader_apply_action(
    state: Any,
    actor_id: str,
    action: dict[str, Any],
    emit_event: Callable[..., dict[str, Any]],
) -> bool:
    """Apply DayTrader-specific investment actions against task state."""

    action_type = action.get("type")
    if action_type not in ("make_individual_investment", "make_group_investment"):
        return False
    payload = action.get("payload", {})
    if not isinstance(payload, dict):
        return False
    task_state = state.task_state
    if not isinstance(task_state, dict) or task_state.get("task_type") != "daytrader":
        return False
    participants = task_state.get("participants", {})
    if not isinstance(participants, dict) or actor_id not in participants:
        return False

    private_participants = _get_or_init_private_participants(state)
    me = private_participants.get(actor_id)
    if not isinstance(me, dict):
        return False

    invest_price = payload.get("invest_price")
    if not isinstance(invest_price, (int, float)):
        return False
    invest_price = float(invest_price)

    rules = task_state.get("rules", {})
    max_trade_price = rules.get("max_trade_price")

    if action_type == "make_individual_investment":
        if invest_price <= 0:
            return False
        min_trade_price = rules.get("min_trade_price")
        if isinstance(min_trade_price, (int, float)) and invest_price < float(min_trade_price):
            return False
        if isinstance(max_trade_price, (int, float)) and invest_price > float(max_trade_price):
            return False
        money = float(me.get("money", 0.0))
        if money < invest_price:
            emit_event(
                event_type="action_rejected",
                actor_id=actor_id,
                visibility="system",
                payload={"action": action, "error_message": "Insufficient funds."},
            )
            return True
        reward = invest_price * 2.0
        me["money"] = money - invest_price + reward
        me.setdefault("investment_history", []).append({
            "investment_amount": invest_price,
            "investment_type": "individual",
            "reward": reward,
            "money_before": money,
            "money_after": me["money"],
            "step_index": task_state.get("steps_taken", 0),
        })
        emit_event(
            event_type="individual_investment_made",
            actor_id=actor_id,
            visibility="private",
            payload={
                "invest_price": invest_price,
                "reward": reward,
                "money_after": me["money"],
                "recipients": [actor_id],
            },
        )
        return True

    # make_group_investment
    if invest_price < 0:
        return False
    if isinstance(max_trade_price, (int, float)) and invest_price > float(max_trade_price):
        return False
    money = float(me.get("money", 0.0))
    if invest_price > 0 and money < invest_price:
        emit_event(
            event_type="action_rejected",
            actor_id=actor_id,
            visibility="system",
            payload={"action": action, "error_message": "Insufficient funds."},
        )
        return True
    me["money"] = money - invest_price
    group_pool = state.buffers.get(_GROUP_POOL_KEY)
    if not isinstance(group_pool, dict):
        group_pool = {}
        state.buffers[_GROUP_POOL_KEY] = group_pool
    group_pool[actor_id] = float(group_pool.get(actor_id, 0.0)) + invest_price
    me.setdefault("investment_history", []).append({
        "investment_amount": invest_price,
        "investment_type": "group",
        "money_before": money,
        "money_after": me["money"],
        "step_index": task_state.get("steps_taken", 0),
    })
    emit_event(
        event_type="group_investment_contributed",
        actor_id=actor_id,
        visibility="private",
        payload={
            "invest_price": invest_price,
            "money_after": me["money"],
            "recipients": [actor_id],
        },
    )
    return True


def daytrader_settle_group_pool(
    state: Any,
    emit_event: Callable[..., dict[str, Any]],
) -> None:
    """Distribute group pool to all participants at end of decision phase.

    Each participant receives the total pooled amount regardless of their
    individual contribution (public goods game).
    """

    task_state = state.task_state
    if not isinstance(task_state, dict):
        return
    participants = task_state.get("participants", {})
    if not isinstance(participants, dict) or not participants:
        return
    private_participants = state.buffers.get(_PRIVATE_KEY)
    if not isinstance(private_participants, dict):
        return
    group_pool = state.buffers.get(_GROUP_POOL_KEY, {})
    if not isinstance(group_pool, dict):
        group_pool = {}
    total = sum(float(v) for v in group_pool.values() if isinstance(v, (int, float)))
    contributions_snapshot = dict(group_pool)
    for agent_id in participants.keys():
        if not isinstance(agent_id, str) or not agent_id:
            continue
        me = private_participants.get(agent_id)
        if not isinstance(me, dict):
            continue
        money = float(me.get("money", 0.0))
        me["money"] = money + total
        emit_event(
            event_type="group_pool_settled",
            actor_id=agent_id,
            visibility="private",
            payload={
                "group_pool_total": total,
                "contributions": contributions_snapshot,
                "money_after": me["money"],
                "recipients": [agent_id],
            },
        )
    state.buffers[_GROUP_POOL_KEY] = {}


def daytrader_snapshot_round_start_money(state: Any) -> None:
    """Record each participant's current money as the baseline for the next round's earnings calculation."""
    private_participants = state.buffers.get(_PRIVATE_KEY)
    if not isinstance(private_participants, dict):
        return
    state.buffers[_ROUND_START_MONEY_KEY] = {
        agent_id: float(info.get("money", 0.0))
        for agent_id, info in private_participants.items()
        if isinstance(info, dict)
    }


def daytrader_award_round_bonus(
    state: Any,
    emit_event: Callable[..., dict[str, Any]],
) -> None:
    """Award $90 to the top earner(s) of the just-settled decision round.

    Skipped for round 1. Ties split the bonus equally.
    """
    task_state = state.task_state
    if not isinstance(task_state, dict):
        return
    round_index = task_state.get("round_index", 1)
    if not isinstance(round_index, int) or round_index < 2:
        return
    participants = task_state.get("participants", {})
    if not isinstance(participants, dict) or not participants:
        return
    private_participants = state.buffers.get(_PRIVATE_KEY)
    if not isinstance(private_participants, dict):
        return
    starting_money = float(task_state.get("starting_money", 200.0))
    round_start = state.buffers.get(_ROUND_START_MONEY_KEY, {})

    earnings: dict[str, float] = {}
    for agent_id in participants:
        current = float(private_participants.get(agent_id, {}).get("money", 0.0)
                        if isinstance(private_participants.get(agent_id), dict) else 0.0)
        baseline = float(round_start.get(agent_id, starting_money))
        earnings[agent_id] = current - baseline

    max_earning = max(earnings.values())
    winners = sorted(a for a, e in earnings.items() if e == max_earning)
    bonus_each = int(_ROUND_BONUS // len(winners))

    for winner in winners:
        me = private_participants.get(winner)
        if not isinstance(me, dict):
            continue
        me["money"] = float(me.get("money", 0.0)) + bonus_each
        emit_event(
            event_type="round_bonus_awarded",
            actor_id=winner,
            visibility="public",
            payload={
                "round_index": round_index,
                "bonus": bonus_each,
                "earnings_this_round": earnings[winner],
                "winners": winners,
                "money_after": me["money"],
            },
        )


DAYTRADER_TASK = TaskDefinition(
    name="daytrader",
    init_state=daytrader_init_state,
    step=daytrader_step,
    apply_action=daytrader_apply_action,
)
