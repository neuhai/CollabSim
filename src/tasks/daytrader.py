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
    return {
        "task_type": "daytrader",
        "target_rounds": target_rounds,
        "target_steps": target_rounds * 2,
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
    """Advance DayTrader task by one step."""

    if state.get("complete") is True:
        return state
    steps_taken = state.get("steps_taken", 0)
    target_rounds = state.get("target_rounds", 0)
    if not isinstance(steps_taken, int) or not isinstance(target_rounds, int):
        raise ValueError("daytrader state steps/rounds must be integers.")
    state["steps_taken"] = steps_taken + 1
    state["rounds_completed"] = state["steps_taken"] // 2
    if state["rounds_completed"] >= target_rounds:
        state["complete"] = True
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


DAYTRADER_TASK = TaskDefinition(
    name="daytrader",
    init_state=daytrader_init_state,
    step=daytrader_step,
    apply_action=daytrader_apply_action,
)
