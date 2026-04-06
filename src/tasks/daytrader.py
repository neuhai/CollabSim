"""DayTrader task runtime with investment actions."""

from __future__ import annotations

from typing import Any, Callable

from src.tasks.registry import TaskDefinition


def daytrader_init_state(config: dict[str, Any]) -> dict[str, Any]:
    """Initialize DayTrader task state."""

    task_cfg = config.get("task", {})
    target_steps = task_cfg.get("target_steps", 30)
    if not isinstance(target_steps, int) or target_steps <= 0:
        raise ValueError("task.target_steps must be a positive integer for daytrader.")
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
        participants[agent_id] = {
            "money": float(starting_money),
            "investment_history": [],
        }
    return {
        "task_type": "daytrader",
        "target_steps": target_steps,
        "steps_taken": 0,
        "complete": False,
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
    target_steps = state.get("target_steps", 0)
    if not isinstance(steps_taken, int) or not isinstance(target_steps, int):
        raise ValueError("daytrader state steps must be integers.")
    state["steps_taken"] = steps_taken + 1
    if state["steps_taken"] >= target_steps:
        state["complete"] = True
    return state


def daytrader_apply_action(
    state: Any,
    actor_id: str,
    action: dict[str, Any],
    emit_event: Callable[..., dict[str, Any]],
) -> bool:
    """Apply DayTrader-specific actions against task state."""

    if action.get("type") != "make_investment":
        return False
    payload = action.get("payload", {})
    if not isinstance(payload, dict):
        return False
    task_state = state.task_state
    if not isinstance(task_state, dict) or task_state.get("task_type") != "daytrader":
        return False
    participants = task_state.get("participants", {})
    if not isinstance(participants, dict):
        return False
    me = participants.get(actor_id)
    if not isinstance(me, dict):
        return False

    invest_price = payload.get("invest_price")
    invest_decision_type = payload.get("invest_decision_type")
    if not isinstance(invest_price, (int, float)) or float(invest_price) <= 0:
        return False
    if invest_decision_type not in ("individual", "group"):
        return False
    rules = task_state.get("rules", {})
    min_trade_price = rules.get("min_trade_price")
    max_trade_price = rules.get("max_trade_price")
    if isinstance(min_trade_price, (int, float)) and float(invest_price) < float(min_trade_price):
        return False
    if isinstance(max_trade_price, (int, float)) and float(invest_price) > float(max_trade_price):
        return False
    money = me.get("money", 0.0)
    if not isinstance(money, (int, float)) or float(money) < float(invest_price):
        emit_event(
            event_type="action_rejected",
            actor_id=actor_id,
            visibility="system",
            payload={"action": {"type": "make_investment", "payload": payload}, "error_message": "Insufficient funds."},
        )
        return True

    me["money"] = float(money) - float(invest_price)
    history = me.setdefault("investment_history", [])
    if not isinstance(history, list):
        history = []
        me["investment_history"] = history
    history.append(
        {
            "investment_amount": float(invest_price),
            "investment_type": invest_decision_type,
            "money_before": float(money),
            "money_after": me["money"],
            "step_index": task_state.get("steps_taken", 0),
        }
    )
    emit_event(
        event_type="investment_made",
        actor_id=actor_id,
        visibility="public",
        payload={
            "invest_price": float(invest_price),
            "invest_decision_type": invest_decision_type,
            "money_after": me["money"],
        },
    )
    return True


DAYTRADER_TASK = TaskDefinition(
    name="daytrader",
    init_state=daytrader_init_state,
    step=daytrader_step,
    apply_action=daytrader_apply_action,
)
