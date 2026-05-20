"""ShapeFactory task runtime with domain-specific actions."""

from __future__ import annotations

from typing import Any, Callable

from src.tasks.registry import TaskDefinition


def shapefactory_init_state(config: dict[str, Any]) -> dict[str, Any]:
    """Initialize ShapeFactory task state."""

    task_cfg = config.get("task", {})
    target_steps = task_cfg.get("target_steps", 30)
    if not isinstance(target_steps, int) or target_steps <= 0:
        raise ValueError("task.target_steps must be a positive integer for shapefactory.")
    shape_options = task_cfg.get(
        "shape_options",
        ["circle", "square", "triangle", "rectangle", "diamond", "pentagon", "hexagon", "star"],
    )
    if not isinstance(shape_options, list) or not shape_options:
        raise ValueError("task.shape_options must be a non-empty list for shapefactory.")
    shape_options = [item for item in shape_options if isinstance(item, str) and item]
    if not shape_options:
        raise ValueError("task.shape_options must contain at least one non-empty shape string.")

    starting_money = task_cfg.get("starting_money", 200.0)
    regular_cost = task_cfg.get("regular_cost", 40.0)
    specialty_cost = task_cfg.get("specialty_cost", 15.0)
    min_trade_price = task_cfg.get("min_trade_price", 15.0)
    max_trade_price = task_cfg.get("max_trade_price", 100.0)
    incentive_money = task_cfg.get("incentive_money", 60.0)
    max_production_num = task_cfg.get("max_production_num", 3)
    shapes_order = task_cfg.get("shapes_order", 4)
    specialties = task_cfg.get("specialties", {})

    if not isinstance(specialties, dict):
        specialties = {}
    agents = config.get("agents", [])
    participants: dict[str, dict[str, Any]] = {}
    agent_ids: list[str] = []
    for idx, agent in enumerate(agents):
        if not isinstance(agent, dict):
            continue
        agent_id = agent.get("id")
        if not isinstance(agent_id, str) or not agent_id:
            continue
        agent_ids.append(agent_id)
        specialty = specialties.get(agent_id)
        if not isinstance(specialty, str) or specialty not in shape_options:
            specialty = shape_options[idx % len(shape_options)]
        tasks = [shape_options[(idx + i) % len(shape_options)] for i in range(max(int(shapes_order), 1))]
        participants[agent_id] = {
            "money": float(starting_money),
            "specialty": specialty,
            "production_number": 0,
            "order_progress": 0,
            "inventory": [],
            "tasks": tasks,
            "in_production": [],
        }

    protocol_cfg = config.get("protocol", {})
    if not isinstance(protocol_cfg, dict):
        protocol_cfg = {}
    prod_time = task_cfg.get("production_time")
    if prod_time is None:
        delay = protocol_cfg.get("produce_shape_delay_sec")
        prod_time = float(delay) if isinstance(delay, (int, float)) else 10.0
    else:
        prod_time = float(prod_time)

    return {
        "task_type": "shapefactory",
        "target_steps": target_steps,
        "steps_taken": 0,
        "complete": False,
        "shape_options": shape_options,
        "participants": participants,
        "agent_ids": agent_ids,
        "pending_offers": [],
        "completed_trades": [],
        "next_offer_seq": 1,
        "rules": {
            "regular_cost": float(regular_cost),
            "specialty_cost": float(specialty_cost),
            "min_trade_price": float(min_trade_price),
            "max_trade_price": float(max_trade_price),
            "incentive_money": float(incentive_money),
            "max_production_num": int(max_production_num),
            "starting_money": float(starting_money),
            "shapes_order": int(max(int(shapes_order), 1)),
            "production_time": float(prod_time),
        },
    }


def shapefactory_step(state: dict[str, Any]) -> dict[str, Any]:
    """Advance ShapeFactory task by one controller step."""

    if state.get("complete") is True:
        return state
    steps_taken = state.get("steps_taken", 0)
    target_steps = state.get("target_steps", 0)
    if not isinstance(steps_taken, int) or not isinstance(target_steps, int):
        raise ValueError("shapefactory state steps must be integers.")
    state["steps_taken"] = steps_taken + 1
    if state["steps_taken"] >= target_steps:
        state["complete"] = True
    return state


def shapefactory_apply_action(
    state: Any,
    actor_id: str,
    action: dict[str, Any],
    emit_event: Callable[..., dict[str, Any]],
) -> bool:
    """Apply ShapeFactory-specific actions against task state."""

    if not isinstance(actor_id, str) or not actor_id:
        return False
    task_state = state.task_state
    if not isinstance(task_state, dict):
        return False
    if task_state.get("task_type") != "shapefactory":
        return False

    action_type = action.get("type")
    payload = action.get("payload", {})
    if not isinstance(payload, dict):
        return False

    participants = task_state.get("participants", {})
    if not isinstance(participants, dict):
        return False
    me = participants.get(actor_id)
    if not isinstance(me, dict):
        return False

    if action_type == "produce_shape":
        return _apply_produce_shape(task_state, actor_id, me, payload, emit_event)
    if action_type == "propose_trade_offer":
        return _apply_propose_trade_offer(task_state, actor_id, participants, payload, emit_event)
    if action_type == "trade_response":
        return _apply_trade_response(task_state, actor_id, participants, payload, emit_event)
    if action_type == "cancel_trade_offer":
        return _apply_cancel_trade_offer(task_state, actor_id, payload, emit_event)
    if action_type == "fulfill_order":
        return _apply_fulfill_order(task_state, actor_id, me, payload, emit_event)
    return False


def _apply_produce_shape(
    task_state: dict[str, Any],
    actor_id: str,
    me: dict[str, Any],
    payload: dict[str, Any],
    emit_event: Callable[..., dict[str, Any]],
) -> bool:
    shape = payload.get("shape")
    quantity = payload.get("quantity", 1)
    if not isinstance(shape, str) or not shape:
        return False
    if not isinstance(quantity, int) or quantity <= 0:
        return False
    rules = task_state.get("rules", {})
    specialty = me.get("specialty")
    unit_cost = rules.get("specialty_cost") if specialty == shape else rules.get("regular_cost")
    if not isinstance(unit_cost, (int, float)):
        return False
    max_production_num = rules.get("max_production_num", 0)
    production_number = me.get("production_number", 0)
    if not isinstance(max_production_num, int) or not isinstance(production_number, int):
        return False
    if production_number + quantity > max_production_num:
        emit_event(
            event_type="action_rejected",
            actor_id=actor_id,
            visibility="system",
            payload={"action": {"type": "produce_shape", "payload": payload}, "error_message": "Production cap reached."},
        )
        return True
    total_cost = float(unit_cost) * float(quantity)
    money = me.get("money", 0.0)
    if not isinstance(money, (int, float)) or float(money) < total_cost:
        emit_event(
            event_type="action_rejected",
            actor_id=actor_id,
            visibility="system",
            payload={"action": {"type": "produce_shape", "payload": payload}, "error_message": "Insufficient funds."},
        )
        return True
    me["money"] = float(money) - total_cost
    inventory = me.setdefault("inventory", [])
    if isinstance(inventory, list):
        for _ in range(quantity):
            inventory.append(shape)
    me["production_number"] = production_number + quantity
    emit_event(
        event_type="shape_produced",
        actor_id=actor_id,
        visibility="public",
        payload={"shape": shape, "quantity": quantity, "money_after": me["money"]},
    )
    return True


def _apply_propose_trade_offer(
    task_state: dict[str, Any],
    actor_id: str,
    participants: dict[str, dict[str, Any]],
    payload: dict[str, Any],
    emit_event: Callable[..., dict[str, Any]],
) -> bool:
    offer_type = payload.get("offer_type")
    shape = payload.get("shape")
    target_id = payload.get("target_id")
    price = payload.get("price_per_unit")
    quantity = payload.get("quantity", 1)
    if offer_type not in ("buy", "sell"):
        return False
    if not isinstance(shape, str) or not shape:
        return False
    if not isinstance(target_id, str) or not target_id:
        return False
    if not isinstance(price, (int, float)) or float(price) <= 0:
        return False
    if not isinstance(quantity, int) or quantity <= 0:
        return False
    target = participants.get(target_id)
    me = participants.get(actor_id)
    if not isinstance(target, dict) or not isinstance(me, dict):
        return False
    rules = task_state.get("rules", {})
    min_trade_price = rules.get("min_trade_price")
    max_trade_price = rules.get("max_trade_price")
    if isinstance(min_trade_price, (int, float)) and float(price) < float(min_trade_price):
        return False
    if isinstance(max_trade_price, (int, float)) and float(price) > float(max_trade_price):
        return False
    if offer_type == "sell":
        inventory = me.get("inventory", [])
        if not isinstance(inventory, list) or inventory.count(shape) < quantity:
            return False
    if offer_type == "buy":
        money = me.get("money", 0.0)
        if not isinstance(money, (int, float)) or float(money) < float(price) * float(quantity):
            return False
    offer_seq = task_state.get("next_offer_seq", 1)
    if not isinstance(offer_seq, int):
        offer_seq = 1
    offer_id = f"offer_{stateful_step(task_state)}_{offer_seq}"
    task_state["next_offer_seq"] = offer_seq + 1
    offer = {
        "id": offer_id,
        "from": actor_id,
        "to": target_id,
        "offer_type": offer_type,
        "shape": shape,
        "quantity": quantity,
        "price_per_unit": float(price),
        "status": "pending",
    }
    pending_offers = task_state.setdefault("pending_offers", [])
    if isinstance(pending_offers, list):
        pending_offers.append(offer)
    emit_event(
        event_type="trade_offer_created",
        actor_id=actor_id,
        visibility="public",
        payload=offer,
    )
    return True


def _apply_trade_response(
    task_state: dict[str, Any],
    actor_id: str,
    participants: dict[str, dict[str, Any]],
    payload: dict[str, Any],
    emit_event: Callable[..., dict[str, Any]],
) -> bool:
    transaction_id = payload.get("transaction_id")
    response_type = payload.get("response_type")
    if not isinstance(transaction_id, str) or not transaction_id:
        return False
    if response_type not in ("accept", "decline"):
        return False
    pending_offers = task_state.get("pending_offers", [])
    if not isinstance(pending_offers, list):
        return False
    offer = None
    for item in pending_offers:
        if isinstance(item, dict) and item.get("id") == transaction_id and item.get("status") == "pending":
            offer = item
            break
    if not isinstance(offer, dict):
        return False
    if offer.get("to") != actor_id:
        return False

    if response_type == "accept":
        if not _execute_trade_offer(offer, participants):
            return False
        offer["status"] = "accepted"
    else:
        offer["status"] = "declined"
    pending_offers[:] = [item for item in pending_offers if not (isinstance(item, dict) and item.get("id") == transaction_id)]
    completed = task_state.setdefault("completed_trades", [])
    if isinstance(completed, list):
        completed.append(dict(offer))
    # Include offer economics on the event so downstream metrics (e.g. analysis/task_metrics)
    # can attribute prices without re-scanning prior trade_offer_created events.
    respond_payload: dict[str, Any] = {
        "transaction_id": transaction_id,
        "response_type": response_type,
    }
    frm = offer.get("from")
    if isinstance(frm, str) and frm:
        respond_payload["initiator_id"] = frm
    to_id = offer.get("to")
    if isinstance(to_id, str) and to_id:
        respond_payload["target_id"] = to_id
    ppu = offer.get("price_per_unit")
    if isinstance(ppu, (int, float)):
        respond_payload["price_per_unit"] = float(ppu)
    emit_event(
        event_type="trade_offer_responded",
        actor_id=actor_id,
        visibility="public",
        payload=respond_payload,
    )
    return True


def _apply_cancel_trade_offer(
    task_state: dict[str, Any],
    actor_id: str,
    payload: dict[str, Any],
    emit_event: Callable[..., dict[str, Any]],
) -> bool:
    transaction_id = payload.get("transaction_id")
    if not isinstance(transaction_id, str) or not transaction_id:
        return False
    pending_offers = task_state.get("pending_offers", [])
    if not isinstance(pending_offers, list):
        return False
    offer = None
    for item in pending_offers:
        if isinstance(item, dict) and item.get("id") == transaction_id:
            offer = item
            break
    if not isinstance(offer, dict):
        return False
    if offer.get("from") != actor_id:
        return False
    pending_offers[:] = [item for item in pending_offers if not (isinstance(item, dict) and item.get("id") == transaction_id)]
    offer["status"] = "cancelled"
    completed = task_state.setdefault("completed_trades", [])
    if isinstance(completed, list):
        completed.append(dict(offer))
    emit_event(
        event_type="trade_offer_cancelled",
        actor_id=actor_id,
        visibility="public",
        payload={"transaction_id": transaction_id},
    )
    return True


def _apply_fulfill_order(
    task_state: dict[str, Any],
    actor_id: str,
    me: dict[str, Any],
    payload: dict[str, Any],
    emit_event: Callable[..., dict[str, Any]],
) -> bool:
    order_indices = payload.get("order_indices")
    if not isinstance(order_indices, list) or not order_indices:
        return False
    if not all(isinstance(idx, int) and idx >= 0 for idx in order_indices):
        return False
    tasks = me.get("tasks", [])
    inventory = me.get("inventory", [])
    if not isinstance(tasks, list) or not isinstance(inventory, list):
        return False
    selected: list[str] = []
    for idx in order_indices:
        if idx >= len(tasks):
            return False
        selected.append(tasks[idx])
    shadow = list(inventory)
    for shape in selected:
        if shape not in shadow:
            return False
        shadow.remove(shape)
    for shape in selected:
        inventory.remove(shape)
    for idx in sorted(order_indices, reverse=True):
        tasks.pop(idx)
    incentive_money = task_state.get("rules", {}).get("incentive_money", 0.0)
    current_money = me.get("money", 0.0)
    if not isinstance(current_money, (int, float)):
        return False
    me["money"] = float(current_money) + float(incentive_money) * float(len(selected))
    order_progress = me.get("order_progress", 0)
    if not isinstance(order_progress, int):
        order_progress = 0
    me["order_progress"] = order_progress + len(selected)
    emit_event(
        event_type="order_fulfilled",
        actor_id=actor_id,
        visibility="public",
        payload={"fulfilled_count": len(selected), "money_after": me["money"]},
    )
    return True


def _execute_trade_offer(offer: dict[str, Any], participants: dict[str, dict[str, Any]]) -> bool:
    from_id = offer.get("from")
    to_id = offer.get("to")
    offer_type = offer.get("offer_type")
    shape = offer.get("shape")
    quantity = offer.get("quantity", 1)
    price_per_unit = offer.get("price_per_unit")
    if not all(isinstance(item, str) and item for item in (from_id, to_id, offer_type, shape)):
        return False
    if offer_type not in ("buy", "sell"):
        return False
    if not isinstance(quantity, int) or quantity <= 0:
        return False
    if not isinstance(price_per_unit, (int, float)) or float(price_per_unit) <= 0:
        return False
    sender = participants.get(from_id)
    receiver = participants.get(to_id)
    if not isinstance(sender, dict) or not isinstance(receiver, dict):
        return False
    total_price = float(price_per_unit) * float(quantity)
    if offer_type == "sell":
        seller = sender
        buyer = receiver
    else:
        seller = receiver
        buyer = sender
    seller_inventory = seller.get("inventory", [])
    buyer_inventory = buyer.get("inventory", [])
    seller_money = seller.get("money", 0.0)
    buyer_money = buyer.get("money", 0.0)
    if not isinstance(seller_inventory, list) or seller_inventory.count(shape) < quantity:
        return False
    if not isinstance(buyer_inventory, list):
        return False
    if not isinstance(seller_money, (int, float)) or not isinstance(buyer_money, (int, float)):
        return False
    if float(buyer_money) < total_price:
        return False
    for _ in range(quantity):
        seller_inventory.remove(shape)
        buyer_inventory.append(shape)
    seller["money"] = float(seller_money) + total_price
    buyer["money"] = float(buyer_money) - total_price
    return True


def stateful_step(task_state: dict[str, Any]) -> int:
    step = task_state.get("steps_taken", 0)
    if isinstance(step, int) and step >= 0:
        return step
    return 0


SHAPEFACTORY_TASK = TaskDefinition(
    name="shapefactory",
    init_state=shapefactory_init_state,
    step=shapefactory_step,
    apply_action=shapefactory_apply_action,
)
