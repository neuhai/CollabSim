"""Action schema definitions and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ActionValidationError(Exception):
    """Raised when an action payload fails validation."""

    message: str

    def __str__(self) -> str:
        return self.message


ACTION_TYPES: tuple[str, ...] = (
    "message",
    "decide",
    "propose",
    "respond",
    "transfer",
    "produce_shape",
    "propose_trade_offer",
    "trade_response",
    "cancel_trade_offer",
    "fulfill_order",
    "make_investment",
    "make_individual_investment",
    "make_group_investment",
    "update_map_progress",
    "draw",
    "erase",
    "undo",
    "reset",
    "do_nothing",
)


def validate_action(action: Mapping[str, Any]) -> None:
    """Validate a single action envelope and payload.

    Args:
        action: Action envelope dictionary.

    Raises:
        ActionValidationError: If validation fails.
    """

    if not isinstance(action, Mapping):
        raise ActionValidationError("Action must be a mapping/object.")

    for key in ("type", "actor_id", "timestamp", "payload"):
        if key not in action:
            raise ActionValidationError(f"Action missing required field: {key}")

    action_type = action["type"]
    if action_type not in ACTION_TYPES:
        raise ActionValidationError(f"Unsupported action type: {action_type}")

    payload = action["payload"]
    if not isinstance(payload, Mapping):
        raise ActionValidationError("Action payload must be a mapping/object.")

    if action_type == "message":
        _validate_message(payload)
    elif action_type == "decide":
        _validate_decide(payload)
    elif action_type == "propose":
        _validate_propose(payload)
    elif action_type == "respond":
        _validate_respond(payload)
    elif action_type == "transfer":
        _validate_transfer(payload)
    elif action_type == "produce_shape":
        _validate_produce_shape(payload)
    elif action_type == "propose_trade_offer":
        _validate_propose_trade_offer(payload)
    elif action_type == "trade_response":
        _validate_trade_response(payload)
    elif action_type == "cancel_trade_offer":
        _validate_cancel_trade_offer(payload)
    elif action_type == "fulfill_order":
        _validate_fulfill_order(payload)
    elif action_type == "make_investment":
        _validate_make_investment(payload)
    elif action_type == "make_individual_investment":
        _validate_make_individual_investment(payload)
    elif action_type == "make_group_investment":
        _validate_make_group_investment(payload)
    elif action_type == "update_map_progress":
        _validate_update_map_progress(payload)
    elif action_type == "draw":
        _validate_map_draw(payload)
    elif action_type == "erase":
        _validate_map_erase(payload)
    elif action_type == "undo":
        _validate_map_undo_reset(payload)
    elif action_type == "reset":
        _validate_map_undo_reset(payload)
    elif action_type == "do_nothing":
        _validate_do_nothing(payload)


def _validate_message(payload: Mapping[str, Any]) -> None:
    _require_fields(payload, ("channel", "content", "content_type"))
    channel = payload["channel"]
    if channel not in ("broadcast", "direct"):
        raise ActionValidationError("message.channel must be broadcast or direct.")
    if channel == "direct":
        recipients = payload.get("recipients")
        if not isinstance(recipients, list) or not recipients:
            raise ActionValidationError("message.recipients must be a non-empty list.")
    if payload["content_type"] not in ("text", "json"):
        raise ActionValidationError("message.content_type must be text or json.")


def _validate_decide(payload: Mapping[str, Any]) -> None:
    _require_fields(payload, ("decision_id", "choice", "reveal"))
    if payload["reveal"] not in ("sequential", "aggregated", "simultaneous"):
        raise ActionValidationError("decide.reveal must be sequential, aggregated, or simultaneous.")


def _validate_propose(payload: Mapping[str, Any]) -> None:
    _require_fields(payload, ("proposal_id", "target_ids", "terms"))
    target_ids = payload["target_ids"]
    if not isinstance(target_ids, list) or not target_ids:
        raise ActionValidationError("propose.target_ids must be a non-empty list.")
    if not isinstance(payload["terms"], Mapping):
        raise ActionValidationError("propose.terms must be an object.")


def _validate_respond(payload: Mapping[str, Any]) -> None:
    _require_fields(payload, ("proposal_id", "response"))
    response = payload["response"]
    if response not in ("accept", "reject", "counter"):
        raise ActionValidationError("respond.response must be accept, reject, or counter.")
    if response == "counter":
        if "counter_terms" not in payload or not isinstance(payload["counter_terms"], Mapping):
            raise ActionValidationError("respond.counter_terms required for counter response.")


def _validate_transfer(payload: Mapping[str, Any]) -> None:
    _require_fields(payload, ("resource_id", "amount", "to"))
    amount = payload["amount"]
    if not isinstance(amount, (int, float)) or amount <= 0:
        raise ActionValidationError("transfer.amount must be a positive number.")


def _validate_produce_shape(payload: Mapping[str, Any]) -> None:
    _require_fields(payload, ("shape",))
    shape = payload.get("shape")
    if not isinstance(shape, str) or not shape:
        raise ActionValidationError("produce_shape.shape must be a non-empty string.")
    quantity = payload.get("quantity", 1)
    if not isinstance(quantity, int) or quantity <= 0:
        raise ActionValidationError("produce_shape.quantity must be a positive integer.")


def _validate_propose_trade_offer(payload: Mapping[str, Any]) -> None:
    _require_fields(payload, ("offer_type", "shape", "price_per_unit", "target_id"))
    offer_type = payload.get("offer_type")
    if offer_type not in ("buy", "sell"):
        raise ActionValidationError("propose_trade_offer.offer_type must be buy or sell.")
    shape = payload.get("shape")
    if not isinstance(shape, str) or not shape:
        raise ActionValidationError("propose_trade_offer.shape must be a non-empty string.")
    target_id = payload.get("target_id")
    if not isinstance(target_id, str) or not target_id:
        raise ActionValidationError("propose_trade_offer.target_id must be a non-empty string.")
    price_per_unit = payload.get("price_per_unit")
    if not isinstance(price_per_unit, (int, float)) or float(price_per_unit) <= 0:
        raise ActionValidationError("propose_trade_offer.price_per_unit must be a positive number.")
    quantity = payload.get("quantity", 1)
    if not isinstance(quantity, int) or quantity <= 0:
        raise ActionValidationError("propose_trade_offer.quantity must be a positive integer.")


def _validate_trade_response(payload: Mapping[str, Any]) -> None:
    _require_fields(payload, ("transaction_id", "response_type"))
    transaction_id = payload.get("transaction_id")
    if not isinstance(transaction_id, str) or not transaction_id:
        raise ActionValidationError("trade_response.transaction_id must be a non-empty string.")
    response_type = payload.get("response_type")
    if response_type not in ("accept", "decline"):
        raise ActionValidationError("trade_response.response_type must be accept or decline.")


def _validate_cancel_trade_offer(payload: Mapping[str, Any]) -> None:
    _require_fields(payload, ("transaction_id",))
    transaction_id = payload.get("transaction_id")
    if not isinstance(transaction_id, str) or not transaction_id:
        raise ActionValidationError("cancel_trade_offer.transaction_id must be a non-empty string.")


def _validate_fulfill_order(payload: Mapping[str, Any]) -> None:
    _require_fields(payload, ("order_indices",))
    order_indices = payload.get("order_indices")
    if not isinstance(order_indices, list) or not order_indices:
        raise ActionValidationError("fulfill_order.order_indices must be a non-empty list.")
    if not all(isinstance(idx, int) and idx >= 0 for idx in order_indices):
        raise ActionValidationError("fulfill_order.order_indices must contain non-negative integers.")


def _validate_make_investment(payload: Mapping[str, Any]) -> None:
    _require_fields(payload, ("invest_price", "invest_decision_type"))
    invest_price = payload.get("invest_price")
    if not isinstance(invest_price, (int, float)) or float(invest_price) <= 0:
        raise ActionValidationError("make_investment.invest_price must be a positive number.")
    invest_decision_type = payload.get("invest_decision_type")
    if invest_decision_type not in ("individual", "group"):
        raise ActionValidationError("make_investment.invest_decision_type must be individual or group.")


def _validate_make_individual_investment(payload: Mapping[str, Any]) -> None:
    _require_fields(payload, ("invest_price",))
    invest_price = payload.get("invest_price")
    if not isinstance(invest_price, (int, float)) or float(invest_price) <= 0:
        raise ActionValidationError("make_individual_investment.invest_price must be a positive number.")


def _validate_make_group_investment(payload: Mapping[str, Any]) -> None:
    _require_fields(payload, ("invest_price",))
    invest_price = payload.get("invest_price")
    if not isinstance(invest_price, (int, float)) or float(invest_price) < 0:
        raise ActionValidationError("make_group_investment.invest_price must be a non-negative number.")


def _validate_update_map_progress(payload: Mapping[str, Any]) -> None:
    _require_fields(payload, ("map_progress",))
    map_progress = payload.get("map_progress")
    if not isinstance(map_progress, Mapping):
        raise ActionValidationError("update_map_progress.map_progress must be an object.")


def _validate_map_draw(payload: Mapping[str, Any]) -> None:
    cells = payload.get("cells")
    if cells is None:
        cells = payload.get("drawn_points")
    if not isinstance(cells, list) or not cells:
        raise ActionValidationError("draw.cells must be a non-empty list of [row, col].")
    for item in cells:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ActionValidationError("draw.cells entries must be [row, col].")
        row, col = item
        if not isinstance(row, int) or not isinstance(col, int):
            raise ActionValidationError("draw.cells rows/cols must be integers.")


def _validate_map_erase(payload: Mapping[str, Any]) -> None:
    cells = payload.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ActionValidationError("erase.cells must be a non-empty list of [row, col].")
    for item in cells:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ActionValidationError("erase.cells entries must be [row, col].")
        row, col = item
        if not isinstance(row, int) or not isinstance(col, int):
            raise ActionValidationError("erase.cells rows/cols must be integers.")


def _validate_map_undo_reset(payload: Mapping[str, Any]) -> None:
    for key in payload:
        if key == "reason":
            val = payload.get("reason")
            if val is not None and not isinstance(val, str):
                raise ActionValidationError("undo/reset.reason must be a string when provided.")
            continue
        raise ActionValidationError(f"undo/reset payload does not support field: {key}")


def _validate_do_nothing(payload: Mapping[str, Any]) -> None:
    if "reason" in payload and not isinstance(payload.get("reason"), str):
        raise ActionValidationError("do_nothing.reason must be a string when provided.")


def _require_fields(payload: Mapping[str, Any], fields: tuple[str, ...]) -> None:
    for field in fields:
        if field not in payload:
            raise ActionValidationError(f"Payload missing required field: {field}")
