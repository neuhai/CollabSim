"""Metrics computation scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MetricsResult:
    """Container for computed metric outputs."""

    per_agent: dict[str, dict[str, float]]
    per_run: dict[str, float]


def _compute_event_metrics(
    event_log: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    per_run: dict[str, float] = {}
    per_agent: dict[str, dict[str, float]] = {}
    if not event_log:
        return per_run, per_agent
    event_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    messages_sent = 0
    messages_received = 0
    transfer_count = 0
    transfer_amount_total = 0.0
    proposal_created = 0
    proposal_responded = 0
    decision_choice_count = 0
    trade_offers_created = 0
    trade_offers_responded = 0
    trade_offers_cancelled = 0
    trade_accept_count = 0
    orders_fulfilled_count = 0
    investments_count = 0
    investment_amount_total = 0.0
    map_progress_updates = 0
    for event in event_log:
        event_type = event.get("event_type")
        if isinstance(event_type, str) and event_type:
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        payload = event.get("payload", {})
        if event_type == "action_submitted" and isinstance(payload, dict):
            action = payload.get("action", {})
            if isinstance(action, dict):
                action_type = action.get("type")
                if isinstance(action_type, str) and action_type:
                    action_counts[action_type] = action_counts.get(action_type, 0) + 1
        if event_type == "message_delivered" and isinstance(payload, dict):
            messages_sent += 1
            recipients = payload.get("recipients")
            if isinstance(recipients, list):
                messages_received += len([item for item in recipients if isinstance(item, str) and item])
            actor_id = event.get("actor_id")
            if isinstance(actor_id, str) and actor_id:
                sender = per_agent.setdefault(actor_id, {})
                sender["messages_sent"] = sender.get("messages_sent", 0.0) + 1.0
            if isinstance(recipients, list):
                for recipient in recipients:
                    if not isinstance(recipient, str) or not recipient:
                        continue
                    receiver = per_agent.setdefault(recipient, {})
                    receiver["messages_received"] = receiver.get("messages_received", 0.0) + 1.0
        if event_type == "resource_transferred" and isinstance(payload, dict):
            transfer_count += 1
            amount = payload.get("amount")
            if isinstance(amount, (int, float)):
                transfer_amount_total += float(amount)
            sender_id = payload.get("from")
            if isinstance(sender_id, str) and sender_id:
                sender = per_agent.setdefault(sender_id, {})
                sender["transfers_sent"] = sender.get("transfers_sent", 0.0) + 1.0
                if isinstance(amount, (int, float)):
                    sender["transfer_amount_sent"] = sender.get("transfer_amount_sent", 0.0) + float(amount)
            receiver_id = payload.get("to")
            if isinstance(receiver_id, str) and receiver_id:
                receiver = per_agent.setdefault(receiver_id, {})
                receiver["transfers_received"] = receiver.get("transfers_received", 0.0) + 1.0
                if isinstance(amount, (int, float)):
                    receiver["transfer_amount_received"] = receiver.get("transfer_amount_received", 0.0) + float(amount)
        if event_type == "proposal_created":
            proposal_created += 1
        if event_type == "proposal_responded":
            proposal_responded += 1
        if event_type == "decision_revealed" and isinstance(payload, dict):
            choices = payload.get("choices")
            if isinstance(choices, list):
                decision_choice_count += len(choices)
        if event_type == "trade_offer_created":
            trade_offers_created += 1
        if event_type == "trade_offer_responded" and isinstance(payload, dict):
            trade_offers_responded += 1
            if payload.get("response_type") == "accept":
                trade_accept_count += 1
        if event_type == "trade_offer_cancelled":
            trade_offers_cancelled += 1
        if event_type == "order_fulfilled" and isinstance(payload, dict):
            fulfilled_count = payload.get("fulfilled_count")
            if isinstance(fulfilled_count, (int, float)):
                orders_fulfilled_count += int(fulfilled_count)
        if event_type == "investment_made" and isinstance(payload, dict):
            investments_count += 1
            invest_price = payload.get("invest_price")
            if isinstance(invest_price, (int, float)):
                investment_amount_total += float(invest_price)
        if event_type == "map_progress_updated":
            map_progress_updates += 1
            actor_id = event.get("actor_id")
            if isinstance(actor_id, str) and actor_id:
                agent = per_agent.setdefault(actor_id, {})
                agent["map_progress_updates"] = agent.get("map_progress_updates", 0.0) + 1.0
    for event_type, count in sorted(event_counts.items()):
        per_run[f"event_count_{event_type}"] = float(count)
    for action_type, count in sorted(action_counts.items()):
        per_run[f"action_submitted_{action_type}_count"] = float(count)
    per_run["messages_sent"] = float(messages_sent)
    per_run["messages_received"] = float(messages_received)
    if transfer_count > 0:
        per_run["transfers_count"] = float(transfer_count)
        per_run["transfer_amount_total"] = transfer_amount_total
        per_run["transfer_amount_mean"] = transfer_amount_total / float(transfer_count)
    if proposal_created > 0:
        per_run["proposal_created"] = float(proposal_created)
        per_run["proposal_responded"] = float(proposal_responded)
        per_run["proposal_response_rate"] = float(proposal_responded) / float(proposal_created)
    if decision_choice_count > 0:
        per_run["decision_choice_count"] = float(decision_choice_count)
    if trade_offers_created > 0:
        per_run["trade_offers_created"] = float(trade_offers_created)
        per_run["trade_offers_responded"] = float(trade_offers_responded)
        per_run["trade_offers_cancelled"] = float(trade_offers_cancelled)
        per_run["trade_offer_response_rate"] = float(trade_offers_responded) / float(trade_offers_created)
        per_run["trade_accept_rate"] = float(trade_accept_count) / float(trade_offers_created)
    if orders_fulfilled_count > 0:
        per_run["orders_fulfilled_count"] = float(orders_fulfilled_count)
    if investments_count > 0:
        per_run["investments_count"] = float(investments_count)
        per_run["investment_amount_total"] = float(investment_amount_total)
        per_run["investment_amount_mean"] = float(investment_amount_total) / float(investments_count)
    if map_progress_updates > 0:
        per_run["map_progress_updates"] = float(map_progress_updates)
    return per_run, per_agent


def _compute_probe_metrics(probe_log: list[dict[str, Any]]) -> dict[str, float]:
    total = len(probe_log)
    if total == 0:
        return {}
    answered = 0
    confidence_sum = 0.0
    confidence_count = 0
    construct_counts: dict[str, int] = {}
    for record in probe_log:
        if record.get("answer") is not None:
            answered += 1
        confidence = record.get("confidence")
        if isinstance(confidence, (int, float)):
            confidence_sum += float(confidence)
            confidence_count += 1
        construct = record.get("construct")
        if isinstance(construct, str) and construct:
            construct_counts[construct] = construct_counts.get(construct, 0) + 1
    metrics: dict[str, float] = {
        "probe_records": float(total),
        "probe_answered": float(answered),
        "probe_unanswered": float(total - answered),
        "probe_response_rate": float(answered) / float(total),
    }
    if confidence_count > 0:
        metrics["probe_confidence_mean"] = confidence_sum / float(confidence_count)
    for construct, count in sorted(construct_counts.items()):
        metrics[f"probe_construct_{construct}"] = float(count)
    return metrics


def _compute_probe_agent_metrics(probe_log: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    per_agent: dict[str, dict[str, float]] = {}
    if not probe_log:
        return per_agent
    totals: dict[str, int] = {}
    answered: dict[str, int] = {}
    confidence_sum: dict[str, float] = {}
    confidence_count: dict[str, int] = {}
    construct_counts: dict[str, dict[str, int]] = {}
    for record in probe_log:
        actor_id = record.get("actor_id")
        if not isinstance(actor_id, str) or not actor_id:
            continue
        totals[actor_id] = totals.get(actor_id, 0) + 1
        if record.get("answer") is not None:
            answered[actor_id] = answered.get(actor_id, 0) + 1
        confidence = record.get("confidence")
        if isinstance(confidence, (int, float)):
            confidence_sum[actor_id] = confidence_sum.get(actor_id, 0.0) + float(confidence)
            confidence_count[actor_id] = confidence_count.get(actor_id, 0) + 1
        construct = record.get("construct")
        if isinstance(construct, str) and construct:
            per_construct = construct_counts.setdefault(actor_id, {})
            per_construct[construct] = per_construct.get(construct, 0) + 1
    for actor_id, total in totals.items():
        agent_metrics: dict[str, float] = {
            "probe_records": float(total),
            "probe_answered": float(answered.get(actor_id, 0)),
            "probe_unanswered": float(total - answered.get(actor_id, 0)),
            "probe_response_rate": float(answered.get(actor_id, 0)) / float(total),
        }
        count = confidence_count.get(actor_id, 0)
        if count > 0:
            agent_metrics["probe_confidence_mean"] = confidence_sum.get(actor_id, 0.0) / float(count)
        for construct, count in sorted(construct_counts.get(actor_id, {}).items()):
            agent_metrics[f"probe_construct_{construct}"] = float(count)
        per_agent[actor_id] = agent_metrics
    return per_agent


def compute_metrics(
    event_log: list[dict[str, Any]],
    probe_log: list[dict[str, Any]],
    task_outcome: dict[str, Any] | None = None,
) -> MetricsResult:
    """Compute placeholder metrics from logs.

    This scaffolding returns a minimal counter-task metric when available.
    """

    per_run: dict[str, float] = {}
    if task_outcome is not None and task_outcome.get("task_type") == "counter":
        steps_taken = task_outcome.get("steps_taken")
        target_steps = task_outcome.get("target_steps")
        complete = task_outcome.get("complete")
        if isinstance(steps_taken, int) and isinstance(target_steps, int) and target_steps > 0:
            per_run["steps_taken"] = float(steps_taken)
            per_run["target_steps"] = float(target_steps)
            if steps_taken > 0:
                per_run["efficiency"] = float(target_steps) / float(steps_taken)
        if isinstance(complete, bool):
            per_run["completed"] = 1.0 if complete else 0.0
    if task_outcome is not None and task_outcome.get("task_type") == "accumulator":
        steps_taken = task_outcome.get("steps_taken")
        target_value = task_outcome.get("target_value")
        current_value = task_outcome.get("current_value")
        increment = task_outcome.get("increment")
        complete = task_outcome.get("complete")
        if isinstance(steps_taken, int):
            per_run["steps_taken"] = float(steps_taken)
        if isinstance(target_value, int):
            per_run["target_value"] = float(target_value)
        if isinstance(current_value, int):
            per_run["current_value"] = float(current_value)
            if current_value > 0 and isinstance(target_value, int):
                per_run["efficiency"] = float(target_value) / float(current_value)
        if isinstance(increment, int):
            per_run["increment"] = float(increment)
        if isinstance(complete, bool):
            per_run["completed"] = 1.0 if complete else 0.0
    if task_outcome is not None and task_outcome.get("task_type") == "hidden_profile":
        steps_taken = task_outcome.get("steps_taken")
        target_steps = task_outcome.get("target_steps")
        complete = task_outcome.get("complete")
        if isinstance(steps_taken, int) and steps_taken >= 0:
            per_run["steps_taken"] = float(steps_taken)
        if isinstance(target_steps, int) and target_steps > 0:
            per_run["target_steps"] = float(target_steps)
            if isinstance(steps_taken, int) and steps_taken > 0:
                per_run["efficiency"] = float(target_steps) / float(steps_taken)
        if isinstance(complete, bool):
            per_run["completed"] = 1.0 if complete else 0.0
    if task_outcome is not None and task_outcome.get("task_type") == "shapefactory":
        steps_taken = task_outcome.get("steps_taken")
        target_steps = task_outcome.get("target_steps")
        completed_trades = task_outcome.get("completed_trades")
        pending_offers = task_outcome.get("pending_offers")
        complete = task_outcome.get("complete")
        if isinstance(steps_taken, int):
            per_run["steps_taken"] = float(steps_taken)
        if isinstance(target_steps, int) and target_steps > 0:
            per_run["target_steps"] = float(target_steps)
            if isinstance(steps_taken, int) and steps_taken > 0:
                per_run["efficiency"] = float(target_steps) / float(steps_taken)
        if isinstance(completed_trades, int):
            per_run["completed_trades"] = float(completed_trades)
        if isinstance(pending_offers, int):
            per_run["pending_offers"] = float(pending_offers)
        if isinstance(complete, bool):
            per_run["completed"] = 1.0 if complete else 0.0
    if task_outcome is not None and task_outcome.get("task_type") == "daytrader":
        steps_taken = task_outcome.get("steps_taken")
        target_steps = task_outcome.get("target_steps")
        investments_count = task_outcome.get("investments_count")
        complete = task_outcome.get("complete")
        if isinstance(steps_taken, int):
            per_run["steps_taken"] = float(steps_taken)
        if isinstance(target_steps, int) and target_steps > 0:
            per_run["target_steps"] = float(target_steps)
            if isinstance(steps_taken, int) and steps_taken > 0:
                per_run["efficiency"] = float(target_steps) / float(steps_taken)
        if isinstance(investments_count, int):
            per_run["task_investments_count"] = float(investments_count)
        if isinstance(complete, bool):
            per_run["completed"] = 1.0 if complete else 0.0
    if task_outcome is not None and task_outcome.get("task_type") == "maptask":
        steps_taken = task_outcome.get("steps_taken")
        target_steps = task_outcome.get("target_steps")
        map_progress_updates = task_outcome.get("map_progress_updates")
        complete = task_outcome.get("complete")
        if isinstance(steps_taken, int):
            per_run["steps_taken"] = float(steps_taken)
        if isinstance(target_steps, int) and target_steps > 0:
            per_run["target_steps"] = float(target_steps)
            if isinstance(steps_taken, int) and steps_taken > 0:
                per_run["efficiency"] = float(target_steps) / float(steps_taken)
        if isinstance(map_progress_updates, int):
            per_run["task_map_progress_updates"] = float(map_progress_updates)
        if isinstance(complete, bool):
            per_run["completed"] = 1.0 if complete else 0.0
    event_per_run, event_per_agent = _compute_event_metrics(event_log)
    per_run.update(event_per_run)
    per_run.update(_compute_probe_metrics(probe_log))
    per_agent = _compute_probe_agent_metrics(probe_log)
    for agent_id, metrics in event_per_agent.items():
        merged = per_agent.setdefault(agent_id, {})
        merged.update(metrics)
    return MetricsResult(per_agent=per_agent, per_run=per_run)
