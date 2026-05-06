"""Task-specific outcome metrics for all four experiment types.

Each public function accepts a :class:`~analysis.trace_parser.Trace` and
returns two dicts:
  per_run   — scalar metrics describing the whole session
  per_agent — {agent_id: {metric: value, …}}

Caller convenience:
  compute_task_metrics(trace) dispatches to the right function automatically.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from analysis.trace_parser import Trace


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #

def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _message_stats(events: list[dict[str, Any]], actor_id: str | None = None
                   ) -> dict[str, float | None]:
    """Count messages and compute average length from message_delivered events."""
    msgs = [
        e for e in events
        if e.get("event_type") == "message_delivered"
        and (actor_id is None or e.get("actor_id") == actor_id)
    ]
    count = float(len(msgs))
    lengths = []
    for e in msgs:
        content = (e.get("payload") or {}).get("content", "")
        if isinstance(content, str):
            lengths.append(float(len(content.split())))
    return {
        "messages_sent": count,
        "avg_message_length_words": _mean(lengths),
    }


# ------------------------------------------------------------------ #
# ShapeFactory
# ------------------------------------------------------------------ #

def _shapefactory_metrics(trace: Trace) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """
    Key metrics (mirrors collaborator's log_analysis.py):
      per-agent : final_wealth, wealth_gain, successful_trades,
                  avg_trade_price, min_trade_price, max_trade_price,
                  messages_sent, avg_message_length_words, trade_efficiency
      per-run   : total_successful_trades, trade_accept_rate,
                  avg_trade_price, session_avg_wealth, wealth_gini
    """
    per_agent: dict[str, dict[str, float]] = defaultdict(dict)
    per_run: dict[str, float] = {}

    # Starting money from manifest config
    task_cfg = (trace.manifest.get("config") or {}).get("task") or {}
    starting_money = float(task_cfg.get("starting_money", 200.0))

    # Trade events
    trade_created = trace.events_of_type("trade_offer_created")
    trade_responded = trace.events_of_type("trade_offer_responded")
    accepted = [e for e in trade_responded if (e.get("payload") or {}).get("response_type") == "accept"]

    trade_prices: list[float] = []
    successful_trades_by_agent: dict[str, int] = defaultdict(int)
    trade_prices_by_agent: dict[str, list[float]] = defaultdict(list)

    for e in accepted:
        payload = e.get("payload") or {}
        price = payload.get("price_per_unit")
        actor = e.get("actor_id")  # the agent who accepted (buyer)
        initiator = payload.get("initiator_id") or payload.get("target_id")
        for aid in filter(None, [actor, initiator]):
            if isinstance(aid, str):
                successful_trades_by_agent[aid] += 1
                if isinstance(price, (int, float)):
                    trade_prices_by_agent[aid].append(float(price))
        if isinstance(price, (int, float)):
            trade_prices.append(float(price))

    # Wealth from final task_state via observation_built events or task_complete
    wealth_by_agent: dict[str, float] = {}
    for e in reversed(trace.events):
        payload = e.get("payload") or {}
        obs = payload.get("observation") or {}
        state = obs.get("state") or {}
        task_state = state.get("task_state") or {}
        participants = task_state.get("participants") or {}
        if isinstance(participants, dict) and participants:
            for aid, info in participants.items():
                if isinstance(info, dict) and "money" in info and aid not in wealth_by_agent:
                    wealth_by_agent[aid] = float(info["money"])
        if len(wealth_by_agent) >= len(trace.agent_ids):
            break

    # Fallback: reconstruct from transfer events if observation snapshots unavailable
    if not wealth_by_agent:
        balance: dict[str, float] = {aid: starting_money for aid in trace.agent_ids}
        for e in trace.events:
            p = e.get("payload") or {}
            if e.get("event_type") == "resource_transferred":
                frm = p.get("from")
                to = p.get("to")
                amt = p.get("amount")
                if isinstance(amt, (int, float)):
                    if isinstance(frm, str) and frm in balance:
                        balance[frm] -= float(amt)
                    if isinstance(to, str) and to in balance:
                        balance[to] += float(amt)
        wealth_by_agent = dict(balance)

    all_wealth = list(wealth_by_agent.values())
    per_run["session_avg_wealth"] = _mean(all_wealth) or 0.0
    per_run["total_successful_trades"] = float(sum(successful_trades_by_agent.values()) // 2 or len(accepted))
    per_run["trade_offers_created"] = float(len(trade_created))
    per_run["trade_accept_rate"] = len(accepted) / len(trade_created) if trade_created else 0.0
    per_run["avg_trade_price"] = _mean(trade_prices) or 0.0
    per_run["min_trade_price"] = min(trade_prices) if trade_prices else 0.0
    per_run["max_trade_price"] = max(trade_prices) if trade_prices else 0.0

    # Wealth Gini coefficient
    if len(all_wealth) >= 2 and sum(all_wealth) > 0:
        n = len(all_wealth)
        s = sorted(all_wealth)
        gini = sum(abs(s[i] - s[j]) for i in range(n) for j in range(n)) / (2 * n * sum(s))
        per_run["wealth_gini"] = gini

    msg_stats = _message_stats(trace.events)
    per_run["messages_sent_total"] = msg_stats["messages_sent"]

    for aid in trace.agent_ids:
        wealth = wealth_by_agent.get(aid, starting_money)
        trades = successful_trades_by_agent.get(aid, 0)
        prices = trade_prices_by_agent.get(aid, [])
        msgs = _message_stats(trace.events, aid)
        msg_count = msgs["messages_sent"]
        per_agent[aid] = {
            "final_wealth": wealth,
            "wealth_gain": wealth - starting_money,
            "successful_trades": float(trades),
            "avg_trade_price": _mean(prices) or 0.0,
            "min_trade_price": min(prices) if prices else 0.0,
            "max_trade_price": max(prices) if prices else 0.0,
            "messages_sent": msg_count,
            "avg_message_length_words": msgs["avg_message_length_words"] or 0.0,
            # trades per message sent — communication efficiency
            "trade_efficiency": float(trades) / msg_count if msg_count > 0 else 0.0,
        }

    return per_run, dict(per_agent)


# ------------------------------------------------------------------ #
# DayTrader
# ------------------------------------------------------------------ #

def _daytrader_metrics(trace: Trace) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """
    Key metrics:
      per-agent : final_balance, net_return, total_individual_invested,
                  total_group_invested, group_investment_rate,
                  individual_investment_count, group_investment_count
      per-run   : rounds_completed, cooperation_rate (fraction of investment
                  actions that were group), avg_group_pool_per_round,
                  avg_net_return
    """
    per_agent: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "final_balance": 0.0,
        "net_return": 0.0,
        "total_individual_invested": 0.0,
        "total_group_invested": 0.0,
        "individual_investment_count": 0,
        "group_investment_count": 0,
    })
    per_run: dict[str, float] = {}

    task_cfg = (trace.manifest.get("config") or {}).get("task") or {}
    starting_money = float(task_cfg.get("starting_money", 200.0))

    # Individual investments
    for e in trace.events_of_type("individual_investment_made"):
        aid = e.get("actor_id")
        p = e.get("payload") or {}
        invest = p.get("invest_price", 0.0)
        if isinstance(aid, str) and isinstance(invest, (int, float)):
            per_agent[aid]["total_individual_invested"] += float(invest)
            per_agent[aid]["individual_investment_count"] += 1

    # Group contributions
    for e in trace.events_of_type("group_investment_contributed"):
        aid = e.get("actor_id")
        p = e.get("payload") or {}
        invest = p.get("invest_price", 0.0)
        if isinstance(aid, str) and isinstance(invest, (int, float)):
            per_agent[aid]["total_group_invested"] += float(invest)
            per_agent[aid]["group_investment_count"] += 1

    # Group pool settlements — track pool totals per round
    pool_totals: list[float] = []
    for e in trace.events_of_type("group_pool_settled"):
        p = e.get("payload") or {}
        total = p.get("group_pool_total")
        if isinstance(total, (int, float)) and total not in pool_totals:
            pool_totals.append(float(total))
        # Final balance after settlement per agent
        aid = e.get("actor_id")
        money_after = p.get("money_after")
        if isinstance(aid, str) and isinstance(money_after, (int, float)):
            per_agent[aid]["final_balance"] = float(money_after)

    # If no settlement events, try to get balance from observation snapshots
    for aid in trace.agent_ids:
        if per_agent[aid]["final_balance"] == 0.0:
            per_agent[aid]["final_balance"] = starting_money

    # Net return and group investment rate per agent
    total_actions = 0
    total_group_actions = 0
    net_returns: list[float] = []
    for aid in trace.agent_ids:
        d = per_agent[aid]
        d["net_return"] = d["final_balance"] - starting_money
        net_returns.append(d["net_return"])
        ind_c = d["individual_investment_count"]
        grp_c = d["group_investment_count"]
        total_inv = ind_c + grp_c
        d["group_investment_rate"] = grp_c / total_inv if total_inv > 0 else 0.0
        total_actions += total_inv
        total_group_actions += grp_c
        # cast counts to float for CSV
        d["individual_investment_count"] = float(d["individual_investment_count"])
        d["group_investment_count"] = float(d["group_investment_count"])

    # Task state for rounds_completed
    task_state = {}
    for e in reversed(trace.events):
        p = e.get("payload") or {}
        obs = p.get("observation") or {}
        ts = (obs.get("state") or {}).get("task_state") or {}
        if ts.get("task_type") == "daytrader":
            task_state = ts
            break

    per_run["rounds_completed"] = float(task_state.get("rounds_completed", 0))
    per_run["cooperation_rate"] = total_group_actions / total_actions if total_actions > 0 else 0.0
    per_run["avg_group_pool_per_round"] = _mean(pool_totals) or 0.0
    per_run["avg_net_return"] = _mean(net_returns) or 0.0
    per_run["messages_sent_total"] = _message_stats(trace.events)["messages_sent"]

    return per_run, {k: dict(v) for k, v in per_agent.items()}


# ------------------------------------------------------------------ #
# Hidden Profile
# ------------------------------------------------------------------ #

def _hidden_profile_metrics(trace: Trace) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """
    Key metrics:
      per-agent : initial_vote, final_vote, vote_changed,
                  messages_sent, avg_message_length_words
      per-run   : consensus_reached, vote_change_rate,
                  information_pooling_score (fraction of messages that
                  contain private candidate info from probe logs),
                  messages_sent_total
    """
    per_agent: dict[str, dict[str, Any]] = {aid: {} for aid in trace.agent_ids}
    per_run: dict[str, float] = {}

    # Votes
    for e in trace.events_of_type("hidden_profile_initial_vote_submitted"):
        aid = e.get("actor_id")
        choice = (e.get("payload") or {}).get("choice")
        if isinstance(aid, str) and isinstance(choice, str):
            per_agent.setdefault(aid, {})["initial_vote"] = choice

    for e in trace.events_of_type("hidden_profile_final_vote_submitted"):
        aid = e.get("actor_id")
        choice = (e.get("payload") or {}).get("choice")
        if isinstance(aid, str) and isinstance(choice, str):
            per_agent.setdefault(aid, {})["final_vote"] = choice

    vote_changed_count = 0
    final_votes: list[str] = []
    for aid, d in per_agent.items():
        iv = d.get("initial_vote")
        fv = d.get("final_vote")
        changed = (iv is not None and fv is not None and iv != fv)
        d["vote_changed"] = 1.0 if changed else 0.0
        if changed:
            vote_changed_count += 1
        if isinstance(fv, str):
            final_votes.append(fv)
        msgs = _message_stats(trace.events, aid)
        d["messages_sent"] = msgs["messages_sent"]
        d["avg_message_length_words"] = msgs["avg_message_length_words"] or 0.0

    # Consensus: all final votes identical
    per_run["consensus_reached"] = 1.0 if len(set(final_votes)) == 1 and final_votes else 0.0
    per_run["vote_change_rate"] = vote_changed_count / len(per_agent) if per_agent else 0.0

    # Vote distribution
    for candidate in set(final_votes):
        per_run[f"final_votes_for_{candidate}"] = float(final_votes.count(candidate))

    per_run["messages_sent_total"] = _message_stats(trace.events)["messages_sent"]

    # Information pooling score:
    # Count how many broadcast messages contain text that references candidate info
    # that was in private facts (proxied by checking if messages mention candidate
    # names — a rough but computable signal without NLP).
    candidate_mentions: dict[str, int] = defaultdict(int)
    for e in trace.events_of_type("message_delivered"):
        content = ((e.get("payload") or {}).get("content") or "").lower()
        for label in ["candidate a", "candidate b", "candidate c", "candidate d"]:
            if label in content:
                candidate_mentions[label] += 1

    total_msgs = int(per_run["messages_sent_total"])
    if total_msgs > 0:
        per_run["msg_fraction_mentioning_candidate_c"] = (
            candidate_mentions.get("candidate c", 0) / total_msgs
        )
    for label, count in candidate_mentions.items():
        per_run[f"candidate_mentions_{label.replace(' ', '_')}"] = float(count)

    return per_run, {k: {mk: (float(mv) if isinstance(mv, (int, float)) else mv)
                         for mk, mv in v.items()} for k, v in per_agent.items()}


# ------------------------------------------------------------------ #
# MapTask
# ------------------------------------------------------------------ #

def _maptask_metrics(trace: Trace) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """
    Key metrics:
      per-agent : messages_sent, avg_message_length_words, map_progress_updates
      per-run   : route_score, route_score_max, route_similarity,
                  route_completion_rate, map_progress_updates_total,
                  messages_sent_total, communication_efficiency
                  (route_score / messages_sent_total)
    """
    per_agent: dict[str, dict[str, float]] = {}
    per_run: dict[str, float] = {}

    # Route scoring from final task_state
    task_state: dict[str, Any] = {}
    for e in reversed(trace.events):
        p = e.get("payload") or {}
        obs = p.get("observation") or {}
        ts = (obs.get("state") or {}). get("task_state") or {}
        if ts.get("task_type") == "maptask":
            task_state = ts
            break

    route_score = task_state.get("maptask_route_score")
    route_score_max = task_state.get("maptask_route_score_max")
    route_similarity = task_state.get("maptask_route_similarity")

    if isinstance(route_score, (int, float)):
        per_run["route_score"] = float(route_score)
    if isinstance(route_score_max, (int, float)):
        per_run["route_score_max"] = float(route_score_max)
        if isinstance(route_score, (int, float)) and route_score_max > 0:
            per_run["route_completion_rate"] = float(route_score) / float(route_score_max)
    if isinstance(route_similarity, (int, float)):
        per_run["route_similarity"] = float(route_similarity)

    # Map progress updates
    updates_total = len(trace.events_of_type("map_progress_updated"))
    per_run["map_progress_updates_total"] = float(updates_total)

    # Message stats
    msg_stats = _message_stats(trace.events)
    per_run["messages_sent_total"] = msg_stats["messages_sent"]
    total_msgs = msg_stats["messages_sent"]
    if isinstance(route_score, (int, float)) and total_msgs > 0:
        per_run["communication_efficiency"] = float(route_score) / total_msgs

    for aid in trace.agent_ids:
        msgs = _message_stats(trace.events, aid)
        upd = float(sum(
            1 for e in trace.events_by_actor(aid)
            if e.get("event_type") == "map_progress_updated"
        ))
        per_agent[aid] = {
            "messages_sent": msgs["messages_sent"],
            "avg_message_length_words": msgs["avg_message_length_words"] or 0.0,
            "map_progress_updates": upd,
        }

    return per_run, per_agent


# ------------------------------------------------------------------ #
# Dispatcher
# ------------------------------------------------------------------ #

_DISPATCH = {
    "shapefactory": _shapefactory_metrics,
    "daytrader": _daytrader_metrics,
    "hidden_profile": _hidden_profile_metrics,
    "maptask": _maptask_metrics,
}


def compute_task_metrics(
    trace: Trace,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Dispatch to the correct task metrics function and return (per_run, per_agent)."""
    task_type = trace.task_type
    fn = _DISPATCH.get(task_type or "")
    if fn is None:
        return {}, {}
    return fn(trace)


def task_summary_rows(trace: Trace) -> list[dict[str, Any]]:
    """Flatten task metrics into one row per agent plus one run-level row."""
    per_run, per_agent = compute_task_metrics(trace)
    run_id = trace.manifest.get("run_id", str(trace.run_dir.name))
    task_type = trace.task_type or "unknown"
    rows: list[dict[str, Any]] = []

    # Run-level row
    rows.append({"run_id": run_id, "task_type": task_type, "agent_id": "_run_", **per_run})

    # Per-agent rows
    for agent_id, metrics in per_agent.items():
        rows.append({"run_id": run_id, "task_type": task_type, "agent_id": agent_id, **metrics})

    return rows
