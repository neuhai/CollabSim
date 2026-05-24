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
    """Count messages and compute average length (word count as token proxy) from message_delivered events."""
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
        "avg_message_length_tokens": _mean(lengths),
    }


def _run_message_aggregates(
    events: list[dict[str, Any]], agent_ids: list[str]
) -> dict[str, float]:
    """Per-run averages: avg messages sent per agent and avg message length per agent."""
    counts: list[float] = []
    lengths: list[float] = []
    for aid in agent_ids:
        stats = _message_stats(events, aid)
        counts.append(stats["messages_sent"])
        if stats["avg_message_length_tokens"] is not None:
            lengths.append(stats["avg_message_length_tokens"])
    return {
        "avg_messages_sent_per_agent": _mean(counts) or 0.0,
        "avg_message_length_tokens_per_agent": _mean(lengths) or 0.0,
    }


def _daytrader_run_investment_aggregates(
    per_agent: dict[str, dict[str, float]], agent_ids: list[str]
) -> dict[str, float]:
    """Per-run averages of per-agent individual/group investment count and money metrics."""
    ind_counts: list[float] = []
    ind_money: list[float] = []
    grp_counts: list[float] = []
    grp_money: list[float] = []
    for aid in agent_ids:
        d = per_agent.get(aid, {})
        ind_counts.append(float(d.get("avg_individual_investment_count", 0.0)))
        ind_money.append(float(d.get("avg_individual_investment_money", 0.0)))
        grp_counts.append(float(d.get("avg_group_investment_count", 0.0)))
        grp_money.append(float(d.get("avg_group_investment_money", 0.0)))
    return {
        "avg_individual_investment_count_per_agent": _mean(ind_counts) or 0.0,
        "avg_individual_investment_money_per_agent": _mean(ind_money) or 0.0,
        "avg_group_investment_count_per_agent": _mean(grp_counts) or 0.0,
        "avg_group_investment_money_per_agent": _mean(grp_money) or 0.0,
    }


def _fulfilled_slots_from_order_fulfilled_events(events: list[dict[str, Any]]) -> dict[str, int]:
    """Sum fulfilled_count from order_fulfilled events per actor."""
    out: dict[str, int] = defaultdict(int)
    for e in events:
        if e.get("event_type") != "order_fulfilled":
            continue
        aid = e.get("actor_id")
        fc = (e.get("payload") or {}).get("fulfilled_count")
        if isinstance(aid, str) and isinstance(fc, (int, float)):
            out[aid] += int(fc)
    return dict(out)


def _last_order_progress_from_shapefactory_observations(
    trace: Trace, agent_ids: list[str],
) -> dict[str, int]:
    """Recover order_progress from the latest shapefactory observation snapshot."""
    found: dict[str, int] = {}
    for e in reversed(trace.events):
        if e.get("event_type") != "observation_built":
            continue
        payload = e.get("payload") or {}
        obs = payload.get("observation") or {}
        state = obs.get("state") or {}
        task_state = state.get("task_state") or {}
        if task_state.get("task_type") != "shapefactory":
            continue
        participants = task_state.get("participants") or {}
        if not isinstance(participants, dict):
            continue
        for aid in agent_ids:
            if aid in found:
                continue
            info = participants.get(aid)
            if isinstance(info, dict):
                op = info.get("order_progress")
                if isinstance(op, (int, float)):
                    found[aid] = int(op)
        if len(found) >= len(agent_ids):
            break
    return found


_MAPTASK_FOLLOWER_DRAW_ACTIONS = frozenset({"draw", "erase", "undo", "reset"})


def _maptask_follower_ids(trace: Trace) -> list[str]:
    """Resolve MapTask follower agent ids from manifest or run summary."""
    task_cfg = (trace.manifest.get("config") or {}).get("task") or {}
    roles = task_cfg.get("roles") or {}
    if isinstance(roles, dict):
        followers = [aid for aid, role in roles.items() if role == "follower" and isinstance(aid, str)]
        if followers:
            return followers

    config_agents = (trace.manifest.get("config") or {}).get("agents") or []
    followers = [
        a["id"]
        for a in config_agents
        if isinstance(a, dict) and a.get("role") == "follower" and isinstance(a.get("id"), str)
    ]
    if followers:
        return followers

    summary_per_agent = trace.summary.get("per_agent") or {}
    return [
        aid
        for aid, data in summary_per_agent.items()
        if isinstance(aid, str) and isinstance(data, dict) and data.get("role") == "follower"
    ]


def _follower_drawing_action_counts(
    trace: Trace, follower_ids: set[str],
) -> dict[str, int]:
    """Count validated draw/erase/undo/reset actions by MapTask followers."""
    counts = {action: 0 for action in _MAPTASK_FOLLOWER_DRAW_ACTIONS}
    for e in trace.events_of_type("action_validated"):
        aid = e.get("actor_id")
        if not isinstance(aid, str) or aid not in follower_ids:
            continue
        action_type = ((e.get("payload") or {}).get("action") or {}).get("type")
        if action_type in _MAPTASK_FOLLOWER_DRAW_ACTIONS:
            counts[action_type] += 1
    return counts


def _revision_rate_from_drawing_counts(counts: dict[str, int]) -> float | None:
    """Revision rate = 1 - draw / (draw + erase + undo + reset)."""
    draw = counts.get("draw", 0)
    total = sum(counts.get(action, 0) for action in _MAPTASK_FOLLOWER_DRAW_ACTIONS)
    if total <= 0:
        return None
    return 1.0 - (draw / total)


def _shapefactory_fulfilled_slots_by_agent(
    trace: Trace,
    agent_ids: list[str],
    summary_per_agent: dict[str, Any],
) -> dict[str, int]:
    """Per-agent own order slots fulfilled (prefers run_summary, else last obs, else events)."""
    from_events = _fulfilled_slots_from_order_fulfilled_events(trace.events)
    from_obs = _last_order_progress_from_shapefactory_observations(trace, agent_ids)
    out: dict[str, int] = {}
    for aid in agent_ids:
        val: int | None = None
        sdata = summary_per_agent.get(aid)
        if isinstance(sdata, dict):
            op = sdata.get("order_progress")
            if isinstance(op, (int, float)):
                val = int(op)
        if val is None and aid in from_obs:
            val = from_obs[aid]
        if val is None:
            val = int(from_events.get(aid, 0))
        out[aid] = val
    return out


# ------------------------------------------------------------------ #
# ShapeFactory
# ------------------------------------------------------------------ #

def _shapefactory_metrics(trace: Trace) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """
    Key metrics (mirrors collaborator's log_analysis.py):
      per-agent : final_wealth, wealth_gain, successful_trades,
                  avg_trade_price, min_trade_price, max_trade_price,
                  messages_sent, avg_message_length_words, trade_efficiency,
                  fulfilled_order_slots, order_fully_fulfilled
      per-run   : total_successful_trades, trade_accept_rate,
                  avg_trade_price, session_avg_wealth, wealth_gini,
                  agents_fully_fulfilled_own_order_count,
                  avg/min/max_fulfilled_order_slots_per_agent,
                  messages_per_successful_trade
    """
    per_agent: dict[str, dict[str, float]] = defaultdict(dict)
    per_run: dict[str, float] = {}

    # Starting money from manifest config
    task_cfg = (trace.manifest.get("config") or {}).get("task") or {}
    starting_money = float(task_cfg.get("starting_money", 200.0))
    try:
        shapes_order_target = int(max(int(task_cfg.get("shapes_order", 4)), 1))
    except (TypeError, ValueError):
        shapes_order_target = 4

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

    # Prefer run_summary.json for final balances (most authoritative)
    wealth_by_agent: dict[str, float] = {}
    summary_per_agent = trace.summary.get("per_agent") or {}
    for aid, sdata in summary_per_agent.items():
        if isinstance(sdata, dict) and isinstance(sdata.get("final_balance"), (int, float)):
            wealth_by_agent[aid] = float(sdata["final_balance"])

    # Fallback: wealth from final task_state via observation_built events
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

    fulfilled_slots_by_agent = _shapefactory_fulfilled_slots_by_agent(
        trace, trace.agent_ids, summary_per_agent,
    )
    slot_counts = [float(fulfilled_slots_by_agent.get(aid, 0)) for aid in trace.agent_ids]
    full_count = sum(
        1 for aid in trace.agent_ids
        if fulfilled_slots_by_agent.get(aid, 0) >= shapes_order_target
    )
    per_run["agents_fully_fulfilled_own_order_count"] = float(full_count)
    per_run["avg_fulfilled_order_slots_per_agent"] = _mean(slot_counts) or 0.0
    per_run["min_fulfilled_order_slots_per_agent"] = min(slot_counts) if slot_counts else 0.0
    per_run["max_fulfilled_order_slots_per_agent"] = max(slot_counts) if slot_counts else 0.0

    msg_stats = _message_stats(trace.events)
    per_run["messages_sent_total"] = msg_stats["messages_sent"]
    per_run.update(_run_message_aggregates(trace.events, trace.agent_ids))
    total_trades = float(per_run["total_successful_trades"])
    if total_trades > 0:
        per_run["messages_per_successful_trade"] = float(per_run["messages_sent_total"]) / total_trades

    # Action type distribution (validated actions only)
    action_counts_by_agent: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    action_counts_total: dict[str, int] = defaultdict(int)
    for e in trace.events_of_type("action_validated"):
        aid = e.get("actor_id")
        action_type = ((e.get("payload") or {}).get("action") or {}).get("type")
        if isinstance(aid, str) and isinstance(action_type, str):
            action_counts_by_agent[aid][action_type] += 1
            action_counts_total[action_type] += 1
    for action_type, count in action_counts_total.items():
        per_run[f"action_{action_type}_count"] = float(count)

    for aid in trace.agent_ids:
        wealth = wealth_by_agent.get(aid, starting_money)
        trades = successful_trades_by_agent.get(aid, 0)
        prices = trade_prices_by_agent.get(aid, [])
        msgs = _message_stats(trace.events, aid)
        msg_count = msgs["messages_sent"]
        fulfilled_sl = float(fulfilled_slots_by_agent.get(aid, 0))
        agent_data: dict[str, float] = {
            "final_wealth": wealth,
            "wealth_gain": wealth - starting_money,
            "successful_trades": float(trades),
            "avg_trade_price": _mean(prices) or 0.0,
            "min_trade_price": min(prices) if prices else 0.0,
            "max_trade_price": max(prices) if prices else 0.0,
            "messages_sent": msg_count,
            "avg_message_length_tokens": msgs["avg_message_length_tokens"] or 0.0,
            "trade_efficiency": float(trades) / msg_count if msg_count > 0 else 0.0,
            "messages_per_successful_trade": msg_count / float(trades) if trades > 0 else 0.0,
            "fulfilled_order_slots": fulfilled_sl,
            "order_fully_fulfilled": 1.0 if fulfilled_sl >= float(shapes_order_target) else 0.0,
        }
        for action_type, count in action_counts_by_agent.get(aid, {}).items():
            agent_data[f"action_{action_type}_count"] = float(count)
        per_agent[aid] = agent_data

    return per_run, dict(per_agent)


# ------------------------------------------------------------------ #
# DayTrader
# ------------------------------------------------------------------ #

def _daytrader_metrics(trace: Trace) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """
    Key metrics:
      per-agent : final_balance, net_return, total_individual_invested,
                  total_group_invested, group_investment_rate,
                  individual_investment_count, group_investment_count,
                  avg_individual_investment_count, avg_individual_investment_money,
                  avg_group_investment_count, avg_group_investment_money
      per-run   : rounds_completed, cooperation_rate (fraction of investment
                  actions that were group), avg_group_pool_per_round,
                  avg_net_return, plus per-agent averages of the avg_* investment metrics
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

    # Prefer run_summary.json final balances when available
    summary_per_agent = trace.summary.get("per_agent") or {}
    for aid, sdata in summary_per_agent.items():
        if isinstance(sdata, dict) and isinstance(sdata.get("final_balance"), (int, float)):
            per_agent[aid]["final_balance"] = float(sdata["final_balance"])

    # Fallback for agents with no settlement events
    for aid in trace.agent_ids:
        if per_agent[aid]["final_balance"] == 0.0:
            per_agent[aid]["final_balance"] = starting_money

    # Task state for rounds_completed
    task_state = {}
    for e in reversed(trace.events):
        p = e.get("payload") or {}
        obs = p.get("observation") or {}
        ts = (obs.get("state") or {}).get("task_state") or {}
        if ts.get("task_type") == "daytrader":
            task_state = ts
            break

    rounds_completed = float(task_state.get("rounds_completed", 0))
    rounds_divisor = rounds_completed if rounds_completed > 0 else 1.0

    # Net return, investment rates, and per-agent avg investment count/money
    total_actions = 0
    total_group_actions = 0
    net_returns: list[float] = []
    for aid in trace.agent_ids:
        d = per_agent[aid]
        d["net_return"] = d["final_balance"] - starting_money
        net_returns.append(d["net_return"])
        ind_c = int(d["individual_investment_count"])
        grp_c = int(d["group_investment_count"])
        total_ind = float(d["total_individual_invested"])
        total_grp = float(d["total_group_invested"])
        total_inv = ind_c + grp_c
        d["group_investment_rate"] = grp_c / total_inv if total_inv > 0 else 0.0
        total_actions += total_inv
        total_group_actions += grp_c
        d["individual_investment_count"] = float(ind_c)
        d["group_investment_count"] = float(grp_c)
        d["avg_individual_investment_count"] = float(ind_c) / rounds_divisor
        d["avg_group_investment_count"] = float(grp_c) / rounds_divisor
        d["avg_individual_investment_money"] = total_ind / float(ind_c) if ind_c > 0 else 0.0
        d["avg_group_investment_money"] = total_grp / float(grp_c) if grp_c > 0 else 0.0

    per_run["rounds_completed"] = rounds_completed
    per_run["cooperation_rate"] = total_group_actions / total_actions if total_actions > 0 else 0.0
    per_run["avg_group_pool_per_round"] = _mean(pool_totals) or 0.0
    per_run["avg_net_return"] = _mean(net_returns) or 0.0
    per_run["messages_sent_total"] = _message_stats(trace.events)["messages_sent"]
    per_run.update(_run_message_aggregates(trace.events, trace.agent_ids))
    per_run.update(_daytrader_run_investment_aggregates(dict(per_agent), trace.agent_ids))

    # Winner: agent with highest final balance
    if trace.agent_ids:
        winner_id = max(trace.agent_ids, key=lambda aid: per_agent[aid]["final_balance"])
        per_run["winner"] = winner_id

    # Per-agent message stats
    for aid in trace.agent_ids:
        msgs = _message_stats(trace.events, aid)
        per_agent[aid]["messages_sent"] = msgs["messages_sent"]
        per_agent[aid]["avg_message_length_tokens"] = msgs["avg_message_length_tokens"] or 0.0

    return per_run, {k: dict(v) for k, v in per_agent.items()}


# ------------------------------------------------------------------ #
# Hidden Profile
# ------------------------------------------------------------------ #

def _hidden_profile_metrics(trace: Trace) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """
    Key metrics:
      per-agent : initial_vote, final_vote, vote_changed, initial_vote_correct,
                  final_vote_correct, messages_sent, avg_message_length_tokens
      per-run   : consensus_reached, vote_change_rate, initial_vote_accuracy,
                  final_vote_accuracy, messages_sent_total
    """
    per_agent: dict[str, dict[str, Any]] = {aid: {} for aid in trace.agent_ids}
    per_run: dict[str, float] = {}

    # Votes come from run_summary.json task_summary (not from events)
    task_summary = trace.summary.get("task_summary") or {}
    initial_votes_map: dict[str, str] = task_summary.get("initial_votes") or {}
    final_votes_map: dict[str, str] = task_summary.get("final_votes") or {}

    for aid in per_agent:
        if aid in initial_votes_map:
            per_agent[aid]["initial_vote"] = initial_votes_map[aid]
        if aid in final_votes_map:
            per_agent[aid]["final_vote"] = final_votes_map[aid]

    # Correct answer from task config phase_rules (optional)
    task_cfg = (trace.manifest.get("config") or {}).get("task") or {}
    phase_rules = task_cfg.get("phase_rules") or {}
    correct_answer: str | None = phase_rules.get("correct_answer")
    if isinstance(correct_answer, str):
        correct_answer = correct_answer.strip()

    vote_changed_count = 0
    initial_correct_count = 0
    final_correct_count = 0
    final_votes: list[str] = []
    agent_count = len(per_agent)

    for aid, d in per_agent.items():
        iv = d.get("initial_vote")
        fv = d.get("final_vote")
        changed = (iv is not None and fv is not None and iv != fv)
        d["vote_changed"] = 1.0 if changed else 0.0
        if changed:
            vote_changed_count += 1
        if isinstance(fv, str):
            final_votes.append(fv)
        if correct_answer is not None:
            iv_correct = isinstance(iv, str) and iv.lower() == correct_answer.lower()
            fv_correct = isinstance(fv, str) and fv.lower() == correct_answer.lower()
            d["initial_vote_correct"] = 1.0 if iv_correct else 0.0
            d["final_vote_correct"] = 1.0 if fv_correct else 0.0
            if iv_correct:
                initial_correct_count += 1
            if fv_correct:
                final_correct_count += 1
        msgs = _message_stats(trace.events, aid)
        d["messages_sent"] = msgs["messages_sent"]
        d["avg_message_length_tokens"] = msgs["avg_message_length_tokens"] or 0.0

    # Consensus: all final votes identical
    per_run["consensus_reached"] = 1.0 if len(set(final_votes)) == 1 and final_votes else 0.0
    per_run["vote_change_rate"] = vote_changed_count / agent_count if agent_count > 0 else 0.0

    if correct_answer is not None and agent_count > 0:
        per_run["initial_vote_accuracy"] = initial_correct_count / agent_count
        per_run["final_vote_accuracy"] = final_correct_count / agent_count

    # Vote distribution
    for candidate in set(final_votes):
        per_run[f"final_votes_for_{candidate}"] = float(final_votes.count(candidate))

    per_run["messages_sent_total"] = _message_stats(trace.events)["messages_sent"]
    per_run.update(_run_message_aggregates(trace.events, trace.agent_ids))

    # Candidate mention fractions across all messages
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

def _maptask_drawing_accuracy_steps(trace: Trace) -> list[dict[str, Any]]:
    """Collect score_board snapshots after each map_progress_updated event."""

    steps: list[dict[str, Any]] = []
    for idx, event in enumerate(trace.events):
        if event.get("event_type") != "map_progress_updated":
            continue
        payload = event.get("payload") or {}
        snap = payload.get("drawing_accuracy")
        if not isinstance(snap, dict):
            continue
        step_entry: dict[str, Any] = {
            "step_index": len(steps),
            "sim_step": event.get("step"),
            "timestamp": event.get("timestamp"),
            "actor_id": event.get("actor_id"),
            **snap,
        }
        steps.append(step_entry)
    return steps


def _apply_maptask_route_scalars(per_run: dict[str, Any], route_score: float, route_score_max: float,
                                 route_similarity: float | None) -> None:
    per_run["route_score"] = route_score
    per_run["route_score_max"] = route_score_max
    if route_score_max > 0:
        per_run["route_completion_rate"] = route_score / route_score_max
        per_run["follower_accuracy"] = route_score / route_score_max
    if route_similarity is not None:
        per_run["route_similarity"] = route_similarity


def _maptask_metrics(trace: Trace) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    """
    Key metrics (score_board.txt-based):
      per-agent : messages_sent, avg_message_length_tokens, map_progress_updates
      per-run   : route_score, route_score_max, route_similarity, drawing_score_steps,
                  drawing_score_final, map_progress_updates_total, communication_efficiency,
                  follower_draw/erase/undo/reset counts, revision_rate
    """
    per_agent: dict[str, dict[str, float]] = {}
    per_run: dict[str, Any] = {}

    task_summary = trace.summary.get("task_summary") or {}
    route_score = task_summary.get("route_score")
    route_score_max = task_summary.get("route_score_max")
    route_similarity = task_summary.get("route_similarity")

    # Fallback: final drawing_accuracy from last map_progress_updated
    steps = _maptask_drawing_accuracy_steps(trace)
    if steps:
        per_run["drawing_score_steps"] = steps
        per_run["drawing_score_final"] = dict(steps[-1])
        per_run["drawing_score_step_count"] = float(len(steps))
        last_ratio = steps[-1].get("ratio_vs_ground_truth_route")
        if isinstance(last_ratio, (int, float)):
            per_run["drawing_score_final_ratio"] = float(last_ratio)
        ratios = [
            float(s["ratio_vs_ground_truth_route"])
            for s in steps
            if isinstance(s.get("ratio_vs_ground_truth_route"), (int, float))
        ]
        if ratios:
            per_run["drawing_score_peak_ratio"] = max(ratios)

    if not isinstance(route_score, (int, float)) and steps:
        route_score = steps[-1].get("score_board_sum_drawn_cells")
    if not isinstance(route_score_max, (int, float)) and steps:
        route_score_max = steps[-1].get("max_route_score_board_sum")
    if not isinstance(route_similarity, (int, float)) and steps:
        route_similarity = steps[-1].get("ratio_vs_ground_truth_route")

    if isinstance(route_score, (int, float)) and isinstance(route_score_max, (int, float)):
        rs = float(route_score)
        rsm = float(route_score_max)
        sim = float(route_similarity) if isinstance(route_similarity, (int, float)) else None
        _apply_maptask_route_scalars(per_run, rs, rsm, sim)

    updates_total = len(trace.events_of_type("map_progress_updated"))
    per_run["map_progress_updates_total"] = float(updates_total)

    msg_stats = _message_stats(trace.events)
    per_run["messages_sent_total"] = msg_stats["messages_sent"]
    total_msgs = msg_stats["messages_sent"]
    if isinstance(per_run.get("route_score"), (int, float)) and total_msgs > 0:
        per_run["communication_efficiency"] = float(per_run["route_score"]) / total_msgs
    per_run.update(_run_message_aggregates(trace.events, trace.agent_ids))

    follower_ids = set(_maptask_follower_ids(trace))
    follower_drawing_counts = _follower_drawing_action_counts(trace, follower_ids)
    for action_type, count in follower_drawing_counts.items():
        per_run[f"follower_action_{action_type}_count"] = float(count)
    revision_rate = _revision_rate_from_drawing_counts(follower_drawing_counts)
    if revision_rate is not None:
        per_run["revision_rate"] = revision_rate

    for aid in trace.agent_ids:
        msgs = _message_stats(trace.events, aid)
        upd = float(sum(
            1 for e in trace.events_by_actor(aid)
            if e.get("event_type") == "map_progress_updated"
        ))
        agent_data: dict[str, float] = {
            "messages_sent": msgs["messages_sent"],
            "avg_message_length_tokens": msgs["avg_message_length_tokens"] or 0.0,
            "map_progress_updates": upd,
        }
        if aid in follower_ids:
            follower_counts = _follower_drawing_action_counts(trace, {aid})
            for action_type, count in follower_counts.items():
                agent_data[f"action_{action_type}_count"] = float(count)
            agent_revision = _revision_rate_from_drawing_counts(follower_counts)
            if agent_revision is not None:
                agent_data["revision_rate"] = agent_revision
        per_agent[aid] = agent_data

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


def _resolve_task_type(trace: Trace) -> str | None:
    task_type = trace.task_type
    if task_type:
        return task_type
    summary_type = trace.summary.get("task_type")
    if isinstance(summary_type, str) and summary_type:
        return summary_type
    return None


def compute_task_metrics(
    trace: Trace,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    """Dispatch to the correct task metrics function and return (per_run, per_agent)."""
    task_type = _resolve_task_type(trace)
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
