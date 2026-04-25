"""Experiment controller skeleton with deterministic loop."""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import heapq
import hashlib
import json
import random
import time
from typing import Any, Callable

from src.agents.interface import ActionProposal, Observation
from src.config.schema import validate_config_schema
from src.core.actions import ActionValidationError, validate_action
from src.data.events import EventValidationError, validate_event
from src.data.logging import (
    RunPaths,
    append_jsonl,
    build_run_manifest,
    log_probe_records,
    write_metrics,
    write_run_manifest,
)
from src.metrics.metrics import compute_metrics
from src.probe.probe import ProbeResponse
from src.probe.registry import ProbeRegistry, ProbeValidationError
from src.tasks import NOOP_TASK
from src.tasks.registry import TaskRegistry
from src.utils.env import load_text_file


@dataclass
class ExperimentState:
    """State container for a single experiment run."""

    agents: dict[str, Any]
    task_state: dict[str, Any]
    resources: dict[str, Any]
    turn_state: dict[str, Any]
    buffers: dict[str, Any]
    step_index: int = 0
    event_log: list[dict[str, Any]] = field(default_factory=list)
    probe_log: list[dict[str, Any]] = field(default_factory=list)
    pending_actions: list[dict[str, Any]] = field(default_factory=list)
    visibility_map: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    agent_memory: dict[str, dict[str, Any]] = field(default_factory=dict)
    observation_cache: dict[str, Observation] = field(default_factory=dict)
    agent_memory_limits: dict[str, int] = field(default_factory=dict)
    visibility_defaults: dict[str, list[str]] = field(default_factory=dict)
    visibility_defaults_by_agent: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    visibility_overrides_by_agent: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    probe_event_index: int = 0
    probe_index: int = 0
    probe_action_counts_by_agent: dict[str, int] = field(default_factory=dict)
    agent_status: dict[str, str] = field(default_factory=dict)
    pending_context_updates: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_context_hash: dict[str, str] = field(default_factory=dict)
    context_memory: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    sim_time_ms: int = 0


def build_state_snapshot(state: ExperimentState) -> dict[str, Any]:
    """Build a deterministic snapshot of replay-relevant state.

    Args:
        state: Current experiment state container.

    Returns:
        JSON-serializable snapshot used for deterministic hashing.

    Research rationale:
        Ensures deterministic replay checks can compare state evolution
        without relying on non-serializable or nondeterministic objects.
    """

    return {
        "step_index": state.step_index,
        "task_state": state.task_state,
        "resources": state.resources,
        "turn_state": state.turn_state,
        "buffers": state.buffers,
    }


def compute_state_hash(state: ExperimentState) -> str:
    """Compute a deterministic hash of replay-relevant state."""

    snapshot = build_state_snapshot(state)
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ExperimentController:
    """Deterministic controller loop (skeleton)."""

    def __init__(
        self,
        initial_state: ExperimentState,
        config: dict[str, Any] | None = None,
        run_paths: RunPaths | None = None,
        run_id: str | None = None,
        task_hook: Callable[[ExperimentState], None] | None = None,
        task_factory: Callable[[dict[str, Any]], Callable[[ExperimentState], None]] | None = None,
        task_registry: Any | None = None,
        probe_registry: ProbeRegistry | None = None,
    ) -> None:
        if config is not None:
            validate_config_schema(config)
        self.state = initial_state
        self.run_paths = run_paths
        self.config = config
        self.run_id = run_id
        self.probe_registry = probe_registry
        self.task_definition: Any | None = None
        self.probe_template_ids = []
        if isinstance(config, dict):
            probe_cfg = config.get("probe", {})
            if isinstance(probe_cfg, dict):
                templates = probe_cfg.get("templates", [])
                if isinstance(templates, list):
                    self.probe_template_ids = [item for item in templates if isinstance(item, str)]
        if self.probe_registry is not None and self.probe_template_ids:
            missing: list[str] = []
            for template_id in self.probe_template_ids:
                try:
                    self.probe_registry.get(template_id)
                except KeyError:
                    missing.append(template_id)
            if missing:
                raise ValueError(f"Probe templates not registered: {', '.join(missing)}")
        if self.run_paths is not None and not self.run_id:
            raise ValueError("run_id is required when run_paths is provided.")
        if task_hook is not None:
            self.task_hook = task_hook
        elif task_factory is not None and config is not None:
            self.task_hook = task_factory(config)
        elif task_registry is not None and config is not None:
            task_type = config.get("task", {}).get("type")
            if task_type is None:
                raise ValueError("config.task.type is required for task_registry.")
            task_def = task_registry.get(task_type)
            self.task_definition = task_def
            self.state.task_state = task_def.init_state(config)
            self.task_hook = lambda state: task_def.step(state.task_state)
        elif task_registry is not None and config is None:
            raise ValueError("config is required when providing a task_registry.")
        elif config is not None:
            registry = TaskRegistry()
            registry.register(NOOP_TASK)
            task_type = config.get("task", {}).get("type", "noop")
            task_def = registry.get(task_type)
            self.task_definition = task_def
            self.state.task_state = task_def.init_state(config)
            self.task_hook = lambda state: task_def.step(state.task_state)
        else:
            self.task_hook = None
        self._last_flushed_index = 0
        self._event_counter = 0
        self._message_counter = 0
        self._scheduled_counter = 0
        self._scheduled_events: list[tuple[int, int, str, dict[str, Any]]] = []
        self._pending_realtime_message_queue: list[dict[str, Any]] = []
        self._wall_start_monotonic: float | None = None
        self._wall_duration_sec: float | None = None
        self._turn_rng = random.Random(self._turn_order_seed())
        self.event_listener: Callable[[dict[str, Any]], None] | None = None
        self.runtime_listener: Callable[[str, dict[str, Any]], None] | None = None

    def run(self, max_steps: int | None = None) -> ExperimentState:
        """Run the controller loop up to max_steps or until task complete."""

        if self._step_mode() == "time":
            self._reset_agents()
            self._run_time_mode(max_steps=max_steps)
            self._export_metrics()
            self._flush_logs()
            self._write_manifest()
            return self.state

        resolved_max_steps = self._resolve_max_steps(max_steps)
        self._reset_agents()
        termination_condition = self._termination_condition()
        if self._step_mode() == "event":
            self._schedule_context_updates()
            noop_streak = 0
            noop_limit = self._noop_consecutive_cycles()
            while self.state.step_index < resolved_max_steps:
                if termination_condition == "task_complete" and self.state.task_state.get("complete") is True:
                    break
                if (
                    not self._uses_maptask_event_pipeline()
                    and not self.state.pending_actions
                    and not self.state.pending_context_updates
                ):
                    break
                self.step()
                all_noop = self.state.turn_state.get("all_actions_do_nothing") is True
                if all_noop:
                    noop_streak += 1
                else:
                    noop_streak = 0
                if noop_limit > 0 and noop_streak >= noop_limit:
                    break
        else:
            while self.state.step_index < resolved_max_steps:
                if termination_condition == "task_complete" and self.state.task_state.get("complete") is True:
                    break
                self.step()
        self._export_metrics()
        self._flush_logs()
        self._write_manifest()
        return self.state

    def _termination_condition(self) -> str:
        if not isinstance(self.config, dict):
            return "task_complete"
        protocol = self.config.get("protocol", {})
        if not isinstance(protocol, dict):
            return "task_complete"
        termination = protocol.get("termination", {})
        if not isinstance(termination, dict):
            return "task_complete"
        condition = termination.get("condition")
        if isinstance(condition, str) and condition:
            return condition
        return "task_complete"

    def _step_mode(self) -> str:
        if not isinstance(self.config, dict):
            return "event"
        protocol = self.config.get("protocol", {})
        if not isinstance(protocol, dict):
            return "event"
        mode = protocol.get("step_mode")
        if isinstance(mode, str) and mode:
            return mode
        return "event"

    def _run_time_mode(self, max_steps: int | None = None) -> None:
        duration_sec = self._duration_seconds()
        allow_unbounded_hidden_profile = (
            self._task_type() == "hidden_profile" and self._termination_condition() == "task_complete"
        )
        if duration_sec <= 0 and not allow_unbounded_hidden_profile:
            raise ValueError("experiment.duration_sec or experiment.duration_ms must be a positive number for time mode.")
        interval_sec = self._agent_trigger_interval_sec()
        if interval_sec <= 0:
            raise ValueError("protocol.agent_trigger_interval_sec must be a positive number for time mode.")

        self._wall_start_monotonic = time.monotonic()
        self._wall_duration_sec = duration_sec if duration_sec > 0 else None
        self._pending_realtime_message_queue = []
        self.state.sim_time_ms = 0
        self.state.observation_cache.clear()
        for agent_id in self.state.agents.keys():
            self._set_agent_status(agent_id, "idle")

        end_time = (
            self._wall_start_monotonic + duration_sec if duration_sec > 0 else float("inf")
        )
        next_cycle_at = self._wall_start_monotonic
        noop_streak_by_agent: dict[str, int] = {}
        noop_limit = self._noop_consecutive_cycles()
        delayed_actions: list[tuple[float, str, dict[str, Any]]] = []
        agent_ids = [agent_id for agent_id in self.state.agents.keys() if isinstance(agent_id, str) and agent_id]
        previous_hidden_profile_phase: str | None = None
        previous_daytrader_phase: str | None = None
        daytrader_decision_phase_queried = False

        while True:
            now = time.monotonic()
            elapsed_sec = max(0.0, now - self._wall_start_monotonic)
            self.state.sim_time_ms = int(elapsed_sec * 1000)
            if self._task_type() == "hidden_profile":
                phase = self._hidden_profile_phase()
                if phase != previous_hidden_profile_phase:
                    if phase in {"initial", "final"}:
                        next_cycle_at = min(next_cycle_at, now)
                previous_hidden_profile_phase = phase
            if self._task_type() == "daytrader":
                self._sync_daytrader_round_phase()
                phase = self._daytrader_phase()
                if phase == "group_chat":
                    turns = self.state.task_state.get("group_chat_turns", 0)
                    if not isinstance(turns, int):
                        turns = 0
                    if turns == 0 and not self._pending_realtime_message_queue:
                        bootstrap_targets = self._daytrader_group_chat_bootstrap_targets(agent_ids)
                        for target in bootstrap_targets:
                            self._enqueue_realtime_trigger(target, phase_lock="group_chat")
                if phase != previous_daytrader_phase:
                    next_cycle_at = min(next_cycle_at, now)
                    if phase == "decision":
                        daytrader_decision_phase_queried = False
                previous_daytrader_phase = phase

            self._process_due_realtime_actions(delayed_actions, now)
            self._flush_logs()

            if now >= end_time:
                break
            if isinstance(max_steps, int) and max_steps > 0 and self.state.step_index >= max_steps:
                break
            if self._termination_condition() == "task_complete" and self.state.task_state.get("complete") is True:
                break
            if self._pending_realtime_message_queue:
                ready_queue_entries: list[dict[str, Any]] = []
                seen: set[str] = set()
                remaining_queue: list[dict[str, Any]] = []
                for entry in self._pending_realtime_message_queue:
                    if not isinstance(entry, dict):
                        continue
                    recipient = entry.get("agent_id")
                    if (
                        isinstance(recipient, str)
                        and recipient
                        and recipient not in seen
                        and self._is_agent_idle(recipient)
                    ):
                        ready_queue_entries.append(entry)
                        seen.add(recipient)
                        continue
                    remaining_queue.append(entry)
                self._pending_realtime_message_queue = remaining_queue
                if ready_queue_entries:
                    trigger_entries = ready_queue_entries
                    if self._task_type() == "daytrader" and self._daytrader_phase() == "group_chat":
                        # Keep free chat conversational: trigger one agent at a time in queue order.
                        trigger_entries = ready_queue_entries[:1]
                        if len(ready_queue_entries) > 1:
                            self._pending_realtime_message_queue = ready_queue_entries[1:] + self._pending_realtime_message_queue
                    trigger_targets = [entry.get("agent_id") for entry in trigger_entries if isinstance(entry.get("agent_id"), str)]
                    phase_lock_by_agent: dict[str, str] = {}
                    for entry in trigger_entries:
                        recipient = entry.get("agent_id")
                        phase_lock = entry.get("daytrader_phase_lock")
                        if isinstance(recipient, str) and phase_lock in {"decision", "group_chat"}:
                            phase_lock_by_agent[recipient] = phase_lock
                    action_types = self._trigger_idle_agents_parallel(
                        trigger_targets,
                        delayed_actions,
                        daytrader_phase_lock_by_agent=phase_lock_by_agent or None,
                    )
                    for recipient, action_type in action_types.items():
                        if action_type == "do_nothing":
                            noop_streak_by_agent[recipient] = noop_streak_by_agent.get(recipient, 0) + 1
                        elif isinstance(action_type, str):
                            noop_streak_by_agent[recipient] = 0
                    self._flush_logs()
                    continue

            if now >= next_cycle_at:
                should_trigger_cycle = True
                if self._task_type() == "daytrader":
                    if self._daytrader_phase() == "decision":
                        should_trigger_cycle = not daytrader_decision_phase_queried
                    else:
                        # In group chat, only bootstrap once (A then B) and then rely on message-triggered thinking.
                        should_trigger_cycle = False
                if not should_trigger_cycle:
                    next_cycle_at += interval_sec
                    continue
                idle_agents = [agent_id for agent_id in sorted(agent_ids) if self._is_agent_idle(agent_id)]
                cycle_active_agents = len(idle_agents)
                action_types = self._trigger_idle_agents_parallel(idle_agents, delayed_actions)
                if self._task_type() == "daytrader" and self._daytrader_phase() == "decision":
                    daytrader_decision_phase_queried = True
                for agent_id, action_type in action_types.items():
                    if action_type == "do_nothing":
                        noop_streak_by_agent[agent_id] = noop_streak_by_agent.get(agent_id, 0) + 1
                    elif isinstance(action_type, str):
                        noop_streak_by_agent[agent_id] = 0
                if cycle_active_agents > 0:
                    for agent_id in agent_ids:
                        if agent_id not in noop_streak_by_agent:
                            noop_streak_by_agent[agent_id] = 0
                next_cycle_at += interval_sec
                self._flush_logs()
                if (
                    noop_limit > 0
                    and agent_ids
                    and all(noop_streak_by_agent.get(agent_id, 0) >= noop_limit for agent_id in agent_ids)
                ):
                    break
                continue

            earliest_due = min((due for due, _, _ in delayed_actions), default=end_time)
            wake_at = min(next_cycle_at, earliest_due, end_time)
            sleep_sec = max(0.0, wake_at - time.monotonic())
            if sleep_sec > 0:
                time.sleep(sleep_sec)

    def _trigger_idle_agents_parallel(
        self,
        agent_ids: list[str],
        delayed_actions: list[tuple[float, str, dict[str, Any]]],
        daytrader_phase_lock_by_agent: dict[str, str] | None = None,
    ) -> dict[str, str | None]:
        if not agent_ids:
            return {}
        payloads: list[dict[str, Any]] = []
        for agent_id in agent_ids:
            if not isinstance(agent_id, str) or not agent_id:
                continue
            if not self._is_agent_idle(agent_id):
                continue
            self._set_agent_status(agent_id, "busy")
            payload: dict[str, Any] = {"agent_id": agent_id}
            if isinstance(daytrader_phase_lock_by_agent, dict):
                phase_lock = daytrader_phase_lock_by_agent.get(agent_id)
                if phase_lock in {"decision", "group_chat"}:
                    payload["daytrader_phase_lock"] = phase_lock
            payloads.append(payload)
        if not payloads:
            return {}
        return self._handle_agent_decide_batch_realtime(payloads, delayed_actions)

    def _handle_agent_decide_batch_realtime(
        self,
        payloads: list[dict[str, Any]],
        delayed_actions: list[tuple[float, str, dict[str, Any]]],
    ) -> dict[str, str | None]:
        requested_ids: list[str] = []
        daytrader_phase_lock_by_agent: dict[str, str] = {}
        for payload in payloads:
            agent_id = payload.get("agent_id")
            if not isinstance(agent_id, str) or agent_id not in self.state.agents:
                continue
            if self.state.agent_status.get(agent_id) != "busy":
                continue
            requested_ids.append(agent_id)
            phase_lock = payload.get("daytrader_phase_lock")
            if phase_lock in {"decision", "group_chat"}:
                daytrader_phase_lock_by_agent[agent_id] = phase_lock
        if not requested_ids:
            return {}

        observation_map: dict[str, Observation] = {}
        self._sync_daytrader_round_phase()
        for agent_id in requested_ids:
            agent = self.state.agents.get(agent_id)
            if agent is None:
                continue
            self._apply_visibility_map_from_config(agent_id)
            self._load_agent_memory(agent_id, agent)
            observation_map[agent_id] = self._get_cached_observation(agent_id)

        proposals: dict[str, Any] = {}
        max_workers = min(self._max_concurrent_requests(), len(observation_map))
        if max_workers <= 1:
            for agent_id, observation in observation_map.items():
                proposals[agent_id] = self._invoke_agent_context_update(agent_id, observation)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._invoke_agent_context_update, agent_id, observation): agent_id
                    for agent_id, observation in observation_map.items()
                }
                for future in as_completed(futures):
                    agent_id = futures[future]
                    try:
                        proposals[agent_id] = future.result()
                    except Exception:
                        proposals[agent_id] = None

        action_types: dict[str, str | None] = {}
        validated_actions: dict[str, dict[str, Any]] = {}
        for agent_id in sorted(requested_ids):
            observation = observation_map.get(agent_id)
            if observation is None:
                self._set_agent_status(agent_id, "idle")
                action_types[agent_id] = None
                continue
            proposal = proposals.get(agent_id)
            action = self._finalize_time_proposal(
                agent_id,
                observation,
                proposal,
                trigger_probe=False,
                daytrader_phase_lock=daytrader_phase_lock_by_agent.get(agent_id),
            )
            if action is None:
                self._set_agent_status(agent_id, "idle")
                action_types[agent_id] = None
                continue
            action_type = action.get("type")
            action_types[agent_id] = action_type if isinstance(action_type, str) else None
            validated_actions[agent_id] = action

        self._maybe_probe_after_valid_actions_batch(validated_actions)

        for agent_id in sorted(requested_ids):
            action = validated_actions.get(agent_id)
            if action is None:
                continue
            duration_ms = self._action_duration_ms(action)
            if duration_ms <= 0:
                self._apply_realtime_action_completion(agent_id, action)
                continue
            # For delayed completion (e.g. produce_shape), keep agent idle so it can
            # continue receiving requests while task effects are finalized later.
            self._set_agent_status(agent_id, "idle")
            delayed_actions.append((time.monotonic() + (duration_ms / 1000.0), agent_id, action))
        self._ensure_daytrader_freechat_bootstrap(validated_actions)
        return action_types

    def _duration_seconds(self) -> float:
        duration_ms = self._duration_ms()
        if duration_ms <= 0:
            return 0.0
        return float(duration_ms) / 1000.0

    def _agent_trigger_interval_sec(self) -> float:
        if not isinstance(self.config, dict):
            return 10.0
        protocol = self.config.get("protocol", {})
        if not isinstance(protocol, dict):
            return 10.0
        value = protocol.get("agent_trigger_interval_sec")
        if isinstance(value, (int, float)) and float(value) > 0:
            return float(value)
        return 10.0

    def _noop_consecutive_cycles(self) -> int:
        if not isinstance(self.config, dict):
            return 3
        protocol = self.config.get("protocol", {})
        if not isinstance(protocol, dict):
            return 3
        termination = protocol.get("termination", {})
        if not isinstance(termination, dict):
            return 3
        value = termination.get("noop_consecutive_cycles")
        if isinstance(value, int) and value > 0:
            return value
        return 3

    def _process_due_realtime_actions(
        self,
        delayed_actions: list[tuple[float, str, dict[str, Any]]],
        now: float,
    ) -> None:
        if not delayed_actions:
            return
        due = [entry for entry in delayed_actions if entry[0] <= now]
        if not due:
            return
        delayed_actions[:] = [entry for entry in delayed_actions if entry[0] > now]
        due.sort(key=lambda item: item[0])
        for _, agent_id, action in due:
            self._apply_realtime_action_completion(agent_id, action)

    def _apply_realtime_action_completion(self, agent_id: str, action: dict[str, Any]) -> None:
        self._notify_runtime(
            "step_start",
            {
                "step_index": self.state.step_index + 1,
                "sim_time_ms": self.state.sim_time_ms,
            },
        )
        self._apply_action(action, agent_id)
        self._advance_daytrader_phase_from_action(agent_id, action)
        self.state.step_index += 1
        self._sync_daytrader_round_phase()
        self.state.observation_cache.clear()
        self.state.turn_state["message_counts"] = {}
        if self.task_hook is not None:
            self.task_hook(self.state)
        state_hash = compute_state_hash(self.state)
        self._emit_event(
            event_type="state_updated",
            actor_id="system",
            visibility="system",
            payload={
                "step_index": self.state.step_index,
                "sim_time_ms": self.state.sim_time_ms,
                "state_delta": {"step_index": self.state.step_index},
                "resulting_state_hash": state_hash,
            },
            event_id=f"step_{self.state.step_index}",
        )
        self._maybe_log_probe(had_action=True, actor_id=agent_id)
        self._set_agent_status(agent_id, "idle")

    def _max_concurrent_requests(self) -> int:
        if not isinstance(self.config, dict):
            return max(1, len(self.state.agents))
        protocol = self.config.get("protocol", {})
        if not isinstance(protocol, dict):
            return max(1, len(self.state.agents))
        value = protocol.get("max_concurrent_requests")
        if isinstance(value, int) and value > 0:
            return value
        # Backward compatibility for older configs.
        value = protocol.get("max_concurrent_thinking")
        if isinstance(value, int) and value > 0:
            return value
        return max(1, len(self.state.agents))

    def _enqueue_event(self, when_ms: int, kind: str, payload: dict[str, Any]) -> None:
        self._scheduled_counter += 1
        heapq.heappush(self._scheduled_events, (when_ms, self._scheduled_counter, kind, payload))

    def _duration_ms(self) -> int:
        if not isinstance(self.config, dict):
            return 0
        experiment = self.config.get("experiment", {})
        if not isinstance(experiment, dict):
            return 0
        duration_ms = experiment.get("duration_ms")
        if isinstance(duration_ms, (int, float)) and duration_ms > 0:
            return int(duration_ms)
        duration_sec = experiment.get("duration_sec")
        if isinstance(duration_sec, (int, float)) and duration_sec > 0:
            return int(duration_sec * 1000)
        return 0

    def _proactive_wakeup_interval_ms(self) -> int:
        if not isinstance(self.config, dict):
            return 15000
        protocol = self.config.get("protocol", {})
        if not isinstance(protocol, dict):
            return 15000
        interval = protocol.get("proactive_wakeup_interval_ms")
        if isinstance(interval, int) and interval > 0:
            return interval
        return 15000

    def _action_duration_ms(self, action: dict[str, Any]) -> int:
        action_type = action.get("type")
        if not isinstance(action_type, str):
            return 0
        # In time mode, actions execute immediately except production, which can be delayed.
        if self._step_mode() == "time":
            if action_type == "produce_shape":
                return int(self._produce_shape_delay_sec() * 1000)
            return 0
        if isinstance(self.config, dict):
            protocol = self.config.get("protocol", {})
            if isinstance(protocol, dict):
                durations = protocol.get("action_durations_ms")
                if isinstance(durations, dict):
                    value = durations.get(action_type)
                    if isinstance(value, int) and value >= 0:
                        return value
        if action_type == "produce_shape" and isinstance(self.config, dict):
            task_cfg = self.config.get("task", {})
            if isinstance(task_cfg, dict):
                production_time = task_cfg.get("production_time")
                if isinstance(production_time, (int, float)) and production_time >= 0:
                    return int(production_time * 1000)
        defaults = {
            "communicate": 120,
            "produce_shape": 2000,
            "propose_trade_offer": 200,
            "trade_response": 150,
            "fulfill_order": 600,
            "do_nothing": 0,
        }
        return defaults.get(action_type, 0)

    def _produce_shape_delay_sec(self) -> float:
        if not isinstance(self.config, dict):
            return 10.0
        protocol = self.config.get("protocol", {})
        if not isinstance(protocol, dict):
            return 10.0
        value = protocol.get("produce_shape_delay_sec")
        if isinstance(value, (int, float)) and float(value) > 0:
            return float(value)
        return 10.0

    def _handle_agent_wakeup(self, payload: dict[str, Any]) -> None:
        agent_id = payload.get("agent_id")
        if not isinstance(agent_id, str) or agent_id not in self.state.agents:
            return
        if not self._is_agent_idle(agent_id):
            return
        # No synthetic pre-thinking delay: wakeup triggers request immediately.
        self._set_agent_status(agent_id, "busy")
        self._enqueue_event(self.state.sim_time_ms, "agent_decide", {"agent_id": agent_id})

    def _handle_agent_decide(self, payload: dict[str, Any]) -> None:
        agent_id = payload.get("agent_id")
        if not isinstance(agent_id, str) or agent_id not in self.state.agents:
            return
        if self.state.agent_status.get(agent_id) != "busy":
            return
        action = self._context_update_agent_time(agent_id)
        if action is None:
            self._set_agent_status(agent_id, "idle")
            return
        duration_ms = self._action_duration_ms(action)
        if duration_ms <= 0:
            self._handle_action_complete({"agent_id": agent_id, "action": action})
            return
        self._set_agent_status(agent_id, "executing")
        self._enqueue_event(
            self.state.sim_time_ms + duration_ms,
            "action_complete",
            {"agent_id": agent_id, "action": action},
        )

    def _handle_agent_decide_batch(self, payloads: list[dict[str, Any]]) -> dict[str, str | None]:
        requested_ids: list[str] = []
        for payload in payloads:
            agent_id = payload.get("agent_id")
            if not isinstance(agent_id, str) or agent_id not in self.state.agents:
                continue
            if self.state.agent_status.get(agent_id) != "busy":
                continue
            requested_ids.append(agent_id)
        if not requested_ids:
            return {}

        # Build observations in controller thread for consistent visibility filtering and cached snapshots.
        observation_map: dict[str, Observation] = {}
        for agent_id in requested_ids:
            agent = self.state.agents.get(agent_id)
            if agent is None:
                continue
            self._apply_visibility_map_from_config(agent_id)
            self._load_agent_memory(agent_id, agent)
            observation_map[agent_id] = self._get_cached_observation(agent_id)

        proposals: dict[str, Any] = {}
        max_workers = min(self._max_concurrent_requests(), len(observation_map))
        if max_workers <= 1:
            for agent_id, observation in observation_map.items():
                proposals[agent_id] = self._invoke_agent_context_update(agent_id, observation)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._invoke_agent_context_update, agent_id, observation): agent_id
                    for agent_id, observation in observation_map.items()
                }
                for future in as_completed(futures):
                    agent_id = futures[future]
                    try:
                        proposals[agent_id] = future.result()
                    except Exception:
                        proposals[agent_id] = None

        # Commit in deterministic order.
        action_types: dict[str, str | None] = {}
        validated_actions: dict[str, dict[str, Any]] = {}
        for agent_id in sorted(requested_ids):
            observation = observation_map.get(agent_id)
            if observation is None:
                self._set_agent_status(agent_id, "idle")
                action_types[agent_id] = None
                continue
            proposal = proposals.get(agent_id)
            action = self._finalize_time_proposal(agent_id, observation, proposal, trigger_probe=False)
            if action is None:
                self._set_agent_status(agent_id, "idle")
                action_types[agent_id] = None
                continue
            action_type = action.get("type")
            action_types[agent_id] = action_type if isinstance(action_type, str) else None
            validated_actions[agent_id] = action

        self._maybe_probe_after_valid_actions_batch(validated_actions)

        for agent_id in sorted(requested_ids):
            action = validated_actions.get(agent_id)
            if action is None:
                continue
            duration_ms = self._action_duration_ms(action)
            if duration_ms <= 0:
                self._handle_action_complete({"agent_id": agent_id, "action": action})
                continue
            self._set_agent_status(agent_id, "executing")
            self._enqueue_event(
                self.state.sim_time_ms + duration_ms,
                "action_complete",
                {"agent_id": agent_id, "action": action},
            )
        return action_types

    def _invoke_agent_context_update(self, agent_id: str, observation: Observation) -> Any:
        agent = self.state.agents.get(agent_id)
        if agent is None or not hasattr(agent, "context_update"):
            return None
        self._notify_runtime(
            "context_update_start",
            {
                "agent_id": agent_id,
                "step_index": self.state.step_index,
                "sim_time_ms": self.state.sim_time_ms,
                **self._daytrader_runtime_fields(),
            },
        )
        proposal = agent.context_update(observation)
        self._notify_runtime(
            "context_update_end",
            {
                "agent_id": agent_id,
                "step_index": self.state.step_index,
                "sim_time_ms": self.state.sim_time_ms,
                **self._daytrader_runtime_fields(),
            },
        )
        return proposal

    def _finalize_time_proposal(
        self,
        agent_id: str,
        observation: Observation,
        proposal: Any,
        trigger_probe: bool = True,
        daytrader_phase_lock: str | None = None,
    ) -> dict[str, Any] | None:
        agent = self.state.agents.get(agent_id)
        if agent is None:
            return None
        self._capture_agent_memory(agent_id, agent)
        actions, context_payload = self._extract_actions_from_proposal(proposal)
        if self._log_observation_events() and context_payload is not None:
            self._emit_event(
                event_type="context_update",
                actor_id=agent_id,
                visibility="system",
                payload={
                    "agent_id": agent_id,
                    "step_index": self.state.step_index,
                    **context_payload,
                },
            )
        if not actions:
            return None
        self._record_context_memory(agent_id, observation)
        for action in actions:
            normalized = action
            if "actor_id" not in normalized:
                normalized = {**normalized, "actor_id": agent_id}
            if "timestamp" not in normalized:
                normalized = {**normalized, "timestamp": self.state.step_index}
            if daytrader_phase_lock in {"decision", "group_chat"}:
                normalized = {**normalized, "_daytrader_phase_lock": daytrader_phase_lock}
            processed = self._process_submitted_action(normalized, apply_immediately=False, trigger_probe=trigger_probe)
            if processed is not None:
                return processed
        return None

    def _handle_action_complete(self, payload: dict[str, Any]) -> None:
        agent_id = payload.get("agent_id")
        action = payload.get("action")
        if not isinstance(agent_id, str) or not isinstance(action, dict):
            return
        self._notify_runtime(
            "step_start",
            {
                "step_index": self.state.step_index + 1,
                "sim_time_ms": self.state.sim_time_ms,
                **self._daytrader_runtime_fields(),
            },
        )
        self._apply_action(action, agent_id)
        self._advance_daytrader_phase_from_action(agent_id, action)
        self.state.step_index += 1
        self._sync_daytrader_round_phase()
        self.state.observation_cache.clear()
        if self.task_hook is not None:
            self.task_hook(self.state)
        state_hash = compute_state_hash(self.state)
        self._emit_event(
            event_type="state_updated",
            actor_id="system",
            visibility="system",
            payload={
                "step_index": self.state.step_index,
                "sim_time_ms": self.state.sim_time_ms,
                "state_delta": {"step_index": self.state.step_index},
                "resulting_state_hash": state_hash,
            },
            event_id=f"step_{self.state.step_index}",
        )
        self._maybe_log_probe(had_action=True, actor_id=agent_id)
        self._set_agent_status(agent_id, "idle")

    def _handle_heartbeat(self, payload: dict[str, Any]) -> None:
        interval = payload.get("interval_ms")
        if not isinstance(interval, int) or interval <= 0:
            interval = self._proactive_wakeup_interval_ms()
        for agent_id in self.state.agents.keys():
            if self._is_agent_idle(agent_id):
                self._enqueue_event(self.state.sim_time_ms, "agent_wakeup", {"agent_id": agent_id, "reason": "heartbeat"})
        self._enqueue_event(self.state.sim_time_ms + interval, "heartbeat", {"interval_ms": interval})

    def _resolve_max_steps(self, max_steps: int | None) -> int:
        if max_steps is not None:
            return max_steps
        if isinstance(self.config, dict):
            experiment = self.config.get("experiment", {})
            if isinstance(experiment, dict):
                config_steps = experiment.get("max_steps")
                if isinstance(config_steps, int) and config_steps > 0:
                    return config_steps
        raise ValueError("max_steps must be provided when config.experiment.max_steps is missing.")

    def _reset_agents(self) -> None:
        if not isinstance(self.state.agents, dict) or not self.state.agents:
            return
        seed = None
        if isinstance(self.config, dict):
            seed = self.config.get("experiment", {}).get("seed")
        for agent in self.state.agents.values():
            if not hasattr(agent, "reset"):
                continue
            try:
                agent.reset(seed)
            except Exception:
                continue

    def step(self) -> None:
        """Execute a single controller step (placeholder)."""

        self._notify_runtime(
            "step_start",
            {
                "step_index": self.state.step_index + 1,
                **self._daytrader_runtime_fields(),
            },
        )
        self.state.step_index += 1
        self._sync_daytrader_round_phase()
        self._apply_proposal_expiry()
        self._apply_decision_timeouts()
        self.state.observation_cache.clear()
        self.state.turn_state["message_counts"] = {}
        had_action, had_non_noop_action = self._process_actions()
        if self.task_hook is not None:
            self.task_hook(self.state)
        state_hash = compute_state_hash(self.state)
        self._emit_event(
            event_type="state_updated",
            actor_id="system",
            visibility="system",
            payload={
                "step_index": self.state.step_index,
                "state_delta": {"step_index": self.state.step_index},
                "resulting_state_hash": state_hash,
            },
            event_id=f"step_{self.state.step_index}",
        )
        self._maybe_log_probe(had_action)
        self._schedule_context_updates()
        self._process_context_updates()
        late_had_action = self.state.turn_state.pop("_had_action_in_context_updates", False) is True
        late_had_non_noop = self.state.turn_state.pop("_had_non_noop_in_context_updates", False) is True
        if late_had_action:
            had_action = True
        if late_had_non_noop:
            had_non_noop_action = True
        self.state.turn_state["all_actions_do_nothing"] = had_action and not had_non_noop_action
        self._flush_logs()

    def _apply_proposal_expiry(self) -> None:
        proposals = self.state.buffers.get("proposals")
        if not isinstance(proposals, dict) or not proposals:
            return
        expired: list[str] = []
        for proposal_id, entry in proposals.items():
            if not isinstance(entry, dict):
                continue
            expires_at = entry.get("expires_at")
            if expires_at is None:
                continue
            if isinstance(expires_at, int) and self.state.step_index >= expires_at:
                expired.append(proposal_id)
        for proposal_id in expired:
            proposals.pop(proposal_id, None)
            self._emit_event(
                event_type="proposal_expired",
                actor_id="system",
                visibility="public",
                payload={"proposal_id": proposal_id},
            )

    def log_probe(self, record: dict[str, Any]) -> None:
        """Log a probe record for this run."""

        self._emit_probe_event(record)
        self.state.probe_log.append(record)
        if self.run_paths is None:
            return
        log_probe_records(self.run_paths, [record])

    def log_probe_response(
        self,
        template_id: str,
        response: ProbeResponse,
        actor_id: str | None = None,
        timestamp: str | None = None,
        prompt: str | None = None,
    ) -> None:
        """Validate and log a probe response record."""

        if self.probe_template_ids and template_id not in self.probe_template_ids:
            raise ProbeValidationError(f"Probe template not enabled: {template_id}")
        if self.probe_registry is not None:
            self.probe_registry.validate_response(template_id, response)
            template = self.probe_registry.get(template_id)
            construct = template.construct
            if prompt is None:
                prompt = template.prompt
        else:
            construct = None
            if prompt is None:
                prompt = None
        record = {
            "probe_id": response.probe_id,
            "template_id": template_id,
            "construct": construct,
            "actor_id": actor_id,
            "answer": response.answer,
            "confidence": response.confidence,
            "structured_fields": response.structured_fields,
            "prompt": prompt,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        }
        self.log_probe(record)

    def _emit_probe_event(self, record: dict[str, Any]) -> None:
        probe_id = record.get("probe_id")
        if not isinstance(probe_id, str) or not probe_id:
            return
        payload = {
            "probe_id": probe_id,
            "template_id": record.get("template_id"),
            "construct": record.get("construct"),
            "actor_id": record.get("actor_id"),
            "answer": record.get("answer"),
            "confidence": record.get("confidence"),
            "structured_fields": record.get("structured_fields"),
            "prompt": record.get("prompt"),
        }
        if record.get("answer") is None:
            self._emit_event(
                event_type="probe_asked",
                actor_id="system",
                visibility="system",
                payload=payload,
            )
            return
        self._emit_event(
            event_type="probe_answered",
            actor_id="system",
            visibility="system",
            payload=payload,
        )

    def _process_actions(self) -> tuple[bool, bool]:
        if not self.state.pending_actions:
            return False, False
        pending = list(self.state.pending_actions)
        self.state.pending_actions.clear()
        if self._step_mode() != "event":
            processed_any = False
            had_non_noop = False
            for action in pending:
                processed = self._process_submitted_action(action, apply_immediately=True)
                if not isinstance(processed, dict):
                    continue
                processed_any = True
                if processed.get("type") != "do_nothing":
                    had_non_noop = True
            return processed_any, had_non_noop

        validated_actions: list[tuple[str, dict[str, Any]]] = []
        for action in pending:
            processed = self._process_submitted_action(action, apply_immediately=False, trigger_probe=False)
            if not isinstance(processed, dict):
                continue
            actor_id = processed.get("actor_id")
            if not isinstance(actor_id, str) or not actor_id:
                continue
            validated_actions.append((actor_id, processed))

        self._maybe_probe_after_validated_actions_event(validated_actions)

        for actor_id, action in validated_actions:
            self._apply_action(action, actor_id)
        had_non_noop = any(action.get("type") != "do_nothing" for _, action in validated_actions)
        return bool(validated_actions), had_non_noop

    def _extract_actions_from_proposal(self, proposal: Any) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        actions: list[dict[str, Any]] = []
        payload: dict[str, Any] = {}

        if isinstance(proposal, ActionProposal):
            raw_actions: Any = proposal.action
            if isinstance(raw_actions, dict):
                maybe_actions = raw_actions.get("actions")
                if isinstance(maybe_actions, list):
                    raw_actions = maybe_actions
                else:
                    raw_actions = [raw_actions]
            elif not isinstance(raw_actions, list):
                raw_actions = []
            actions = [item for item in raw_actions if isinstance(item, dict)]
            if actions:
                payload["action" if len(actions) == 1 else "actions"] = actions[0] if len(actions) == 1 else actions
            if isinstance(proposal.rationale, str):
                payload["rationale"] = proposal.rationale
            if isinstance(proposal.alternatives, list):
                payload["alternatives"] = proposal.alternatives
            return actions, payload or None

        if isinstance(proposal, dict):
            maybe_actions = proposal.get("actions")
            if isinstance(maybe_actions, list):
                actions = [item for item in maybe_actions if isinstance(item, dict)]
            else:
                action = proposal.get("action")
                if isinstance(action, dict):
                    actions = [action]
            if actions:
                payload["action" if len(actions) == 1 else "actions"] = actions[0] if len(actions) == 1 else actions
            rationale = proposal.get("rationale")
            if isinstance(rationale, str):
                payload["rationale"] = rationale
            alternatives = proposal.get("alternatives")
            if isinstance(alternatives, list):
                payload["alternatives"] = alternatives
            return actions, payload or None

        return actions, None

    def _process_submitted_action(
        self,
        action: dict[str, Any],
        apply_immediately: bool,
        trigger_probe: bool = True,
    ) -> dict[str, Any] | None:
        actor_id = self._extract_actor_id(action)
        action = self._apply_action_defaults(action, actor_id=actor_id)
        self._emit_event(
            event_type="action_submitted",
            actor_id=actor_id,
            visibility="system",
            payload={"action": action},
        )
        if not self._is_action_enabled(action):
            self._emit_event(
                event_type="action_rejected",
                actor_id=actor_id,
                visibility="system",
                payload={"action": action, "error_message": "Action type not enabled by config."},
            )
            return None
        try:
            validate_action(action)
        except ActionValidationError as exc:
            self._emit_event(
                event_type="action_rejected",
                actor_id=actor_id,
                visibility="system",
                payload={"action": action, "error_message": str(exc)},
            )
            return None
        precondition_error = self._check_action_preconditions(action, actor_id)
        if precondition_error is not None:
            self._emit_event(
                event_type="action_rejected",
                actor_id=actor_id,
                visibility="system",
                payload={"action": action, "error_message": precondition_error},
            )
            return None
        self._emit_event(
            event_type="action_validated",
            actor_id=actor_id,
            visibility="system",
            payload={"action": action, "checks_passed": ["schema"]},
        )
        if trigger_probe:
            self._maybe_probe_after_valid_action(action, actor_id)
        if apply_immediately:
            self._apply_action(action, actor_id)
        return action

    def _maybe_probe_after_valid_action(self, action: dict[str, Any], actor_id: str) -> None:
        action_type = action.get("type")
        if action_type == "do_nothing":
            return
        if not isinstance(actor_id, str) or not actor_id:
            return
        records = self._log_self_probe_records(actor_id)
        if not records:
            return
        cadence = "per_action"
        relevant_agents = {actor_id}
        self._notify_probe_start(cadence, records, relevant_agents=relevant_agents)
        stats = self._collect_probe_responses(records, relevant_agents=relevant_agents)
        self._notify_probe_end(cadence, records, stats)

    def _maybe_probe_after_valid_actions_batch(self, actions_by_agent: dict[str, dict[str, Any]]) -> None:
        if not actions_by_agent:
            return
        records: list[dict[str, Any]] = []
        relevant_agents: set[str] = set()
        for actor_id in sorted(actions_by_agent.keys()):
            action = actions_by_agent.get(actor_id)
            if not isinstance(action, dict):
                continue
            if action.get("type") == "do_nothing":
                continue
            relevant_agents.add(actor_id)
            records.extend(self._log_self_probe_records(actor_id))
        if not records or not relevant_agents:
            return
        cadence = "per_action"
        self._notify_probe_start(cadence, records, relevant_agents=relevant_agents)
        stats = self._collect_probe_responses(records, relevant_agents=relevant_agents)
        self._notify_probe_end(cadence, records, stats)

    def _maybe_probe_after_validated_actions_event(
        self,
        validated_actions: list[tuple[str, dict[str, Any]]],
    ) -> None:
        if not validated_actions:
            return
        records: list[dict[str, Any]] = []
        relevant_agents: set[str] = set()
        for actor_id, action in validated_actions:
            if not isinstance(actor_id, str) or not actor_id:
                continue
            if not isinstance(action, dict):
                continue
            if action.get("type") == "do_nothing":
                continue
            relevant_agents.add(actor_id)
            records.extend(self._log_self_probe_records(actor_id))
        if not records or not relevant_agents:
            return
        cadence = "per_action"
        self._notify_probe_start(cadence, records, relevant_agents=relevant_agents)
        stats = self._collect_probe_responses(records, relevant_agents=relevant_agents)
        self._notify_probe_end(cadence, records, stats)

    def _context_update_agent_time(self, agent_id: str) -> dict[str, Any] | None:
        if not isinstance(self.state.agents, dict) or not self.state.agents:
            return None
        agent = self.state.agents.get(agent_id)
        if agent is None:
            return None
        if not hasattr(agent, "context_update"):
            return None
        self._sync_daytrader_round_phase()
        self._apply_visibility_map_from_config(agent_id)
        self._load_agent_memory(agent_id, agent)
        observation = self._get_cached_observation(agent_id)
        self._notify_runtime(
            "context_update_start",
            {
                "agent_id": agent_id,
                "step_index": self.state.step_index,
            },
        )
        proposal = agent.context_update(observation)
        self._notify_runtime(
            "context_update_end",
            {
                "agent_id": agent_id,
                "step_index": self.state.step_index,
            },
        )
        self._capture_agent_memory(agent_id, agent)
        if isinstance(proposal, ActionProposal):
            action = proposal.action
            if self._log_observation_events():
                self._emit_event(
                    event_type="context_update",
                    actor_id=agent_id,
                    visibility="system",
                    payload={
                        "agent_id": agent_id,
                        "step_index": self.state.step_index,
                        "action": proposal.action,
                        "rationale": proposal.rationale,
                        "alternatives": proposal.alternatives,
                    },
                )
        elif isinstance(proposal, dict):
            action = proposal.get("action")
        else:
            action = None
        if not isinstance(action, dict):
            return None
        if "actor_id" not in action:
            action = {**action, "actor_id": agent_id}
        if "timestamp" not in action:
            action = {**action, "timestamp": self.state.step_index}
        self._record_context_memory(agent_id, observation)
        return self._process_submitted_action(action, apply_immediately=False)

    def _maybe_log_probe(self, had_action: bool, actor_id: str | None = None) -> None:
        _ = had_action
        _ = actor_id
        # Probing now happens immediately after each validated non-do_nothing action.
        return

    def _probe_every_n_actions(self) -> int:
        if not isinstance(self.config, dict):
            return 1
        probe_cfg = self.config.get("probe", {})
        if not isinstance(probe_cfg, dict):
            return 1
        value = probe_cfg.get("every_n_actions")
        if isinstance(value, int) and value > 0:
            return value
        return 1

    def _notify_probe_start(
        self,
        cadence: str,
        records: list[dict[str, Any]],
        relevant_agents: set[str] | None = None,
    ) -> None:
        self._notify_runtime(
            "probe_start",
            {
                "step_index": self.state.step_index,
                "sim_time_ms": self.state.sim_time_ms,
                "cadence": cadence,
                "record_count": len(records),
                "request_count": self._estimate_probe_requests(records, relevant_agents=relevant_agents),
            },
        )

    def _notify_probe_end(self, cadence: str, records: list[dict[str, Any]], stats: dict[str, int]) -> None:
        self._notify_runtime(
            "probe_end",
            {
                "step_index": self.state.step_index,
                "sim_time_ms": self.state.sim_time_ms,
                "cadence": cadence,
                "record_count": len(records),
                "request_count": stats.get("total", 0),
                "processed_count": stats.get("processed", 0),
                "success_count": stats.get("success", 0),
                "error_count": stats.get("error", 0),
            },
        )

    def _estimate_probe_requests(
        self,
        records: list[dict[str, Any]],
        relevant_agents: set[str] | None = None,
    ) -> int:
        if not records:
            return 0
        if not isinstance(self.state.agents, dict) or not self.state.agents:
            return 0
        responders = {
            agent_id
            for agent_id, agent in self.state.agents.items()
            if isinstance(agent_id, str) and hasattr(agent, "respond_probe")
        }
        if relevant_agents is not None:
            responders = {agent_id for agent_id in responders if agent_id in relevant_agents}
        eligible_calls = 0
        for agent_id in responders:
            has_record = False
            for record in records:
                target_actor_id = record.get("target_actor_id")
                if isinstance(target_actor_id, str) and target_actor_id != agent_id:
                    continue
                about_agent_id = record.get("about_agent_id")
                if isinstance(about_agent_id, str) and about_agent_id == agent_id:
                    continue
                if relevant_agents is not None and isinstance(about_agent_id, str) and about_agent_id not in relevant_agents:
                    continue
                has_record = True
                break
            if has_record:
                eligible_calls += 1
        return eligible_calls

    def _log_probe_placeholder(self, relevant_about_agents: set[str] | None = None) -> list[dict[str, Any]]:
        template_id, construct, prompt = self._next_probe_template()
        questions = self._probe_questions(prompt)
        records: list[dict[str, Any]] = []
        for question in questions:
            for about_agent_id in self._expand_probe_targets(question):
                if (
                    relevant_about_agents is not None
                    and isinstance(about_agent_id, str)
                    and about_agent_id not in relevant_about_agents
                ):
                    continue
                formatted = question.replace("{other_agent_id}", about_agent_id) if about_agent_id else question
                record: dict[str, Any] = {
                    "probe_id": f"probe_{self.state.step_index}_{len(records)+1}",
                    "answer": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                if template_id is not None:
                    record["template_id"] = template_id
                if construct is not None:
                    record["construct"] = construct
                if formatted:
                    record["prompt"] = formatted
                if about_agent_id:
                    record["about_agent_id"] = about_agent_id
                self.log_probe(record)
                records.append(record)
        return records

    def _probe_cadence(self) -> str | None:
        if not isinstance(self.config, dict):
            return None
        probe_cfg = self.config.get("probe", {})
        if not isinstance(probe_cfg, dict):
            return None
        cadence = probe_cfg.get("cadence")
        if isinstance(cadence, str) and cadence:
            return cadence
        return None

    def _probe_event_types(self) -> set[str]:
        if not isinstance(self.config, dict):
            return set()
        probe_cfg = self.config.get("probe", {})
        if not isinstance(probe_cfg, dict):
            return set()
        events = probe_cfg.get("events")
        if not isinstance(events, list):
            return set()
        return {event for event in events if isinstance(event, str) and event}

    def _collect_probe_responses(
        self,
        records: list[dict[str, Any]],
        relevant_agents: set[str] | None = None,
    ) -> dict[str, int]:
        stats = {
            "total": self._estimate_probe_requests(records, relevant_agents=relevant_agents),
            "processed": 0,
            "success": 0,
            "error": 0,
        }
        if not records:
            return stats
        if not isinstance(self.state.agents, dict) or not self.state.agents:
            return stats

        responders: list[str] = []
        for agent_id, agent in self.state.agents.items():
            if not isinstance(agent_id, str) or not hasattr(agent, "respond_probe"):
                continue
            if relevant_agents is not None and agent_id not in relevant_agents:
                continue
            responders.append(agent_id)

        request_payloads: list[tuple[str, list[dict[str, Any]], Observation, str]] = []
        for agent_id in responders:
            agent_records = []
            for record in records:
                target_actor_id = record.get("target_actor_id")
                if isinstance(target_actor_id, str) and target_actor_id != agent_id:
                    continue
                about_agent_id = record.get("about_agent_id")
                if isinstance(about_agent_id, str) and about_agent_id == agent_id:
                    continue
                if relevant_agents is not None and isinstance(about_agent_id, str) and about_agent_id not in relevant_agents:
                    continue
                agent_records.append(record)
            if not agent_records:
                continue
            observation = self._get_cached_observation(agent_id)
            prompt = self._build_batched_probe_prompt(agent_records)
            request_payloads.append((agent_id, agent_records, observation, prompt))

        responses_by_agent: dict[str, Any] = {}
        failed_agents: set[str] = set()
        max_workers = min(self._max_concurrent_requests(), len(request_payloads))
        if max_workers <= 1:
            for agent_id, _agent_records, observation, prompt in request_payloads:
                agent = self.state.agents.get(agent_id)
                if agent is None:
                    failed_agents.add(agent_id)
                    continue
                try:
                    responses_by_agent[agent_id] = agent.respond_probe(
                        probe_id=f"probe_batch_{self.state.step_index}_{agent_id}",
                        prompt=prompt,
                        construct="batched_probe",
                        observation=observation,
                    )
                except Exception:
                    failed_agents.add(agent_id)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for agent_id, _agent_records, observation, prompt in request_payloads:
                    agent = self.state.agents.get(agent_id)
                    if agent is None:
                        failed_agents.add(agent_id)
                        continue
                    futures[
                        executor.submit(
                            agent.respond_probe,
                            probe_id=f"probe_batch_{self.state.step_index}_{agent_id}",
                            prompt=prompt,
                            construct="batched_probe",
                            observation=observation,
                        )
                    ] = agent_id
                for future in as_completed(futures):
                    agent_id = futures[future]
                    try:
                        responses_by_agent[agent_id] = future.result()
                    except Exception:
                        failed_agents.add(agent_id)

        for agent_id, agent_records, _observation, _prompt in request_payloads:
            stats["processed"] += 1
            if agent_id in failed_agents:
                stats["error"] += 1
                self._notify_probe_progress(stats)
                continue

            response = responses_by_agent.get(agent_id)
            parsed = self._parse_batched_probe_response(response)
            if parsed is None:
                stats["error"] += 1
                self._notify_probe_progress(stats)
                continue

            had_validation_error = False
            logged_any_response = False
            for record in agent_records:
                probe_id = record.get("probe_id")
                template_id = record.get("template_id")
                source_prompt = record.get("prompt")
                if not isinstance(probe_id, str) or not probe_id:
                    continue
                if not isinstance(template_id, str) or not template_id:
                    continue
                item = parsed.get(probe_id, {})
                answer = item.get("answer")
                confidence = item.get("confidence")
                structured_fields = item.get("structured_fields")
                if answer is None:
                    had_validation_error = True
                    continue
                try:
                    self.log_probe_response(
                        template_id,
                        ProbeResponse(
                            probe_id=probe_id,
                            answer=answer,
                            confidence=confidence if isinstance(confidence, (int, float)) else None,
                            structured_fields=structured_fields if isinstance(structured_fields, dict) else None,
                        ),
                        actor_id=agent_id,
                        prompt=source_prompt if isinstance(source_prompt, str) else None,
                    )
                    logged_any_response = True
                except ProbeValidationError:
                    had_validation_error = True
                    continue
            if logged_any_response and not had_validation_error:
                stats["success"] += 1
            else:
                stats["error"] += 1
            self._notify_probe_progress(stats)
        return stats

    def _notify_probe_progress(self, stats: dict[str, int]) -> None:
        processed = stats.get("processed", 0)
        total = stats.get("total", 0)
        if total <= 0:
            return
        if processed % 5 != 0 and processed != total:
            return
        self._notify_runtime(
            "probe_progress",
            {
                "step_index": self.state.step_index,
                "sim_time_ms": self.state.sim_time_ms,
                "processed_count": processed,
                "request_count": total,
                "success_count": stats.get("success", 0),
                "error_count": stats.get("error", 0),
            },
        )

    def _resolve_probe_relevant_agents(self, new_events: list[dict[str, Any]]) -> set[str] | None:
        if not isinstance(self.state.agents, dict) or not self.state.agents:
            return None
        all_agents = {
            agent_id for agent_id in self.state.agents.keys() if isinstance(agent_id, str) and agent_id
        }
        if not new_events:
            return None
        relevant: set[str] = set()
        trigger_action_types = {
            "communicate",
            "propose_trade_offer",
            "trade_response",
            "cancel_trade_offer",
            "fulfill_order",
        }
        for event in new_events:
            if not isinstance(event, dict):
                continue
            event_type = event.get("event_type")
            payload = event.get("payload", {})
            visibility = event.get("visibility")
            if event_type != "action_submitted":
                if visibility == "public" and isinstance(event_type, str) and "trade" in event_type:
                    # Public trade updates are observable by everyone.
                    return all_agents
                continue
            if not isinstance(payload, dict):
                continue
            action = payload.get("action")
            if not isinstance(action, dict):
                continue
            action_type = action.get("type")
            if action_type not in trigger_action_types:
                continue
            actor_id = action.get("actor_id")
            if isinstance(actor_id, str) and actor_id in all_agents:
                relevant.add(actor_id)
            action_payload = action.get("payload")
            if not isinstance(action_payload, dict):
                continue
            if action_type == "communicate":
                recipients = action_payload.get("recipients")
                if isinstance(recipients, list):
                    for value in recipients:
                        if isinstance(value, str) and value in all_agents:
                            relevant.add(value)
            elif action_type == "propose_trade_offer":
                target_id = action_payload.get("target_id")
                if isinstance(target_id, str) and target_id in all_agents:
                    relevant.add(target_id)
            elif action_type in {"trade_response", "cancel_trade_offer"}:
                counterparty = self._resolve_trade_counterparty(action_payload, actor_id)
                if isinstance(counterparty, str) and counterparty in all_agents:
                    relevant.add(counterparty)
        return relevant or None

    def _resolve_probe_pairs(self, new_events: list[dict[str, Any]]) -> list[tuple[str, str]]:
        if not isinstance(self.state.agents, dict) or not self.state.agents:
            return []
        all_agents = {
            agent_id for agent_id in self.state.agents.keys() if isinstance(agent_id, str) and agent_id
        }
        pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for event in new_events:
            if not isinstance(event, dict) or event.get("event_type") != "action_submitted":
                continue
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            action = payload.get("action")
            if not isinstance(action, dict):
                continue
            actor_id = action.get("actor_id")
            if not isinstance(actor_id, str) or actor_id not in all_agents:
                continue
            action_type = action.get("type")
            action_payload = action.get("payload")
            if not isinstance(action_payload, dict):
                continue
            recipients: set[str] = set()
            if action_type == "communicate":
                raw = action_payload.get("recipients")
                if isinstance(raw, list):
                    recipients.update(
                        value for value in raw if isinstance(value, str) and value in all_agents and value != actor_id
                    )
            else:
                target_id = action_payload.get("target_id")
                if isinstance(target_id, str) and target_id in all_agents and target_id != actor_id:
                    recipients.add(target_id)
                if action_type in {"trade_response", "cancel_trade_offer"} and not recipients:
                    counterparty = self._resolve_trade_counterparty(action_payload, actor_id)
                    if isinstance(counterparty, str) and counterparty in all_agents and counterparty != actor_id:
                        recipients.add(counterparty)
            for recipient in sorted(recipients):
                pair = (actor_id, recipient)
                if pair in seen:
                    continue
                seen.add(pair)
                pairs.append(pair)
        return pairs

    def _self_probe_questions(self) -> list[str]:
        return [
            "What immediate goal is other agents pursuing with respect to you?",
            "What relevant information do you believe  other agents currently knows?",
            "What action do you expect  other agents to take next?",
        ]

    def _log_self_probe_records(self, actor_id: str) -> list[dict[str, Any]]:
        template_id, construct, _ = self._next_probe_template()
        questions = self._self_probe_questions()
        records: list[dict[str, Any]] = []
        for question in questions:
            record: dict[str, Any] = {
                "probe_id": f"probe_{self.state.step_index}_{len(records)+1}",
                "answer": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "target_actor_id": actor_id,
                "prompt": question,
            }
            if template_id is not None:
                record["template_id"] = template_id
            if construct is not None:
                record["construct"] = construct
            self.log_probe(record)
            records.append(record)
        return records

    def _resolve_trade_counterparty(self, action_payload: dict[str, Any], actor_id: Any) -> str | None:
        transaction_id = action_payload.get("transaction_id")
        if not isinstance(transaction_id, str) or not isinstance(actor_id, str):
            return None
        task_state = self.state.task_state
        if not isinstance(task_state, dict):
            return None
        for key in ("pending_offers", "completed_trades"):
            offers = task_state.get(key)
            if not isinstance(offers, list):
                continue
            for offer in offers:
                if not isinstance(offer, dict) or offer.get("id") != transaction_id:
                    continue
                from_id = offer.get("from")
                to_id = offer.get("to")
                if isinstance(from_id, str) and from_id != actor_id:
                    return from_id
                if isinstance(to_id, str) and to_id != actor_id:
                    return to_id
        return None

    def _build_batched_probe_prompt(self, records: list[dict[str, Any]]) -> str:
        lines = [
            "Answer all probe items in one JSON object.",
            "Return strict JSON only with this shape:",
            '{"answer":{"responses":[{"probe_id":"...","answer":"...","confidence":0.0,"structured_fields":{}}]}}',
            "Use confidence in [0,1]. Keep each answer grounded in current observation and visible events.",
            "Items:",
        ]
        for record in records:
            probe_id = record.get("probe_id")
            question = record.get("prompt")
            about_agent = record.get("about_agent_id")
            if not isinstance(probe_id, str) or not isinstance(question, str):
                continue
            if isinstance(about_agent, str) and about_agent:
                lines.append(f'- probe_id="{probe_id}", about_agent_id="{about_agent}", question="{question}"')
            else:
                lines.append(f'- probe_id="{probe_id}", question="{question}"')
        return "\n".join(lines)

    def _parse_batched_probe_response(self, response: Any) -> dict[str, dict[str, Any]] | None:
        payload: Any = None
        if isinstance(response, ProbeResponse):
            payload = response.answer
        elif isinstance(response, dict):
            payload = response.get("answer")
            if payload is None:
                payload = response
        else:
            payload = response

        parsed: Any = None
        if isinstance(payload, dict):
            parsed = payload
        elif isinstance(payload, str):
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                return None
        else:
            return None

        responses = None
        if isinstance(parsed, dict):
            direct = parsed.get("responses")
            wrapped = parsed.get("answer")
            if isinstance(direct, list):
                responses = direct
            elif isinstance(wrapped, dict):
                nested = wrapped.get("responses")
                if isinstance(nested, list):
                    responses = nested
        if not isinstance(responses, list):
            return None
        result: dict[str, dict[str, Any]] = {}
        for item in responses:
            if not isinstance(item, dict):
                continue
            probe_id = item.get("probe_id")
            if not isinstance(probe_id, str) or not probe_id:
                continue
            result[probe_id] = {
                "answer": item.get("answer"),
                "confidence": item.get("confidence"),
                "structured_fields": item.get("structured_fields"),
            }
        return result or None

    def _probe_questions(self, fallback_prompt: str | None) -> list[str]:
        if not isinstance(self.config, dict):
            return [fallback_prompt] if fallback_prompt else []
        probe_cfg = self.config.get("probe", {})
        if not isinstance(probe_cfg, dict):
            return [fallback_prompt] if fallback_prompt else []
        questions_path = probe_cfg.get("questions_path")
        if isinstance(questions_path, str) and questions_path:
            text = load_text_file(questions_path)
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, list):
                return [item for item in payload if isinstance(item, str) and item]
        questions = probe_cfg.get("questions")
        if isinstance(questions, list):
            return [item for item in questions if isinstance(item, str) and item]
        return [fallback_prompt] if fallback_prompt else []

    def _expand_probe_targets(self, question: str) -> list[str | None]:
        if "{other_agent_id}" not in question:
            return [None]
        if not isinstance(self.state.agents, dict):
            return [None]
        return [agent_id for agent_id in self.state.agents.keys() if isinstance(agent_id, str) and agent_id]

    def _next_probe_template(self) -> tuple[str | None, str | None, str | None]:
        if not self.probe_template_ids:
            return None, None, None
        template_id = self.probe_template_ids[self.state.probe_index % len(self.probe_template_ids)]
        self.state.probe_index += 1
        if self.probe_registry is None:
            return template_id, None, None
        try:
            template = self.probe_registry.get(template_id)
        except KeyError:
            return template_id, None, None
        return template.template_id, template.construct, template.prompt

    def _context_update_agent(self, agent_id: str) -> None:
        if not isinstance(self.state.agents, dict) or not self.state.agents:
            return
        agent = self.state.agents.get(agent_id)
        if agent is None:
            return
        if not hasattr(agent, "context_update"):
            return
        if not self._is_agent_idle(agent_id):
            return
        self._sync_daytrader_round_phase()
        self._apply_visibility_map_from_config(agent_id)
        self._load_agent_memory(agent_id, agent)
        observation = self._get_cached_observation(agent_id)
        self._set_agent_status(agent_id, "busy")
        self._notify_runtime(
            "context_update_start",
            {
                "agent_id": agent_id,
                "step_index": self.state.step_index,
            },
        )
        proposal = agent.context_update(observation)
        self._notify_runtime(
            "context_update_end",
            {
                "agent_id": agent_id,
                "step_index": self.state.step_index,
            },
        )
        self._set_agent_status(agent_id, "idle")
        self._capture_agent_memory(agent_id, agent)
        actions, context_payload = self._extract_actions_from_proposal(proposal)
        if self._log_observation_events() and context_payload is not None:
            self._emit_event(
                event_type="context_update",
                actor_id=agent_id,
                visibility="system",
                payload={
                    "agent_id": agent_id,
                    "step_index": self.state.step_index,
                    **context_payload,
                },
            )
        if not actions:
            return
        for action in actions:
            normalized = action
            if "actor_id" not in normalized and isinstance(agent_id, str):
                normalized = {**normalized, "actor_id": agent_id}
            if "timestamp" not in normalized:
                normalized = {**normalized, "timestamp": self.state.step_index}
            self.state.pending_actions.append(normalized)
        self._record_context_memory(agent_id, observation)

    def _schedule_context_updates(self) -> None:
        if not isinstance(self.state.agents, dict) or not self.state.agents:
            return
        if self._step_mode() == "event":
            for agent_id in self.state.agents.keys():
                self.state.pending_context_updates[agent_id] = {"signature": f"event_step_{self.state.step_index}"}
            return
        for agent_id in self.state.agents.keys():
            signature = self._context_signature(agent_id)
            if signature is None:
                continue
            if signature != self.state.last_context_hash.get(agent_id):
                self.state.pending_context_updates[agent_id] = {"signature": signature}

    def _process_context_updates(self) -> None:
        if not self.state.pending_context_updates:
            return
        pending = dict(self.state.pending_context_updates)
        if self._uses_maptask_event_pipeline():
            agent_ids = self._resolve_maptask_query_order()
            had_action_in_context = False
            had_non_noop_in_context = False
            validated_actions_for_probe: list[tuple[str, dict[str, Any]]] = []
            follower_retry_budget: dict[str, int] = {}
            for agent_id in agent_ids:
                if agent_id not in pending:
                    continue
                if not self._is_agent_idle(agent_id):
                    continue
                self.state.pending_context_updates.pop(agent_id, None)
                self._context_update_agent(agent_id)
                self.state.last_context_hash[agent_id] = pending[agent_id].get("signature", "")
                context_had_action, context_had_non_noop, validated_actions, rejected_actors = (
                    self._drain_pending_actions_for_maptask_event()
                )
                had_action_in_context = had_action_in_context or context_had_action
                had_non_noop_in_context = had_non_noop_in_context or context_had_non_noop
                validated_actions_for_probe.extend(validated_actions)
                if (
                    agent_id in rejected_actors
                    and self._is_maptask_follower(agent_id)
                    and follower_retry_budget.get(agent_id, 0) < 1
                ):
                    follower_retry_budget[agent_id] = follower_retry_budget.get(agent_id, 0) + 1
                    self.state.observation_cache.pop(agent_id, None)
                    self._context_update_agent(agent_id)
                    retry_had_action, retry_had_non_noop, retry_validated_actions, _ = (
                        self._drain_pending_actions_for_maptask_event()
                    )
                    had_action_in_context = had_action_in_context or retry_had_action
                    had_non_noop_in_context = had_non_noop_in_context or retry_had_non_noop
                    validated_actions_for_probe.extend(retry_validated_actions)
            self._maybe_probe_after_validated_actions_event(validated_actions_for_probe)
            self.state.turn_state["_had_action_in_context_updates"] = had_action_in_context
            self.state.turn_state["_had_non_noop_in_context_updates"] = had_non_noop_in_context
            return

        agent_ids = self._resolve_turn_order()
        if self._turn_taking() == "sequential":
            agent_ids = agent_ids[:1]
        for agent_id in agent_ids:
            if agent_id not in pending:
                continue
            if not self._is_agent_idle(agent_id):
                continue
            self.state.pending_context_updates.pop(agent_id, None)
            self._context_update_agent(agent_id)
            self.state.last_context_hash[agent_id] = pending[agent_id].get("signature", "")

    def _drain_pending_actions_for_maptask_event(
        self,
    ) -> tuple[bool, bool, list[tuple[str, dict[str, Any]]], set[str]]:
        if not self.state.pending_actions:
            return False, False, [], set()
        pending = list(self.state.pending_actions)
        self.state.pending_actions.clear()
        validated_actions: list[tuple[str, dict[str, Any]]] = []
        rejected_actors: set[str] = set()
        for action in pending:
            processed = self._process_submitted_action(action, apply_immediately=False, trigger_probe=False)
            if not isinstance(processed, dict):
                continue
            actor_id = processed.get("actor_id")
            if not isinstance(actor_id, str) or not actor_id:
                continue
            validated_actions.append((actor_id, processed))
            before_events = len(self.state.event_log)
            self._apply_action(processed, actor_id)
            if self._has_new_update_map_rejection(actor_id, start_index=before_events):
                rejected_actors.add(actor_id)
        had_non_noop = any(action.get("type") != "do_nothing" for _, action in validated_actions)
        return bool(validated_actions), had_non_noop, validated_actions, rejected_actors

    def _has_new_update_map_rejection(self, actor_id: str, start_index: int) -> bool:
        if start_index < 0:
            start_index = 0
        for event in self.state.event_log[start_index:]:
            if not isinstance(event, dict):
                continue
            if event.get("event_type") != "action_rejected":
                continue
            if event.get("actor_id") != actor_id:
                continue
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            action = payload.get("action")
            if isinstance(action, dict) and action.get("type") == "update_map_progress":
                return True
        return False

    def _is_maptask_follower(self, agent_id: str) -> bool:
        participants = self.state.task_state.get("participants")
        if not isinstance(participants, dict):
            return False
        entry = participants.get(agent_id)
        if not isinstance(entry, dict):
            return False
        return entry.get("role") == "follower"

    def _resolve_maptask_query_order(self) -> list[str]:
        if not isinstance(self.state.agents, dict):
            return []
        participants = self.state.task_state.get("participants")
        if not isinstance(participants, dict):
            return sorted(self.state.agents.keys())

        guiders: list[str] = []
        followers: list[str] = []
        others: list[str] = []
        for agent_id in sorted(self.state.agents.keys()):
            participant = participants.get(agent_id)
            role = participant.get("role") if isinstance(participant, dict) else None
            if role == "guider":
                guiders.append(agent_id)
            elif role == "follower":
                followers.append(agent_id)
            else:
                others.append(agent_id)
        return [*guiders, *followers, *others]

    def _experiment_type(self) -> str | None:
        if not isinstance(self.config, dict):
            return None
        experiment = self.config.get("experiment", {})
        if not isinstance(experiment, dict):
            return None
        value = experiment.get("type")
        if isinstance(value, str) and value:
            return value
        return None

    def _uses_maptask_event_pipeline(self) -> bool:
        if self._step_mode() != "event":
            return False
        experiment_type = self._experiment_type()
        if experiment_type == "maptask":
            return True
        return self._task_type() == "maptask"

    def _context_signature(self, agent_id: str) -> str | None:
        try:
            snapshot = build_state_snapshot(self.state)
            visible_state = self._filter_state_for_agent(snapshot, agent_id)
            visible_events = self._filter_visible_events_for_agent(agent_id)
            payload = json.dumps(
                {"state": visible_state, "events": visible_events},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()
        except Exception:
            return None

    def _record_context_memory(self, agent_id: str, observation: Observation) -> None:
        limit = self._memory_turn_limit()
        if limit <= 0:
            return
        entry = {
            "step_index": observation.step_index,
            "state": observation.state,
            "visible_events": observation.visible_events,
        }
        history = self.state.context_memory.setdefault(agent_id, [])
        history.append(entry)
        if len(history) > limit:
            self.state.context_memory[agent_id] = history[-limit:]

    def _memory_turn_limit(self) -> int:
        if not isinstance(self.config, dict):
            return 7
        protocol = self.config.get("protocol", {})
        if not isinstance(protocol, dict):
            return 7
        limit = protocol.get("memory_turn_limit")
        if isinstance(limit, int) and limit > 0:
            return limit
        return 7

    def _is_agent_idle(self, agent_id: str) -> bool:
        status = self.state.agent_status.get(agent_id, "idle")
        return status == "idle"

    def _set_agent_status(self, agent_id: str, status: str) -> None:
        self.state.agent_status[agent_id] = status

    def _turn_taking(self) -> str:
        if self.config is None:
            return "simultaneous"
        protocol = self.config.get("protocol", {})
        if not isinstance(protocol, dict):
            return "simultaneous"
        strategy = protocol.get("turn_taking")
        if isinstance(strategy, str) and strategy:
            return strategy
        return "simultaneous"

    def _resolve_turn_order(self) -> list[str]:
        if not isinstance(self.state.agents, dict):
            return []
        order = self.state.turn_state.get("order")
        if (
            self._turn_taking() == "sequential"
            and isinstance(order, list)
            and all(isinstance(item, str) for item in order)
        ):
            if set(order) == set(self.state.agents.keys()):
                return order
        resolved = sorted(self.state.agents.keys())
        if self._turn_taking() != "sequential":
            # Simultaneous mode should not enforce a fixed ABC... order.
            # Shuffle each scheduling cycle while staying reproducible via seed.
            self._turn_rng.shuffle(resolved)
            self.state.turn_state["last_order"] = resolved
            return resolved
        self.state.turn_state["order"] = resolved
        return resolved

    def _turn_order_seed(self) -> int:
        if not isinstance(self.config, dict):
            return 0
        experiment = self.config.get("experiment", {})
        if not isinstance(experiment, dict):
            return 0
        seed = experiment.get("seed")
        if isinstance(seed, int):
            return seed
        return 0

    def _apply_visibility_map_from_config(self, agent_id: str) -> None:
        if self.config is None:
            return
        controls = self.config.get("controls", {})
        if not isinstance(controls, dict):
            return
        visibility_map = controls.get("visibility_map")
        if not isinstance(visibility_map, dict):
            return
        entry = visibility_map.get(agent_id)
        if isinstance(entry, dict) and agent_id not in self.state.visibility_map:
            self.state.visibility_map[agent_id] = entry

    def _build_observation_for_agent(self, agent_id: str) -> Observation:
        snapshot = build_state_snapshot(self.state)
        visible_state = self._filter_state_for_agent(snapshot, agent_id)
        visible_events = self._filter_visible_events_for_agent(agent_id)
        return Observation(
            state=visible_state,
            visible_events=visible_events,
            step_index=self.state.step_index,
            memory=self._context_memory_for_agent(agent_id),
        )

    def _context_memory_for_agent(self, agent_id: str) -> dict[str, Any] | None:
        history = self.state.context_memory.get(agent_id)
        if not isinstance(history, list) or not history:
            return None
        limit = self._memory_turn_limit()
        trimmed = history[-limit:] if limit > 0 else history
        return {"turns": copy.deepcopy(trimmed)}

    def _get_cached_observation(self, agent_id: str) -> Observation:
        cached = self.state.observation_cache.get(agent_id)
        if cached is not None:
            return cached
        observation = self._build_observation_for_agent(agent_id)
        self.state.observation_cache[agent_id] = observation
        if self._log_observation_events():
            observation_payload = {
                "state": copy.deepcopy(observation.state),
                "visible_events": copy.deepcopy(observation.visible_events),
                "memory": copy.deepcopy(observation.memory),
            }
            self._emit_event(
                event_type="observation_built",
                actor_id=agent_id,
                visibility="system",
                payload={
                    "agent_id": agent_id,
                    "step_index": observation.step_index,
                    "observation": observation_payload,
                    "context": self._build_agent_input_context(agent_id),
                },
            )
        return observation

    def _filter_state_for_agent(self, snapshot: dict[str, Any], agent_id: str) -> dict[str, Any]:
        self._apply_visibility_overrides_from_config(agent_id)
        visibility = self.state.visibility_map.get(agent_id)
        self._apply_visibility_defaults_from_config(agent_id)
        defaults = self._visibility_defaults_for_agent(agent_id)
        if not visibility and not defaults:
            return snapshot
        filtered: dict[str, Any] = {}
        for section in ("task_state", "resources", "turn_state", "buffers"):
            section_value = snapshot.get(section, {})
            if section == "task_state" and isinstance(section_value, dict):
                section_value = self._apply_private_fact_visibility(section_value, agent_id)
            allowed = visibility.get(section, []) if isinstance(visibility, dict) else []
            if not isinstance(allowed, list) or not allowed:
                allowed = defaults.get(section, []) if isinstance(defaults, dict) else []
            if not isinstance(allowed, list) or not allowed:
                continue
            allowed = self._apply_visibility_overrides(section, allowed, section_value, agent_id)
            allowed = self._apply_visibility_excludes(section, allowed, section_value, agent_id)
            if "*" in allowed:
                filtered[section] = section_value
                continue
            if isinstance(section_value, dict):
                filtered[section] = {key: value for key, value in section_value.items() if key in allowed}
            else:
                filtered[section] = section_value
        filtered["step_index"] = snapshot.get("step_index")
        return filtered

    def _apply_private_fact_visibility(self, task_state: dict[str, Any], agent_id: str) -> dict[str, Any]:
        if not self._is_asymmetric_visibility():
            return task_state
        private_facts = task_state.get("private_facts")
        if not isinstance(private_facts, dict):
            return task_state
        if agent_id not in private_facts:
            return task_state
        return {
            **task_state,
            "private_facts": {agent_id: private_facts.get(agent_id)},
        }

    def _is_asymmetric_visibility(self) -> bool:
        if not isinstance(self.config, dict):
            return False
        controls = self.config.get("controls", {})
        if not isinstance(controls, dict):
            return False
        distribution = controls.get("information_distribution", {})
        if not isinstance(distribution, dict):
            return False
        return distribution.get("visibility") == "asymmetric"

    def _apply_visibility_defaults_from_config(self, agent_id: str) -> None:
        if self.state.visibility_defaults_by_agent.get(agent_id) is not None:
            return
        if not isinstance(self.config, dict):
            return
        controls = self.config.get("controls", {})
        if not isinstance(controls, dict):
            return
        per_agent = controls.get("visibility_defaults_by_agent")
        if isinstance(per_agent, dict) and agent_id in per_agent:
            entry = per_agent.get(agent_id)
            if isinstance(entry, dict):
                self.state.visibility_defaults_by_agent[agent_id] = {
                    key: value
                    for key, value in entry.items()
                    if isinstance(key, str) and isinstance(value, list)
                }
        defaults = controls.get("visibility_defaults")
        if not isinstance(defaults, dict):
            return
        if not self.state.visibility_defaults:
            self.state.visibility_defaults = {
                key: value
                for key, value in defaults.items()
                if isinstance(key, str) and isinstance(value, list)
            }

    def _visibility_defaults_for_agent(self, agent_id: str) -> dict[str, list[str]] | None:
        if agent_id in self.state.visibility_defaults_by_agent:
            return self.state.visibility_defaults_by_agent.get(agent_id)
        if self.state.visibility_defaults:
            return self.state.visibility_defaults
        return None

    def _apply_visibility_overrides_from_config(self, agent_id: str) -> None:
        if self.state.visibility_overrides_by_agent.get(agent_id) is not None:
            return
        if not isinstance(self.config, dict):
            return
        controls = self.config.get("controls", {})
        if not isinstance(controls, dict):
            return
        per_agent_overrides = controls.get("visibility_overrides_by_agent")
        if isinstance(per_agent_overrides, dict) and agent_id in per_agent_overrides:
            entry = per_agent_overrides.get(agent_id)
            if isinstance(entry, dict):
                self.state.visibility_overrides_by_agent[agent_id] = {
                    key: value
                    for key, value in entry.items()
                    if isinstance(key, str) and isinstance(value, list)
                }

    def _apply_visibility_overrides(
        self,
        section: str,
        allowed: list[str],
        section_value: Any,
        agent_id: str,
    ) -> list[str]:
        if not isinstance(self.config, dict):
            return allowed
        controls = self.config.get("controls", {})
        if not isinstance(controls, dict):
            return allowed
        if not isinstance(section_value, dict):
            return allowed
        per_agent = self.state.visibility_overrides_by_agent.get(agent_id)
        entry = None
        if isinstance(per_agent, dict):
            entry = per_agent.get(section)
        if entry is None:
            overrides = controls.get("visibility_overrides")
            if isinstance(overrides, dict):
                entry = overrides.get(section)
        if not isinstance(entry, list) or not entry:
            return allowed
        merged = list(allowed)
        for field in entry:
            if isinstance(field, str) and field and field not in merged:
                merged.append(field)
        return merged

    def _apply_visibility_excludes(
        self,
        section: str,
        allowed: list[str],
        section_value: Any,
        agent_id: str,
    ) -> list[str]:
        if not isinstance(self.config, dict):
            return allowed
        controls = self.config.get("controls", {})
        if not isinstance(controls, dict):
            return allowed
        per_agent = controls.get("visibility_excludes_by_agent")
        entry = None
        if isinstance(per_agent, dict):
            entry = per_agent.get(agent_id)
            if isinstance(entry, dict):
                entry = entry.get(section)
        if entry is None:
            excludes = controls.get("visibility_excludes")
            if isinstance(excludes, dict):
                entry = excludes.get(section)
        if not isinstance(entry, list) or not entry:
            return allowed
        exclude_set = {field for field in entry if isinstance(field, str) and field}
        if not exclude_set:
            return allowed
        if "*" in allowed:
            if isinstance(section_value, dict):
                return [key for key in section_value.keys() if key not in exclude_set]
            return allowed
        return [field for field in allowed if field not in exclude_set]

    def _filter_visible_events_for_agent(self, agent_id: str) -> list[dict[str, Any]]:
        visible: list[dict[str, Any]] = []
        events = self._visible_event_window(self.state.event_log, agent_id)
        allowed_types, per_agent_include = self._visible_event_types(agent_id)
        excluded_types = self._visible_event_types_exclude(agent_id, per_agent_include)
        for event in events:
            if allowed_types is not None and event.get("event_type") not in allowed_types:
                continue
            if excluded_types is not None and event.get("event_type") in excluded_types:
                continue
            visibility = event.get("visibility")
            if visibility == "public":
                visible.append(event)
                continue
            if visibility == "private":
                payload = event.get("payload", {})
                recipients = payload.get("recipients") if isinstance(payload, dict) else None
                if isinstance(recipients, list) and agent_id in recipients:
                    visible.append(event)
                continue
            if visibility == "system":
                if event.get("event_type") == "action_rejected" and event.get("actor_id") == agent_id:
                    visible.append(event)
        return visible

    def _visible_event_window(self, events: list[dict[str, Any]], agent_id: str) -> list[dict[str, Any]]:
        window = None
        if isinstance(self.config, dict):
            protocol = self.config.get("protocol", {})
            if isinstance(protocol, dict):
                per_agent = protocol.get("visible_event_window_by_agent")
                if isinstance(per_agent, dict):
                    entry = per_agent.get(agent_id)
                    if isinstance(entry, int):
                        window = entry
                if window is None:
                    window = protocol.get("visible_event_window")
        if isinstance(window, int) and window > 0:
            return events[-window:]
        return events

    def _visible_event_types(self, agent_id: str) -> tuple[set[str] | None, bool]:
        if not isinstance(self.config, dict):
            return None, False
        protocol = self.config.get("protocol", {})
        if not isinstance(protocol, dict):
            return None, False
        per_agent = protocol.get("visible_event_types_by_agent")
        if isinstance(per_agent, dict):
            entry = per_agent.get(agent_id)
            if isinstance(entry, list) and entry:
                allowed = {value for value in entry if isinstance(value, str) and value}
                return allowed or None, True
        types = protocol.get("visible_event_types")
        if not isinstance(types, list) or not types:
            return None, False
        allowed = {value for value in types if isinstance(value, str) and value}
        return allowed or None, False

    def _visible_event_types_exclude(self, agent_id: str, per_agent_include: bool) -> set[str] | None:
        if not isinstance(self.config, dict):
            return None
        protocol = self.config.get("protocol", {})
        if not isinstance(protocol, dict):
            return None
        per_agent = protocol.get("visible_event_types_exclude_by_agent")
        if isinstance(per_agent, dict):
            entry = per_agent.get(agent_id)
            if isinstance(entry, list) and entry:
                excluded = {value for value in entry if isinstance(value, str) and value}
                return excluded or None
        if per_agent_include:
            return None
        types = protocol.get("visible_event_types_exclude")
        if not isinstance(types, list) or not types:
            return None
        excluded = {value for value in types if isinstance(value, str) and value}
        return excluded or None

    def _capture_agent_memory(self, agent_id: str, agent: Any) -> None:
        if not hasattr(agent, "serialize"):
            return
        try:
            memory = agent.serialize()
        except Exception:
            return
        if isinstance(memory, dict):
            self._apply_agent_memory_limits_from_config(agent_id)
            self.state.agent_memory[agent_id] = self._limit_agent_memory(memory, agent_id)

    def _load_agent_memory(self, agent_id: str, agent: Any) -> None:
        if not hasattr(agent, "load"):
            return
        memory = self.state.agent_memory.get(agent_id)
        if not isinstance(memory, dict):
            return
        try:
            agent.load(memory)
        except Exception:
            return

    def _limit_agent_memory(self, memory: dict[str, Any], agent_id: str) -> dict[str, Any]:
        limit = self.state.agent_memory_limits.get(agent_id)
        if limit is None and isinstance(self.config, dict):
            protocol = self.config.get("protocol", {})
            if isinstance(protocol, dict):
                limit = protocol.get("agent_memory_key_limit")
        if not isinstance(limit, int) or limit <= 0:
            return memory
        keys = sorted(memory.keys())
        trimmed_keys = keys[:limit]
        return {key: memory[key] for key in trimmed_keys}

    def _limit_observation_memory(
        self,
        memory: dict[str, Any] | None,
        agent_id: str,
    ) -> dict[str, Any] | None:
        if memory is None:
            return None
        limit = None
        if isinstance(self.config, dict):
            protocol = self.config.get("protocol", {})
            if isinstance(protocol, dict):
                per_agent = protocol.get("observation_memory_key_limit_by_agent")
                if isinstance(per_agent, dict):
                    entry = per_agent.get(agent_id)
                    if isinstance(entry, int):
                        limit = entry
                if limit is None:
                    limit = protocol.get("observation_memory_key_limit")
        if not isinstance(limit, int) or limit <= 0:
            return memory
        keys = sorted(memory.keys())
        trimmed_keys = keys[:limit]
        return {key: memory[key] for key in trimmed_keys}

    def _apply_agent_memory_limits_from_config(self, agent_id: str) -> None:
        if self.state.agent_memory_limits.get(agent_id) is not None:
            return
        if not isinstance(self.config, dict):
            return
        protocol = self.config.get("protocol", {})
        if not isinstance(protocol, dict):
            return
        per_agent = protocol.get("agent_memory_key_limit_by_agent")
        if not isinstance(per_agent, dict):
            return
        limit = per_agent.get(agent_id)
        if isinstance(limit, int) and limit > 0:
            self.state.agent_memory_limits[agent_id] = limit

    def _apply_action(self, action: dict[str, Any], actor_id: str) -> None:
        self._queue_targeted_realtime_triggers(action, actor_id)
        if self._apply_task_action(action, actor_id):
            return
        action_type = action.get("type")
        payload = action.get("payload", {})
        if action_type == "decide" and isinstance(payload, dict):
            self._apply_decision_action(actor_id, payload)
        if action_type == "communicate" and isinstance(payload, dict):
            self._apply_communicate_action(actor_id, payload)
        if action_type == "propose" and isinstance(payload, dict):
            self._apply_propose_action(actor_id, payload)
        if action_type == "respond" and isinstance(payload, dict):
            self._apply_respond_action(actor_id, payload)
        if action_type == "transfer" and isinstance(payload, dict):
            self._apply_transfer_action(actor_id, payload)
        if action_type == "do_nothing":
            return

    def _queue_targeted_realtime_triggers(self, action: dict[str, Any], actor_id: str) -> None:
        if self._step_mode() != "time":
            return
        if not isinstance(action, dict):
            return
        action_type = action.get("type")
        payload = action.get("payload")
        if not isinstance(payload, dict):
            return
        if action_type == "communicate":
            # communicate already has dedicated trigger handling in _apply_communicate_action.
            return
        triggerable = {
            "propose_trade_offer",
            "trade_response",
            "cancel_trade_offer",
            "fulfill_order",
        }
        if action_type not in triggerable:
            return

        targets: set[str] = set()
        target_id = payload.get("target_id")
        if isinstance(target_id, str) and target_id and target_id != actor_id:
            targets.add(target_id)

        # Trade response/cancel typically only include transaction_id.
        if action_type in {"trade_response", "cancel_trade_offer"} and not targets:
            counterparty = self._resolve_trade_counterparty(payload, actor_id)
            if isinstance(counterparty, str) and counterparty and counterparty != actor_id:
                targets.add(counterparty)

        for target in sorted(targets):
            self._enqueue_realtime_trigger(target, phase_lock=None)

    def _apply_task_action(self, action: dict[str, Any], actor_id: str) -> bool:
        task_def = self.task_definition
        if task_def is None:
            return False
        apply_action = getattr(task_def, "apply_action", None)
        if not callable(apply_action):
            return False
        try:
            handled = apply_action(self.state, actor_id, action, self._emit_event)
        except Exception:
            return False
        return handled is True

    def _apply_decision_action(self, actor_id: str, payload: dict[str, Any]) -> None:
        decision_id = payload.get("decision_id")
        choice = payload.get("choice")
        reveal = payload.get("reveal")
        if not isinstance(decision_id, str):
            return
        buffer_store = self.state.buffers.setdefault("decisions", {})
        meta_store = self.state.buffers.setdefault("decision_meta", {})
        if not isinstance(buffer_store, dict):
            return
        if not isinstance(meta_store, dict):
            return
        entries = buffer_store.setdefault(decision_id, [])
        if isinstance(entries, list):
            entries.append({"actor_id": actor_id, "choice": choice})
        if reveal == "sequential":
            ordered = self._sorted_decision_choices(entries)
            self._emit_event(
                event_type="decision_revealed",
                actor_id=actor_id,
                visibility="public",
                payload={"decision_id": decision_id, "choices": ordered},
            )
            buffer_store.pop(decision_id, None)
            meta_store.pop(decision_id, None)
            return
        if decision_id not in meta_store:
            meta_store[decision_id] = {
                "created_step": self.state.step_index,
                "reveal": reveal,
            }
        if reveal in ("aggregated", "simultaneous") and self._should_reveal_decision(entries, reveal):
            ordered = self._sorted_decision_choices(entries)
            self._emit_event(
                event_type="decision_revealed",
                actor_id=actor_id,
                visibility="public",
                payload={"decision_id": decision_id, "choices": ordered},
            )
            buffer_store.pop(decision_id, None)
            meta_store.pop(decision_id, None)
            return
        self._emit_event(
            event_type="decision_buffered",
            actor_id=actor_id,
            visibility="system",
            payload={"decision_id": decision_id, "buffer_size": len(entries) if isinstance(entries, list) else 0},
        )

    def _apply_decision_timeouts(self) -> None:
        timeout_steps = self._decision_timeout_steps()
        if timeout_steps is None:
            return
        decisions = self.state.buffers.get("decisions")
        meta_store = self.state.buffers.get("decision_meta")
        if not isinstance(decisions, dict) or not isinstance(meta_store, dict):
            return
        expired: list[str] = []
        for decision_id, meta in meta_store.items():
            if not isinstance(meta, dict):
                continue
            created_step = meta.get("created_step")
            if not isinstance(created_step, int):
                continue
            if self.state.step_index >= created_step + timeout_steps:
                expired.append(decision_id)
        for decision_id in expired:
            entries = decisions.get(decision_id, [])
            self._emit_event(
                event_type="decision_timed_out",
                actor_id="system",
                visibility="system",
                payload={"decision_id": decision_id},
            )
            self._emit_event(
                event_type="decision_revealed",
                actor_id="system",
                visibility="public",
                payload={"decision_id": decision_id, "choices": self._sorted_decision_choices(entries)},
            )
            decisions.pop(decision_id, None)
            meta_store.pop(decision_id, None)

    def _apply_communicate_action(self, actor_id: str, payload: dict[str, Any]) -> None:
        channel = payload.get("channel")
        recipients = self._resolve_recipients(channel, payload, actor_id)
        message_id = self._next_message_id()
        visibility = "private" if channel == "direct" else "public"
        self._emit_event(
            event_type="message_delivered",
            actor_id=actor_id,
            visibility=visibility,
            payload={
                "message_id": message_id,
                "recipients": recipients,
                "channel": channel,
                "content": payload.get("content"),
                "content_type": payload.get("content_type"),
            },
        )
        self._increment_message_count(actor_id)
        daytrader_phase_lock: str | None = None
        if self._task_type() == "daytrader" and self._daytrader_phase() == "group_chat":
            # Keep message-triggered free chat within free-chat constraints,
            # even if wall-clock crosses into decision phase before the reply arrives.
            daytrader_phase_lock = "group_chat"
        for recipient in recipients:
            if recipient == actor_id and channel == "direct":
                continue
            if self._step_mode() == "time":
                # Real-time mode processes recipients strictly by message arrival order.
                self._enqueue_realtime_trigger(recipient, phase_lock=daytrader_phase_lock)

    def _apply_propose_action(self, actor_id: str, payload: dict[str, Any]) -> None:
        proposal_id = payload.get("proposal_id")
        if not isinstance(proposal_id, str):
            return
        buffer_store = self.state.buffers.setdefault("proposals", {})
        if not isinstance(buffer_store, dict):
            return
        expires_at = payload.get("expires_at")
        if not isinstance(expires_at, int):
            expires_at = self._default_proposal_expiry()
        buffer_store[proposal_id] = {
            "actor_id": actor_id,
            "target_ids": payload.get("target_ids", []),
            "terms": payload.get("terms"),
            "expires_at": expires_at,
            "responses": [],
        }
        self._emit_event(
            event_type="proposal_created",
            actor_id=actor_id,
            visibility="public",
            payload={
                "proposal_id": proposal_id,
                "target_ids": payload.get("target_ids", []),
                "terms": payload.get("terms"),
                "expires_at": expires_at,
            },
        )

    def _apply_respond_action(self, actor_id: str, payload: dict[str, Any]) -> None:
        proposal_id = payload.get("proposal_id")
        if not isinstance(proposal_id, str):
            return
        buffer_store = self.state.buffers.setdefault("proposals", {})
        if not isinstance(buffer_store, dict):
            return
        entry = buffer_store.get(proposal_id)
        if not isinstance(entry, dict):
            return
        responses = entry.setdefault("responses", [])
        if isinstance(responses, list):
            responses.append(
                {
                    "actor_id": actor_id,
                    "response": payload.get("response"),
                    "counter_terms": payload.get("counter_terms"),
                }
            )
        self._emit_event(
            event_type="proposal_responded",
            actor_id=actor_id,
            visibility="public",
            payload={
                "proposal_id": proposal_id,
                "response": payload.get("response"),
                "counter_terms": payload.get("counter_terms"),
            },
        )
        if payload.get("response") in ("accept", "reject"):
            buffer_store.pop(proposal_id, None)

    def _apply_transfer_action(self, actor_id: str, payload: dict[str, Any]) -> None:
        resource_id = payload.get("resource_id")
        target_id = payload.get("to")
        amount = payload.get("amount")
        if not isinstance(resource_id, str) or not isinstance(target_id, str):
            return
        if not isinstance(amount, (int, float)):
            return
        resource_store = self.state.resources.setdefault(resource_id, {})
        if not isinstance(resource_store, dict):
            return
        resource_store[actor_id] = float(resource_store.get(actor_id, 0.0)) - float(amount)
        resource_store[target_id] = float(resource_store.get(target_id, 0.0)) + float(amount)
        self._emit_event(
            event_type="resource_transferred",
            actor_id=actor_id,
            visibility="public",
            payload={
                "resource_id": resource_id,
                "from": actor_id,
                "to": target_id,
                "amount": float(amount),
                "from_balance": resource_store.get(actor_id),
                "to_balance": resource_store.get(target_id),
            },
        )

    def _default_proposal_expiry(self) -> int | None:
        if self.config is None:
            return None
        protocol = self.config.get("protocol", {})
        if not isinstance(protocol, dict):
            return None
        expiry_steps = protocol.get("proposal_expiry_steps")
        if not isinstance(expiry_steps, int) or expiry_steps <= 0:
            return None
        return self.state.step_index + expiry_steps

    def _decision_timeout_steps(self) -> int | None:
        if self.config is None:
            return None
        protocol = self.config.get("protocol", {})
        if not isinstance(protocol, dict):
            return None
        timeout = protocol.get("decision_timeout_steps")
        if not isinstance(timeout, int) or timeout <= 0:
            return None
        return timeout

    def _resolve_recipients(self, channel: Any, payload: dict[str, Any], actor_id: str) -> list[str]:
        if channel == "direct":
            raw = payload.get("recipients")
            if isinstance(raw, list):
                return [item for item in raw if isinstance(item, str) and item]
            return []
        if channel == "broadcast":
            # Broadcast can target multiple selected recipients, or all others when recipients omitted.
            raw = payload.get("recipients")
            if isinstance(raw, list) and raw:
                seen: set[str] = set()
                selected: list[str] = []
                for item in raw:
                    if not isinstance(item, str) or not item or item == actor_id or item in seen:
                        continue
                    seen.add(item)
                    selected.append(item)
                return selected
            if isinstance(self.state.agents, dict):
                return [
                    agent_id
                    for agent_id in self.state.agents.keys()
                    if isinstance(agent_id, str) and agent_id and agent_id != actor_id
                ]
            return []
        if isinstance(self.state.agents, dict):
            return [agent_id for agent_id in self.state.agents.keys() if isinstance(agent_id, str) and agent_id]
        return []

    def _log_observation_events(self) -> bool:
        if not isinstance(self.config, dict):
            return False
        logging_cfg = self.config.get("logging", {})
        if not isinstance(logging_cfg, dict):
            return False
        return logging_cfg.get("observation_events") is True

    def _build_agent_input_context(self, agent_id: str) -> dict[str, Any]:
        prompt_cfg = {}
        if isinstance(self.config, dict):
            prompt_cfg = self.config.get("prompts", {})
        if not isinstance(prompt_cfg, dict):
            prompt_cfg = {}
        task_path = prompt_cfg.get("task", "prompts/task_instructions.md")
        return_format_path = prompt_cfg.get("return_format", "prompts/return_format.md")
        task_instructions = load_text_file(task_path)
        return_format = load_text_file(return_format_path)
        action_space = self.config.get("action_space", {}) if isinstance(self.config, dict) else {}
        enabled = action_space.get("enabled", []) if isinstance(action_space, dict) else []
        allowed_actions = [item for item in enabled if isinstance(item, str)]
        if "do_nothing" not in allowed_actions:
            allowed_actions.append("do_nothing")
        if self._task_type() == "hidden_profile":
            phase = self._hidden_profile_phase()
            if phase in {"initial", "final"}:
                allowed_actions = [item for item in allowed_actions if item in {"decide", "do_nothing"}]
                if "decide" not in allowed_actions:
                    allowed_actions.insert(0, "decide")
            elif phase == "discussion":
                allowed_actions = [item for item in allowed_actions if item in {"communicate", "do_nothing"}]
                if "communicate" not in allowed_actions:
                    allowed_actions.insert(0, "communicate")
        persona_profile = None
        agent = self.state.agents.get(agent_id) if isinstance(self.state.agents, dict) else None
        if agent is not None:
            persona_profile = getattr(agent, "persona_profile", None)
        decide_cfg = action_space.get("decide", {}) if isinstance(action_space, dict) else {}
        decide_reveal = decide_cfg.get("reveal") if isinstance(decide_cfg, dict) else None
        if "{decide_reveal}" in return_format:
            reveal_value = decide_reveal if isinstance(decide_reveal, str) else "aggregated"
            return_format = return_format.replace("{decide_reveal}", reveal_value)
        protocol = self.config.get("protocol", {}) if isinstance(self.config, dict) else {}
        elapsed_time_sec = None
        remaining_time_sec = None
        if self._wall_start_monotonic is not None and self._wall_duration_sec is not None:
            elapsed_time_sec = max(0.0, time.monotonic() - self._wall_start_monotonic)
            remaining_time_sec = max(0.0, self._wall_duration_sec - elapsed_time_sec)
        return {
            "persona_profile": persona_profile,
            "task_instructions": task_instructions,
            "protocol": protocol,
            "allowed_actions": allowed_actions,
            "return_format": return_format,
            "elapsed_time_sec": elapsed_time_sec,
            "remaining_time_sec": remaining_time_sec,
        }

    def _apply_action_defaults(self, action: dict[str, Any], actor_id: str | None = None) -> dict[str, Any]:
        payload = action.get("payload")
        if not isinstance(payload, dict):
            return action
        action_type = action.get("type")
        if action_type == "decide":
            if payload.get("reveal") is not None:
                return action
            default_reveal = self._default_decide_reveal()
            if default_reveal is None:
                return action
            return {**action, "payload": {**payload, "reveal": default_reveal}}
        if action_type == "communicate":
            channel = payload.get("channel")
            if channel != "broadcast":
                return action
            recipients = payload.get("recipients")
            if recipients is not None:
                return action
            if not isinstance(actor_id, str) or not actor_id:
                return action
            resolved = self._resolve_recipients("broadcast", payload, actor_id)
            if not resolved:
                return action
            return {**action, "payload": {**payload, "recipients": resolved}}
        return action

    def _default_decide_reveal(self) -> str | None:
        if self.config is None:
            return None
        action_space = self.config.get("action_space", {})
        if not isinstance(action_space, dict):
            return None
        decide_cfg = action_space.get("decide", {})
        if not isinstance(decide_cfg, dict):
            return None
        reveal = decide_cfg.get("reveal")
        if isinstance(reveal, str) and reveal:
            return reveal
        return None

    def _check_action_preconditions(self, action: dict[str, Any], actor_id: str) -> str | None:
        action_type = action.get("type")
        payload = action.get("payload")
        daytrader_error = self._check_daytrader_round_preconditions(action)
        if daytrader_error is not None:
            return daytrader_error
        hidden_profile_error = self._check_hidden_profile_phase_preconditions(action, actor_id)
        if hidden_profile_error is not None:
            return hidden_profile_error
        if action_type != "communicate" or not isinstance(payload, dict):
            return None
        mode = self._communication_mode()
        channel = payload.get("channel")
        if mode == "direct" and channel == "broadcast":
            return "Broadcast messages not allowed in direct-only mode."
        if channel == "direct":
            recipients = payload.get("recipients")
            if not isinstance(recipients, list) or not recipients:
                return "Direct communication requires a non-empty recipients list."
            if not all(isinstance(item, str) and item for item in recipients):
                return "communicate.recipients must contain non-empty participant ids."
            if actor_id in recipients:
                return "You cannot send a direct message to yourself."
            if len(set(recipients)) != len(recipients):
                return "communicate.recipients must not contain duplicates."
        if channel == "broadcast":
            recipients = payload.get("recipients")
            if recipients is not None:
                if not isinstance(recipients, list) or not recipients:
                    return "broadcast recipients must be a non-empty list when provided."
                if not all(isinstance(item, str) and item for item in recipients):
                    return "broadcast recipients must contain non-empty participant ids."
                if actor_id in recipients:
                    return "broadcast recipients must not include yourself."
                if len(set(recipients)) != len(recipients):
                    return "broadcast recipients must not contain duplicates."
        if self._step_mode() != "time":
            limit = self._max_messages_per_turn()
            if limit is not None and self._message_count(actor_id) >= limit:
                return "Communication limit reached for this turn."
        return None

    def _check_daytrader_round_preconditions(self, action: dict[str, Any]) -> str | None:
        if self._task_type() != "daytrader":
            return None
        action_type = action.get("type")
        phase = self._daytrader_phase_for_action(action)
        if phase == "decision":
            allowed = {"make_investment", "do_nothing"}
            if action_type in allowed:
                return None
            allowed_text = ", ".join(sorted(allowed))
            return f"DayTrader decision phase only allows: {allowed_text}."
        allowed = {"communicate", "do_nothing"}
        if action_type not in allowed:
            allowed_text = ", ".join(sorted(allowed))
            return f"DayTrader group_chat phase only allows: {allowed_text}."
        return None

    def _sync_daytrader_round_phase(self) -> None:
        if self._task_type() != "daytrader":
            return
        task_state = self.state.task_state
        if not isinstance(task_state, dict):
            return
        round_index = task_state.get("round_index")
        if not isinstance(round_index, int) or round_index <= 0:
            round_index = 1
        phase = task_state.get("phase")
        if phase not in {"decision", "group_chat"}:
            phase = "decision"
        task_state["round_index"] = round_index
        task_state["phase"] = phase
        task_state["phase_step_in_round"] = 1 if phase == "decision" else 2
        participant_ids = self._daytrader_agent_ids()
        if "decision_pending_agents" not in task_state or not isinstance(task_state.get("decision_pending_agents"), list):
            task_state["decision_pending_agents"] = participant_ids
        if "group_chat_turns" not in task_state or not isinstance(task_state.get("group_chat_turns"), int):
            task_state["group_chat_turns"] = 0
        if "group_chat_silent_agents" not in task_state or not isinstance(task_state.get("group_chat_silent_agents"), list):
            task_state["group_chat_silent_agents"] = []

    def _daytrader_phase_for_action(self, action: dict[str, Any]) -> str:
        phase_lock = action.get("_daytrader_phase_lock")
        if phase_lock in {"decision", "group_chat"}:
            return phase_lock
        self._sync_daytrader_round_phase()
        return self._daytrader_phase()

    def _daytrader_phase(self) -> str:
        if self._task_type() != "daytrader":
            return "decision"
        task_state = self.state.task_state
        if isinstance(task_state, dict):
            phase = task_state.get("phase")
            if phase in {"decision", "group_chat"}:
                return phase
        return "decision"

    def _daytrader_group_chat_bootstrap_targets(self, agent_ids: list[str]) -> list[str]:
        ordered = [agent_id for agent_id in sorted(agent_ids) if isinstance(agent_id, str) and agent_id]
        if len(ordered) <= 2:
            return ordered
        return ordered[:2]

    def _enqueue_realtime_trigger(self, agent_id: str, phase_lock: str | None = None) -> None:
        if not isinstance(agent_id, str) or not agent_id:
            return
        normalized_phase_lock = phase_lock if phase_lock in {"decision", "group_chat"} else None
        self._pending_realtime_message_queue = [
            entry
            for entry in self._pending_realtime_message_queue
            if not (isinstance(entry, dict) and entry.get("agent_id") == agent_id)
        ]
        self._pending_realtime_message_queue.append(
            {"agent_id": agent_id, "daytrader_phase_lock": normalized_phase_lock}
        )

    def _ensure_daytrader_freechat_bootstrap(self, validated_actions: dict[str, dict[str, Any]]) -> None:
        if self._task_type() != "daytrader":
            return
        self._sync_daytrader_round_phase()
        task_state = self.state.task_state
        if not isinstance(task_state, dict):
            return
        phase = task_state.get("phase")
        if phase == "decision":
            participant_ids = self._daytrader_agent_ids()
            if not participant_ids:
                return
            pending = task_state.get("decision_pending_agents")
            pending_count = len(pending) if isinstance(pending, list) else len(participant_ids)
            if pending_count > 0:
                return
            task_state["phase"] = "group_chat"
            task_state["phase_step_in_round"] = 2
            task_state["group_chat_turns"] = 0
            task_state["group_chat_silent_agents"] = []
            self._pending_realtime_message_queue = []
        if task_state.get("phase") != "group_chat":
            return
        turns = task_state.get("group_chat_turns", 0)
        if not isinstance(turns, int):
            turns = 0
        if turns > 0:
            return
        if self._pending_realtime_message_queue:
            return
        bootstrap_targets = self._daytrader_group_chat_bootstrap_targets(self._daytrader_agent_ids())
        for target in bootstrap_targets:
            self._enqueue_realtime_trigger(target, phase_lock="group_chat")

    def _daytrader_agent_ids(self) -> list[str]:
        task_state = self.state.task_state
        if isinstance(task_state, dict):
            participants = task_state.get("participants")
            if isinstance(participants, dict):
                return sorted(agent_id for agent_id in participants.keys() if isinstance(agent_id, str) and agent_id)
        if isinstance(self.state.agents, dict):
            return sorted(agent_id for agent_id in self.state.agents.keys() if isinstance(agent_id, str) and agent_id)
        return []

    def _daytrader_group_chat_max_turns(self) -> int:
        if not isinstance(self.config, dict):
            return 6
        task_cfg = self.config.get("task", {})
        if not isinstance(task_cfg, dict):
            return 6
        phase_rules = task_cfg.get("phase_rules", {})
        if not isinstance(phase_rules, dict):
            return 6
        value = phase_rules.get("group_chat_max_turns")
        if isinstance(value, int) and value > 0:
            return value
        return 6

    def _advance_daytrader_phase_from_action(self, actor_id: str, action: dict[str, Any]) -> None:
        if self._task_type() != "daytrader":
            return
        task_state = self.state.task_state
        if not isinstance(task_state, dict):
            return
        self._sync_daytrader_round_phase()
        phase = self._daytrader_phase_for_action(action)
        agent_ids = self._daytrader_agent_ids()
        if not agent_ids:
            return
        if phase == "decision":
            pending_raw = task_state.get("decision_pending_agents")
            pending: set[str] = {
                item for item in pending_raw if isinstance(item, str) and item
            } if isinstance(pending_raw, list) else set(agent_ids)
            pending.discard(actor_id)
            task_state["decision_pending_agents"] = sorted(pending)
            if pending:
                return
            task_state["phase"] = "group_chat"
            task_state["phase_step_in_round"] = 2
            task_state["group_chat_turns"] = 0
            task_state["group_chat_silent_agents"] = []
            self._pending_realtime_message_queue = []
            bootstrap_targets = self._daytrader_group_chat_bootstrap_targets(agent_ids)
            self._pending_realtime_message_queue.extend(
                {"agent_id": target, "daytrader_phase_lock": "group_chat"} for target in bootstrap_targets
            )
            return

        turns = task_state.get("group_chat_turns", 0)
        if not isinstance(turns, int):
            turns = 0
        turns += 1
        task_state["group_chat_turns"] = turns
        action_type = action.get("type")
        silent_raw = task_state.get("group_chat_silent_agents")
        silent_agents: set[str] = {
            item for item in silent_raw if isinstance(item, str) and item
        } if isinstance(silent_raw, list) else set()
        if action_type == "do_nothing":
            if isinstance(actor_id, str) and actor_id:
                silent_agents.add(actor_id)
        else:
            silent_agents.clear()
        task_state["group_chat_silent_agents"] = sorted(silent_agents)
        if len(silent_agents) < len(agent_ids) and turns < self._daytrader_group_chat_max_turns():
            return
        round_index = task_state.get("round_index", 1)
        if not isinstance(round_index, int) or round_index <= 0:
            round_index = 1
        round_index += 1
        task_state["round_index"] = round_index
        task_state["phase"] = "decision"
        task_state["phase_step_in_round"] = 1
        task_state["decision_pending_agents"] = list(agent_ids)
        task_state["group_chat_turns"] = 0
        task_state["group_chat_silent_agents"] = []
        self._pending_realtime_message_queue = [
            entry
            for entry in self._pending_realtime_message_queue
            if isinstance(entry, dict) and entry.get("daytrader_phase_lock") != "group_chat"
        ]

    def _daytrader_runtime_fields(self) -> dict[str, Any]:
        if self._task_type() != "daytrader":
            return {}
        task_state = self.state.task_state
        if not isinstance(task_state, dict):
            return {}
        round_index = task_state.get("round_index")
        phase = task_state.get("phase")
        if not isinstance(round_index, int) or round_index <= 0:
            return {}
        if phase not in {"decision", "group_chat"}:
            phase = "decision"
        return {
            "daytrader_round_index": round_index,
            "daytrader_phase": phase,
        }

    def _daytrader_round_phase(self, step_index: int) -> tuple[int, str]:
        if step_index <= 0:
            step_index = 1
        round_index = ((step_index - 1) // 2) + 1
        phase = "decision" if step_index % 2 == 1 else "group_chat"
        return round_index, phase

    def _check_hidden_profile_phase_preconditions(self, action: dict[str, Any], actor_id: str) -> str | None:
        if self._task_type() != "hidden_profile":
            return None
        action_type = action.get("type")
        payload = action.get("payload")
        phase = self._hidden_profile_phase()
        participant = self._hidden_profile_participant_state(actor_id)
        phase_rules = self._hidden_profile_phase_rules()
        initial_decision_id = phase_rules.get("initial_vote_decision_id", "initial_vote")
        final_decision_id = phase_rules.get("final_vote_decision_id", "final_vote")
        if not isinstance(initial_decision_id, str) or not initial_decision_id:
            initial_decision_id = "initial_vote"
        if not isinstance(final_decision_id, str) or not final_decision_id:
            final_decision_id = "final_vote"
        discussion_actions_raw = phase_rules.get("discussion_action_types")
        discussion_actions: set[str] = {"communicate"}
        if isinstance(discussion_actions_raw, list):
            configured = {item for item in discussion_actions_raw if isinstance(item, str) and item}
            if configured:
                discussion_actions = configured

        if action_type == "communicate":
            if phase != "discussion":
                return "Communication is only allowed during hidden_profile discussion phase."
            if "communicate" not in discussion_actions:
                return "Communication is disabled by hidden_profile phase_rules.discussion_action_types."
            return None

        if action_type == "do_nothing":
            if phase == "discussion":
                return None
            if phase in {"initial", "final"}:
                return None

        if phase == "discussion":
            if action_type in discussion_actions:
                return None
            allowed = ", ".join(sorted(set(discussion_actions) | {"do_nothing"}))
            return f"Discussion phase only allows these actions: {allowed}."

        if action_type != "decide" or not isinstance(payload, dict):
            return "Voting phases only allow decide or do_nothing actions in hidden_profile."

        decision_id = payload.get("decision_id")
        if not isinstance(decision_id, str) or not decision_id:
            return "hidden_profile voting requires decide.decision_id."
        choice = payload.get("choice")
        if not isinstance(choice, str) or not choice:
            return "hidden_profile voting requires a non-empty decide.choice."

        if phase == "initial":
            if decision_id != initial_decision_id:
                return f"Initial phase only allows decide.decision_id = {initial_decision_id}."
            if isinstance(participant, dict) and isinstance(participant.get("initial_vote"), str):
                return "Initial vote already submitted for this participant."
            return None

        if phase == "final":
            if decision_id != final_decision_id:
                return f"Final phase only allows decide.decision_id = {final_decision_id}."
            if isinstance(participant, dict):
                initial_vote = participant.get("initial_vote")
                if not isinstance(initial_vote, str) or not initial_vote:
                    return "Final vote requires an existing initial vote first."
                if isinstance(participant.get("final_vote"), str):
                    return "Final vote already submitted for this participant."
            return None

        return "Discussion phase only allows configured discussion actions."

    def _task_type(self) -> str | None:
        if not isinstance(self.config, dict):
            return None
        task_cfg = self.config.get("task", {})
        if not isinstance(task_cfg, dict):
            return None
        task_type = task_cfg.get("type")
        if isinstance(task_type, str) and task_type:
            return task_type
        return None

    def _hidden_profile_phase(self) -> str:
        task_state = self.state.task_state
        if not isinstance(task_state, dict):
            return "discussion"
        participants = task_state.get("participants")
        if not isinstance(participants, dict) or not participants:
            task_state["phase"] = "discussion"
            return "discussion"

        if not all(
            isinstance(item, dict) and isinstance(item.get("initial_vote"), str) and item.get("initial_vote")
            for item in participants.values()
        ):
            task_state["phase"] = "initial"
            return "initial"

        if self._step_mode() != "time":
            target_steps = task_state.get("target_steps")
            if not isinstance(target_steps, int) or target_steps <= 1:
                task_state["phase"] = "discussion"
                return "discussion"
            if self.state.step_index >= target_steps:
                task_state["phase"] = "final"
                return "final"
            task_state["phase"] = "discussion"
            return "discussion"

        elapsed_sec = self._elapsed_time_seconds()
        discussion_started_at_sec = task_state.get("discussion_started_at_sec")
        if not isinstance(discussion_started_at_sec, (int, float)):
            discussion_started_at_sec = elapsed_sec
            task_state["discussion_started_at_sec"] = discussion_started_at_sec

        duration_sec = self._hidden_profile_discussion_duration_sec()
        if elapsed_sec >= float(discussion_started_at_sec) + duration_sec:
            task_state["phase"] = "final"
            return "final"
        task_state["phase"] = "discussion"
        return "discussion"

    def _hidden_profile_participant_state(self, actor_id: str) -> dict[str, Any] | None:
        task_state = self.state.task_state
        if not isinstance(task_state, dict):
            return None
        participants = task_state.get("participants")
        if not isinstance(participants, dict):
            return None
        entry = participants.get(actor_id)
        return entry if isinstance(entry, dict) else None

    def _hidden_profile_phase_rules(self) -> dict[str, Any]:
        task_state = self.state.task_state
        if not isinstance(task_state, dict):
            return {}
        phase_rules = task_state.get("phase_rules")
        if not isinstance(phase_rules, dict):
            return {}
        return phase_rules

    def _hidden_profile_discussion_duration_sec(self) -> float:
        phase_rules = self._hidden_profile_phase_rules()
        duration_sec = phase_rules.get("discussion_duration_sec")
        if isinstance(duration_sec, (int, float)) and float(duration_sec) >= 0:
            return float(duration_sec)
        return 60.0

    def _elapsed_time_seconds(self) -> float:
        if self._wall_start_monotonic is None:
            return float(self.state.sim_time_ms) / 1000.0
        return max(0.0, time.monotonic() - self._wall_start_monotonic)

    def _communication_mode(self) -> str | None:
        if self.config is None:
            return None
        controls = self.config.get("controls", {})
        if not isinstance(controls, dict):
            return None
        communication = controls.get("communication", {})
        if not isinstance(communication, dict):
            return None
        mode = communication.get("mode")
        if isinstance(mode, str) and mode:
            return mode
        return None

    def _max_messages_per_turn(self) -> int | None:
        if self.config is None:
            return None
        controls = self.config.get("controls", {})
        if not isinstance(controls, dict):
            return None
        communication = controls.get("communication", {})
        if not isinstance(communication, dict):
            return None
        limit = communication.get("max_messages_per_turn")
        if isinstance(limit, int) and limit > 0:
            return limit
        return None

    def _message_count(self, actor_id: str) -> int:
        counts = self.state.turn_state.get("message_counts")
        if not isinstance(counts, dict):
            return 0
        count = counts.get(actor_id, 0)
        return count if isinstance(count, int) else 0

    def _increment_message_count(self, actor_id: str) -> None:
        counts = self.state.turn_state.setdefault("message_counts", {})
        if not isinstance(counts, dict):
            self.state.turn_state["message_counts"] = {actor_id: 1}
            return
        counts[actor_id] = self._message_count(actor_id) + 1

    def _should_reveal_decision(self, entries: Any, reveal: Any) -> bool:
        if not isinstance(entries, list):
            return False
        if not isinstance(self.state.agents, dict) or not self.state.agents:
            return False
        if reveal == "aggregated":
            quorum = self._decision_quorum()
            return len(entries) >= quorum
        return len(entries) >= len(self.state.agents)

    def _sorted_decision_choices(self, entries: Any) -> list[dict[str, Any]]:
        if not isinstance(entries, list):
            return []
        def _key(item: dict[str, Any]) -> str:
            actor_id = item.get("actor_id") if isinstance(item, dict) else None
            return actor_id if isinstance(actor_id, str) else ""
        return sorted((item for item in entries if isinstance(item, dict)), key=_key)

    def _decision_quorum(self) -> int:
        if self.config is None:
            return len(self.state.agents)
        protocol = self.config.get("protocol", {})
        if not isinstance(protocol, dict):
            return len(self.state.agents)
        quorum = protocol.get("decision_quorum")
        if isinstance(quorum, int) and quorum > 0:
            return min(quorum, len(self.state.agents))
        return len(self.state.agents)

    def _is_action_enabled(self, action: Any) -> bool:
        if not isinstance(action, dict):
            return False
        action_type = action.get("type")
        if not isinstance(action_type, str):
            return False
        if action_type == "do_nothing":
            return True
        if self.config is None:
            return True
        action_space = self.config.get("action_space", {})
        if not isinstance(action_space, dict):
            return True
        enabled = action_space.get("enabled", [])
        if isinstance(enabled, list) and enabled:
            return action_type in enabled
        return True

    def _extract_actor_id(self, action: Any) -> str:
        if isinstance(action, dict):
            actor_id = action.get("actor_id")
            if isinstance(actor_id, str) and actor_id:
                return actor_id
        return "unknown"

    def _emit_event(
        self,
        event_type: str,
        actor_id: str,
        visibility: str,
        payload: dict[str, Any],
        event_id: str | None = None,
    ) -> dict[str, Any]:
        event = {
            "event_id": event_id or self._next_event_id(),
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actor_id": actor_id,
            "visibility": visibility,
            "payload": payload,
        }
        try:
            validate_event(event)
        except EventValidationError as exc:
            raise ValueError(f"Invalid event emitted: {exc}") from exc
        self.state.event_log.append(event)
        listener = self.event_listener
        if callable(listener):
            try:
                listener(event)
            except Exception:
                pass
        return event

    def _notify_runtime(self, name: str, payload: dict[str, Any]) -> None:
        listener = self.runtime_listener
        if callable(listener):
            try:
                listener(name, payload)
            except Exception:
                pass

    def _next_event_id(self) -> str:
        self._event_counter += 1
        return f"event_{self._event_counter}"

    def _next_message_id(self) -> str:
        self._message_counter += 1
        return f"msg_{self._message_counter}"

    def _flush_logs(self) -> None:
        if self.run_paths is None:
            return
        new_events = self.state.event_log[self._last_flushed_index :]
        if new_events:
            # For runs that have already produced some events, reload the full
            # in-memory log and write it as a single pretty-printed JSON array.
            # This keeps the on-disk format as JSON (not JSONL) with indentation.
            from src.data.logging import write_events_json

            write_events_json(self.run_paths.events_path, self.state.event_log)
            self._last_flushed_index = len(self.state.event_log)

    def _export_metrics(self) -> None:
        if self.run_paths is None:
            return
        task_outcome = self._build_task_outcome()
        result = compute_metrics(
            event_log=self.state.event_log,
            probe_log=self.state.probe_log,
            task_outcome=task_outcome,
        )
        write_metrics(self.run_paths, {"per_agent": result.per_agent, "per_run": result.per_run})
        if isinstance(task_outcome, dict) and task_outcome.get("task_type") == "maptask":
            score_map_text = task_outcome.get("maptask_score_map_text")
            if isinstance(score_map_text, str) and score_map_text:
                score_map_path = self.run_paths.run_dir / "maptask_score_map.txt"
                score_map_path.write_text(score_map_text, encoding="utf-8")
            follower_map_text = task_outcome.get("maptask_follower_map_text")
            if isinstance(follower_map_text, str) and follower_map_text:
                follower_map_path = self.run_paths.run_dir / "maptask_follower_working_map.txt"
                follower_map_path.write_text(follower_map_text, encoding="utf-8")

    def _write_manifest(self) -> None:
        if self.run_paths is None or self.config is None or self.run_id is None:
            return
        manifest = build_run_manifest(
            self.config,
            run_id=self.run_id,
            probe_registry=self.probe_registry,
        )
        write_run_manifest(self.run_paths, manifest)

    def _build_task_outcome(self) -> dict[str, Any] | None:
        task_cfg = self.config.get("task") if self.config is not None else None
        task_type = task_cfg.get("type") if isinstance(task_cfg, dict) else None
        if not isinstance(task_type, str):
            return None
        outcome: dict[str, Any] = {
            "task_type": task_type,
            "complete": self.state.task_state.get("complete"),
        }
        if task_type == "counter":
            for key in ("steps_taken", "target_steps"):
                if key in self.state.task_state:
                    outcome[key] = self.state.task_state.get(key)
        if task_type == "accumulator":
            for key in ("steps_taken", "target_value", "current_value", "increment"):
                if key in self.state.task_state:
                    outcome[key] = self.state.task_state.get(key)
        if task_type == "hidden_profile":
            for key in ("steps_taken", "target_steps"):
                if key in self.state.task_state:
                    outcome[key] = self.state.task_state.get(key)
            participants = self.state.task_state.get("participants")
            if isinstance(participants, dict):
                initial_votes: dict[str, str] = {}
                final_votes: dict[str, str] = {}
                for agent_id, entry in participants.items():
                    if not isinstance(agent_id, str) or not isinstance(entry, dict):
                        continue
                    initial_vote = entry.get("initial_vote")
                    final_vote = entry.get("final_vote")
                    if isinstance(initial_vote, str) and initial_vote:
                        initial_votes[agent_id] = initial_vote
                    if isinstance(final_vote, str) and final_vote:
                        final_votes[agent_id] = final_vote
                if initial_votes:
                    outcome["initial_votes"] = initial_votes
                if final_votes:
                    outcome["final_votes"] = final_votes
        if task_type == "shapefactory":
            for key in ("steps_taken", "target_steps"):
                if key in self.state.task_state:
                    outcome[key] = self.state.task_state.get(key)
            pending_offers = self.state.task_state.get("pending_offers")
            completed_trades = self.state.task_state.get("completed_trades")
            if isinstance(pending_offers, list):
                outcome["pending_offers"] = len(pending_offers)
            if isinstance(completed_trades, list):
                outcome["completed_trades"] = len(completed_trades)
        if task_type == "daytrader":
            for key in ("steps_taken", "target_rounds", "rounds_completed", "target_steps"):
                if key in self.state.task_state:
                    outcome[key] = self.state.task_state.get(key)
            participants = self.state.task_state.get("participants")
            if isinstance(participants, dict):
                investments_count = 0
                for participant in participants.values():
                    if not isinstance(participant, dict):
                        continue
                    history = participant.get("investment_history")
                    if isinstance(history, list):
                        investments_count += len(history)
                outcome["investments_count"] = investments_count
        if task_type == "maptask":
            for key in ("steps_taken", "target_steps"):
                if key in self.state.task_state:
                    outcome[key] = self.state.task_state.get(key)
            participants = self.state.task_state.get("participants")
            if isinstance(participants, dict):
                updates = 0
                for participant in participants.values():
                    if not isinstance(participant, dict):
                        continue
                    progress = participant.get("map_progress")
                    if isinstance(progress, dict):
                        updates += len(progress)
                outcome["map_progress_updates"] = updates
                route_score = self._maptask_route_similarity(participants)
                if route_score is not None:
                    outcome.update(route_score)
        return outcome

    def _maptask_route_similarity(self, participants: dict[str, Any]) -> dict[str, Any] | None:
        guider_map_text: str | None = None
        follower_map_text: str | None = None
        follower_points: set[tuple[int, int]] = set()
        anchor_points: set[tuple[int, int]] = set()

        for participant in participants.values():
            if not isinstance(participant, dict):
                continue
            role = participant.get("role")
            map_info = participant.get("map")
            if isinstance(map_info, dict):
                special_points = map_info.get("special_points")
                if isinstance(special_points, dict):
                    for key in ("start", "finish"):
                        entry = special_points.get(key)
                        if not isinstance(entry, dict):
                            continue
                        point = self._coerce_cell_point(entry.get("cell"))
                        if point is not None:
                            anchor_points.add(point)
                if role == "guider":
                    text = map_info.get("map_text")
                    if isinstance(text, str) and text:
                        guider_map_text = text
            if role == "follower":
                working_text = participant.get("map_working_text")
                if isinstance(working_text, str) and working_text:
                    follower_map_text = working_text
                raw_points = participant.get("drawn_route_points")
                if isinstance(raw_points, list):
                    for item in raw_points:
                        point = self._coerce_cell_point(item)
                        if point is not None:
                            follower_points.add(point)

        if guider_map_text is None:
            return None
        gt_points = self._extract_route_points_from_text(guider_map_text).union(anchor_points)
        if not gt_points:
            return None

        score_map = self._build_maptask_score_map(guider_map_text, gt_points)
        total_score = 0
        for point in follower_points:
            total_score += score_map.get(point, 0)
        max_score = len(gt_points) * 3
        similarity = float(total_score) / float(max_score) if max_score > 0 else 0.0
        if similarity > 1.0:
            similarity = 1.0
        return {
            "maptask_route_score": float(total_score),
            "maptask_route_score_max": float(max_score),
            "maptask_route_similarity": similarity,
            "maptask_gt_points": float(len(gt_points)),
            "maptask_follower_points": float(len(follower_points)),
            "maptask_score_map_text": self._render_maptask_score_map(guider_map_text, score_map),
            "maptask_follower_map_text": follower_map_text,
        }

    def _extract_route_points_from_text(self, map_text: str) -> set[tuple[int, int]]:
        points: set[tuple[int, int]] = set()
        for row_idx, line in enumerate(map_text.splitlines()):
            for col_idx, cell in enumerate(line):
                if cell == ".":
                    points.add((row_idx, col_idx))
        return points

    def _coerce_cell_point(self, value: Any) -> tuple[int, int] | None:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        row, col = value
        if not isinstance(row, int) or not isinstance(col, int):
            return None
        return (row, col)

    def _build_maptask_score_map(
        self,
        map_text: str,
        gt_points: set[tuple[int, int]],
    ) -> dict[tuple[int, int], int]:
        lines = map_text.splitlines()
        rows = len(lines)
        cols = max((len(line) for line in lines), default=0)
        score_map: dict[tuple[int, int], int] = {}
        for row in range(rows):
            for col in range(cols):
                point = (row, col)
                if point in gt_points:
                    score_map[point] = 3
                    continue
                best_distance = self._nearest_chebyshev_distance(point, gt_points)
                if best_distance is None:
                    continue
                if best_distance == 1:
                    score_map[point] = 2
                elif best_distance == 2:
                    score_map[point] = 1
        return score_map

    def _nearest_chebyshev_distance(
        self,
        point: tuple[int, int],
        gt_points: set[tuple[int, int]],
    ) -> int | None:
        if not gt_points:
            return None
        row, col = point
        best: int | None = None
        for gt_row, gt_col in gt_points:
            distance = max(abs(row - gt_row), abs(col - gt_col))
            if best is None or distance < best:
                best = distance
                if best == 0:
                    return 0
        return best

    def _render_maptask_score_map(
        self,
        map_text: str,
        score_map: dict[tuple[int, int], int],
    ) -> str:
        lines = map_text.splitlines()
        rows = len(lines)
        cols = max((len(line) for line in lines), default=0)
        rendered: list[str] = []
        for row in range(rows):
            chars: list[str] = []
            for col in range(cols):
                score = score_map.get((row, col), 0)
                if score <= 0:
                    chars.append(" ")
                else:
                    chars.append(str(score))
            rendered.append("".join(chars))
        return "\n".join(rendered)
