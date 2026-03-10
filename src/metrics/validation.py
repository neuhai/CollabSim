"""Validation helpers for synthetic/adversarial metric checks."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Sequence

from src.metrics.metrics import MetricsResult, compute_metrics


@dataclass(frozen=True)
class MetricScenario:
    """Scenario describing inputs for metric validation."""

    name: str
    task_outcome: dict[str, Any] | None
    event_log: list[dict[str, Any]] | None = None
    probe_log: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class ValidationReport:
    """Summary of validation outcomes."""

    passed: bool
    failures: list[str]


def validate_event_metrics(
    scenarios: Sequence[MetricScenario],
) -> ValidationReport:
    """Validate event-derived metrics for basic numeric sanity.

    Args:
        scenarios: Scenarios to validate for non-negative, finite metrics.

    Returns:
        ValidationReport with pass/fail and diagnostic messages.
    """

    failures: list[str] = []
    for scenario in scenarios:
        metrics = _compute(scenario)
        for key, value in metrics.per_run.items():
            if not _is_event_metric_key(key):
                continue
            if not isinstance(value, (int, float)):
                failures.append(f"{scenario.name}: {key} not numeric.")
                continue
            if not isfinite(float(value)):
                failures.append(f"{scenario.name}: {key} not finite.")
            if float(value) < 0.0:
                failures.append(f"{scenario.name}: {key} is negative.")
        for agent_id, agent_metrics in metrics.per_agent.items():
            for key, value in agent_metrics.items():
                if key not in {"messages_sent", "messages_received"}:
                    continue
                if not isinstance(value, (int, float)):
                    failures.append(f"{scenario.name}: {agent_id} {key} not numeric.")
                    continue
                if not isfinite(float(value)):
                    failures.append(f"{scenario.name}: {agent_id} {key} not finite.")
                if float(value) < 0.0:
                    failures.append(f"{scenario.name}: {agent_id} {key} is negative.")
    return ValidationReport(passed=not failures, failures=failures)


def validate_counter_metrics(
    high: MetricScenario,
    low: MetricScenario,
    adversarial: Sequence[MetricScenario],
) -> ValidationReport:
    """Validate counter metrics against synthetic and adversarial scenarios.

    Args:
        high: Scenario expected to yield higher efficiency.
        low: Scenario expected to yield lower efficiency.
        adversarial: Scenarios to check for robustness (no NaN/negatives).

    Returns:
        ValidationReport with pass/fail and diagnostic messages.
    """

    failures: list[str] = []

    high_metrics = _compute(high)
    low_metrics = _compute(low)

    high_eff = high_metrics.per_run.get("efficiency")
    low_eff = low_metrics.per_run.get("efficiency")
    if high_eff is None or low_eff is None:
        failures.append("Missing efficiency metric for monotonicity check.")
    elif high_eff <= low_eff:
        failures.append(
            f"Efficiency monotonicity failed: {high_eff} <= {low_eff}."
        )

    for scenario in adversarial:
        metrics = _compute(scenario)
        for key, value in metrics.per_run.items():
            if not isinstance(value, (int, float)):
                failures.append(f"{scenario.name}: {key} not numeric.")
                continue
            if not isfinite(float(value)):
                failures.append(f"{scenario.name}: {key} not finite.")
            if float(value) < 0.0:
                failures.append(f"{scenario.name}: {key} is negative.")

    return ValidationReport(passed=not failures, failures=failures)


def _compute(scenario: MetricScenario) -> MetricsResult:
    event_log = scenario.event_log or []
    probe_log = scenario.probe_log or []
    return compute_metrics(
        event_log=event_log,
        probe_log=probe_log,
        task_outcome=scenario.task_outcome,
    )


def _is_event_metric_key(key: str) -> bool:
    if key in {"messages_sent", "messages_received"}:
        return True
    if key.startswith("event_count_"):
        return True
    if key.startswith("action_submitted_") and key.endswith("_count"):
        return True
    return False
