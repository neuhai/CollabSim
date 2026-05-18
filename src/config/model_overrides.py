"""Optional global overrides for every agent ``model`` block in experiment configs."""

from __future__ import annotations

from typing import Any


def apply_agent_model_overrides(
    config: dict[str, Any],
    *,
    provider: str | None = None,
    name: str | None = None,
    temperature: float | None = None,
) -> int:
    """Mutate ``config`` agents in-place. Returns count of agents updated.

    Missing ``model`` objects are created. Only supplied fields are written.
    """

    if provider is None and name is None and temperature is None:
        return 0

    touched = 0
    agents = config.get("agents")
    if not isinstance(agents, list):
        return 0
    for agent in agents:
        if not isinstance(agent, dict):
            continue
        if not isinstance(agent.get("id"), str) or not agent.get("id"):
            continue
        model = agent.get("model")
        if model is None or not isinstance(model, dict):
            model = {}
            agent["model"] = model
        if provider is not None:
            stripped = provider.strip()
            if stripped:
                model["provider"] = stripped
        if name is not None:
            stripped = name.strip()
            if stripped:
                model["name"] = stripped
        if temperature is not None:
            model["temperature"] = temperature
        touched += 1
    return touched
