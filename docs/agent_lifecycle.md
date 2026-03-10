## Agent Lifecycle

Controllers call `AgentInterface.reset()` at the start of a run, passing the
configured experiment seed when available. This ensures deterministic agent
state initialization before any observations are issued.
