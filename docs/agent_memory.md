## Agent Memory

Controllers persist `AgentInterface.serialize()` snapshots per agent and pass them back via
`Observation.memory` on the next step. Controllers also call `AgentInterface.load()` with the
stored memory before requesting a new action proposal. This enables reproducible agent state
tracking without exposing private state through the shared task state.

The optional `protocol.agent_memory_key_limit` config limits the number of keys
stored per agent by retaining the first N keys in sorted order.

The optional `protocol.observation_memory_key_limit` config limits the number of
memory keys attached to each observation.

The optional `protocol.agent_memory_key_limit_by_agent` map overrides memory
limits per agent id.

The optional `protocol.observation_memory_key_limit_by_agent` map overrides
observation memory limits per agent id.
