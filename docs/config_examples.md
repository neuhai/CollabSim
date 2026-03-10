## Config Examples

Minimal example with counter task:

```yaml
experiment:
  id: "counter_v1"
  seed: 17
  max_steps: 5
agents:
  - id: "A"
    role: "tester"
    model: { provider: "local", name: "dummy", temperature: 0.0 }
action_space:
  enabled: ["decide"]
controls: {}
task:
  type: "counter"
  target_steps: 3
protocol:
  turn_taking: "sequential"
  termination: { condition: "max_steps" }
probe:
  cadence: "per_action"
  templates: ["grounding_v1"]
logging:
  trace_schema_version: "v0"
```

Accumulator example:

```yaml
experiment:
  id: "accumulator_v1"
  seed: 42
  max_steps: 8
agents:
  - id: "A"
    role: "tester"
    model: { provider: "local", name: "dummy", temperature: 0.0 }
action_space:
  enabled: ["decide"]
controls: {}
task:
  type: "accumulator"
  target_value: 10
  increment: 2
protocol:
  turn_taking: "sequential"
  termination: { condition: "max_steps" }
probe:
  cadence: "per_action"
  templates: ["grounding_v1"]
logging:
  trace_schema_version: "v0"
```

Hidden-profile example:

```yaml
experiment:
  id: "hidden_profile_v1"
  seed: 7
  max_steps: 6
agents:
  - id: "A"
    role: "planner"
    model: { provider: "local", name: "dummy", temperature: 0.0 }
  - id: "B"
    role: "planner"
    model: { provider: "local", name: "dummy", temperature: 0.0 }
action_space:
  enabled: ["communicate", "decide"]
controls: {}
task:
  type: "hidden_profile"
  target_steps: 3
  shared_facts:
    - "Objective is to pick the optimal plan."
  private_facts:
    A:
      - "Plan X is low risk."
    B:
      - "Plan Y yields higher reward."
protocol:
  turn_taking: "sequential"
  termination: { condition: "max_steps" }
probe:
  cadence: "per_action"
  templates: ["grounding_v1", "coordination_v1"]
logging:
  trace_schema_version: "v0"
```
