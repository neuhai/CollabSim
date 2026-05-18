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
  step_mode: "event"
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
  step_mode: "event"
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
  enabled: ["message", "decide"]
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
  step_mode: "event"
  termination: { condition: "max_steps" }
probe:
  cadence: "per_action"
  templates: ["grounding_v1", "coordination_v1"]
logging:
  trace_schema_version: "v0"
```

ShapeFactory-style example:

```yaml
experiment:
  id: "shapefactory_v1"
  seed: 42
  max_steps: 10
agents:
  - id: "A"
    role: "builder"
    model: { provider: "local", name: "dummy", temperature: 0.0 }
  - id: "B"
    role: "trader"
    model: { provider: "local", name: "dummy", temperature: 0.0 }
action_space:
  enabled: ["message", "produce_shape", "propose_trade_offer", "trade_response", "cancel_trade_offer", "fulfill_order"]
controls: {}
task:
  type: "shapefactory"
  target_steps: 10
  starting_money: 200
  shape_options: ["circle", "square", "triangle"]
protocol:
  turn_taking: "simultaneous"
  step_mode: "event"
  termination: { condition: "max_steps" }
probe:
  cadence: "per_action"
  templates: ["grounding_v1"]
logging:
  trace_schema_version: "v0"
```

DayTrader-style example:

```yaml
experiment:
  id: "daytrader_v1"
  seed: 17
  max_steps: 10
agents:
  - id: "A"
    role: "investor"
    model: { provider: "local", name: "dummy", temperature: 0.0 }
  - id: "B"
    role: "investor"
    model: { provider: "local", name: "dummy", temperature: 0.0 }
action_space:
  enabled: ["message", "make_investment"]
controls: {}
task:
  type: "daytrader"
  target_steps: 8
  starting_money: 200
protocol:
  turn_taking: "simultaneous"
  step_mode: "event"
  termination: { condition: "max_steps" }
probe:
  cadence: "per_action"
  templates: ["coordination_v1"]
logging:
  trace_schema_version: "v0"
```

MapTask-style example:

```yaml
experiment:
  id: "maptask_v1"
  seed: 29
  max_steps: 12
agents:
  - id: "A"
    role: "guider"
    model: { provider: "local", name: "dummy", temperature: 0.0 }
  - id: "B"
    role: "follower"
    model: { provider: "local", name: "dummy", temperature: 0.0 }
action_space:
  enabled: ["message", "update_map_progress"]
controls: {}
task:
  type: "maptask"
  target_steps: 10
  roles: { A: "guider", B: "follower" }
protocol:
  turn_taking: "simultaneous"
  step_mode: "event"
  termination: { condition: "max_steps" }
probe:
  cadence: "per_action"
  templates: ["situation_awareness_v1"]
logging:
  trace_schema_version: "v0"
```
