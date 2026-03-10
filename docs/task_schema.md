## Task Schema Overview

Tasks are configured under `task` in the experiment config. Each task type
defines its own parameters and internal state.

Common fields:
- `type`: registered task identifier (e.g., `counter`, `accumulator`)

Counter task fields:
- `target_steps`: integer number of steps to execute

Accumulator task fields:
- `target_value`: integer target total
- `increment`: integer increment per step

Hidden-profile task fields:
- `target_steps`: integer number of steps to execute
- `shared_facts`: list of shared facts available to all agents
- `private_facts`: mapping of agent_id to private fact lists

Notes:
- Task state is initialized by the task implementation and stored in the run
  logs and metrics.
