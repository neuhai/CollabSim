## Config Schema Overview

This document summarizes the top-level configuration keys expected by the CLI
and controller. See `configs/README.md` for runnable examples.

Top-level keys:
- `experiment`: run id, seed, and runtime limits
- `prompts`: optional prompt path overrides
- `agents`: roster of agent metadata and model specs
- `action_space`: enabled action primitives and constraints
- `controls`: communication and information constraints
- `task`: task type and task-specific parameters
- `protocol`: turn-taking, termination, and step-mode settings
- `probe`: probe cadence and template ids
- `logging`: trace schema version and output settings

Notes:
- All keys are required by the config validator even if they are empty.
- `task.type` selects a registered task in the controller.
