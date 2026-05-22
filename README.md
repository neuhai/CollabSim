# CodeRepo_Agent-Collaboration-Simulation

## Setup

[uv](https://docs.astral.sh/uv/) is used for dependencies and lockfile (`uv.lock`).

```bash
uv sync
```

Optional dev tools (includes `pytest`):

```bash
uv sync --group dev
```

Run tests:

```bash
uv run pytest
```

## Bulk LLM settings (all agents)

Without editing each YAML, set **environment variables** (e.g. in `.env` or your shell) before `uv run python -m src.cli …`:

- `COLLABSIM_MODEL_PROVIDER` — e.g. `azure`, `litellm`, `sglang`, `openai`
- `COLLABSIM_MODEL_NAME` — deployment or LiteLLM model id
- `COLLABSIM_MODEL_TEMPERATURE` — optional float

Or pass flags; any flag you pass overrides the corresponding env var for that field only: `--model-provider`, `--model-name`, `--model-temperature`.

Example:

```bash
export COLLABSIM_MODEL_PROVIDER=litellm
export COLLABSIM_MODEL_NAME=gpt-4o
uv run python -m src.cli configs/study_conditions/shapefactory/baseline.yml --print-actions
```

## Experiments (4)

Run all study conditions for all four tasks (parallel within each task; skips conditions that already have results):

```bash
./configs/study_conditions/run_task_batch.sh all
```

Quick smoke (10 steps per condition):

```bash
./configs/study_conditions/run_task_batch.sh all smoke
```

Optional flags: `--collaboration [true]`, `--jobs N`, `--force`.