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

### 1) ShapeFactory

```bash
uv run python -m src.cli configs/shapefactory_time_mode_example.yml --print-actions
```

### 2) MapTask

Formal run:

```bash
uv run python -m src.cli configs/maptask_example.yml --print-actions
```

Debug run:

```bash
uv run python -m src.cli configs/maptask_example.yml --max-steps 3 --dry-run --print-actions
```

### 3) DayTrader

```bash
uv run python -m src.cli configs/daytrader_example.yml --print-actions
```

### 4) Hidden Profile

```bash
uv run python -m src.cli configs/hidden_profile_azure.yml --print-actions
```