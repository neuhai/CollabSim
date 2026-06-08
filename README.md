# CollabSim

<p align="center">
    <a href="https://arxiv.org/abs/2606.06399">
        <img src="https://img.shields.io/badge/arXiv-2606.06399-B31B1B.svg?style=plastic&logo=arxiv" alt="arXiv">
    </a>
    <a href="https://github.com/neuhai/CollabSim">
        <img src="https://img.shields.io/badge/GitHub-CollabSim-black?style=plastic&logo=github" alt="GitHub">
    </a>
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=plastic" alt="License: MIT">
    </a>
</p>

<p align="center">
    <b>CollabSim: A CSCW-Grounded Methodology for Investigating Collaborative Competence of LLM Agents through Controlled Multi-Agent Experiments</b>
</p>

<p align="center">
    Jiaju Chen, Bo Sun, Yuxuan Lu, Yun Wang, Dakuo Wang, Bingsheng Yao
</p>

**CollabSim** is a configurable simulation framework for systematically assessing LLM agents' collaborative competence under controlled interaction conditions. It draws on experimental paradigms from Computer-Supported Cooperative Work (CSCW) research on distributed human teams, where collaborators coordinate through structured text-based channels without co-presence.

Researchers can vary task constraints such as **communication bandwidth**, **information visibility**, and **team size**, so that the effects of specific interaction conditions on collaborative behavior can be examined in isolation. CollabSim also includes a **probing module** that elicits each agent's reported mental-model awareness of the task state, partner intentions, and own reasoning after actions, enabling analysis of internal collaborative states beyond observable behavior.

## Collaborative Tasks

CollabSim implements four CSCW-inspired tasks that reflect core process-level coordination challenges:

| Task | CSCW paradigm | Challenge |
|------|---------------|-----------|
| **Shape Factory** | Resource coordination (Bos et al., 2004) | Coordinating production and trade under cost asymmetry |
| **DayTrader** | Social-dilemma negotiation (Bos et al., 2002) | Balancing individual and collective incentives |
| **Hidden Profile** | Information pooling (Stasser & Titus, 1985) | Integrating distributed private information |
| **Map Task** | Referential grounding (Anderson et al., 1991) | Establishing shared spatial understanding through language alone |

Each task is fully configurable via YAML. Example configs live in `configs/`, and paper study conditions are in `configs/study_conditions/`.

### Study Conditions

Pre-configured condition sweeps vary one interaction dimension at a time:

| Task | Conditions |
|------|------------|
| **Shape Factory** | `baseline`, `bandwidth_1_msg_per_sim_min`, `awareness_dashboard`, `group_size_6`, `group_size_8`, `group_size_10` |
| **DayTrader** | `baseline`, `bandwidth_1_msg_per_5_actions`, `group_size_6`, `group_size_9` |
| **Hidden Profile** | `baseline`, `bandwidth_max_words_5` |
| **Map Task** | `baseline`, `bandwidth_max_words_5`, `canvas_visibility` |

Condition names map directly to YAML files under `configs/study_conditions/<task>/`.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/neuhai/CollabSim.git
cd CollabSim
uv sync
```

Optional dev dependencies (includes `pytest`):

```bash
uv sync --group dev
```

Run tests:

```bash
uv run pytest
```

## Environment & LLM Setup

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

**Never commit `.env` or API keys.** The repo `.gitignore` already excludes `.env`.

### Bulk model overrides (recommended)

Set these environment variables to control the LLM for **all agents** without editing each YAML:

| Variable | Description | Example |
|----------|-------------|---------|
| `COLLABSIM_MODEL_PROVIDER` | Agent backend | `litellm`, `openai`, `azure`, `sglang` |
| `COLLABSIM_MODEL_NAME` | Model or deployment id | `gpt-4o` |
| `COLLABSIM_MODEL_TEMPERATURE` | Sampling temperature (optional) | `0.0` |

CLI flags (`--model-provider`, `--model-name`, `--model-temperature`) override the corresponding env var for that run only.

```bash
export COLLABSIM_MODEL_PROVIDER=litellm
export COLLABSIM_MODEL_NAME=gpt-4o
uv run python -m src.cli configs/study_conditions/shapefactory/baseline.yml --print-actions
```

### Provider-specific credentials

**LiteLLM** (recommended default — supports OpenAI, Anthropic, Azure, Bedrock, and OpenAI-compatible endpoints):

```bash
export OPENAI_API_KEY=your-key-here
# Optional proxy:
# export LITELLM_API_BASE=http://127.0.0.1:4000
# export LITELLM_API_KEY=your-proxy-key
```

See `configs/litellm_example.yml`.

**OpenAI**:

```bash
export OPENAI_API_KEY=your-key-here
```

**Azure OpenAI**:

```bash
export AZURE_OPENAI_API_KEY=your-key-here
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
export AZURE_OPENAI_DEPLOYMENT=your-deployment-name   # optional if set in YAML model.name
export AZURE_OPENAI_API_VERSION=2024-02-15-preview    # optional
```

See `configs/hidden_profile_azure.yml`.

**SGLang** (local open-source models):

```bash
# Terminal 1 — start inference server
uv run python scripts/start_sglang_server.py meta-llama/Llama-3.1-8B-Instruct --tp 1

# Terminal 2 — run experiment
export SGLANG_HOST=127.0.0.1
export SGLANG_PORT=30000
export COLLABSIM_MODEL_PROVIDER=sglang
export COLLABSIM_MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
uv run python -m src.cli configs/sglang_example.yml --print-actions
```

See `configs/sglang_example.yml` for a full example.

## Running Experiments

### Single run

```bash
uv run python -m src.cli configs/study_conditions/daytrader/baseline.yml \
  --run-id my_run \
  --print-actions
```

Useful flags:

- `--validate-only` — check config without running
- `--max-steps N` — cap simulation length (useful for smoke tests)
- `--output-dir PATH` — override log output directory
- `--collaboration` — append collaboration-priming instructions (`prompts/collaboration_module.md`) to each agent's initial prompt
- `--wandb` — enable Weights & Biases logging (optional; requires `wandb login`)

### Quick smoke test (10 steps)

```bash
uv run python -m src.cli configs/study_conditions/hidden_profile/baseline.yml \
  --max-steps 10 --print-actions
```

### Batch study conditions

Run all conditions for one task, or all four tasks:

```bash
# All four tasks
./configs/study_conditions/run_task_batch.sh all

# One task
./configs/study_conditions/run_task_batch.sh shapefactory

# Smoke mode (10 steps per condition)
./configs/study_conditions/run_task_batch.sh all smoke

# With collaboration priming (results saved to <condition>_collab folders)
./configs/study_conditions/run_task_batch.sh all --collaboration
```

Batch runner options:

| Flag | Description |
|------|-------------|
| `--jobs N` | Parallel conditions (default: all pending) |
| `--force` | Re-run even if results already exist |
| `--retry-failed` | Re-run incomplete conditions |
| `--list-failed` | List incomplete conditions without running |
| `--conditions a,b` | Run only named conditions |
| `--no-wandb-upload` | Skip W&B artifact upload after batch |

Results are written under `experiments/study_conditions/`. When W&B is enabled, set `WANDB_PROJECT` (default: `collabsim`).

### Web GUI (optional)

Launch a minimal Flask UI for uploading configs and inspecting runs:

```bash
uv run flask --app src.gui.app run --debug
```

Runs are saved to `experiments/gui_runs/`.

## Output & Analysis

Each run produces structured logs under the configured `logging.output_dir`:

- `events.jsonl` — full event trace (actions, messages, state changes)
- `actions.jsonl` — agent action records
- `probes.jsonl` — mental-state probe responses
- `metrics.json` — task outcome metrics
- `run_manifest.json` — experiment metadata and config snapshot

Analyze one or more run directories:

```bash
# Single run
uv run python -m analysis.report --run experiments/study_conditions/daytrader/baseline

# All runs under a folder
uv run python -m analysis.report --runs experiments/study_conditions/daytrader
```

Outputs CSVs to `analysis_output/` by default: `task_metrics.csv`, `probe_metrics.csv`, `combined.csv`.

Additional plotting scripts are in `analysis/` (e.g. `plot_probe_confidence_curves.py`, `plot_persona_heatmaps.py`).

## Customization

CollabSim is designed to be extended through YAML configuration and pluggable components.

### Interaction conditions

Edit `controls` in a config YAML to vary collaboration constraints:

- **Communication bandwidth** — `controls.communication.max_messages_per_turn`, `max_message_words`, `min_sim_interval_between_communicate_sec`
- **Communication mode** — `controls.communication.mode`: `broadcast` or `direct`
- **Information visibility** — `controls.information_distribution`, `controls.visibility_defaults`, `controls.visibility_map`
- **Team size** — add or remove entries under `agents`

See `docs/config_fields.md` and `docs/config_schema.md` for the full field reference.

### Prompts & personas

Prompt templates live in `prompts/`. Override paths per experiment under `prompts:` in the YAML (e.g. `task`, `action`, `probe`, `persona_profiles`). Persona profiles are defined in `prompts/persona_profiles.json`.

### Probing module

Configure probes under `probe:` in the YAML:

```yaml
probe:
  cadence: per_action          # or per_turn, per_agent_n_actions, on_event
  templates: ["grounding_v1", "coordination_v1"]
  questions_path: prompts/interview_questions_hidden_profile.json
```

Template definitions are in `configs/probe_templates.yml`. See `docs/probe_templates.md` and `docs/probe_response_schema.md`.

### Collaboration priming

Pass `--collaboration` (or set `experiment.collaboration: true`) to inject CSCW-grounded collaboration instructions from `prompts/collaboration_module.md` into each agent's system prompt. This tests whether explicit collaboration guidance improves agent behavior.

### Adding a new task

1. Implement task logic in `src/tasks/<your_task>.py` (register `init_state`, `step`, `apply_action`).
2. Register it in `src/tasks/registry.py`.
3. Add task instructions and return-format prompts under `prompts/`.
4. Create a YAML config referencing `task.type: <your_task>`.

Reference implementations: `src/tasks/shapefactory.py`, `src/tasks/daytrader.py`, `src/tasks/hidden_profile.py`, `src/tasks/maptask.py`.

### Adding a new study condition

Copy an existing condition YAML in `configs/study_conditions/<task>/`, modify the varied dimension, and update `logging.output_dir`. Run it directly or include it in a batch via `--conditions`.

## Project Structure

```
CollabSim/
├── configs/                  # Experiment YAML configs and study conditions
├── prompts/                  # Agent prompt templates and persona profiles
├── src/
│   ├── agents/               # LLM agent backends (LiteLLM, OpenAI, Azure, SGLang)
│   ├── controller/           # Simulation controller and stepping logic
│   ├── probe/                # Mental-state probing module
│   ├── tasks/                # Task implementations (Shape Factory, DayTrader, etc.)
│   └── cli.py                # CLI entrypoint
├── analysis/                 # Post-run metrics and plotting
├── docs/                     # Schema and design documentation
└── scripts/                  # Utility scripts (SGLang server, persona generation)
```

## Documentation

Detailed schemas and examples are in `docs/`:

- [Config fields](docs/config_fields.md) · [Config schema](docs/config_schema.md) · [Config examples](docs/config_examples.md)
- [Event schema](docs/event_schema.md) · [Action schema](docs/action_schema.md) · [Metrics](docs/metrics.md)
- [Probe templates](docs/probe_templates.md) · [Agent lifecycle](docs/agent_lifecycle.md)

## Citation

If you use CollabSim in your research, please cite our paper:

```bibtex
@article{chen2026collabsim,
  title   = {CollabSim: A CSCW-Grounded Methodology for Investigating Collaborative Competence of LLM Agents through Controlled Multi-Agent Experiments},
  author  = {Chen, Jiaju and Sun, Bo and Lu, Yuxuan and Wang, Yun and Wang, Dakuo and Yao, Bingsheng},
  year    = {2026},
  eprint  = {2606.06399},
  archivePrefix = {arXiv},
  primaryClass  = {cs.HC},
  url     = {https://arxiv.org/abs/2606.06399}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
