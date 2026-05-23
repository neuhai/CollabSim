"""Four-panel persona study heatmaps (models x conditions) with YlGnBu styling.

Data layout (default paths under experiments/gpt5.5/):
  - GPT-5.5  -> study_conditions-persona
  - Claude   -> claude-persona (2)
  - Qwen3 / Llama -> set paths when available (rows show as empty until then)

Each cell shows a numeric STD-like metric and an arrow vs that model's baseline
(single arrow for small change, double for |delta| >= 2 * arrow_threshold).

Usage:
  uv run python -m analysis.plot_persona_heatmaps
  uv run python -m analysis.plot_persona_heatmaps --out analysis_output/persona_heatmaps.png
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "experiments" / "gpt5.5"

MetricFn = Callable[[dict[str, Any]], float | None]

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _per_run(metrics: dict[str, Any], key: str) -> float | None:
    val = (metrics.get("per_run") or {}).get(key)
    return float(val) if isinstance(val, (int, float)) and not math.isnan(val) else None


def _agent_stdev(metrics: dict[str, Any], key: str) -> float | None:
    vals = [
        float(agent[key])
        for agent in (metrics.get("per_agent") or {}).values()
        if isinstance(agent, dict) and isinstance(agent.get(key), (int, float))
    ]
    if len(vals) < 2:
        return None
    return statistics.stdev(vals)


def metric_probe_grounding_std(metrics: dict[str, Any]) -> float | None:
    return _per_run(metrics, "probe_grounding_confidence_std")


def metric_probe_coordination_std(metrics: dict[str, Any]) -> float | None:
    return _per_run(metrics, "probe_coordination_confidence_std")


def metric_lang_std(metrics: dict[str, Any]) -> float | None:
    return _agent_stdev(metrics, "avg_message_length_tokens")


def metric_task_std(metrics: dict[str, Any]) -> float | None:
    return _agent_stdev(metrics, "map_progress_updates")


# ---------------------------------------------------------------------------
# Panel specification
# ---------------------------------------------------------------------------

ColumnSpec = dict[str, Any]

PANELS: list[dict[str, Any]] = [
    {
        "title": "Shape Factory",
        "task": "shapefactory",
        "metric": metric_probe_grounding_std,
        "metric_label": "Grounding STD",
        "columns": [
            {"label": "+BW↓", "condition": "bandwidth_1_msg_per_sim_min"},
            # N=10 group size; rename to awareness_dashboard if you meant +Dash only.
            {"label": "+Dash\nN=10", "condition": "group_size_10"},
        ],
    },
    {
        "title": "Map Task",
        "task": "maptask",
        "metric": metric_lang_std,
        "metric_label": "Lang STD",
        "columns": [
            {"label": "+BW↓", "condition": "bandwidth_max_words_5"},
            {"label": "+Canvas", "condition": "canvas_visibility"},
        ],
    },
    {
        "title": "Hidden Profile",
        "task": "hidden_profile",
        "metric": metric_probe_grounding_std,
        "metric_label": "Grounding STD",
        "columns": [
            {"label": "+BW↓", "condition": "bandwidth_max_words_5"},
            {
                "label": "+Theory",
                "condition": "bandwidth_max_words_5",
                "metric": metric_probe_coordination_std,
                "metric_label": "Coordination STD",
            },
        ],
    },
    {
        "title": "DayTrader",
        "task": "daytrader",
        "metric": metric_probe_grounding_std,
        "metric_label": "Grounding STD",
        "columns": [
            {"label": "+BW↓", "condition": "bandwidth_1_msg_per_5_actions"},
            {"label": "N=9", "condition": "group_size_9"},
            {"label": "+Theory", "condition": "group_size_6"},
        ],
    },
]

MODELS: list[tuple[str, Path | None]] = [
    ("Claude", DEFAULT_DATA_ROOT / "claude-persona (2)"),
    ("GPT-5.5", DEFAULT_DATA_ROOT / "study_conditions-persona"),
    ("Qwen3", None),
    ("Llama", None),
]

BASELINE_CONDITION = "baseline"
ARROW_THRESHOLD = 0.005


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_metrics(model_root: Path, task: str, condition: str) -> dict[str, Any] | None:
    path = model_root / task / condition / "metrics.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_metric(column: ColumnSpec, panel: dict[str, Any]) -> tuple[MetricFn, str]:
    metric_fn = column.get("metric") or panel["metric"]
    label = column.get("metric_label") or panel.get("metric_label", "STD")
    return metric_fn, label


def cell_value(
    model_root: Path | None,
    panel: dict[str, Any],
    column: ColumnSpec,
) -> float | None:
    if model_root is None:
        return None
    condition = column["condition"]
    metrics = load_metrics(model_root, panel["task"], condition)
    if metrics is None:
        return None
    metric_fn, _ = resolve_metric(column, panel)
    return metric_fn(metrics)


def baseline_value(
    model_root: Path | None,
    panel: dict[str, Any],
    column: ColumnSpec,
) -> float | None:
    if model_root is None:
        return None
    metrics = load_metrics(model_root, panel["task"], BASELINE_CONDITION)
    if metrics is None:
        return None
    metric_fn, _ = resolve_metric(column, panel)
    return metric_fn(metrics)


def delta_arrow(value: float | None, baseline: float | None) -> str:
    if value is None or baseline is None:
        return "—"
    delta = value - baseline
    if abs(delta) < ARROW_THRESHOLD:
        return "→"
    if delta > 0:
        return "↑↑" if delta >= 2 * ARROW_THRESHOLD else "↑"
    return "↓↓" if delta <= -2 * ARROW_THRESHOLD else "↓"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _text_color(norm_value: float) -> str:
    return "white" if norm_value > 0.62 else "black"


def draw_panel(
    ax: plt.Axes,
    panel: dict[str, Any],
    cmap: mcolors.Colormap,
    models: list[tuple[str, Path | None]],
    *,
    show_ylabel: bool,
) -> ScalarMappable | None:
    columns: list[ColumnSpec] = panel["columns"]
    n_rows = len(models)
    n_cols = len(columns)

    values = np.full((n_rows, n_cols), np.nan, dtype=float)
    arrows = [[""] * n_cols for _ in range(n_rows)]
    labels = [[""] * n_cols for _ in range(n_rows)]

    for r, (_, model_root) in enumerate(models):
        for c, column in enumerate(columns):
            val = cell_value(model_root, panel, column)
            base = baseline_value(model_root, panel, column)
            if val is not None:
                values[r, c] = val
            arrows[r][c] = delta_arrow(val, base)
            labels[r][c] = "—" if val is None else f"{val:.2f}\n{arrows[r][c]}"

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(panel["title"], fontsize=12, fontweight="bold", pad=8)
        ax.set_axis_off()
        return None

    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if math.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_aspect("equal")

    for r in range(n_rows):
        for c in range(n_cols):
            val = values[r, c]
            face = "#f0f0f0" if math.isnan(val) else cmap(norm(val))
            rect = Rectangle(
                (c - 0.5, r - 0.5),
                1,
                1,
                facecolor=face,
                edgecolor="white",
                linewidth=2,
            )
            ax.add_patch(rect)
            if not math.isnan(val):
                nv = norm(val)
                ax.text(
                    c,
                    r,
                    labels[r][c],
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=_text_color(nv),
                    linespacing=1.1,
                )
            else:
                ax.text(c, r, "—", ha="center", va="center", fontsize=10, color="#666666")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([col["label"] for col in columns], fontsize=9)
    ax.set_yticks(range(n_rows))
    if show_ylabel:
        ax.set_yticklabels([name for name, _ in models], fontsize=10)
    else:
        ax.set_yticklabels([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(panel["title"], fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("")
    return sm


def resolve_models(data_root: Path) -> list[tuple[str, Path | None]]:
    """Map model display names to experiment directories under data_root."""
    resolved: list[tuple[str, Path | None]] = []
    for name, configured in MODELS:
        if configured is None:
            guess = data_root / f"{name.lower().replace('.', '').replace('-', '')}-persona"
            resolved.append((name, guess if guess.is_dir() else None))
            continue
        path = configured if configured.is_absolute() else data_root / configured.name
        resolved.append((name, path if path.is_dir() else None))
    return resolved


def plot_figure(
    out_path: Path,
    *,
    dpi: int = 160,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> None:
    models = resolve_models(data_root)

    cmap = plt.cm.YlGnBu
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    for ax, panel in zip(axes.flatten(), PANELS, strict=True):
        sm = draw_panel(ax, panel, cmap, models, show_ylabel=(ax.get_subplotspec().colspan.start == 0))
        if sm is not None:
            cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, aspect=18)
            cbar.ax.tick_params(labelsize=7)
            metric_label = panel.get("metric_label", "STD")
            cbar.set_label(metric_label, fontsize=8)

    fig.suptitle(
        "Persona study — condition heatmaps (value + Δ vs baseline)",
        fontsize=14,
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Parent folder containing model run dirs (default: experiments/gpt5.5)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "analysis_output" / "persona_heatmaps.png",
    )
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()
    plot_figure(args.out, dpi=args.dpi, data_root=args.data_root)


if __name__ == "__main__":
    main()
