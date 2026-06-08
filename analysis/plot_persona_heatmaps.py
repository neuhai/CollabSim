"""Four-panel persona study heatmaps (models x all study conditions) with YlGnBu styling.

Conditions and display labels are read from configs/study_conditions/<task>/*.yml:
  baseline              -> Base
  *bandwidth*           -> +BW↓
  awareness_dashboard   -> +Dash
  canvas_visibility     -> +Canvas
  group_size_<n>        -> N=<n>

Each cell: metric value + arrow vs Base (same model, same task).

Usage:
  uv run python -m analysis.plot_persona_heatmaps
  uv run python -m analysis.plot_persona_heatmaps --out analysis_output/persona_heatmaps.png
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs" / "study_conditions"
DEFAULT_DATA_ROOT = ROOT / "experiments" / "gpt5.5"

MetricFn = Callable[[dict[str, Any]], float | None]

BASELINE_CONDITION = "baseline"
ARROW_THRESHOLD = 0.005
# Per-task skips: SF baseline is already 6 agents, so N=6 duplicates Base.
# DayTrader baseline is 3 agents — keep N=6.
EXCLUDED_CONDITIONS_BY_TASK: dict[str, frozenset[str]] = {
    "shapefactory": frozenset({"group_size_6"}),
}

# Custom ablation gradient: #ffffd9 (low) → … → #061d58 (high)
_ABLATION_CMAP_STOPS: list[str] = [
    "#ffffd9",
    "#97d5b7",
    "#40b9c3",
    "#2493c0",
    "#24499e",
    "#061d58",
]


def make_ablation_cmap() -> mcolors.LinearSegmentedColormap:
    """Smooth yellow–teal–blue gradient for heatmap cells."""
    return mcolors.LinearSegmentedColormap.from_list(
        "ablation_yellow_blue",
        _ABLATION_CMAP_STOPS,
        N=256,
    )


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


def metric_lang_std(metrics: dict[str, Any]) -> float | None:
    return _agent_stdev(metrics, "avg_message_length_tokens")


# ---------------------------------------------------------------------------
# Condition discovery + labels
# ---------------------------------------------------------------------------

ColumnSpec = dict[str, Any]

TASK_SPECS: list[tuple[str, str, MetricFn, str]] = [
    ("Shape Factory", "shapefactory", metric_probe_grounding_std, "Grounding STD"),
    ("Map Task", "maptask", metric_lang_std, "Lang STD"),
    ("Hidden Profile", "hidden_profile", metric_probe_grounding_std, "Grounding STD"),
    ("DayTrader", "daytrader", metric_probe_grounding_std, "Grounding STD"),
]

MODELS: list[tuple[str, Path | None]] = [
    ("Claude", DEFAULT_DATA_ROOT / "claude-persona (2)"),
    ("GPT-5.5", DEFAULT_DATA_ROOT / "study_conditions-persona"),
    ("Qwen3", None),
    ("Llama", None),
]

# Multi-model experiment layout under experiments/final_res (and copies).
MODEL_SPECS: list[tuple[str, str]] = [
    ("Llama", "llama"),
    ("Qwen", "qwen"),
    ("GPT", "gpt"),
    ("Claude", "claude"),
]


def model_group_dir(data_root: Path, model_slug: str, *, collab: bool) -> Path:
    mode = "collab" if collab else "persona"
    return data_root / f"{model_slug}_{mode}"


def condition_slug_to_label(slug: str) -> str:
    if slug == BASELINE_CONDITION:
        return "Base"
    if "bandwidth" in slug:
        return "+BW↓"
    if slug == "awareness_dashboard":
        return "+Dash"
    if slug == "canvas_visibility":
        return "+Canvas"
    m = re.fullmatch(r"group_size_(\d+)", slug)
    if m:
        return f"N={m.group(1)}"
    return slug


def condition_sort_key(slug: str) -> tuple[int, int | str]:
    if slug == BASELINE_CONDITION:
        return (0, 0)
    if "bandwidth" in slug:
        return (1, 0)
    if slug == "awareness_dashboard":
        return (2, 0)
    if slug == "canvas_visibility":
        return (2, 0)
    m = re.fullmatch(r"group_size_(\d+)", slug)
    if m:
        return (3, int(m.group(1)))
    return (4, slug)


def discover_conditions(task: str) -> list[str]:
    cfg_dir = CONFIGS_DIR / task
    if not cfg_dir.is_dir():
        raise FileNotFoundError(f"Missing config dir: {cfg_dir}")
    excluded = EXCLUDED_CONDITIONS_BY_TASK.get(task, frozenset())
    slugs = [p.stem for p in cfg_dir.glob("*.yml") if p.stem not in excluded]
    if not slugs:
        raise ValueError(f"No *.yml configs in {cfg_dir}")
    return sorted(slugs, key=condition_sort_key)


def build_panels() -> list[dict[str, Any]]:
    panels: list[dict[str, Any]] = []
    for title, task, metric_fn, metric_label in TASK_SPECS:
        columns = [
            {"label": condition_slug_to_label(slug), "condition": slug}
            for slug in discover_conditions(task)
        ]
        panels.append(
            {
                "title": title,
                "task": task,
                "metric": metric_fn,
                "metric_label": metric_label,
                "columns": columns,
            }
        )
    return panels


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_metrics(model_root: Path, task: str, condition: str) -> dict[str, Any] | None:
    path = model_root / task / condition / "metrics.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_metric(column: ColumnSpec, panel: dict[str, Any]) -> MetricFn:
    return column.get("metric") or panel["metric"]


def cell_value(
    model_root: Path | None,
    panel: dict[str, Any],
    column: ColumnSpec,
) -> float | None:
    if model_root is None:
        return None
    metrics = load_metrics(model_root, panel["task"], column["condition"])
    if metrics is None:
        return None
    return resolve_metric(column, panel)(metrics)


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
    return resolve_metric(column, panel)(metrics)


def delta_arrow(
    value: float | None,
    baseline: float | None,
    *,
    is_baseline_column: bool,
) -> str:
    if is_baseline_column:
        return "—"
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
    # Switch to light text once cells enter the mid-blue part of the gradient.
    return "white" if norm_value > 0.48 else "#1a1a1a"


def draw_panel(
    ax: plt.Axes,
    panel: dict[str, Any],
    cmap: mcolors.Colormap,
    models: list[tuple[str, Path | None]],
    *,
    show_ylabel: bool,
    label_fontsize: float,
) -> ScalarMappable | None:
    columns: list[ColumnSpec] = panel["columns"]
    n_rows = len(models)
    n_cols = len(columns)

    values = np.full((n_rows, n_cols), np.nan, dtype=float)
    labels = [[""] * n_cols for _ in range(n_rows)]

    for r, (_, model_root) in enumerate(models):
        for c, column in enumerate(columns):
            val = cell_value(model_root, panel, column)
            base = baseline_value(model_root, panel, column)
            is_base = column["condition"] == BASELINE_CONDITION
            if val is not None:
                values[r, c] = val
            labels[r][c] = "—" if val is None else f"{val:.2f}"

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

    cell_font = 9 if n_cols > 4 else 10
    for r in range(n_rows):
        for c in range(n_cols):
            val = values[r, c]
            face = "#f5f5f5" if math.isnan(val) else cmap(norm(val))
            ax.add_patch(
                Rectangle(
                    (c - 0.5, r - 0.5),
                    1,
                    1,
                    facecolor=face,
                    edgecolor="white",
                    linewidth=1.5,
                )
            )
            txt = labels[r][c]
            color = "#aaaaaa" if math.isnan(val) else _text_color(norm(val))
            ax.text(
                c, r, txt,
                ha="center", va="center",
                fontsize=cell_font, color=color,
                fontweight="semibold",
            )

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [col["label"] for col in columns],
        fontsize=label_fontsize,
        rotation=30 if n_cols > 4 else 0,
        ha="right" if n_cols > 4 else "center",
    )
    ax.set_yticks(range(n_rows))
    if show_ylabel:
        ax.set_yticklabels([name for name, _ in models], fontsize=10)
    else:
        ax.set_yticklabels([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        f"{panel['title']}\n({panel.get('metric_label', '')})",
        fontsize=11, fontweight="bold", pad=10, linespacing=1.4,
    )
    return sm


def resolve_models(data_root: Path) -> list[tuple[str, Path | None]]:
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
    panels = build_panels()
    models = resolve_models(data_root)
    max_cols = max(len(p["columns"]) for p in panels)

    cmap = make_ablation_cmap()
    fig_w = max(12.0, 1.5 * max_cols + 4.5)
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, 8.0), constrained_layout=True)
    fig.patch.set_facecolor("white")

    label_fs = 8 if max_cols > 4 else 9
    for ax, panel in zip(axes.flatten(), panels, strict=True):
        sm = draw_panel(
            ax,
            panel,
            cmap,
            models,
            show_ylabel=(ax.get_subplotspec().colspan.start == 0),
            label_fontsize=label_fs,
        )
        if sm is not None:
            cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, aspect=28)
            cbar.outline.set_visible(False)
            cbar.ax.tick_params(labelsize=7, length=2, width=0.5)
            cbar.ax.yaxis.set_tick_params(pad=2)

    fig.suptitle(
        "Condition ablation across tasks and models",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")
    for panel in panels:
        labels = [c["label"] for c in panel["columns"]]
        print(f"  {panel['title']}: {labels}")


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
