"""Probe confidence curves — eight figures (4 tasks × persona/collab).

Data: experiments/final_res, all four models (Llama, Qwen, GPT, Claude).
Per model × condition: probe session → agents → mean, then mean across models.

Each run is rescaled so its actual end = 100% task progress; values are kept at
0%, 25%, 50%, 75%, and 100% only, then connected with a smooth interpolated curve.

Usage:
  uv run python -m analysis.plot_probe_confidence_curves

Writes eight single-panel PNGs plus ``all_probe_confidence_curves.png`` (2×4 composite).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import matplotlib.pyplot as plt
import numpy as np
import yaml

from analysis.plot_persona_heatmaps import (
    MODEL_SPECS,
    condition_sort_key,
    discover_conditions,
    model_group_dir,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs" / "study_conditions"
DEFAULT_DATA_ROOT = ROOT / "experiments" / "final_res"
DEFAULT_OUT_DIR = ROOT / "analysis_output"
COMPOSITE_FILENAME = "all_probe_confidence_curves.png"
TASKS_ORDER = ["shapefactory", "daytrader", "hidden_profile", "maptask"]

PALETTE: list[str] = [
    "#8857a8",  # baseline
    "#7c9fd9",  # bandwidth
    "#ace1af",  # information visibility
    "#fee55d",  # group size N=6
    "#fe8162",  # group size N=8
    "#c9a0dc",  # group size N=9
    "#4db8a8",  # group size N=10
]

GROUP_SIZE_COLORS: dict[int, str] = {
    6: PALETTE[3],
    8: PALETTE[4],
    9: PALETTE[5],
    10: PALETTE[6],
}

CONSTRUCT_BY_TASK: dict[str, str] = {
    "shapefactory": "grounding",
    "maptask": "situation_awareness",
    "hidden_profile": "grounding",
    "daytrader": "grounding",
}

TASK_DEFAULT_MAX_STEPS: dict[str, int] = {
    "maptask": 120,
    "hidden_profile": 30,
}

PROGRESS_PCTS: tuple[float, ...] = (0.0, 25.0, 50.0, 75.0, 100.0)
PROGRESS_N_BINS = len(PROGRESS_PCTS)
SMOOTH_CURVE_POINTS = 200

# Title, legend, Task progress, axis labels
UI_FONT_SIZE = 30
TICK_FONT_SIZE = 24  # x-axis % ticks slightly smaller
LINE_WIDTH = 3.0
HANDLE_LENGTH = 5.2

TASK_DISPLAY: dict[str, str] = {
    "shapefactory": "Shape Factory",
    "maptask": "Map Task",
    "hidden_profile": "Hidden Profile",
    "daytrader": "Day Trader",
}

PANELS_BASE: list[dict[str, Any]] = [
    {"task": "shapefactory", "output_stem": "Shape_factory"},
    {"task": "daytrader", "output_stem": "Day_Trader"},
    {"task": "hidden_profile", "output_stem": "Hidden_profile"},
    {"task": "maptask", "output_stem": "Map_task"},
]

STYLE_RC: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Lato", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
}

MODEL_SLUGS = [slug for _, slug in MODEL_SPECS]


def build_panels() -> list[dict[str, Any]]:
    panels: list[dict[str, Any]] = []
    for base in PANELS_BASE:
        for collab in (False, True):
            mode = "collab" if collab else "persona"
            panels.append(
                {
                    **base,
                    "collab": collab,
                    "output_name": f"{base['output_stem']}_{mode}",
                    "mode_label": "Theory-Informed" if collab else "Persona",
                }
            )
    return panels


def panel_for(task: str, collab: bool) -> dict[str, Any]:
    for panel in build_panels():
        if panel["task"] == task and panel["collab"] is collab:
            return panel
    raise KeyError(f"No panel for task={task!r} collab={collab}")


def xlabel_for_task(_task: str) -> str:
    return "Task progress (%)"


def legend_label(slug: str) -> str:
    if slug == "baseline":
        return "Baseline"
    if "bandwidth" in slug:
        return "Limited Comm.\nBandwidth"
    if slug in ("awareness_dashboard", "canvas_visibility"):
        return "Info. Visibility"
    m = re.fullmatch(r"group_size_(\d+)", slug)
    if m:
        return f"Group Size (N={m.group(1)})"
    return slug.replace("_", " ").title()


def color_for_condition(slug: str) -> str:
    """One color per condition type (consistent across tasks)."""
    if slug == "baseline":
        return PALETTE[0]
    if "bandwidth" in slug:
        return PALETTE[1]
    if slug in ("awareness_dashboard", "canvas_visibility"):
        return PALETTE[2]
    m = re.fullmatch(r"group_size_(\d+)", slug)
    if m:
        n = int(m.group(1))
        if n in GROUP_SIZE_COLORS:
            return GROUP_SIZE_COLORS[n]
        return PALETTE[3 + ((n - 6) % 4)]
    return PALETTE[0]


def legend_sort_key(label: str) -> tuple[int, int | str]:
    if label == "Baseline":
        return (0, 0)
    if label.startswith("Limited Comm."):
        return (1, 0)
    if label == "Info. Visibility":
        return (2, 0)
    m = re.search(r"N=(\d+)", label)
    if m:
        return (3, int(m.group(1)))
    return (4, label)


def iter_json_objects(path: Path) -> Iterator[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    dec = json.JSONDecoder()
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        obj, end = dec.raw_decode(text, i)
        yield obj
        i = end


def probe_step_index(row: dict[str, Any]) -> int | None:
    probe_id = str(row.get("probe_id", ""))
    parts = probe_id.split("_")
    if len(parts) >= 2 and parts[0] == "probe" and parts[1].isdigit():
        step_index = int(parts[1])
        return step_index if step_index >= 0 else None
    return None


def config_max_steps(task: str, condition: str) -> int:
    cfg_path = CONFIGS_DIR / task / f"{condition}.yml"
    if cfg_path.is_file():
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        ms = (data.get("experiment") or {}).get("max_steps")
        if isinstance(ms, int) and ms > 0:
            return ms
    return TASK_DEFAULT_MAX_STEPS.get(task, 120)


def resolve_run_end_step(
    run_dir: Path,
    records: list[dict[str, Any]],
    task: str,
    condition: str,
) -> int:
    """Probe progress 100% = last probe step (not simulation steps_taken)."""
    probe_steps = [
        s for row in records if (s := probe_step_index(row)) is not None
    ]
    probe_max = max(probe_steps) if probe_steps else 0
    if probe_max > 0:
        return probe_max
    taken = 0
    summary_path = run_dir / "run_summary.json"
    if summary_path.is_file():
        ts = json.loads(summary_path.read_text(encoding="utf-8")).get("task_summary") or {}
        st = ts.get("steps_taken")
        if isinstance(st, int) and st > 0:
            taken = st
    if taken > 0:
        return taken
    return config_max_steps(task, condition)


def progress_bin(step_index: int, run_end_step: int) -> int | None:
    if run_end_step <= 0:
        return None
    if step_index <= 0:
        return 0
    if step_index >= run_end_step:
        return PROGRESS_N_BINS - 1
    pct = 100.0 * step_index / run_end_step
    idx = int(pct / 25.0)
    return min(PROGRESS_N_BINS - 1, max(0, idx))


def progress_bin_center(bin_idx: int) -> float:
    return PROGRESS_PCTS[bin_idx]


def load_probe_records(path: Path) -> list[dict[str, Any]]:
    return list(iter_json_objects(path))


def _session_means_by_bucket(
    records: list[dict[str, Any]],
    construct: str,
    *,
    run_end_step: int,
) -> dict[int, float]:
    dedup: dict[tuple[int, str, str], list[float]] = defaultdict(list)

    for row in records:
        if row.get("construct") != construct:
            continue
        conf = row.get("confidence")
        if not isinstance(conf, (int, float)):
            continue
        probe_id = str(row.get("probe_id", ""))
        actor = row.get("actor_id")
        if actor is None:
            continue

        step_index = probe_step_index(row)
        if step_index is None:
            continue
        bucket = progress_bin(step_index, run_end_step)
        if bucket is None:
            continue
        dedup[(bucket, probe_id, str(actor))].append(float(conf))

    sessions: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (bucket, _probe_id, actor), confs in dedup.items():
        sessions[bucket][actor].append(statistics.mean(confs))

    out: dict[int, float] = {}
    for bucket, actor_map in sessions.items():
        actor_means = [
            statistics.mean(confs) for confs in actor_map.values() if confs
        ]
        if actor_means:
            out[bucket] = statistics.mean(actor_means)
    return out


def rescaled_progress_grid(
    records: list[dict[str, Any]],
    construct: str,
    run_end_step: int,
) -> dict[int, float]:
    """One run → 0/25/50/75/100% bins; last bin always includes the true run end."""
    return _session_means_by_bucket(records, construct, run_end_step=run_end_step)


def mean_progress_grids(grids: list[dict[int, float]]) -> tuple[list[float], list[float]]:
    """Average across models after each run was rescaled to 0–100%."""
    if not grids:
        return [], []
    accum: dict[int, list[float]] = defaultdict(list)
    for grid in grids:
        for bin_idx, y in grid.items():
            accum[bin_idx].append(y)
    if not accum:
        return [], []
    bins = sorted(accum)
    return [progress_bin_center(b) for b in bins], [statistics.mean(accum[b]) for b in bins]


def fill_progress_milestones(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    """Ensure 0% and 100% anchors; carry last value forward for missing bins."""
    if not xs:
        return [], []
    by_pct = dict(zip(xs, ys, strict=True))
    filled_x: list[float] = []
    filled_y: list[float] = []
    last: float | None = None
    for pct in PROGRESS_PCTS:
        if pct in by_pct:
            last = by_pct[pct]
        if last is None:
            continue
        filled_x.append(pct)
        filled_y.append(last)
    if not filled_x:
        return [], []
    if filled_x[0] > 0.0:
        filled_x.insert(0, 0.0)
        filled_y.insert(0, filled_y[0])
    if filled_x[-1] < 100.0:
        filled_x.append(100.0)
        filled_y.append(filled_y[-1])
    return filled_x, filled_y


def smooth_progress_curve(
    xs: list[float], ys: list[float], *, n_points: int = SMOOTH_CURVE_POINTS
) -> tuple[np.ndarray, np.ndarray]:
    """PCHIP smooth line from 0% to 100% through milestone means."""
    xs, ys = fill_progress_milestones(xs, ys)
    if not xs:
        return np.array([]), np.array([])
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    x_dense = np.linspace(0.0, 100.0, n_points)
    if len(x) == 1:
        return x_dense, np.full(n_points, y[0])
    try:
        from scipy.interpolate import PchipInterpolator

        interp = PchipInterpolator(x, y, extrapolate=True)
        y_dense = np.clip(interp(x_dense), 0.0, 1.0)
    except ImportError:
        y_dense = np.clip(np.interp(x_dense, x, y), 0.0, 1.0)
    return x_dense, y_dense


def available_conditions(task: str, data_root: Path, *, collab: bool) -> list[str]:
    slugs = discover_conditions(task)
    found: list[str] = []
    for slug in slugs:
        for model_slug in MODEL_SLUGS:
            run_dir = model_group_dir(data_root, model_slug, collab=collab) / task / slug
            if (run_dir / "probes.jsonl").is_file():
                found.append(slug)
                break
    return sorted(set(found), key=condition_sort_key)


def _style_legend_frame(legend: plt.Legend) -> None:
    frame = legend.get_frame()
    frame.set_linewidth(0.8)
    frame.set_edgecolor("#333333")
    frame.set_facecolor("white")
    frame.set_alpha(1.0)


def dedupe_legend_entries(
    handles: list[Any], labels: list[str]
) -> tuple[list[Any], list[str]]:
    by_label: dict[str, Any] = {}
    for handle, label in zip(handles, labels, strict=True):
        if label not in by_label:
            by_label[label] = handle
    ordered = sorted(by_label.items(), key=lambda item: legend_sort_key(item[0]))
    return [h for _, h in ordered], [lab for lab, _ in ordered]


def style_axes(ax: plt.Axes, *, ylim: tuple[float, float] | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        color="#d0d0d0",
        alpha=0.9,
    )
    ax.set_axisbelow(True)
    ax.tick_params(
        axis="x",
        which="major",
        labelsize=TICK_FONT_SIZE,
        length=4,
        width=0.8,
    )
    ax.tick_params(
        axis="y",
        which="major",
        labelsize=TICK_FONT_SIZE,
        length=4,
        width=0.8,
    )
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ymin, ymax = ax.get_ylim()
        pad = max(0.02, (ymax - ymin) * 0.06)
        ax.set_ylim(max(0.0, ymin - pad), min(1.0, ymax + pad))

    ax.set_xlim(0, 100)
    ax.set_xticks(list(PROGRESS_PCTS))
    ax.set_xticklabels(
        [f"{int(p)}%" for p in PROGRESS_PCTS], fontsize=TICK_FONT_SIZE
    )


def collect_condition_series(
    task: str,
    data_root: Path,
    *,
    collab: bool,
) -> list[tuple[list[float], list[float]]]:
    """Per-condition (xs, ys) at progress milestones, before smoothing."""
    construct = CONSTRUCT_BY_TASK[task]
    conditions = available_conditions(task, data_root, collab=collab)
    series_list: list[tuple[list[float], list[float]]] = []
    for slug in conditions:
        per_model_progress: list[dict[int, float]] = []
        for model_slug in MODEL_SLUGS:
            run_dir = model_group_dir(data_root, model_slug, collab=collab) / task / slug
            probes_path = run_dir / "probes.jsonl"
            if not probes_path.is_file():
                continue
            records = load_probe_records(probes_path)
            end_step = resolve_run_end_step(run_dir, records, task, slug)
            grid = rescaled_progress_grid(records, construct, end_step)
            if grid:
                per_model_progress.append(grid)
        xs, ys = mean_progress_grids(per_model_progress)
        if xs:
            series_list.append((xs, ys))
    return series_list


def ylim_with_padding(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 1.0)
    ymin, ymax = min(values), max(values)
    pad = max(0.02, (ymax - ymin) * 0.06)
    return (max(0.0, ymin - pad), min(1.0, ymax + pad))


def compute_row_ylims(data_root: Path) -> dict[bool, tuple[float, float]]:
    """Shared y-axis per composite row (persona vs collab)."""
    ylims: dict[bool, tuple[float, float]] = {}
    for collab in (False, True):
        all_y: list[float] = []
        for task in TASKS_ORDER:
            for _xs, ys in collect_condition_series(task, data_root, collab=collab):
                filled_x, filled_y = fill_progress_milestones(_xs, ys)
                if filled_x:
                    fx, fy = smooth_progress_curve(filled_x, filled_y)
                    all_y.extend(float(v) for v in fy)
                else:
                    all_y.extend(ys)
        ylims[collab] = ylim_with_padding(all_y)
    return ylims


def plot_panel(
    ax: plt.Axes,
    task: str,
    data_root: Path,
    panel: dict[str, Any],
    *,
    ylim: tuple[float, float] | None = None,
    show_ylabel: bool = True,
    show_xlabel: bool = True,
    title_pad: float = 10,
) -> tuple[int, list[Any], list[str]]:
    construct = CONSTRUCT_BY_TASK[task]
    collab = panel["collab"]
    conditions = available_conditions(task, data_root, collab=collab)
    if not conditions:
        ax.set_visible(False)
        return 0, [], []

    plotted = 0
    handles: list[Any] = []
    labels: list[str] = []
    for slug in conditions:
        per_model_progress: list[dict[int, float]] = []
        for model_slug in MODEL_SLUGS:
            run_dir = model_group_dir(data_root, model_slug, collab=collab) / task / slug
            probes_path = run_dir / "probes.jsonl"
            if not probes_path.is_file():
                continue
            records = load_probe_records(probes_path)
            end_step = resolve_run_end_step(run_dir, records, task, slug)
            grid = rescaled_progress_grid(records, construct, end_step)
            if grid:
                per_model_progress.append(grid)

        xs, ys = mean_progress_grids(per_model_progress)
        if not xs:
            continue

        plot_x, plot_y = smooth_progress_curve(xs, ys)
        (line,) = ax.plot(
            plot_x,
            plot_y,
            color=color_for_condition(slug),
            linewidth=LINE_WIDTH,
            solid_capstyle="round",
            label=legend_label(slug),
        )
        handles.append(line)
        labels.append(legend_label(slug))
        plotted += 1

    if show_xlabel:
        ax.set_xlabel(xlabel_for_task(task), fontsize=UI_FONT_SIZE, labelpad=8)
    if show_ylabel:
        ax.set_ylabel("Confidence", fontsize=UI_FONT_SIZE, labelpad=10)
    style_axes(ax, ylim=ylim)

    title = f"{TASK_DISPLAY[task]}\n({panel['mode_label']})"
    ax.set_title(title, fontsize=UI_FONT_SIZE, pad=title_pad)
    return plotted, handles, labels


def build_single_figure(
    data_root: Path,
    panel: dict[str, Any],
    *,
    ylim: tuple[float, float],
) -> plt.Figure:
    plt.rcParams.update(STYLE_RC)
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), facecolor="white")
    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.14, right=0.97)
    plot_panel(ax, panel["task"], data_root, panel, ylim=ylim, show_ylabel=True)
    return fig


def _add_shared_row_xlabel(fig: plt.Figure, axes_row: list[plt.Axes]) -> None:
    """Task progress label centered under a row (bottom row only in composite)."""
    fig.canvas.draw()
    row_left = min(ax.get_position().x0 for ax in axes_row)
    row_right = max(ax.get_position().x1 for ax in axes_row)
    row_bottom = min(ax.get_position().y0 for ax in axes_row)
    fig.text(
        (row_left + row_right) / 2,
        row_bottom - 0.058,
        xlabel_for_task(""),
        ha="center",
        va="top",
        fontsize=UI_FONT_SIZE,
    )


def _place_composite_legend(
    fig: plt.Figure,
    chart_axes: list[plt.Axes],
    ax_leg: plt.Axes,
    handles: list[Any],
    labels: list[str],
) -> None:
    """Legend vertically centered on all chart panels."""
    fig.canvas.draw()
    plot_bottom = min(ax.get_position().y0 for ax in chart_axes)
    plot_top = max(ax.get_position().y1 for ax in chart_axes)
    plot_center = (plot_bottom + plot_top) / 2
    leg_box = ax_leg.get_position()
    legend = ax_leg.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(leg_box.x0, plot_center),
        bbox_transform=fig.transFigure,
        frameon=True,
        fancybox=False,
        shadow=False,
        fontsize=UI_FONT_SIZE,
        borderpad=0.35,
        handlelength=HANDLE_LENGTH,
        handletextpad=0.8,
        labelspacing=0.50,
        borderaxespad=0.0,
    )
    _style_legend_frame(legend)
    ax_leg.axis("off")


def build_composite_figure(
    data_root: Path,
    row_ylims: dict[bool, tuple[float, float]],
) -> plt.Figure:
    """2×4 panels + one shared legend on the right (both rows)."""
    plt.rcParams.update(STYLE_RC)
    fig = plt.figure(figsize=(30, 12), facecolor="white")
    gs = fig.add_gridspec(
        2,
        5,
        width_ratios=[0.52, 0.52, 0.52, 0.52, 0.48],
        wspace=0.22,
        hspace=0.58,
        left=0.11,
        right=0.93,
        top=0.90,
        bottom=0.16,
    )

    all_handles: list[Any] = []
    all_labels: list[str] = []
    chart_axes: list[plt.Axes] = []

    for row_idx, collab in enumerate((False, True)):
        axes_row: list[plt.Axes] = []
        for col_idx, task in enumerate(TASKS_ORDER):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            axes_row.append(ax)
            chart_axes.append(ax)
            panel = panel_for(task, collab)
            _plotted, handles, labels = plot_panel(
                ax,
                task,
                data_root,
                panel,
                ylim=row_ylims[collab],
                show_ylabel=(col_idx == 0),
                show_xlabel=False,
                title_pad=6,
            )
            all_handles.extend(handles)
            all_labels.extend(labels)

        if row_idx == 1:
            _add_shared_row_xlabel(fig, axes_row)

    dedup_h, dedup_l = dedupe_legend_entries(all_handles, all_labels)
    ax_leg = fig.add_subplot(gs[:, 4])
    if dedup_h:
        _place_composite_legend(fig, chart_axes, ax_leg, dedup_h, dedup_l)
    else:
        ax_leg.axis("off")

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = build_panels()
    row_ylims = compute_row_ylims(data_root)
    for panel in panels:
        task = panel["task"]
        mode = "collab" if panel["collab"] else "persona"
        conds = available_conditions(task, data_root, collab=panel["collab"])
        labels = [legend_label(c) for c in conds]
        print(f"{task}/{mode}: {', '.join(labels)}")

        out_path = out_dir / f"{panel['output_name']}.png"
        fig = build_single_figure(data_root, panel, ylim=row_ylims[panel["collab"]])
        fig.savefig(out_path, dpi=150, facecolor="white")
        plt.close(fig)
        print(f"Wrote {out_path}")

    composite_path = out_dir / COMPOSITE_FILENAME
    composite_fig = build_composite_figure(data_root, row_ylims)
    composite_fig.savefig(
        composite_path, dpi=175, facecolor="white", bbox_inches="tight", pad_inches=0.10
    )
    plt.close(composite_fig)
    print(f"Wrote {composite_path}")


if __name__ == "__main__":
    main()
