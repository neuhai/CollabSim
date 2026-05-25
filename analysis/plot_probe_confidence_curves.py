"""Probe confidence curves — all study conditions per task (2x2 panel figure).

Per probe_N round: each actor averages probe_N_1/_2/_3, then mean across agents.
One line per condition. Map task uses situation_awareness; others use grounding.

Usage:
  uv run python -m analysis.plot_probe_confidence_curves
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import matplotlib.pyplot as plt

from analysis.plot_persona_heatmaps import (
    condition_slug_to_label,
    condition_sort_key,
    discover_conditions,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "experiments" / "gpt5.5" / "study_conditions-persona"
DEFAULT_OUT = ROOT / "analysis_output" / "probe_confidence_panels.png"

CONDITION_COLORS: list[str] = [
    "#A07FA8",
    "#A0BBE4",
    "#F6C3CE",
    "#A2CFCE",
    "#FFE15D",
]

CONSTRUCT_BY_TASK: dict[str, str] = {
    "shapefactory": "grounding",
    "maptask": "situation_awareness",
    "hidden_profile": "grounding",
    "daytrader": "grounding",
}

PANELS: list[dict[str, Any]] = [
    {
        "task": "shapefactory",
        "panel_id": "a",
        "title": "Shape factory",
        "subtitle": "Resource coordination, 6 agents",
        "legend_loc": "lower right",
    },
    {
        "task": "maptask",
        "panel_id": "b",
        "title": "Map task",
        "subtitle": "Social dilemma, 3 agents",
        "legend_loc": "upper left",
    },
    {
        "task": "hidden_profile",
        "panel_id": "c",
        "title": "Hidden profile",
        "subtitle": "Information pooling, 3 agents",
        "legend_loc": "lower right",
    },
    {
        "task": "daytrader",
        "panel_id": "d",
        "title": "DayTrader",
        "subtitle": "Social dilemma, 3 agents",
        "legend_loc": "upper left",
    },
]

STYLE_RC: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "legend.fontsize": 9,
}


def legend_label(slug: str) -> str:
    label = condition_slug_to_label(slug)
    if label == "+BW↓":
        return "+BW ↓"
    return label


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


def probe_round(probe_id: str) -> int | None:
    parts = probe_id.split("_")
    if len(parts) >= 2 and parts[0] == "probe" and parts[1].isdigit():
        return int(parts[1])
    return None


def load_probe_records(path: Path) -> list[dict[str, Any]]:
    return list(iter_json_objects(path))


def aggregate_mean_series(
    records: list[dict[str, Any]], construct: str
) -> tuple[list[float], list[float]]:
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
        rnd = probe_round(probe_id)
        if rnd is None:
            continue
        dedup[(rnd, probe_id, str(actor))].append(float(conf))

    sessions: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (rnd, _probe_id, actor), confs in dedup.items():
        sessions[rnd][actor].append(statistics.mean(confs))

    xs: list[float] = []
    ys: list[float] = []
    for i, rnd in enumerate(sorted(sessions.keys())):
        actor_means = [
            statistics.mean(confs) for confs in sessions[rnd].values() if confs
        ]
        if not actor_means:
            continue
        xs.append(float(i))
        ys.append(statistics.mean(actor_means))
    return xs, ys


def available_conditions(task: str, data_root: Path) -> list[str]:
    task_dir = data_root / task
    if not task_dir.is_dir():
        return []
    has_probes = {
        p.name
        for p in task_dir.iterdir()
        if p.is_dir() and (p / "probes.jsonl").is_file()
    }
    slugs = [
        slug
        for slug in discover_conditions(task)
        if slug in has_probes and not slug.endswith("_collab")
    ]
    return sorted(slugs, key=condition_sort_key)


def style_axes(ax: plt.Axes) -> None:
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
    ax.tick_params(axis="both", which="major", labelsize=10, length=4, width=0.8)
    ymin, ymax = ax.get_ylim()
    pad = max(0.02, (ymax - ymin) * 0.06)
    ax.set_ylim(max(0.0, ymin - pad), min(1.0, ymax + pad))


def plot_panel(ax: plt.Axes, task: str, data_root: Path, panel: dict[str, Any]) -> None:
    construct = CONSTRUCT_BY_TASK[task]
    conditions = available_conditions(task, data_root)
    if not conditions:
        ax.set_visible(False)
        return

    for i, slug in enumerate(conditions):
        probes_path = data_root / task / slug / "probes.jsonl"
        records = load_probe_records(probes_path)
        xs, ys = aggregate_mean_series(records, construct)
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            color=CONDITION_COLORS[i % len(CONDITION_COLORS)],
            linewidth=2.0,
            solid_capstyle="round",
            label=legend_label(slug),
        )

    panel_id = panel["panel_id"]
    ax.set_title(
        f"({panel_id}) {panel['title']}",
        fontsize=13,
        fontweight="bold",
        loc="left",
        pad=18,
        color="#111111",
    )
    ax.text(
        0.0,
        1.02,
        panel["subtitle"],
        transform=ax.transAxes,
        fontsize=10,
        color="#666666",
        va="bottom",
        ha="left",
    )

    ax.set_xlabel("Probing round", labelpad=6)
    ax.set_ylabel("Probe confidence", labelpad=8)
    style_axes(ax)

    legend_loc = panel.get("legend_loc", "lower right")
    legend_kwargs: dict[str, Any] = {
        "frameon": True,
        "fancybox": False,
        "shadow": False,
        "borderpad": 0.35,
        "handlelength": 2.4,
        "handletextpad": 0.6,
        "borderaxespad": 0.6,
    }
    if legend_loc == "upper left":
        legend_kwargs.update(
            {"loc": "upper left", "bbox_to_anchor": (0.02, 0.98)}
        )
    else:
        legend_kwargs.update(
            {"loc": "lower right", "bbox_to_anchor": (0.98, 0.02)}
        )

    legend = ax.legend(**legend_kwargs)
    frame = legend.get_frame()
    frame.set_linewidth(0.8)
    frame.set_edgecolor("#333333")
    frame.set_facecolor("white")
    frame.set_alpha(1.0)


def build_figure(data_root: Path) -> plt.Figure:
    plt.rcParams.update(STYLE_RC)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="white")
    fig.subplots_adjust(
        hspace=0.38,
        wspace=0.22,
        top=0.92,
        bottom=0.08,
        left=0.09,
        right=0.97,
    )

    for ax, panel in zip(axes.flat, PANELS, strict=True):
        plot_panel(ax, panel["task"], data_root, panel)

    fig.text(
        0.99,
        0.995,
        "Y axis: probe confidence (0–1)",
        ha="right",
        va="top",
        fontsize=10,
        color="#444444",
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    data_root = args.data_root.resolve()

    for panel in PANELS:
        conds = available_conditions(panel["task"], data_root)
        labels = [legend_label(c) for c in conds]
        print(f"{panel['task']}: {', '.join(labels)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(data_root)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
