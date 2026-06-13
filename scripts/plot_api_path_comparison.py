#!/usr/bin/env python3
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from figure_data import (
    MODE_ORDER,
    ModeResult,
    figures_dir,
    load_pyplot,
    load_query_results,
    parse_query_id,
    query_run_by_id,
    read_json,
    resolve_run_root,
    selected_functional_match,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot selected API paths across modes for one query in a completed run folder."
    )
    parser.add_argument("run_folder", type=Path, help="Run folder containing qXX_* result folders.")
    parser.add_argument("--query", default=None, help="Query id to plot. Defaults to the first discovered query.")
    parser.add_argument("--output-dir", type=Path, help="Output directory. Defaults to <run_folder>/figures.")
    parser.add_argument("--png-dpi", type=int, default=300, help="PNG export resolution.")
    return parser.parse_args()


def first_query_id(run_root: Path) -> str:
    from figure_data import discover_query_runs

    return discover_query_runs(run_root)[0].query_id


def load_subtasks(query_dir: Path) -> list[dict]:
    path = query_dir / "0_decomposer.json"
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Decomposer output must be a JSON list: {path}")
    subtasks = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Subtask {index} in {path} is not an object.")
        subtask_id = str(item.get("id", index))
        description = str(item.get("description", "")).strip()
        subtasks.append({"id": subtask_id, "description": description})
    if not subtasks:
        raise ValueError(f"No subtasks found in {path}")
    return subtasks


def planner_steps(query_dir: Path, mode: str) -> list[dict]:
    path = query_dir / mode / "4_planner.json"
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Planner output must be a JSON object: {path}")
    workflow = payload.get("execution_workflow")
    if not isinstance(workflow, dict) or not isinstance(workflow.get("steps"), list):
        raise ValueError(f"Planner output does not contain execution_workflow.steps: {path}")
    steps = [step for step in workflow["steps"] if isinstance(step, dict)]
    steps.sort(key=lambda step: int(step.get("step", 0)))
    return steps


def wrapped_label(text: str, width: int = 24) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=True, break_on_hyphens=False))


def result_by_mode(results: list[ModeResult]) -> dict[str, ModeResult]:
    return {result.mode: result for result in results}


def generate(run_folder: str | Path, query: str | None = None, output_dir: str | Path | None = None, png_dpi: int = 300) -> Path:
    run_root = resolve_run_root(run_folder)
    query_id = parse_query_id(query) if query else first_query_id(run_root)
    query_run = query_run_by_id(run_root, query_id)
    subtasks = load_subtasks(query_run.path)
    results = result_by_mode(load_query_results(query_run))

    plt = load_pyplot()
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 12, "axes.labelsize": 9})

    mode_count = len(MODE_ORDER)
    subtask_count = len(subtasks)
    fig_width = max(13.5, 2.9 * subtask_count + 4.0)
    fig_height = max(4.8, 1.35 * mode_count + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(-1.15, subtask_count + 1.25)
    ax.set_ylim(-0.95, mode_count + 0.8)
    ax.axis("off")
    ax.set_title(f"{query_id} Selected API Path Comparison Across Modes", loc="left", pad=16, fontweight="bold")

    for idx, subtask in enumerate(subtasks):
        ax.text(idx, mode_count + 0.28, f"Subtask {subtask['id']}", ha="center", va="center", fontweight="bold", color="#344054")
        ax.plot([idx, idx], [-0.28, mode_count - 0.55], color="#E5EAF0", linewidth=0.8, zorder=0)

    for mode_index, mode in enumerate(MODE_ORDER):
        y = mode_count - mode_index - 1
        if mode_index % 2 == 1:
            ax.add_patch(plt.Rectangle((-0.8, y - 0.42), subtask_count + 1.85, 0.84, color="#F6F8FA", zorder=0))
        ax.text(-0.52, y, mode, ha="right", va="center", fontweight="bold", color="#182230")

        steps = planner_steps(query_run.path, mode)
        by_subtask = {str(step.get("subtask_id")): step for step in steps}
        for idx, subtask in enumerate(subtasks):
            step = by_subtask.get(str(subtask["id"]))
            if step is None:
                continue
            api_id = str(step.get("api_id", "")).strip()
            fm = selected_functional_match(query_run, mode, str(subtask["id"]), api_id)
            if fm is None:
                fm = step.get("functional_match_label")
            fm_int = int(fm) if fm in (0, 1, "0", "1") else 0
            face = "#E6F4EE" if fm_int == 1 else "#FFF1E8"
            edge = "#2E8B65" if fm_int == 1 else "#E5531A"
            ax.add_patch(plt.Rectangle((idx - 0.39, y - 0.28), 0.78, 0.56, facecolor=face, edgecolor=edge, linewidth=1.1))
            ax.text(idx, y + 0.05, wrapped_label(api_id, 22), ha="center", va="center", fontweight="bold", fontsize=7.1)
            ax.text(idx, y - 0.22, f"FM={fm_int}", ha="center", va="center", color=edge, fontsize=7.2)
            if idx < subtask_count - 1:
                ax.annotate(
                    "",
                    xy=(idx + 0.59, y),
                    xytext=(idx + 0.41, y),
                    arrowprops={"arrowstyle": "-|>", "color": "#667085", "lw": 0.9},
                )

        metric = results[mode]
        bar_x = subtask_count + 0.28
        ax.text(bar_x, mode_count + 0.28, "Workflow scores", ha="left", va="center", fontweight="bold", color="#344054")
        bars = (
            ("FC", metric.functional_coverage, "#2563EB"),
            ("QoS", metric.normalized_qos, "#059669"),
            ("Score", metric.score, "#7C3AED"),
        )
        for offset, (label, value, color) in enumerate(bars):
            yy = y + 0.2 - offset * 0.19
            ax.text(bar_x, yy, label, ha="left", va="center", color="#475467", fontsize=7)
            ax.add_patch(plt.Rectangle((bar_x + 0.18, yy - 0.045), 0.58, 0.09, facecolor="#EDF2F7", edgecolor="#D0D7E2", linewidth=0.4))
            ax.add_patch(plt.Rectangle((bar_x + 0.18, yy - 0.045), 0.58 * max(0, min(1, value)), 0.09, facecolor=color, edgecolor=color, linewidth=0.4))
            ax.text(bar_x + 0.84, yy, f"{value:.3f}", ha="left", va="center", fontsize=6.8)

    ax.text(
        -0.72,
        -0.65,
        "Node fill indicates Functional Match label for the assigned subtask. Score bars are rounded visual summaries.",
        ha="left",
        va="center",
        color="#667085",
        fontsize=7,
    )
    out_dir = figures_dir(run_root, output_dir)
    output_path = out_dir / f"selected_api_path_{query_id}.png"
    fig.savefig(output_path, dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    args = parse_args()
    output = generate(args.run_folder, args.query, args.output_dir, args.png_dpi)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
