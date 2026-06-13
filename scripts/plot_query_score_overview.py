#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from figure_data import (
    MODE_COLORS,
    MODE_ORDER,
    best_modes_for_query,
    figures_dir,
    group_results_by_query,
    load_all_results,
    load_pyplot,
    query_sort_key,
    resolve_run_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate grouped query-score bars and winner/tied-winner heatmap from a run folder."
    )
    parser.add_argument("run_folder", type=Path, help="Run folder containing qXX_* result folders.")
    parser.add_argument("--output-dir", type=Path, help="Output directory. Defaults to <run_folder>/figures.")
    parser.add_argument("--png-dpi", type=int, default=300, help="PNG export resolution.")
    return parser.parse_args()


def plot_grouped_scores(plt, grouped, out_dir: Path, png_dpi: int) -> Path:
    query_ids = sorted(grouped, key=query_sort_key)
    fig_width = max(10.0, 0.68 * len(query_ids) + 5.5)
    fig, ax = plt.subplots(figsize=(fig_width, 5.0), constrained_layout=True)
    x = list(range(len(query_ids)))
    width = 0.19
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for offset, mode in zip(offsets, MODE_ORDER):
        values = [next(row.score for row in grouped[query_id] if row.mode == mode) for query_id in query_ids]
        ax.bar(
            [pos + offset for pos in x],
            values,
            width,
            label=mode,
            color=MODE_COLORS[mode],
            edgecolor="#FFFFFF",
            linewidth=0.4,
        )
    ax.set_title("Grouped Query-Level Final Score by Mode", pad=10)
    ax.set_xlabel("Query ID")
    ax.set_ylabel("QoS-adjusted composition score")
    ax.set_xticks(x)
    ax.set_xticklabels(query_ids, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.08)
    ax.legend(loc="upper right", ncol=2, frameon=True)
    ax.set_axisbelow(True)
    ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    output = out_dir / "query_composition_scores.png"
    fig.savefig(output, dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_winner_heatmap(plt, grouped, out_dir: Path, png_dpi: int) -> Path:
    import numpy as np
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    query_ids = sorted(grouped, key=query_sort_key)
    matrix = []
    for query_id in query_ids:
        best_modes = best_modes_for_query(grouped[query_id])
        is_tie = len(best_modes) > 1
        matrix.append([2 if mode in best_modes and is_tie else 1 if mode in best_modes else 0 for mode in MODE_ORDER])

    fig_height = max(5.6, 0.34 * len(query_ids) + 2.2)
    fig, ax = plt.subplots(figsize=(9.2, fig_height), constrained_layout=True)
    cmap = ListedColormap(["#F3F4F6", "#54A24B", "#F58518"])
    ax.imshow(np.array(matrix), cmap=cmap, vmin=0, vmax=2, aspect="auto")
    ax.set_title("Winner or Tied-Winner Heatmap", pad=12)
    ax.set_xlabel("Mode")
    ax.set_ylabel("Query ID")
    ax.set_xticks(range(len(MODE_ORDER)))
    ax.set_xticklabels(MODE_ORDER, rotation=35, ha="right")
    ax.set_yticks(range(len(query_ids)))
    ax.set_yticklabels(query_ids)
    ax.set_xticks([idx - 0.5 for idx in range(1, len(MODE_ORDER))], minor=True)
    ax.set_yticks([idx - 0.5 for idx in range(1, len(query_ids))], minor=True)
    ax.grid(which="minor", color="#D0D5DD", linewidth=0.65, alpha=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)
    legend = (
        Patch(color="#F3F4F6", label="not best"),
        Patch(color="#54A24B", label="unique best"),
        Patch(color="#F58518", label="tied best"),
    )
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    output = out_dir / "winner_status_heatmap.png"
    fig.savefig(output, dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)
    return output


def generate(run_folder: str | Path, output_dir: str | Path | None = None, png_dpi: int = 300) -> list[Path]:
    run_root = resolve_run_root(run_folder)
    grouped = group_results_by_query(load_all_results(run_root))
    out_dir = figures_dir(run_root, output_dir)
    plt = load_pyplot()
    return [
        plot_grouped_scores(plt, grouped, out_dir, png_dpi),
        plot_winner_heatmap(plt, grouped, out_dir, png_dpi),
    ]


def main() -> int:
    args = parse_args()
    for output in generate(args.run_folder, args.output_dir, args.png_dpi):
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
