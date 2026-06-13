#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

from figure_data import (
    MODE_COLORS,
    MODE_LABELS,
    MODE_ORDER,
    figures_dir,
    group_results_by_query,
    load_all_results,
    load_pyplot,
    parse_query_ids,
    query_span_token,
    resolve_run_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate small-multiple query score panels for a comma-separated query list."
    )
    parser.add_argument("run_folder", type=Path, help="Run folder containing qXX_* result folders.")
    parser.add_argument(
        "--queries",
        required=True,
        help="Comma-separated query ids to include, for example q01,q02,q03 or 1,2,3.",
    )
    parser.add_argument("--output-dir", type=Path, help="Output directory. Defaults to <run_folder>/figures.")
    parser.add_argument("--png-dpi", type=int, default=300, help="PNG export resolution.")
    return parser.parse_args()


def generate(
    run_folder: str | Path,
    queries: str,
    output_dir: str | Path | None = None,
    png_dpi: int = 300,
) -> Path:
    run_root = resolve_run_root(run_folder)
    query_ids = parse_query_ids(queries)
    grouped = group_results_by_query(load_all_results(run_root))
    missing = [query_id for query_id in query_ids if query_id not in grouped]
    if missing:
        available = ", ".join(grouped)
        raise ValueError(f"Requested queries not found: {', '.join(missing)}. Available: {available}")

    plt = load_pyplot()
    plt.rcParams.update({"axes.titlesize": 18, "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10})
    columns = min(3, len(query_ids))
    rows = math.ceil(len(query_ids) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 3.9 * rows), sharey=True)
    if not isinstance(axes, (list, tuple)):
        import numpy as np

        axes = np.array(axes)
    axes_flat = list(axes.ravel())

    for ax, query_id in zip(axes_flat, query_ids):
        query_rows = grouped[query_id]
        values = [next(row.score for row in query_rows if row.mode == mode) for mode in MODE_ORDER]
        bars = ax.bar(
            [MODE_LABELS[mode] for mode in MODE_ORDER],
            values,
            color=[MODE_COLORS[mode] for mode in MODE_ORDER],
            edgecolor="#222222",
            linewidth=0.7,
        )
        ax.set_title(query_id, fontweight="bold", pad=8)
        ax.set_ylim(0.0, 1.06)
        ax.set_xlabel("Mode")
        ax.set_ylabel("Score (0-1)")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.75)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for tick in ax.get_xticklabels():
            tick.set_rotation(0)
            tick.set_ha("center")
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.025, 1.03),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    for ax in axes_flat[len(query_ids) :]:
        ax.axis("off")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=MODE_COLORS[mode], ec="#222222", label=MODE_LABELS[mode])
        for mode in MODE_ORDER
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_dir = figures_dir(run_root, output_dir)
    output = out_dir / f"query_score_panels_{query_span_token(query_ids)}.png"
    fig.savefig(output, dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> int:
    args = parse_args()
    print(generate(args.run_folder, args.queries, args.output_dir, args.png_dpi))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
