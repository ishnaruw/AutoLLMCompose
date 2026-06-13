#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from figure_data import MODE_ORDER, figures_dir, load_csv_matrix, load_pyplot, resolve_run_root


MATRICES = (
    ("Spearman", "spearman_matrix.csv"),
    ("Average Overlap@K", "average_overlap_matrix.csv"),
    ("RBO", "rbo_matrix.csv"),
    ("Jaccard@K", "jaccard_matrix.csv"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ranking-similarity heatmaps from <run_folder>/ranking_eval.")
    parser.add_argument("run_folder", type=Path, help="Run folder containing ranking_eval matrix CSV files.")
    parser.add_argument("--output-dir", type=Path, help="Output directory. Defaults to <run_folder>/figures.")
    parser.add_argument("--png-dpi", type=int, default=300, help="PNG export resolution.")
    return parser.parse_args()


def generate(run_folder: str | Path, output_dir: str | Path | None = None, png_dpi: int = 300) -> Path:
    run_root = resolve_run_root(run_folder)
    ranking_dir = run_root / "ranking_eval"
    if not ranking_dir.exists():
        raise FileNotFoundError(f"Missing ranking_eval directory: {ranking_dir}")

    plt = load_pyplot()
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.8), constrained_layout=True)
    image = None
    for ax, (title, filename) in zip(axes.ravel(), MATRICES):
        modes, matrix = load_csv_matrix(ranking_dir / filename)
        values = np.array(matrix, dtype=float)
        image = ax.imshow(values, cmap="YlGnBu", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels(modes, rotation=35, ha="right")
        ax.set_yticks(range(len(modes)))
        ax.set_yticklabels(modes)
        ax.set_xticks([idx - 0.5 for idx in range(1, len(modes))], minor=True)
        ax.set_yticks([idx - 0.5 for idx in range(1, len(modes))], minor=True)
        ax.grid(which="minor", color="#FFFFFF", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row_index in range(len(MODE_ORDER)):
            for col_index in range(len(MODE_ORDER)):
                value = values[row_index, col_index]
                ax.text(
                    col_index,
                    row_index,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color="white" if value >= 0.75 else "#222222",
                    fontweight="bold" if row_index == col_index else "normal",
                    fontsize=9,
                )
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.86, pad=0.025)
    colorbar.set_label("Similarity")
    out_dir = figures_dir(run_root, output_dir)
    output = out_dir / "ranking_similarity_heatmaps.png"
    fig.savefig(output, dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> int:
    args = parse_args()
    print(generate(args.run_folder, args.output_dir, args.png_dpi))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
