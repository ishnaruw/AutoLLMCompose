#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import plot_api_path_comparison
import plot_mode_summary
import plot_query_score_overview
import plot_query_score_panels
import plot_ranking_similarity
from figure_data import discover_query_runs, parse_query_id, resolve_run_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate all research figures supported by the plotting scripts for one run folder."
    )
    parser.add_argument("run_folder", type=Path, help="Run folder containing qXX_* result folders.")
    parser.add_argument(
        "--path-query",
        help="Query id for selected API path comparison. Defaults to the first discovered query.",
    )
    parser.add_argument(
        "--panel-queries",
        action="append",
        default=[],
        help=(
            "Comma-separated query ids for one query-panel output. "
            "May be repeated, for example --panel-queries q01,q02,q03 --panel-queries q04,q05."
        ),
    )
    parser.add_argument("--output-dir", type=Path, help="Output directory. Defaults to <run_folder>/figures.")
    parser.add_argument("--png-dpi", type=int, default=300, help="PNG export resolution.")
    return parser.parse_args()


def default_panel_groups(run_root: Path) -> list[str]:
    query_ids = [run.query_id for run in discover_query_runs(run_root)]
    groups = []
    for start in range(0, len(query_ids), 5):
        groups.append(",".join(query_ids[start : start + 5]))
    return groups


def main() -> int:
    args = parse_args()
    run_root = resolve_run_root(args.run_folder)
    path_query = parse_query_id(args.path_query) if args.path_query else discover_query_runs(run_root)[0].query_id
    panel_groups = args.panel_queries or default_panel_groups(run_root)

    outputs = []
    outputs.append(plot_api_path_comparison.generate(run_root, path_query, args.output_dir, args.png_dpi))
    outputs.extend(plot_mode_summary.generate(run_root, args.output_dir, args.png_dpi))
    outputs.append(plot_ranking_similarity.generate(run_root, args.output_dir, args.png_dpi))
    outputs.extend(plot_query_score_overview.generate(run_root, args.output_dir, args.png_dpi))
    for group in panel_groups:
        outputs.append(plot_query_score_panels.generate(run_root, group, args.output_dir, args.png_dpi))

    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
