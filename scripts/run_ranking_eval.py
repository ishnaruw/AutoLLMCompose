from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ranking_metrics import (  # noqa: E402
    DEFAULT_RBO_P,
    DEFAULT_INCLUSION_POLICY,
    MODE_ORDER,
    cases_to_frame,
    evaluate_parent_runs,
)


def _write_outputs(bundle, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric, matrix in bundle.matrices.items():
        matrix.to_csv(output_dir / f"{metric}_matrix.csv")
    for metric, matrix in bundle.pairwise_counts.items():
        matrix.to_csv(output_dir / f"{metric}_included_counts.csv")

    bundle.pairwise_scores.to_csv(output_dir / "pairwise_scores.csv", index=False)
    cases_to_frame(bundle.cases).to_csv(output_dir / "included_cases.csv", index=False)
    bundle.invalid_cases.to_csv(output_dir / "invalid_cases.csv", index=False)
    bundle.raw_rows.to_csv(output_dir / "loaded_rows.csv", index=False)
    (output_dir / "warnings.json").write_text(
        json.dumps(bundle.warnings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "inclusion_policy": bundle.inclusion_policy,
                "selected_modes": bundle.selected_modes,
                "included_cases": len(bundle.cases),
                "invalid_mode_subtask_cases": len(bundle.invalid_cases),
                "discovered_run_dirs": len(bundle.discovered_run_dirs),
                "loaded_reports": bundle.loaded_report_paths,
                "warnings": len(bundle.warnings),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate AutoLLMCompose mode-ranking agreement from completed query run outputs."
    )
    parser.add_argument(
        "parent_runs_dir",
        type=Path,
        help="Parent directory containing qXX_timestamp query run folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for outputs. Default: <parent_runs_dir>/ranking_eval.",
    )
    parser.add_argument(
        "--rbo-p",
        type=float,
        default=DEFAULT_RBO_P,
        help="RBO persistence parameter. Default: 0.9",
    )
    parser.add_argument(
        "--inclusion-policy",
        default=DEFAULT_INCLUSION_POLICY,
        help="Internal/debug option. Final reporting uses strict selected-mode evaluation.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=MODE_ORDER,
        choices=MODE_ORDER,
        help="Modes to include. Default: all four modes.",
    )
    args = parser.parse_args()

    parent_runs_dir = args.parent_runs_dir
    if not parent_runs_dir.is_absolute():
        parent_runs_dir = (PROJECT_ROOT / parent_runs_dir).resolve()
    if not parent_runs_dir.is_dir():
        raise FileNotFoundError(f"Missing experiment run folder: {parent_runs_dir}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = parent_runs_dir / "ranking_eval"
    elif not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()

    bundle = evaluate_parent_runs(
        parent_runs_dir,
        p=args.rbo_p,
        inclusion_policy=args.inclusion_policy,
        selected_modes=args.modes,
    )
    print(f"Evaluation inclusion policy: {bundle.inclusion_policy}")
    print(f"Selected modes: {', '.join(bundle.selected_modes)}")
    print(f"Discovered query run folders: {len(bundle.discovered_run_dirs)}")
    print(f"Loaded ranking reports: {len(bundle.loaded_report_paths)}")
    print(f"Included query/subtask cases: {len(bundle.cases)}")
    print(f"Excluded invalid mode/subtask cases: {len(bundle.invalid_cases)}")
    if "included_cases" in bundle.pairwise_scores.columns:
        print(f"Included pairwise comparisons: {int(bundle.pairwise_scores['included_cases'].sum())}")
    print(f"Warnings: {len(bundle.warnings)}")
    for warning in bundle.warnings[:12]:
        print(f"- {warning}")
    if len(bundle.warnings) > 12:
        print(f"- ... {len(bundle.warnings) - 12} more")

    for metric, matrix in bundle.matrices.items():
        print(f"\n{metric}")
        print(matrix.round(4).to_string())

    _write_outputs(bundle, output_dir)
    print(f"\nWrote ranking evaluation outputs to {output_dir}")


if __name__ == "__main__":
    main()
