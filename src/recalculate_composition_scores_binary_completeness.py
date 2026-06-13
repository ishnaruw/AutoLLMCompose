#!/usr/bin/env python3
"""
Recalculate composition-level scores for existing AutoLLMCompose runs using the
current binary-completeness formula.

This script updates the files that the Streamlit dashboard reads:

  Ranking Evaluation page:
    - evaluation/query_<qid>_composition_qos_eval_rows.json
    - evaluation/query_<qid>_composition_qos_eval.xlsx, Planned_Workflow sheet

  Composition Visualizations page:
    - evaluation/query_<qid>_composition_qos_eval_rows.json
    - evaluation/query_<qid>_composition_qos_eval.xlsx, Planned_Workflow sheet

It also refreshes related run metadata:
    - evaluation/query_<qid>_composition_qos_eval_summary.json
    - evaluation/query_<qid>_composition_validity_issues.json
    - evaluation/query_<qid>_composition_validity_issues.log
    - evaluation_result.json
    - meta.json
    - run.log, append-only note

It does not rerun decomposition, retrieval, functional refinement, ranking,
selection, or planning. It only reruns:
    src.eval.composition_qos_eval.evaluate_composition_qos()

Current scoring formula expected from the evaluator:
    QoS_Adjusted_Composition_Score =
        Composition_Completeness * (
            0.7 * Functional_Coverage + 0.3 * Normalized_QoS_Score
        )

where Composition_Completeness is binary.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

COMPOSITION_ARTIFACT_KEYS = {
    "rows_json": "composition_qos_eval_rows_json",
    "summary_json": "composition_qos_eval_summary_json",
    "excel": "composition_qos_eval_excel",
    "composition_validity_issues_json": "composition_validity_issues_json",
    "composition_validity_issues_log": "composition_validity_issues_log",
}

SUMMARY_KEYS_TO_MIRROR = [
    "composition_validity_summary",
    "score_summary",
    "summary_rows",
    "scoring_formula",
    "valid_mode_count",
    "complete_mode_count",
    "best_modes_by_qos_adjusted_composition_score",
    "best_qos_adjusted_composition_score",
    "best_mode_by_qos_adjusted_composition_score",
    "is_qos_adjusted_composition_score_tie",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: Path, default: Any = None) -> Any:
    if default is None:
        default = {}
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def rel_to(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def is_run_dir(path: Path) -> bool:
    if not path.is_dir() or path.name.startswith(".") or path.name == "__MACOSX":
        return False
    return (path / "meta.json").exists() and (path / "0_decomposer.json").exists()


def discover_runs(target: Path, recursive: bool = False) -> list[Path]:
    target = target.expanduser().resolve()
    if is_run_dir(target):
        return [target]
    if not target.exists():
        raise FileNotFoundError(f"Runs directory does not exist: {target}")
    runs = [p for p in (target.rglob("*") if recursive else target.iterdir()) if is_run_dir(p)]
    return sorted(runs, key=lambda p: p.name)


def get_query_id(run_dir: Path) -> str:
    meta = read_json(run_dir / "meta.json", {})
    qid = str(meta.get("query_id") or "").strip()
    if qid:
        return qid
    return run_dir.name.split("_", 1)[0] if "_" in run_dir.name else run_dir.name


def expected_artifacts(run_dir: Path, query_id: str) -> list[Path]:
    eval_dir = run_dir / "evaluation"
    return [
        eval_dir / f"query_{query_id}_composition_qos_eval_rows.json",
        eval_dir / f"query_{query_id}_composition_qos_eval_summary.json",
        eval_dir / f"query_{query_id}_composition_qos_eval.xlsx",
        eval_dir / f"query_{query_id}_composition_validity_issues.json",
        eval_dir / f"query_{query_id}_composition_validity_issues.log",
        run_dir / "evaluation_result.json",
        run_dir / "meta.json",
    ]


def backup_existing(run_dir: Path, query_id: str, run_stamp: str) -> str:
    backup_dir = run_dir / "evaluation" / f"score_recalc_backup_{run_stamp}"
    copied = 0
    for src in expected_artifacts(run_dir, query_id):
        if not src.exists():
            continue
        dst = backup_dir / (f"root_{src.name}" if src.parent == run_dir else src.name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return str(backup_dir) if copied else ""


def normalize_result_paths(result: dict[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for key in COMPOSITION_ARTIFACT_KEYS:
        value = result.get(key)
        if not value:
            raise KeyError(f"Evaluator did not return {key}")
        paths[key] = Path(value).expanduser().resolve()
    return paths


def patch_summary_file(summary_path: Path, run_dir: Path, project_root: Path) -> dict[str, Any]:
    summary = read_json(summary_path, {})
    for key in [
        "rows_json",
        "excel",
        "composition_validity_issues_json",
        "composition_validity_issues_log",
        "candidate_api_rankings_rows_json",
    ]:
        value = summary.get(key)
        if not value:
            continue
        p = Path(value)
        if not p.is_absolute():
            candidate = (run_dir / p).resolve()
            p = candidate if candidate.exists() else (project_root / p).resolve()
        summary[key] = rel_to(p, run_dir)

    summary["composition_score_recalculated_at_utc"] = utc_now()
    summary["composition_score_recalculation_formula"] = (
        "Composition_Completeness * (0.7 * Functional_Coverage + 0.3 * Normalized_QoS_Score)"
    )
    write_json(summary_path, summary)
    return summary


def scores_by_mode(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        mode = str(row.get("Mode") or "").strip()
        if not mode:
            continue
        out[mode] = {
            "QoS_Adjusted_Composition_Score": row.get("QoS_Adjusted_Composition_Score"),
            "Composition_Completeness": row.get("Composition_Completeness"),
            "Functional_Coverage": row.get("Functional_Coverage"),
            "Normalized_QoS_Score": row.get("Normalized_QoS_Score"),
            "Composition_Validity": row.get("Composition_Validity"),
            "Total_Response_Time_s": row.get("Total_Response_Time_s"),
            "Bottleneck_Throughput_kbps": row.get("Bottleneck_Throughput_kbps"),
            "Average_Workflow_Availability": row.get("Average_Workflow_Availability"),
        }
    return out


def patch_evaluation_result(run_dir: Path, result_paths: dict[str, Path], summary: dict[str, Any]) -> None:
    path = run_dir / "evaluation_result.json"
    payload = read_json(path, {})

    for result_key, eval_result_key in COMPOSITION_ARTIFACT_KEYS.items():
        payload[eval_result_key] = rel_to(result_paths[result_key], run_dir)

    for key in SUMMARY_KEYS_TO_MIRROR:
        if key in summary:
            payload[key] = summary[key]

    rows = read_json(result_paths["rows_json"], [])
    if isinstance(rows, list):
        payload["composition_scores_by_mode"] = scores_by_mode(rows)

    score_summary = summary.get("score_summary") if isinstance(summary.get("score_summary"), dict) else {}
    for key, value in score_summary.items():
        if key.startswith("best_") or key.startswith("is_") or key.endswith("_count") or key == "scoring_formula":
            payload[key] = value

    payload["composition_score_recalculated_at_utc"] = utc_now()
    payload["composition_score_recalculation_formula"] = (
        "Composition_Completeness * (0.7 * Functional_Coverage + 0.3 * Normalized_QoS_Score)"
    )
    payload["composition_score_recalculation_note"] = (
        "Dashboard-facing composition rows, summary, Excel report, evaluation_result.json, and meta.json were refreshed from saved run artifacts."
    )
    write_json(path, payload)


def patch_meta(run_dir: Path, result_paths: dict[str, Path], summary: dict[str, Any]) -> None:
    path = run_dir / "meta.json"
    meta = read_json(path, {})

    timing = meta.get("timing")
    if not isinstance(timing, dict):
        timing = {}
        meta["timing"] = timing
    stages = timing.get("stages")
    if not isinstance(stages, dict):
        stages = {}
        timing["stages"] = stages

    stage = stages.get("composition_qos_evaluation")
    if not isinstance(stage, dict):
        stage = {}
        stages["composition_qos_evaluation"] = stage

    stage.update(
        {
            "status": "recalculated",
            "recalculated_at_utc": utc_now(),
            "rows_json": rel_to(result_paths["rows_json"], run_dir),
            "summary_json": rel_to(result_paths["summary_json"], run_dir),
            "excel": rel_to(result_paths["excel"], run_dir),
            "issues_json": rel_to(result_paths["composition_validity_issues_json"], run_dir),
            "issues_log": rel_to(result_paths["composition_validity_issues_log"], run_dir),
            "score_summary": summary.get("score_summary"),
            "composition_validity_summary": summary.get("composition_validity_summary"),
            "scoring_formula": summary.get("scoring_formula"),
        }
    )

    meta["composition_score_recalculated_at_utc"] = utc_now()
    meta["composition_score_recalculation_formula"] = (
        "Composition_Completeness * (0.7 * Functional_Coverage + 0.3 * Normalized_QoS_Score)"
    )
    write_json(path, meta)


def append_run_log(run_dir: Path, summary: dict[str, Any]) -> None:
    best = summary.get("best_modes_by_qos_adjusted_composition_score")
    if not best and isinstance(summary.get("score_summary"), dict):
        best = summary["score_summary"].get("best_modes_by_qos_adjusted_composition_score")
    best_text = ",".join(str(x) for x in best) if isinstance(best, list) else str(best or "unknown")
    with (run_dir / "run.log").open("a", encoding="utf-8") as f:
        f.write(
            f"[{utc_now()}] recalculated composition scores with binary completeness formula; "
            f"best_modes_by_qos_adjusted_composition_score={best_text}\n"
        )


def verify_dashboard_rows(rows_path: Path) -> dict[str, Any]:
    rows = read_json(rows_path, [])
    if not isinstance(rows, list):
        raise ValueError(f"Dashboard rows file is not a list: {rows_path}")
    score_map = scores_by_mode(rows)
    return {
        "row_count": len(rows),
        "dashboard_rows_json": str(rows_path),
        "no_qos_score": score_map.get("no_qos", {}).get("QoS_Adjusted_Composition_Score"),
        "qos_pure_llm_score": score_map.get("qos_pure_llm", {}).get("QoS_Adjusted_Composition_Score"),
        "qos_topsis_score": score_map.get("qos_topsis", {}).get("QoS_Adjusted_Composition_Score"),
        "qos_hybrid_score": score_map.get("qos_hybrid", {}).get("QoS_Adjusted_Composition_Score"),
        "qos_topsis_completeness": score_map.get("qos_topsis", {}).get("Composition_Completeness"),
        "qos_topsis_functional_coverage": score_map.get("qos_topsis", {}).get("Functional_Coverage"),
        "qos_topsis_normalized_qos": score_map.get("qos_topsis", {}).get("Normalized_QoS_Score"),
    }


def process_run(
    run_dir: Path,
    *,
    project_root: Path,
    run_stamp: str,
    dry_run: bool,
    make_backup: bool,
    write_run_log: bool,
) -> dict[str, Any]:
    query_id = get_query_id(run_dir)
    eval_dir = run_dir / "evaluation"
    candidate_rows = eval_dir / f"query_{query_id}_candidate_api_rankings_rows.json"
    if not candidate_rows.exists():
        raise FileNotFoundError(f"Missing candidate ranking rows: {candidate_rows}")

    if dry_run:
        return {
            "status": "dry_run_ok",
            "run_name": run_dir.name,
            "query_id": query_id,
            "run_dir": str(run_dir),
            "dashboard_rows_json": str(eval_dir / f"query_{query_id}_composition_qos_eval_rows.json"),
        }

    backup_dir = backup_existing(run_dir, query_id, run_stamp) if make_backup else ""

    from src.eval.composition_qos_eval import evaluate_composition_qos

    result = evaluate_composition_qos(query_dir=run_dir, query_id=query_id, output_dir=eval_dir)
    result_paths = normalize_result_paths(result)
    summary = patch_summary_file(result_paths["summary_json"], run_dir, project_root)
    patch_evaluation_result(run_dir, result_paths, summary)
    patch_meta(run_dir, result_paths, summary)
    if write_run_log:
        append_run_log(run_dir, summary)

    verification = verify_dashboard_rows(result_paths["rows_json"])
    return {
        "status": "updated",
        "run_name": run_dir.name,
        "query_id": query_id,
        "run_dir": str(run_dir),
        "backup_dir": backup_dir,
        "summary_json": str(result_paths["summary_json"]),
        "excel": str(result_paths["excel"]),
        **verification,
    }


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    write_json(path, rows)
    csv_path = path.with_suffix(".csv")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh existing run composition scores using the binary-completeness formula."
    )
    parser.add_argument("--runs-dir", required=True, type=Path, help="Parent directory of qXX_* run folders, or one run folder.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Project root containing src/. Default: current directory.")
    parser.add_argument("--recursive", action="store_true", help="Search recursively for run folders.")
    parser.add_argument("--only", nargs="*", help="Optional query IDs or exact run folder names, e.g. q06 q18_20260525T083005.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed without writing files.")
    parser.add_argument("--no-backup", action="store_true", help="Do not back up old composition artifacts.")
    parser.add_argument("--no-run-log", action="store_true", help="Do not append a note to run.log.")
    parser.add_argument("--report-name", default=None, help="Optional report JSON filename written under runs-dir.")
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    runs_dir = args.runs_dir.expanduser().resolve()
    if not (project_root / "src").exists():
        print(f"ERROR: project root does not contain src/: {project_root}", file=sys.stderr)
        return 2
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        runs = discover_runs(runs_dir, recursive=args.recursive)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.only:
        wanted = set(args.only)
        runs = [r for r in runs if r.name in wanted or get_query_id(r) in wanted]

    if not runs:
        print(f"No run directories found under: {runs_dir}", file=sys.stderr)
        return 1

    run_stamp = stamp()
    report_rows: list[dict[str, Any]] = []

    for run_dir in runs:
        try:
            row = process_run(
                run_dir,
                project_root=project_root,
                run_stamp=run_stamp,
                dry_run=args.dry_run,
                make_backup=not args.no_backup,
                write_run_log=not args.no_run_log,
            )
            report_rows.append(row)
            print(
                f"[{row['status']}] {row['run_name']} "
                f"no_qos={row.get('no_qos_score')} "
                f"qos_pure_llm={row.get('qos_pure_llm_score')} "
                f"qos_topsis={row.get('qos_topsis_score')} "
                f"qos_hybrid={row.get('qos_hybrid_score')}"
            )
        except Exception as exc:
            row = {
                "status": "error",
                "run_name": run_dir.name,
                "query_id": get_query_id(run_dir),
                "run_dir": str(run_dir),
                "error": f"{type(exc).__name__}: {exc}",
            }
            report_rows.append(row)
            print(f"[error] {run_dir.name}: {row['error']}", file=sys.stderr)

    if args.dry_run:
        print("\nDry run only. No files were changed.")
        return 0 if all(row.get("status") != "error" for row in report_rows) else 3

    report_base = runs_dir if runs_dir.is_dir() else runs_dir.parent
    report_name = args.report_name or f"composition_binary_score_refresh_report_{run_stamp}.json"
    report_path = report_base / report_name
    latest_path = report_base / "composition_binary_score_refresh_report_latest.json"
    write_report(report_path, report_rows)
    write_report(latest_path, report_rows)

    print(f"\nWrote report: {report_path}")
    print(f"Wrote CSV:    {report_path.with_suffix('.csv')}")
    print(f"Wrote latest: {latest_path}")
    print(f"Wrote latest CSV: {latest_path.with_suffix('.csv')}")
    print("\nStreamlit reads evaluation/query_*_composition_qos_eval_rows.json for score values.")
    print("After running, click 'Reload reports' and 'Reload visualization data', or restart Streamlit if cached values remain.")

    return 0 if all(row.get("status") != "error" for row in report_rows) else 3


if __name__ == "__main__":
    raise SystemExit(main())
