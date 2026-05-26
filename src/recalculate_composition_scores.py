#!/usr/bin/env python3
"""
Recalculate AutoLLMCompose composition-level QoS scores for existing run folders.

This script does NOT rerun decomposition, retrieval, ranking, selection, or planning.
It only reruns src.eval.composition_qos_eval.evaluate_composition_qos on saved run outputs,
then refreshes the composition-evaluation pointers/summaries in evaluation_result.json and meta.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PATH_KEYS_IN_SUMMARY = [
    "rows_json",
    "excel",
    "composition_validity_issues_json",
    "composition_validity_issues_log",
    "candidate_api_rankings_rows_json",
]

EVAL_RESULT_COMPOSITION_KEYS = {
    "rows_json": "composition_qos_eval_rows_json",
    "summary_json": "composition_qos_eval_summary_json",
    "excel": "composition_qos_eval_excel",
    "composition_validity_issues_json": "composition_validity_issues_json",
    "composition_validity_issues_log": "composition_validity_issues_log",
}


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


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def is_run_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if path.name.startswith(".") or path.name == "__MACOSX":
        return False
    return (path / "meta.json").exists() and (path / "0_decomposer.json").exists()


def discover_runs(runs_dir: Path, recursive: bool = False) -> List[Path]:
    if recursive:
        candidates = [p for p in runs_dir.rglob("*") if is_run_dir(p)]
    else:
        candidates = [p for p in runs_dir.iterdir() if is_run_dir(p)]
    return sorted(candidates, key=lambda p: p.name)


def get_query_id(run_dir: Path) -> str:
    meta = read_json(run_dir / "meta.json", {})
    qid = str(meta.get("query_id") or "").strip()
    if qid:
        return qid
    # Expected folder style: q06_20260525T162701
    name = run_dir.name
    return name.split("_", 1)[0] if "_" in name else name


def expected_eval_paths(run_dir: Path, query_id: str) -> List[Path]:
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


def backup_existing_files(run_dir: Path, query_id: str, stamp: str) -> Optional[Path]:
    backup_dir = run_dir / "evaluation" / f"recalc_backup_{stamp}"
    copied = 0
    for src in expected_eval_paths(run_dir, query_id):
        if not src.exists():
            continue
        # Keep metadata backups inside the same backup folder but preserve labels.
        if src.parent == run_dir:
            dst = backup_dir / f"root_{src.name}"
        else:
            dst = backup_dir / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return backup_dir if copied else None


def patch_summary_paths(summary_path: Path, project_root: Path) -> Dict[str, Any]:
    summary = read_json(summary_path, {})
    for key in PATH_KEYS_IN_SUMMARY:
        value = summary.get(key)
        if not value:
            continue
        p = Path(value)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        # Prefer stable repo-relative paths when possible.
        summary[key] = rel_to(p, project_root)
    write_json(summary_path, summary)
    return summary


def update_evaluation_result(run_dir: Path, result_paths: Dict[str, Path], summary: Dict[str, Any]) -> None:
    path = run_dir / "evaluation_result.json"
    payload = read_json(path, {})

    for result_key, eval_key in EVAL_RESULT_COMPOSITION_KEYS.items():
        result_path = result_paths.get(result_key)
        if result_path:
            payload[eval_key] = rel_to(result_path, run_dir)

    # Refresh composition-level summaries that dashboards may read.
    for key in [
        "composition_validity_summary",
        "score_summary",
        "summary_rows",
        "valid_mode_count",
        "complete_mode_count",
        "scoring_formula",
    ]:
        if key in summary:
            payload[key] = summary[key]

    # Preserve and refresh convenience best-mode keys emitted by the evaluator.
    for key, value in summary.items():
        if key.startswith("best_") or key.startswith("is_"):
            payload[key] = value

    payload["composition_qos_recalculated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload["composition_qos_recalculation_note"] = (
        "Recomputed from saved planner/selection outputs using current src.eval.composition_qos_eval."
    )
    write_json(path, payload)


def update_meta(run_dir: Path, result_paths: Dict[str, Path], summary: Dict[str, Any]) -> None:
    path = run_dir / "meta.json"
    meta = read_json(path, {})
    stages = meta.setdefault("timing", {}).setdefault("stages", {})
    comp_stage = stages.setdefault("composition_qos_evaluation", {})
    comp_stage["status"] = "recalculated"
    comp_stage["recalculated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    comp_stage["rows_json"] = rel_to(result_paths["rows_json"], run_dir)
    comp_stage["summary_json"] = rel_to(result_paths["summary_json"], run_dir)
    comp_stage["excel"] = rel_to(result_paths["excel"], run_dir)
    comp_stage["issues_json"] = rel_to(result_paths["composition_validity_issues_json"], run_dir)
    comp_stage["issues_log"] = rel_to(result_paths["composition_validity_issues_log"], run_dir)
    comp_stage["composition_validity_summary"] = summary.get("composition_validity_summary")
    comp_stage["score_summary"] = summary.get("score_summary")
    meta["composition_qos_eval_recalculated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_json(path, meta)


def append_run_log(run_dir: Path, summary: Dict[str, Any]) -> None:
    score_summary = summary.get("score_summary") or {}
    best_modes = score_summary.get("best_modes_by_qos_adjusted_composition_score") or summary.get(
        "best_modes_by_qos_adjusted_composition_score"
    )
    if isinstance(best_modes, list):
        best_text = ",".join(best_modes)
    else:
        best_text = str(best_modes or "unknown")
    line = (
        f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] "
        f"composition QoS scores recalculated with current evaluator; "
        f"best_mode_by_qos_adjusted_composition_score={best_text}\n"
    )
    with (run_dir / "run.log").open("a", encoding="utf-8") as f:
        f.write(line)


def summarize_scores(summary: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for row in summary.get("summary_rows", []) or []:
        mode = row.get("Mode")
        if not mode:
            continue
        out[f"{mode}_score"] = row.get("QoS_Adjusted_Composition_Score")
        out[f"{mode}_rank"] = row.get("Rank_By_QoS_Adjusted_Score")
    score_summary = summary.get("score_summary", {}) or {}
    best_modes = score_summary.get("best_modes_by_qos_adjusted_composition_score")
    out["best_modes"] = ",".join(best_modes) if isinstance(best_modes, list) else best_modes
    out["tie"] = score_summary.get("is_qos_adjusted_composition_score_tie")
    return out


def run_one(run_dir: Path, project_root: Path, backup: bool, stamp: str, dry_run: bool, write_log: bool) -> Dict[str, Any]:
    query_id = get_query_id(run_dir)
    eval_dir = run_dir / "evaluation"
    candidate_rows = eval_dir / f"query_{query_id}_candidate_api_rankings_rows.json"
    if not candidate_rows.exists():
        raise FileNotFoundError(f"Missing candidate ranking rows: {candidate_rows}")

    if dry_run:
        return {
            "run_dir": str(run_dir),
            "query_id": query_id,
            "status": "dry_run_ok",
            "candidate_rows": str(candidate_rows),
        }

    backup_dir = backup_existing_files(run_dir, query_id, stamp) if backup else None

    from src.eval.composition_qos_eval import evaluate_composition_qos

    result = evaluate_composition_qos(query_dir=run_dir, query_id=query_id, output_dir=eval_dir)
    result_paths = {
        "rows_json": Path(result["rows_json"]),
        "summary_json": Path(result["summary_json"]),
        "excel": Path(result["excel"]),
        "composition_validity_issues_json": Path(result["composition_validity_issues_json"]),
        "composition_validity_issues_log": Path(result["composition_validity_issues_log"]),
    }
    summary = patch_summary_paths(result_paths["summary_json"], project_root)
    update_evaluation_result(run_dir, result_paths, summary)
    update_meta(run_dir, result_paths, summary)
    if write_log:
        append_run_log(run_dir, summary)

    row = {
        "run_dir": str(run_dir),
        "query_id": query_id,
        "status": "updated",
        "backup_dir": str(backup_dir) if backup_dir else "",
        "summary_json": str(result_paths["summary_json"]),
        "rows_json": str(result_paths["rows_json"]),
        "excel": str(result_paths["excel"]),
    }
    row.update(summarize_scores(summary))
    return row


def write_report(report_json: Path, rows: List[Dict[str, Any]]) -> None:
    write_json(report_json, rows)
    csv_path = report_json.with_suffix(".csv")
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recalculate composition QoS scores for existing AutoLLMCompose run directories."
    )
    parser.add_argument("--runs-dir", required=True, type=Path, help="Directory containing qXX_* run folders.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="AutoLLMCompose project root containing src/. Default: current working directory.",
    )
    parser.add_argument("--recursive", action="store_true", help="Search recursively for run folders.")
    parser.add_argument("--only", nargs="*", help="Optional run folder names or query IDs to include, e.g. q06 q18_...")
    parser.add_argument("--dry-run", action="store_true", help="List runs that would be processed without writing files.")
    parser.add_argument("--no-backup", action="store_true", help="Do not backup old eval/metadata files before overwriting.")
    parser.add_argument("--no-run-log", action="store_true", help="Do not append recalculation note to each run.log.")
    parser.add_argument("--report-name", default=None, help="Optional report filename under runs-dir.")
    args = parser.parse_args()

    runs_dir = args.runs_dir.expanduser().resolve()
    project_root = args.project_root.expanduser().resolve()
    src_dir = project_root / "src"
    if not src_dir.exists():
        print(f"ERROR: project root does not contain src/: {project_root}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(project_root))

    runs = discover_runs(runs_dir, recursive=args.recursive)
    if args.only:
        wanted = set(args.only)
        runs = [r for r in runs if r.name in wanted or get_query_id(r) in wanted]
    if not runs:
        print(f"No run directories found under: {runs_dir}")
        return 1

    stamp = now_stamp()
    report_rows: List[Dict[str, Any]] = []
    for run_dir in runs:
        try:
            row = run_one(
                run_dir=run_dir,
                project_root=project_root,
                backup=not args.no_backup,
                stamp=stamp,
                dry_run=args.dry_run,
                write_log=not args.no_run_log,
            )
            print(f"[{row['status']}] {run_dir.name} ({row.get('query_id')})")
            report_rows.append(row)
        except Exception as exc:
            row = {
                "run_dir": str(run_dir),
                "query_id": get_query_id(run_dir),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[error] {run_dir.name}: {row['error']}", file=sys.stderr)
            report_rows.append(row)

    report_name = args.report_name or f"composition_recalc_report_{stamp}.json"
    report_path = runs_dir / report_name
    if not args.dry_run:
        write_report(report_path, report_rows)
        print(f"\nWrote report: {report_path}")
        print(f"Wrote CSV:    {report_path.with_suffix('.csv')}")
    else:
        print("\nDry run only. No files were changed.")
    return 0 if all(row.get("status") != "error" for row in report_rows) else 3


if __name__ == "__main__":
    raise SystemExit(main())
