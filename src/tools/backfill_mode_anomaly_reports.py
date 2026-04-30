from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.config import CONFIG
from src.eval.audit_api_duplicates import collect_duplicate_audit_for_run
from src.eval.audit_api_hallucinations import collect_hallucination_audit_for_run
from src.eval.mode_anomaly_report import collect_ranking_anomaly_audit_for_run, write_mode_anomaly_excel

RUN_DIR_PATTERN = re.compile(r"^q\d+_\d{8}T\d{6}$", flags=re.IGNORECASE)


def _normalize_run_dir(path: Path) -> Path:
    path = path.resolve()
    if path.is_dir() and (path / "0_decomposer.json").exists():
        return path
    if path.name in {"evaluation", "functional_match_eval"} and path.parent.is_dir() and (path.parent / "0_decomposer.json").exists():
        return path.parent.resolve()
    raise ValueError(f"Expected a query run directory or its evaluation/functional_match_eval directory, got: {path}")


def _query_id_from_run_dir(run_dir: Path) -> str:
    return run_dir.name.split("_", 1)[0]


def _default_eval_dir(run_dir: Path) -> Path:
    for name in ("evaluation", "functional_match_eval"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    return run_dir / "evaluation"


def _safe_load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _provider_root_for_run(run_dir: Path) -> Path:
    return run_dir.resolve().parent


def _find_aggregate_audit_json(provider_root: Path, filename: str) -> Optional[Path]:
    candidates = [
        provider_root / filename,
        provider_root.parent / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _filter_duplicate_audit_for_run(audit: Dict[str, Any], run_dir_name: str) -> Dict[str, Any]:
    return {
        "summary_rows": [
            row for row in audit.get("summary_rows", [])
            if isinstance(row, dict) and str(row.get("Run_Dir", "")).strip() == run_dir_name
        ],
        "duplicate_rows": [
            row for row in audit.get("duplicate_rows", [])
            if isinstance(row, dict) and str(row.get("Run_Dir", "")).strip() == run_dir_name
        ],
    }


def _filter_hallucination_audit_for_run(
    audit: Dict[str, Any],
    run_dir_name: str,
    provider_label: str,
) -> Dict[str, Any]:
    def _matches(row: Any) -> bool:
        if not isinstance(row, dict):
            return False
        if str(row.get("Run_Dir", "")).strip() != run_dir_name:
            return False
        provider_value = str(row.get("Provider_Dir", "")).strip()
        return not provider_value or provider_value == provider_label

    return {
        "mode_summary_rows": [row for row in audit.get("mode_summary_rows", []) if _matches(row)],
        "mode_detail_rows": [row for row in audit.get("mode_detail_rows", []) if _matches(row)],
    }


def _load_aggregate_audits_for_provider(provider_root: Path) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    duplicate_path = _find_aggregate_audit_json(provider_root, "api_duplicate_audit.json")
    hallucination_path = _find_aggregate_audit_json(provider_root, "api_hallucination_audit.json")
    duplicate_audit = _safe_load_json(duplicate_path) if duplicate_path else None
    hallucination_audit = _safe_load_json(hallucination_path) if hallucination_path else None
    return duplicate_audit, hallucination_audit


def _iter_run_dirs(root_dir: Path) -> Iterable[Path]:
    root_dir = root_dir.resolve()
    if (root_dir / "0_decomposer.json").exists():
        yield root_dir
        return
    for child in sorted(root_dir.iterdir()):
        if child.is_dir() and RUN_DIR_PATTERN.match(child.name) and (child / "0_decomposer.json").exists():
            yield child.resolve()


def _output_dir_for_run(run_dir: Path, write_into: str) -> Path:
    return run_dir if write_into == "run_dir" else _default_eval_dir(run_dir)


def _report_path(run_dir: Path, output_dir: Path) -> Path:
    return output_dir / f"query_{_query_id_from_run_dir(run_dir)}_mode_anomalies.xlsx"


def _build_report_for_run(
    run_dir: Path,
    output_dir: Path,
    *,
    aggregate_duplicate_audit: Optional[Dict[str, Any]] = None,
    aggregate_hallucination_audit: Optional[Dict[str, Any]] = None,
    provider_label: Optional[str] = None,
) -> Path:
    if aggregate_duplicate_audit is not None:
        duplicate_audit = _filter_duplicate_audit_for_run(aggregate_duplicate_audit, run_dir.name)
    else:
        duplicate_audit = collect_duplicate_audit_for_run(run_dir)

    if aggregate_hallucination_audit is not None and provider_label:
        hallucination_audit = _filter_hallucination_audit_for_run(
            aggregate_hallucination_audit,
            run_dir.name,
            provider_label,
        )
    else:
        hallucination_audit = collect_hallucination_audit_for_run(run_dir, CONFIG.catalog_no_qos_path)

    ranking_anomaly_audit = collect_ranking_anomaly_audit_for_run(run_dir, query_id=_query_id_from_run_dir(run_dir))
    out_path = _report_path(run_dir, output_dir)
    return write_mode_anomaly_excel(duplicate_audit, hallucination_audit, out_path, ranking_anomaly_audit)


def _backfill_single(run_path: Path, output_dir: Path | None) -> List[Path]:
    run_dir = _normalize_run_dir(run_path)
    target_dir = output_dir.resolve() if output_dir else _default_eval_dir(run_dir)
    provider_root = _provider_root_for_run(run_dir)
    provider_label = provider_root.name
    aggregate_duplicate_audit, aggregate_hallucination_audit = _load_aggregate_audits_for_provider(provider_root)
    return [
        _build_report_for_run(
            run_dir,
            target_dir,
            aggregate_duplicate_audit=aggregate_duplicate_audit,
            aggregate_hallucination_audit=aggregate_hallucination_audit,
            provider_label=provider_label,
        )
    ]


def _backfill_root(root_dir: Path, write_into: str) -> List[Path]:
    root_dir = root_dir.resolve()
    provider_label = root_dir.name
    aggregate_duplicate_audit, aggregate_hallucination_audit = _load_aggregate_audits_for_provider(root_dir)
    written: List[Path] = []
    for run_dir in _iter_run_dirs(root_dir):
        target_dir = _output_dir_for_run(run_dir, write_into)
        written.append(
            _build_report_for_run(
                run_dir,
                target_dir,
                aggregate_duplicate_audit=aggregate_duplicate_audit,
                aggregate_hallucination_audit=aggregate_hallucination_audit,
                provider_label=provider_label,
            )
        )
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill query_<id>_mode_anomalies.xlsx reports for historical runs."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Single query run directory, or its evaluation/functional_match_eval directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory to write the single-run report into.",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        help="Provider/root directory containing multiple query runs to backfill.",
    )
    parser.add_argument(
        "--write-into",
        choices=["eval_dir", "run_dir"],
        default="eval_dir",
        help="For --root-dir backfills, write each report into the run's evaluation/functional_match_eval dir or run dir.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    written: List[Path] = []

    if not args.run_dir and not args.root_dir:
        raise SystemExit("Provide at least one of --run-dir or --root-dir.")

    if args.output_dir and not args.run_dir:
        raise SystemExit("--output-dir can only be used together with --run-dir.")

    if args.run_dir:
        written.extend(_backfill_single(args.run_dir, args.output_dir))

    if args.root_dir:
        written.extend(_backfill_root(args.root_dir, args.write_into))

    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
