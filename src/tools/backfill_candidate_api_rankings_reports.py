from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.config import CONFIG
from src.eval.candidate_api_rankings_excel import enrich_functional_match_rows_with_anomaly_flags, write_candidate_api_rankings_excel
from src.eval.audit_api_duplicates import collect_duplicate_audit_for_run
from src.eval.audit_api_hallucinations import collect_hallucination_audit_for_run

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


def _safe_load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_audit_json(path: Path) -> Dict[str, Any] | None:
    data = _safe_load_json(path)
    return data if isinstance(data, dict) else None


def _load_or_collect_duplicate_audit(run_dir: Path, eval_dir: Path, query_id: str) -> Dict[str, Any]:
    path = eval_dir / f"query_{query_id}_duplicate_audit.json"
    audit = _load_audit_json(path)
    return audit if audit is not None else collect_duplicate_audit_for_run(run_dir)


def _load_or_collect_hallucination_audit(run_dir: Path, eval_dir: Path, query_id: str) -> Dict[str, Any]:
    path = eval_dir / f"query_{query_id}_hallucination_audit.json"
    audit = _load_audit_json(path)
    return audit if audit is not None else collect_hallucination_audit_for_run(run_dir, CONFIG.catalog_no_qos_path)


def _backfill_single(run_path: Path, output_dir: Path | None) -> Path:
    run_dir = _normalize_run_dir(run_path)
    eval_dir = output_dir.resolve() if output_dir else _default_eval_dir(run_dir)
    query_id = _query_id_from_run_dir(run_dir)
    rows_path = eval_dir / f"query_{query_id}_candidate_api_rankings_rows.json"
    rows = _safe_load_json(rows_path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected list rows JSON at {rows_path}, got: {type(rows).__name__}")

    duplicate_audit = _load_or_collect_duplicate_audit(run_dir, eval_dir, query_id)
    hallucination_audit = _load_or_collect_hallucination_audit(run_dir, eval_dir, query_id)
    enriched_rows = enrich_functional_match_rows_with_anomaly_flags(
        rows,
        duplicate_audit=duplicate_audit,
        hallucination_audit=hallucination_audit,
    )

    eval_dir.mkdir(parents=True, exist_ok=True)
    rows_path.write_text(json.dumps(enriched_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    out_path = eval_dir / f"query_{query_id}_candidate_api_rankings.xlsx"
    write_candidate_api_rankings_excel(
        enriched_rows,
        out_path,
        duplicate_audit=duplicate_audit,
        hallucination_audit=hallucination_audit,
    )
    return out_path


def _iter_run_dirs(root_dir: Path) -> Iterable[Path]:
    root_dir = root_dir.resolve()
    if (root_dir / "0_decomposer.json").exists():
        yield root_dir
        return
    for child in sorted(root_dir.iterdir()):
        if child.is_dir() and RUN_DIR_PATTERN.match(child.name) and (child / "0_decomposer.json").exists():
            yield child.resolve()


def _backfill_root(root_dir: Path) -> List[Path]:
    written: List[Path] = []
    for run_dir in _iter_run_dirs(root_dir):
        written.append(_backfill_single(run_dir, None))
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill query_<id>_candidate_api_rankings.xlsx and rows JSON with anomaly flags for historical runs."
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    written: List[Path] = []

    if not args.run_dir and not args.root_dir:
        raise SystemExit("Provide at least one of --run-dir or --root-dir.")

    if args.output_dir and not args.run_dir:
        raise SystemExit("--output-dir can only be used together with --run-dir.")

    if args.run_dir:
        written.append(_backfill_single(args.run_dir, args.output_dir))

    if args.root_dir:
        written.extend(_backfill_root(args.root_dir))

    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
