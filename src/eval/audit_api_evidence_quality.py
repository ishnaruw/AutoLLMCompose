from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.core.api_formatting import _as_dict, _find_endpoint_detail, _load_tool_json, normalize_api_for_ranking


GENERIC_DESCRIPTION_PHRASES = {
    "api endpoint",
    "get data",
    "get information",
    "returns data",
    "returns information",
    "search api",
    "this endpoint",
}


def _pct(count: int, total: int) -> float:
    return round((float(count) / float(total) * 100.0), 2) if total else 0.0


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalized_for_match(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _is_generic_description(description: str) -> bool:
    compact = _normalized_for_match(description)
    if not compact:
        return False
    if compact in GENERIC_DESCRIPTION_PHRASES:
        return True
    if len(compact) <= 80 and any(phrase in compact for phrase in GENERIC_DESCRIPTION_PHRASES):
        return True
    return bool(
        re.fullmatch(
            r"(get|fetch|retrieve|return|returns|search|list) "
            r"(api|data|details|information|results|items|content)",
            compact,
        )
    )


def _safe_read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _iter_json_records(data: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
        return

    if not isinstance(data, dict):
        return

    for key in ("apis", "candidates", "items", "results", "ranked"):
        value = data.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yield item
            return

    dict_values = [value for value in data.values() if isinstance(value, dict)]
    if dict_values and len(dict_values) == len(data):
        for key, value in data.items():
            row = dict(value)
            row.setdefault("api_id", key)
            yield row
        return

    yield data


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if isinstance(item, dict):
                item = dict(item)
                item.setdefault("_source_line", line_no)
                yield item


def _iter_input_records(path: Path) -> Iterable[Dict[str, Any]]:
    if path.is_dir():
        files = sorted(path.rglob("1_retriever_s*.json"))
        if not files:
            files = sorted(path.rglob("2_ranked_s*.json"))
        for file_path in files:
            data = _safe_read_json(file_path)
            for item in _iter_json_records(data):
                item = dict(item)
                item.setdefault("_source_path", str(file_path))
                yield item
        return

    if path.suffix.lower() == ".jsonl":
        yield from _iter_jsonl(path)
        return

    data = _safe_read_json(path)
    for item in _iter_json_records(data):
        item = dict(item)
        item.setdefault("_source_path", str(path))
        yield item


def _first_text(*values: Any) -> str:
    for value in values:
        text = _clean_text(value)
        if text:
            return text
    return ""


def _has_endpoint_details(value: Any) -> bool:
    details = _as_dict(value)
    return bool(details.get("required_parameters") or details.get("optional_parameters"))


def _toolbench_endpoint_match(record: Dict[str, Any]) -> Optional[bool]:
    compressed = _as_dict(record.get("compressed"))
    service = _as_dict(record.get("service"))
    category = _first_text(compressed.get("category"), service.get("category"), record.get("category"))
    file_name = _first_text(
        service.get("_file"),
        record.get("_file"),
        compressed.get("_file"),
        service.get("file_name"),
        record.get("file_name"),
        compressed.get("file_name"),
    )
    existing_details = any(
        _has_endpoint_details(source)
        for source in (
            record.get("endpoint_details"),
            compressed.get("endpoint_details"),
            service.get("endpoint_details"),
        )
    )
    tool_json_available = bool(_load_tool_json(category, file_name)) if category and file_name else False
    if not (tool_json_available or existing_details):
        return None

    try:
        endpoint_detail = _find_endpoint_detail(record, compressed, service)
    except Exception:
        return None

    endpoint_details = _as_dict(endpoint_detail.get("endpoint_details"))
    return bool(
        _clean_text(endpoint_detail.get("description"))
        or endpoint_details.get("required_parameters")
        or endpoint_details.get("optional_parameters")
    )


def audit_records(
    records: Iterable[Dict[str, Any]],
    *,
    description_short_chars: int,
    subtask_text: str,
    limit: Optional[int] = None,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []

    for idx, record in enumerate(records, start=1):
        if limit is not None and len(rows) >= limit:
            break
        normalized = normalize_api_for_ranking(record, subtask_text=subtask_text, include_qos_rank=False)
        description = _clean_text(normalized.get("description"))
        parameters = normalized.get("parameters") if isinstance(normalized.get("parameters"), list) else []
        useful_parameter_count = sum(
            1
            for param in parameters
            if isinstance(param, dict) and _clean_text(param.get("name")) and _clean_text(param.get("description"))
        )
        endpoint_match = _toolbench_endpoint_match(record)
        missing_description = not description
        short_description = bool(description) and len(description) < description_short_chars
        generic_description = _is_generic_description(description)

        rows.append(
            {
                "row_index": idx,
                "source_path": record.get("_source_path", ""),
                "source_line": record.get("_source_line", ""),
                "api_id": normalized.get("api_id", ""),
                "name": normalized.get("name", ""),
                "tool_name": normalized.get("tool_name", ""),
                "description_length": len(description),
                "missing_description": int(missing_description),
                "short_description": int(short_description),
                "generic_description": int(generic_description),
                "useful_parameter_count": useful_parameter_count,
                "has_useful_parameters": int(useful_parameter_count > 0),
                "toolbench_endpoint_match": "" if endpoint_match is None else int(endpoint_match),
                "description": description,
            }
        )

    total = len(rows)
    missing = sum(int(row["missing_description"]) for row in rows)
    short = sum(int(row["short_description"]) for row in rows)
    generic = sum(int(row["generic_description"]) for row in rows)
    with_params = sum(int(row["has_useful_parameters"]) for row in rows)
    description_issue_union = sum(
        1
        for row in rows
        if row["missing_description"] or row["short_description"] or row["generic_description"]
    )
    endpoint_detectable = sum(1 for row in rows if row["toolbench_endpoint_match"] != "")
    endpoint_matches = sum(1 for row in rows if row["toolbench_endpoint_match"] == 1)

    summary: Dict[str, Any] = {
        "total_apis_checked": total,
        "missing_descriptions_count": missing,
        "missing_descriptions_pct": _pct(missing, total),
        "short_descriptions_count": short,
        "short_descriptions_pct": _pct(short, total),
        "generic_descriptions_count": generic,
        "generic_descriptions_pct": _pct(generic, total),
        "with_useful_parameters_count": with_params,
        "with_useful_parameters_pct": _pct(with_params, total),
        "description_issue_union_count": description_issue_union,
        "description_issue_union_pct": _pct(description_issue_union, total),
        "toolbench_endpoint_detectable_count": endpoint_detectable,
        "toolbench_endpoint_match_count": endpoint_matches,
        "toolbench_endpoint_match_pct": _pct(endpoint_matches, endpoint_detectable),
        "description_short_chars": description_short_chars,
        "decision_rule": (
            "If more than 20-25% of APIs have missing, generic, or very short descriptions, "
            "consider adding endpoint_path to ranking prompts. Otherwise, keep endpoint_path "
            "out and use it only internally for metadata lookup."
        ),
    }
    return summary, rows


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_index",
        "source_path",
        "source_line",
        "api_id",
        "name",
        "tool_name",
        "description_length",
        "missing_description",
        "short_description",
        "generic_description",
        "useful_parameter_count",
        "has_useful_parameters",
        "toolbench_endpoint_match",
        "description",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _print_summary(summary: Dict[str, Any], input_path: Path) -> None:
    total = int(summary["total_apis_checked"])
    print("API evidence quality audit")
    print(f"Input: {input_path}")
    print(f"Total APIs checked: {total}")
    print(
        "Missing descriptions: "
        f"{summary['missing_descriptions_count']} ({summary['missing_descriptions_pct']}%)"
    )
    print(
        "Short descriptions: "
        f"{summary['short_descriptions_count']} ({summary['short_descriptions_pct']}%)"
    )
    print(
        "Generic descriptions: "
        f"{summary['generic_descriptions_count']} ({summary['generic_descriptions_pct']}%)"
    )
    print(
        "With useful parameters: "
        f"{summary['with_useful_parameters_count']} ({summary['with_useful_parameters_pct']}%)"
    )
    if summary["toolbench_endpoint_detectable_count"]:
        print(
            "With ToolBench endpoint match: "
            f"{summary['toolbench_endpoint_match_count']}/"
            f"{summary['toolbench_endpoint_detectable_count']} "
            f"({summary['toolbench_endpoint_match_pct']}%)"
        )
    else:
        print("With ToolBench endpoint match: n/a")
    print(
        "Missing/generic/very short descriptions: "
        f"{summary['description_issue_union_count']} ({summary['description_issue_union_pct']}%)"
    )
    print(summary["decision_rule"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit normalized compact API evidence quality.")
    parser.add_argument("--input", required=True, type=Path, help="Catalog JSONL, candidate JSON, or run directory.")
    parser.add_argument("--output-json", type=Path, help="Optional JSON summary output path.")
    parser.add_argument("--output-csv", type=Path, help="Optional per-API CSV detail output path.")
    parser.add_argument("--limit", type=int, help="Optional maximum number of records to audit.")
    parser.add_argument("--short-chars", type=int, default=40, help="Description length threshold. Default: 40.")
    parser.add_argument("--subtask-text", default="", help="Optional subtask text used by parameter selection.")
    args = parser.parse_args()

    records = _iter_input_records(args.input)
    summary, rows = audit_records(
        records,
        description_short_chars=max(1, int(args.short_chars)),
        subtask_text=args.subtask_text,
        limit=args.limit,
    )
    _print_summary(summary, args.input)

    if args.output_json:
        _write_json(args.output_json, {"input": str(args.input), "summary": summary})
    if args.output_csv:
        _write_csv(args.output_csv, rows)


if __name__ == "__main__":
    main()
