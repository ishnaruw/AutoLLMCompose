#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


MODE_ORDER = ("no_qos", "qos_pure_llm", "qos_topsis", "qos_hybrid")
QUERY_DIR_RE = re.compile(r"^(q\d{2})_")
TIE_TOLERANCE = 1e-12
POOR_QOS_THRESHOLD = 0.25

WEIGHT_SETTINGS: tuple[tuple[int, int], ...] = (
    (0, 100),
    (25, 75),
    (50, 50),
    (70, 30),
    (75, 25),
    (100, 0),
)

WEIGHT_PURPOSE = {
    (0, 100): "QoS-only extreme",
    (25, 75): "QoS-dominant setting",
    (50, 50): "Balanced setting",
    (70, 30): "Selected primary setting",
    (75, 25): "Conservative robustness setting",
    (100, 0): "Functional-only extreme",
}


@dataclass(frozen=True)
class WorkflowRow:
    query_id: str
    query_text: str
    run_folder: str
    mode: str
    composition_completeness: float
    functional_coverage: float
    normalized_qos: float
    source_file: str


@dataclass(frozen=True)
class QueryBest:
    query_id: str
    alpha_percent: int
    beta_percent: int
    best_modes: tuple[str, ...]
    selected_score: float
    runner_up_score: float | None
    selected_completeness_mean: float
    selected_functional_mean: float
    selected_functional_min: float
    selected_qos_mean: float
    selected_qos_min: float

    @property
    def tie_count(self) -> int:
        return len(self.best_modes)

    @property
    def best_status(self) -> str:
        return "tied best" if self.tie_count > 1 else "unique best"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate thesis Table 5.2 and Table 5.3 weight-sensitivity summaries "
            "from an AutoLLMCompose experiment run folder."
        )
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Experiment run directory, e.g. results/logs/.../fireworks_gpt-oss-120b",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run_dir>/weigh_sensitivity.",
    )
    parser.add_argument(
        "--poor-qos-threshold",
        type=float,
        default=POOR_QOS_THRESHOLD,
        help="Candidate-level normalized QoS threshold for poor QoS risk.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def as_float(value: Any, *, field: str, path: Path | None = None) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        location = f" in {path}" if path else ""
        raise ValueError(f"Missing numeric value for {field}{location}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        location = f" in {path}" if path else ""
        raise ValueError(f"Invalid numeric value for {field}{location}: {value!r}") from exc
    if not math.isfinite(number):
        location = f" in {path}" if path else ""
        raise ValueError(f"Non-finite numeric value for {field}{location}: {value!r}")
    return number


def optional_float(value: Any) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def norm_subtask_id(value: Any) -> str:
    text = str(value or "").strip()
    return text[:-2] if text.endswith(".0") else text


def weight_label(alpha_percent: int, beta_percent: int) -> str:
    return f"{alpha_percent}/{beta_percent}"


def source_root_from_run(run_dir: Path) -> Path:
    current = run_dir.resolve()
    for parent in (current, *current.parents):
        if parent.name == "AutoLLMCompose":
            return parent
    return current.parent


def resolve_run_relative_path(run_dir: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    root = source_root_from_run(run_dir)
    candidate = root / path
    if candidate.exists():
        return candidate
    return run_dir / path


def discover_query_dirs(run_dir: Path) -> dict[str, Path]:
    query_dirs: dict[str, Path] = {}
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        match = QUERY_DIR_RE.match(child.name)
        if match:
            query_dirs[match.group(1)] = child
    if not query_dirs:
        raise FileNotFoundError(f"No query directories matching qNN_* found under {run_dir}")
    return query_dirs


def load_workflows(run_dir: Path) -> list[WorkflowRow]:
    summary_path = run_dir / "summary" / "all_15_query_composition_results.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing composition summary: {summary_path}")
    rows: list[WorkflowRow] = []
    seen: set[tuple[str, str]] = set()
    for raw in read_csv_dicts(summary_path):
        query_id = (raw.get("Query_ID") or "").strip()
        mode = (raw.get("Mode") or "").strip()
        if not query_id or mode not in MODE_ORDER:
            continue
        key = (query_id, mode)
        if key in seen:
            raise ValueError(f"Duplicate workflow row for {query_id}/{mode} in {summary_path}")
        seen.add(key)
        rows.append(
            WorkflowRow(
                query_id=query_id,
                query_text=raw.get("Query_Text", ""),
                run_folder=raw.get("Run_Folder", ""),
                mode=mode,
                composition_completeness=as_float(
                    raw.get("Composition_Completeness"),
                    field="Composition_Completeness",
                    path=summary_path,
                ),
                functional_coverage=as_float(
                    raw.get("Functional_Coverage"),
                    field="Functional_Coverage",
                    path=summary_path,
                ),
                normalized_qos=as_float(
                    raw.get("Normalized_QoS_Score"),
                    field="Normalized_QoS_Score",
                    path=summary_path,
                ),
                source_file=raw.get("Source_File", ""),
            )
        )
    return rows


def adjusted_score(row: WorkflowRow, alpha_percent: int, beta_percent: int) -> float:
    alpha = alpha_percent / 100.0
    beta = beta_percent / 100.0
    return row.composition_completeness * (
        alpha * row.functional_coverage + beta * row.normalized_qos
    )


def compute_query_best(workflows: list[WorkflowRow]) -> dict[tuple[int, int], list[QueryBest]]:
    by_query: dict[str, list[WorkflowRow]] = defaultdict(list)
    for row in workflows:
        by_query[row.query_id].append(row)

    output: dict[tuple[int, int], list[QueryBest]] = {}
    for alpha_percent, beta_percent in WEIGHT_SETTINGS:
        best_rows: list[QueryBest] = []
        for query_id in sorted(by_query):
            query_rows = by_query[query_id]
            scored = [
                (adjusted_score(row, alpha_percent, beta_percent), row)
                for row in query_rows
            ]
            if not scored:
                continue
            best_score = max(score for score, _row in scored)
            winners = [
                row
                for score, row in scored
                if math.isclose(score, best_score, rel_tol=0.0, abs_tol=TIE_TOLERANCE)
            ]
            runner_scores = sorted(
                {
                    round(score, 15)
                    for score, _row in scored
                    if not math.isclose(score, best_score, rel_tol=0.0, abs_tol=TIE_TOLERANCE)
                },
                reverse=True,
            )
            runner_up = runner_scores[0] if runner_scores else None
            best_rows.append(
                QueryBest(
                    query_id=query_id,
                    alpha_percent=alpha_percent,
                    beta_percent=beta_percent,
                    best_modes=tuple(mode for mode in MODE_ORDER if any(row.mode == mode for row in winners)),
                    selected_score=sum(
                        adjusted_score(row, alpha_percent, beta_percent) for row in winners
                    )
                    / len(winners),
                    runner_up_score=runner_up,
                    selected_completeness_mean=sum(row.composition_completeness for row in winners)
                    / len(winners),
                    selected_functional_mean=sum(row.functional_coverage for row in winners)
                    / len(winners),
                    selected_functional_min=min(row.functional_coverage for row in winners),
                    selected_qos_mean=sum(row.normalized_qos for row in winners) / len(winners),
                    selected_qos_min=min(row.normalized_qos for row in winners),
                )
            )
        output[(alpha_percent, beta_percent)] = best_rows
    return output


def table_5_2_rows(query_best: dict[tuple[int, int], list[QueryBest]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for alpha_percent, beta_percent in WEIGHT_SETTINGS:
        best_rows = query_best[(alpha_percent, beta_percent)]
        if not best_rows:
            continue
        rows.append(
            {
                "Weight Setting": weight_label(alpha_percent, beta_percent),
                "Purpose": WEIGHT_PURPOSE[(alpha_percent, beta_percent)],
                "Mean Selected Score": f"{sum(row.selected_score for row in best_rows) / len(best_rows):.6f}",
                "Mean Selected Functional Coverage": f"{sum(row.selected_functional_mean for row in best_rows) / len(best_rows):.6f}",
                "Minimum Selected Functional Coverage": f"{min(row.selected_functional_min for row in best_rows):.6f}",
                "Mean Selected Normalized QoS": f"{sum(row.selected_qos_mean for row in best_rows) / len(best_rows):.6f}",
                "Unique Best Queries": sum(1 for row in best_rows if row.tie_count == 1),
                "Tied-Best Queries": sum(1 for row in best_rows if row.tie_count > 1),
            }
        )
    return rows


def selected_query_metric_rows(query_best: dict[tuple[int, int], list[QueryBest]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for order, (alpha_percent, beta_percent) in enumerate(WEIGHT_SETTINGS, start=1):
        for best in query_best[(alpha_percent, beta_percent)]:
            rows.append(
                {
                    "Query_ID": best.query_id,
                    "Weight_Order": order,
                    "Weight_Setting": weight_label(alpha_percent, beta_percent),
                    "Alpha_Percent": alpha_percent,
                    "Beta_Percent": beta_percent,
                    "Best_Mode_or_Tied_Modes": ", ".join(best.best_modes),
                    "Tie_Count": best.tie_count,
                    "Best_Status": best.best_status,
                    "Selected_Adjusted_Score": best.selected_score,
                    "Runner_Up_Adjusted_Score": best.runner_up_score,
                    "Score_Margin_To_Runner_Up": (
                        "" if best.runner_up_score is None else best.selected_score - best.runner_up_score
                    ),
                    "Selected_Composition_Completeness_Mean": best.selected_completeness_mean,
                    "Selected_Functional_Coverage_Mean": best.selected_functional_mean,
                    "Selected_Functional_Coverage_Min_Across_Ties": best.selected_functional_min,
                    "Selected_Normalized_QoS_Mean": best.selected_qos_mean,
                    "Selected_Normalized_QoS_Min_Across_Ties": best.selected_qos_min,
                }
            )
    return rows


def load_included_cases(run_dir: Path) -> dict[tuple[str, str], int]:
    path = run_dir / "ranking_eval" / "included_cases.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Top-N case file: {path}")
    topn: dict[tuple[str, str], int] = {}
    for raw in read_csv_dicts(path):
        query_id = (raw.get("query_id") or "").strip()
        subtask_id = norm_subtask_id(raw.get("subtask_id"))
        if not query_id or not subtask_id:
            continue
        k = int(as_float(raw.get("k"), field="k", path=path))
        topn[(query_id, subtask_id)] = k
    return topn


def load_candidate_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    query_dirs = discover_query_dirs(run_dir)
    for query_id, query_dir in sorted(query_dirs.items()):
        path = query_dir / "evaluation" / f"query_{query_id}_candidate_api_rankings_rows.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing candidate ranking rows: {path}")
        data = read_json(path)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list in {path}")
        for raw in data:
            if not isinstance(raw, dict):
                continue
            mode = str(raw.get("Mode") or "").strip()
            subtask_id = norm_subtask_id(raw.get("Sub Task"))
            api_id = str(raw.get("Selected_API") or "").strip()
            if mode not in MODE_ORDER or not subtask_id or not api_id:
                continue
            rows.append(
                {
                    "query_id": query_id,
                    "mode": mode,
                    "subtask_id": subtask_id,
                    "api_id": api_id,
                    "mode_rank": int(as_float(raw.get("Mode Rank"), field="Mode Rank", path=path)),
                    "retrieved_rank": int(
                        as_float(raw.get("Retrieved Rank"), field="Retrieved Rank", path=path)
                    ),
                    "functional_match_label": optional_float(raw.get("Functional Match (0/1)")),
                    "rt_s": optional_float(raw.get("QoS_RT_s")),
                    "tp_kbps": optional_float(raw.get("QoS_TP_kbps")),
                    "availability": optional_float(raw.get("QoS Availability")),
                    "subtask_purpose": raw.get("Subtask_Purpose", ""),
                }
            )
    return rows


def minmax_high(value: float, minimum: float, maximum: float) -> float:
    if math.isclose(maximum, minimum, rel_tol=0.0, abs_tol=0.0):
        return 1.0
    return max(0.0, min(1.0, (value - minimum) / (maximum - minimum)))


def minmax_low(value: float, minimum: float, maximum: float) -> float:
    if math.isclose(maximum, minimum, rel_tol=0.0, abs_tol=0.0):
        return 1.0
    return max(0.0, min(1.0, (maximum - value) / (maximum - minimum)))


def candidate_qos_lookup(candidate_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in candidate_rows:
        key = (row["query_id"], row["subtask_id"])
        grouped[key].setdefault(row["api_id"], row)

    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for (query_id, subtask_id), by_api in grouped.items():
        complete_rows = [
            row
            for row in by_api.values()
            if row["rt_s"] is not None
            and row["tp_kbps"] is not None
            and row["availability"] is not None
        ]
        if not complete_rows:
            continue
        rt_values = [row["rt_s"] for row in complete_rows]
        tp_values = [row["tp_kbps"] for row in complete_rows]
        availability_values = [row["availability"] for row in complete_rows]
        rt_min, rt_max = min(rt_values), max(rt_values)
        tp_min, tp_max = min(tp_values), max(tp_values)
        availability_min, availability_max = min(availability_values), max(availability_values)
        for row in complete_rows:
            rt_norm = minmax_low(row["rt_s"], rt_min, rt_max)
            tp_norm = minmax_high(row["tp_kbps"], tp_min, tp_max)
            availability_norm = minmax_high(row["availability"], availability_min, availability_max)
            lookup[(query_id, subtask_id, row["api_id"])] = {
                "query_id": query_id,
                "subtask_id": subtask_id,
                "api_id": row["api_id"],
                "functional_match_label": row["functional_match_label"],
                "QoS_RT_s": row["rt_s"],
                "QoS_TP_kbps": row["tp_kbps"],
                "QoS_Availability": row["availability"],
                "normalized_response_time_score": rt_norm,
                "normalized_throughput_score": tp_norm,
                "normalized_availability_score": availability_norm,
                "candidate_normalized_qos_score": (rt_norm + tp_norm + availability_norm) / 3.0,
            }
    return lookup


def checked_topn_rows(
    *,
    run_dir: Path,
    query_best: dict[tuple[int, int], list[QueryBest]],
    candidate_rows: list[dict[str, Any]],
    topn_by_case: dict[tuple[str, str], int],
    qos_lookup: dict[tuple[str, str, str], dict[str, Any]],
    poor_qos_threshold: float,
) -> list[dict[str, Any]]:
    by_query_mode_subtask: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        by_query_mode_subtask[(row["query_id"], row["mode"], row["subtask_id"])].append(row)
    for rows in by_query_mode_subtask.values():
        rows.sort(key=lambda item: (item["mode_rank"], item["api_id"]))

    output: list[dict[str, Any]] = []
    for alpha_percent, beta_percent in WEIGHT_SETTINGS:
        for best in query_best[(alpha_percent, beta_percent)]:
            for mode in best.best_modes:
                subtask_ids = sorted(
                    {
                        subtask_id
                        for query_id, candidate_mode, subtask_id in by_query_mode_subtask
                        if query_id == best.query_id and candidate_mode == mode
                    },
                    key=lambda value: int(value) if value.isdigit() else value,
                )
                for subtask_id in subtask_ids:
                    top_n = topn_by_case.get((best.query_id, subtask_id))
                    if top_n is None:
                        raise KeyError(f"Missing included_cases k for {best.query_id} subtask {subtask_id}")
                    for candidate in by_query_mode_subtask[(best.query_id, mode, subtask_id)]:
                        if candidate["mode_rank"] > top_n:
                            continue
                        normalized = qos_lookup.get(
                            (best.query_id, subtask_id, candidate["api_id"]),
                            {},
                        )
                        functional = candidate["functional_match_label"]
                        candidate_qos = normalized.get("candidate_normalized_qos_score")
                        functional_invalid = functional is None or int(functional) != 1
                        poor_qos = candidate_qos is None or candidate_qos <= poor_qos_threshold
                        reasons: list[str] = []
                        if functional_invalid:
                            reasons.append("functional_match_label != 1")
                        if poor_qos:
                            if candidate_qos is None:
                                reasons.append("candidate_normalized_qos_score missing")
                            else:
                                reasons.append(
                                    f"candidate_normalized_qos_score={candidate_qos:.6f} <= {poor_qos_threshold:.6f}"
                                )
                        output.append(
                            {
                                "Query_ID": best.query_id,
                                "Weight_Setting": weight_label(alpha_percent, beta_percent),
                                "Alpha_Percent": alpha_percent,
                                "Beta_Percent": beta_percent,
                                "Best_Mode": mode,
                                "Best_Mode_or_Tied_Modes": ", ".join(best.best_modes),
                                "Subtask_ID": subtask_id,
                                "Top_N": top_n,
                                "API_ID": candidate["api_id"],
                                "Mode_Rank": candidate["mode_rank"],
                                "Retrieved_Rank": candidate["retrieved_rank"],
                                "Functional_Match_Label": functional if functional is not None else "",
                                "Candidate_Normalized_QoS_Score": (
                                    "" if candidate_qos is None else candidate_qos
                                ),
                                "Normalized_Response_Time_Score": normalized.get(
                                    "normalized_response_time_score", ""
                                ),
                                "Normalized_Throughput_Score": normalized.get(
                                    "normalized_throughput_score", ""
                                ),
                                "Normalized_Availability_Score": normalized.get(
                                    "normalized_availability_score", ""
                                ),
                                "QoS_RT_s": candidate["rt_s"] if candidate["rt_s"] is not None else "",
                                "QoS_TP_kbps": (
                                    candidate["tp_kbps"] if candidate["tp_kbps"] is not None else ""
                                ),
                                "QoS_Availability": (
                                    candidate["availability"]
                                    if candidate["availability"] is not None
                                    else ""
                                ),
                                "Subtask_Purpose": candidate["subtask_purpose"],
                                "Triggers_Functional_Invalid": functional_invalid,
                                "Triggers_Poor_QoS": poor_qos,
                                "Triggers_Functional_Invalid_Or_Poor_QoS": (
                                    functional_invalid or poor_qos
                                ),
                                "Reason": "; ".join(reasons),
                            }
                        )
    return output


def table_5_3_rows(checked_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_weight: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in checked_rows:
        by_weight[(int(row["Alpha_Percent"]), int(row["Beta_Percent"]))].append(row)

    rows: list[dict[str, Any]] = []
    for alpha_percent, beta_percent in WEIGHT_SETTINGS:
        weight_rows = by_weight[(alpha_percent, beta_percent)]
        functional_invalid = [
            row for row in weight_rows if bool(row["Triggers_Functional_Invalid"])
        ]
        poor_qos = [row for row in weight_rows if bool(row["Triggers_Poor_QoS"])]
        rows.append(
            {
                "Weight Setting": weight_label(alpha_percent, beta_percent),
                "Checked Top-N Candidate Rows": len(weight_rows),
                "Functional-Invalid Candidate Count": len(functional_invalid),
                "Poor-QoS Candidate Count": len(poor_qos),
                "Queries with Functional-Invalid Top-N": len(
                    {row["Query_ID"] for row in functional_invalid}
                ),
                "Queries with Poor-QoS Top-N": len({row["Query_ID"] for row in poor_qos}),
            }
        )
    return rows


def candidate_qos_rows(qos_lookup: dict[tuple[str, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for _key, row in sorted(
            qos_lookup.items(),
            key=lambda item: (item[0][0], int(item[0][1]) if item[0][1].isdigit() else item[0][1], item[0][2]),
        )
    ]


def markdown_table(fieldnames: list[str], rows: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fieldnames) + " |")
    return "\n".join(lines)


def write_markdown(output_dir: Path, table52: list[dict[str, Any]], table53: list[dict[str, Any]]) -> None:
    fields52 = [
        "Weight Setting",
        "Purpose",
        "Mean Selected Score",
        "Mean Selected Functional Coverage",
        "Minimum Selected Functional Coverage",
        "Mean Selected Normalized QoS",
        "Unique Best Queries",
        "Tied-Best Queries",
    ]
    fields53 = [
        "Weight Setting",
        "Checked Top-N Candidate Rows",
        "Functional-Invalid Candidate Count",
        "Poor-QoS Candidate Count",
        "Queries with Functional-Invalid Top-N",
        "Queries with Poor-QoS Top-N",
    ]
    text = (
        "# Weight Sensitivity Tables\n\n"
        "## Table 5.2: Representative alpha and beta Sensitivity of Selected Best Workflows\n\n"
        f"{markdown_table(fields52, table52)}\n\n"
        "## Table 5.3: Candidate-Level Top-N Risk Summary Across Weight Settings\n\n"
        f"{markdown_table(fields53, table53)}\n"
    )
    (output_dir / "weight_sensitivity_tables.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "weigh_sensitivity"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    workflows = load_workflows(run_dir)
    query_best = compute_query_best(workflows)
    table52 = table_5_2_rows(query_best)

    topn_by_case = load_included_cases(run_dir)
    candidate_rows = load_candidate_rows(run_dir)
    qos_lookup = candidate_qos_lookup(candidate_rows)
    checked_rows = checked_topn_rows(
        run_dir=run_dir,
        query_best=query_best,
        candidate_rows=candidate_rows,
        topn_by_case=topn_by_case,
        qos_lookup=qos_lookup,
        poor_qos_threshold=args.poor_qos_threshold,
    )
    table53 = table_5_3_rows(checked_rows)

    table52_fields = [
        "Weight Setting",
        "Purpose",
        "Mean Selected Score",
        "Mean Selected Functional Coverage",
        "Minimum Selected Functional Coverage",
        "Mean Selected Normalized QoS",
        "Unique Best Queries",
        "Tied-Best Queries",
    ]
    table53_fields = [
        "Weight Setting",
        "Checked Top-N Candidate Rows",
        "Functional-Invalid Candidate Count",
        "Poor-QoS Candidate Count",
        "Queries with Functional-Invalid Top-N",
        "Queries with Poor-QoS Top-N",
    ]
    selected_fields = [
        "Query_ID",
        "Weight_Order",
        "Weight_Setting",
        "Alpha_Percent",
        "Beta_Percent",
        "Best_Mode_or_Tied_Modes",
        "Tie_Count",
        "Best_Status",
        "Selected_Adjusted_Score",
        "Runner_Up_Adjusted_Score",
        "Score_Margin_To_Runner_Up",
        "Selected_Composition_Completeness_Mean",
        "Selected_Functional_Coverage_Mean",
        "Selected_Functional_Coverage_Min_Across_Ties",
        "Selected_Normalized_QoS_Mean",
        "Selected_Normalized_QoS_Min_Across_Ties",
    ]
    checked_fields = [
        "Query_ID",
        "Weight_Setting",
        "Alpha_Percent",
        "Beta_Percent",
        "Best_Mode",
        "Best_Mode_or_Tied_Modes",
        "Subtask_ID",
        "Top_N",
        "API_ID",
        "Mode_Rank",
        "Retrieved_Rank",
        "Functional_Match_Label",
        "Candidate_Normalized_QoS_Score",
        "Normalized_Response_Time_Score",
        "Normalized_Throughput_Score",
        "Normalized_Availability_Score",
        "QoS_RT_s",
        "QoS_TP_kbps",
        "QoS_Availability",
        "Subtask_Purpose",
        "Triggers_Functional_Invalid",
        "Triggers_Poor_QoS",
        "Triggers_Functional_Invalid_Or_Poor_QoS",
        "Reason",
    ]
    candidate_qos_fields = [
        "query_id",
        "subtask_id",
        "api_id",
        "functional_match_label",
        "QoS_RT_s",
        "QoS_TP_kbps",
        "QoS_Availability",
        "normalized_response_time_score",
        "normalized_throughput_score",
        "normalized_availability_score",
        "candidate_normalized_qos_score",
    ]

    write_csv(output_dir / "table_5_2_representative_alpha_beta_sensitivity.csv", table52, table52_fields)
    write_csv(output_dir / "table_5_3_candidate_topn_risk_summary.csv", table53, table53_fields)
    write_csv(output_dir / "selected_query_metrics.csv", selected_query_metric_rows(query_best), selected_fields)
    write_csv(output_dir / "checked_topn_candidates.csv", checked_rows, checked_fields)
    write_csv(output_dir / "candidate_normalized_qos_scores.csv", candidate_qos_rows(qos_lookup), candidate_qos_fields)
    write_markdown(output_dir, table52, table53)
    write_json(
        output_dir / "methodology.json",
        {
            "run_dir": str(run_dir),
            "output_dir": str(output_dir),
            "weight_settings": [weight_label(alpha, beta) for alpha, beta in WEIGHT_SETTINGS],
            "workflow_score_formula": (
                "Composition_Completeness * (alpha * Functional_Coverage + "
                "beta * Normalized_QoS_Score)"
            ),
            "tie_tolerance": TIE_TOLERANCE,
            "topn_source": "ranking_eval/included_cases.csv column k",
            "candidate_functional_invalid_definition": "Functional Match (0/1) != 1 or missing",
            "candidate_qos_normalization": (
                "Within each query/subtask candidate pool: response time min-max normalized "
                "with lower better, throughput min-max normalized with higher better, "
                "availability min-max normalized with higher better; average of the three components."
            ),
            "poor_qos_threshold": args.poor_qos_threshold,
            "query_count": len({row.query_id for row in workflows}),
            "workflow_rows": len(workflows),
            "checked_topn_candidate_rows": len(checked_rows),
        },
    )

    print(f"Wrote Table 5.2 and Table 5.3 outputs to {output_dir}")


if __name__ == "__main__":
    main()
