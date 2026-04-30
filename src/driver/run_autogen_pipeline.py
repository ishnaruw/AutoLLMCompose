from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.agents.decomposer import decompose_goal
from src.agents.planner import planner_call
from src.agents.ranker import InvalidRankingOutput, rank_subtask
from src.agents.qos_scorer_llm import InvalidQosScoringOutput, score_qos_llm
from src.agents.retriever import collect_candidates
from src.config import CONFIG
from src.core.run_logging import (
    clear_run_log,
    configure_model_usage,
    configure_run_log,
    current_model_usage_path,
    log_error_event,
    log_invalid_case_event,
    log_line,
)
from src.core.retry import call_with_backoff
from src.eval.functional_match_eval import evaluate_query, evaluate_retrieval_functional_match
from src.eval.audit_api_duplicates import collect_duplicate_audit_for_run
from src.eval.audit_api_hallucinations import collect_hallucination_audit_for_run
from src.eval.mode_anomaly_report import write_mode_anomaly_excel
from src.eval.topsis_eval import _extract_qos, _run_topsis_pydecision
from src.llm.autogen_runner import run_autogen_agent
from src.llm.backends import GROQ_MULTI_MODEL_SENTINEL, groq_experiment_model_pool, make_backend
from src.tools.fetch_services import fetch_services

DECOMPOSER_SYS = (
    "You are a decomposition agent for API discovery. "
    "Your job is to split a user request into 2 to 5 ordered API-retrieval subtasks when the request contains multiple distinct capabilities. "
    "Do not collapse multiple functions into one subtask. "
    "Return strict JSON only."
)

RANKER_SYS = (
    "You are a ranking agent. Given the original user query, a single subtask, and "
    "a list of candidate APIs from a catalog, rank the candidates best-to-worst for that subtask. "
    "Follow the prompt strictly and return valid JSON."
)

QOS_SCORER_SYS = (
    "You are a QoS scoring agent. Given only api ids and QoS metrics, produce a relative QoS-only ranking and score. "
    "Return strict JSON only."
)

PLANNER_SYS = (
    "You are an orchestration planner that composes a logical API workflow "
    "using only the selected APIs provided. Preserve the ordered subtasks and return valid JSON."
)

MODE_ORDER = ["no_qos", "qos_pure_llm", "qos_topsis", "qos_hybrid"]
ALL_QUERIES_PATH = Path("data/queries/all_user_query.jsonl")

PROVIDER_POLICY = {
    "mistral": {"sleep_after_query": 0.5},
    "groq": {"sleep_after_query": 0.8},
    "together": {"sleep_after_query": 0.5},
    "gemini": {"sleep_after_query": 0.4},
    "azure_foundry": {"sleep_after_query": 0.2},
    "azure": {"sleep_after_query": 0.2},
    "lmstudio": {"sleep_after_query": 0.0},
    "_default": {"sleep_after_query": 0.4},
}


def load_queries(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Queries file not found: {path.resolve()}")
    queries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip().lstrip("\ufeff")
            if line:
                try:
                    queries.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    preview = line[:200]
                    raise ValueError(
                        f"Invalid JSON in queries file {path} at line {line_no}: {exc.msg}. "
                        f"Line content starts with: {preview}"
                    ) from exc
    return queries


def _parse_query_selection(selection: str, total_queries: int) -> List[int]:
    text = (selection or "").strip().lower()
    if not text:
        raise ValueError("Enter a query number, all, a range like 1-5, or comma-separated numbers like 1,3,5.")

    if text == "all":
        return list(range(total_queries))

    if text.startswith("range"):
        text = text[len("range") :].strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    text = text.replace(" ", "")

    def _to_index(value: str) -> int:
        if not value.isdigit():
            raise ValueError(f"Invalid query number: {value}")
        number = int(value)
        if number < 1 or number > total_queries:
            raise ValueError(f"Query number {number} is outside 1-{total_queries}.")
        return number - 1

    if "," in text:
        parts = [part for part in text.split(",") if part]
        if not parts:
            raise ValueError("No query numbers found.")
        indices = [_to_index(part) for part in parts]
    elif "-" in text:
        parts = text.split("-", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError("Ranges must look like 1-5.")
        start = _to_index(parts[0])
        end = _to_index(parts[1])
        if start > end:
            raise ValueError("Range start must be less than or equal to range end.")
        indices = list(range(start, end + 1))
    else:
        indices = [_to_index(text)]

    selected: List[int] = []
    seen: set[int] = set()
    for index in indices:
        if index not in seen:
            selected.append(index)
            seen.add(index)
    return selected


def choose_queries_interactive(queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not queries:
        raise ValueError(f"No queries loaded from {ALL_QUERIES_PATH}.")

    print(f"\nLoaded {len(queries)} queries from {ALL_QUERIES_PATH}:")
    for idx, query in enumerate(queries, start=1):
        qid = query.get("id", f"q{idx:02d}")
        title = query.get("title", "")
        print(f"  {idx:>2}) {qid} | {title}")

    print("\nWhich queries do you want to run?")
    print("  - Single query: 1")
    print("  - All queries: all")
    print("  - Range: 1-5 or [1-5]")
    print("  - Comma-separated: 1,3,5")

    while True:
        selection = input("Enter query selection: ").strip()
        try:
            indices = _parse_query_selection(selection, len(queries))
        except ValueError as exc:
            print(f"Invalid selection: {exc}")
            continue

        selected = [queries[index] for index in indices]
        selected_labels = ", ".join(str(query.get("id", f"q{index + 1:02d}")) for index, query in zip(indices, selected))
        print(f"Selected {len(selected)} quer{'y' if len(selected) == 1 else 'ies'}: {selected_labels}\n")
        return selected


def choose_provider_interactive() -> str:
    options = [
        ("mistral", "Mistral"),
        ("groq", "Groq"),
        ("fireworks", "Fireworks AI"),
        ("azure_foundry", "Azure (DeepSeek via Foundry endpoint)"),
        ("lmstudio", "LM Studio (local, meta-llama-3.1-8b-instruct)"),
    ]
    print("\nSelect model provider:")
    for i, (_, label) in enumerate(options, start=1):
        print(f"  {i}) {label}")
    while True:
        choice = input("Enter choice number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            provider = options[int(choice) - 1][0]
            print(f"Selected: {options[int(choice) - 1][1]}\n")
            return provider
        print("Invalid choice. Try again.")


def choose_groq_model_interactive() -> str:
    models = groq_experiment_model_pool()
    print("Select Groq model mode:")
    print("  1) Multi-model failover (recommended)")
    for idx, model_name in enumerate(models, start=2):
        print(f"  {idx}) Single model: {model_name}")
    while True:
        choice = input("Enter choice number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models) + 1:
            selected = int(choice)
            if selected == 1:
                print(f"Selected: Groq multi-model failover ({', '.join(models)})\n")
                return GROQ_MULTI_MODEL_SENTINEL
            model_name = models[selected - 2]
            print(f"Selected: Groq single model {model_name}\n")
            return model_name
        print("Invalid choice. Try again.")


def choose_provider_and_model_interactive() -> tuple[str, str | None]:
    provider = choose_provider_interactive()
    if provider == "groq":
        return provider, choose_groq_model_interactive()
    return provider, None


def _safe_name(text: str) -> str:
    text = (text or "unknown").strip()
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() or ch in {"-", "_"} else "_")
    safe = "".join(out).strip("_")
    return safe or "unknown"


def _model_dir_name(model_tag: str) -> str:
    provider, sep, model_name = (model_tag or "").partition(":")
    if not sep:
        return _safe_name(model_tag)
    model_leaf = model_name.rstrip("/").rsplit("/", 1)[-1] or model_name
    return f"{_safe_name(provider)}_{_safe_name(model_leaf)}"


def _run_dir(model_tag: str, query_id: str | None = None, run_tag: str | None = None) -> Path:
    run_id = time.strftime("%Y%m%dT%H%M%S")
    run_name = f"{_safe_name(query_id)}_{run_id}" if query_id else run_id
    base_dir = Path("results/logs")
    if run_tag:
        base_dir = base_dir / _safe_name(run_tag)
    out = base_dir / _model_dir_name(model_tag) / run_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _fetch_catalog_subset(api_ids: List[str], with_qos: bool) -> Dict[str, Dict[str, Any]]:
    wanted = set(str(x) for x in api_ids if x)
    found: Dict[str, Dict[str, Any]] = {}
    offset = 0
    while True:
        batch = fetch_services(category=None, offset=offset, limit=500, with_qos=with_qos)
        if not batch:
            break
        for item in batch:
            api_id = str(item.get("api_id", ""))
            if api_id in wanted:
                found[api_id] = item
        offset += len(batch)
        if len(found) >= len(wanted):
            break
    return found


def _build_llm_call(backend):
    def _lmstudio_limits(role_name: str) -> Dict[str, Any]:
        limits: Dict[str, Any] = {}
        if getattr(backend, "provider", "") != "lmstudio":
            return limits
        limits["timeout_seconds"] = CONFIG.lmstudio_timeout_seconds
        if role_name == "ranker":
            limits["max_tokens"] = CONFIG.lmstudio_ranker_max_tokens
        return limits

    def _invoke(fn, *, name: str) -> str:
        if getattr(backend, "multi_model_mode", lambda: False)():
            return fn()
        return call_with_backoff(fn, name=name)

    def llm_call(role_name: str, system_msg: str, prompt: str) -> str:
        temp = 0.2 if role_name == "planner" else 0.0
        limits = _lmstudio_limits(role_name)
        if CONFIG.use_autogen_agents:
            return _invoke(
                lambda: run_autogen_agent(
                    backend=backend,
                    role_name=role_name,
                    system_message=system_msg,
                    prompt=prompt,
                    temperature=temp,
                    force_json=True,
                    max_tokens=limits.get("max_tokens"),
                    timeout_seconds=limits.get("timeout_seconds"),
                ),
                name=role_name,
            )
        return _invoke(
            lambda: backend.chat_json(
                system_msg,
                prompt,
                temperature=temp,
                force_json=True,
                max_tokens=limits.get("max_tokens"),
                timeout_seconds=limits.get("timeout_seconds"),
            ),
            name=role_name,
        )
    return llm_call


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _to_run_relative(path: Path | None, run_dir: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(run_dir.resolve()))
    except Exception:
        return str(path)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class _RunMetaTracker:
    def __init__(self, meta_path: Path, payload: Dict[str, Any]) -> None:
        self.meta_path = meta_path
        self.payload = payload
        self._active_stages: Dict[str, float] = {}
        self._run_started_perf = time.perf_counter()

    def persist(self) -> None:
        _write_json(self.meta_path, self.payload)

    def update(self, **kwargs: Any) -> None:
        self.payload.update(kwargs)
        self.persist()

    def set_stage(self, key: str, data: Dict[str, Any]) -> None:
        self.payload.setdefault("timing", {}).setdefault("stages", {})[key] = data
        self.persist()

    def start_stage(self, key: str, **extra: Any) -> None:
        stage = self.payload.setdefault("timing", {}).setdefault("stages", {}).setdefault(key, {})
        stage.update({"status": "running", "started_at": _now_iso()})
        stage.update({k: v for k, v in extra.items() if v is not None})
        self._active_stages[key] = time.perf_counter()
        self.persist()

    def finish_stage(self, key: str, *, status: str = "completed", **extra: Any) -> None:
        stage = self.payload.setdefault("timing", {}).setdefault("stages", {}).setdefault(key, {})
        started_perf = self._active_stages.pop(key, None)
        stage["status"] = status
        stage["ended_at"] = _now_iso()
        if started_perf is not None:
            stage["duration_seconds"] = round(time.perf_counter() - started_perf, 3)
        stage.update({k: v for k, v in extra.items() if v is not None})
        self.persist()

    def record_invocation(
        self,
        key: str,
        *,
        started_at: str,
        ended_at: str,
        duration_seconds: float,
        status: str = "completed",
        **extra: Any,
    ) -> None:
        stage = self.payload.setdefault("timing", {}).setdefault("stages", {}).setdefault(
            key,
            {"invocations": 0, "duration_seconds": 0.0},
        )
        if not stage.get("started_at"):
            stage["started_at"] = started_at
        stage["ended_at"] = ended_at
        stage["invocations"] = int(stage.get("invocations", 0)) + 1
        stage["duration_seconds"] = round(float(stage.get("duration_seconds", 0.0)) + float(duration_seconds), 3)
        if status == "failed":
            stage["status"] = "failed"
            stage["failed_invocations"] = int(stage.get("failed_invocations", 0)) + 1
        elif stage.get("status") != "failed":
            stage["status"] = "completed"
        stage.update({k: v for k, v in extra.items() if v is not None})
        self.persist()

    def finish_run(self, *, status: str, error: str | None = None) -> None:
        run_meta = self.payload.setdefault("timing", {}).setdefault("run", {})
        run_meta["ended_at"] = _now_iso()
        run_meta["duration_seconds"] = round(time.perf_counter() - self._run_started_perf, 3)
        self.payload["status"] = status
        if error:
            self.payload["error"] = error
        else:
            self.payload.pop("error", None)
        self.persist()


def _timed_invocation(
    tracker: _RunMetaTracker,
    stage_key: str,
    fn,
    **extra: Any,
):
    started_at = _now_iso()
    started_perf = time.perf_counter()
    status = "completed"
    try:
        result = fn()
        return result, round(time.perf_counter() - started_perf, 3)
    except Exception:
        status = "failed"
        raise
    finally:
        ended_at = _now_iso()
        duration_seconds = round(time.perf_counter() - started_perf, 3)
        tracker.record_invocation(
            stage_key,
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=duration_seconds,
            status=status,
            **extra,
        )


def _build_shared_retrieval(user_goal: str, subtasks: List[Dict[str, Any]], out_dir: Path) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    retrieved_by_subtask: Dict[str, List[Dict[str, Any]]] = {}
    pick_ids: List[str] = []
    seen = set()
    for sub in subtasks:
        sub_id = str(sub.get("id", "unknown"))
        retrieved = collect_candidates(subtask_goal=str(sub.get("description", "")), index_dir=str(CONFIG.shared_index_dir), top_k=CONFIG.rag_top_k)
        for idx, item in enumerate(retrieved, start=1):
            item["retrieved_rank"] = idx
        retrieved_by_subtask[sub_id] = retrieved
        _write_json(out_dir / f"1_retriever_s{sub_id}.json", retrieved)
        for item in retrieved:
            api_id = str(item.get("api_id", ""))
            if api_id and api_id not in seen:
                seen.add(api_id)
                pick_ids.append(api_id)
    return retrieved_by_subtask, _fetch_catalog_subset(pick_ids, with_qos=False), _fetch_catalog_subset(pick_ids, with_qos=True)


def _candidate_rows(retrieved: List[Dict[str, Any]], id_to_service: Dict[str, Dict[str, Any]], *, enrich: Dict[str, Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in retrieved:
        api_id = str(r.get("api_id", ""))
        row = {
            "api_id": api_id,
            "rag_score": r.get("rag_score", 0.0),
            "retrieved_rank": r.get("retrieved_rank"),
            "compressed": r.get("compressed", {}),
            "service": id_to_service.get(api_id, {}),
        }
        if enrich and api_id in enrich:
            row.update(enrich[api_id])
        rows.append(row)
    return rows


def _compute_topsis_metadata(retrieved: List[Dict[str, Any]], id_to_service: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    rows = []
    api_ids = []
    for r in retrieved:
        api_id = str(r.get("api_id", ""))
        svc = id_to_service.get(api_id, {})
        qos = (svc.get("qos") or {}) if isinstance(svc.get("qos"), dict) else {}
        vec = _extract_qos(qos)
        if vec is not None:
            rows.append(vec)
            api_ids.append(api_id)
    out: Dict[str, Dict[str, Any]] = {}
    if rows:
        import numpy as np
        scores, ranking = _run_topsis_pydecision(np.asarray(rows, dtype=float), [1.0, 1.0, 1.0])
        for idx, api_id in enumerate(api_ids):
            out[api_id] = {"topsis_score": float(scores[idx])}
        for rank, row_idx in enumerate(ranking, start=1):
            out[api_ids[row_idx]]["topsis_rank"] = rank
    # Missing QoS at bottom with valid rank
    next_rank = len(api_ids) + 1
    for r in retrieved:
        api_id = str(r.get("api_id", ""))
        if api_id not in out:
            out[api_id] = {"topsis_score": None, "topsis_rank": next_rank}
            next_rank += 1
    return out


def _deterministic_topsis_ranking(retrieved: List[Dict[str, Any]], topsis_meta: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched = []
    for r in retrieved:
        api_id = str(r.get("api_id", ""))
        m = topsis_meta.get(api_id, {})
        enriched.append((int(m.get("topsis_rank") or 10**9), api_id, m.get("topsis_score")))
    enriched.sort(key=lambda x: x[0])
    return [{"api_id": api_id, "reason": "Deterministic QoS ordering."} for _, api_id, _ in enriched]


def _functional_match_value(entry: Dict[str, Any]) -> int:
    return int(
        entry.get(
            "Functional Match (0/1)",
            entry.get("Functional Match Label", entry.get("functional_match", entry.get("relevant", 0))),
        )
        or 0
    )


def _deterministic_hybrid_ranking(retrieved: List[Dict[str, Any]], topsis_meta: Dict[str, Dict[str, Any]], sub_id: str, functional_match_map: Dict[tuple[str, str], Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply Functional Candidate Refinement for qos_hybrid only."""
    matching_enriched = []
    non_matching_enriched = []

    for r in retrieved:
        api_id = str(r.get("api_id", ""))
        m = topsis_meta.get(api_id, {})
        topsis_rank = int(m.get("topsis_rank") or 10**9)
        topsis_score = m.get("topsis_score")
        match_entry = functional_match_map.get((sub_id, api_id), {})
        is_functional_match = _functional_match_value(match_entry) == 1

        if is_functional_match:
            matching_enriched.append((topsis_rank, api_id, topsis_score))
        else:
            non_matching_enriched.append((topsis_rank, api_id, topsis_score))

    matching_enriched.sort(key=lambda x: (x[0], x[1]))
    non_matching_enriched.sort(key=lambda x: (x[0], x[1]))
    combined = matching_enriched + non_matching_enriched

    ranked: List[Dict[str, Any]] = []
    for topsis_rank, api_id, _ in combined:
        match_entry = functional_match_map.get((sub_id, api_id), {})
        match_label = "functional match" if _functional_match_value(match_entry) == 1 else "not a functional match"
        ranked.append(
            {
                "api_id": api_id,
                "reason": f"Functional Candidate Refinement: {match_label}; ordered by TOPSIS rank {topsis_rank}.",
            }
        )
    return ranked


def _write_ranked(
    mode_dir: Path,
    sub_id: str,
    ranked: List[Dict[str, Any]],
    retrieved: List[Dict[str, Any]],
    id_to_service: Dict[str, Dict[str, Any]],
    extras: Dict[str, Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    rag_map = {str(r.get("api_id")): r for r in retrieved}
    full = []
    ranked_full = [item for item in ranked if str(item.get("api_id", "")) in rag_map]

    for idx, item in enumerate(ranked_full, start=1):
        api_id = str(item.get("api_id", ""))
        base = rag_map.get(api_id, {})
        row = {
            "api_id": api_id,
            "mode_rank": idx,
            "llm_reported_rank": item.get("llm_reported_rank"),
            "retrieved_rank": base.get("retrieved_rank"),
            "rag_score": base.get("rag_score"),
            "reason": item.get("reason", ""),
            "service": id_to_service.get(api_id, {}),
        }
        if extras and api_id in extras:
            row.update(extras[api_id])
        full.append(row)
    _write_json(mode_dir / f"2_ranked_s{sub_id}.json", full)
    return full


def _write_invalid_ranked(mode_dir: Path, sub_id: str, failure: Dict[str, Any]) -> List[Dict[str, Any]]:
    row = {
        "api_id": "",
        "mode_rank": None,
        "retrieved_rank": None,
        "rag_score": None,
        "reason": failure.get("failure_reason", "invalid_ranking_case"),
        "service": {},
        **failure,
    }
    _write_json(mode_dir / f"2_ranked_s{sub_id}.json", [row])
    _write_json(mode_dir / "debug" / f"failure_s{sub_id}.json", row)
    return []


def _ranking_failure_record(
    *,
    metadata: Dict[str, Any],
    query_id: str | None,
    subtask_id: str,
    mode: str,
    out_dir: Path,
) -> Dict[str, Any]:
    record = {
        "failure_flag": True,
        "query_id": query_id,
        "subtask_id": subtask_id,
        "mode": mode,
        "exclude_from_ranking_eval": True,
        **metadata,
    }
    record.setdefault("failure_stage", "llm_ranking")
    record.setdefault("failure_reason", "parse_error")
    record["ranked_file"] = _to_run_relative(out_dir / mode / f"2_ranked_s{subtask_id}.json", out_dir)
    return record


def _is_expected_invalid_evaluation_case(record: Dict[str, Any]) -> bool:
    if record.get("error"):
        return False
    stage = str(record.get("failure_stage") or "")
    reason = str(record.get("failure_reason") or "")
    if reason.endswith("_after_retries"):
        reason = reason[: -len("_after_retries")]
    expected_reasons = {
        "parse_error",
        "invalid_json",
        "duplicate_ranked_apis",
        "unknown_api_ids",
        "incomplete_ranked_api_list",
        "missing_ranked_apis",
        "incomplete_qos_scores",
        "missing_api_scores",
    }
    return stage in {"llm_ranking", "qos_llm_scoring"} and reason in expected_reasons


def _load_functional_match_map(rows_path: Path) -> Dict[tuple[str, str], Dict[str, Any]]:
    data = json.loads(rows_path.read_text(encoding="utf-8"))
    out = {}
    for row in data:
        sub_id = str(row.get("Sub Task", row.get("subtask_id", "")))
        api_id = str(row.get("Selected_API", row.get("api_id", "")))
        if not sub_id or not api_id:
            continue
        functional_match = _functional_match_value(row)
        comment = row.get("Comments", row.get("comment", ""))
        out[(sub_id, api_id)] = {
            "Functional Match (0/1)": functional_match,
            "functional_match": functional_match,
            "Comments": comment,
            "comment": comment,
        }
    return out


def _load_planner_top_n(rows_path: Path) -> Dict[str, int]:
    data = json.loads(rows_path.read_text(encoding="utf-8"))
    matching_api_ids_by_subtask: Dict[str, set[str]] = {}
    for row in data:
        sub_id = str(row.get("Sub Task", row.get("subtask_id", "")))
        api_id = str(row.get("Selected_API", row.get("api_id", "")))
        if not sub_id or not api_id or _functional_match_value(row) != 1:
            continue
        matching_api_ids_by_subtask.setdefault(sub_id, set()).add(api_id)
    return {sub_id: len(api_ids) for sub_id, api_ids in matching_api_ids_by_subtask.items()}


def _deterministic_select_and_plan(mode_name: str, subtasks: List[Dict[str, Any]], ranked_full: Dict[str, List[Dict[str, Any]]], planner_top_n: Dict[str, int], llm_call, user_goal: str, out_dir: Path, planner_prompt_path: str) -> Dict[str, Any]:
    mode_dir = out_dir / mode_name
    selected_all = []
    for sub in subtasks:
        sub_id = str(sub.get("id", "unknown"))
        ranked_rows = ranked_full.get(sub_id, [])
        dynamic_top_n = int(planner_top_n.get(sub_id, 0) or 0)
        fallback_used = dynamic_top_n <= 0
        selected_limit = dynamic_top_n if dynamic_top_n > 0 else CONFIG.selector_top_n
        selected_limit = min(selected_limit, len(ranked_rows))
        selected_rows = ranked_rows[:selected_limit]
        top_n_source = "functional_candidate_refinement_k" if dynamic_top_n > 0 else "selector_top_n_fallback"
        selected = []
        for idx, r in enumerate(selected_rows, start=1):
            row = dict(r)
            row["selected_rank"] = idx
            row["score"] = (len(selected_rows) - idx + 1) / float(len(selected_rows) or 1)
            row["subtask_id"] = sub_id
            row["selector_reason"] = "Deterministic selection from mode rank using a shared per-subtask top-n cutoff."
            row["planner_top_n"] = selected_limit
            row["planner_top_n_source"] = top_n_source
            row["fallback_used"] = fallback_used
            selected.append(row)
        selected_all.extend(selected)
        _write_json(mode_dir / f"3_selected_s{sub_id}.json", selected)
        _write_json(
            mode_dir / f"3_selected_trace_s{sub_id}.json",
            {
                "planner_top_n": selected_limit,
                "planner_top_n_source": top_n_source,
                "dynamic_top_n_from_functional_match": dynamic_top_n,
                "fallback_used": fallback_used,
                "available_ranked": len(ranked_rows),
                "selected_count": len(selected),
            },
        )
    planner = planner_call(llm_call=lambda p: llm_call("planner", PLANNER_SYS, p), user_goal=user_goal, ranked_top=selected_all, subtasks=subtasks, prompt_path=planner_prompt_path)
    _write_json(mode_dir / "4_planner.json", planner)
    return {"selected": selected_all, "planner": planner}


def run_autogen_once(user_goal: str, provider: str | None = None, model: str | None = None, query_id: str | None = None, query_title: str | None = None, run_tag: str | None = None) -> Tuple[Path, Path | None]:
    backend = make_backend(provider=provider, model=model)
    model_tag = backend.name()
    llm_call = _build_llm_call(backend)
    effective_run_tag = CONFIG.run_tag if run_tag is None else run_tag
    out_dir = _run_dir(model_tag, query_id=query_id, run_tag=effective_run_tag)
    run_label = query_id or out_dir.name
    provider_label = provider or backend.name().split(":", 1)[0]
    run_log_path = configure_run_log(out_dir / "run.log")
    model_usage_path = configure_model_usage(
        provider=provider_label,
        model_tag=backend.name(),
        multi_model_mode=backend.multi_model_mode(),
        model_pool=backend.model_pool(),
    )
    meta_tracker = _RunMetaTracker(
        out_dir / "meta.json",
        {
            "query_id": query_id,
            "query_title": query_title,
            "user_goal": user_goal,
            "provider": provider_label,
            "model_tag": backend.name(),
            "active_model": backend.active_model_name(),
            "multi_model_mode": backend.multi_model_mode(),
            "model_pool": backend.model_pool(),
            "model_switches": [],
            "num_subtasks": 0,
            "evaluation_triggered": False,
            "evaluation_dir": "evaluation",
            "planner_enabled": CONFIG.planner_enabled,
            "modes": MODE_ORDER,
            "log_file": _to_run_relative(run_log_path, out_dir) if run_log_path else None,
            "model_usage_file": _to_run_relative(model_usage_path, out_dir) if model_usage_path else None,
            "ranking_failures": [],
            "status": "running",
            "timing": {
                "run": {
                    "started_at": _now_iso(),
                },
                "stages": {},
            },
        },
    )
    meta_tracker.persist()
    if backend.multi_model_mode():
        log_line(
            f"[{run_label}] run started | provider={provider_label} | mode={backend.name()} "
            f"| active_model={backend.active_model_name()} | pool={backend.model_pool()}"
        )
    else:
        log_line(f"[{run_label}] run started | provider={provider_label} | model={backend.name()}")

    subtasks: List[Dict[str, Any]] = []
    retrieved_by_subtask: Dict[str, List[Dict[str, Any]]] = {}
    no_qos_services: Dict[str, Dict[str, Any]] = {}
    with_qos_services: Dict[str, Dict[str, Any]] = {}
    ranked_full_by_mode: Dict[str, Dict[str, List[Dict[str, Any]]]] = {m: {} for m in MODE_ORDER}
    functional_match_map: Dict[tuple[str, str], Dict[str, Any]] = {}
    planner_top_n: Dict[str, int] = {}
    retrieval_functional_match_rows_path: Path | None = None
    eval_out: Path | None = None
    candidate_api_rankings_rows_path: Path | None = None
    mode_anomaly_xlsx: Path | None = None
    duplicate_audit_json: Path | None = None
    hallucination_audit_json: Path | None = None
    summary_selected = {"planner_enabled": CONFIG.planner_enabled}
    ranking_failures: List[Dict[str, Any]] = []
    run_status = "completed"
    run_error: str | None = None

    def _record_invalid_mode(metadata: Dict[str, Any], *, mode: str, subtask_id: str) -> List[Dict[str, Any]]:
        nonlocal run_status
        record = _ranking_failure_record(
            metadata=metadata,
            query_id=query_id,
            subtask_id=subtask_id,
            mode=mode,
            out_dir=out_dir,
        )
        ranking_failures.append(record)
        ranked_rows = _write_invalid_ranked(out_dir / mode, subtask_id, record)
        if _is_expected_invalid_evaluation_case(record):
            log_invalid_case_event(record)
        else:
            log_error_event(
                {
                    "event_type": "ranking_mode_failure",
                    **record,
                }
            )
        meta_tracker.update(ranking_failures=ranking_failures)
        if run_status == "completed":
            run_status = "completed_with_warnings"
        log_line(
            f"[{run_label}] subtask={subtask_id} mode={mode} marked invalid "
            f"stage={record.get('failure_stage')} reason={record.get('failure_reason')}"
        )
        return ranked_rows

    try:
        meta_tracker.start_stage("decomposer")
        log_line(f"[{run_label}] starting decomposition")
        try:
            raw_subtasks = decompose_goal(llm_call=lambda p: llm_call("decomposer", DECOMPOSER_SYS, p), user_goal=user_goal)
            subtasks = raw_subtasks
            _write_json(out_dir / "debug" / "0_decomposer_raw.json", raw_subtasks)
            _write_json(out_dir / "0_decomposer.json", subtasks)
            meta_tracker.update(num_subtasks=len(subtasks))
            meta_tracker.finish_stage("decomposer", subtask_count=len(subtasks))
            log_line(f"[{run_label}] finished decomposition ({len(subtasks)} subtasks)")
        except Exception:
            meta_tracker.finish_stage("decomposer", status="failed")
            raise

        run_config = CONFIG.as_dict()
        run_config.update(
            {
                "provider": provider_label,
                "model": backend.name(),
                "active_model": backend.active_model_name(),
                "multi_model_mode": backend.multi_model_mode(),
                "model_pool": backend.model_pool(),
                "query_id": query_id,
                "query_title": query_title,
                "modes": MODE_ORDER,
                "model_usage_file": _to_run_relative(current_model_usage_path(), out_dir),
            }
        )
        _write_json(out_dir / "run_config.json", run_config)

        meta_tracker.start_stage("retrieval")
        log_line(f"[{run_label}] starting shared retrieval")
        try:
            retrieved_by_subtask, no_qos_services, with_qos_services = _build_shared_retrieval(user_goal, subtasks, out_dir)
            total_retrieved = sum(len(items) for items in retrieved_by_subtask.values())
            meta_tracker.finish_stage("retrieval", subtask_count=len(subtasks), retrieved_candidate_count=total_retrieved)
            log_line(f"[{run_label}] finished shared retrieval ({total_retrieved} retrieved candidates)")
        except Exception:
            meta_tracker.finish_stage("retrieval", status="failed")
            raise

        eval_dir = out_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_cache = eval_dir / "functional_match_cache.json"
        meta_tracker.start_stage("retrieval_functional_match_evaluation")
        log_line(f"[{run_label}] starting Functional Candidate Refinement labeling")
        try:
            retrieval_functional_match_rows_path = evaluate_retrieval_functional_match(
                query_dir=out_dir,
                query_id=query_id,
                output_dir=eval_dir,
                cache_path=eval_cache,
                provider=provider or "azure",
                model=model,
            )
            functional_match_map = _load_functional_match_map(retrieval_functional_match_rows_path)
            meta_tracker.finish_stage(
                "retrieval_functional_match_evaluation",
                rows_json=_to_run_relative(retrieval_functional_match_rows_path, out_dir),
            )
            log_line(f"[{run_label}] finished Functional Candidate Refinement labeling")
        except Exception as e:
            (out_dir / "retrieval_functional_match_error.txt").write_text(str(e), encoding="utf-8")
            meta_tracker.finish_stage(
                "retrieval_functional_match_evaluation",
                status="failed",
                error_file="retrieval_functional_match_error.txt",
            )
            if run_status == "completed":
                run_status = "completed_with_warnings"
            log_line(f"[{run_label}] Functional Candidate Refinement labeling failed: {e}")

        meta_tracker.start_stage("ranking")
        log_line(f"[{run_label}] starting ranking stages")
        try:
            for sub in subtasks:
                sub_id = str(sub.get("id", "unknown"))
                retrieved = retrieved_by_subtask[sub_id]
                log_line(f"[{run_label}] subtask={sub_id} starting ranking bundle")

                no_qos_candidates = _candidate_rows(retrieved, no_qos_services)
                try:
                    no_qos_ranked, no_qos_duration = _timed_invocation(
                        meta_tracker,
                        "ranker_no_qos",
                        lambda: rank_subtask(
                            llm_call=lambda p: llm_call("ranker", RANKER_SYS, p),
                            user_query=user_goal,
                            subtask=sub,
                            candidates=no_qos_candidates,
                            prompt_path="prompts/ranker_no_qos.md",
                            debug_raw_path=str(out_dir / "no_qos" / "debug" / f"2_ranker_raw_s{sub_id}.txt"),
                            use_compact_api_evidence=True,
                            include_qos_rank=False,
                        ),
                    )
                    ranked_full_by_mode["no_qos"][sub_id] = _write_ranked(
                        out_dir / "no_qos",
                        sub_id,
                        no_qos_ranked,
                        retrieved,
                        no_qos_services,
                    )
                    log_line(f"[{run_label}] subtask={sub_id} finished no_qos ranking in {no_qos_duration:.2f}s")
                except InvalidRankingOutput as exc:
                    ranked_full_by_mode["no_qos"][sub_id] = _record_invalid_mode(
                        exc.metadata,
                        mode="no_qos",
                        subtask_id=sub_id,
                    )

                pure_qos_candidates = []
                for r in retrieved:
                    api_id = str(r.get("api_id", ""))
                    svc = with_qos_services.get(api_id, {})
                    qos = (svc.get("qos") or {}) if isinstance(svc.get("qos"), dict) else {}
                    pure_qos_candidates.append({
                        "api_id": api_id,
                        "rt_ms": qos.get("rt_ms"),
                        "tp_rps": qos.get("tp_rps"),
                        "availability": qos.get("availability"),
                    })
                try:
                    pure_qos_meta, qos_duration = _timed_invocation(
                        meta_tracker,
                        "qos_scorer",
                        lambda: score_qos_llm(
                            llm_call=lambda p: llm_call("qos_scorer", QOS_SCORER_SYS, p),
                            candidates=pure_qos_candidates,
                            prompt_path="prompts/qos_score_llm.md",
                            debug_raw_path=str(out_dir / "qos_pure_llm" / "debug" / f"1_qos_scores_raw_s{sub_id}.txt"),
                            batch_size=0,
                        ),
                    )
                    _write_json(out_dir / "qos_pure_llm" / "debug" / f"1_qos_scores_s{sub_id}.json", pure_qos_meta)
                    log_line(f"[{run_label}] subtask={sub_id} finished qos scoring in {qos_duration:.2f}s")
                except InvalidQosScoringOutput as exc:
                    _write_json(out_dir / "qos_pure_llm" / "debug" / f"1_qos_scores_s{sub_id}.json", exc.metadata)
                    ranked_full_by_mode["qos_pure_llm"][sub_id] = _record_invalid_mode(
                        exc.metadata,
                        mode="qos_pure_llm",
                        subtask_id=sub_id,
                    )
                else:
                    pure_candidates = _candidate_rows(retrieved, with_qos_services, enrich=pure_qos_meta)
                    try:
                        pure_ranked, pure_rank_duration = _timed_invocation(
                            meta_tracker,
                            "ranker_qos_pure_llm",
                            lambda: rank_subtask(
                                llm_call=lambda p: llm_call("ranker", RANKER_SYS, p),
                                user_query=user_goal,
                                subtask=sub,
                                candidates=pure_candidates,
                                prompt_path="prompts/ranker_qos_pure_llm.md",
                                debug_raw_path=str(out_dir / "qos_pure_llm" / "debug" / f"2_ranker_raw_s{sub_id}.txt"),
                                use_compact_api_evidence=True,
                                include_qos_rank=True,
                            ),
                        )
                        ranked_full_by_mode["qos_pure_llm"][sub_id] = _write_ranked(
                            out_dir / "qos_pure_llm",
                            sub_id,
                            pure_ranked,
                            retrieved,
                            with_qos_services,
                            pure_qos_meta,
                        )
                        log_line(f"[{run_label}] subtask={sub_id} finished qos_pure_llm ranking in {pure_rank_duration:.2f}s")
                    except InvalidRankingOutput as exc:
                        ranked_full_by_mode["qos_pure_llm"][sub_id] = _record_invalid_mode(
                            exc.metadata,
                            mode="qos_pure_llm",
                            subtask_id=sub_id,
                        )

                topsis_meta, topsis_duration = _timed_invocation(
                    meta_tracker,
                    "qos_topsis",
                    lambda: _compute_topsis_metadata(retrieved, with_qos_services),
                )
                topsis_ranked = _deterministic_topsis_ranking(retrieved, topsis_meta)
                ranked_full_by_mode["qos_topsis"][sub_id] = _write_ranked(out_dir / "qos_topsis", sub_id, topsis_ranked, retrieved, with_qos_services, topsis_meta)
                log_line(f"[{run_label}] subtask={sub_id} finished qos_topsis scoring in {topsis_duration:.2f}s")

                hybrid_ranked, hybrid_duration = _timed_invocation(
                    meta_tracker,
                    "qos_hybrid",
                    lambda: _deterministic_hybrid_ranking(retrieved, topsis_meta, sub_id, functional_match_map),
                )
                ranked_full_by_mode["qos_hybrid"][sub_id] = _write_ranked(out_dir / "qos_hybrid", sub_id, hybrid_ranked, retrieved, with_qos_services, topsis_meta)
                log_line(f"[{run_label}] subtask={sub_id} finished qos_hybrid ranking in {hybrid_duration:.2f}s")
                log_line(f"[{run_label}] subtask={sub_id} finished ranking bundle")
            meta_tracker.finish_stage("ranking", subtask_count=len(subtasks), ranking_failure_count=len(ranking_failures))
            log_line(f"[{run_label}] finished ranking stages")
        except Exception:
            meta_tracker.finish_stage("ranking", status="failed")
            raise

        meta_tracker.update(evaluation_triggered=True)
        meta_tracker.start_stage("evaluation_outputs")
        log_line(f"[{run_label}] building final functional match report outputs from retrieval cache")
        try:
            eval_out = evaluate_query(query_dir=out_dir, query_id=query_id, output_dir=eval_dir, cache_path=eval_cache, provider=provider or "azure", model=model)
            log_line(f"[{run_label}] finished building final functional match report outputs")
            candidate_api_rankings_rows_path = eval_dir / f"query_{query_id}_candidate_api_rankings_rows.json"
            if candidate_api_rankings_rows_path.exists():
                planner_top_n = _load_planner_top_n(candidate_api_rankings_rows_path)

            duplicate_audit = collect_duplicate_audit_for_run(out_dir)
            duplicate_audit_json = eval_dir / f"query_{query_id}_duplicate_audit.json"
            _write_json(duplicate_audit_json, duplicate_audit)

            hallucination_audit = collect_hallucination_audit_for_run(out_dir, CONFIG.catalog_no_qos_path)
            hallucination_audit_json = eval_dir / f"query_{query_id}_hallucination_audit.json"
            _write_json(hallucination_audit_json, hallucination_audit)

            mode_anomaly_xlsx = eval_dir / f"query_{query_id}_mode_anomalies.xlsx"
            write_mode_anomaly_excel(duplicate_audit, hallucination_audit, mode_anomaly_xlsx)

            _write_json(
                out_dir / "evaluation_result.json",
                {
                    "evaluation_dir": _to_run_relative(eval_dir, out_dir),
                    "candidate_api_rankings_excel": _to_run_relative(eval_out, out_dir),
                    "candidate_api_rankings_rows_json": _to_run_relative(candidate_api_rankings_rows_path, out_dir),
                    "retrieval_functional_match_rows_json": _to_run_relative(retrieval_functional_match_rows_path, out_dir),
                    "duplicate_audit_json": _to_run_relative(duplicate_audit_json, out_dir),
                    "hallucination_audit_json": _to_run_relative(hallucination_audit_json, out_dir),
                    "mode_anomaly_excel": _to_run_relative(mode_anomaly_xlsx, out_dir),
                    "cache_path": _to_run_relative(eval_cache, out_dir),
                },
            )
            meta_tracker.finish_stage("evaluation_outputs", status="completed")
        except Exception as e:
            (out_dir / "evaluation_error.txt").write_text(str(e), encoding="utf-8")
            meta_tracker.finish_stage("evaluation_outputs", status="failed", error_file="evaluation_error.txt")
            if run_status == "completed":
                run_status = "completed_with_warnings"
            log_line(f"[{run_label}] evaluation output generation failed: {e}")

        if CONFIG.planner_enabled:
            meta_tracker.start_stage("planner")
            log_line(f"[{run_label}] starting planner")
            try:
                for mode in MODE_ORDER:
                    planner_prompt = "prompts/planner_no_qos.md" if mode == "no_qos" else "prompts/planner.md"
                    result = _deterministic_select_and_plan(mode, subtasks, ranked_full_by_mode[mode], planner_top_n, llm_call, user_goal, out_dir, planner_prompt)
                    summary_selected[f"{mode}_selected"] = len(result["selected"])
                meta_tracker.finish_stage("planner", status="completed")
                log_line(f"[{run_label}] finished planner")
            except Exception:
                meta_tracker.finish_stage("planner", status="failed")
                raise
        else:
            summary_selected["planner_skipped_reason"] = "planner disabled in pipeline_config"
            meta_tracker.set_stage("planner", {"status": "skipped", "reason": "planner disabled in pipeline_config"})

        meta_tracker.update(summary=summary_selected)
        log_line(f"Saved run to {out_dir}")
        if eval_out is not None:
            log_line(f"Saved evaluation to {eval_out}")
        return out_dir, eval_out
    except Exception as e:
        run_status = "failed"
        run_error = str(e)
        log_error_event(
            {
                "event_type": "pipeline_failure",
                "query_id": query_id,
                "failure_stage": "pipeline",
                "failure_reason": type(e).__name__,
                "error": str(e),
            }
        )
        log_line(f"[{run_label}] run failed: {e}")
        raise
    finally:
        meta_tracker.update(
            active_model=backend.active_model_name(),
            model_pool=backend.model_pool(),
            model_switches=backend.failover_events(),
        )
        meta_tracker.finish_run(status=run_status, error=run_error)
        clear_run_log()


if __name__ == "__main__":
    queries = choose_queries_interactive(load_queries(ALL_QUERIES_PATH))
    provider, model = choose_provider_and_model_interactive()
    for i, q in enumerate(queries, start=1):
        goal = q.get("goal", "")
        qid = q.get("id", f"q{i:02d}")
        title = q.get("title", "")
        print("\n" + "=" * 80)
        print(f"Running query {i}/{len(queries)} | {qid} | {title}")
        print(f"User goal: {goal}")
        print("=" * 80)
        run_autogen_once(user_goal=goal, provider=provider, model=model, query_id=qid, query_title=title)
