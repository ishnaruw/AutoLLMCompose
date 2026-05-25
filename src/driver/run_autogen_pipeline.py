from __future__ import annotations

from src.core.runtime_bootstrap import harden_scientific_runtime

harden_scientific_runtime()

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.agents.decomposer import decompose_goal
from src.agents.evaluator import EvaluationAgent
from src.agents.functional_refiner import FunctionalRefinerAgent
from src.agents.planner import planner_call
from src.agents.ranker import InvalidRankingOutput, rank_subtask
from src.agents.qos_scorer_llm import InvalidQosScoringOutput, score_qos_llm
from src.agents.retriever import RagRetrieverAgent
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
from src.eval.topsis_eval import _extract_qos, _run_topsis_pydecision
from src.llm.autogen_gateway import call_autogen_gateway
from src.llm.backends import (
    GROQ_MULTI_MODEL_SENTINEL,
    fireworks_model_options,
    groq_experiment_model_pool,
    make_backend,
)
from src.tools.fetch_services import catalog_path, fetch_services

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
    "You are a QoS scoring agent. Given only candidate IDs and QoS metrics, produce a relative QoS-only ranking and score. "
    "Return strict JSON only with one top-level scores array."
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
    "lmstudio_qwen": {"sleep_after_query": 0.0},
    "_default": {"sleep_after_query": 0.4},
}


class PlannerPrecheckFailure(RuntimeError):
    def __init__(self, message: str, payload: Dict[str, Any], selection_trace: Dict[str, Dict[str, Any]]) -> None:
        self.payload = payload
        self.selection_trace = selection_trace
        super().__init__(message)


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
        ("lmstudio", "LM Studio (local, meta-llama-3.1-8b-instruct)"),
        ("lmstudio_qwen", "LM Studio Qwen (local, qwen2.5-3b-instruct.gguf)"),
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


def choose_fireworks_model_interactive() -> str:
    models = fireworks_model_options()
    print("Select Fireworks model:")
    for idx, model_name in enumerate(models, start=1):
        print(f"  {idx}) {model_name}")
    while True:
        choice = input("Enter choice number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            model_name = models[int(choice) - 1]
            print(f"Selected: Fireworks model {model_name}\n")
            return model_name
        print("Invalid choice. Try again.")


def choose_provider_and_model_interactive() -> tuple[str, str | None]:
    provider = choose_provider_interactive()
    if provider == "groq":
        return provider, choose_groq_model_interactive()
    if provider in {"fireworks", "fireworks_ai"}:
        return provider, choose_fireworks_model_interactive()
    return provider, None


def _query_lookup(queries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        str(query.get("id", f"q{idx:02d}")).lower(): query
        for idx, query in enumerate(queries, start=1)
    }


def _select_queries_by_ids(queries: List[Dict[str, Any]], raw_ids: str) -> List[Dict[str, Any]]:
    query_ids = [part.strip().lower() for part in (raw_ids or "").split(",") if part.strip()]
    if not query_ids:
        raise ValueError("At least one query id is required.")
    lookup = _query_lookup(queries)
    missing = [query_id for query_id in query_ids if query_id not in lookup]
    if missing:
        raise ValueError(f"Unknown query id(s): {', '.join(missing)}")
    return [lookup[query_id] for query_id in query_ids]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AutoLLMCompose query-level pipeline experiments.")
    parser.add_argument(
        "--query-ids",
        help="Comma-separated query ids to run, such as q01,q03. If omitted, interactive selection is used.",
    )
    parser.add_argument("--query-id", action="append", help="Query id to run. Can be passed multiple times.")
    parser.add_argument("--provider", help="LLM provider, such as mistral, fireworks, groq, lmstudio, or lmstudio_qwen.")
    parser.add_argument("--model", help="Model name for the selected provider.")
    parser.add_argument("--run-tag", help="Optional folder under results/logs for this batch.")
    parser.add_argument("--queries-path", default=str(ALL_QUERIES_PATH), help="Path to JSONL query file.")
    return parser.parse_args()


def _run_from_args(args: argparse.Namespace) -> None:
    queries = load_queries(Path(args.queries_path))
    raw_query_ids = args.query_ids
    if args.query_id:
        raw_query_ids = ",".join(args.query_id if raw_query_ids is None else [raw_query_ids, *args.query_id])

    selected_queries = _select_queries_by_ids(queries, raw_query_ids or "")
    provider = args.provider
    model = args.model
    for i, q in enumerate(selected_queries, start=1):
        goal = q.get("goal", "")
        qid = q.get("id", f"q{i:02d}")
        title = q.get("title", "")
        print("\n" + "=" * 80)
        print(f"Running query {i}/{len(selected_queries)} | {qid} | {title}")
        print(f"User goal: {goal}")
        print("=" * 80)
        run_autogen_once(
            user_goal=goal,
            provider=provider,
            model=model,
            query_id=qid,
            query_title=title,
            run_tag=args.run_tag,
        )


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
        provider = getattr(backend, "provider", "")
        if provider not in {"lmstudio", "lmstudio_qwen"}:
            return limits
        limits["timeout_seconds"] = CONFIG.lmstudio_timeout_seconds
        if role_name.startswith("ranker") and provider == "lmstudio":
            limits["max_tokens"] = CONFIG.lmstudio_ranker_max_tokens
        return limits

    def _invoke(fn, *, name: str) -> str:
        if getattr(backend, "multi_model_mode", lambda: False)():
            return fn()
        max_retries = CONFIG.planner_max_retries if name.startswith("planner") else 8
        return call_with_backoff(fn, name=name, max_retries=max_retries)

    def llm_call(role_name: str, system_msg: str, prompt: str) -> str:
        temp = CONFIG.planner_temperature if role_name.startswith("planner") else 0.0
        limits = _lmstudio_limits(role_name)
        if role_name.startswith("planner"):
            limits["timeout_seconds"] = CONFIG.planner_timeout_seconds
        elif role_name.startswith("ranker"):
            limits["timeout_seconds"] = CONFIG.ranker_timeout_seconds
        elif role_name.startswith("qos_scorer"):
            limits["timeout_seconds"] = CONFIG.qos_scorer_timeout_seconds
        return _invoke(
            lambda: call_autogen_gateway(
                backend=backend,
                role_name=role_name,
                system_message=system_msg,
                user_prompt=prompt,
                temperature=temp,
                force_json=True,
                max_tokens=limits.get("max_tokens"),
                timeout_s=limits.get("timeout_seconds"),
                metadata={"source": "run_autogen_pipeline"},
            ),
            name=role_name,
        )
    return llm_call


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _merge_json_file(path: Path, updates: Dict[str, Any]) -> None:
    payload: Dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except Exception:
            payload = {}
    payload.update(updates)
    _write_json(path, payload)


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
    retriever_agent = RagRetrieverAgent(index_dir=str(CONFIG.shared_index_dir))
    for sub in subtasks:
        sub_id = str(sub.get("id", "unknown"))
        retrieved = retriever_agent.retrieve(str(sub.get("description", "")), top_k=CONFIG.rag_top_k)
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
    ranked_full = [item for item in ranked if str(item.get("api_id", "")).strip()]
    if ranked_full and all(item.get("llm_reported_rank") is not None for item in ranked_full):
        ranked_full = sorted(ranked_full, key=lambda item: int(item.get("llm_reported_rank")))
    passthrough_keys = [
        "ranking_anomaly",
        "ranking_anomaly_reason",
        "ranking_anomaly_stage",
        "candidate_id",
        "expected_api_count",
        "expected_candidate_count",
        "actual_api_count",
        "actual_candidate_count",
        "returned_api_count",
        "returned_candidate_count",
        "expected_rank_count",
        "actual_rank_count",
        "returned_rank_count",
        "missing_rank_values",
        "missing_rank_candidate_ids",
        "missing_rank_api_ids",
        "non_integer_rank_values",
        "non_integer_rank_candidate_ids",
        "non_integer_rank_api_ids",
        "duplicate_rank_values",
        "rank_values_out_of_range",
        "duplicate_candidate_ids",
        "duplicate_api_ids",
        "missing_candidate_ids",
        "missing_api_ids",
        "unknown_candidate_ids",
        "unknown_api_ids",
        "is_unknown_api_id",
        "is_unknown_candidate_id",
    ]

    for idx, item in enumerate(ranked_full, start=1):
        api_id = str(item.get("api_id", ""))
        base = rag_map.get(api_id, {})
        try:
            mode_rank = int(item.get("llm_reported_rank"))
        except Exception:
            mode_rank = idx
        row = {
            "api_id": api_id,
            "candidate_id": item.get("candidate_id", ""),
            "mode_rank": mode_rank,
            "llm_reported_rank": item.get("llm_reported_rank"),
            "retrieved_rank": base.get("retrieved_rank"),
            "rag_score": base.get("rag_score"),
            "reason": item.get("reason", ""),
            "service": id_to_service.get(api_id, {}),
        }
        for key in passthrough_keys:
            if key in item:
                row[key] = item.get(key)
        if extras and api_id in extras:
            row.update(extras[api_id])
        full.append(row)
    _write_json(mode_dir / f"2_ranked_s{sub_id}.json", full)
    return full


def _coerce_optional_int(value: Any) -> int | None:
    try:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return int(value)
    except Exception:
        return None


def _write_invalid_ranked(
    mode_dir: Path,
    sub_id: str,
    failure: Dict[str, Any],
    invalid_ranked_items: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    failure_fields = {
        **failure,
        "ranking_anomaly": True,
        "ranking_anomaly_reason": failure.get("failure_reason", "invalid_ranking_case"),
        "ranking_anomaly_stage": failure.get("failure_stage", "llm_ranking"),
        "invalid_output_row": True,
    }
    rows: List[Dict[str, Any]] = []
    for returned_position, item in enumerate(invalid_ranked_items or [], start=1):
        api_id = str(item.get("api_id", "")).strip()
        reported_rank = item.get("llm_reported_rank")
        item_passthrough = {
            key: item.get(key)
            for key in [
                "functional_reason",
                "qos_reason",
                "is_unknown_api_id",
                "is_unknown_candidate_id",
            ]
            if key in item
        }
        rows.append(
            {
                "api_id": api_id,
                "candidate_id": item.get("candidate_id", ""),
                "mode_rank": _coerce_optional_int(reported_rank) or returned_position,
                "llm_reported_rank": reported_rank,
                "returned_position": returned_position,
                "retrieved_rank": None,
                "rag_score": None,
                "reason": item.get("reason") or failure.get("failure_reason", "invalid_ranking_case"),
                "service": {},
                **item_passthrough,
                **failure_fields,
            }
        )

    if not rows:
        rows = [
            {
                "api_id": "",
                "mode_rank": None,
                "retrieved_rank": None,
                "rag_score": None,
                "reason": failure.get("failure_reason", "invalid_ranking_case"),
                "service": {},
                **failure_fields,
            }
        ]

    _write_json(mode_dir / f"2_ranked_s{sub_id}.json", rows)
    _write_json(mode_dir / "debug" / f"failure_s{sub_id}.json", {"failure": failure, "rows": rows})
    return rows


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
    stage = str(record.get("failure_stage") or "")
    reason = str(record.get("failure_reason") or "")
    if reason.endswith("_after_retries"):
        reason = reason[: -len("_after_retries")]
    if record.get("error") and reason not in {"timeout", "llm_transport_error", "groq_prompt_too_large"}:
        return False
    expected_reasons = {
        "empty_response",
        "parse_error",
        "invalid_json",
        "llm_transport_error",
        "llm_call_error",
        "missing_required_key",
        "wrong_json_type",
        "timeout",
        "duplicate_candidate_ids",
        "duplicate_ranked_apis",
        "duplicate_api_ids",
        "unknown_candidate_ids",
        "unknown_api_ids",
        "incomplete_candidate_id_list",
        "missing_candidate_ids",
        "incomplete_ranked_api_list",
        "missing_ranked_apis",
        "missing_rank_values",
        "duplicate_rank_values",
        "non_integer_rank_values",
        "rank_values_out_of_range",
        "incomplete_rank_sequence",
        "invalid_rank_sequence",
        "groq_prompt_too_large",
        "incomplete_qos_scores",
        "missing_api_scores",
        "missing_score",
        "invalid_score_range",
        "invalid_score_value",
        "qos_score_formula_mismatch",
        "missing_label",
        "invalid_label_value",
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


def _load_planner_top_n_from_retrieval_match(rows_path: Path) -> Dict[str, int]:
    data = json.loads(rows_path.read_text(encoding="utf-8"))
    matching_api_ids_by_subtask: Dict[str, set[str]] = {}

    for row in data:
        sub_id = str(row.get("Sub Task", row.get("subtask_id", "")))
        api_id = str(row.get("Selected_API", row.get("api_id", "")))

        if not sub_id or not api_id:
            continue

        if _functional_match_value(row) == 1:
            matching_api_ids_by_subtask.setdefault(sub_id, set()).add(api_id)

    return {
        sub_id: len(api_ids)
        for sub_id, api_ids in matching_api_ids_by_subtask.items()
    }


def _load_planner_top_n(rows_path: Path) -> Dict[str, int]:
    # Deprecated: planner K should come from retrieval_functional_match_rows_json,
    # not mode-expanded candidate_api_rankings_rows_json.
    return _load_planner_top_n_from_retrieval_match(rows_path)


def _summarize_retry_outcomes(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "retry_outcomes.log"
    summary: Dict[str, Any] = {
        "total_events": 0,
        "successes": 0,
        "failures": 0,
        "by_stage": {},
        "by_role": {},
        "max_attempts": CONFIG.llm_validation_max_retries + 1,
        "max_validation_retries": CONFIG.llm_validation_max_retries,
        "log_file": _to_run_relative(path, run_dir) if path.exists() else None,
    }
    if not path.exists():
        return summary

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except Exception:
            continue
        outcome = str(event.get("outcome", ""))
        stage = str(event.get("stage", "unknown"))
        role = str(event.get("role", "unknown"))
        summary["total_events"] += 1
        if outcome == "success":
            summary["successes"] += 1
        elif outcome == "failed":
            summary["failures"] += 1

        for bucket_name, key in [("by_stage", stage), ("by_role", role)]:
            bucket = summary[bucket_name].setdefault(key, {"successes": 0, "failures": 0, "total_events": 0})
            bucket["total_events"] += 1
            if outcome == "success":
                bucket["successes"] += 1
            elif outcome == "failed":
                bucket["failures"] += 1

    total = int(summary["successes"]) + int(summary["failures"])
    summary["success_rate"] = round(float(summary["successes"]) / total, 4) if total else None
    summary["failure_rate"] = round(float(summary["failures"]) / total, 4) if total else None
    return summary


def _deterministic_select_and_plan(mode_name: str, subtasks: List[Dict[str, Any]], ranked_full: Dict[str, List[Dict[str, Any]]], planner_top_n: Dict[str, int], llm_call, user_goal: str, out_dir: Path, planner_prompt_path: str) -> Dict[str, Any]:
    mode_dir = out_dir / mode_name
    selected_all = []
    selection_trace: Dict[str, Dict[str, Any]] = {}
    missing_due_to_ranking_failure: List[str] = []
    subtask_failure_reasons: Dict[str, str] = {}
    for sub in subtasks:
        sub_id = str(sub.get("id", "unknown"))
        ranked_rows = ranked_full.get(sub_id, [])
        dynamic_top_n = int(planner_top_n.get(sub_id, 0) or 0)
        fallback_used = dynamic_top_n <= 0
        selected_limit = dynamic_top_n if dynamic_top_n > 0 else CONFIG.selector_top_n
        requested_limit = selected_limit
        selected_limit = min(selected_limit, len(ranked_rows))
        selected_rows = ranked_rows[:selected_limit]
        top_n_source = "retrieval_functional_match_k" if dynamic_top_n > 0 else "selector_top_n_fallback"
        selected = []
        for idx, r in enumerate(selected_rows, start=1):
            if r.get("failure_flag") or r.get("invalid_output_row"):
                continue
            api_id = str(r.get("api_id") or "").strip()
            if not api_id:
                continue
            row = dict(r)
            row["selected_rank"] = idx
            row["selection_order"] = idx
            row.pop("score", None)
            row["subtask_id"] = sub_id
            row["selector_reason"] = "Deterministic selection from mode rank using the shared retrieval-level functional-match K."
            row["planner_top_n"] = selected_limit
            row["planner_requested_top_n"] = requested_limit
            row["planner_top_n_source"] = top_n_source
            row["fallback_used"] = fallback_used
            selected.append(row)
        selected_all.extend(selected)
        invalid_ranked_reasons = [
            str(r.get("failure_reason") or r.get("ranking_anomaly_reason") or "ranking_failure")
            for r in ranked_rows
            if isinstance(r, dict) and (r.get("failure_flag") or r.get("invalid_output_row"))
        ]
        if not selected and invalid_ranked_reasons:
            missing_due_to_ranking_failure.append(sub_id)
            subtask_failure_reasons[sub_id] = invalid_ranked_reasons[0]
        _write_json(mode_dir / f"3_selected_s{sub_id}.json", selected)
        _write_json(
            mode_dir / f"3_selected_trace_s{sub_id}.json",
            {
                "planner_top_n": selected_limit,
                "planner_requested_top_n": requested_limit,
                "planner_top_n_source": top_n_source,
                "retrieval_functional_match_k": dynamic_top_n,
                "fallback_used": fallback_used,
                "available_ranked": len(ranked_rows),
                "selected_count": len(selected),
                "invalid_ranked_rows": len(invalid_ranked_reasons),
            },
        )
        selection_trace[sub_id] = {
            "planner_top_k": selected_limit,
            "planner_requested_top_k": requested_limit,
            "planner_top_k_source": top_n_source,
            "retrieval_functional_match_k": dynamic_top_n,
            "fallback_used": fallback_used,
            "available_ranked": len(ranked_rows),
            "selected_count": len(selected),
            "invalid_ranked_rows": len(invalid_ranked_reasons),
        }
    if missing_due_to_ranking_failure:
        missing_text = ", ".join(missing_due_to_ranking_failure)
        message = (
            f"Skipping planner for mode {mode_name} because selected APIs are missing for "
            f"required subtask(s): {missing_text}. Cause: upstream ranking failure."
        )
        failure_payload = {
            "failure_stage": "planner_precheck",
            "failure_reason": "missing_selected_apis_due_to_ranking_failure",
            "mode": mode_name,
            "missing_subtask_ids": missing_due_to_ranking_failure,
            "subtask_failure_reasons": subtask_failure_reasons,
            "planner_called": False,
        }
        _write_json(mode_dir / "planner_failure.json", failure_payload)
        (mode_dir / "planner_error.txt").write_text(message, encoding="utf-8")
        raise PlannerPrecheckFailure(message, failure_payload, selection_trace)
    planner_role = f"planner_{mode_name}"
    planner = planner_call(llm_call=lambda p: llm_call(planner_role, PLANNER_SYS, p), user_goal=user_goal, ranked_top=selected_all, subtasks=subtasks, prompt_path=planner_prompt_path)
    _write_json(mode_dir / "4_planner.json", planner)
    return {"selected": selected_all, "planner": planner, "selection_trace": selection_trace}


def _planner_selection_k_summary(query_id: str | None, subtasks: List[Dict[str, Any]], by_mode: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    by_subtask: Dict[str, Dict[str, Any]] = {}
    for sub in subtasks:
        sub_id = str(sub.get("id", "unknown"))
        by_subtask[sub_id] = {
            mode: mode_trace.get(sub_id, {})
            for mode, mode_trace in by_mode.items()
        }
    return {
        "query_id": query_id,
        "selection_stage": "planner_input_selection",
        "selection_rule": "For each subtask, K equals the number of unique APIs in the shared retrieved candidate pool labeled Functional Match=1. If K=0, CONFIG.selector_top_n is used as fallback. Each mode then passes its own top-K ranked APIs to the planner.",
        "fallback_selector_top_n": CONFIG.selector_top_n,
        "by_mode": by_mode,
        "by_subtask": by_subtask,
    }


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
            "composition_qos_eval_enabled": CONFIG.composition_qos_eval_enabled,
            "modes": MODE_ORDER,
            "llm_validation_policy": {
                "max_validation_retries": CONFIG.llm_validation_max_retries,
                "max_attempts": CONFIG.llm_validation_max_retries + 1,
                "purpose": "bounded recovery for structurally invalid LLM outputs",
            },
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
    functional_refinement_summary_path: Path | None = None
    eval_out: Path | None = None
    mode_anomaly_xlsx: Path | None = None
    ranking_anomaly_audit_json: Path | None = None
    duplicate_audit_json: Path | None = None
    hallucination_audit_json: Path | None = None
    composition_qos_rows_json: Path | None = None
    composition_qos_summary_json: Path | None = None
    composition_qos_excel: Path | None = None
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
        invalid_ranked_items = record.pop("_invalid_ranked_items", None)
        ranking_failures.append(record)
        _write_invalid_ranked(
            out_dir / mode,
            subtask_id,
            record,
            invalid_ranked_items=invalid_ranked_items if isinstance(invalid_ranked_items, list) else None,
        )
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
        return []

    try:
        meta_tracker.start_stage("decomposer")
        log_line(f"[{run_label}] starting decomposition")
        try:
            raw_subtasks = decompose_goal(llm_call=lambda p: llm_call("decomposer", DECOMPOSER_SYS, p), user_goal=user_goal)
            subtasks = raw_subtasks
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
                "retrieval_agent": "RagRetrieverAgent",
                "retrieval_agent_mode": "deterministic_faiss",
                "functional_refiner_agent": "FunctionalRefinerAgent",
                "functional_refiner_agent_mode": "llm_binary_functional_labeling",
                "evaluation_agent": "EvaluationAgent",
                "evaluation_agent_mode": "deterministic_adapter",
                "model_usage_file": _to_run_relative(current_model_usage_path(), out_dir),
            }
        )
        _write_json(out_dir / "run_config.json", run_config)

        meta_tracker.start_stage("retrieval")
        log_line(f"[{run_label}] starting shared retrieval")
        try:
            retrieved_by_subtask, no_qos_services, with_qos_services = _build_shared_retrieval(user_goal, subtasks, out_dir)
            total_retrieved = sum(len(items) for items in retrieved_by_subtask.values())
            meta_tracker.finish_stage(
                "retrieval",
                subtask_count=len(subtasks),
                retrieved_candidate_count=total_retrieved,
                retrieval_agent="RagRetrieverAgent",
                retrieval_agent_mode="deterministic_faiss",
            )
            log_line(f"[{run_label}] finished shared retrieval ({total_retrieved} retrieved candidates)")
        except Exception:
            meta_tracker.finish_stage("retrieval", status="failed")
            raise

        eval_dir = out_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_cache = eval_dir / "functional_match_cache.json"
        evaluation_agent = EvaluationAgent(catalog_no_qos_path=catalog_path(with_qos=False))
        functional_refiner_agent = FunctionalRefinerAgent()
        functional_refinement_summary_path = eval_dir / f"query_{query_id or out_dir.name}_functional_refinement_summary.json"
        if CONFIG.functional_refinement_enabled:
            meta_tracker.start_stage("functional_refinement")
            log_line(f"[{run_label}] starting Functional Refiner Agent")
            try:
                retrieval_functional_match_rows_path = functional_refiner_agent.refine_candidates(
                    query_dir=out_dir,
                    query_id=query_id,
                    output_dir=eval_dir,
                    cache_path=eval_cache,
                    provider=provider or "azure",
                    model=model,
                )
                functional_match_map = _load_functional_match_map(retrieval_functional_match_rows_path)
                meta_tracker.update(
                    functional_refinement_status="completed",
                    functional_refinement_rows_json=_to_run_relative(retrieval_functional_match_rows_path, out_dir),
                    functional_refinement_summary_json=_to_run_relative(functional_refinement_summary_path, out_dir)
                    if functional_refinement_summary_path.exists()
                    else None,
                )
                meta_tracker.finish_stage(
                    "functional_refinement",
                    rows_json=_to_run_relative(retrieval_functional_match_rows_path, out_dir),
                    summary_json=_to_run_relative(functional_refinement_summary_path, out_dir)
                    if functional_refinement_summary_path.exists()
                    else None,
                    functional_refiner_agent="FunctionalRefinerAgent",
                    functional_refiner_agent_mode="llm_binary_functional_labeling",
                    functional_refinement_status="completed",
                )
                log_line(f"[{run_label}] finished Functional Refiner Agent")
            except Exception as e:
                (out_dir / "functional_refinement_error.txt").write_text(str(e), encoding="utf-8")
                meta_tracker.update(functional_refinement_status="failed")
                meta_tracker.finish_stage(
                    "functional_refinement",
                    status="failed",
                    error_file="functional_refinement_error.txt",
                    functional_refiner_agent="FunctionalRefinerAgent",
                    functional_refiner_agent_mode="llm_binary_functional_labeling",
                    functional_refinement_status="failed",
                )
                if run_status == "completed":
                    run_status = "completed_with_warnings"
                log_line(f"[{run_label}] Functional Refiner Agent failed: {e}")
        else:
            meta_tracker.update(functional_refinement_status="skipped")
            meta_tracker.set_stage(
                "functional_refinement",
                {
                    "status": "skipped",
                    "reason": "functional_refinement_enabled is false",
                    "functional_refiner_agent": "FunctionalRefinerAgent",
                    "functional_refinement_status": "skipped",
                },
            )
            log_line(f"[{run_label}] skipped Functional Refiner Agent")

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
                            llm_call=lambda p: llm_call("ranker_no_qos", RANKER_SYS, p),
                            user_query=user_goal,
                            subtask=sub,
                            candidates=no_qos_candidates,
                            prompt_path="prompts/ranker_no_qos.md",
                            debug_raw_path=None,
                            use_compact_api_evidence=True,
                            include_qos_rank=False,
                            max_validation_retries=CONFIG.llm_validation_max_retries,
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
                        "rt_s": qos.get("rt_s"),
                        "tp_kbps": qos.get("tp_kbps"),
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
                            debug_raw_path=None,
                            batch_size=CONFIG.qos_llm_batch_size,
                            max_validation_retries=CONFIG.llm_validation_max_retries,
                            validate_formula=CONFIG.qos_llm_validate_formula,
                            formula_audit=CONFIG.qos_llm_formula_audit,
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
                                llm_call=lambda p: llm_call("ranker_qos_pure_llm", RANKER_SYS, p),
                                user_query=user_goal,
                                subtask=sub,
                                candidates=pure_candidates,
                                prompt_path="prompts/ranker_qos_pure_llm.md",
                                debug_raw_path=None,
                                use_compact_api_evidence=True,
                                include_qos_rank=True,
                                max_validation_retries=CONFIG.llm_validation_max_retries,
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
            retry_summary = _summarize_retry_outcomes(out_dir)
            meta_tracker.update(retry_summary=retry_summary)
            meta_tracker.finish_stage(
                "ranking",
                subtask_count=len(subtasks),
                ranking_failure_count=len(ranking_failures),
                retry_summary=retry_summary,
            )
            log_line(f"[{run_label}] finished ranking stages")
        except Exception:
            meta_tracker.finish_stage("ranking", status="failed")
            raise

        meta_tracker.update(evaluation_triggered=True)
        meta_tracker.start_stage("evaluation_outputs")
        log_line(f"[{run_label}] building final functional match report outputs from retrieval cache")
        try:
            evaluation_outputs = evaluation_agent.build_evaluation_outputs(
                query_dir=out_dir,
                query_id=query_id,
                output_dir=eval_dir,
                cache_path=eval_cache,
                provider=provider or "azure",
                model=model,
                retrieval_functional_match_rows_path=retrieval_functional_match_rows_path,
            )
            eval_out = evaluation_outputs["candidate_api_rankings_excel"]
            log_line(f"[{run_label}] finished building final functional match report outputs")
            retrieval_functional_match_rows_path = evaluation_outputs["retrieval_functional_match_rows_json"]
            if retrieval_functional_match_rows_path and retrieval_functional_match_rows_path.exists():
                planner_top_n = _load_planner_top_n_from_retrieval_match(retrieval_functional_match_rows_path)

            _write_json(
                out_dir / "evaluation_result.json",
                {
                    "evaluation_dir": _to_run_relative(evaluation_outputs["evaluation_dir"], out_dir),
                    "candidate_api_rankings_excel": _to_run_relative(evaluation_outputs["candidate_api_rankings_excel"], out_dir),
                    "candidate_api_rankings_rows_json": _to_run_relative(evaluation_outputs["candidate_api_rankings_rows_json"], out_dir),
                    "retrieval_functional_match_rows_json": _to_run_relative(evaluation_outputs["retrieval_functional_match_rows_json"], out_dir)
                    if evaluation_outputs["retrieval_functional_match_rows_json"]
                    else None,
                    "duplicate_audit_json": _to_run_relative(evaluation_outputs["duplicate_audit_json"], out_dir),
                    "hallucination_audit_json": _to_run_relative(evaluation_outputs["hallucination_audit_json"], out_dir),
                    "ranking_anomaly_audit_json": _to_run_relative(evaluation_outputs["ranking_anomaly_audit_json"], out_dir),
                    "mode_anomaly_excel": _to_run_relative(evaluation_outputs["mode_anomaly_excel"], out_dir),
                    "cache_path": _to_run_relative(evaluation_outputs["cache_path"], out_dir),
                    "evaluation_agent": "EvaluationAgent",
                    "evaluation_agent_mode": "deterministic_adapter",
                },
            )
            meta_tracker.finish_stage(
                "evaluation_outputs",
                status="completed",
                evaluation_agent="EvaluationAgent",
                evaluation_agent_mode="deterministic_adapter",
            )
        except Exception as e:
            (out_dir / "evaluation_error.txt").write_text(str(e), encoding="utf-8")
            meta_tracker.finish_stage("evaluation_outputs", status="failed", error_file="evaluation_error.txt")
            if run_status == "completed":
                run_status = "completed_with_warnings"
            log_line(f"[{run_label}] evaluation output generation failed: {e}")

        if CONFIG.planner_enabled:
            meta_tracker.start_stage("planner")
            log_line(f"[{run_label}] starting planner")
            planner_failures: List[Dict[str, Any]] = []
            planner_selection_k_by_mode_subtask: Dict[str, Dict[str, Dict[str, Any]]] = {}
            planner_selection_k_summary_path = eval_dir / f"query_{query_id}_planner_selection_k_summary.json"
            for mode in MODE_ORDER:
                try:
                    planner_prompt = "prompts/planner_no_qos.md" if mode == "no_qos" else "prompts/planner_qos.md"
                    result = _deterministic_select_and_plan(mode, subtasks, ranked_full_by_mode[mode], planner_top_n, llm_call, user_goal, out_dir, planner_prompt)
                    summary_selected[f"{mode}_selected"] = len(result["selected"])
                    planner_selection_k_by_mode_subtask[mode] = result.get("selection_trace", {})
                except Exception as exc:
                    if isinstance(exc, PlannerPrecheckFailure):
                        failure = {**exc.payload, "error": str(exc)}
                        planner_selection_k_by_mode_subtask[mode] = exc.selection_trace
                    else:
                        failure = {"mode": mode, "failure_reason": type(exc).__name__, "error": str(exc)}
                    planner_failures.append(failure)
                    summary_selected[f"{mode}_planner_status"] = "failed"
                    summary_selected[f"{mode}_planner_error"] = str(exc)
                    (out_dir / mode / "planner_error.txt").parent.mkdir(parents=True, exist_ok=True)
                    (out_dir / mode / "planner_error.txt").write_text(str(exc), encoding="utf-8")
                    if run_status == "completed":
                        run_status = "completed_with_warnings"
                    log_error_event(
                        {
                            "event_type": "planner_mode_failure",
                            "query_id": query_id,
                            "mode": mode,
                            "failure_stage": failure.get("failure_stage", "planner"),
                            **failure,
                        }
                    )
                    log_line(f"[{run_label}] mode={mode} planner failed: {exc}")
            planner_selection_k_summary = _planner_selection_k_summary(query_id, subtasks, planner_selection_k_by_mode_subtask)
            _write_json(planner_selection_k_summary_path, planner_selection_k_summary)
            _merge_json_file(
                out_dir / "evaluation_result.json",
                {
                    "planner_selection_k_summary_json": _to_run_relative(planner_selection_k_summary_path, out_dir),
                    "planner_selection_k_by_mode_subtask": planner_selection_k_by_mode_subtask,
                },
            )
            summary_selected["planner_selection_k_summary_json"] = _to_run_relative(planner_selection_k_summary_path, out_dir)
            log_line(
                f"[{run_label}] planner selection K summary saved to "
                f"{_to_run_relative(planner_selection_k_summary_path, out_dir)}"
            )
            if planner_failures:
                meta_tracker.finish_stage(
                    "planner",
                    status="completed_with_warnings",
                    failure_count=len(planner_failures),
                    failures=planner_failures,
                    planner_selection_k_summary_json=_to_run_relative(planner_selection_k_summary_path, out_dir),
                    planner_selection_k_by_mode_subtask=planner_selection_k_by_mode_subtask,
                )
            else:
                meta_tracker.finish_stage(
                    "planner",
                    status="completed",
                    planner_selection_k_summary_json=_to_run_relative(planner_selection_k_summary_path, out_dir),
                    planner_selection_k_by_mode_subtask=planner_selection_k_by_mode_subtask,
                )
            log_line(f"[{run_label}] finished planner")
        else:
            summary_selected["planner_skipped_reason"] = "planner disabled in pipeline_config"
            meta_tracker.set_stage("planner", {"status": "skipped", "reason": "planner disabled in pipeline_config"})

        if CONFIG.planner_enabled and CONFIG.composition_qos_eval_enabled:
            meta_tracker.start_stage("composition_qos_evaluation")
            log_line(f"[{run_label}] starting composition QoS evaluation")
            try:
                composition_result = evaluation_agent.evaluate_composition_qos(query_dir=out_dir, query_id=query_id, output_dir=out_dir / "evaluation")
                composition_qos_rows_json = composition_result.get("rows_json")
                composition_qos_summary_json = composition_result.get("summary_json")
                composition_qos_excel = composition_result.get("excel")
                composition_validity_issues_json = composition_result.get("composition_validity_issues_json")
                composition_validity_issues_log = composition_result.get("composition_validity_issues_log")
                composition_validity_summary = composition_result.get("composition_validity_summary")
                _merge_json_file(
                    out_dir / "evaluation_result.json",
                    {
                        "composition_qos_eval_rows_json": _to_run_relative(composition_qos_rows_json, out_dir),
                        "composition_qos_eval_summary_json": _to_run_relative(composition_qos_summary_json, out_dir),
                        "composition_qos_eval_excel": _to_run_relative(composition_qos_excel, out_dir),
                        "composition_validity_issues_json": _to_run_relative(composition_validity_issues_json, out_dir),
                        "composition_validity_issues_log": _to_run_relative(composition_validity_issues_log, out_dir),
                        "composition_validity_summary": composition_validity_summary,
                        "evaluation_agent": "EvaluationAgent",
                        "evaluation_agent_mode": "deterministic_adapter",
                    },
                )
                meta_tracker.finish_stage(
                    "composition_qos_evaluation",
                    status="completed",
                    rows_json=_to_run_relative(composition_qos_rows_json, out_dir),
                    summary_json=_to_run_relative(composition_qos_summary_json, out_dir),
                    excel=_to_run_relative(composition_qos_excel, out_dir),
                    issues_json=_to_run_relative(composition_validity_issues_json, out_dir),
                    issues_log=_to_run_relative(composition_validity_issues_log, out_dir),
                    composition_validity_summary=composition_validity_summary,
                    evaluation_agent="EvaluationAgent",
                    evaluation_agent_mode="deterministic_adapter",
                )
                log_line(f"[{run_label}] finished composition QoS evaluation")
            except Exception as e:
                (out_dir / "composition_qos_eval_error.txt").write_text(str(e), encoding="utf-8")
                _merge_json_file(
                    out_dir / "evaluation_result.json",
                    {
                        "composition_qos_eval_skipped_reason": None,
                        "composition_qos_eval_error": str(e),
                    },
                )
                meta_tracker.finish_stage("composition_qos_evaluation", status="failed", error_file="composition_qos_eval_error.txt")
                if run_status == "completed":
                    run_status = "completed_with_warnings"
                log_line(f"[{run_label}] composition QoS evaluation failed: {e}")
        else:
            if not CONFIG.planner_enabled:
                skipped_reason = "planner disabled in pipeline_config"
            else:
                skipped_reason = "composition_qos_eval disabled in pipeline_config"
            summary_selected["composition_qos_eval_skipped_reason"] = skipped_reason
            _merge_json_file(
                out_dir / "evaluation_result.json",
                {
                    "composition_qos_eval_skipped_reason": skipped_reason,
                },
            )
            meta_tracker.set_stage("composition_qos_evaluation", {"status": "skipped", "reason": skipped_reason})

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
    args = _parse_args()
    if args.query_ids or args.query_id:
        _run_from_args(args)
    else:
        queries = choose_queries_interactive(load_queries(Path(args.queries_path)))
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
