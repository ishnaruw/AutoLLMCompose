from __future__ import annotations

from src.core.runtime_bootstrap import harden_scientific_runtime

harden_scientific_runtime()

import argparse
import json
import math
import re
import time
from datetime import datetime
from itertools import product
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
    log_warning_event,
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
FUNCTIONAL_REFINER_MODES = {"qos_hybrid"}
TOPSIS_METADATA_MODES = {"qos_topsis", "qos_hybrid"}
ALL_QUERIES_PATH = Path("data/queries/all_user_query.jsonl")
PLANNER_CANDIDATE_MODES = {"fixed_one", "top_n_ablation"}
HYBRID_WORKFLOW_SELECTORS = {"workflow_topsis", "relative_to_best"}
WORKFLOW_TOPSIS_WEIGHTS = {
    "response_time": 1.0 / 3.0,
    "throughput": 1.0 / 3.0,
    "availability": 1.0 / 3.0,
}

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


class PlannerSelectionValidationFailure(RuntimeError):
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


def _copy_retrieved_by_subtask(retrieved_by_subtask: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    return {
        str(subtask_id): [dict(item) for item in items]
        for subtask_id, items in retrieved_by_subtask.items()
    }


def _write_retrieval_view(view_dir: Path, subtasks: List[Dict[str, Any]], retrieved_by_subtask: Dict[str, List[Dict[str, Any]]]) -> None:
    for sub in subtasks:
        sub_id = str(sub.get("id", "unknown"))
        _write_json(view_dir / f"1_retriever_s{sub_id}.json", retrieved_by_subtask.get(sub_id, []))


def _zero_functional_retrieval_retry_query(user_goal: str, subtask: Dict[str, Any]) -> str:
    description = str(subtask.get("description") or "").strip()
    base = " ".join(part for part in [description, str(user_goal or "").strip()] if part)
    text = base.lower()
    expansion_terms = [
        "direct endpoint required action domain dataset result output functional match",
        "analysis prediction classification recommendation score signal indicator",
    ]

    has_stock_domain = bool(re.search(r"\b(?:stock|stocks|equity|equities|share|shares|market)\b", text))
    has_signal_action = bool(
        re.search(r"\b(?:trend|trends|signal|signals|buy|sell|predict|prediction|forecast|indicator|technical|ml|machine learning)\b", text)
    )
    if has_stock_domain and has_signal_action:
        expansion_terms.append(
            "stock market technical indicator screener candlestick pattern price target buy sell signal trend analysis"
        )

    has_delivery_action = bool(re.search(r"\b(?:send|deliver|email|mail|sms|message|notify|notification)\b", text))
    if has_delivery_action:
        expansion_terms.append("send message notification email sms delivery communication endpoint")

    return " ".join([*expansion_terms, base]).strip()


def _merge_retrieval_retry_candidates(
    original: List[Dict[str, Any]],
    retry: List[Dict[str, Any]],
    *,
    max_candidates: int,
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def add_rows(rows: List[Dict[str, Any]], source: str) -> None:
        for row in rows:
            api_id = str(row.get("api_id") or "").strip()
            if not api_id or api_id in seen or len(merged) >= max_candidates:
                continue
            item = dict(row)
            item["retrieval_retry_source"] = source
            merged.append(item)
            seen.add(api_id)

    add_rows(retry, "zero_functional_match_retry")
    add_rows(original, "initial_retrieval")
    for idx, item in enumerate(merged, start=1):
        item["retrieved_rank"] = idx
    return merged


def _subtask_functional_match_count(
    retrieved: List[Dict[str, Any]],
    subtask_id: str,
    functional_match_map: Dict[tuple[str, str], Dict[str, Any]],
) -> int:
    count = 0
    for item in retrieved:
        api_id = str(item.get("api_id") or "").strip()
        if api_id and _functional_match_value(functional_match_map.get((subtask_id, api_id), {})) == 1:
            count += 1
    return count


def _retry_zero_functional_retrieval(
    *,
    user_goal: str,
    subtasks: List[Dict[str, Any]],
    out_dir: Path,
    retrieved_by_subtask: Dict[str, List[Dict[str, Any]]],
    functional_match_map: Dict[tuple[str, str], Dict[str, Any]],
    no_qos_services: Dict[str, Dict[str, Any]],
    with_qos_services: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not bool(getattr(CONFIG, "zero_functional_retrieval_retry_enabled", False)):
        return []

    retry_top_k = max(1, int(getattr(CONFIG, "zero_functional_retrieval_retry_top_k", CONFIG.rag_top_k) or CONFIG.rag_top_k))
    max_candidates = max(1, int(getattr(CONFIG, "rag_top_k", retry_top_k) or retry_top_k))
    retriever_agent: RagRetrieverAgent | None = None
    traces: List[Dict[str, Any]] = []
    new_api_ids: List[str] = []
    known_api_ids = set(no_qos_services) | set(with_qos_services)

    for sub in subtasks:
        sub_id = str(sub.get("id", "unknown"))
        original = retrieved_by_subtask.get(sub_id, [])
        if not original:
            continue
        if _subtask_functional_match_count(original, sub_id, functional_match_map) > 0:
            continue

        retry_query = _zero_functional_retrieval_retry_query(user_goal, sub)
        if retriever_agent is None:
            retriever_agent = RagRetrieverAgent(index_dir=str(CONFIG.shared_index_dir))
        retry_rows = retriever_agent.retrieve(retry_query, top_k=retry_top_k)
        merged = _merge_retrieval_retry_candidates(original, retry_rows, max_candidates=max_candidates)
        original_ids = [str(item.get("api_id") or "") for item in original]
        merged_ids = [str(item.get("api_id") or "") for item in merged]
        if merged_ids == original_ids[: len(merged_ids)]:
            continue

        retrieved_by_subtask[sub_id] = merged
        _write_json(out_dir / f"1_retriever_s{sub_id}.json", merged)
        trace = {
            "event_type": "zero_functional_retrieval_retry",
            "subtask_id": sub_id,
            "initial_candidate_count": len(original),
            "retry_candidate_count": len(retry_rows),
            "merged_candidate_count": len(merged),
            "initial_functional_match_count": 0,
            "retry_query": retry_query,
            "initial_api_ids": original_ids,
            "merged_api_ids": merged_ids,
        }
        _write_json(out_dir / "debug" / f"retrieval_zero_functional_retry_s{sub_id}.json", trace)
        log_warning_event({"stage": "retrieval", **trace})
        log_line(
            f"[{out_dir.name}] subtask={sub_id} zero functional matches; "
            f"reran retrieval and merged {len(merged)} candidates"
        )
        traces.append(trace)

        for api_id in merged_ids:
            if api_id and api_id not in known_api_ids and api_id not in new_api_ids:
                new_api_ids.append(api_id)

    if new_api_ids:
        no_qos_services.update(_fetch_catalog_subset(new_api_ids, with_qos=False))
        with_qos_services.update(_fetch_catalog_subset(new_api_ids, with_qos=True))

    return traces


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


def _functional_refinement_enrichment(
    retrieved: List[Dict[str, Any]],
    *,
    subtask_id: str,
    functional_match_map: Dict[tuple[str, str], Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in retrieved:
        api_id = str(row.get("api_id") or "").strip()
        if not api_id:
            continue
        match_entry = functional_match_map.get((subtask_id, api_id))
        if not isinstance(match_entry, dict) or not match_entry:
            continue
        reason = (
            match_entry.get("functional_refiner_reason")
            or match_entry.get("Comments")
            or match_entry.get("comment")
            or ""
        )
        enrichment = {
            "functional_match_label": _functional_match_value(match_entry),
        }
        if reason:
            enrichment["functional_refiner_reason"] = str(reason)
        out[api_id] = enrichment
    return out


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
            entry.get("Functional Match Label", entry.get("functional_match_label", entry.get("functional_match", entry.get("relevant", 0)))),
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
        "functional_first_ranking_violation",
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


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _qos_metrics_for_workflow(row: Dict[str, Any]) -> Tuple[float | None, float | None, float | None]:
    service = row.get("service")
    service = service if isinstance(service, dict) else {}
    qos = service.get("qos")
    qos = qos if isinstance(qos, dict) else {}

    rt_s = _float_or_none(row.get("rt_s"))
    if rt_s is None:
        rt_s = _float_or_none(row.get("QoS_RT_s"))
    if rt_s is None:
        rt_s = _float_or_none(qos.get("rt_s"))
    if rt_s is None:
        rt_ms = _float_or_none(row.get("rt_ms"))
        if rt_ms is None:
            rt_ms = _float_or_none(qos.get("rt_ms"))
        if rt_ms is not None:
            rt_s = rt_ms / 1000.0

    tp_kbps = _float_or_none(row.get("tp_kbps"))
    if tp_kbps is None:
        tp_kbps = _float_or_none(row.get("QoS_TP_kbps"))
    if tp_kbps is None:
        tp_kbps = _float_or_none(qos.get("tp_kbps"))
    if tp_kbps is None:
        tp_kbps = _float_or_none(row.get("tp_rps"))
    if tp_kbps is None:
        tp_kbps = _float_or_none(qos.get("tp_rps"))

    availability = _float_or_none(row.get("availability"))
    if availability is None:
        availability = _float_or_none(row.get("QoS Availability"))
    if availability is None:
        availability = _float_or_none(qos.get("availability"))
    return rt_s, tp_kbps, availability


def _clamp_01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalize_workflow_metric_relative_to_best(value: float | None, best_value: float | None, *, higher_better: bool) -> float:
    if value is None or best_value is None:
        return 0.0
    if higher_better:
        if best_value <= 0:
            return 0.0
        return _clamp_01(value / best_value)
    if value <= 0:
        return 0.0
    return _clamp_01(best_value / value)


def _planner_candidate_mode() -> str:
    mode = str(getattr(CONFIG, "planner_candidate_mode", "fixed_one") or "fixed_one").strip()
    if mode not in PLANNER_CANDIDATE_MODES:
        raise ValueError(
            f"Invalid planner_candidate_mode={mode!r}; expected one of {sorted(PLANNER_CANDIDATE_MODES)}."
        )
    return mode


def _hybrid_workflow_selector() -> str:
    selector = str(getattr(CONFIG, "hybrid_workflow_selector", "workflow_topsis") or "workflow_topsis").strip()
    if selector not in HYBRID_WORKFLOW_SELECTORS:
        raise ValueError(
            f"Invalid hybrid_workflow_selector={selector!r}; expected one of {sorted(HYBRID_WORKFLOW_SELECTORS)}."
        )
    return selector


def _planner_top_n_cap() -> int:
    try:
        return max(1, int(getattr(CONFIG, "planner_top_n_cap", 5) or 5))
    except Exception:
        return 5


def _planner_requested_top_n(subtask_id: str, planner_top_n: Dict[str, int] | None) -> int:
    candidate_cap = _planner_top_n_cap()
    mode_cutoff = max(1, int(getattr(CONFIG, "selector_top_n", 5) or 5))
    functional_cutoff = 0
    if planner_top_n:
        try:
            functional_cutoff = int(planner_top_n.get(subtask_id) or 0)
        except Exception:
            functional_cutoff = 0
    existing_cutoff = functional_cutoff if functional_cutoff > 0 else mode_cutoff
    return max(1, min(existing_cutoff, candidate_cap))


def _valid_ranked_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid = []
    for row in rows:
        if not isinstance(row, dict) or row.get("failure_flag") or row.get("invalid_output_row"):
            continue
        api_id = str(row.get("api_id") or "").strip()
        if api_id:
            valid.append(row)
    return valid


def _functional_refiner_metadata(
    row: Dict[str, Any],
    subtask_id: str,
    api_id: str,
    functional_match_map: Dict[tuple[str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    match_entry = functional_match_map.get((subtask_id, api_id), {})
    label_source = match_entry if match_entry else row
    reason = (
        match_entry.get("functional_refiner_reason")
        or match_entry.get("Comments")
        or match_entry.get("comment")
        or row.get("functional_refiner_reason")
        or row.get("Comments")
        or row.get("comment")
    )
    return {
        "functional_match_label": _functional_match_value(label_source),
        "functional_refiner_reason": reason,
    }


def _short_text(value: Any, limit: int = 240) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def _row_topsis_rank(row: Dict[str, Any]) -> int:
    try:
        return int(row.get("topsis_rank") or 10**9)
    except Exception:
        return 10**9


def _row_mode_rank(row: Dict[str, Any]) -> int:
    try:
        return int(row.get("mode_rank") or row.get("llm_reported_rank") or 10**9)
    except Exception:
        return 10**9


def _qos_values_for_row(row: Dict[str, Any]) -> Dict[str, float | None]:
    rt_s, tp_kbps, availability = _qos_metrics_for_workflow(row)
    return {
        "rt_s": rt_s,
        "tp_kbps": tp_kbps,
        "availability": availability,
    }


def _rank_source_for_mode(mode_name: str) -> str:
    return {
        "no_qos": "no_qos_ranking",
        "qos_pure_llm": "qos_pure_llm_ranking",
        "qos_topsis": "topsis_ranking",
        "qos_hybrid": "qos_hybrid_pareto_expanded_pool",
    }.get(mode_name, f"{mode_name}_ranking")


def _selection_provenance(row: Dict[str, Any], *, mode_name: str, selected_by: str, planner_override_attempted: bool) -> Dict[str, Any]:
    provenance = {
        "mode": mode_name,
        "subtask_id": str(row.get("subtask_id", "")),
        "api_id": str(row.get("api_id", "")),
        "selection_order": row.get("selection_order"),
        "selected_by": selected_by,
        "planner_override_attempted": planner_override_attempted,
    }
    if mode_name in TOPSIS_METADATA_MODES:
        provenance["topsis_rank"] = row.get("topsis_rank")
        provenance["topsis_score"] = row.get("topsis_score")
    if mode_name in FUNCTIONAL_REFINER_MODES:
        provenance["functional_match_label"] = row.get("functional_match_label")
        provenance["functional_refiner_reason"] = row.get("functional_refiner_reason")
    if row.get("planner_candidate_mode") == "top_n_ablation":
        for key in ["mode_rank", "rank_source", "qos_values", "short_rank_reason"]:
            if row.get(key) is not None:
                provenance[key] = row.get(key)
    for key in [
        "selected_by_view",
        "pareto_status",
        "hybrid_candidate_views",
        "hybrid_selector_objective",
        "hybrid_pool_strategy",
        "hybrid_pool_size_before_expansion",
        "hybrid_pool_size_after_expansion",
        "hybrid_pool_size_after_pareto",
        "pareto_filtered_count",
        "pareto_filter_fallback_used",
        "balanced_relative_qos_score",
        "hybrid_total_combinations_before_cap",
        "hybrid_total_combinations_after_cap",
        "hybrid_max_workflow_combinations",
        "hybrid_pool_trimmed",
        "hybrid_pool_sizes_before_cap",
        "hybrid_pool_sizes_after_cap",
        "projected_composition_score",
        "projected_normalized_qos_score",
        "workflow_total_response_time",
        "workflow_bottleneck_throughput",
        "workflow_average_availability",
        "hybrid_workflow_selector",
        "workflow_topsis_score",
        "workflow_topsis_weights",
        "workflow_topsis_rank",
        "workflow_selector_fallback_used",
    ]:
        if key in row:
            provenance[key] = row.get(key)
    return provenance


def _strip_reserved_mode_metadata(row: Dict[str, Any], mode_name: str) -> Dict[str, Any]:
    cleaned = dict(row)
    if mode_name not in FUNCTIONAL_REFINER_MODES:
        cleaned.pop("functional_match_label", None)
        cleaned.pop("functional_refiner_reason", None)
    if mode_name not in TOPSIS_METADATA_MODES:
        cleaned.pop("topsis_rank", None)
        cleaned.pop("topsis_score", None)
    return cleaned


def _prepare_selected_row(
    row: Dict[str, Any],
    *,
    subtask_id: str,
    selection_order: int,
    mode_name: str,
    selected_by: str,
    selector_reason: str,
    functional_match_map: Dict[tuple[str, str], Dict[str, Any]],
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    api_id = str(row.get("api_id") or "").strip()
    selected = dict(row)
    selected["api_id"] = api_id
    selected["selected_rank"] = selection_order
    selected["selection_order"] = selection_order
    selected.pop("score", None)
    selected["subtask_id"] = subtask_id
    selected["selector_reason"] = selector_reason
    selected["selected_by"] = selected_by
    selected["planner_override_attempted"] = False
    if mode_name in FUNCTIONAL_REFINER_MODES:
        selected.update(_functional_refiner_metadata(selected, subtask_id, api_id, functional_match_map))
    if extra:
        selected.update(extra)
    selected = _strip_reserved_mode_metadata(selected, mode_name)
    selected["selection_provenance"] = _selection_provenance(
        selected,
        mode_name=mode_name,
        selected_by=selected_by,
        planner_override_attempted=False,
    )
    return selected


def _select_top_ranked_workflow(
    *,
    mode_name: str,
    subtasks: List[Dict[str, Any]],
    ranked_full: Dict[str, List[Dict[str, Any]]],
    functional_match_map: Dict[tuple[str, str], Dict[str, Any]],
    planner_candidate_mode: str,
    planner_top_n: Dict[str, int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], List[str], Dict[str, str]]:
    selected_all: List[Dict[str, Any]] = []
    selection_trace: Dict[str, Dict[str, Any]] = {}
    missing_due_to_ranking_failure: List[str] = []
    subtask_failure_reasons: Dict[str, str] = {}

    for sub in subtasks:
        sub_id = str(sub.get("id", "unknown"))
        ranked_rows = ranked_full.get(sub_id, [])
        valid_rows = _valid_ranked_rows(ranked_rows)
        selected: List[Dict[str, Any]] = []
        if valid_rows:
            if planner_candidate_mode == "top_n_ablation":
                requested_top_n = _planner_requested_top_n(sub_id, planner_top_n)
                selected = [
                    _prepare_selected_row(
                        row,
                        subtask_id=sub_id,
                        selection_order=idx,
                        mode_name=mode_name,
                        selected_by="planner_candidate_ablation",
                        selector_reason=(
                            "Top-N planner ablation candidate from the mode ranking. "
                            "The planner may choose one candidate for this subtask."
                        ),
                        functional_match_map=functional_match_map,
                        extra={
                            "planner_candidate_mode": planner_candidate_mode,
                            "planner_top_n": min(requested_top_n, len(valid_rows)),
                            "planner_requested_top_n": requested_top_n,
                            "planner_top_n_cap": _planner_top_n_cap(),
                            "planner_top_n_source": _rank_source_for_mode(mode_name),
                            "rank_source": _rank_source_for_mode(mode_name),
                            "qos_values": _qos_values_for_row(row),
                            "short_rank_reason": _short_text(row.get("reason")),
                            "fallback_used": False,
                        },
                    )
                    for idx, row in enumerate(valid_rows[:requested_top_n], start=1)
                ]
            else:
                selected = [
                    _prepare_selected_row(
                        valid_rows[0],
                        subtask_id=sub_id,
                        selection_order=1,
                        mode_name=mode_name,
                        selected_by="ranking_mode",
                        selector_reason=(
                            "Fixed primary API selected by ranking mode. "
                            "The planner may compose this API but may not replace or re-rank it."
                        ),
                        functional_match_map=functional_match_map,
                        extra={
                            "planner_top_n": 1,
                            "planner_requested_top_n": 1,
                            "planner_top_n_source": "fixed_primary_api",
                            "fallback_used": False,
                        },
                    )
                ]

        invalid_ranked_reasons = [
            str(r.get("failure_reason") or r.get("ranking_anomaly_reason") or "ranking_failure")
            for r in ranked_rows
            if isinstance(r, dict) and (r.get("failure_flag") or r.get("invalid_output_row"))
        ]
        if not selected:
            missing_due_to_ranking_failure.append(sub_id)
            subtask_failure_reasons[sub_id] = invalid_ranked_reasons[0] if invalid_ranked_reasons else "no_valid_ranked_candidates"
        selected_all.extend(selected)
        selection_trace[sub_id] = {
            "planner_candidate_mode": planner_candidate_mode,
            "planner_top_n_cap": _planner_top_n_cap(),
            "planner_top_k": len(selected),
            "planner_requested_top_k": _planner_requested_top_n(sub_id, planner_top_n)
            if planner_candidate_mode == "top_n_ablation"
            else 1,
            "planner_top_k_source": _rank_source_for_mode(mode_name)
            if planner_candidate_mode == "top_n_ablation"
            else "fixed_primary_api",
            "retrieval_functional_match_k": None,
            "fallback_used": False,
            "available_ranked": len(ranked_rows),
            "selected_count": len(selected),
            "planner_candidates_per_subtask": len(selected),
            "selected_api_ids": [row.get("api_id") for row in selected],
            "invalid_ranked_rows": len(invalid_ranked_reasons),
        }

    return selected_all, selection_trace, missing_due_to_ranking_failure, subtask_failure_reasons


def _top_n_by_topsis(rows: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            _row_topsis_rank(row),
            -float(_float_or_none(row.get("topsis_score")) or 0.0),
            str(row.get("api_id") or ""),
        ),
    )[:n]


def _top_n_by_qos_metric(rows: List[Dict[str, Any]], n: int, metric_index: int, *, higher_better: bool) -> List[Dict[str, Any]]:
    metric_rows = []
    for row in rows:
        metric = _qos_metrics_for_workflow(row)[metric_index]
        if metric is not None:
            metric_rows.append((metric, row))
    metric_rows.sort(
        key=lambda item: (
            -float(item[0]) if higher_better else float(item[0]),
            str(item[1].get("api_id") or ""),
        )
    )
    return [row for _, row in metric_rows[:n]]


def _annotate_balanced_relative_qos(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    metrics = [_qos_metrics_for_workflow(row) for row in rows]
    rt_values = [rt_s for rt_s, _, _ in metrics if rt_s is not None and rt_s > 0]
    tp_values = [tp_kbps for _, tp_kbps, _ in metrics if tp_kbps is not None and tp_kbps > 0]
    availability_values = [availability for _, _, availability in metrics if availability is not None and availability > 0]
    best_rt = min(rt_values) if rt_values else None
    best_tp = max(tp_values) if tp_values else None
    best_availability = max(availability_values) if availability_values else None

    annotated: List[Dict[str, Any]] = []
    for row, (rt_s, tp_kbps, availability) in zip(rows, metrics):
        rt_score = _normalize_workflow_metric_relative_to_best(rt_s, best_rt, higher_better=False)
        tp_score = _normalize_workflow_metric_relative_to_best(tp_kbps, best_tp, higher_better=True)
        availability_score = _normalize_workflow_metric_relative_to_best(
            availability,
            best_availability,
            higher_better=True,
        )
        enriched = dict(row)
        enriched["balanced_relative_qos_score"] = round(float((rt_score + tp_score + availability_score) / 3.0), 6)
        annotated.append(enriched)
    return annotated


def _dedupe_rows_by_api_id(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for row in rows:
        api_id = str(row.get("api_id") or "").strip()
        if not api_id or api_id in seen:
            continue
        seen.add(api_id)
        deduped.append(row)
    return deduped


def _row_dominates_qos(challenger: Dict[str, Any], current: Dict[str, Any]) -> bool:
    challenger_rt, challenger_tp, challenger_availability = _qos_metrics_for_workflow(challenger)
    current_rt, current_tp, current_availability = _qos_metrics_for_workflow(current)
    if (
        challenger_rt is None
        or challenger_tp is None
        or challenger_availability is None
        or current_rt is None
        or current_tp is None
        or current_availability is None
    ):
        return False
    return (
        challenger_rt <= current_rt
        and challenger_tp >= current_tp
        and challenger_availability >= current_availability
        and (
            challenger_rt < current_rt
            or challenger_tp > current_tp
            or challenger_availability > current_availability
        )
    )


def _pareto_filter_hybrid_pool(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pareto_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        dominated = any(
            other_idx != idx and _row_dominates_qos(other, row)
            for other_idx, other in enumerate(rows)
        )
        if not dominated:
            pareto_rows.append(row)
    return pareto_rows


def _build_pareto_expanded_hybrid_pool(functional_rows: List[Dict[str, Any]], candidate_cap: int) -> Dict[str, Any]:
    annotated_rows = _annotate_balanced_relative_qos(functional_rows)
    expanded_rows: List[Dict[str, Any]] = []
    views_by_api: Dict[str, List[str]] = {}

    def add_view(rows: List[Dict[str, Any]], view: str) -> None:
        for row in rows:
            api_id = str(row.get("api_id") or "").strip()
            if api_id:
                views_by_api.setdefault(api_id, [])
                if view not in views_by_api[api_id]:
                    views_by_api[api_id].append(view)
            enriched = dict(row)
            enriched.setdefault("selected_by_view", view)
            expanded_rows.append(enriched)

    add_view(_top_n_by_topsis(annotated_rows, candidate_cap), "topsis")
    add_view(_top_n_by_qos_metric(annotated_rows, candidate_cap, 0, higher_better=False), "response_time")
    add_view(_top_n_by_qos_metric(annotated_rows, candidate_cap, 1, higher_better=True), "throughput")
    add_view(_top_n_by_qos_metric(annotated_rows, candidate_cap, 2, higher_better=True), "availability")
    add_view(
        sorted(
            annotated_rows,
            key=lambda row: (
                -float(row.get("balanced_relative_qos_score") or 0.0),
                _row_topsis_rank(row),
                str(row.get("api_id") or ""),
            ),
        )[:candidate_cap],
        "balanced_qos",
    )

    expanded_deduped = _dedupe_rows_by_api_id(expanded_rows)
    pareto_pool = _pareto_filter_hybrid_pool(expanded_deduped)
    pareto_fallback_used = bool(expanded_deduped and not pareto_pool)
    final_pool = expanded_deduped if pareto_fallback_used else pareto_pool
    pareto_api_ids = {str(row.get("api_id") or "").strip() for row in pareto_pool}
    final_pool = [
        {
            **row,
            "selected_by_view": "pareto" if str(row.get("api_id") or "").strip() in pareto_api_ids else row.get("selected_by_view"),
            "hybrid_candidate_views": views_by_api.get(str(row.get("api_id") or "").strip(), []),
            "pareto_status": "pareto" if str(row.get("api_id") or "").strip() in pareto_api_ids else "pareto_fallback",
        }
        for row in final_pool
    ]
    return {
        "pool": final_pool,
        "hybrid_pool_strategy": "pareto_expanded_multi_qos_view",
        "hybrid_pool_size_before_expansion": len(functional_rows),
        "hybrid_pool_size_after_expansion": len(expanded_deduped),
        "hybrid_pool_size_after_pareto": len(final_pool),
        "pareto_filtered_count": len(expanded_deduped) - len(pareto_pool),
        "pareto_filter_fallback_used": pareto_fallback_used,
    }


def _hybrid_max_workflow_combinations() -> int:
    try:
        return max(1, int(getattr(CONFIG, "hybrid_max_workflow_combinations", 5000) or 5000))
    except Exception:
        return 5000


def _candidate_pool_combination_count(pool_sizes: List[int]) -> int:
    total = 1
    for size in pool_sizes:
        total *= max(0, int(size))
    return total


def _hybrid_pareto_priority(row: Dict[str, Any]) -> int:
    status = str(row.get("pareto_status") or "").strip().lower()
    if status == "pareto":
        return 0
    if status == "pareto_fallback":
        return 1
    return 2


def _hybrid_candidate_priority_key(row: Dict[str, Any]) -> tuple:
    projected_score = _float_or_none(row.get("projected_composition_score"))
    balanced_score = _float_or_none(row.get("balanced_relative_qos_score"))
    priority_score = projected_score if projected_score is not None else balanced_score
    rt_s, tp_kbps, availability = _qos_metrics_for_workflow(row)
    topsis_score = _float_or_none(row.get("topsis_score"))
    return (
        -float(priority_score) if priority_score is not None else float("inf"),
        _hybrid_pareto_priority(row),
        _row_topsis_rank(row),
        -float(topsis_score) if topsis_score is not None else float("inf"),
        float(rt_s) if rt_s is not None else float("inf"),
        -float(availability) if availability is not None else float("inf"),
        -float(tp_kbps) if tp_kbps is not None else float("inf"),
        str(row.get("api_id") or ""),
    )


def _cap_hybrid_candidate_pools(
    pools: List[List[Dict[str, Any]]],
    subtask_ids: List[str],
    max_combinations: int,
) -> Tuple[List[List[Dict[str, Any]]], Dict[str, Any]]:
    pool_sizes_before = [len(pool) for pool in pools]
    total_before = _candidate_pool_combination_count(pool_sizes_before)
    cap = max(1, int(max_combinations))
    metadata = {
        "hybrid_total_combinations_before_cap": total_before,
        "hybrid_total_combinations_after_cap": total_before,
        "hybrid_max_workflow_combinations": cap,
        "hybrid_pool_trimmed": False,
        "hybrid_pool_sizes_before_cap": {sub_id: size for sub_id, size in zip(subtask_ids, pool_sizes_before)},
        "hybrid_pool_sizes_after_cap": {sub_id: size for sub_id, size in zip(subtask_ids, pool_sizes_before)},
    }
    if total_before <= cap:
        return pools, metadata

    sorted_pools = [sorted(pool, key=_hybrid_candidate_priority_key) for pool in pools]
    pool_sizes_after = list(pool_sizes_before)
    total_after = total_before
    while total_after > cap and any(size > 1 for size in pool_sizes_after):
        largest_idx = min(
            (idx for idx, size in enumerate(pool_sizes_after) if size > 1),
            key=lambda idx: (-pool_sizes_after[idx], str(subtask_ids[idx]), idx),
        )
        pool_sizes_after[largest_idx] -= 1
        total_after = _candidate_pool_combination_count(pool_sizes_after)

    capped_pools = [pool[:size] for pool, size in zip(sorted_pools, pool_sizes_after)]
    metadata.update(
        {
            "hybrid_total_combinations_after_cap": total_after,
            "hybrid_pool_trimmed": True,
            "hybrid_pool_sizes_after_cap": {sub_id: size for sub_id, size in zip(subtask_ids, pool_sizes_after)},
        }
    )
    return capped_pools, metadata


def _finite_workflow_metric(value: Any) -> float | None:
    number = _float_or_none(value)
    if number is None or not math.isfinite(number):
        return None
    return number


def _round_float_or_none(value: Any, digits: int = 6) -> float | None:
    number = _finite_workflow_metric(value)
    if number is None:
        return None
    return round(float(number), digits)


def _annotate_relative_to_best_workflow_records(combination_records: List[Dict[str, Any]]) -> None:
    response_time_values = [
        record["total_response_time"]
        for record in combination_records
        if record["total_response_time"] is not None
    ]
    throughput_values = [
        record["bottleneck_throughput"]
        for record in combination_records
        if record["bottleneck_throughput"] is not None
    ]
    availability_values = [
        record["average_availability"]
        for record in combination_records
        if record["average_availability"] is not None
    ]
    best_response_time = min(response_time_values) if response_time_values else None
    best_throughput = max(throughput_values) if throughput_values else None
    best_availability = max(availability_values) if availability_values else None

    for record in combination_records:
        response_time_score = _normalize_workflow_metric_relative_to_best(
            record["total_response_time"],
            best_response_time,
            higher_better=False,
        )
        throughput_score = _normalize_workflow_metric_relative_to_best(
            record["bottleneck_throughput"],
            best_throughput,
            higher_better=True,
        )
        availability_score = _normalize_workflow_metric_relative_to_best(
            record["average_availability"],
            best_availability,
            higher_better=True,
        )
        normalized_qos_score = (response_time_score + throughput_score + availability_score) / 3.0
        projected_composition_score = 1.0 * (0.7 * 1.0 + 0.3 * normalized_qos_score)
        record["workflow_qos_score"] = normalized_qos_score
        record["projected_normalized_qos_score"] = normalized_qos_score
        record["projected_composition_score"] = projected_composition_score
        record["normalized_response_time"] = response_time_score
        record["normalized_throughput"] = throughput_score
        record["normalized_availability"] = availability_score


def _relative_to_best_workflow_sort_key(record: Dict[str, Any]) -> tuple:
    return (
        -float(record["projected_composition_score"]),
        float(record["total_response_time"]) if record["total_response_time"] is not None else float("inf"),
        -float(record["average_availability"]) if record["average_availability"] is not None else float("inf"),
        -float(record["bottleneck_throughput"]) if record["bottleneck_throughput"] is not None else float("inf"),
        -float(record["average_topsis_score"]) if record["average_topsis_score"] is not None else float("inf"),
        record["api_ids"],
    )


def _workflow_topsis_sort_key(record: Dict[str, Any]) -> tuple:
    score = _finite_workflow_metric(record.get("workflow_topsis_score"))
    return (
        -float(score) if score is not None else float("inf"),
        float(record["total_response_time"]) if record["total_response_time"] is not None else float("inf"),
        -float(record["average_availability"]) if record["average_availability"] is not None else float("inf"),
        -float(record["bottleneck_throughput"]) if record["bottleneck_throughput"] is not None else float("inf"),
        -float(record["average_topsis_score"]) if record["average_topsis_score"] is not None else float("inf"),
        record["api_ids"],
    )


def _annotate_workflow_topsis_records(combination_records: List[Dict[str, Any]]) -> bool:
    if not combination_records:
        return False

    weights = [
        WORKFLOW_TOPSIS_WEIGHTS["response_time"],
        WORKFLOW_TOPSIS_WEIGHTS["throughput"],
        WORKFLOW_TOPSIS_WEIGHTS["availability"],
    ]
    for record in combination_records:
        record["workflow_topsis_weights"] = dict(WORKFLOW_TOPSIS_WEIGHTS)

    if len(combination_records) == 1:
        combination_records[0]["workflow_topsis_score"] = 1.0
        combination_records[0]["workflow_topsis_rank"] = 1
        return True

    matrix: List[List[float]] = []
    for record in combination_records:
        row = [
            _finite_workflow_metric(record.get("total_response_time")),
            _finite_workflow_metric(record.get("bottleneck_throughput")),
            _finite_workflow_metric(record.get("average_availability")),
        ]
        if any(value is None for value in row):
            return False
        matrix.append([float(value) for value in row if value is not None])

    column_norms = [
        math.sqrt(sum(row[col_idx] ** 2 for row in matrix))
        for col_idx in range(3)
    ]
    if any(norm <= 0.0 or not math.isfinite(norm) for norm in column_norms):
        return False

    weighted = [
        [
            (row[col_idx] / column_norms[col_idx]) * weights[col_idx]
            for col_idx in range(3)
        ]
        for row in matrix
    ]
    if any(not math.isfinite(value) for row in weighted for value in row):
        return False

    ideal_best = [
        min(row[0] for row in weighted),
        max(row[1] for row in weighted),
        max(row[2] for row in weighted),
    ]
    ideal_worst = [
        max(row[0] for row in weighted),
        min(row[1] for row in weighted),
        min(row[2] for row in weighted),
    ]

    for record, row in zip(combination_records, weighted):
        distance_to_best = math.sqrt(sum((row[idx] - ideal_best[idx]) ** 2 for idx in range(3)))
        distance_to_worst = math.sqrt(sum((row[idx] - ideal_worst[idx]) ** 2 for idx in range(3)))
        denominator = distance_to_best + distance_to_worst
        if denominator <= 0.0 or not math.isfinite(denominator):
            return False
        score = distance_to_worst / denominator
        if not math.isfinite(score):
            return False
        record["workflow_topsis_score"] = score

    for rank, record in enumerate(sorted(combination_records, key=_workflow_topsis_sort_key), start=1):
        record["workflow_topsis_rank"] = rank
    return True


def _select_workflow_hybrid(
    *,
    subtasks: List[Dict[str, Any]],
    ranked_full: Dict[str, List[Dict[str, Any]]],
    functional_match_map: Dict[tuple[str, str], Dict[str, Any]],
    run_label: str,
    planner_candidate_mode: str,
    planner_top_n: Dict[str, int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], List[str], Dict[str, str]]:
    candidate_cap = max(1, int(getattr(CONFIG, "selector_top_n", 5) or 5))
    workflow_selector = _hybrid_workflow_selector()
    pools: List[List[Dict[str, Any]]] = []
    subtask_ids: List[str] = []
    pool_metadata_by_subtask: Dict[str, Dict[str, Any]] = {}
    selection_trace: Dict[str, Dict[str, Any]] = {}
    missing_due_to_ranking_failure: List[str] = []
    subtask_failure_reasons: Dict[str, str] = {}

    for sub in subtasks:
        sub_id = str(sub.get("id", "unknown"))
        ranked_rows = ranked_full.get(sub_id, [])
        valid_rows = _valid_ranked_rows(ranked_rows)
        functional_rows = [
            row
            for row in valid_rows
            if _functional_refiner_metadata(row, sub_id, str(row.get("api_id") or ""), functional_match_map)["functional_match_label"] == 1
        ]
        fallback_used = not functional_rows
        if functional_rows:
            pool_result = _build_pareto_expanded_hybrid_pool(functional_rows, candidate_cap)
            pool = pool_result["pool"]
            source = "functional_match_pareto_expanded_multi_qos_view"
            pool_metadata = {
                key: pool_result[key]
                for key in [
                    "hybrid_pool_strategy",
                    "hybrid_pool_size_before_expansion",
                    "hybrid_pool_size_after_expansion",
                    "hybrid_pool_size_after_pareto",
                    "pareto_filtered_count",
                    "pareto_filter_fallback_used",
                ]
            }
        else:
            pool = sorted(valid_rows, key=lambda row: (_row_mode_rank(row), str(row.get("api_id") or "")))[:candidate_cap]
            source = "hybrid_no_functional_match_fallback"
            pool_metadata = {
                "hybrid_pool_strategy": "hybrid_no_functional_match_fallback",
                "hybrid_pool_size_before_expansion": 0,
                "hybrid_pool_size_after_expansion": 0,
                "hybrid_pool_size_after_pareto": 0,
                "pareto_filtered_count": 0,
                "pareto_filter_fallback_used": False,
            }
            log_line(
                f"[{run_label}] subtask={sub_id} mode=qos_hybrid hybrid_no_functional_match_fallback "
                f"using selector_top_n={candidate_cap}"
            )

        invalid_ranked_reasons = [
            str(r.get("failure_reason") or r.get("ranking_anomaly_reason") or "ranking_failure")
            for r in ranked_rows
            if isinstance(r, dict) and (r.get("failure_flag") or r.get("invalid_output_row"))
        ]
        if not pool:
            missing_due_to_ranking_failure.append(sub_id)
            subtask_failure_reasons[sub_id] = invalid_ranked_reasons[0] if invalid_ranked_reasons else "no_valid_ranked_candidates"

        pools.append(pool)
        subtask_ids.append(sub_id)
        pool_metadata_by_subtask[sub_id] = pool_metadata
        selection_trace[sub_id] = {
            "planner_candidate_mode": planner_candidate_mode,
            "planner_top_n_cap": _planner_top_n_cap(),
            "planner_top_k": 0,
            "planner_requested_top_k": _planner_requested_top_n(sub_id, planner_top_n)
            if planner_candidate_mode == "top_n_ablation"
            else 1,
            "planner_top_k_source": "workflow_hybrid_selector",
            "hybrid_workflow_selector": workflow_selector,
            "candidate_cap_per_subtask": candidate_cap,
            "workflow_candidate_pool_source": source,
            "hybrid_no_functional_match_fallback": fallback_used,
            **pool_metadata,
            "available_ranked": len(ranked_rows),
            "functional_candidate_count": len(functional_rows),
            "workflow_candidate_pool_count": len(pool),
            "selected_count": 0,
            "invalid_ranked_rows": len(invalid_ranked_reasons),
        }

    if any(not pool for pool in pools):
        return [], selection_trace, missing_due_to_ranking_failure, subtask_failure_reasons

    if planner_candidate_mode == "top_n_ablation":
        selected_all: List[Dict[str, Any]] = []
        for sub_id, pool in zip(subtask_ids, pools):
            requested_top_n = _planner_requested_top_n(sub_id, planner_top_n)
            ranked_pool = sorted(
                pool,
                key=lambda row: (
                    -float(row.get("balanced_relative_qos_score") or 0.0),
                    _row_topsis_rank(row),
                    _row_mode_rank(row),
                    str(row.get("api_id") or ""),
                ),
            )[:requested_top_n]
            for idx, row in enumerate(ranked_pool, start=1):
                selected = _prepare_selected_row(
                    row,
                    subtask_id=sub_id,
                    selection_order=idx,
                    mode_name="qos_hybrid",
                    selected_by="planner_candidate_ablation",
                    selector_reason=(
                        "Top-N planner ablation candidate from the Pareto-expanded hybrid pool. "
                        "The planner may choose one candidate for this subtask."
                    ),
                    functional_match_map=functional_match_map,
                    extra={
                        "planner_candidate_mode": planner_candidate_mode,
                        "planner_top_n": len(ranked_pool),
                        "planner_requested_top_n": requested_top_n,
                        "planner_top_n_cap": _planner_top_n_cap(),
                        "planner_top_n_source": "qos_hybrid_pareto_expanded_pool",
                        "rank_source": "qos_hybrid_pareto_expanded_pool",
                        **pool_metadata_by_subtask.get(sub_id, {}),
                        "selected_by_view": row.get("selected_by_view"),
                        "pareto_status": row.get("pareto_status"),
                        "hybrid_candidate_views": row.get("hybrid_candidate_views"),
                        "balanced_relative_qos_score": row.get("balanced_relative_qos_score"),
                        "qos_values": _qos_values_for_row(row),
                        "short_rank_reason": _short_text(row.get("reason")),
                        "fallback_used": bool(selection_trace.get(sub_id, {}).get("hybrid_no_functional_match_fallback")),
                    },
                )
                selected_all.append(selected)

            selection_trace[sub_id]["planner_top_k"] = len(ranked_pool)
            selection_trace[sub_id]["selected_count"] = len(ranked_pool)
            selection_trace[sub_id]["planner_candidates_per_subtask"] = len(ranked_pool)
            selection_trace[sub_id]["planner_top_k_source"] = "qos_hybrid_pareto_expanded_pool"
            selection_trace[sub_id]["selected_api_ids"] = [str(row.get("api_id") or "") for row in ranked_pool]

        return selected_all, selection_trace, missing_due_to_ranking_failure, subtask_failure_reasons

    pools, combination_cap_metadata = _cap_hybrid_candidate_pools(
        pools,
        subtask_ids,
        _hybrid_max_workflow_combinations(),
    )
    for sub_id in subtask_ids:
        selection_trace[sub_id].update(combination_cap_metadata)
        selection_trace[sub_id]["workflow_candidate_pool_count"] = combination_cap_metadata["hybrid_pool_sizes_after_cap"].get(
            sub_id,
            selection_trace[sub_id].get("workflow_candidate_pool_count", 0),
        )
    if combination_cap_metadata["hybrid_pool_trimmed"]:
        log_line(
            f"[{run_label}] mode=qos_hybrid hybrid_pool_trimmed "
            f"before={combination_cap_metadata['hybrid_total_combinations_before_cap']} "
            f"after={combination_cap_metadata['hybrid_total_combinations_after_cap']} "
            f"cap={combination_cap_metadata['hybrid_max_workflow_combinations']} "
            f"sizes_before={combination_cap_metadata['hybrid_pool_sizes_before_cap']} "
            f"sizes_after={combination_cap_metadata['hybrid_pool_sizes_after_cap']}"
        )

    combination_records: List[Dict[str, Any]] = []
    for combo in product(*pools):
        rt_values: List[float] = []
        tp_values: List[float] = []
        av_values: List[float] = []
        for row in combo:
            rt_s, tp_kbps, availability = _qos_metrics_for_workflow(row)
            if rt_s is not None:
                rt_values.append(rt_s)
            if tp_kbps is not None:
                tp_values.append(tp_kbps)
            if availability is not None:
                av_values.append(availability)

        total_response_time = sum(rt_values) if len(rt_values) == len(combo) else None
        bottleneck_throughput = min(tp_values) if len(tp_values) == len(combo) else None
        average_availability = sum(av_values) / len(av_values) if len(av_values) == len(combo) else None
        topsis_scores = [_float_or_none(row.get("topsis_score")) for row in combo]
        present_topsis_scores = [score for score in topsis_scores if score is not None]
        api_ids = tuple(str(row.get("api_id") or "") for row in combo)
        combination_records.append(
            {
                "combo": combo,
                "api_ids": api_ids,
                "total_response_time": total_response_time,
                "bottleneck_throughput": bottleneck_throughput,
                "average_availability": average_availability,
                "average_topsis_score": sum(present_topsis_scores) / len(present_topsis_scores) if present_topsis_scores else None,
            }
        )

    # Keep the previous selector's relative-to-best score available for provenance
    # and as the configured fallback; final evaluator scoring is unchanged.
    _annotate_relative_to_best_workflow_records(combination_records)
    for record in combination_records:
        record.setdefault("workflow_topsis_weights", dict(WORKFLOW_TOPSIS_WEIGHTS))
        record.setdefault("workflow_topsis_score", None)
        record.setdefault("workflow_topsis_rank", None)

    workflow_selector_fallback_used = False
    effective_selector = workflow_selector
    if workflow_selector == "workflow_topsis":
        try:
            topsis_available = _annotate_workflow_topsis_records(combination_records)
        except Exception as exc:
            topsis_available = False
            log_warning_event(
                {
                    "event_type": "qos_hybrid_workflow_topsis_failed",
                    "run_label": run_label,
                    "mode": "qos_hybrid",
                    "fallback": "relative_to_best",
                    "error": str(exc),
                }
            )
        if topsis_available:
            winner = sorted(combination_records, key=_workflow_topsis_sort_key)[0]
        else:
            workflow_selector_fallback_used = True
            effective_selector = "relative_to_best"
            winner = sorted(combination_records, key=_relative_to_best_workflow_sort_key)[0]
    else:
        winner = sorted(combination_records, key=_relative_to_best_workflow_sort_key)[0]

    selector_objective = "workflow_topsis" if effective_selector == "workflow_topsis" else "final_score_aligned_relative_to_best"
    selector_reason = (
        "Workflow-level hybrid selector chose this fixed API from functionally matched candidates "
        "using workflow-level TOPSIS over total response time, bottleneck throughput, and average availability."
        if effective_selector == "workflow_topsis"
        else (
            "Workflow-level hybrid selector chose this fixed API from functionally matched candidates "
            "using projected final-score-aligned relative QoS."
        )
    )

    selected_all: List[Dict[str, Any]] = []
    for sub_id, row in zip(subtask_ids, winner["combo"]):
        selected = _prepare_selected_row(
            row,
            subtask_id=sub_id,
            selection_order=1,
            mode_name="qos_hybrid",
            selected_by="workflow_hybrid_selector",
            selector_reason=selector_reason,
            functional_match_map=functional_match_map,
            extra={
                "planner_top_n": 1,
                "planner_requested_top_n": 1,
                "planner_top_n_source": "workflow_hybrid_selector",
                "hybrid_selector_objective": selector_objective,
                "hybrid_workflow_selector": workflow_selector,
                **pool_metadata_by_subtask.get(sub_id, {}),
                **combination_cap_metadata,
                "balanced_relative_qos_score": row.get("balanced_relative_qos_score"),
                "workflow_qos_score": _round_float_or_none(winner.get("workflow_qos_score")),
                "projected_composition_score": _round_float_or_none(winner.get("projected_composition_score")),
                "projected_normalized_qos_score": _round_float_or_none(winner.get("projected_normalized_qos_score")),
                "workflow_total_response_time": _round_float_or_none(winner.get("total_response_time")),
                "workflow_bottleneck_throughput": winner["bottleneck_throughput"],
                "workflow_average_availability": winner["average_availability"],
                "workflow_topsis_score": _round_float_or_none(winner.get("workflow_topsis_score")),
                "workflow_topsis_weights": winner.get("workflow_topsis_weights") or dict(WORKFLOW_TOPSIS_WEIGHTS),
                "workflow_topsis_rank": winner.get("workflow_topsis_rank"),
                "workflow_selector_fallback_used": workflow_selector_fallback_used,
            },
        )
        selected_all.append(selected)
        selection_trace[sub_id]["planner_top_k"] = 1
        selection_trace[sub_id]["selected_count"] = 1
        selection_trace[sub_id]["planner_candidates_per_subtask"] = 1
        selection_trace[sub_id]["selected_api_id"] = selected["api_id"]
        selection_trace[sub_id]["selected_api_ids"] = [selected["api_id"]]
        selection_trace[sub_id]["selected_by"] = "workflow_hybrid_selector"
        selection_trace[sub_id]["hybrid_selector_objective"] = selector_objective
        selection_trace[sub_id]["hybrid_workflow_selector"] = workflow_selector
        if row.get("balanced_relative_qos_score") is not None:
            selection_trace[sub_id]["balanced_relative_qos_score"] = row.get("balanced_relative_qos_score")
        selection_trace[sub_id]["projected_composition_score"] = _round_float_or_none(winner.get("projected_composition_score"))
        selection_trace[sub_id]["projected_normalized_qos_score"] = _round_float_or_none(winner.get("projected_normalized_qos_score"))
        selection_trace[sub_id]["workflow_total_response_time"] = _round_float_or_none(winner.get("total_response_time"))
        selection_trace[sub_id]["workflow_bottleneck_throughput"] = winner["bottleneck_throughput"]
        selection_trace[sub_id]["workflow_average_availability"] = winner["average_availability"]
        selection_trace[sub_id]["workflow_qos_score"] = _round_float_or_none(winner.get("workflow_qos_score"))
        selection_trace[sub_id]["workflow_topsis_score"] = _round_float_or_none(winner.get("workflow_topsis_score"))
        selection_trace[sub_id]["workflow_topsis_weights"] = winner.get("workflow_topsis_weights") or dict(WORKFLOW_TOPSIS_WEIGHTS)
        selection_trace[sub_id]["workflow_topsis_rank"] = winner.get("workflow_topsis_rank")
        selection_trace[sub_id]["workflow_selector_fallback_used"] = workflow_selector_fallback_used

    return selected_all, selection_trace, missing_due_to_ranking_failure, subtask_failure_reasons


def _planner_api_ids(plan: Dict[str, Any]) -> List[str]:
    api_ids: List[str] = []
    containers = [
        ((plan.get("primary_plan") or {}).get("steps") if isinstance(plan.get("primary_plan"), dict) else []),
        ((plan.get("execution_workflow") or {}).get("steps") if isinstance(plan.get("execution_workflow"), dict) else []),
        plan.get("selected_api_ids") if isinstance(plan.get("selected_api_ids"), list) else [],
    ]
    for container in containers:
        for item in container if isinstance(container, list) else []:
            api_id = item.get("api_id") if isinstance(item, dict) else item
            if isinstance(api_id, str) and api_id.strip() and api_id.strip() not in api_ids:
                api_ids.append(api_id.strip())
    return api_ids


def _unapproved_planner_api_ids(plan: Dict[str, Any], allowed_api_ids: set[str]) -> List[str]:
    return [api_id for api_id in _planner_api_ids(plan) if api_id not in allowed_api_ids]


def _planner_steps(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps = (plan.get("primary_plan") or {}).get("steps") if isinstance(plan.get("primary_plan"), dict) else []
    if isinstance(steps, list) and steps:
        return [step for step in steps if isinstance(step, dict)]
    execution_steps = (plan.get("execution_workflow") or {}).get("steps") if isinstance(plan.get("execution_workflow"), dict) else []
    return [step for step in execution_steps if isinstance(step, dict)] if isinstance(execution_steps, list) else []


def _planner_api_ids_by_subtask(plan: Dict[str, Any]) -> Dict[str, List[str]]:
    by_subtask: Dict[str, List[str]] = {}
    step_groups = [
        (plan.get("primary_plan") or {}).get("steps") if isinstance(plan.get("primary_plan"), dict) else [],
        (plan.get("execution_workflow") or {}).get("steps") if isinstance(plan.get("execution_workflow"), dict) else [],
    ]
    for steps in step_groups:
        for step in steps if isinstance(steps, list) else []:
            if not isinstance(step, dict):
                continue
            subtask_id = str(step.get("subtask_id") or "").strip()
            api_id = str(step.get("api_id") or "").strip()
            if not subtask_id or not api_id:
                continue
            by_subtask.setdefault(subtask_id, [])
            if api_id not in by_subtask[subtask_id]:
                by_subtask[subtask_id].append(api_id)
    return by_subtask


def _planner_one_api_per_subtask_issues(
    plan: Dict[str, Any],
    subtasks: List[Dict[str, Any]],
    selected_all: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    by_subtask = _planner_api_ids_by_subtask(plan)
    allowed_by_subtask: Dict[str, set[str]] = {}
    candidate_by_pair: Dict[tuple[str, str], Dict[str, Any]] = {}
    for row in selected_all or []:
        subtask_id = str(row.get("subtask_id") or "").strip()
        api_id = str(row.get("api_id") or "").strip()
        if subtask_id and api_id:
            allowed_by_subtask.setdefault(subtask_id, set()).add(api_id)
            candidate_by_pair[(subtask_id, api_id)] = row
    primary_steps = (
        (plan.get("primary_plan") or {}).get("steps")
        if isinstance(plan.get("primary_plan"), dict)
        else []
    )
    execution_steps = (
        (plan.get("execution_workflow") or {}).get("steps")
        if isinstance(plan.get("execution_workflow"), dict)
        else []
    )
    primary_by_pair = {
        (str(step.get("subtask_id") or "").strip(), str(step.get("api_id") or "").strip()): step
        for step in primary_steps
        if isinstance(step, dict)
    } if isinstance(primary_steps, list) else {}
    execution_by_pair = {
        (str(step.get("subtask_id") or "").strip(), str(step.get("api_id") or "").strip()): step
        for step in execution_steps
        if isinstance(step, dict)
    } if isinstance(execution_steps, list) else {}
    issues: List[Dict[str, Any]] = []
    for sub in subtasks:
        subtask_id = str(sub.get("id", "unknown"))
        api_ids = by_subtask.get(subtask_id, [])
        if len(api_ids) != 1:
            issues.append({"subtask_id": subtask_id, "selected_api_ids": api_ids, "selected_count": len(api_ids)})
            continue
        allowed_api_ids = allowed_by_subtask.get(subtask_id)
        if allowed_api_ids is not None and api_ids[0] not in allowed_api_ids:
            issues.append(
                {
                    "subtask_id": subtask_id,
                    "selected_api_ids": api_ids,
                    "selected_count": len(api_ids),
                    "reason": "api_not_provided_for_subtask",
                    "allowed_api_ids_for_subtask": sorted(allowed_api_ids),
                }
            )
            continue
        candidate = candidate_by_pair.get((subtask_id, api_ids[0]), {})
        try:
            selection_order = int(candidate.get("selection_order") or candidate.get("selected_rank") or 0)
        except Exception:
            selection_order = 0
        if selection_order > 1:
            primary_reason = primary_by_pair.get((subtask_id, api_ids[0]), {}).get("planner_override_reason")
            execution_reason = execution_by_pair.get((subtask_id, api_ids[0]), {}).get("planner_override_reason")
            if not str(primary_reason or "").strip() or not str(execution_reason or "").strip():
                issues.append(
                    {
                        "subtask_id": subtask_id,
                        "selected_api_ids": api_ids,
                        "selected_count": len(api_ids),
                        "reason": "missing_planner_override_reason_for_rank_override",
                        "selection_order": selection_order,
                        "primary_plan_has_planner_override_reason": bool(str(primary_reason or "").strip()),
                        "execution_workflow_has_planner_override_reason": bool(str(execution_reason or "").strip()),
                    }
                )
    return issues


def _planner_one_api_retry_goal(user_goal: str, issues: List[Dict[str, Any]]) -> str:
    return (
        f"{user_goal}\n\n"
        "Planner validation retry: choose exactly one API per subtask from the provided candidates. "
        "Do not omit subtasks and do not use multiple APIs for the same subtask. "
        "If you choose a candidate whose selection_order is greater than 1, include planner_override_reason "
        "on both the primary_plan step and execution_workflow step using the candidate metadata.\n"
        f"Issues to fix: {json.dumps(issues, ensure_ascii=False)}"
    )


def _planner_fixed_one_retry_goal(user_goal: str, issues: List[Dict[str, Any]]) -> str:
    return (
        f"{user_goal}\n\n"
        "Planner validation retry: fixed_one mode received exactly one selected API per subtask. "
        "Use the selected API assigned to each subtask exactly as provided. "
        "Do not replace, re-rank, swap, or substitute APIs across subtasks.\n"
        f"Issues to fix: {json.dumps(issues, ensure_ascii=False)}"
    )


def _planner_retry_goal(
    user_goal: str,
    unapproved_api_ids: List[str],
    allowed_api_ids: set[str],
    *,
    planner_candidate_mode: str = "fixed_one",
) -> str:
    if planner_candidate_mode == "top_n_ablation":
        return (
            f"{user_goal}\n\n"
            "Planner validation retry: your previous workflow used API IDs that were not in the provided "
            f"candidate alternatives: {sorted(unapproved_api_ids)}. Use only these provided API IDs: "
            f"{sorted(allowed_api_ids)}. Choose exactly one API per subtask from the provided candidates."
        )
    return (
        f"{user_goal}\n\n"
        "Planner validation retry: your previous workflow used API IDs that were not fixed by the selection stage: "
        f"{sorted(unapproved_api_ids)}. Use only these selected API IDs: {sorted(allowed_api_ids)}. "
        "Do not replace, re-rank, or substitute selected APIs."
    )


def _annotate_planner_provenance(
    planner: Dict[str, Any],
    selected_all: List[Dict[str, Any]],
    *,
    mode_name: str,
    planner_override_attempted: bool,
) -> Dict[str, Any]:
    provenance_by_pair: Dict[tuple[str, str], Dict[str, Any]] = {}
    provenance_by_api: Dict[str, Dict[str, Any]] = {}
    for row in selected_all:
        row = dict(row)
        row["planner_override_attempted"] = planner_override_attempted
        provenance = _selection_provenance(
            row,
            mode_name=mode_name,
            selected_by=str(row.get("selected_by") or "ranking_mode"),
            planner_override_attempted=planner_override_attempted,
        )
        provenance_by_pair[(str(row.get("subtask_id", "")), str(row.get("api_id", "")))] = provenance
        provenance_by_api[str(row.get("api_id", ""))] = provenance

    planned_provenance: List[Dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for steps in [
        (planner.get("primary_plan") or {}).get("steps") if isinstance(planner.get("primary_plan"), dict) else [],
        (planner.get("execution_workflow") or {}).get("steps") if isinstance(planner.get("execution_workflow"), dict) else [],
    ]:
        for step in steps if isinstance(steps, list) else []:
            if not isinstance(step, dict):
                continue
            sid = str(step.get("subtask_id", ""))
            api_id = str(step.get("api_id", ""))
            provenance = provenance_by_pair.get((sid, api_id)) or provenance_by_api.get(api_id)
            if not provenance:
                continue
            step.update(provenance)
            pair = (sid, api_id)
            if pair not in seen_pairs:
                planned_provenance.append(provenance)
                seen_pairs.add(pair)

    planner["planner_provenance"] = planned_provenance
    planner["planner_override_attempted"] = planner_override_attempted
    return planner


def _annotate_planner_selection_diagnostics(
    planner: Dict[str, Any],
    selected_all: List[Dict[str, Any]],
    subtasks: List[Dict[str, Any]],
    selection_trace: Dict[str, Dict[str, Any]],
    *,
    mode_name: str,
    query_dir: Path,
    planner_candidate_mode: str,
) -> Dict[str, Any]:
    candidates_by_pair: Dict[tuple[str, str], Dict[str, Any]] = {}
    candidates_by_subtask: Dict[str, List[Dict[str, Any]]] = {}
    for row in selected_all:
        subtask_id = str(row.get("subtask_id") or "")
        api_id = str(row.get("api_id") or "")
        candidates_by_pair[(subtask_id, api_id)] = row
        candidates_by_subtask.setdefault(subtask_id, []).append(row)

    planner_candidates_per_subtask = {
        str(sub.get("id", "unknown")): len(candidates_by_subtask.get(str(sub.get("id", "unknown")), []))
        for sub in subtasks
    }
    selected_api_ids = _planner_api_ids(planner)
    selected_by_subtask = _planner_api_ids_by_subtask(planner)
    step_reason_by_pair: Dict[tuple[str, str], str | None] = {}
    for step in _planner_steps(planner):
        subtask_id = str(step.get("subtask_id") or "")
        api_id = str(step.get("api_id") or "")
        step_reason_by_pair[(subtask_id, api_id)] = step.get("planner_override_reason") or step.get("why")

    records: List[Dict[str, Any]] = []
    for sub in subtasks:
        subtask_id = str(sub.get("id", "unknown"))
        api_ids = selected_by_subtask.get(subtask_id, [])
        trace = selection_trace.setdefault(subtask_id, {})
        trace["planner_candidate_mode"] = planner_candidate_mode
        trace["planner_top_n_cap"] = _planner_top_n_cap()
        trace["planner_candidates_per_subtask"] = planner_candidates_per_subtask.get(subtask_id, 0)
        trace["planner_candidate_api_ids"] = [str(row.get("api_id") or "") for row in candidates_by_subtask.get(subtask_id, [])]
        trace["selected_api_ids"] = api_ids
        if len(api_ids) != 1:
            trace["planner_selected_rank"] = None
            trace["planner_override_rank1"] = None
            trace["planner_override_reason"] = "planner_did_not_select_exactly_one_api_for_subtask"
            records.append(
                {
                    "subtask_id": subtask_id,
                    "api_id": None,
                    "planner_selected_rank": None,
                    "planner_override_rank1": None,
                    "planner_override_reason": trace["planner_override_reason"],
                    "selected_api_ids": api_ids,
                }
            )
            continue

        api_id = api_ids[0]
        candidate = candidates_by_pair.get((subtask_id, api_id), {})
        selected_rank = candidate.get("selection_order") or candidate.get("selected_rank")
        try:
            selected_rank_int = int(selected_rank)
        except Exception:
            selected_rank_int = None
        override_rank1 = bool(selected_rank_int and selected_rank_int > 1)
        override_reason = step_reason_by_pair.get((subtask_id, api_id)) if override_rank1 else None
        trace["planner_selected_rank"] = selected_rank_int
        trace["planner_override_rank1"] = override_rank1
        trace["planner_override_reason"] = override_reason
        trace["planner_selected_api_id"] = api_id
        record = {
            "subtask_id": subtask_id,
            "api_id": api_id,
            "planner_selected_rank": selected_rank_int,
            "planner_override_rank1": override_rank1,
            "planner_override_reason": override_reason,
            "selected_api_ids": api_ids,
        }
        records.append(record)
        if override_rank1:
            payload = {
                "event_type": "planner_rank1_override",
                "query_dir": str(query_dir),
                "mode": mode_name,
                "subtask_id": subtask_id,
                "api_id": api_id,
                "planner_selected_rank": selected_rank_int,
                "planner_override_reason": override_reason,
                "planner_candidate_mode": planner_candidate_mode,
            }
            log_warning_event(payload)
            log_line(
                f"[{query_dir.name}] mode={mode_name} subtask={subtask_id} "
                f"planner_rank1_override rank={selected_rank_int} api_id={api_id}"
            )

    planner["planner_candidate_mode"] = planner_candidate_mode
    planner["planner_top_n_cap"] = _planner_top_n_cap()
    planner["planner_candidates_per_subtask"] = planner_candidates_per_subtask
    planner["planner_selected_api_ids"] = selected_api_ids
    planner["planner_selection_diagnostics"] = records
    return planner


def _write_selected_outputs(mode_dir: Path, selected_all: List[Dict[str, Any]], selection_trace: Dict[str, Dict[str, Any]], subtasks: List[Dict[str, Any]]) -> None:
    selected_by_subtask: Dict[str, List[Dict[str, Any]]] = {}
    for row in selected_all:
        selected_by_subtask.setdefault(str(row.get("subtask_id", "unknown")), []).append(row)

    for sub in subtasks:
        sub_id = str(sub.get("id", "unknown"))
        selected = selected_by_subtask.get(sub_id, [])
        _write_json(mode_dir / f"3_selected_s{sub_id}.json", selected)
        _write_json(mode_dir / f"3_selected_trace_s{sub_id}.json", selection_trace.get(sub_id, {}))


def _deterministic_select_and_plan(
    mode_name: str,
    subtasks: List[Dict[str, Any]],
    ranked_full: Dict[str, List[Dict[str, Any]]],
    planner_top_n: Dict[str, int],
    llm_call,
    user_goal: str,
    out_dir: Path,
    planner_prompt_path: str,
    functional_match_map: Dict[tuple[str, str], Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    mode_dir = out_dir / mode_name
    functional_match_map = functional_match_map or {}
    planner_candidate_mode = _planner_candidate_mode()

    # This preserves stage separation. It is not a hybrid rescue rule:
    # ranking/selection chooses APIs, planner composes only those APIs, and
    # the evaluator scores the finished workflow.
    if mode_name == "qos_hybrid":
        selected_all, selection_trace, missing_due_to_ranking_failure, subtask_failure_reasons = _select_workflow_hybrid(
            subtasks=subtasks,
            ranked_full=ranked_full,
            functional_match_map=functional_match_map,
            run_label=out_dir.name,
            planner_candidate_mode=planner_candidate_mode,
            planner_top_n=planner_top_n,
        )
    else:
        selected_all, selection_trace, missing_due_to_ranking_failure, subtask_failure_reasons = _select_top_ranked_workflow(
            mode_name=mode_name,
            subtasks=subtasks,
            ranked_full=ranked_full,
            functional_match_map=functional_match_map,
            planner_candidate_mode=planner_candidate_mode,
            planner_top_n=planner_top_n,
        )

    _write_selected_outputs(mode_dir, selected_all, selection_trace, subtasks)
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
    allowed_api_ids = {str(row.get("api_id") or "").strip() for row in selected_all if str(row.get("api_id") or "").strip()}
    planner = planner_call(
        llm_call=lambda p: llm_call(planner_role, PLANNER_SYS, p),
        user_goal=user_goal,
        ranked_top=selected_all,
        subtasks=subtasks,
        prompt_path=planner_prompt_path,
        planner_candidate_mode=planner_candidate_mode,
        planner_top_n_cap=_planner_top_n_cap(),
    )
    planner_override_attempted = False
    unapproved_api_ids = _unapproved_planner_api_ids(planner, allowed_api_ids)
    if unapproved_api_ids:
        planner_override_attempted = True
        log_error_event(
            {
                "event_type": "planner_selected_unapproved_api",
                "query_dir": str(out_dir),
                "mode": mode_name,
                "unapproved_api_ids": unapproved_api_ids,
                "allowed_api_ids": sorted(allowed_api_ids),
            }
        )
        log_line(
            f"[{out_dir.name}] mode={mode_name} planner_selected_unapproved_api "
            f"unapproved={unapproved_api_ids}; retrying once"
        )
        planner = planner_call(
            llm_call=lambda p: llm_call(planner_role, PLANNER_SYS, p),
            user_goal=_planner_retry_goal(
                user_goal,
                unapproved_api_ids,
                allowed_api_ids,
                planner_candidate_mode=planner_candidate_mode,
            ),
            ranked_top=selected_all,
            subtasks=subtasks,
            prompt_path=planner_prompt_path,
            planner_candidate_mode=planner_candidate_mode,
            planner_top_n_cap=_planner_top_n_cap(),
        )
        retry_unapproved_api_ids = _unapproved_planner_api_ids(planner, allowed_api_ids)
        if retry_unapproved_api_ids:
            message = (
                f"Planner for mode {mode_name} used unapproved API IDs after retry: "
                f"{', '.join(retry_unapproved_api_ids)}"
            )
            failure_payload = {
                "failure_stage": "planner_validation",
                "failure_reason": "planner_selected_unapproved_api",
                "mode": mode_name,
                "unapproved_api_ids": retry_unapproved_api_ids,
                "allowed_api_ids": sorted(allowed_api_ids),
                "planner_called": True,
                "planner_retry_count": 1,
            }
            _write_json(mode_dir / "planner_failure.json", failure_payload)
            _write_json(
                mode_dir / "debug" / "planner_unapproved_api_attempts.json",
                {
                    "first_unapproved_api_ids": unapproved_api_ids,
                    "retry_unapproved_api_ids": retry_unapproved_api_ids,
                    "allowed_api_ids": sorted(allowed_api_ids),
                },
            )
            (mode_dir / "planner_error.txt").write_text(message, encoding="utf-8")
            raise PlannerSelectionValidationFailure(message, failure_payload, selection_trace)

    one_api_issues = (
        _planner_one_api_per_subtask_issues(planner, subtasks, selected_all)
        if planner_candidate_mode == "top_n_ablation"
        else []
    )
    if one_api_issues:
        planner_override_attempted = True
        log_warning_event(
            {
                "event_type": "planner_one_api_per_subtask_retry",
                "query_dir": str(out_dir),
                "mode": mode_name,
                "issues": one_api_issues,
                "planner_candidate_mode": planner_candidate_mode,
            }
        )
        log_line(
            f"[{out_dir.name}] mode={mode_name} planner did not choose exactly one API per subtask; retrying once"
        )
        planner = planner_call(
            llm_call=lambda p: llm_call(planner_role, PLANNER_SYS, p),
            user_goal=_planner_one_api_retry_goal(user_goal, one_api_issues),
            ranked_top=selected_all,
            subtasks=subtasks,
            prompt_path=planner_prompt_path,
            planner_candidate_mode=planner_candidate_mode,
            planner_top_n_cap=_planner_top_n_cap(),
        )
        retry_unapproved_api_ids = _unapproved_planner_api_ids(planner, allowed_api_ids)
        retry_one_api_issues = _planner_one_api_per_subtask_issues(planner, subtasks, selected_all)
        if retry_unapproved_api_ids or retry_one_api_issues:
            message = (
                f"Planner for mode {mode_name} failed top-N ablation validation after retry."
            )
            failure_payload = {
                "failure_stage": "planner_validation",
                "failure_reason": "planner_top_n_ablation_validation_failed",
                "mode": mode_name,
                "unapproved_api_ids": retry_unapproved_api_ids,
                "one_api_per_subtask_issues": retry_one_api_issues,
                "allowed_api_ids": sorted(allowed_api_ids),
                "planner_called": True,
                "planner_retry_count": 1,
                "planner_candidate_mode": planner_candidate_mode,
                "planner_top_n_cap": _planner_top_n_cap(),
            }
            _write_json(mode_dir / "planner_failure.json", failure_payload)
            (mode_dir / "planner_error.txt").write_text(message, encoding="utf-8")
            raise PlannerSelectionValidationFailure(message, failure_payload, selection_trace)

    fixed_one_issues = (
        _planner_one_api_per_subtask_issues(planner, subtasks, selected_all)
        if planner_candidate_mode == "fixed_one"
        else []
    )
    if fixed_one_issues:
        planner_override_attempted = True
        log_warning_event(
            {
                "event_type": "planner_fixed_selection_retry",
                "query_dir": str(out_dir),
                "mode": mode_name,
                "issues": fixed_one_issues,
                "planner_candidate_mode": planner_candidate_mode,
            }
        )
        log_line(
            f"[{out_dir.name}] mode={mode_name} planner did not preserve fixed selected APIs; retrying once"
        )
        planner = planner_call(
            llm_call=lambda p: llm_call(planner_role, PLANNER_SYS, p),
            user_goal=_planner_fixed_one_retry_goal(user_goal, fixed_one_issues),
            ranked_top=selected_all,
            subtasks=subtasks,
            prompt_path=planner_prompt_path,
            planner_candidate_mode=planner_candidate_mode,
            planner_top_n_cap=_planner_top_n_cap(),
        )
        retry_unapproved_api_ids = _unapproved_planner_api_ids(planner, allowed_api_ids)
        retry_fixed_one_issues = _planner_one_api_per_subtask_issues(planner, subtasks, selected_all)
        if retry_unapproved_api_ids or retry_fixed_one_issues:
            message = (
                f"Planner for mode {mode_name} failed fixed_one selection validation after retry."
            )
            failure_payload = {
                "failure_stage": "planner_validation",
                "failure_reason": "planner_fixed_selection_validation_failed",
                "mode": mode_name,
                "unapproved_api_ids": retry_unapproved_api_ids,
                "fixed_one_issues": retry_fixed_one_issues,
                "allowed_api_ids": sorted(allowed_api_ids),
                "planner_called": True,
                "planner_retry_count": 1,
                "planner_candidate_mode": planner_candidate_mode,
            }
            _write_json(mode_dir / "planner_failure.json", failure_payload)
            (mode_dir / "planner_error.txt").write_text(message, encoding="utf-8")
            raise PlannerSelectionValidationFailure(message, failure_payload, selection_trace)

    planner = _annotate_planner_provenance(
        planner,
        selected_all,
        mode_name=mode_name,
        planner_override_attempted=planner_override_attempted,
    )
    planner = _annotate_planner_selection_diagnostics(
        planner,
        selected_all,
        subtasks,
        selection_trace,
        mode_name=mode_name,
        query_dir=out_dir,
        planner_candidate_mode=planner_candidate_mode,
    )
    _write_selected_outputs(mode_dir, selected_all, selection_trace, subtasks)
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
    planner_candidate_mode = _planner_candidate_mode()
    selection_rule = (
        "For every mode, selection fixes exactly one primary API per subtask before planning. "
        "qos_hybrid uses workflow-level functional-first QoS selection; other modes use their top ranked valid API. "
        "The planner composes the fixed APIs and may not replace or re-rank them."
    )
    if planner_candidate_mode == "top_n_ablation":
        selection_rule = (
            "Top-N planner ablation is enabled. Each mode sends up to planner_top_n_cap candidates per subtask "
            "from its own ranking or candidate pool. The planner must choose exactly one API per subtask, prefer "
            "selection_order 1, and provide planner_override_reason when choosing a lower-ranked candidate."
        )
    return {
        "query_id": query_id,
        "selection_stage": "planner_input_selection",
        "selection_rule": selection_rule,
        "planner_candidate_mode": planner_candidate_mode,
        "planner_top_n_cap": _planner_top_n_cap(),
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
            "planner_candidate_mode": _planner_candidate_mode(),
            "planner_top_n_cap": _planner_top_n_cap(),
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
    hybrid_retrieved_by_subtask: Dict[str, List[Dict[str, Any]]] = {}
    no_qos_services: Dict[str, Dict[str, Any]] = {}
    with_qos_services: Dict[str, Dict[str, Any]] = {}
    ranked_full_by_mode: Dict[str, Dict[str, List[Dict[str, Any]]]] = {m: {} for m in MODE_ORDER}
    functional_match_map: Dict[tuple[str, str], Dict[str, Any]] = {}
    hybrid_functional_match_map: Dict[tuple[str, str], Dict[str, Any]] = {}
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
    summary_selected = {
        "planner_enabled": CONFIG.planner_enabled,
        "planner_candidate_mode": _planner_candidate_mode(),
        "planner_top_n_cap": _planner_top_n_cap(),
    }
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
            hybrid_retrieved_by_subtask = _copy_retrieved_by_subtask(retrieved_by_subtask)
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
                hybrid_functional_match_map = dict(functional_match_map)
                zero_functional_retry_traces = _retry_zero_functional_retrieval(
                    user_goal=user_goal,
                    subtasks=subtasks,
                    out_dir=out_dir / "qos_hybrid",
                    retrieved_by_subtask=hybrid_retrieved_by_subtask,
                    functional_match_map=functional_match_map,
                    no_qos_services=no_qos_services,
                    with_qos_services=with_qos_services,
                )
                hybrid_functional_match_rows_path: Path | None = None
                if zero_functional_retry_traces:
                    hybrid_view_dir = out_dir / "qos_hybrid"
                    _write_json(
                        hybrid_view_dir / "meta.json",
                        {
                            "query_id": query_id,
                            "user_goal": user_goal,
                        },
                    )
                    _write_json(hybrid_view_dir / "0_decomposer.json", subtasks)
                    _write_retrieval_view(hybrid_view_dir, subtasks, hybrid_retrieved_by_subtask)
                    hybrid_eval_dir = eval_dir / "qos_hybrid"
                    hybrid_functional_match_rows_path = functional_refiner_agent.refine_candidates(
                        query_dir=hybrid_view_dir,
                        query_id=query_id,
                        output_dir=hybrid_eval_dir,
                        cache_path=eval_cache,
                        provider=provider or "azure",
                        model=model,
                    )
                    hybrid_functional_match_map = _load_functional_match_map(hybrid_functional_match_rows_path)
                zero_functional_retry_status = "completed" if zero_functional_retry_traces else "not_needed"
                meta_tracker.update(
                    functional_refinement_status="completed",
                    functional_refinement_rows_json=_to_run_relative(retrieval_functional_match_rows_path, out_dir),
                    hybrid_functional_refinement_rows_json=_to_run_relative(hybrid_functional_match_rows_path, out_dir)
                    if hybrid_functional_match_rows_path
                    else None,
                    functional_refinement_summary_json=_to_run_relative(functional_refinement_summary_path, out_dir)
                    if functional_refinement_summary_path.exists()
                    else None,
                    zero_functional_retrieval_retry_count=len(zero_functional_retry_traces),
                    zero_functional_retrieval_retry_status=zero_functional_retry_status,
                    zero_functional_retrieval_retry_scope="qos_hybrid",
                    zero_functional_retrieval_retries=zero_functional_retry_traces,
                )
                meta_tracker.finish_stage(
                    "functional_refinement",
                    rows_json=_to_run_relative(retrieval_functional_match_rows_path, out_dir),
                    hybrid_rows_json=_to_run_relative(hybrid_functional_match_rows_path, out_dir)
                    if hybrid_functional_match_rows_path
                    else None,
                    summary_json=_to_run_relative(functional_refinement_summary_path, out_dir)
                    if functional_refinement_summary_path.exists()
                    else None,
                    functional_refiner_agent="FunctionalRefinerAgent",
                    functional_refiner_agent_mode="llm_binary_functional_labeling",
                    functional_refinement_status="completed",
                    zero_functional_retrieval_retry_count=len(zero_functional_retry_traces),
                    zero_functional_retrieval_retry_status=zero_functional_retry_status,
                    zero_functional_retrieval_retry_scope="qos_hybrid",
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
                    pure_enrichment: Dict[str, Dict[str, Any]] = {
                        api_id: dict(meta)
                        for api_id, meta in pure_qos_meta.items()
                        if isinstance(meta, dict)
                    }
                    for api_id, functional_meta in _functional_refinement_enrichment(
                        retrieved,
                        subtask_id=sub_id,
                        functional_match_map=functional_match_map,
                    ).items():
                        pure_enrichment.setdefault(api_id, {}).update(functional_meta)
                    pure_candidates = _candidate_rows(retrieved, with_qos_services, enrich=pure_enrichment)
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
                                include_functional_match_label=True,
                                enforce_functional_first=True,
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

                hybrid_retrieved = hybrid_retrieved_by_subtask.get(sub_id, retrieved)
                hybrid_topsis_meta = (
                    topsis_meta
                    if hybrid_retrieved is retrieved
                    else _compute_topsis_metadata(hybrid_retrieved, with_qos_services)
                )
                hybrid_ranked, hybrid_duration = _timed_invocation(
                    meta_tracker,
                    "qos_hybrid",
                    lambda: _deterministic_hybrid_ranking(
                        hybrid_retrieved,
                        hybrid_topsis_meta,
                        sub_id,
                        hybrid_functional_match_map,
                    ),
                )
                ranked_full_by_mode["qos_hybrid"][sub_id] = _write_ranked(
                    out_dir / "qos_hybrid",
                    sub_id,
                    hybrid_ranked,
                    hybrid_retrieved,
                    with_qos_services,
                    hybrid_topsis_meta,
                )
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
            log_line(
                f"[{run_label}] starting planner "
                f"candidate_mode={_planner_candidate_mode()} top_n_cap={_planner_top_n_cap()}"
            )
            planner_failures: List[Dict[str, Any]] = []
            planner_selection_k_by_mode_subtask: Dict[str, Dict[str, Dict[str, Any]]] = {}
            planner_selection_k_summary_path = eval_dir / f"query_{query_id}_planner_selection_k_summary.json"
            for mode in MODE_ORDER:
                try:
                    planner_prompt = "prompts/planner_no_qos.md" if mode == "no_qos" else "prompts/planner_qos.md"
                    result = _deterministic_select_and_plan(
                        mode,
                        subtasks,
                        ranked_full_by_mode[mode],
                        planner_top_n,
                        llm_call,
                        user_goal,
                        out_dir,
                        planner_prompt,
                        functional_match_map=functional_match_map,
                    )
                    summary_selected[f"{mode}_selected"] = len(result["selected"])
                    planner_selection_k_by_mode_subtask[mode] = result.get("selection_trace", {})
                except Exception as exc:
                    if isinstance(exc, (PlannerPrecheckFailure, PlannerSelectionValidationFailure)):
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
