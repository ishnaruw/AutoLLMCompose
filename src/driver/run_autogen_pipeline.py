from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.agents.decomposer import decompose_goal
from src.agents.planner import planner_call
from src.agents.ranker import rank_subtask
from src.agents.retriever import collect_candidates
from src.agents.selector import select_ranked_relevant_apis
from src.config import CONFIG, QUERIES_PATH
from src.core.retry import call_with_backoff
from src.eval.api_relevancy_eval import evaluate_query
from src.eval.topsis_eval import _extract_qos, _run_topsis_pydecision
from src.llm.backends import make_backend
from src.tools.fetch_services import load_catalog_map

DECOMPOSER_SYS = "You are a task decomposition agent. Return strict JSON only."
RANKER_SYS = "You are an API ranking agent. Return strict JSON only."
PLANNER_SYS = "You are an API workflow planner. Return strict JSON only."

MODE_ORDER = ["no_qos", "qos_pure_llm", "qos_topsis"]

PROVIDER_POLICY = {
    "mistral": {"sleep_after_query": 0.0},
    "groq": {"sleep_after_query": 0.0},
    "azure_foundry": {"sleep_after_query": 0.0},
    "lmstudio": {"sleep_after_query": 0.0},
    "_default": {"sleep_after_query": 0.0},
}


def load_queries(path: Path = QUERIES_PATH) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Queries file not found: {path.resolve()}")
    queries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def choose_provider_interactive() -> str:
    options = [
        ("mistral", "Mistral"),
        ("groq", "Groq"),
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


def _safe_name(text: str) -> str:
    text = (text or "unknown").strip()
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() or ch in {"-", "_"} else "_")
    safe = "".join(out).strip("_")
    return safe or "unknown"


def _run_dir(model_tag: str, query_id: str | None = None, run_tag: str | None = None) -> Path:
    run_id = time.strftime("%Y%m%dT%H%M%S")
    run_name = f"{_safe_name(query_id)}_{run_id}" if CONFIG.prefix_run_dir_with_query_id and query_id else run_id
    base_dir = Path("results/logs")
    if run_tag:
        base_dir = base_dir / _safe_name(run_tag)
    out = base_dir / model_tag.replace(":", "_") / run_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _get_policy(provider: str | None) -> Dict[str, Any]:
    return PROVIDER_POLICY.get(provider or "_default", PROVIDER_POLICY["_default"])


def _throttle_after_query(provider: str | None) -> None:
    delay = float((_get_policy(provider) or {}).get("sleep_after_query", 0.0) or 0.0)
    if delay > 0:
        time.sleep(delay)


def _build_llm_call(backend):
    def llm_call(role_name: str, system_msg: str, prompt: str) -> str:
        temp = 0.2 if role_name == "planner" else 0.0
        return call_with_backoff(
            lambda: backend.chat_json(system_msg, prompt, temperature=temp, force_json=True),
            name=role_name,
        )
    return llm_call


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _normalize_service_with_qos(base_service: Dict[str, Any], with_qos_service: Dict[str, Any] | None) -> Dict[str, Any]:
    row = dict(base_service or {})
    if isinstance(with_qos_service, dict):
        qos = with_qos_service.get("qos") if isinstance(with_qos_service.get("qos"), dict) else None
        if qos:
            row["qos"] = qos
        else:
            qos_fields = {k: with_qos_service.get(k) for k in ["rt_ms", "tp_rps", "availability"] if with_qos_service.get(k) is not None}
            if qos_fields:
                row["qos"] = qos_fields
        for key in ["rt_ms", "tp_rps", "availability"]:
            if key in with_qos_service and with_qos_service.get(key) is not None:
                row[key] = with_qos_service.get(key)
    return row


def _prepare_shared_candidates(
    *,
    user_goal: str,
    subtasks: List[Dict[str, Any]],
    index_dir: Path,
    no_qos_catalog: Dict[str, Dict[str, Any]],
    with_qos_catalog: Dict[str, Dict[str, Any]],
) -> tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    shared_retrieved_by_subtask: Dict[str, List[Dict[str, Any]]] = {}
    shared_id_to_service_no_qos: Dict[str, Dict[str, Any]] = {}
    shared_id_to_service_with_qos: Dict[str, Dict[str, Any]] = {}

    for sub in subtasks:
        sub_id = str(sub.get("id", "unknown"))
        retrieved = collect_candidates(
            user_query=user_goal,
            subtask_goal=str(sub.get("description", "")),
            index_dir=str(index_dir),
            top_k=CONFIG.rag_top_k,
        )
        retrieved = retrieved[: CONFIG.rag_top_k]
        for idx, item in enumerate(retrieved, start=1):
            api_id = str(item.get("api_id", "")).strip()
            item["retrieved_rank"] = idx
            if not api_id:
                continue
            no_qos_service = no_qos_catalog.get(api_id, {})
            with_qos_service = with_qos_catalog.get(api_id, {})
            shared_id_to_service_no_qos[api_id] = dict(no_qos_service)
            shared_id_to_service_with_qos[api_id] = _normalize_service_with_qos(no_qos_service, with_qos_service)
        shared_retrieved_by_subtask[sub_id] = retrieved
    return shared_retrieved_by_subtask, shared_id_to_service_no_qos, shared_id_to_service_with_qos


def _rank_subtasks_llm(
    *,
    llm_call,
    user_goal: str,
    subtasks: List[Dict[str, Any]],
    retrieved_by_subtask: Dict[str, List[Dict[str, Any]]],
    id_to_service: Dict[str, Dict[str, Any]],
    ranker_prompt_path: str,
    mode_dir: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    ranked_by_subtask: Dict[str, List[Dict[str, Any]]] = {}
    for sub in subtasks:
        sub_id = str(sub.get("id", "unknown"))
        candidates_for_ranker: List[Dict[str, Any]] = []
        for r in retrieved_by_subtask.get(sub_id, []):
            api_id = str(r.get("api_id", "")).strip()
            if not api_id:
                continue
            candidates_for_ranker.append(
                {
                    "api_id": api_id,
                    "rag_score": r.get("rag_score", 0.0),
                    "retrieved_rank": r.get("retrieved_rank"),
                    "compressed": r.get("compressed", {}),
                    "service": id_to_service.get(api_id, {}),
                }
            )
        ranked = rank_subtask(
            llm_call=lambda p: llm_call("ranker", RANKER_SYS, p),
            user_query=user_goal,
            subtask=sub,
            candidates=candidates_for_ranker,
            prompt_path=ranker_prompt_path,
            debug_raw_path=str(mode_dir / f"2_ranker_raw_s{sub_id}.txt"),
        )
        for item in ranked:
            api_id = str(item.get("api_id", "")).strip()
            retrieved = next((x for x in retrieved_by_subtask.get(sub_id, []) if str(x.get("api_id", "")).strip() == api_id), {})
            item["retrieved_rank"] = retrieved.get("retrieved_rank")
            item["rag_score"] = retrieved.get("rag_score", 0.0)
            item["service"] = id_to_service.get(api_id, {})
        ranked_by_subtask[sub_id] = ranked
        _write_json(mode_dir / f"2_ranked_s{sub_id}.json", ranked)
    return ranked_by_subtask


def _rank_subtasks_topsis(
    *,
    subtasks: List[Dict[str, Any]],
    retrieved_by_subtask: Dict[str, List[Dict[str, Any]]],
    id_to_service_with_qos: Dict[str, Dict[str, Any]],
    mode_dir: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    weights = list(CONFIG.qos_metric_weights)
    ranked_by_subtask: Dict[str, List[Dict[str, Any]]] = {}
    for sub in subtasks:
        sub_id = str(sub.get("id", "unknown"))
        retrieved = retrieved_by_subtask.get(sub_id, [])
        valid_rows: List[Dict[str, Any]] = []
        valid_matrix: List[List[float]] = []
        missing_rows: List[Dict[str, Any]] = []
        for item in retrieved:
            api_id = str(item.get("api_id", "")).strip()
            service = id_to_service_with_qos.get(api_id, {})
            qos_source = service.get("qos") if isinstance(service.get("qos"), dict) else service
            qos_vals = _extract_qos(qos_source if isinstance(qos_source, dict) else {})
            row = {
                "api_id": api_id,
                "reason": "TOPSIS QoS ranking.",
                "retrieved_rank": item.get("retrieved_rank"),
                "rag_score": item.get("rag_score", 0.0),
                "service": service,
            }
            if qos_vals is None:
                row["reason"] = "TOPSIS QoS ranking: missing QoS, placed at bottom."
                missing_rows.append(row)
            else:
                valid_rows.append(row)
                valid_matrix.append(qos_vals)

        ranked: List[Dict[str, Any]] = []
        if valid_rows:
            import numpy as np
            X = np.asarray(valid_matrix, dtype=float)
            scores, ranking_idx = _run_topsis_pydecision(X, weights)
            for order_idx, row_idx in enumerate(ranking_idx, start=1):
                row = dict(valid_rows[row_idx])
                row["mode_rank"] = order_idx
                row["reason"] = "TOPSIS QoS ranking with equal metric weights."
                ranked.append(row)
        for miss in missing_rows:
            miss = dict(miss)
            miss["mode_rank"] = len(ranked) + 1
            ranked.append(miss)
        ranked_by_subtask[sub_id] = ranked
        _write_json(mode_dir / f"2_ranked_s{sub_id}.json", ranked)
    return ranked_by_subtask


def _run_deterministic_selector(
    *,
    subtasks: List[Dict[str, Any]],
    ranked_by_subtask: Dict[str, List[Dict[str, Any]]],
    relevancy_by_subtask: Dict[str, Dict[str, Dict[str, Any]]],
    mode_dir: Path,
) -> List[Dict[str, Any]]:
    selected_all: List[Dict[str, Any]] = []
    for sub in subtasks:
        sub_id = str(sub.get("id", "unknown"))
        selected, trace = select_ranked_relevant_apis(
            ranked_candidates=ranked_by_subtask.get(sub_id, []),
            relevancy_map=relevancy_by_subtask.get(sub_id, {}),
            fallback_top_n=CONFIG.selector_fallback_top_n,
        )
        for idx, item in enumerate(selected, start=1):
            item["selected_rank"] = idx
            item["subtask_id"] = sub_id
        selected_all.extend(selected)
        _write_json(mode_dir / f"3_selected_s{sub_id}.json", selected)
        _write_json(mode_dir / f"3_selected_trace_s{sub_id}.json", trace)
    return selected_all


def _run_planner(*, llm_call, user_goal: str, subtasks: List[Dict[str, Any]], selected_all: List[Dict[str, Any]], prompt_path: str) -> Dict[str, Any]:
    return planner_call(
        llm_call=lambda p: llm_call("planner", PLANNER_SYS, p),
        user_goal=user_goal,
        ranked_top=selected_all,
        subtasks=subtasks,
        prompt_path=prompt_path,
    )


def run_autogen_once(
    user_goal: str,
    provider: str | None = None,
    model: str | None = None,
    query_id: str | None = None,
    query_title: str | None = None,
    run_tag: str | None = None,
) -> Tuple[Path, Path | None]:
    backend = make_backend(provider=provider, model=model)
    model_tag = backend.name()
    llm_call = _build_llm_call(backend)
    out_dir = _run_dir(model_tag, query_id=query_id, run_tag=run_tag)

    raw_subtasks = decompose_goal(
        llm_call=lambda p: llm_call("decomposer", DECOMPOSER_SYS, p),
        user_goal=user_goal,
    )
    subtasks = raw_subtasks
    _write_json(out_dir / "0_decomposer_raw.json", raw_subtasks)
    _write_json(out_dir / "0_subtask_normalizer.json", {"disabled": True, "subtasks": subtasks})
    _write_json(out_dir / "0_decomposer.json", subtasks)

    no_qos_catalog = load_catalog_map(with_qos=False)
    with_qos_catalog = load_catalog_map(with_qos=True)

    shared_dir = out_dir / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    shared_retrieved_by_subtask, id_to_service_no_qos, id_to_service_with_qos = _prepare_shared_candidates(
        user_goal=user_goal,
        subtasks=subtasks,
        index_dir=CONFIG.shared_index_dir,
        no_qos_catalog=no_qos_catalog,
        with_qos_catalog=with_qos_catalog,
    )
    for sub in subtasks:
        sub_id = str(sub.get("id", "unknown"))
        _write_json(shared_dir / f"1_retriever_s{sub_id}.json", shared_retrieved_by_subtask.get(sub_id, []))

    no_qos_dir = out_dir / "no_qos"
    qos_pure_dir = out_dir / "qos_pure_llm"
    qos_topsis_dir = out_dir / "qos_topsis"
    for d in [no_qos_dir, qos_pure_dir, qos_topsis_dir]:
        d.mkdir(parents=True, exist_ok=True)

    no_qos_ranked = _rank_subtasks_llm(
        llm_call=llm_call,
        user_goal=user_goal,
        subtasks=subtasks,
        retrieved_by_subtask=shared_retrieved_by_subtask,
        id_to_service=id_to_service_no_qos,
        ranker_prompt_path="prompts/ranker_no_qos.md",
        mode_dir=no_qos_dir,
    )
    qos_pure_ranked = _rank_subtasks_llm(
        llm_call=llm_call,
        user_goal=user_goal,
        subtasks=subtasks,
        retrieved_by_subtask=shared_retrieved_by_subtask,
        id_to_service=id_to_service_with_qos,
        ranker_prompt_path="prompts/ranker.md",
        mode_dir=qos_pure_dir,
    )
    qos_topsis_ranked = _rank_subtasks_topsis(
        subtasks=subtasks,
        retrieved_by_subtask=shared_retrieved_by_subtask,
        id_to_service_with_qos=id_to_service_with_qos,
        mode_dir=qos_topsis_dir,
    )

    eval_out: Path | None = None
    relevancy_by_subtask: Dict[str, Dict[str, Dict[str, Any]]] = {}
    try:
        eval_dir = out_dir / "relevancy_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_cache = eval_dir / "relevancy_cache.json"
        eval_out, relevancy_by_subtask = evaluate_query(
            query_dir=out_dir,
            query_id=query_id,
            output_dir=eval_dir,
            cache_path=eval_cache,
            provider=provider or "azure_foundry",
            model=model,
        )
        _write_json(
            out_dir / "evaluation_result.json",
            {
                "excel": str(eval_out),
                "output_dir": str(eval_dir),
                "cache_path": str(eval_cache),
            },
        )
    except Exception as e:
        (out_dir / "evaluation_error.txt").write_text(str(e), encoding="utf-8")

    no_qos_selected = _run_deterministic_selector(
        subtasks=subtasks,
        ranked_by_subtask=no_qos_ranked,
        relevancy_by_subtask=relevancy_by_subtask,
        mode_dir=no_qos_dir,
    )
    qos_pure_selected = _run_deterministic_selector(
        subtasks=subtasks,
        ranked_by_subtask=qos_pure_ranked,
        relevancy_by_subtask=relevancy_by_subtask,
        mode_dir=qos_pure_dir,
    )
    qos_topsis_selected = _run_deterministic_selector(
        subtasks=subtasks,
        ranked_by_subtask=qos_topsis_ranked,
        relevancy_by_subtask=relevancy_by_subtask,
        mode_dir=qos_topsis_dir,
    )

    _write_json(no_qos_dir / "4_planner.json", _run_planner(llm_call=llm_call, user_goal=user_goal, subtasks=subtasks, selected_all=no_qos_selected, prompt_path="prompts/planner_no_qos.md"))
    _write_json(qos_pure_dir / "4_planner.json", _run_planner(llm_call=llm_call, user_goal=user_goal, subtasks=subtasks, selected_all=qos_pure_selected, prompt_path="prompts/planner.md"))
    _write_json(qos_topsis_dir / "4_planner.json", _run_planner(llm_call=llm_call, user_goal=user_goal, subtasks=subtasks, selected_all=qos_topsis_selected, prompt_path="prompts/planner.md"))

    run_config = CONFIG.as_dict()
    run_config.update({"provider": provider, "model": backend.name(), "query_id": query_id, "query_title": query_title, "modes": MODE_ORDER})
    _write_json(out_dir / "run_config.json", run_config)
    _write_json(out_dir / "meta.json", {
        "query_id": query_id,
        "query_title": query_title,
        "user_goal": user_goal,
        "model_tag": backend.name(),
        "num_subtasks": len(subtasks),
        "shared_retrieval": True,
        "shared_rag_top_k": CONFIG.rag_top_k,
        "evaluation_triggered": True,
        "modes": MODE_ORDER,
    })

    print(f"Saved run to {out_dir}")
    if eval_out is not None:
        print(f"Saved evaluation to {eval_out}")
    _throttle_after_query(provider)
    return out_dir, eval_out


if __name__ == "__main__":
    provider = choose_provider_interactive()
    queries = load_queries()
    print(f"Loaded {len(queries)} queries from file.\n")
    for i, q in enumerate(queries, start=1):
        goal = q.get("goal", "")
        qid = q.get("id", f"q{i:02d}")
        title = q.get("title", "")
        print("\n" + "=" * 80)
        print(f"Running query {i}/{len(queries)} | {qid} | {title}")
        print(f"User goal: {goal}")
        print("=" * 80)
        run_autogen_once(user_goal=goal, provider=provider, model=None, query_id=qid, query_title=title, run_tag="run_5")
