from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.agents.decomposer import decompose_goal
from src.agents.planner import planner_call
from src.agents.ranker import rank_subtask
from src.agents.retriever import collect_candidates
from src.agents.selector import select_top_apis_for_subtask
from src.config import CONFIG
from src.core.retry import call_with_backoff
from src.eval.api_relevancy_eval import evaluate_query
from src.llm.backends import make_backend
from src.tools.fetch_services import fetch_services

DECOMPOSER_SYS = (
    "You are a task decomposition agent. "
    "You break a user goal into ordered subtasks. "
    "Return strict JSON as instructed by the prompt."
)

RANKER_SYS = (
    "You are a ranking agent. Given the original user query, a single subtask, and "
    "a list of candidate APIs from a catalog, rank the candidates best-to-worst for that subtask. "
    "Follow the prompt strictly and return valid JSON."
)

PLANNER_SYS = (
    "You are an orchestration planner that composes a logical API workflow "
    "using only the selected APIs provided. Preserve the ordered subtasks and return valid JSON."
)

MODE_ORDER = ["no_qos", "qos_pure_llm", "qos_topsis"]

PROVIDER_POLICY = {
    "mistral": {"sleep_after_query": 0.5},
    "groq": {"sleep_after_query": 0.8},
    "gemini": {"sleep_after_query": 0.4},
    "azure_foundry": {"sleep_after_query": 0.2},
    "azure": {"sleep_after_query": 0.2},
    "lmstudio": {"sleep_after_query": 0.0},
    "_default": {"sleep_after_query": 0.4},
}


def load_queries(path: Path = CONFIG.queries_path) -> List[Dict[str, Any]]:
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


def _run_dir(model_tag: str, query_id: str | None = None) -> Path:
    run_id = time.strftime("%Y%m%dT%H%M%S")
    run_name = f"{_safe_name(query_id)}_{run_id}" if CONFIG.prefix_run_dir_with_query_id and query_id else run_id
    out = Path("results/logs") / model_tag.replace(":", "_") / run_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _get_policy(provider: str | None) -> Dict[str, Any]:
    return PROVIDER_POLICY.get(provider or "_default", PROVIDER_POLICY["_default"])


def _throttle_after_query(provider: str | None) -> None:
    delay = float((_get_policy(provider) or {}).get("sleep_after_query", 0.0) or 0.0)
    if delay > 0:
        time.sleep(delay)


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


def _rank_subtasks(
    *,
    llm_call,
    user_goal: str,
    subtasks: List[Dict[str, Any]],
    retrieved_by_subtask: Dict[str, List[Dict[str, Any]]],
    id_to_service: Dict[str, Dict[str, Any]],
    ranker_prompt_path: str,
) -> Dict[str, List[Dict[str, Any]]]:
    ranked_by_subtask: Dict[str, List[Dict[str, Any]]] = {}
    for sub in subtasks:
        sub_id = str(sub.get("id", "unknown"))
        candidates_for_ranker: List[Dict[str, Any]] = []
        for r in retrieved_by_subtask.get(sub_id, []):
            api_id = r.get("api_id")
            if not api_id:
                continue
            candidates_for_ranker.append(
                {
                    "api_id": api_id,
                    "rag_score": r.get("rag_score", 0.0),
                    "compressed": r.get("compressed", {}),
                    "service": id_to_service.get(str(api_id), {}),
                }
            )

        ranked_by_subtask[sub_id] = rank_subtask(
            llm_call=lambda p: llm_call("ranker", RANKER_SYS, p),
            user_query=user_goal,
            subtask=sub,
            candidates=candidates_for_ranker,
            prompt_path=ranker_prompt_path,
        )
    return ranked_by_subtask


def _build_ranked_pool_for_subtask(
    *,
    subtask: Dict[str, Any],
    ranked: List[Dict[str, Any]],
    retrieved: List[Dict[str, Any]],
    id_to_service: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rag_map = {str(r.get("api_id")): float(r.get("rag_score", 0.0) or 0.0) for r in retrieved}
    pool: List[Dict[str, Any]] = []
    for idx, r in enumerate((ranked or [])[: CONFIG.ranker_pool_n], start=1):
        api_id = str(r.get("api_id", ""))
        if not api_id:
            continue
        pool.append(
            {
                "api_id": api_id,
                "rank": idx,
                "subtask_id": subtask.get("id"),
                "rag_score": rag_map.get(api_id, 0.0),
                "reason": r.get("reason", "") or "",
                "service": id_to_service.get(api_id, {}),
            }
        )
    return pool


def _add_selected_scores(selected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    n = len(selected)
    for i, item in enumerate(selected, start=1):
        row = dict(item)
        row["selected_rank"] = i
        row["score"] = (n - i + 1) / float(n) if n else 0.0
        out.append(row)
    return out


def _run_mode(
    *,
    mode_name: str,
    user_goal: str,
    subtasks: List[Dict[str, Any]],
    llm_call,
    out_dir: Path,
    with_qos: bool,
    index_dir: Path,
    ranker_prompt_path: str,
    planner_prompt_path: str,
    selector_mode: str,
    shared_ranked_by_subtask: Dict[str, List[Dict[str, Any]]] | None = None,
    shared_retrieved_by_subtask: Dict[str, List[Dict[str, Any]]] | None = None,
    shared_id_to_service: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    mode_dir = out_dir / mode_name
    mode_dir.mkdir(parents=True, exist_ok=True)

    if shared_retrieved_by_subtask is None or shared_id_to_service is None:
        retrieved_by_subtask: Dict[str, List[Dict[str, Any]]] = {}
        pick_ids: List[str] = []
        seen = set()
        for sub in subtasks:
            sub_id = str(sub.get("id", "unknown"))
            retrieved = collect_candidates(
                user_query=user_goal,
                subtask_goal=str(sub.get("description", "")),
                index_dir=str(index_dir),
                top_k=CONFIG.rag_top_k,
            )
            retrieved_by_subtask[sub_id] = retrieved
            for item in retrieved:
                api_id = str(item.get("api_id", ""))
                if api_id and api_id not in seen:
                    seen.add(api_id)
                    pick_ids.append(api_id)
        id_to_service = _fetch_catalog_subset(pick_ids, with_qos=with_qos)
    else:
        retrieved_by_subtask = shared_retrieved_by_subtask
        id_to_service = shared_id_to_service

    if shared_ranked_by_subtask is None:
        ranked_by_subtask = _rank_subtasks(
            llm_call=llm_call,
            user_goal=user_goal,
            subtasks=subtasks,
            retrieved_by_subtask=retrieved_by_subtask,
            id_to_service=id_to_service,
            ranker_prompt_path=ranker_prompt_path,
        )
    else:
        ranked_by_subtask = shared_ranked_by_subtask

    selected_all: List[Dict[str, Any]] = []
    selector_traces: Dict[str, Any] = {}

    for sub in subtasks:
        sub_id = str(sub.get("id", "unknown"))
        ranked_pool = _build_ranked_pool_for_subtask(
            subtask=sub,
            ranked=ranked_by_subtask.get(sub_id, []),
            retrieved=retrieved_by_subtask.get(sub_id, []),
            id_to_service=id_to_service,
        )
        selected, trace = select_top_apis_for_subtask(
            subtask_id=sub_id,
            ranked_pool=ranked_pool,
            mode_override=selector_mode,
            top_n=CONFIG.selector_top_n,
            topsis_top_k=CONFIG.topsis_top_k,
            min_qos_candidates=CONFIG.topsis_min_qos_candidates,
        )
        selected = _add_selected_scores(selected)
        selected_all.extend(selected)
        selector_traces[sub_id] = trace

        _write_json(mode_dir / f"1_retriever_s{sub_id}.json", retrieved_by_subtask.get(sub_id, []))
        _write_json(mode_dir / f"2_ranker_pool_s{sub_id}.json", ranked_pool)
        _write_json(mode_dir / f"3_selected_s{sub_id}.json", selected)
        _write_json(mode_dir / f"3_selected_trace_s{sub_id}.json", trace)

    planner = planner_call(
        llm_call=lambda p: llm_call("planner", PLANNER_SYS, p),
        user_goal=user_goal,
        ranked_top=selected_all,
        subtasks=subtasks,
        prompt_path=planner_prompt_path,
    )
    _write_json(mode_dir / "4_planner.json", planner)

    return {
        "retrieved_by_subtask": retrieved_by_subtask,
        "ranked_by_subtask": ranked_by_subtask,
        "selected": selected_all,
        "planner": planner,
        "selector_traces": selector_traces,
        "id_to_service": id_to_service,
    }


def run_autogen_once(
    user_goal: str,
    provider: str | None = None,
    model: str | None = None,
    query_id: str | None = None,
    query_title: str | None = None,
) -> Tuple[Path, Path | None]:
    backend = make_backend(provider=provider, model=model)
    model_tag = backend.name()
    llm_call = _build_llm_call(backend)
    out_dir = _run_dir(model_tag, query_id=query_id)

    subtasks = decompose_goal(
        llm_call=lambda p: llm_call("decomposer", DECOMPOSER_SYS, p),
        user_goal=user_goal,
    )
    _write_json(out_dir / "0_decomposer.json", subtasks)

    no_qos_result = _run_mode(
        mode_name="no_qos",
        user_goal=user_goal,
        subtasks=subtasks,
        llm_call=llm_call,
        out_dir=out_dir,
        with_qos=False,
        index_dir=CONFIG.no_qos_index_dir,
        ranker_prompt_path="prompts/ranker_no_qos.md",
        planner_prompt_path="prompts/planner_no_qos.md",
        selector_mode="PURE_LLM",
    )

    qos_retrieved_by_subtask: Dict[str, List[Dict[str, Any]]] = {}
    qos_pick_ids: List[str] = []
    seen_qos = set()
    for sub in subtasks:
        sub_id = str(sub.get("id", "unknown"))
        retrieved = collect_candidates(
            user_query=user_goal,
            subtask_goal=str(sub.get("description", "")),
            index_dir=str(CONFIG.with_qos_index_dir),
            top_k=CONFIG.rag_top_k,
        )
        qos_retrieved_by_subtask[sub_id] = retrieved
        for item in retrieved:
            api_id = str(item.get("api_id", ""))
            if api_id and api_id not in seen_qos:
                seen_qos.add(api_id)
                qos_pick_ids.append(api_id)

    qos_id_to_service = _fetch_catalog_subset(qos_pick_ids, with_qos=True)
    qos_ranked_by_subtask = _rank_subtasks(
        llm_call=llm_call,
        user_goal=user_goal,
        subtasks=subtasks,
        retrieved_by_subtask=qos_retrieved_by_subtask,
        id_to_service=qos_id_to_service,
        ranker_prompt_path="prompts/ranker.md",
    )

    qos_pure_result = _run_mode(
        mode_name="qos_pure_llm",
        user_goal=user_goal,
        subtasks=subtasks,
        llm_call=llm_call,
        out_dir=out_dir,
        with_qos=True,
        index_dir=CONFIG.with_qos_index_dir,
        ranker_prompt_path="prompts/ranker.md",
        planner_prompt_path="prompts/planner.md",
        selector_mode="PURE_LLM",
        shared_ranked_by_subtask=qos_ranked_by_subtask,
        shared_retrieved_by_subtask=qos_retrieved_by_subtask,
        shared_id_to_service=qos_id_to_service,
    )

    qos_topsis_result = _run_mode(
        mode_name="qos_topsis",
        user_goal=user_goal,
        subtasks=subtasks,
        llm_call=llm_call,
        out_dir=out_dir,
        with_qos=True,
        index_dir=CONFIG.with_qos_index_dir,
        ranker_prompt_path="prompts/ranker.md",
        planner_prompt_path="prompts/planner.md",
        selector_mode="TOPSIS",
        shared_ranked_by_subtask=qos_ranked_by_subtask,
        shared_retrieved_by_subtask=qos_retrieved_by_subtask,
        shared_id_to_service=qos_id_to_service,
    )

    run_config = CONFIG.as_dict()
    run_config.update(
        {
            "provider": provider,
            "model": backend.name(),
            "query_id": query_id,
            "query_title": query_title,
            "modes": MODE_ORDER,
        }
    )
    _write_json(out_dir / "run_config.json", run_config)
    _write_json(
        out_dir / "meta.json",
        {
            "query_id": query_id,
            "query_title": query_title,
            "user_goal": user_goal,
            "model_tag": backend.name(),
            "num_subtasks": len(subtasks),
            "evaluation_triggered": True,
            "qos_ranker_pool_shared": True,
            "modes": MODE_ORDER,
            "summary": {
                "no_qos_selected": len(no_qos_result["selected"]),
                "qos_pure_llm_selected": len(qos_pure_result["selected"]),
                "qos_topsis_selected": len(qos_topsis_result["selected"]),
            },
        },
    )

    eval_out: Path | None = None
    try:
        eval_out = evaluate_query(
            query_dir=out_dir,
            query_id=query_id,
            provider=provider or "azure",
            model=model,
        )
        _write_json(out_dir / "evaluation_result.json", {"excel": str(eval_out)})
    except Exception as e:
        (out_dir / "evaluation_error.txt").write_text(str(e), encoding="utf-8")

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

        run_autogen_once(
            user_goal=goal,
            provider=provider,
            model=None,
            query_id=qid,
            query_title=title,
        )
