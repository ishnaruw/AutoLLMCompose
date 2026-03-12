# src/driver/run_autogen_pipeline.py

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.config import CONFIG
from src.llm.backends import make_backend
from src.tools.fetch_services import fetch_services
from src.agents.decomposer import decompose_goal
from src.agents.retriever import collect_candidates
from src.agents.ranker import rank_subtask
from src.agents.selector import select_top_apis_for_subtask
from src.agents.planner import planner_call
from src.core.retry import call_with_backoff


QUERIES_PATH = Path(CONFIG.queries_path)

DECOMPOSER_SYS = (
    "You are a task decomposition agent. "
    "You break a user goal into ordered subtasks. "
    "Return strict JSON as instructed by the prompt."
)

RANKER_SYS = (
    "You are a ranking agent. Given the original user query, a single subtask, and "
    "a list of candidate APIs from a catalog, rank the candidates best-to-worst for "
    "that subtask. Follow the prompt strictly and return valid JSON."
)

PLANNER_SYS = (
    "You are an orchestration planner that composes a logical sequential API workflow "
    "using only the selected APIs provided. Follow the prompt strictly and return valid JSON."
)

PROVIDER_POLICY = {
    "mistral":       {"sleep_after_query": 0.5},
    "groq":          {"sleep_after_query": 0.8},
    "gemini":        {"sleep_after_query": 0.4},
    "azure_foundry": {"sleep_after_query": 0.2},
    "azure":         {"sleep_after_query": 0.2},
    "_default":      {"sleep_after_query": 0.4},
}


def load_queries(path: Path = QUERIES_PATH) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Queries file not found: {path.resolve()}")
    queries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
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


def _run_dir(model_tag: str) -> Path:
    run_id = time.strftime("%Y%m%dT%H%M%S")
    d = Path(CONFIG.results_root) / model_tag / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_policy(provider: str | None) -> Dict[str, Any]:
    provider = provider or "_default"
    return PROVIDER_POLICY.get(provider, PROVIDER_POLICY["_default"])


def throttle(provider: str | None) -> None:
    t = float(get_policy(provider).get("sleep_after_query", 0.0) or 0.0)
    if t > 0:
        time.sleep(t)


def _mode_dir(base: Path, mode_name: str) -> Path:
    d = base / mode_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fetch_services_by_ids(api_ids: List[str], *, with_qos: bool) -> Dict[str, Dict[str, Any]]:
    wanted = set(api_ids)
    out: Dict[str, Dict[str, Any]] = {}
    offset = 0
    while True:
        batch = fetch_services(category=None, offset=offset, limit=200, with_qos=with_qos)
        if not batch:
            break
        for item in batch:
            aid = item.get("api_id")
            if aid in wanted:
                out[aid] = item
        offset += 200
    return out


def _retrieve_for_subtasks(
    *,
    user_goal: str,
    subtasks: List[Dict[str, Any]],
    index_dir: str,
) -> Dict[str, List[Dict[str, Any]]]:
    retrieved_by_subtask: Dict[str, List[Dict[str, Any]]] = {}
    for sub in subtasks:
        sub_id = str(sub.get("id", "unknown"))
        retrieved_by_subtask[sub_id] = collect_candidates(
            user_query=user_goal,
            subtask_goal=sub["description"],
            index_dir=index_dir,
            top_k=CONFIG.rag_top_k,
            debug_dir=None,
        )
    return retrieved_by_subtask


def _rank_for_subtasks(
    *,
    user_goal: str,
    subtasks: List[Dict[str, Any]],
    retrieved_by_subtask: Dict[str, List[Dict[str, Any]]],
    id_to_service: Dict[str, Dict[str, Any]],
    prompt_path: str,
    llm_call,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    ranked_by_subtask: Dict[str, List[Dict[str, Any]]] = {}
    ranked_pool_with_service: List[Dict[str, Any]] = []

    for sub in subtasks:
        sub_id = str(sub.get("id", "unknown"))
        retrieved = retrieved_by_subtask.get(sub_id, [])
        candidates_for_ranker: List[Dict[str, Any]] = []
        for r in retrieved:
            api_id = r.get("api_id")
            if not api_id:
                continue
            candidates_for_ranker.append({
                "api_id": api_id,
                "rag_score": r.get("rag_score", 0.0),
                "compressed": r.get("compressed", {}),
                "service": id_to_service.get(api_id, {}),
            })

        ranked = rank_subtask(
            llm_call=llm_call,
            user_query=user_goal,
            subtask=sub,
            candidates=candidates_for_ranker,
            prompt_path=prompt_path,
            debug_raw_path=None,
        )
        ranked_by_subtask[sub_id] = ranked

        rag_map = {r.get("api_id"): float(r.get("rag_score", 0.0) or 0.0) for r in retrieved}
        for idx, r in enumerate(ranked[: CONFIG.ranker_pool_n], start=1):
            api_id = r.get("api_id")
            if not api_id:
                continue
            ranked_pool_with_service.append({
                "api_id": api_id,
                "rank": idx,
                "subtask_id": sub.get("id"),
                "rag_score": rag_map.get(api_id, 0.0),
                "service": id_to_service.get(api_id, {}),
                "reason": r.get("reason", "") or "",
            })

    return ranked_by_subtask, ranked_pool_with_service


def _select_for_subtasks(
    *,
    subtasks: List[Dict[str, Any]],
    ranked_pool_with_service: List[Dict[str, Any]],
    selector_mode: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    selected_with_service: List[Dict[str, Any]] = []
    selector_traces: Dict[str, Any] = {}

    for sub in subtasks:
        sub_id = str(sub.get("id", "unknown"))
        pool_for_sub = [c for c in ranked_pool_with_service if str(c.get("subtask_id")) == sub_id]
        selected, trace = select_top_apis_for_subtask(
            subtask_id=sub_id,
            ranked_pool=pool_for_sub,
            mode_override=selector_mode,
            top_n=CONFIG.selector_top_n,
            topsis_top_k=CONFIG.topsis_top_k,
            min_qos_candidates=CONFIG.topsis_min_qos_candidates,
        )
        selector_traces[sub_id] = trace

        target_n = len(selected)
        for j, c in enumerate(selected, start=1):
            score = (target_n - j + 1) / float(target_n) if target_n else 0.0
            c2 = dict(c)
            c2["score"] = score
            c2["selected_rank"] = j
            selected_with_service.append(c2)


    return selected_with_service, selector_traces


def _write_mode_outputs(
    *,
    mode_dir: Path,
    mode_suffix: str,
    retrieved_by_subtask: Dict[str, List[Dict[str, Any]]],
    ranked_pool_with_service: List[Dict[str, Any]],
    selected_with_service: List[Dict[str, Any]],
    plan: Dict[str, Any],
    selector_traces: Dict[str, Any],
) -> None:
    (mode_dir / f"1_retriever_{mode_suffix}.json").write_text(json.dumps(retrieved_by_subtask, indent=2, ensure_ascii=False))
    (mode_dir / f"2_ranker_pool_{mode_suffix}.json").write_text(json.dumps(ranked_pool_with_service, indent=2, ensure_ascii=False))
    (mode_dir / f"3_selected_{mode_suffix}.json").write_text(json.dumps(selected_with_service, indent=2, ensure_ascii=False))
    (mode_dir / f"4_planner_{mode_suffix}.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    (mode_dir / f"selector_trace_{mode_suffix}.json").write_text(json.dumps(selector_traces, indent=2, ensure_ascii=False))


def run_autogen_once(user_goal: str, provider: str | None = None, model: str | None = None):
    backend = make_backend(provider=provider, model=model)
    model_tag = backend.name()

    def llm_call(role_name: str, system_msg: str, prompt: str) -> str:
        temp = 0.2 if role_name == "planner" else 0
        return call_with_backoff(
            lambda: backend.chat_json(system_msg, prompt, temperature=temp, force_json=True),
            name=role_name,
        )

    out = _run_dir(model_tag)
    # write meta early so failed runs still record context
    top_meta = {
        "model_tag": model_tag,
        "provider": model_tag.split(":")[0],
        "model": model_tag.split(":")[1] if ":" in model_tag else model_tag,
        "user_goal": user_goal,
        "modes": ["no_qos", "qos_pure_llm", "qos_topsis"],
        "status": "started",
    }
    (out / "meta.json").write_text(json.dumps(top_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # 0) shared decomposition
    subtasks = decompose_goal(
        llm_call=lambda p: llm_call("decomposer", DECOMPOSER_SYS, p),
        user_goal=user_goal,
    )
    (out / "0_decomposer.json").write_text(json.dumps(subtasks, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "run_config.json").write_text(json.dumps({
        "rag_index_dir_qos": CONFIG.rag_index_dir_qos,
        "rag_index_dir_no_qos": CONFIG.rag_index_dir_no_qos,
        "rag_top_k": CONFIG.rag_top_k,
        "ranker_max_candidates": CONFIG.ranker_max_candidates,
        "ranker_pool_n": CONFIG.ranker_pool_n,
        "selector_top_n": CONFIG.selector_top_n,
        "topsis_top_k": CONFIG.topsis_top_k,
        "topsis_min_qos_candidates": CONFIG.topsis_min_qos_candidates,
    }, indent=2), encoding="utf-8")

    # --- no_qos mode ---
    no_qos_dir = _mode_dir(out, "no_qos")
    retrieved_no_qos = _retrieve_for_subtasks(
        user_goal=user_goal,
        subtasks=subtasks,
        index_dir=CONFIG.rag_index_dir_no_qos,
    )
    no_qos_ids = sorted({r["api_id"] for rows in retrieved_no_qos.values() for r in rows if r.get("api_id")})
    no_qos_services = _fetch_services_by_ids(no_qos_ids, with_qos=False)
    _, ranker_pool_no_qos = _rank_for_subtasks(
        user_goal=user_goal,
        subtasks=subtasks,
        retrieved_by_subtask=retrieved_no_qos,
        id_to_service=no_qos_services,
        prompt_path="prompts/ranker_no_qos.md",
        llm_call=lambda p: llm_call("ranker", RANKER_SYS, p),
    )
    selected_no_qos, trace_no_qos = _select_for_subtasks(
        subtasks=subtasks,
        ranked_pool_with_service=ranker_pool_no_qos,
        selector_mode="PURE_LLM",
    )
    plan_no_qos = planner_call(
        llm_call=lambda p: llm_call("planner", PLANNER_SYS, p),
        user_goal=user_goal,
        ranked_top=selected_no_qos,
        subtasks=subtasks,
        prompt_path="prompts/planner_no_qos.md",
    )
    _write_mode_outputs(
        mode_dir=no_qos_dir,
        mode_suffix="no_qos",
        retrieved_by_subtask=retrieved_no_qos,
        ranked_pool_with_service=ranker_pool_no_qos,
        selected_with_service=selected_no_qos,
        plan=plan_no_qos,
        selector_traces=trace_no_qos,
    )

    # --- shared with_qos retrieval + ranking ---
    retrieved_qos = _retrieve_for_subtasks(
        user_goal=user_goal,
        subtasks=subtasks,
        index_dir=CONFIG.rag_index_dir_qos,
    )
    qos_ids = sorted({r["api_id"] for rows in retrieved_qos.values() for r in rows if r.get("api_id")})
    qos_services = _fetch_services_by_ids(qos_ids, with_qos=True)
    _, ranker_pool_qos = _rank_for_subtasks(
        user_goal=user_goal,
        subtasks=subtasks,
        retrieved_by_subtask=retrieved_qos,
        id_to_service=qos_services,
        prompt_path="prompts/ranker.md",
        llm_call=lambda p: llm_call("ranker", RANKER_SYS, p),
    )

    # --- qos pure llm selector ---
    qos_pure_dir = _mode_dir(out, "qos_pure_llm")
    selected_qos_pure, trace_qos_pure = _select_for_subtasks(
        subtasks=subtasks,
        ranked_pool_with_service=ranker_pool_qos,
        selector_mode="PURE_LLM",
    )
    plan_qos_pure = planner_call(
        llm_call=lambda p: llm_call("planner", PLANNER_SYS, p),
        user_goal=user_goal,
        ranked_top=selected_qos_pure,
        subtasks=subtasks,
        prompt_path="prompts/planner.md",
    )
    _write_mode_outputs(
        mode_dir=qos_pure_dir,
        mode_suffix="qos_pure_llm",
        retrieved_by_subtask=retrieved_qos,
        ranked_pool_with_service=ranker_pool_qos,
        selected_with_service=selected_qos_pure,
        plan=plan_qos_pure,
        selector_traces=trace_qos_pure,
    )

    # --- qos topsis selector ---
    qos_topsis_dir = _mode_dir(out, "qos_topsis")
    selected_qos_topsis, trace_qos_topsis = _select_for_subtasks(
        subtasks=subtasks,
        ranked_pool_with_service=ranker_pool_qos,
        selector_mode="TOPSIS",
    )
    plan_qos_topsis = planner_call(
        llm_call=lambda p: llm_call("planner", PLANNER_SYS, p),
        user_goal=user_goal,
        ranked_top=selected_qos_topsis,
        subtasks=subtasks,
        prompt_path="prompts/planner.md",
    )
    _write_mode_outputs(
        mode_dir=qos_topsis_dir,
        mode_suffix="qos_topsis",
        retrieved_by_subtask=retrieved_qos,
        ranked_pool_with_service=ranker_pool_qos,
        selected_with_service=selected_qos_topsis,
        plan=plan_qos_topsis,
        selector_traces=trace_qos_topsis,
    )

    top_meta.update({
        "num_subtasks": len(subtasks),
        "status": "completed",
    })
    (out / "meta.json").write_text(json.dumps(top_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved to {out}")
    throttle(provider)

    return {
        "no_qos": plan_no_qos,
        "qos_pure_llm": plan_qos_pure,
        "qos_topsis": plan_qos_topsis,
    }


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

        run_autogen_once(user_goal=goal, provider=provider, model=None)
