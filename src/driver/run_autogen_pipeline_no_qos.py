# src/driver/run_autogen_pipeline_no_qos.py

"""QoS-unaware MAOF pipeline.

This pipeline mirrors src/driver/run_autogen_pipeline.py but:
- Uses the no-QoS API catalog (fetch_services(with_qos=False)).
- Uses a FAISS index built from the no-QoS catalog.
- Uses prompt variants that do not mention QoS.
- Runs the Selector in PURE_LLM mode (no TOPSIS).

It writes outputs to: results/logs_no_qos/<model_tag>/<run_id>/
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from src.llm.backends import make_backend
from src.tools.fetch_services import fetch_services
from src.agents.decomposer import decompose_goal
from src.agents.retriever import collect_candidates
from src.agents.ranker import rank_subtask
from src.agents.selector import select_top_apis_for_subtask
from src.agents.planner import planner_call
from src.core.retry import call_with_backoff
from src.eval.export_excel import export_run_excel


QUERIES_PATH = Path("data/queries/one_user_query.jsonl")
# QUERIES_PATH = Path("data/queries/user_queries.jsonl")

# -------- role prompts --------
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
    "using only the ranked APIs provided. Follow the prompt strictly and return valid JSON."
)


def load_queries(path: Path = QUERIES_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Queries file not found: {path.resolve()}")
    queries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            queries.append(obj)
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
        if not choice.isdigit():
            print("Please enter a number.")
            continue
        idx = int(choice)
        if 1 <= idx <= len(options):
            provider = options[idx - 1][0]
            print(f"Selected: {options[idx - 1][1]}\n")
            return provider
        print("Invalid choice. Try again.")


def _run_dir(model_tag: str) -> Path:
    run_id = time.strftime("%Y%m%dT%H%M%S")
    d = Path(f"results/logs_no_qos/{model_tag}/{run_id}")
    d.mkdir(parents=True, exist_ok=True)
    return d


# Keep the same policy dict as the main pipeline (copy-paste for independence)
PROVIDER_POLICY = {
    "mistral": {"max_batches": 2, "sleep_after_subtask": 0.5, "sleep_after_query": 0.5},
    "groq": {"max_batches": 2, "sleep_after_subtask": 0.6, "sleep_after_query": 0.8},
    "gemini": {"max_batches": 2, "sleep_after_subtask": 0.4, "sleep_after_query": 0.4},
    "azure_foundry": {"max_batches": 3, "sleep_after_subtask": 0.2, "sleep_after_query": 0.2},
    "azure": {"max_batches": 3, "sleep_after_subtask": 0.2, "sleep_after_query": 0.2},
    "_default": {"max_batches": 2, "sleep_after_subtask": 0.4, "sleep_after_query": 0.4},
}


def get_policy(provider: str | None):
    provider = provider or "_default"
    return PROVIDER_POLICY.get(provider, PROVIDER_POLICY["_default"])


def throttle(provider: str | None, phase: str):
    pol = get_policy(provider)
    if phase == "after_subtask":
        t = pol.get("sleep_after_subtask", 0.0) or 0.0
    elif phase == "after_query":
        t = pol.get("sleep_after_query", 0.0) or 0.0
    else:
        t = 0.0
    if t > 0:
        time.sleep(t)


def run_autogen_once(user_goal: str, provider: str | None = None, model: str | None = None):
    """Run one query through the QoS-unaware pipeline."""
    backend = make_backend(provider=provider, model=model)
    model_tag = backend.name()

    policy = get_policy(provider)
    max_batches = 1

    def llm_call(role_name: str, system_msg: str, prompt: str) -> str:
        temp = 0.2 if role_name == "planner" else 0
        return call_with_backoff(
            lambda: backend.chat_json(system_msg, prompt, temperature=temp, force_json=True),
            name=role_name,
        )

    out = _run_dir(model_tag)

    # 1) DECOMPOSER
    subtasks = decompose_goal(
        llm_call=lambda p: llm_call("decomposer", DECOMPOSER_SYS, p),
        user_goal=user_goal,
    )

    # 2) RAG RETRIEVER (no-QoS index)
    # Use explicit env vars to avoid impacting the QoS-aware pipeline.
    index_dir = os.getenv("MAOF_RAG_INDEX_DIR_NO_QOS", "data/index/maof_v1/no_qos")
    top_k = int(os.getenv("MAOF_RAG_TOPK", "60"))

    retrieved_by_subtask: dict[str, list[dict[str, Any]]] = {}
    pick_ids_set: set[str] = set()

    for sub in subtasks:
        sub_goal = sub["description"]
        sub_id = str(sub.get("id", "unknown"))

        debug_dir = out / f"debug_retriever_subtask_{sub_id}"
        debug_dir.mkdir(exist_ok=True)

        retrieved = collect_candidates(
            user_query=user_goal,
            subtask_goal=sub_goal,
            index_dir=index_dir,
            top_k=top_k,
            debug_dir=str(debug_dir),
        )

        retrieved_by_subtask[sub_id] = retrieved
        for r in retrieved:
            if r.get("api_id"):
                pick_ids_set.add(r["api_id"])

    pick_ids = sorted(list(pick_ids_set))

    # 3) GATHER CATALOG ITEMS (NO-QOS)
    cat_items = []
    offset = 0
    while True:
        batch = fetch_services(category=None, offset=offset, limit=200, with_qos=False)
        if not batch:
            break
        cat_items.extend([b for b in batch if b["api_id"] in pick_ids])
        offset += 200

    id_to_service = {s["api_id"]: s for s in cat_items}

    # 4) RANKER (no-QoS prompt)
    ranked_by_subtask: dict[str, list[dict[str, Any]]] = {}

    for sub in subtasks:
        sub_id = str(sub.get("id", "unknown"))
        retrieved = retrieved_by_subtask.get(sub_id, [])

        candidates_for_ranker: List[Dict[str, Any]] = []
        for r in retrieved:
            api_id = r.get("api_id")
            if not api_id:
                continue
            service = id_to_service.get(api_id, {})
            candidates_for_ranker.append(
                {
                    "api_id": api_id,
                    "rag_score": r.get("rag_score", 0.0),
                    "compressed": r.get("compressed", {}),
                    "service": service,
                }
            )

        debug_ranker_dir = out / f"debug_ranker_subtask_{sub_id}"
        debug_ranker_dir.mkdir(exist_ok=True)
        (debug_ranker_dir / "ranker_input.json").write_text(
            json.dumps({"user_query": user_goal, "subtask": sub, "candidates": candidates_for_ranker}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        ranked = rank_subtask(
            llm_call=lambda p: llm_call("ranker", RANKER_SYS, p),
            user_query=user_goal,
            subtask=sub,
            candidates=candidates_for_ranker,
            prompt_path="prompts/ranker_no_qos.md",
            debug_raw_path=str(debug_ranker_dir / "ranker_raw.txt"),
        )
        ranked_by_subtask[sub_id] = ranked
        (debug_ranker_dir / "ranked.json").write_text(json.dumps(ranked, indent=2, ensure_ascii=False), encoding="utf-8")

    # 4.5) SELECTOR: PURE_LLM only (top 8)
    # Force PURE_LLM regardless of environment for this pipeline.
    os.environ["MAOF_SELECTOR_MODE"] = "PURE_LLM"

    ranked_pool_with_service: List[Dict[str, Any]] = []
    selected_with_service: List[Dict[str, Any]] = []
    selector_traces: Dict[str, Any] = {}

    selected_top_n = int(os.getenv("MAOF_SELECTED_TOP_N", "8"))
    ranker_pool_n = int(os.getenv("MAOF_RANKER_POOL_N", "20"))

    for sub in subtasks:
        sub_id = str(sub.get("id", "unknown"))
        rag_map = {r.get("api_id"): float(r.get("rag_score", 0.0) or 0.0) for r in retrieved_by_subtask.get(sub_id, [])}

        ranked_list = (ranked_by_subtask.get(sub_id, []) or [])[:ranker_pool_n]
        pool_for_sub: List[Dict[str, Any]] = []

        for idx, r in enumerate(ranked_list, start=1):
            api_id = r.get("api_id")
            if not api_id:
                continue
            pool_for_sub.append(
                {
                    "api_id": api_id,
                    "rank": idx,
                    "subtask_id": sub.get("id"),
                    "rag_score": rag_map.get(api_id, 0.0),
                    "service": id_to_service.get(api_id, {}),
                    "reason": r.get("reason", "") or "",
                }
            )

        ranked_pool_with_service.extend(pool_for_sub)

        selected, trace = select_top_apis_for_subtask(
            subtask_id=sub_id,
            ranked_pool=pool_for_sub,
            top_n=selected_top_n,
            topsis_top_k=0,
            min_qos_candidates=0,
        )
        selector_traces[sub_id] = trace

        target_n = len(selected)
        for j, c in enumerate(selected, start=1):
            score = (target_n - j + 1) / float(target_n) if target_n else 0.0
            c2 = dict(c)
            c2["score"] = score
            c2["selected_rank"] = j
            selected_with_service.append(c2)

        debug_sel_dir = out / f"debug_selector_subtask_{sub_id}"
        debug_sel_dir.mkdir(exist_ok=True)
        (debug_sel_dir / "selector_input.json").write_text(
            json.dumps({"subtask": sub, "ranked_pool": pool_for_sub}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (debug_sel_dir / "selector_trace.json").write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
        (debug_sel_dir / "selected.json").write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")

    # 5) PLANNER (no-QoS prompt)
    plan = planner_call(
        llm_call=lambda p: llm_call("planner", PLANNER_SYS, p),
        user_goal=user_goal,
        ranked_top=selected_with_service,
        subtasks=subtasks,
        prompt_path="prompts/planner_no_qos.md",
    )

    # 6) LOGS
    (out / "0_decomposer.json").write_text(json.dumps(subtasks, indent=2))
    (out / "1_retriever.json").write_text(json.dumps(retrieved_by_subtask, indent=2, ensure_ascii=False))
    (out / "2_ranker_raw.json").write_text(json.dumps(ranked_by_subtask, indent=2, ensure_ascii=False))
    (out / "3_ranked_pool_with_service.json").write_text(json.dumps(ranked_pool_with_service, indent=2, ensure_ascii=False))
    (out / "4_selected_with_service.json").write_text(json.dumps(selected_with_service, indent=2, ensure_ascii=False))
    (out / "5_planner.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    (out / "selector_traces.json").write_text(json.dumps(selector_traces, indent=2, ensure_ascii=False))
    (out / "meta.json").write_text(
        json.dumps(
            {
                "model_tag": model_tag,
                "provider": model_tag.split(":")[0],
                "model": model_tag.split(":")[1] if ":" in model_tag else model_tag,
                "with_qos": False,
                "pipeline": "no_qos",
                "rag_index_dir": index_dir,
                "rag_topk": top_k,
                "ranker_candidates_used": int(os.getenv("MAOF_RANKER_MAX_CANDIDATES", "25")),
                "ranker_pool_n": int(os.getenv("MAOF_RANKER_POOL_N", "20")),
                "selected_top_n": selected_top_n,
                "selector_mode": "PURE_LLM",
                "user_goal": user_goal,
                "num_subtasks": len(subtasks),
                "max_batches": max_batches,
                "policy": policy,
                "prompts": {
                    "ranker": "prompts/ranker_no_qos.md",
                    "planner": "prompts/planner_no_qos.md",
                },
            },
            indent=2,
        )
    )

    # 7) EXCEL REPORT
    try:
        export_run_excel(run_dir=out)
    except Exception as e:
        (out / "excel_export_error.txt").write_text(str(e), encoding="utf-8")

    print(f"Saved to {out}")
    throttle(provider, "after_query")
    return ranked_by_subtask, plan


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
        )
