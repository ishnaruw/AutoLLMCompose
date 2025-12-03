# src/driver/run_autogen_pipeline.py

import json
import time
from pathlib import Path

from src.llm.backends import make_backend
from src.tools.fetch_services import fetch_services
from src.agents.decomposer import decompose_goal
from src.agents.retriever import collect_candidates
from src.agents.ranker import ranker_call
from src.agents.planner import planner_call

# -------- role prompts --------
DECOMPOSER_SYS = (
    "You are a task decomposition agent. "
    "You break a user goal into ordered subtasks. "
    "Return strict JSON as instructed by the prompt."
)

RETRIEVER_SYS = (
    "You are a retrieval agent that selects relevant APIs from a JSON catalog "
    "based on a subtask goal. Return strict JSON as instructed by the prompt; "
    "never invent services."
)

RANKER_SYS = (
    "You are a ranking agent. Given a user goal, its decomposed subtasks, and "
    "a list of candidate APIs from a catalog, you assign a numeric score to "
    "each API and order them from best to worst. Follow the prompt strictly "
    "and return valid JSON."
)

PLANNER_SYS = (
    "You are an orchestration planner that composes a logical API workflow "
    "using only the ranked APIs provided. You have access to an agent score "
    "and the full catalog entry for each API (including any QoS-related "
    "information if present). Follow the prompt strictly and return valid JSON."
)


def _run_dir(model_tag: str) -> Path:
    run_id = time.strftime("%Y%m%dT%H%M%S")
    d = Path(f"results/logs/{model_tag}/{run_id}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_autogen_once(user_goal: str, with_qos: bool):
    """
    End to end pipeline:

      1) Decompose user_goal into ordered subtasks.
      2) For each subtask, retrieve relevant APIs from the entire catalog (all categories).
      3) Merge all retrieved APIs into a global candidate set.
      4) Rank candidates using an LLM-based score derived from the user goal,
         subtasks, and catalog information.
      5) Plan an orchestration using the ranked APIs with access to their full
         catalog entries (including any QoS-related information).

    Note: The caller no longer provides a category. The LLM selects APIs by
    inspecting the mixed-category catalog directly.
    """
    backend = make_backend()  # Azure or Mistral, chosen by LLM_PROVIDER
    model_tag = backend.name()

    # 0) LLM call wrappers for each role
    decomposer_llm = lambda p: backend.chat_json(DECOMPOSER_SYS, p, temperature=0, force_json=True)
    retriever_llm = lambda p: backend.chat_json(RETRIEVER_SYS, p, temperature=0, force_json=True)
    ranker_llm = lambda p: backend.chat_json(RANKER_SYS, p, temperature=0, force_json=True)
    planner_llm = lambda p: backend.chat_json(PLANNER_SYS, p, temperature=0, force_json=True)

    # 1) DECOMPOSER: user_goal -> subtasks (no categories)
    subtasks = decompose_goal(decomposer_llm, user_goal)

    # 2) RETRIEVER per subtask over the whole catalog (category=None)
    global_keep = {}  # api_id -> reason (merged across subtasks)

    for sub in subtasks:
        sub_goal = sub["description"]

        picks_sub = collect_candidates(
            llm_call=retriever_llm,
            user_goal=sub_goal,
            fetch_fn=fetch_services,
            category=None,          # no category filter, all catalog entries
            with_qos=with_qos,
            max_batches=5,
        )

        for p in picks_sub:
            api_id = p.get("api_id")
            if not api_id:
                continue
            reason = (p.get("reason") or "").strip()
            existing = global_keep.get(api_id)
            if existing:
                merged = (
                    f"{existing} | Also relevant for subtask {sub['id']}: {reason}"
                    if reason else existing
                )
                global_keep[api_id] = merged
            else:
                prefix = f"[Subtask {sub['id']}] "
                global_keep[api_id] = prefix + reason if reason else prefix + "Selected as relevant."

    # Flatten to the original format expected by downstream stages
    picks = [{"api_id": k, "reason": v} for k, v in global_keep.items()]
    pick_ids = [p["api_id"] for p in picks]

    # 3) GATHER CATALOG ITEMS FOR RANKER (across all categories, no filter)
    cat_items = []
    offset = 0
    while True:
        batch = fetch_services(category=None, offset=offset, limit=200, with_qos=with_qos)
        if not batch:
            break
        cat_items.extend([b for b in batch if b["api_id"] in pick_ids])
        offset += 200

    # 4) RANKER (LLM based) over full catalog entries (including qos if present)
    ranked = ranker_call(
        llm_call=ranker_llm,
        services=cat_items,
        user_goal=user_goal,
        subtasks=subtasks,
    )

    # 4.5) Build ranked_top for planner: include score and full catalog service
    # Index catalog items by api_id for quick lookup
    id_to_service = {s["api_id"]: s for s in cat_items}

    ranked_top = []
    if ranked:
        # Pass all ranked candidates to the planner
        for r in ranked:
            api_id = r["api_id"]
            service = id_to_service.get(api_id, {})
            ranked_top.append({
                "api_id": api_id,
                "score": r.get("score", 0.0),
                "service": service,
            })
    else:
        # Fallback: take catalog items when ranker returns nothing
        for s in cat_items:
            ranked_top.append({
                "api_id": s["api_id"],
                "score": 0.0,
                "service": s,
            })

    # 5) PLANNER
    plan = planner_call(
        llm_call=planner_llm,
        user_goal=user_goal,
        ranked_top=ranked_top,
        subtasks=subtasks,
    )

    # 6) LOGS
    out = _run_dir(model_tag)
    (out / "decomposer_autogen.json").write_text(json.dumps(subtasks, indent=2))
    (out / "retriever_autogen.json").write_text(json.dumps(picks, indent=2))
    (out / "ranker_autogen.json").write_text(json.dumps(ranked, indent=2))
    (out / "ranked_for_planner.json").write_text(json.dumps(ranked_top, indent=2))
    (out / "planner_autogen.json").write_text(json.dumps(plan, indent=2))
    (out / "meta.json").write_text(json.dumps({
        "model_tag": model_tag,
        "provider": model_tag.split(":")[0],
        "model": model_tag.split(":")[1] if ":" in model_tag else model_tag,
        "with_qos": with_qos,
        "user_goal": user_goal,
        "num_subtasks": len(subtasks),
    }, indent=2))


    print(f"Saved to {out}")
    return ranked, plan


if __name__ == "__main__":
    # Example single run with an intentionally cross-category goal
    run_autogen_once(
        user_goal=(
            "Build a service that uses a user’s location to discover restaurants, "
            "show reviews, and offer reservations via email."
        ),
        with_qos=True,
    )
