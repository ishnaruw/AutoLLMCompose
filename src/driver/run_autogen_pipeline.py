# src/driver/run_autogen_pipeline.py

import json
import time
from pathlib import Path

from src.llm.backends import make_backend
from src.tools.fetch_services import fetch_services
from src.agents.decomposer import decompose_goal
from src.agents.retriever import collect_candidates
from src.agents.ranker_topsis import make_qos_table, ranker_call
from src.agents.planner import planner_call
from src.core.topsis_verify import topsis_verify

# -------- role prompts --------
DECOMPOSER_SYS = (
    "You are a task decomposition agent. "
    "You break a user goal into ordered subtasks and assign each a coarse API category. "
    "Return strict JSON as instructed by the prompt."
)

RETRIEVER_SYS = (
    "You are a retrieval agent that selects relevant APIs from a JSON catalog based on a subtask goal. "
    "Return strict JSON as instructed by the prompt; never invent services."
)
RANKER_SYS = (
    "You are a QoS evaluator. Apply TOPSIS to rt_ms (cost), tp_rps (benefit), and availability (benefit). "
    "Follow the prompt strictly and return valid JSON."
)
PLANNER_SYS = (
    "You are an orchestration planner that composes a logical API workflow using only the ranked APIs. "
    "Follow the prompt strictly and return valid JSON."
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
      4) Rank candidates by QoS using TOPSIS (LLM + deterministic verifier).
      5) Plan an orchestration using the ranked APIs.

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

        # Reuse existing collect_candidates, but with category=None
        picks_sub = collect_candidates(
            llm_call=retriever_llm,
            user_goal=sub_goal,
            fetch_fn=fetch_services,
            category=None,          # <- no category filter, all catalog entries
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

    # Build QoS table (or placeholders when with_qos=False)
    qos_rows = (
        make_qos_table(cat_items)
        if with_qos
        else [
            {
                "api_id": s["api_id"],
                "rt_ms": None,
                "tp_rps": None,
                "availability": None,
            }
            for s in cat_items
        ]
    )

    # 4) RANKER (LLM TOPSIS) + deterministic verifier
    ranked = ranker_call(
        llm_call=ranker_llm,
        qos_rows=qos_rows,
    )
    verified = topsis_verify(qos_rows)

    # 5) PLANNER
    ranked_top = ranked[:6] if ranked else [
        {"api_id": s["api_id"], "C": 0.0} for s in cat_items[:6]
    ]
    plan = planner_call(
        llm_call=planner_llm,
        user_goal=user_goal,
        ranked_top=ranked_top,
    )

    # 6) LOGS
    out = _run_dir(model_tag)
    (out / "decomposer_autogen.json").write_text(json.dumps(subtasks, indent=2))
    (out / "retriever_autogen.json").write_text(json.dumps(picks, indent=2))
    (out / "ranker_autogen.json").write_text(json.dumps(ranked, indent=2))
    (out / "planner_autogen.json").write_text(json.dumps(plan, indent=2))
    (out / "topsis_verify.json").write_text(json.dumps(verified, indent=2))
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
        user_goal=("Build a service that uses a user’s location to discover restaurants, show reviews, and offer reservations via email. "
        ),
        with_qos=True,
    )
