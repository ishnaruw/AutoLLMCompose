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
from src.core.retry import call_with_backoff
from src.eval.topsis_eval import evaluate_topsis_mode1


QUERIES_PATH = Path("data/queries/one_user_query.jsonl")
# QUERIES_PATH = Path("data/queries/user_queries.jsonl")

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
        ("lmstudio", "LM Studio (local, meta-llama-3.1-8b-instruct)")
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
    d = Path(f"results/logs/{model_tag}/{run_id}")
    d.mkdir(parents=True, exist_ok=True)
    return d


# One place to tune pacing behavior per provider.
# This applies to ALL providers, and you can tune each independently.
PROVIDER_POLICY = {
    "mistral":       {"max_batches": 2, "sleep_after_subtask": 0.5, "sleep_after_query": 0.5},
    "groq":          {"max_batches": 2, "sleep_after_subtask": 0.6, "sleep_after_query": 0.8},
    "gemini":        {"max_batches": 2, "sleep_after_subtask": 0.4, "sleep_after_query": 0.4},
    "azure_foundry": {"max_batches": 3, "sleep_after_subtask": 0.2, "sleep_after_query": 0.2},
    "azure":         {"max_batches": 3, "sleep_after_subtask": 0.2, "sleep_after_query": 0.2},
    # fallback default
    "_default":      {"max_batches": 2, "sleep_after_subtask": 0.4, "sleep_after_query": 0.4},
}


def get_policy(provider: str | None):
    provider = provider or "_default"
    return PROVIDER_POLICY.get(provider, PROVIDER_POLICY["_default"])


def throttle(provider: str | None, phase: str):
    """
    phase:
      - "after_subtask": pause after each subtask retrieval
      - "after_query":   pause after finishing a full query
    """
    pol = get_policy(provider)
    if phase == "after_subtask":
        t = pol.get("sleep_after_subtask", 0.0) or 0.0
    elif phase == "after_query":
        t = pol.get("sleep_after_query", 0.0) or 0.0
    else:
        t = 0.0
    if t > 0:
        time.sleep(t)


def run_autogen_once(user_goal: str, with_qos: bool, provider: str | None = None, model: str | None = None):
    backend = make_backend(provider=provider, model=model)
    model_tag = backend.name()

    policy = get_policy(provider)
    max_batches = 1
    # max_batches = int(policy.get("max_batches", 2))

    # Single call helper: always backoff, for all providers.
    def llm_call(role_name: str, system_msg: str, prompt: str) -> str:
        return call_with_backoff(
            lambda: backend.chat_json(system_msg, prompt, temperature=0, force_json=True),
            name=role_name
        )
    
    out = _run_dir(model_tag)

    # 1) DECOMPOSER
    subtasks = decompose_goal(
        llm_call=lambda p: llm_call("decomposer", DECOMPOSER_SYS, p),
        user_goal=user_goal
    )

    # 2) RETRIEVER per subtask
    global_keep = {}

    for sub in subtasks:
        sub_goal = sub["description"]

        sub_id = sub.get("id", "unknown")

        debug_dir = out / f"debug_retriever_subtask_{sub_id}"
        debug_dir.mkdir(exist_ok=True)

        picks_sub = collect_candidates(
            llm_call=lambda p: llm_call("retriever", RETRIEVER_SYS, p),
            user_goal=sub_goal,
            fetch_fn=fetch_services,
            category=None,
            with_qos=with_qos,
            max_batches=max_batches,
            debug_dir=str(debug_dir),
        )

        for p in picks_sub:
            api_id = p.get("api_id")
            if not api_id:
                continue
            reason = (p.get("reason") or "").strip()
            existing = global_keep.get(api_id)
            if existing:
                merged = f"{existing} | Also relevant for subtask {sub.get('id')}: {reason}" if reason else existing
                global_keep[api_id] = merged
            else:
                prefix = f"[Subtask {sub.get('id')}] "
                global_keep[api_id] = prefix + reason if reason else prefix + "Selected as relevant."

        # ✅ pacing between subtasks (all providers, policy driven)
        throttle(provider, "after_subtask")

    picks = [{"api_id": k, "reason": v} for k, v in global_keep.items()]
    pick_ids = [p["api_id"] for p in picks]

    # 3) GATHER CATALOG ITEMS FOR RANKER
    cat_items = []
    offset = 0
    while True:
        batch = fetch_services(category=None, offset=offset, limit=200, with_qos=with_qos)
        if not batch:
            break
        cat_items.extend([b for b in batch if b["api_id"] in pick_ids])
        offset += 200

    # 4) RANKER
    ranked = ranker_call(
        llm_call=lambda p: llm_call("ranker", RANKER_SYS, p),
        services=cat_items,
        user_goal=user_goal,
        subtasks=subtasks,
    )

    # 4.5) Build ranked_top for planner: include score and full catalog service
    id_to_service = {s["api_id"]: s for s in cat_items}

    ranked_top = []
    if ranked:
        for r in ranked:
            api_id = r["api_id"]
            ranked_top.append({
                "api_id": api_id,
                "score": r.get("score", 0.0),
                "service": id_to_service.get(api_id, {}),
            })
    else:
        for s in cat_items:
            ranked_top.append({"api_id": s["api_id"], "score": 0.0, "service": s})

    # 5) PLANNER
    plan = planner_call(
        llm_call=lambda p: llm_call("planner", PLANNER_SYS, p),
        user_goal=user_goal,
        ranked_top=ranked_top,
        subtasks=subtasks,
    )

    # 5.5) TOPSIS EVALUATION (Mode 1: ranker candidate set per subtask)
    topsis_eval = evaluate_topsis_mode1(
        picks=picks,
        ranked=ranked,
        ranked_top=ranked_top,
        plan=plan,
        top_k_candidates=12,        # matches your ranker pool size range
        min_qos_candidates=5,       # skip TOPSIS if too few have valid QoS
        weights=None,               # TODO later: map from user query
    )


    # 6) LOGS
    (out / "0_decomposer.json").write_text(json.dumps(subtasks, indent=2))
    (out / "1_retriever.json").write_text(json.dumps(picks, indent=2))
    (out / "2_ranker_raw.json").write_text(json.dumps(ranked, indent=2))
    (out / "3_ranked_with_service.json").write_text(json.dumps(ranked_top, indent=2))
    (out / "4_planner.json").write_text(json.dumps(plan, indent=2))
    (out / "5_topsis_eval.json").write_text(json.dumps(topsis_eval, indent=2))
    (out / "meta.json").write_text(json.dumps({
        "model_tag": model_tag,
        "provider": model_tag.split(":")[0],
        "model": model_tag.split(":")[1] if ":" in model_tag else model_tag,
        "with_qos": with_qos,
        "user_goal": user_goal,
        "num_subtasks": len(subtasks),
        "max_batches": max_batches,
        "policy": get_policy(provider),
        "topsis_enabled": True,
        "topsis_mode": "mode1_ranker_candidates",
        "topsis_top_k_candidates": 12,
        "topsis_min_qos_candidates": 5,
        "topsis_outlier_mitigation": "winsorize_p5_p95",
        "topsis_normalization": "vector (inside TOPSIS; implemented by method)",
        "topsis_criteria": ["rt_ms(cost)", "tp_rps(benefit)", "availability(benefit)"],
    }, indent=2))

    print(f"Saved to {out}")

    # ✅ pacing between queries (all providers, policy driven)
    throttle(provider, "after_query")

    return ranked, plan


if __name__ == "__main__":
    provider = choose_provider_interactive()
    with_qos = True

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
            with_qos=with_qos,
            provider=provider,
            model=None,
        )
