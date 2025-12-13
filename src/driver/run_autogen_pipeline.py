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

    # 6) LOGS (step-numbered for easier navigation)
    out = _run_dir(model_tag)

    (out / "0_decomposer.json").write_text(json.dumps(subtasks, indent=2))
    (out / "1_retriever.json").write_text(json.dumps(picks, indent=2))
    (out / "2_ranker_raw.json").write_text(json.dumps(ranked, indent=2))
    (out / "3_ranked_with_service.json").write_text(json.dumps(ranked_top, indent=2))
    (out / "4_planner.json").write_text(json.dumps(plan, indent=2))
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
    # List of user queries to run through the pipeline
    USER_QUERIES = [
        # 1. Weather Alerts via SMS
        "I want to build a service that fetches weather data for a user’s location and sends real-time weather alerts via SMS.",

        # 2. Nearby Restaurant Finder + Reservation
        "Build a service that uses a user’s location to discover restaurants, show reviews, and offer reservations via email.",

        # 3. Cybersecurity Alert System
        "Create a dashboard that checks domains for threats using threat intelligence APIs and sends alerts to admins via SMS.",

        # 4. Travel Planner with Tourist Spots
        "Design a travel app that books flights, suggests hotels, shows top-rated local attractions, and displays weather info.",

        # 5. Stock Analyzer + Buy/Sell Alerts
        "Build a tool that tracks stock prices, analyzes trends using ML, and emails buy/sell signals to users.",

        # 6. Event Finder with Calendar Sync
        "Build a service that finds local events based on interests and syncs selected ones to the user’s Google Calendar.",

        # 7. News Summarizer with SMS Briefing
        "Create a system that fetches top news, summarizes headlines using LLM, and sends SMS digests every morning.",

        # 8. Price Tracker for eCommerce Products
        "Build a bot that tracks product prices across multiple eCommerce platforms and sends alerts when prices drop.",

        # 9. Online Course Recommendation Assistant
        "Create an education bot that suggests courses, checks ratings, and reminds users about enrollment via email.",

        # 10. Gym Finder with Booking
        "Build a service that lists gyms near a user’s location, compares pricing plans, and enables mobile booking.",

        # 11. Loan Simulator with Email Report
        "Create a tool that pulls interest rates, simulates repayment plans, and sends loan summaries to users.",

        # 12. Content Safety Filter for Web Links
        "Design a service that checks whether a given URL is malicious or contains adult content before it opens in-browser.",

        # 13. Retail Promotions via Geo-Fencing
        "Build a mobile app that detects when a user enters a shopping area and sends active retail coupons via push or SMS.",

        # 14. Social Media Topic Tracker
        "Design a system that tracks trending hashtags, top influencers, and returns a daily engagement summary.",

        # 15. Delivery Status Tracker with Email Updates
        "Create a logistics tool that fetches parcel tracking data, predicts ETA, and emails the customer when status changes.",
    ]

    # Run all queries one by one through the pipeline
    with_qos = True  # or False, depending on the experiment

    for i, query in enumerate(USER_QUERIES, start=1):
        print("\n" + "=" * 80)
        print(f"Running query {i}/{len(USER_QUERIES)}")
        print(f"User goal: {query}")
        print("=" * 80)

        run_autogen_once(
            user_goal=query,
            with_qos=with_qos,
        )

