from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.agents.decomposer import decompose_goal
from src.agents.planner import planner_call
from src.agents.ranker import rank_subtask
from src.agents.qos_scorer_llm import score_qos_llm
from src.agents.retriever import collect_candidates
from src.config import CONFIG
from src.core.retry import call_with_backoff
from src.eval.api_relevancy_eval import evaluate_query, evaluate_retrieval_relevancy
from src.eval.audit_api_duplicates import collect_duplicate_audit_for_run
from src.eval.audit_api_hallucinations import collect_hallucination_audit_for_run
from src.eval.mode_anomaly_report import write_mode_anomaly_excel
from src.eval.topsis_eval import _extract_qos, _run_topsis_pydecision
from src.llm.autogen_runner import run_autogen_agent
from src.llm.backends import make_backend
from src.tools.fetch_services import fetch_services

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
    "You are a QoS scoring agent. Given only api ids and QoS metrics, produce a relative QoS-only ranking and score. "
    "Return strict JSON only."
)

PLANNER_SYS = (
    "You are an orchestration planner that composes a logical API workflow "
    "using only the selected APIs provided. Preserve the ordered subtasks and return valid JSON."
)

MODE_ORDER = ["no_qos", "qos_pure_llm", "qos_topsis", "qos_hybrid"]

PROVIDER_POLICY = {
    "mistral": {"sleep_after_query": 0.5},
    "groq": {"sleep_after_query": 0.8},
    "together": {"sleep_after_query": 0.5},
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


def choose_provider_interactive() -> str:
    options = [
        ("mistral", "Mistral"),
        ("groq", "Groq"),
        ("together", "Together AI"),
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
    run_name = f"{_safe_name(query_id)}_{run_id}" if query_id else run_id
    base_dir = Path("results/logs")
    if run_tag:
        base_dir = base_dir / _safe_name(run_tag)
    out = base_dir / model_tag.replace(":", "_") / run_name
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
    def llm_call(role_name: str, system_msg: str, prompt: str) -> str:
        temp = 0.2 if role_name == "planner" else 0.0
        if CONFIG.use_autogen_agents:
            return call_with_backoff(
                lambda: run_autogen_agent(
                    backend=backend,
                    role_name=role_name,
                    system_message=system_msg,
                    prompt=prompt,
                    temperature=temp,
                    force_json=True,
                ),
                name=role_name,
            )
        return call_with_backoff(lambda: backend.chat_json(system_msg, prompt, temperature=temp, force_json=True), name=role_name)
    return llm_call


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _to_run_relative(path: Path | None, run_dir: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(run_dir.resolve()))
    except Exception:
        return str(path)


def _build_shared_retrieval(user_goal: str, subtasks: List[Dict[str, Any]], out_dir: Path) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    retrieved_by_subtask: Dict[str, List[Dict[str, Any]]] = {}
    pick_ids: List[str] = []
    seen = set()
    for sub in subtasks:
        sub_id = str(sub.get("id", "unknown"))
        retrieved = collect_candidates(subtask_goal=str(sub.get("description", "")), index_dir=str(CONFIG.shared_index_dir), top_k=CONFIG.rag_top_k)
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


def _deterministic_hybrid_ranking(retrieved: List[Dict[str, Any]], topsis_meta: Dict[str, Dict[str, Any]], sub_id: str, relevancy_map: Dict[tuple[str, str], Dict[str, Any]]) -> List[Dict[str, Any]]:
    relevant_enriched = []
    non_relevant_enriched = []

    for r in retrieved:
        api_id = str(r.get("api_id", ""))
        m = topsis_meta.get(api_id, {})
        topsis_rank = int(m.get("topsis_rank") or 10**9)
        topsis_score = m.get("topsis_score")
        rel_entry = relevancy_map.get((sub_id, api_id), {})
        is_relevant = int(rel_entry.get("API Relevancy (0/1)", rel_entry.get("relevant", 0)) or 0) == 1

        if is_relevant:
            relevant_enriched.append((topsis_rank, api_id, topsis_score))
        else:
            non_relevant_enriched.append((topsis_rank, api_id, topsis_score))

    relevant_enriched.sort(key=lambda x: x[0])
    non_relevant_enriched.sort(key=lambda x: x[0])
    combined = relevant_enriched + non_relevant_enriched

    ranked: List[Dict[str, Any]] = []
    for topsis_rank, api_id, _ in combined:
        rel_entry = relevancy_map.get((sub_id, api_id), {})
        is_relevant = int(rel_entry.get("API Relevancy (0/1)", rel_entry.get("relevant", 0)) or 0) == 1
        rel_label = "Relevant" if is_relevant else "Non-relevant"
        ranked.append(
            {
                "api_id": api_id,
                "reason": f"Deterministic hybrid: {rel_label} API ordered by TOPSIS rank {topsis_rank}.",
            }
        )
    return ranked


def _write_ranked(mode_dir: Path, sub_id: str, ranked: List[Dict[str, Any]], retrieved: List[Dict[str, Any]], id_to_service: Dict[str, Dict[str, Any]], extras: Dict[str, Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
    rag_map = {str(r.get("api_id")): r for r in retrieved}
    full = []
    for idx, item in enumerate(ranked, start=1):
        api_id = str(item.get("api_id", ""))
        base = rag_map.get(api_id, {})
        row = {
            "api_id": api_id,
            "mode_rank": idx,
            "retrieved_rank": base.get("retrieved_rank"),
            "rag_score": base.get("rag_score"),
            "reason": item.get("reason", ""),
            "service": id_to_service.get(api_id, {}),
        }
        if extras and api_id in extras:
            row.update(extras[api_id])
        full.append(row)
    _write_json(mode_dir / f"2_ranked_s{sub_id}.json", full)
    return full


def _load_relevancy_map(rows_path: Path) -> Dict[tuple[str, str], Dict[str, Any]]:
    data = json.loads(rows_path.read_text(encoding="utf-8"))
    out = {}
    for row in data:
        sub_id = str(row.get("Sub Task", row.get("subtask_id", "")))
        api_id = str(row.get("Selected_API", row.get("api_id", "")))
        if not sub_id or not api_id:
            continue
        relevant = row.get("API Relevancy (0/1)", row.get("relevant", 0))
        comment = row.get("Comments", row.get("comment", ""))
        out[(sub_id, api_id)] = {
            "API Relevancy (0/1)": relevant,
            "Comments": comment,
            "relevant": relevant,
            "comment": comment,
        }
    return out


def _load_planner_top_n(rows_path: Path) -> Dict[str, int]:
    data = json.loads(rows_path.read_text(encoding="utf-8"))
    relevant_api_ids_by_subtask: Dict[str, set[str]] = {}
    for row in data:
        sub_id = str(row.get("Sub Task", row.get("subtask_id", "")))
        api_id = str(row.get("Selected_API", row.get("api_id", "")))
        relevant = row.get("API Relevancy (0/1)", row.get("relevant", 0))
        if not sub_id or not api_id or int(relevant or 0) != 1:
            continue
        relevant_api_ids_by_subtask.setdefault(sub_id, set()).add(api_id)
    return {sub_id: len(api_ids) for sub_id, api_ids in relevant_api_ids_by_subtask.items()}


def _deterministic_select_and_plan(mode_name: str, subtasks: List[Dict[str, Any]], ranked_full: Dict[str, List[Dict[str, Any]]], planner_top_n: Dict[str, int], llm_call, user_goal: str, out_dir: Path, planner_prompt_path: str) -> Dict[str, Any]:
    mode_dir = out_dir / mode_name
    selected_all = []
    for sub in subtasks:
        sub_id = str(sub.get("id", "unknown"))
        ranked_rows = ranked_full.get(sub_id, [])
        dynamic_top_n = int(planner_top_n.get(sub_id, 0) or 0)
        fallback_used = dynamic_top_n <= 0
        selected_limit = dynamic_top_n if dynamic_top_n > 0 else CONFIG.selector_top_n
        selected_limit = min(selected_limit, len(ranked_rows))
        selected_rows = ranked_rows[:selected_limit]
        top_n_source = "final_api_relevancy" if dynamic_top_n > 0 else "selector_top_n_fallback"
        selected = []
        for idx, r in enumerate(selected_rows, start=1):
            row = dict(r)
            row["selected_rank"] = idx
            row["score"] = (len(selected_rows) - idx + 1) / float(len(selected_rows) or 1)
            row["subtask_id"] = sub_id
            row["selector_reason"] = "Deterministic selection from mode rank using a shared per-subtask top-n cutoff."
            row["planner_top_n"] = selected_limit
            row["planner_top_n_source"] = top_n_source
            row["fallback_used"] = fallback_used
            selected.append(row)
        selected_all.extend(selected)
        _write_json(mode_dir / f"3_selected_s{sub_id}.json", selected)
        _write_json(
            mode_dir / f"3_selected_trace_s{sub_id}.json",
            {
                "planner_top_n": selected_limit,
                "planner_top_n_source": top_n_source,
                "dynamic_top_n_from_relevancy": dynamic_top_n,
                "fallback_used": fallback_used,
                "available_ranked": len(ranked_rows),
                "selected_count": len(selected),
            },
        )
    planner = planner_call(llm_call=lambda p: llm_call("planner", PLANNER_SYS, p), user_goal=user_goal, ranked_top=selected_all, subtasks=subtasks, prompt_path=planner_prompt_path)
    _write_json(mode_dir / "4_planner.json", planner)
    return {"selected": selected_all, "planner": planner}


def run_autogen_once(user_goal: str, provider: str | None = None, model: str | None = None, query_id: str | None = None, query_title: str | None = None, run_tag: str | None = None) -> Tuple[Path, Path | None]:
    backend = make_backend(provider=provider, model=model)
    model_tag = backend.name()
    llm_call = _build_llm_call(backend)
    effective_run_tag = CONFIG.run_tag if run_tag is None else run_tag
    out_dir = _run_dir(model_tag, query_id=query_id, run_tag=effective_run_tag)

    raw_subtasks = decompose_goal(llm_call=lambda p: llm_call("decomposer", DECOMPOSER_SYS, p), user_goal=user_goal)
    subtasks = raw_subtasks
    _write_json(out_dir / "debug" / "0_decomposer_raw.json", raw_subtasks)
    _write_json(out_dir / "0_decomposer.json", subtasks)

    run_config = CONFIG.as_dict()
    run_config.update({"provider": provider, "model": backend.name(), "query_id": query_id, "query_title": query_title, "modes": MODE_ORDER})
    _write_json(out_dir / "run_config.json", run_config)
    _write_json(
        out_dir / "meta.json",
        {
            "query_id": query_id,
            "query_title": query_title,
            "user_goal": user_goal,
            "model_tag": backend.name(),
            "num_subtasks": len(subtasks),
            "evaluation_triggered": False,
            "modes": MODE_ORDER,
        },
    )

    retrieved_by_subtask, no_qos_services, with_qos_services = _build_shared_retrieval(user_goal, subtasks, out_dir)

    ranked_full_by_mode: Dict[str, Dict[str, List[Dict[str, Any]]]] = {m: {} for m in MODE_ORDER}
    relevancy_map: Dict[tuple[str, str], Dict[str, Any]] = {}
    planner_top_n: Dict[str, int] = {}
    retrieval_relevancy_rows_path: Path | None = None
    eval_out: Path | None = None
    relevancy_rows_path: Path | None = None
    duplicate_audit_xlsx: Path | None = None
    hallucination_audit_xlsx: Path | None = None
    mode_anomaly_xlsx: Path | None = None
    duplicate_audit_json: Path | None = None
    hallucination_audit_json: Path | None = None
    try:
        eval_dir = out_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_cache = eval_dir / "relevancy_cache.json"
        print(f"[{query_id}] starting retrieval relevancy evaluation")
        retrieval_relevancy_rows_path = evaluate_retrieval_relevancy(
            query_dir=out_dir,
            query_id=query_id,
            output_dir=eval_dir,
            cache_path=eval_cache,
            provider=provider or "azure",
            model=model,
        )
        print(f"[{query_id}] finished retrieval relevancy evaluation")
        relevancy_map = _load_relevancy_map(retrieval_relevancy_rows_path)
    except Exception as e:
        (out_dir / "retrieval_relevancy_error.txt").write_text(str(e), encoding="utf-8")

    print(f"[{query_id}] starting ranking stages")
    for sub in subtasks:
        sub_id = str(sub.get("id", "unknown"))
        retrieved = retrieved_by_subtask[sub_id]

        no_qos_candidates = _candidate_rows(retrieved, no_qos_services)
        no_qos_ranked = rank_subtask(llm_call=lambda p: llm_call("ranker", RANKER_SYS, p), user_query=user_goal, subtask=sub, candidates=no_qos_candidates, prompt_path="prompts/ranker_no_qos.md", debug_raw_path=str(out_dir / "no_qos" / "debug" / f"2_ranker_raw_s{sub_id}.txt"))
        ranked_full_by_mode["no_qos"][sub_id] = _write_ranked(out_dir / "no_qos", sub_id, no_qos_ranked, retrieved, no_qos_services)

        pure_qos_candidates = []
        for r in retrieved:
            api_id = str(r.get("api_id", ""))
            svc = with_qos_services.get(api_id, {})
            qos = (svc.get("qos") or {}) if isinstance(svc.get("qos"), dict) else {}
            pure_qos_candidates.append({
                "api_id": api_id,
                "rt_ms": qos.get("rt_ms"),
                "tp_rps": qos.get("tp_rps"),
                "availability": qos.get("availability"),
            })
        pure_qos_meta = score_qos_llm(
            llm_call=lambda p: llm_call("qos_scorer", QOS_SCORER_SYS, p),
            candidates=pure_qos_candidates,
            prompt_path="prompts/qos_score_llm.md",
            debug_raw_path=str(out_dir / "qos_pure_llm" / "debug" / f"1_qos_scores_raw_s{sub_id}.txt"),
        )
        pure_candidates = _candidate_rows(retrieved, with_qos_services, enrich=pure_qos_meta)
        pure_ranked = rank_subtask(llm_call=lambda p: llm_call("ranker", RANKER_SYS, p), user_query=user_goal, subtask=sub, candidates=pure_candidates, prompt_path="prompts/ranker_qos_pure_llm.md", debug_raw_path=str(out_dir / "qos_pure_llm" / "debug" / f"2_ranker_raw_s{sub_id}.txt"))
        ranked_full_by_mode["qos_pure_llm"][sub_id] = _write_ranked(out_dir / "qos_pure_llm", sub_id, pure_ranked, retrieved, with_qos_services, pure_qos_meta)
        _write_json(out_dir / "qos_pure_llm" / "debug" / f"1_qos_scores_s{sub_id}.json", pure_qos_meta)

        topsis_meta = _compute_topsis_metadata(retrieved, with_qos_services)
        topsis_ranked = _deterministic_topsis_ranking(retrieved, topsis_meta)
        ranked_full_by_mode["qos_topsis"][sub_id] = _write_ranked(out_dir / "qos_topsis", sub_id, topsis_ranked, retrieved, with_qos_services, topsis_meta)

        hybrid_ranked = _deterministic_hybrid_ranking(retrieved, topsis_meta, sub_id, relevancy_map)
        ranked_full_by_mode["qos_hybrid"][sub_id] = _write_ranked(out_dir / "qos_hybrid", sub_id, hybrid_ranked, retrieved, with_qos_services, topsis_meta)

    try:
        eval_dir = out_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_cache = eval_dir / "relevancy_cache.json"
        print(f"[{query_id}] building final api relevancy outputs from retrieval cache")
        eval_out = evaluate_query(query_dir=out_dir, query_id=query_id, output_dir=eval_dir, cache_path=eval_cache, provider=provider or "azure", model=model)
        print(f"[{query_id}] finished building final api relevancy outputs")
        relevancy_rows_path = eval_dir / f"query_{query_id}_api_relevancy_rows.json"
        if relevancy_rows_path.exists():
            planner_top_n = _load_planner_top_n(relevancy_rows_path)

        duplicate_audit = collect_duplicate_audit_for_run(out_dir)
        duplicate_audit_json = eval_dir / f"query_{query_id}_duplicate_audit.json"
        _write_json(duplicate_audit_json, duplicate_audit)

        hallucination_audit = collect_hallucination_audit_for_run(out_dir, CONFIG.catalog_no_qos_path)
        hallucination_audit_json = eval_dir / f"query_{query_id}_hallucination_audit.json"
        _write_json(hallucination_audit_json, hallucination_audit)

        mode_anomaly_xlsx = eval_dir / f"query_{query_id}_mode_anomalies.xlsx"
        write_mode_anomaly_excel(duplicate_audit, hallucination_audit, mode_anomaly_xlsx)

        _write_json(
            out_dir / "evaluation_result.json",
            {
                "evaluation_dir": _to_run_relative(eval_dir, out_dir),
                "api_relevancy_excel": _to_run_relative(eval_out, out_dir),
                "api_relevancy_rows_json": _to_run_relative(relevancy_rows_path, out_dir),
                "retrieval_relevancy_rows_json": _to_run_relative(retrieval_relevancy_rows_path, out_dir),
                "duplicate_audit_json": _to_run_relative(duplicate_audit_json, out_dir),
                "hallucination_audit_json": _to_run_relative(hallucination_audit_json, out_dir),
                "mode_anomaly_excel": _to_run_relative(mode_anomaly_xlsx, out_dir),
                "cache_path": _to_run_relative(eval_cache, out_dir),
            },
        )
    except Exception as e:
        (out_dir / "evaluation_error.txt").write_text(str(e), encoding="utf-8")

    summary_selected = {"planner_enabled": CONFIG.planner_enabled}
    if CONFIG.planner_enabled:
        for mode in MODE_ORDER:
            planner_prompt = "prompts/planner_no_qos.md" if mode == "no_qos" else "prompts/planner.md"
            result = _deterministic_select_and_plan(mode, subtasks, ranked_full_by_mode[mode], planner_top_n, llm_call, user_goal, out_dir, planner_prompt)
            summary_selected[f"{mode}_selected"] = len(result["selected"])
    else:
        summary_selected["planner_skipped_reason"] = "planner disabled in pipeline_config"

    _write_json(
        out_dir / "meta.json",
        {
            "query_id": query_id,
            "query_title": query_title,
            "user_goal": user_goal,
            "model_tag": backend.name(),
            "num_subtasks": len(subtasks),
            "evaluation_triggered": True,
            "evaluation_dir": "evaluation",
            "planner_enabled": CONFIG.planner_enabled,
            "modes": MODE_ORDER,
            "summary": summary_selected,
        },
    )

    print(f"Saved run to {out_dir}")
    if eval_out is not None:
        print(f"Saved evaluation to {eval_out}")
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
        run_autogen_once(user_goal=goal, provider=provider, model=None, query_id=qid, query_title=title)
