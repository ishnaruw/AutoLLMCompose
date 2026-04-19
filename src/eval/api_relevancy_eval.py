from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.core.retry import call_with_backoff
from src.eval.api_relevancy_excel import write_relevancy_excel
from src.eval.api_relevancy_prompt import build_llm_prompt
from src.llm.backends import make_backend
from src.tools.fetch_services import load_catalog_map

MODE_DIRS = ["no_qos", "qos_pure_llm", "qos_topsis"]
CHUNK_SIZE = 3

EVAL_SYS = (
    "You are a strict API relevance evaluator. "
    "Decide only whether an API is functionally relevant to a subtask. "
    "Return strict JSON only."
)


def _safe_read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_subtasks(query_dir: Path) -> List[Dict[str, Any]]:
    data = _safe_read_json(query_dir / "0_decomposer.json") or []
    return data if isinstance(data, list) else []


def _load_meta(query_dir: Path) -> Dict[str, Any]:
    data = _safe_read_json(query_dir / "meta.json") or {}
    return data if isinstance(data, dict) else {}


def _load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_cache(cache: Dict[str, Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def _chunk_list(items: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _normalize_relevant(val: Any) -> Optional[int]:
    if isinstance(val, bool):
        return 1 if val else 0
    if val in (0, 1):
        return int(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in {"1", "true", "yes", "relevant"}:
            return 1
        if s in {"0", "false", "no", "irrelevant", "not relevant"}:
            return 0
    return None


def _parse_results(text: str, expected_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    try:
        data = json.loads(text)
    except Exception:
        return {}

    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        return {}

    expected_set = set(expected_ids)
    out: Dict[str, Dict[str, Any]] = {}
    for r in results:
        if not isinstance(r, dict):
            continue
        api_id = str(r.get("api_id", "")).strip()
        if api_id not in expected_set:
            continue
        rel = _normalize_relevant(r.get("relevant"))
        if rel is None:
            continue
        out[api_id] = {
            "relevant": rel,
            "comment": str(r.get("comment", "")).strip()[:200],
        }
    return out


def _extract_api_info(item: Dict[str, Any], catalog: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    api_id = str(item.get("api_id", "")).strip()
    service = item.get("service") or {}
    catalog_entry = catalog.get(api_id, {})
    merged = dict(catalog_entry)
    if isinstance(service, dict):
        merged.update({k: v for k, v in service.items() if v is not None})
    merged.setdefault("api_id", api_id)
    return merged


def _load_shared_candidates(query_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for path in sorted((query_dir / "shared").glob("1_retriever_s*.json")):
        sid = path.stem.split("_s")[-1]
        data = _safe_read_json(path) or []
        if isinstance(data, list):
            out[sid] = data
    return out


def _load_ranked_files(query_dir: Path) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {mode: {} for mode in MODE_DIRS}
    for mode in MODE_DIRS:
        mode_dir = query_dir / mode
        for path in sorted(mode_dir.glob("2_ranked_s*.json")):
            sid = path.stem.split("_s")[-1]
            data = _safe_read_json(path) or []
            if isinstance(data, list):
                out[mode][sid] = data
    return out


def _build_relevancy_batches(
    query_id: str,
    main_task: str,
    subtasks: List[Dict[str, Any]],
    shared_candidates: Dict[str, List[Dict[str, Any]]],
    catalog: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    batches: List[Dict[str, Any]] = []
    for sub in subtasks:
        sid = str(sub.get("id"))
        purpose = str(sub.get("description", ""))
        apis: List[Dict[str, Any]] = []
        for item in shared_candidates.get(sid, []):
            api_info = _extract_api_info(item, catalog)
            apis.append(
                {
                    "api_id": str(item.get("api_id", "")),
                    "name": api_info.get("name") or api_info.get("title") or api_info.get("operation"),
                    "category": api_info.get("category"),
                    "tool_name": api_info.get("tool_name"),
                    "tool_description": api_info.get("tool_description"),
                    "description": api_info.get("description") or api_info.get("summary") or api_info.get("desc"),
                    "method": api_info.get("method"),
                    "url": api_info.get("url") or api_info.get("endpoint") or api_info.get("path"),
                    "endpoint_details": api_info.get("endpoint_details") or {},
                }
            )
        batches.append(
            {
                "query_id": query_id,
                "main_task": main_task,
                "subtask_id": sid,
                "subtask_description": purpose,
                "apis": apis,
            }
        )
    return batches


def evaluate_query(
    *,
    query_dir: Path,
    query_id: Optional[str],
    provider: str,
    model: Optional[str] = None,
    output_dir: Path,
    cache_path: Path,
) -> Tuple[Path, Dict[str, Dict[str, Dict[str, Any]]]]:
    backend = make_backend(provider=provider, model=model)
    meta = _load_meta(query_dir)
    main_task = str(meta.get("user_goal") or "")
    query_id = query_id or str(meta.get("query_id") or query_dir.name)

    subtasks = _load_subtasks(query_dir)
    shared_candidates = _load_shared_candidates(query_dir)
    ranked_by_mode = _load_ranked_files(query_dir)
    catalog = load_catalog_map(with_qos=True)
    cache = _load_cache(cache_path)

    relevancy_by_subtask: Dict[str, Dict[str, Dict[str, Any]]] = {}
    rows: List[Dict[str, Any]] = []
    llm_calls = 0
    total_cached = 0

    batches = _build_relevancy_batches(query_id, main_task, subtasks, shared_candidates, catalog)
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    for batch in batches:
        sid = batch["subtask_id"]
        apis = batch["apis"]
        sub_results: Dict[str, Dict[str, Any]] = {}
        uncached: List[Dict[str, Any]] = []
        for api in apis:
            key = f"{query_id}_{sid}_{api['api_id']}"
            if key in cache:
                sub_results[api["api_id"]] = cache[key]
                total_cached += 1
            else:
                uncached.append(api)

        for chunk_idx, chunk in enumerate(_chunk_list(uncached, CHUNK_SIZE), start=1):
            prompt = build_llm_prompt(
                query_id=query_id,
                main_task=batch["main_task"],
                subtask_id=sid,
                subtask_description=batch["subtask_description"],
                api_entries=chunk,
            )
            expected_ids = [a["api_id"] for a in chunk]

            def _call() -> str:
                return backend.chat_json(EVAL_SYS, prompt, temperature=0, force_json=True)

            raw = call_with_backoff(_call, name=f"api_relevancy_eval_s{sid}_chunk{chunk_idx}")
            parsed = _parse_results(raw, expected_ids)
            llm_calls += 1
            for api in chunk:
                api_id = api["api_id"]
                val = parsed.get(api_id, {"relevant": 0, "comment": "Missing from LLM response"})
                cache[f"{query_id}_{sid}_{api_id}"] = val
                sub_results[api_id] = val

        relevancy_by_subtask[sid] = sub_results

    _save_cache(cache, cache_path)

    for sub in subtasks:
        sid = str(sub.get("id"))
        purpose = str(sub.get("description", ""))
        retrieved_rows = shared_candidates.get(sid, [])
        retrieved_rank_map = {
            str(item.get("api_id", "")): idx for idx, item in enumerate(retrieved_rows, start=1)
        }
        for mode in MODE_DIRS:
            mode_rows = ranked_by_mode.get(mode, {}).get(sid, [])
            for row in mode_rows:
                api_id = str(row.get("api_id", "")).strip()
                api_info = _extract_api_info(row, catalog)
                qos = api_info.get("qos") if isinstance(api_info.get("qos"), dict) else {}
                rel_info = relevancy_by_subtask.get(sid, {}).get(api_id, {"relevant": 0, "comment": "Missing"})
                rows.append(
                    {
                        "Query_ID": query_id,
                        "Mode": mode,
                        "Sub Task": sid,
                        "Retrieved Rank": retrieved_rank_map.get(api_id),
                        "Mode Rank": row.get("mode_rank") or row.get("rank"),
                        "Subtask_Purpose": purpose,
                        "Selected_API": api_id,
                        "API Relevancy (0/1)": rel_info.get("relevant", 0),
                        "QoS_RT": qos.get("rt_ms"),
                        "QoS_TP": qos.get("tp_rps"),
                        "QoS Availability": qos.get("availability"),
                        "Comments": rel_info.get("comment", ""),
                    }
                )

    rows.sort(
        key=lambda r: (
            int(str(r["Sub Task"])) if str(r["Sub Task"]).isdigit() else 9999,
            MODE_DIRS.index(str(r["Mode"])) if str(r["Mode"]) in MODE_DIRS else 999,
            int(r["Mode Rank"] or 9999),
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "final_rows_preview.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    out_xlsx = output_dir / f"query_{query_id}_api_relevancy.xlsx"
    write_relevancy_excel(rows, out_xlsx)

    summary = {
        "query_id": query_id,
        "query_dir": str(query_dir),
        "provider": provider,
        "model": backend.name(),
        "total_rows": len(rows),
        "unique_apis_evaluated": len({(sid, api_id) for sid, per_sub in relevancy_by_subtask.items() for api_id in per_sub}),
        "cached_results": total_cached,
        "llm_calls": llm_calls,
        "excel": str(out_xlsx),
        "cache": str(cache_path),
    }
    (output_dir / f"query_{query_id}_api_relevancy_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return out_xlsx, relevancy_by_subtask
