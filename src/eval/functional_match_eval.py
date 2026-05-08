from __future__ import annotations

import json
import math
import re
from pathlib import Path
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.config import CONFIG
from src.core.api_formatting import normalize_api_for_ranking
from src.core.candidate_ids import assign_candidate_ids
from src.core.run_logging import log_line, log_warning_event
from src.core.retry import call_with_backoff
from src.eval.candidate_api_rankings_excel import write_candidate_api_rankings_excel
from src.eval.functional_match_prompt import build_llm_prompt
from src.llm.autogen_runner import run_autogen_agent
from src.llm.backends import make_backend

CATALOG_WITH_QOS_PATH = Path("data/processed/api_catalog_sample_balanced/api_repo.with_qos.jsonl")
MODE_DIRS = ["no_qos", "qos_pure_llm", "qos_topsis", "qos_hybrid"]
MODE_ORDER = {name: idx for idx, name in enumerate(MODE_DIRS)}
EVAL_SYS = "You are a strict functional-match evaluator. Decide only whether an API is functionally suitable for a subtask. Return strict JSON only."


def _chunk_size() -> int:
    try:
        return max(1, int(CONFIG.functional_match_chunk_size))
    except Exception:
        return 3


def _safe_read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_jsonl_catalog(path: Path) -> Dict[str, Dict[str, Any]]:
    catalog: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return catalog
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            api_id = obj.get("api_id")
            if api_id:
                catalog[str(api_id)] = obj
    return catalog


def _load_ranked_files(query_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for mode in MODE_DIRS:
        mode_dir = query_dir / mode
        rows: List[Dict[str, Any]] = []
        for path in sorted(mode_dir.glob("2_ranked_s*.json")):
            m = re.search(r"2_ranked_s(\d+)\.json$", path.name)
            sid = m.group(1) if m else None
            data = _safe_read_json(path) or []
            if isinstance(data, list):
                for x in data:
                    if not isinstance(x, dict):
                        continue
                    row = dict(x)
                    if sid is not None:
                        row["subtask_id"] = sid
                    row["_ranked_file"] = path.name
                    rows.append(row)
        out[mode] = rows
    return out


def _load_shared_candidates(query_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for path in sorted(query_dir.glob("1_retriever_s*.json")):
        m = re.search(r"1_retriever_s(\d+)\.json$", path.name)
        sid = m.group(1) if m else None
        if sid is None:
            continue
        data = _safe_read_json(path) or []
        if isinstance(data, list):
            out[sid] = data
    return out


def _build_duplicate_flag_map(
    ranked_by_mode: Dict[str, List[Dict[str, Any]]],
) -> Dict[Tuple[str, str, str], int]:
    counts: Counter[Tuple[str, str, str]] = Counter()
    for mode, items in ranked_by_mode.items():
        for item in items:
            subtask_id = str(item.get("subtask_id") or "").strip()
            api_id = str(item.get("api_id") or "").strip()
            if not subtask_id or not api_id:
                continue
            counts[(mode, subtask_id, api_id)] += 1
    return {key: 1 if count > 1 else 0 for key, count in counts.items()}


def _build_hallucination_flag_map(
    ranked_by_mode: Dict[str, List[Dict[str, Any]]],
    shared_candidates: Dict[str, List[Dict[str, Any]]],
    catalog_ids: set[str],
) -> Dict[Tuple[str, str, str], int]:
    retrieved_sets: Dict[str, set[str]] = {}
    retrieved_catalog_missing_by_subtask: Dict[str, set[str]] = {}

    for subtask_id, items in shared_candidates.items():
        retrieved_ids = {
            str(item.get("api_id") or "").strip()
            for item in items
            if str(item.get("api_id") or "").strip()
        }
        retrieved_sets[subtask_id] = retrieved_ids
        retrieved_catalog_missing_by_subtask[subtask_id] = {
            api_id for api_id in retrieved_ids if api_id not in catalog_ids
        }

    flags: Dict[Tuple[str, str, str], int] = {}
    for mode, items in ranked_by_mode.items():
        for item in items:
            subtask_id = str(item.get("subtask_id") or "").strip()
            api_id = str(item.get("api_id") or "").strip()
            if not subtask_id or not api_id:
                continue
            retrieved_set = retrieved_sets.get(subtask_id, set())
            in_retrieved = api_id in retrieved_set
            in_catalog = api_id in catalog_ids
            retrieved_but_catalog_missing = api_id in retrieved_catalog_missing_by_subtask.get(subtask_id, set())
            flags[(mode, subtask_id, api_id)] = 1 if (not in_retrieved or not in_catalog or retrieved_but_catalog_missing) else 0
    return flags


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


def _save_progress(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _extract_api_info(item: Dict[str, Any], catalog: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    api_id = str(item.get("api_id", ""))
    service = item.get("service") or {}
    catalog_entry = catalog.get(api_id, {})
    merged = dict(catalog_entry)
    merged.update({k: v for k, v in service.items() if v is not None})
    if "qos" not in merged and isinstance(catalog_entry.get("qos"), dict):
        merged["qos"] = catalog_entry.get("qos")
    merged.setdefault("api_id", api_id)
    return merged


def _sort_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _retrieved_rank_sort_key(item: Dict[str, Any]) -> int:
    try:
        return int(item.get("retrieved_rank") or 10**9)
    except Exception:
        return 10**9


def _prompt_api_sort_key(api: Dict[str, Any]) -> Tuple[str, str, str]:
    if not isinstance(api, dict):
        return ("", "", "")
    api_id = api.get("api_id")
    name = api.get("name")
    tool_name = api.get("tool_name")
    primary = api_id or name or tool_name or ""
    return (_sort_text(primary), _sort_text(name), _sort_text(tool_name))


def _warn_api_id_quality(candidates: List[Dict[str, Any]], *, context: str) -> None:
    ids = [str(c.get("api_id") or "").strip() for c in candidates if isinstance(c, dict)]
    missing = sum(1 for api_id in ids if not api_id)
    duplicates = sorted(api_id for api_id, count in Counter(api_id for api_id in ids if api_id).items() if count > 1)
    if not missing and not duplicates:
        return

    parts = []
    if missing:
        parts.append(f"{missing} missing api_id")
    if duplicates:
        preview = ", ".join(duplicates[:5])
        suffix = "..." if len(duplicates) > 5 else ""
        parts.append(f"{len(duplicates)} duplicate api_id value(s): {preview}{suffix}")
    log_line(f"[retrieval_functional_match] warning: {context} has " + "; ".join(parts))
    if missing:
        log_warning_event(
            {
                "warning_type": "missing_api_id",
                "missing_api_id_count": missing,
                "context": context,
                "source": "functional_match_prompt_payload",
            }
        )
    if duplicates:
        log_warning_event(
            {
                "warning_type": "duplicate_api_id_in_prompt_payload",
                "duplicate_api_ids": duplicates,
                "duplicate_api_id_count": len(duplicates),
                "context": context,
                "source": "functional_match_prompt_payload",
            }
        )


def _fixed_retrieval_pool(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    try:
        top_k = max(1, int(CONFIG.rag_top_k))
    except Exception:
        top_k = 40
    return sorted(
        (item for item in items if isinstance(item, dict)),
        key=_retrieved_rank_sort_key,
    )[:top_k]


def _build_shared_retrieval_batches(query_id: str, main_task: str, subtasks: List[Dict[str, Any]], shared_candidates: Dict[str, List[Dict[str, Any]]], catalog: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    batches: List[Dict[str, Any]] = []
    for sub in subtasks:
        sid = str(sub.get("id"))
        purpose = sub.get("description", "")
        candidate_pool = _fixed_retrieval_pool(shared_candidates.get(sid, []))
        apis: List[Dict[str, Any]] = []
        for row in candidate_pool:
            merged = _extract_api_info(row, catalog)
            apis.append(normalize_api_for_ranking(merged, subtask_text=str(purpose), include_qos_rank=False))
        _warn_api_id_quality(
            apis,
            context=f"query={query_id} subtask={sid} top-{len(candidate_pool)} functional-match prompt pool",
        )
        apis = sorted(apis, key=_prompt_api_sort_key)
        apis, _candidate_id_to_api_id, _api_id_to_candidate_id = assign_candidate_ids(apis)
        batches.append({
            "query_id": query_id,
            "main_task": main_task,
            "subtask_id": sid,
            "subtask_description": purpose,
            "apis": apis,
        })
    return batches


def _chunk_list(items: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _normalize_functional_match(val: Any) -> Optional[int]:
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


def _functional_match_value(info: Dict[str, Any]) -> int:
    return int(
        info.get(
            "Functional Match (0/1)",
            info.get("Functional Match Label", info.get("functional_match", info.get("relevant", 0))),
        )
        or 0
    )


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _format_list_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _extract_json_text(raw: str) -> Tuple[str, str | None]:
    text = (raw or "").strip()
    if not text:
        return "", "empty_response"
    try:
        json.loads(text)
        return text, None
    except Exception:
        pass
    match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if not match:
        return "", "invalid_json"
    return match.group(1), None


def _functional_match_label_value(item: Dict[str, Any]) -> Any:
    if "label" in item:
        return item.get("label")
    return item.get("functional_match", item.get("relevant"))


def _functional_match_comment_value(item: Dict[str, Any]) -> str:
    return str(item.get("comment", item.get("reason", item.get("explanation", ""))) or "").strip()[:200]


def _is_failure_row(item: Dict[str, Any]) -> bool:
    return _truthy(item.get("failure_flag")) or _truthy(item.get("exclude_from_ranking_eval"))


def _planner_k_by_subtask(
    *,
    query_id: str,
    subtasks: List[Dict[str, Any]],
    shared_candidates: Dict[str, List[Dict[str, Any]]],
    cache: Dict[str, Dict[str, Any]],
) -> Dict[str, int]:
    planner_k: Dict[str, int] = {}
    for sub in subtasks:
        sid = str(sub.get("id"))
        matched_api_ids: set[str] = set()
        for item in shared_candidates.get(sid, []):
            api_id = str(item.get("api_id", ""))
            if not api_id:
                continue
            rel_info = cache.get(f"{query_id}_{sid}_{api_id}", {})
            if _functional_match_value(rel_info) == 1:
                matched_api_ids.add(api_id)
        planner_k[sid] = len(matched_api_ids)
    return planner_k


def _parse_results(
    text: str,
    expected_ids: List[str],
    candidate_id_to_api_id: Dict[str, str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    parsed, _issue = _parse_results_with_issue(text, expected_ids, candidate_id_to_api_id)
    return parsed


def _parse_results_with_issue(
    text: str,
    expected_ids: List[str],
    candidate_id_to_api_id: Dict[str, str] | None = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any] | None]:
    candidate_id_to_api_id = {
        str(candidate_id).strip(): str(api_id).strip()
        for candidate_id, api_id in (candidate_id_to_api_id or {}).items()
        if str(candidate_id).strip() and str(api_id).strip()
    }
    api_id_to_candidate_id: Dict[str, str] = {}
    for candidate_id, api_id in candidate_id_to_api_id.items():
        api_id_to_candidate_id.setdefault(api_id, candidate_id)
    expected_candidate_ids = list(candidate_id_to_api_id.keys())

    json_text, json_error = _extract_json_text(text)
    if json_error:
        return {}, {
            "reason": json_error,
            "expected_api_count": len(expected_ids),
            "expected_candidate_count": len(expected_candidate_ids),
            "actual_api_count": 0,
            "actual_candidate_count": 0,
        }

    try:
        data = json.loads(json_text)
    except Exception as exc:
        return {}, {
            "reason": "invalid_json",
            "expected_api_count": len(expected_ids),
            "expected_candidate_count": len(expected_candidate_ids),
            "actual_api_count": 0,
            "actual_candidate_count": 0,
            "parse_error": str(exc),
        }
    results = None
    if isinstance(data, dict):
        results = data.get("matches")
        if not isinstance(results, list):
            results = data.get("results")
    if not isinstance(results, list):
        return {}, {
            "reason": "parse_error",
            "expected_api_count": len(expected_ids),
            "expected_candidate_count": len(expected_candidate_ids),
            "actual_api_count": 0,
            "actual_candidate_count": 0,
            "detail": "missing_matches_or_results_list",
        }
    contains_candidate_ids = any(
        isinstance(item, dict) and str(item.get("candidate_id") or "").strip()
        for item in results
    )
    if candidate_id_to_api_id and contains_candidate_ids:
        return _parse_results_by_candidate_id(results, expected_ids, candidate_id_to_api_id, api_id_to_candidate_id)
    return _parse_results_by_api_id(results, expected_ids, expected_candidate_ids)


def _parse_results_by_api_id(
    results: List[Any],
    expected_ids: List[str],
    expected_candidate_ids: List[str] | None = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any] | None]:
    expected_candidate_ids = expected_candidate_ids or []
    expected_set = set(expected_ids)
    out: Dict[str, Dict[str, Any]] = {}
    returned_ids: List[str] = []
    malformed_items = 0
    for r in results:
        if not isinstance(r, dict):
            malformed_items += 1
            continue
        api_id = r.get("api_id")
        if api_id is None:
            malformed_items += 1
            continue
        api_id = str(api_id).strip()
        returned_ids.append(api_id)
        if api_id not in expected_set:
            continue
        functional_match = _normalize_functional_match(_functional_match_label_value(r))
        if functional_match is None:
            malformed_items += 1
            continue
        out[api_id] = {"functional_match": functional_match, "comment": _functional_match_comment_value(r)}

    returned_counts = Counter(returned_ids)
    duplicate_ids = sorted(api_id for api_id, count in returned_counts.items() if count > 1)
    unknown_ids = sorted(api_id for api_id in returned_counts if api_id not in expected_set)
    missing_ids = [api_id for api_id in expected_ids if api_id not in out]
    issue_base = {
        "expected_api_count": len(expected_ids),
        "actual_api_count": len(out),
        "returned_api_count": len(returned_ids),
    }
    if expected_candidate_ids:
        issue_base.update(
            {
                "expected_candidate_count": len(expected_candidate_ids),
                "actual_candidate_count": len(out),
            }
        )
    if malformed_items:
        return out, {**issue_base, "reason": "parse_error", "malformed_result_count": malformed_items}
    if duplicate_ids:
        return out, {**issue_base, "reason": "duplicate_api_ids", "duplicate_api_ids": duplicate_ids}
    if unknown_ids:
        return out, {**issue_base, "reason": "unknown_api_ids", "unknown_api_ids": unknown_ids, "missing_api_ids": missing_ids}
    if missing_ids:
        return out, {**issue_base, "reason": "incomplete_functional_match_results", "missing_api_ids": missing_ids}
    return out, None


def _parse_results_by_candidate_id(
    results: List[Any],
    expected_ids: List[str],
    candidate_id_to_api_id: Dict[str, str],
    api_id_to_candidate_id: Dict[str, str],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any] | None]:
    expected_candidate_ids = list(candidate_id_to_api_id.keys())
    expected_candidate_set = set(expected_candidate_ids)
    out: Dict[str, Dict[str, Any]] = {}
    returned_candidate_ids: List[str] = []
    unknown_legacy_api_ids: List[str] = []
    malformed_items = 0
    for r in results:
        if not isinstance(r, dict):
            malformed_items += 1
            continue
        candidate_id = str(r.get("candidate_id") or "").strip()
        raw_api_id = str(r.get("api_id") or "").strip()
        if candidate_id:
            api_id = candidate_id_to_api_id.get(candidate_id, raw_api_id)
            returned_candidate_ids.append(candidate_id)
        elif raw_api_id:
            api_id = raw_api_id
            candidate_id = api_id_to_candidate_id.get(raw_api_id, "")
            if candidate_id:
                returned_candidate_ids.append(candidate_id)
            else:
                unknown_legacy_api_ids.append(raw_api_id)
        else:
            malformed_items += 1
            continue
        if candidate_id not in expected_candidate_set:
            continue
        functional_match = _normalize_functional_match(_functional_match_label_value(r))
        if functional_match is None:
            malformed_items += 1
            continue
        out[api_id] = {
            "candidate_id": candidate_id,
            "functional_match": functional_match,
            "comment": _functional_match_comment_value(r),
        }

    returned_counts = Counter(returned_candidate_ids)
    duplicate_candidate_ids = sorted(candidate_id for candidate_id, count in returned_counts.items() if count > 1)
    unknown_candidate_ids = sorted(candidate_id for candidate_id in returned_counts if candidate_id not in expected_candidate_set)
    missing_candidate_ids = [candidate_id for candidate_id in expected_candidate_ids if candidate_id not in returned_counts]
    duplicate_api_ids = sorted(
        {
            candidate_id_to_api_id[candidate_id]
            for candidate_id in duplicate_candidate_ids
            if candidate_id in candidate_id_to_api_id
        }
    )
    missing_api_ids = [
        candidate_id_to_api_id[candidate_id]
        for candidate_id in missing_candidate_ids
        if candidate_id in candidate_id_to_api_id
    ]
    unknown_api_ids = sorted(set(unknown_legacy_api_ids))
    issue_base = {
        "expected_api_count": len(expected_ids),
        "actual_api_count": len(out),
        "returned_api_count": len(returned_candidate_ids) + len(unknown_legacy_api_ids),
        "expected_candidate_count": len(expected_candidate_ids),
        "actual_candidate_count": len(out),
        "returned_candidate_count": len(returned_candidate_ids),
    }
    if malformed_items:
        return out, {**issue_base, "reason": "parse_error", "malformed_result_count": malformed_items}
    if duplicate_candidate_ids:
        return out, {
            **issue_base,
            "reason": "duplicate_candidate_ids",
            "duplicate_candidate_ids": duplicate_candidate_ids,
            "duplicate_api_ids": duplicate_api_ids,
        }
    if unknown_candidate_ids:
        return out, {
            **issue_base,
            "reason": "unknown_candidate_ids",
            "unknown_candidate_ids": unknown_candidate_ids,
            "missing_candidate_ids": missing_candidate_ids,
            "missing_api_ids": missing_api_ids,
        }
    if unknown_api_ids:
        return out, {
            **issue_base,
            "reason": "unknown_api_ids",
            "unknown_api_ids": unknown_api_ids,
            "missing_candidate_ids": missing_candidate_ids,
            "missing_api_ids": missing_api_ids,
        }
    if missing_candidate_ids:
        return out, {
            **issue_base,
            "reason": "incomplete_functional_match_results",
            "missing_candidate_ids": missing_candidate_ids,
            "missing_api_ids": missing_api_ids,
        }
    return out, None


def _evaluate_batches(
    *,
    query_id: str,
    batches: List[Dict[str, Any]],
    provider: str,
    model: Optional[str],
    cache_path: Path,
    progress_path: Path,
    stage_name: str,
) -> Dict[Tuple[str, str], Dict[str, Dict[str, Any]]]:
    backend = make_backend(provider=provider, model=model)
    cache = _load_cache(cache_path)
    batch_results: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    chunk_size = _chunk_size()
    total_api_items = sum(len(batch.get("apis", [])) for batch in batches)
    completed_api_items = 0

    def _invoke(fn, *, name: str) -> str:
        if getattr(backend, "multi_model_mode", lambda: False)():
            return fn()
        return call_with_backoff(fn, name=name)

    _save_progress(
        progress_path,
        {
            "stage": stage_name,
            "query_id": query_id,
            "provider": provider,
            "model": backend.name(),
            "active_model": backend.active_model_name(),
            "multi_model_mode": backend.multi_model_mode(),
            "model_pool": backend.model_pool(),
            "chunk_size": chunk_size,
            "status": "running",
            "completed_api_items": 0,
            "total_api_items": total_api_items,
        },
    )

    for batch in batches:
        sid = batch["subtask_id"]
        apis = batch["apis"]
        uncached: List[Dict[str, Any]] = []
        sub_results: Dict[str, Dict[str, Any]] = {}
        for api in apis:
            key = f"{query_id}_{sid}_{api['api_id']}"
            if key in cache:
                sub_results[api["api_id"]] = cache[key]
                completed_api_items += 1
            else:
                uncached.append(api)

        if uncached:
            total_chunks = math.ceil(len(uncached) / float(chunk_size))
            for chunk_idx, chunk in enumerate(_chunk_list(uncached, chunk_size), start=1):
                log_line(
                    f"[{stage_name}] query={query_id} subtask={sid} "
                    f"chunk={chunk_idx}/{total_chunks} completed={completed_api_items}/{total_api_items}"
                )
                prompt = build_llm_prompt(
                    query_id=query_id,
                    main_task=batch["main_task"],
                    subtask_id=sid,
                    subtask_description=batch["subtask_description"],
                    api_entries=chunk,
                )
                expected_ids = [a["api_id"] for a in chunk]
                candidate_id_to_api_id = {
                    str(a.get("candidate_id")): str(a.get("api_id"))
                    for a in chunk
                    if str(a.get("candidate_id") or "").strip() and str(a.get("api_id") or "").strip()
                }
                expected_candidate_ids = list(candidate_id_to_api_id.keys())

                def _call() -> str:
                    timeout_seconds = (
                        CONFIG.lmstudio_timeout_seconds
                        if getattr(backend, "provider", "") in {"lmstudio", "lmstudio_qwen"}
                        else None
                    )
                    if CONFIG.use_autogen_agents:
                        return run_autogen_agent(
                            backend=backend,
                            role_name=f"{stage_name}_evaluator",
                            system_message=EVAL_SYS,
                            prompt=prompt,
                            temperature=0.0,
                            force_json=True,
                            timeout_seconds=timeout_seconds,
                        )
                    return backend.chat_json(
                        EVAL_SYS,
                        prompt,
                        temperature=0,
                        force_json=True,
                        timeout_seconds=timeout_seconds,
                    )

                raw = _invoke(_call, name=f"functional_match_eval_s{sid}_chunk{chunk_idx}")
                parsed, issue = _parse_results_with_issue(raw, expected_ids, candidate_id_to_api_id)
                if issue is not None:
                    log_line(
                        f"[{stage_name}] invalid functional-match output ({issue.get('reason')}) "
                        f"chunk={chunk_idx}/{total_chunks}; expected={issue.get('expected_candidate_count', issue.get('expected_api_count'))} "
                        f"actual={issue.get('actual_candidate_count', issue.get('actual_api_count'))}"
                    )
                    retry_raw = _invoke(_call, name=f"functional_match_eval_retry_s{sid}_chunk{chunk_idx}")
                    retry_parsed, retry_issue = _parse_results_with_issue(retry_raw, expected_ids, candidate_id_to_api_id)
                    if retry_issue is None or len(retry_parsed) >= len(parsed):
                        parsed = retry_parsed

                for api in chunk:
                    api_id = api["api_id"]
                    key = f"{query_id}_{sid}_{api_id}"
                    val = parsed.get(api_id, {"functional_match": 0, "comment": "Missing from LLM response"})
                    cache[key] = val
                    sub_results[api_id] = val
                    completed_api_items += 1

                _save_cache(cache, cache_path)
                _save_progress(
                    progress_path,
                    {
                        "stage": stage_name,
                        "query_id": query_id,
                        "provider": provider,
                        "model": backend.name(),
                        "active_model": backend.active_model_name(),
                        "multi_model_mode": backend.multi_model_mode(),
                        "model_pool": backend.model_pool(),
                        "chunk_size": chunk_size,
                        "status": "running",
                        "current_subtask_id": sid,
                        "current_chunk_index": chunk_idx,
                        "total_chunks_in_subtask": total_chunks,
                        "last_chunk_api_ids": expected_ids,
                        "last_chunk_candidate_ids": expected_candidate_ids,
                        "completed_api_items": completed_api_items,
                        "total_api_items": total_api_items,
                    },
                )

        batch_results[(query_id, sid)] = sub_results

    _save_cache(cache, cache_path)
    _save_progress(
        progress_path,
        {
            "stage": stage_name,
            "query_id": query_id,
            "provider": provider,
            "model": backend.name(),
            "active_model": backend.active_model_name(),
            "multi_model_mode": backend.multi_model_mode(),
            "model_pool": backend.model_pool(),
            "chunk_size": chunk_size,
            "status": "completed",
            "completed_api_items": completed_api_items,
            "total_api_items": total_api_items,
        },
    )
    return batch_results


def evaluate_retrieval_functional_match(*, query_dir: Path, query_id: Optional[str], provider: str, model: Optional[str] = None, output_dir: Path, cache_path: Path) -> Path:
    meta = _load_meta(query_dir)
    main_task = str(meta.get("user_goal") or "")
    query_id = query_id or str(meta.get("query_id") or query_dir.name)
    subtasks = _load_subtasks(query_dir)
    shared_candidates = _load_shared_candidates(query_dir)
    catalog = _load_jsonl_catalog(CATALOG_WITH_QOS_PATH)
    batches = _build_shared_retrieval_batches(query_id, main_task, subtasks, shared_candidates, catalog)
    candidate_ids_by_subtask = {
        batch["subtask_id"]: {
            str(api.get("api_id")): str(api.get("candidate_id"))
            for api in batch.get("apis", [])
            if str(api.get("api_id") or "").strip() and str(api.get("candidate_id") or "").strip()
        }
        for batch in batches
    }
    batch_results = _evaluate_batches(
        query_id=query_id,
        batches=batches,
        provider=provider,
        model=model,
        cache_path=cache_path,
        progress_path=output_dir / "retrieval_functional_match_progress.json",
        stage_name="retrieval_functional_match",
    )

    rows: List[Dict[str, Any]] = []
    for sub in subtasks:
        sid = str(sub.get("id"))
        for item in _fixed_retrieval_pool(shared_candidates.get(sid, [])):
            api_id = str(item.get("api_id", ""))
            rel_info = batch_results.get((query_id, sid), {}).get(api_id, {"functional_match": 0, "comment": "Missing from LLM response"})
            functional_match = _functional_match_value(rel_info)
            rows.append({
                "Query_ID": query_id,
                "Sub Task": sid,
                "Retrieved Rank": item.get("retrieved_rank"),
                "Candidate_ID": rel_info.get("candidate_id", candidate_ids_by_subtask.get(sid, {}).get(api_id, "")),
                "Selected_API": api_id,
                "Functional Match (0/1)": functional_match,
                "Comments": rel_info.get("comment", ""),
            })

    rows.sort(
        key=lambda r: (
            int(str(r["Sub Task"])) if str(r["Sub Task"]).isdigit() else 9999,
            int(r.get("Retrieved Rank") or 9999),
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / f"query_{query_id}_retrieval_functional_match_rows.json"
    rows_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path = output_dir / f"query_{query_id}_retrieval_functional_match_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "query_id": query_id,
                "rows_json": str(rows_path),
                "cache": str(cache_path),
                "total_rows": len(rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return rows_path


def evaluate_query(*, query_dir: Path, query_id: Optional[str], provider: str, model: Optional[str] = None, output_dir: Path, cache_path: Path) -> Path:
    meta = _load_meta(query_dir)
    query_id = query_id or str(meta.get("query_id") or query_dir.name)
    subtasks = _load_subtasks(query_dir)
    ranked_by_mode = _load_ranked_files(query_dir)
    shared_candidates = _load_shared_candidates(query_dir)
    catalog = _load_jsonl_catalog(CATALOG_WITH_QOS_PATH)
    cache = _load_cache(cache_path)
    duplicate_flags = _build_duplicate_flag_map(ranked_by_mode)
    hallucination_flags = _build_hallucination_flag_map(ranked_by_mode, shared_candidates, set(catalog.keys()))
    planner_k_by_subtask = _planner_k_by_subtask(
        query_id=query_id,
        subtasks=subtasks,
        shared_candidates=shared_candidates,
        cache=cache,
    )
    rows: List[Dict[str, Any]] = []
    subtask_map = {str(sub.get("id")): str(sub.get("description", "")) for sub in subtasks}
    retrieved_lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for lookup_subtask_id, lookup_items in shared_candidates.items():
        for lookup_item in lookup_items:
            lookup_api_id = str(lookup_item.get("api_id", "")).strip()
            if lookup_api_id:
                retrieved_lookup.setdefault((str(lookup_subtask_id), lookup_api_id), lookup_item)

    for mode in MODE_DIRS:
        for item in ranked_by_mode.get(mode, []):
            sid = str(item.get("subtask_id") or "")
            purpose = subtask_map.get(sid, "")
            api_id = str(item.get("api_id", ""))
            if _is_failure_row(item):
                rel_info = cache.get(
                    f"{query_id}_{sid}_{api_id}",
                    {"relevant": 0, "comment": item.get("failure_reason", "invalid ranking case")},
                )
                functional_match = _functional_match_value(rel_info)
                catalog_entry = catalog.get(api_id, {})
                service = item.get("service") or {}
                retrieved_item = retrieved_lookup.get((sid, api_id), {})
                if isinstance(service.get("qos"), dict):
                    qos = service.get("qos") or {}
                elif isinstance(catalog_entry.get("qos"), dict):
                    qos = catalog_entry.get("qos") or {}
                else:
                    qos = {}
                rows.append({
                    "Query_ID": query_id,
                    "Mode": mode,
                    "Sub Task": sid,
                    "Retrieved Rank": item.get("retrieved_rank") or retrieved_item.get("retrieved_rank"),
                    "Mode Rank": item.get("mode_rank"),
                    "LLM Reported Rank": item.get("llm_reported_rank", ""),
                    "Subtask_Purpose": purpose,
                    "Candidate_ID": item.get("candidate_id", ""),
                    "Selected_API": api_id,
                    "Ranker Reason": item.get("reason", ""),
                    "Is Hallucinated? (0/1)": hallucination_flags.get((mode, sid, api_id), 0),
                    "Is Duplicated? (0/1)": duplicate_flags.get((mode, sid, api_id), 0),
                    "Functional Match (0/1)": functional_match,
                    "Used in Ranking": "No",
                    "Selected for Planner": "No",
                    "Planner Selection K": planner_k_by_subtask.get(sid, 0),
                    "Failure Flag": 1,
                    "Failure Stage": item.get("failure_stage", ""),
                    "Failure Reason": item.get("failure_reason", ""),
                    "Exclude From Ranking Eval": 1,
                    "Expected Candidate Count": item.get("expected_candidate_count", ""),
                    "Actual Candidate Count": item.get("actual_candidate_count", ""),
                    "Returned Candidate Count": item.get("returned_candidate_count", ""),
                    "Expected API Count": item.get("expected_api_count", ""),
                    "Actual API Count": item.get("actual_api_count", ""),
                    "Returned API Count": item.get("returned_api_count", ""),
                    "Duplicate Candidate IDs": _format_list_field(item.get("duplicate_candidate_ids")),
                    "Duplicate API IDs": _format_list_field(item.get("duplicate_api_ids")),
                    "Missing Candidate IDs": _format_list_field(item.get("missing_candidate_ids")),
                    "Missing API IDs": _format_list_field(item.get("missing_api_ids")),
                    "Unknown Candidate IDs": _format_list_field(item.get("unknown_candidate_ids")),
                    "Unknown API IDs": _format_list_field(item.get("unknown_api_ids")),
                    "Ranking Anomaly": 1 if _truthy(item.get("ranking_anomaly")) else 0,
                    "Ranking Anomaly Reason": item.get("ranking_anomaly_reason", ""),
                    "QoS_RT": qos.get("rt_ms"),
                    "QoS_TP": qos.get("tp_rps"),
                    "QoS Availability": qos.get("availability"),
                    "Comments": rel_info.get("comment", item.get("failure_reason", "invalid ranking case")),
                })
                continue
            rel_info = cache.get(
                f"{query_id}_{sid}_{api_id}",
                {"relevant": 0, "comment": "Missing from retrieval-stage functional match cache"},
            )
            functional_match = _functional_match_value(rel_info)
            planner_k = int(planner_k_by_subtask.get(sid, 0) or 0)
            planner_threshold = planner_k if planner_k > 0 else 5
            mode_rank = item.get("mode_rank")
            try:
                selected_for_planner = int(mode_rank or 0) <= planner_threshold
            except Exception:
                selected_for_planner = False
            service = item.get("service") or {}
            catalog_entry = catalog.get(api_id, {})
            if isinstance(service.get("qos"), dict):
                qos = service.get("qos") or {}
            elif isinstance(catalog_entry.get("qos"), dict):
                qos = catalog_entry.get("qos") or {}
            else:
                qos = {}
            rows.append({
                "Query_ID": query_id,
                "Mode": mode,
                "Sub Task": sid,
                "Retrieved Rank": item.get("retrieved_rank"),
                "Mode Rank": item.get("mode_rank"),
                "LLM Reported Rank": item.get("llm_reported_rank", ""),
                "Subtask_Purpose": purpose,
                "Candidate_ID": item.get("candidate_id", ""),
                "Selected_API": api_id,
                "Ranker Reason": item.get("reason", ""),
                "Is Hallucinated? (0/1)": hallucination_flags.get((mode, sid, api_id), 0),
                "Is Duplicated? (0/1)": duplicate_flags.get((mode, sid, api_id), 0),
                "Functional Match (0/1)": functional_match,
                "Used in Ranking": "Yes" if mode == "qos_hybrid" else "No",
                "Selected for Planner": "Yes" if selected_for_planner else "No",
                "Planner Selection K": planner_k,
                "Failure Flag": 0,
                "Failure Stage": "",
                "Failure Reason": "",
                "Exclude From Ranking Eval": 0,
                "Expected Candidate Count": "",
                "Actual Candidate Count": "",
                "Returned Candidate Count": "",
                "Expected API Count": "",
                "Actual API Count": "",
                "Returned API Count": "",
                "Duplicate Candidate IDs": _format_list_field(item.get("duplicate_candidate_ids")),
                "Duplicate API IDs": _format_list_field(item.get("duplicate_api_ids")),
                "Missing Candidate IDs": _format_list_field(item.get("missing_candidate_ids")),
                "Missing API IDs": _format_list_field(item.get("missing_api_ids")),
                "Unknown Candidate IDs": _format_list_field(item.get("unknown_candidate_ids")),
                "Unknown API IDs": _format_list_field(item.get("unknown_api_ids")),
                "Ranking Anomaly": 1 if _truthy(item.get("ranking_anomaly")) else 0,
                "Ranking Anomaly Reason": item.get("ranking_anomaly_reason", ""),
                "QoS_RT": qos.get("rt_ms"),
                "QoS_TP": qos.get("tp_rps"),
                "QoS Availability": qos.get("availability"),
                "Comments": rel_info.get("comment", ""),
            })

    rows.sort(key=lambda r: (int(str(r["Sub Task"])) if str(r["Sub Task"]).isdigit() else 9999, MODE_ORDER.get(str(r["Mode"]), 999), int(r.get("Mode Rank") or 9999)))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_xlsx = output_dir / f"query_{query_id}_candidate_api_rankings.xlsx"
    write_candidate_api_rankings_excel(rows, out_xlsx)
    (output_dir / f"query_{query_id}_candidate_api_rankings_rows.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / f"query_{query_id}_candidate_api_rankings_summary.json").write_text(
        json.dumps(
            {
                "query_id": query_id,
                "excel": str(out_xlsx),
                "rows_json": str(output_dir / f"query_{query_id}_candidate_api_rankings_rows.json"),
                "source": "retrieval_functional_match_cache",
                "cache": str(cache_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_xlsx
