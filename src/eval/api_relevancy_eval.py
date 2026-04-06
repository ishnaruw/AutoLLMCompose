from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.core.retry import call_with_backoff
from src.eval.api_relevancy_excel import write_relevancy_excel
from src.eval.api_relevancy_prompt import build_llm_prompt
from src.llm.backends import make_backend

CATALOG_WITH_QOS_PATH = Path("data/processed/api_catalog_sample_balanced/api_repo.with_qos.jsonl")
CATALOG_NO_QOS_PATH = Path("data/processed/api_catalog_sample_balanced/api_repo.no_qos.jsonl")
CACHE_PATH = Path("results/relevancy_eval/relevancy_cache.json")
OUT_DIR = Path("results/relevancy_eval")
MODE_DIRS = ["no_qos", "qos_pure_llm", "qos_topsis"]
CHUNK_SIZE = 5

EVAL_SYS = (
    "You are a strict API relevance evaluator. "
    "Decide only whether an API is functionally relevant to a subtask. "
    "Return strict JSON only."
)


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


def _merge_catalogs() -> Dict[str, Dict[str, Any]]:
    with_qos = _load_jsonl_catalog(CATALOG_WITH_QOS_PATH)
    no_qos = _load_jsonl_catalog(CATALOG_NO_QOS_PATH)

    merged: Dict[str, Dict[str, Any]] = {k: dict(v) for k, v in with_qos.items()}
    for api_id, obj in no_qos.items():
        if api_id in merged:
            qos = merged[api_id].get("qos")
            combined = dict(obj)
            if qos is not None:
                combined["qos"] = qos
            merged[api_id] = combined
        else:
            merged[api_id] = dict(obj)
    return merged


def _selected_files_for_mode(mode_dir: Path) -> List[Path]:
    files = sorted(mode_dir.glob("*selected*.json"))
    preferred = [p for p in files if p.name.startswith("3_selected") or p.name.startswith("4_selected")]
    return preferred if preferred else files


def _load_selected_files(query_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for mode in MODE_DIRS:
        mode_dir = query_dir / mode
        rows: List[Dict[str, Any]] = []
        for path in _selected_files_for_mode(mode_dir):
            data = _safe_read_json(path) or []
            if isinstance(data, list):
                rows.extend(x for x in data if isinstance(x, dict))
        out[mode] = rows
    return out


def _load_subtasks(query_dir: Path) -> List[Dict[str, Any]]:
    data = _safe_read_json(query_dir / "0_decomposer.json") or []
    return data if isinstance(data, list) else []


def _load_meta(query_dir: Path) -> Dict[str, Any]:
    data = _safe_read_json(query_dir / "meta.json") or {}
    return data if isinstance(data, dict) else {}


def get_expected_function(subtask_description: str) -> str:
    s = (subtask_description or "").lower()
    rules = [
        (["location", "geolocation", "reverse geocoding", "coordinates"], "Retrieve or resolve user location information"),
        (["restaurant", "nearby", "search"], "Search and retrieve restaurants based on user context or location"),
        (["review", "rating", "feedback"], "Retrieve ratings, reviews, or feedback relevant to the target item"),
        (["course", "courses", "recommend"], "Search, retrieve, or recommend relevant courses"),
        (["email", "remind", "notification", "enrollment"], "Send or manage email reminders or notifications"),
        (["reservation", "booking", "schedule"], "Create, manage, or support reservation or scheduling actions"),
    ]
    for keys, value in rules:
        if any(k in s for k in keys):
            return value
    return f"Perform the core function described by the subtask: {subtask_description.strip()}"


def _load_cache(path: Path = CACHE_PATH) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_cache(cache: Dict[str, Dict[str, Any]], path: Path = CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


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


def _build_subtask_batches(
    query_id: str,
    main_task: str,
    subtasks: List[Dict[str, Any]],
    selected_by_mode: Dict[str, List[Dict[str, Any]]],
    catalog: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    batches: List[Dict[str, Any]] = []
    for sub in subtasks:
        sid = str(sub.get("id"))
        purpose = sub.get("description", "")
        expected = get_expected_function(purpose)
        api_map: Dict[str, Dict[str, Any]] = {}

        for _, rows in selected_by_mode.items():
            for row in rows:
                if str(row.get("subtask_id")) != sid:
                    continue
                api_id = str(row.get("api_id", ""))
                if not api_id:
                    continue
                if api_id not in api_map:
                    merged = _extract_api_info(row, catalog)
                    api_map[api_id] = {
                        "api_id": api_id,
                        "name": merged.get("name") or merged.get("title") or merged.get("operation"),
                        "category": merged.get("category"),
                        "description": merged.get("description") or merged.get("summary") or merged.get("desc"),
                        "method": merged.get("method"),
                        "url": merged.get("url") or merged.get("endpoint") or merged.get("path"),
                    }

        batches.append(
            {
                "query_id": query_id,
                "main_task": main_task,
                "subtask_id": sid,
                "subtask_description": purpose,
                "expected_function": expected,
                "apis": list(api_map.values()),
            }
        )
    return batches


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
        api_id = r.get("api_id")
        if api_id is None:
            continue
        api_id = str(api_id).strip()
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


def evaluate_query(
    *,
    query_dir: Path,
    query_id: Optional[str],
    provider: str,
    model: Optional[str] = None,
) -> Path:
    backend = make_backend(provider=provider, model=model)
    meta = _load_meta(query_dir)
    main_task = str(meta.get("user_goal") or "")
    query_id = query_id or str(meta.get("query_id") or query_dir.name)

    subtasks = _load_subtasks(query_dir)
    selected_by_mode = _load_selected_files(query_dir)
    catalog = _merge_catalogs()
    cache = _load_cache()

    rows: List[Dict[str, Any]] = []
    total_cached = 0
    llm_calls = 0
    parse_failures = 0
    missing_catalog = 0

    batches = _build_subtask_batches(query_id, main_task, subtasks, selected_by_mode, catalog)
    batch_results: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    debug_dir = OUT_DIR / "debug_raw"

    for batch in batches:
        sid = batch["subtask_id"]
        apis = batch["apis"]

        uncached: List[Dict[str, Any]] = []
        sub_results: Dict[str, Dict[str, Any]] = {}

        for api in apis:
            key = f"{query_id}_{sid}_{api['api_id']}"
            if key in cache:
                sub_results[api["api_id"]] = cache[key]
                total_cached += 1
            else:
                uncached.append(api)

        if uncached:
            debug_dir.mkdir(parents=True, exist_ok=True)
            for chunk_idx, chunk in enumerate(_chunk_list(uncached, CHUNK_SIZE), start=1):
                prompt = build_llm_prompt(
                    query_id=query_id,
                    main_task=batch["main_task"],
                    subtask_id=sid,
                    subtask_description=batch["subtask_description"],
                    expected_function=batch["expected_function"],
                    api_entries=chunk,
                )
                expected_ids = [a["api_id"] for a in chunk]

                def _call() -> str:
                    return backend.chat_json(EVAL_SYS, prompt, temperature=0, force_json=True)

                raw = call_with_backoff(_call, name=f"api_relevancy_eval_s{sid}_chunk{chunk_idx}")
                parsed = _parse_results(raw, expected_ids)
                llm_calls += 1

                if len(parsed) != len(expected_ids):
                    retry_raw = call_with_backoff(_call, name=f"api_relevancy_eval_retry_s{sid}_chunk{chunk_idx}")
                    retry_parsed = _parse_results(retry_raw, expected_ids)
                    llm_calls += 1
                    if len(retry_parsed) >= len(parsed):
                        raw = retry_raw
                        parsed = retry_parsed

                if len(parsed) != len(expected_ids):
                    raw_path = debug_dir / f"{query_id}_subtask_{sid}_chunk_{chunk_idx}.json"
                    raw_path.write_text(raw, encoding="utf-8")

                for api in chunk:
                    api_id = api["api_id"]
                    key = f"{query_id}_{sid}_{api_id}"
                    if api_id in parsed:
                        val = parsed[api_id]
                        cache[key] = val
                        sub_results[api_id] = val
                    else:
                        parse_failures += 1
                        sub_results[api_id] = {"relevant": 0, "comment": "Missing from LLM response"}

        batch_results[(query_id, sid)] = sub_results

    _save_cache(cache)

    for mode in MODE_DIRS:
        for item in selected_by_mode.get(mode, []):
            sid = str(item.get("subtask_id"))
            sub = next((s for s in subtasks if str(s.get("id")) == sid), {})
            purpose = sub.get("description", "")
            api_id = str(item.get("api_id", ""))

            api_info = _extract_api_info(item, catalog)
            if not api_info or not api_info.get("api_id"):
                missing_catalog += 1
                rel = 0
                comment = "Missing catalog entry"
                qos = {}
            else:
                rel_info = batch_results.get((query_id, sid), {}).get(
                    api_id,
                    {"relevant": 0, "comment": "Missing from LLM response"},
                )
                rel = rel_info.get("relevant", 0)
                comment = rel_info.get("comment", "")

                service = item.get("service") or {}
                catalog_entry = catalog.get(api_id, {})
                if isinstance(service.get("qos"), dict):
                    qos = service.get("qos") or {}
                elif isinstance(catalog_entry.get("qos"), dict):
                    qos = catalog_entry.get("qos") or {}
                else:
                    qos = {}

            rows.append(
                {
                    "Query_ID": query_id,
                    "Mode": mode,
                    "Sub Task": sid,
                    "Selected Rank": item.get("selected_rank") or item.get("rank"),
                    "Subtask_Purpose": purpose,
                    "Selected_API": api_id,
                    "API Relevancy (0/1)": rel,
                    "QoS_RT": qos.get("rt_ms"),
                    "QoS_TP": qos.get("tp_rps"),
                    "QoS Availability": qos.get("availability"),
                    "Comments": comment,
                }
            )

    rows.sort(
        key=lambda r: (
            r["Mode"],
            int(str(r["Sub Task"])) if str(r["Sub Task"]).isdigit() else 9999,
            int(r["Selected Rank"] or 9999),
        )
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_xlsx = OUT_DIR / f"query_{query_id}_api_relevancy.xlsx"
    write_relevancy_excel(rows, out_xlsx)

    summary = {
        "query_id": query_id,
        "query_dir": str(query_dir),
        "provider": provider,
        "model": backend.name(),
        "total_rows": len(rows),
        "unique_apis_evaluated": len({(r["Sub Task"], r["Selected_API"]) for r in rows}),
        "cached_results": total_cached,
        "llm_calls": llm_calls,
        "parsing_failures": parse_failures,
        "missing_catalog_entries": missing_catalog,
        "excel": str(out_xlsx),
    }
    (OUT_DIR / f"query_{query_id}_api_relevancy_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return out_xlsx


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate API relevancy for one query run directory")
    parser.add_argument("--query-dir", required=True, help="Path to a query run directory")
    parser.add_argument("--query-id", required=False, help="Optional query id override")
    args = parser.parse_args()

    provider = choose_provider_interactive()
    out = evaluate_query(query_dir=Path(args.query_dir), query_id=args.query_id, provider=provider)
    print(f"Saved Excel report to {out}")


if __name__ == "__main__":
    main()
