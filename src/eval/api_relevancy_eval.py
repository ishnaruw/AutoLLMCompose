from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from src.core.retry import call_with_backoff
from src.eval.api_relevancy_excel import write_relevancy_excel
from src.eval.api_relevancy_prompt import build_llm_prompt
from src.llm.backends import make_backend

CATALOG_WITH_QOS_PATH = Path("data/processed/api_catalog_sample_balanced/api_repo.with_qos.jsonl")
DEFAULT_RUNS_ROOT = Path("results/logs/lmstudio_meta-llama-3.1-8b-instruct")
DEFAULT_OUTPUT_ROOT = Path("results/relevancy_eval_runs")
MODE_DIRS = ["no_qos", "qos_pure_llm", "qos_topsis"]
MODE_ORDER = {name: idx for idx, name in enumerate(MODE_DIRS)}
CHUNK_SIZE = 3
TOOLBENCH_TOOLS_ROOT = Path(os.getenv("TOOLBENCH_TOOLS_ROOT", "/Users/ishwaryapns/Documents/Thesis/ToolBench/data/toolenv/tools"))

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


def _tool_json_path(category: Optional[str], file_name: Optional[str]) -> Optional[Path]:
    if not category or not file_name:
        return None
    return TOOLBENCH_TOOLS_ROOT / str(category) / str(file_name)


def _truncate(s: Any, limit: int = 180) -> Any:
    if s is None:
        return None
    text = str(s).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _compact_params(params: Any, limit: int = 4) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(params, list):
        return out
    for p in params[:limit]:
        if not isinstance(p, dict):
            continue
        out.append(
            {
                "name": p.get("name"),
                "type": p.get("type"),
                "description": _truncate(p.get("description"), 100),
            }
        )
    return out


def _find_endpoint_detail(service: Dict[str, Any]) -> Dict[str, Any]:
    tool_path = _tool_json_path(service.get("category"), service.get("_file"))
    tool_json = _safe_read_json(tool_path) if tool_path else None
    if not isinstance(tool_json, dict):
        return {}

    api_list = tool_json.get("api_list")
    if not isinstance(api_list, list):
        return {}

    name = service.get("name")
    method = str(service.get("method") or "").upper()
    url = service.get("url")

    candidates = []
    for ep in api_list:
        if not isinstance(ep, dict):
            continue
        if name and ep.get("name") != name:
            continue
        candidates.append(ep)

    if method:
        method_matches = [ep for ep in candidates if str(ep.get("method") or "").upper() == method]
        if method_matches:
            candidates = method_matches
    if url:
        url_matches = [ep for ep in candidates if ep.get("url") == url]
        if url_matches:
            candidates = url_matches

    endpoint = candidates[0] if candidates else None
    if not isinstance(endpoint, dict):
        return {
            "tool_name": tool_json.get("tool_name") or tool_json.get("name") or tool_json.get("title"),
            "tool_description": _truncate(tool_json.get("tool_description"), 220),
        }

    response_hint = None
    body = endpoint.get("body")
    if isinstance(body, dict):
        response_hint = list(body.keys())[:6]
    elif isinstance(endpoint.get("schema"), dict):
        props = endpoint.get("schema", {}).get("properties")
        if isinstance(props, dict):
            response_hint = list(props.keys())[:6]

    detail: Dict[str, Any] = {
        "tool_name": tool_json.get("tool_name") or tool_json.get("name") or tool_json.get("title"),
        "tool_description": _truncate(tool_json.get("tool_description"), 220),
        "endpoint_details": {
            "required_parameters": _compact_params(endpoint.get("required_parameters"), 4),
            "optional_parameters": _compact_params(endpoint.get("optional_parameters"), 4),
        },
    }
    if response_hint:
        detail["endpoint_details"]["response_fields"] = response_hint
    return detail


def _extract_api_info(item: Dict[str, Any], catalog: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    api_id = str(item.get("api_id", ""))
    service = item.get("service") or {}
    catalog_entry = catalog.get(api_id, {})
    merged = dict(catalog_entry)
    merged.update({k: v for k, v in service.items() if v is not None})
    if "qos" not in merged and isinstance(catalog_entry.get("qos"), dict):
        merged["qos"] = catalog_entry.get("qos")
    merged.setdefault("api_id", api_id)
    detail = _find_endpoint_detail(merged)
    if detail.get("tool_name") and not merged.get("tool_name"):
        merged["tool_name"] = detail.get("tool_name")
    if detail.get("tool_description") and not merged.get("tool_description"):
        merged["tool_description"] = detail.get("tool_description")
    if detail.get("endpoint_details"):
        merged["endpoint_details"] = detail.get("endpoint_details")
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
                        "tool_name": merged.get("tool_name"),
                        "tool_description": merged.get("tool_description"),
                        "description": merged.get("description") or merged.get("summary") or merged.get("desc"),
                        "method": merged.get("method"),
                        "url": merged.get("url") or merged.get("endpoint") or merged.get("path"),
                        "endpoint_details": merged.get("endpoint_details") or {},
                    }

        batches.append(
            {
                "query_id": query_id,
                "main_task": main_task,
                "subtask_id": sid,
                "subtask_description": purpose,
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


def _query_num_from_name(name: str) -> Optional[int]:
    match = re.match(r"q0*(\d+)_", name)
    if match:
        return int(match.group(1))
    return None


def _normalize_query_token(token: str) -> Optional[int]:
    match = re.fullmatch(r"q?0*(\d+)", token.strip().lower())
    return int(match.group(1)) if match else None


def _parse_query_id_filters(tokens: Optional[List[str]]) -> Optional[Set[int]]:
    if not tokens:
        return None
    selected: Set[int] = set()
    for token in tokens:
        token = token.strip().lower()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start = _normalize_query_token(left)
            end = _normalize_query_token(right)
            if start is None or end is None:
                raise ValueError(f"Invalid query range: {token}")
            lo, hi = sorted((start, end))
            selected.update(range(lo, hi + 1))
        else:
            value = _normalize_query_token(token)
            if value is None:
                raise ValueError(f"Invalid query id: {token}")
            selected.add(value)
    return selected


def _discover_query_dirs(runs_root: Path, query_nums: Optional[Set[int]]) -> List[Path]:
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root not found: {runs_root.resolve()}")

    dirs = [p for p in runs_root.iterdir() if p.is_dir() and _query_num_from_name(p.name) is not None]
    dirs.sort(key=lambda p: (_query_num_from_name(p.name) or 9999, p.name))

    if query_nums is None:
        return dirs
    return [p for p in dirs if (_query_num_from_name(p.name) in query_nums)]


def _make_output_dir(base: Path, provider: str, model: Optional[str]) -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    model_part = re.sub(r"[^A-Za-z0-9._-]+", "_", model or "default")
    out = base / f"{provider}_{model_part}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def evaluate_query(
    *,
    query_dir: Path,
    query_id: Optional[str],
    provider: str,
    model: Optional[str] = None,
    output_dir: Path,
    cache_path: Path,
) -> Path:
    backend = make_backend(provider=provider, model=model)
    meta = _load_meta(query_dir)
    main_task = str(meta.get("user_goal") or "")
    query_id = query_id or str(meta.get("query_id") or query_dir.name)

    subtasks = _load_subtasks(query_dir)
    selected_by_mode = _load_selected_files(query_dir)
    catalog = _load_jsonl_catalog(CATALOG_WITH_QOS_PATH)
    cache = _load_cache(cache_path)

    rows: List[Dict[str, Any]] = []
    total_cached = 0
    llm_calls = 0
    parse_failures = 0
    missing_catalog = 0

    batches = _build_subtask_batches(query_id, main_task, subtasks, selected_by_mode, catalog)
    batch_results: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    debug_dir = output_dir / "debug"

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
                    api_entries=chunk,
                )
                expected_ids = [a["api_id"] for a in chunk]
                (debug_dir / f"{query_id}_s{sid}_chunk{chunk_idx}_prompt.txt").write_text(prompt, encoding="utf-8")

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

                (debug_dir / f"{query_id}_s{sid}_chunk{chunk_idx}_response.txt").write_text(raw, encoding="utf-8")
                (debug_dir / f"{query_id}_s{sid}_chunk{chunk_idx}_parsed.json").write_text(
                    json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8"
                )

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

    _save_cache(cache, cache_path)

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
            int(str(r["Sub Task"])) if str(r["Sub Task"]).isdigit() else 9999,
            MODE_ORDER.get(str(r["Mode"]), 999),
            int(r["Selected Rank"] or 9999),
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_xlsx = output_dir / f"query_{query_id}_api_relevancy.xlsx"
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
        "cache": str(cache_path),
    }
    (output_dir / f"query_{query_id}_api_relevancy_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return out_xlsx


def evaluate_many(
    *,
    query_dirs: List[Path],
    provider: str,
    model: Optional[str],
    output_dir: Path,
) -> List[Path]:
    cache_path = output_dir / "relevancy_cache.json"
    excels: List[Path] = []
    for query_dir in query_dirs:
        qnum = _query_num_from_name(query_dir.name)
        qid = f"q{qnum:02d}" if qnum is not None else query_dir.name
        print(f"Evaluating {qid} from {query_dir} ...")
        excels.append(
            evaluate_query(
                query_dir=query_dir,
                query_id=qid,
                provider=provider,
                model=model,
                output_dir=output_dir,
                cache_path=cache_path,
            )
        )
    overall = {
        "provider": provider,
        "model": model or "default_from_env",
        "runs_root": str(query_dirs[0].parent) if query_dirs else "",
        "query_count": len(query_dirs),
        "output_dir": str(output_dir),
        "excel_reports": [str(p) for p in excels],
    }
    (output_dir / "overall_summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    return excels


def main() -> None:
    parser = argparse.ArgumentParser(description="Run API relevancy evaluation on existing MAOF run logs")
    parser.add_argument("--provider", type=str, default=None, help="LLM provider (mistral, groq, azure_foundry, lmstudio)")
    parser.add_argument("--model", type=str, default=None, help="Optional model override for selected provider")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT, help="Root directory containing qXX_* run folders")
    parser.add_argument("--query-dir", type=Path, default=None, help="Evaluate exactly one query run directory")
    parser.add_argument("--query-ids", nargs="*", default=None, help="Query ids or ranges, e.g. 9 or 1-5 9 11-15")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Base directory for saving new evaluation results")
    args = parser.parse_args()

    provider = (args.provider or choose_provider_interactive()).strip().lower()
    output_dir = _make_output_dir(args.output_root, provider, args.model)

    if args.query_dir is not None:
        query_dirs = [args.query_dir]
    else:
        query_nums = _parse_query_id_filters(args.query_ids)
        query_dirs = _discover_query_dirs(args.runs_root, query_nums)

    if not query_dirs:
        raise RuntimeError("No query run directories found to evaluate")

    excels = evaluate_many(
        query_dirs=query_dirs,
        provider=provider,
        model=args.model,
        output_dir=output_dir,
    )

    print("Generated Excel files:")
    for path in excels:
        print(f"- {path}")
    print(f"Saved in: {output_dir}")


if __name__ == "__main__":
    main()
