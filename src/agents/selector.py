
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from src.eval.topsis_eval import _extract_qos, _run_topsis_pydecision


def _selector_mode() -> str:
    mode = (os.getenv("MAOF_SELECTOR_MODE", "PURE_LLM") or "PURE_LLM").strip().upper()
    if mode in {"TOPSIS", "TOPSIS_ENABLED"}:
        return "TOPSIS"
    return "PURE_LLM"


def _get_qos(service: Dict[str, Any]) -> Dict[str, Any]:
    qos = (service or {}).get("qos") or {}
    return qos if isinstance(qos, dict) else {}


def _truncate(s: Any, n: int) -> str:
    s = "" if s is None else str(s)
    s = " ".join(s.strip().split())
    return s if len(s) <= n else (s[: n - 1] + "…")


def _coerce_json(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "{}"
    try:
        json.loads(s)
        return s
    except Exception:
        pass
    m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.DOTALL)
    return m.group(1) if m else "{}"


def _derive_debug_path(path: str | None, suffix: str) -> str | None:
    if not path:
        return None
    p = Path(path)
    return str(p.with_name(f"{p.stem}_{suffix}{p.suffix}"))


def _band_lower_better(val: float, q1: float, q2: float) -> str:
    if val <= q1:
        return "low"
    if val <= q2:
        return "medium"
    return "high"


def _band_higher_better(val: float, q1: float, q2: float) -> str:
    if val >= q2:
        return "high"
    if val >= q1:
        return "medium"
    return "low"


def _compute_qos_labels(candidates: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    rows: List[Tuple[str, float, float, float]] = []
    for c in candidates:
        qos = _get_qos(c.get("service") or {})
        vec = _extract_qos(qos)
        if vec is None:
            continue
        rows.append((str(c.get("api_id")), float(vec[0]), float(vec[1]), float(vec[2])))

    if not rows:
        return {}

    rt = np.asarray([r[1] for r in rows], dtype=float)
    tp = np.asarray([r[2] for r in rows], dtype=float)
    av = np.asarray([r[3] for r in rows], dtype=float)

    rt_q1, rt_q2 = np.quantile(rt, [1/3, 2/3])
    tp_q1, tp_q2 = np.quantile(tp, [1/3, 2/3])
    av_q1, av_q2 = np.quantile(av, [1/3, 2/3])

    out: Dict[str, Dict[str, str]] = {}
    score_map = {
        "latency": {"low": 2, "medium": 1, "high": 0},
        "throughput": {"high": 2, "medium": 1, "low": 0},
        "availability": {"high": 2, "medium": 1, "low": 0},
    }

    for api_id, rt_v, tp_v, av_v in rows:
        lat = _band_lower_better(rt_v, float(rt_q1), float(rt_q2))
        thr = _band_higher_better(tp_v, float(tp_q1), float(tp_q2))
        ava = _band_higher_better(av_v, float(av_q1), float(av_q2))
        total = score_map["latency"][lat] + score_map["throughput"][thr] + score_map["availability"][ava]
        summary = "strong" if total >= 5 else ("moderate" if total >= 3 else "weak")
        out[api_id] = {
            "qos_latency_band": lat,
            "qos_throughput_band": thr,
            "qos_availability_band": ava,
            "qos_summary": summary,
        }
    return out


def _slim_candidate(
    c: Dict[str, Any],
    *,
    include_qos: bool,
    include_topsis: bool,
    qos_labels_by_id: Dict[str, Dict[str, str]] | None = None,
    deemphasize_rank: bool = False,
) -> Dict[str, Any]:
    service = c.get("service") or {}
    if not isinstance(service, dict):
        service = {}

    api_id = str(c.get("api_id"))
    out: Dict[str, Any] = {
        "api_id": c.get("api_id"),
        "category": _truncate(service.get("category") or c.get("category"), 80),
        "tool_name": _truncate(service.get("tool_name") or service.get("_tool"), 120),
        "tool_description": _truncate(service.get("tool_description"), 220),
        "name": _truncate(service.get("name"), 120),
        "description": _truncate(service.get("description"), 240),
        "method": _truncate(service.get("method"), 16),
        "url": _truncate(service.get("url"), 140),
        "rag_score": c.get("rag_score"),
    }

    if include_qos:
        qos = _get_qos(service)
        slim_qos = {}
        for k in ("rt_ms", "tp_rps", "availability"):
            if k in qos:
                slim_qos[k] = qos.get(k)
        if slim_qos:
            out["qos"] = slim_qos
        if qos_labels_by_id and api_id in qos_labels_by_id:
            out.update(qos_labels_by_id[api_id])

    if include_topsis and c.get("topsis_score") is not None:
        out["topsis_score"] = c.get("topsis_score")

    # Put ranker hints later so they remain available but less psychologically dominant.
    out["ranker_rank"] = c.get("rank")
    out["ranker_reason"] = _truncate(c.get("reason"), 120)
    if deemphasize_rank:
        out["ranker_hint"] = "useful semantic hint only"
    return out


def _compute_topsis_scores(candidates: List[Dict[str, Any]]) -> Tuple[Dict[str, float], int]:
    valid_rows: List[Tuple[str, List[float]]] = []
    for c in candidates:
        qos = _get_qos(c.get("service") or {})
        vec = _extract_qos(qos)
        if vec is None:
            continue
        valid_rows.append((str(c.get("api_id")), vec))

    if not valid_rows:
        return {}, 0

    X = np.asarray([v for _, v in valid_rows], dtype=float)
    scores, _ = _run_topsis_pydecision(X, [1.0, 1.0, 1.0])
    return {api_id: float(scores[i]) for i, (api_id, _) in enumerate(valid_rows)}, len(valid_rows)


def _parse_selected(raw: str) -> List[Dict[str, str]]:
    data = json.loads(_coerce_json(raw)) if raw else {}
    picked_raw = data.get("selected", [])
    picked: List[Dict[str, str]] = []
    if isinstance(picked_raw, list):
        for r in picked_raw:
            if not isinstance(r, dict) or not r.get("api_id"):
                continue
            picked.append({"api_id": str(r.get("api_id")), "selector_reason": r.get("reason", "") or ""})
    return picked


def _write_debug(path: str | None, content: str) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _single_pass_select(
    *,
    llm_call: Callable[[str], str],
    prompt: str,
    top_n: int,
    by_id: Dict[str, Dict[str, Any]],
    baseline: List[Dict[str, Any]],
    topsis_scores: Dict[str, float],
    debug_prompt_path: str | None,
    debug_raw_path: str | None,
    debug_parsed_path: str | None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    _write_debug(debug_prompt_path, prompt)
    raw = llm_call(prompt)
    _write_debug(debug_raw_path, raw or "")
    picked = _parse_selected(raw)
    _write_debug(debug_parsed_path, json.dumps({"selected": picked}, indent=2, ensure_ascii=False))

    selected: List[Dict[str, Any]] = []
    seen = set()
    for item in picked:
        api_id = item["api_id"]
        if api_id in seen or api_id not in by_id:
            continue
        row = dict(by_id[api_id])
        row["selector_reason"] = item.get("selector_reason", "")
        if api_id in topsis_scores:
            row["topsis_score"] = topsis_scores[api_id]
        selected.append(row)
        seen.add(api_id)
        if len(selected) >= top_n:
            break

    if len(selected) < top_n:
        for c in baseline:
            api_id = str(c.get("api_id"))
            if api_id in seen:
                continue
            row = dict(c)
            row.setdefault("selector_reason", "Fallback: selector filled remaining slots from fallback order.")
            if api_id in topsis_scores:
                row["topsis_score"] = topsis_scores[api_id]
            selected.append(row)
            seen.add(api_id)
            if len(selected) >= top_n:
                break
    return selected[:top_n], picked


def select_top_apis_for_subtask(
    *,
    llm_call: Callable[[str], str],
    user_query: str,
    subtask: Dict[str, Any],
    selector_candidates: List[Dict[str, Any]],
    mode_override: str | None = None,
    top_n: int = 10,
    topsis_top_k: int = 20,
    min_qos_candidates: int = 5,
    prompt_path: str = "prompts/selector_no_qos.md",
    debug_raw_path: str | None = None,
    debug_prompt_path: str | None = None,
    debug_parsed_path: str | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    mode = (mode_override.strip().upper() if isinstance(mode_override, str) and mode_override.strip() else _selector_mode())
    selector_candidates = [c for c in selector_candidates if isinstance(c, dict) and c.get("api_id")]

    fallback_order = sorted(
        selector_candidates,
        key=lambda x: (
            0 if x.get("rank") is not None else 1,
            int(x.get("rank") or 10**9),
            -(float(x.get("rag_score") or 0.0)),
        ),
    )
    baseline = fallback_order[:top_n]

    trace: Dict[str, Any] = {
        "subtask_id": str(subtask.get("id", "unknown")),
        "mode": mode,
        "top_n": top_n,
        "topsis_top_k": topsis_top_k,
        "min_qos_candidates": min_qos_candidates,
        "selector_candidates": len(selector_candidates),
    }

    include_qos = mode in {"PURE_LLM", "TOPSIS"} and any(_get_qos(c.get("service") or {}) for c in selector_candidates)
    include_topsis = mode in {"PURE_LLM", "TOPSIS"}

    topsis_scores: Dict[str, float] = {}
    qos_labels_by_id: Dict[str, Dict[str, str]] = {}
    qos_candidates = 0
    if include_qos:
        topsis_scores, qos_candidates = _compute_topsis_scores(selector_candidates[: max(topsis_top_k, top_n)])
        qos_labels_by_id = _compute_qos_labels(selector_candidates[: max(topsis_top_k, top_n)])
        trace["qos_candidates"] = qos_candidates
        trace["topsis_status"] = "ok" if qos_candidates >= min_qos_candidates else "insufficient_qos_candidates"

        # QoS-sensitive fallback order for qos_pure_llm and deterministic order for topsis.
        if topsis_scores:
            fallback_order = sorted(
                selector_candidates,
                key=lambda x: (
                    0 if str(x.get("api_id")) in topsis_scores else 1,
                    -(float(topsis_scores.get(str(x.get("api_id")), -1.0))),
                    0 if x.get("rank") is not None else 1,
                    int(x.get("rank") or 10**9),
                    -(float(x.get("rag_score") or 0.0)),
                ),
            )
            baseline = fallback_order[:top_n]

    # Deterministic TOPSIS baseline.
    if mode == "TOPSIS":
        selected: List[Dict[str, Any]] = []
        for c in baseline:
            row = dict(c)
            api_id = str(row.get("api_id"))
            if api_id in topsis_scores:
                row["topsis_score"] = topsis_scores[api_id]
            if api_id in qos_labels_by_id:
                row.update(qos_labels_by_id[api_id])
            row.setdefault("selector_reason", "Deterministic TOPSIS/QoS ordering baseline.")
            selected.append(row)
        trace["selected_by"] = "deterministic_topsis"
        trace["picked_count"] = len(selected)
        return selected[:top_n], trace

    by_id = {str(c.get("api_id")): c for c in selector_candidates}

    # Two-pass qos_pure_llm selector:
    # pass 1 = direct validity / usability shortlist (ignore QoS as much as possible)
    # pass 2 = QoS-aware ordering over the shortlisted valid candidates.
    if mode == "PURE_LLM" and include_qos:
        shortlist_n = min(max(top_n * 2, 12), len(selector_candidates))
        shortlist_payload = [
            _slim_candidate(c, include_qos=False, include_topsis=False, qos_labels_by_id=None, deemphasize_rank=True)
            for c in fallback_order
        ]
        pass1_prompt = (
            "You select a shortlist of functionally valid and directly usable APIs for one subtask.\n\n"
            "Priority order:\n"
            "1) Prefer direct functional match first.\n"
            "2) Prefer APIs that can immediately satisfy the subtask with minimal extra assumptions.\n"
            "3) If no API directly fulfills the subtask, prefer APIs that retrieve the core data required for the subtask.\n"
            "4) Ignore QoS except as a weak tie-break.\n\n"
            "Rules:\n"
            "- Avoid redundant APIs that duplicate the same retrieval function unless they provide clearly different coverage.\n"
            "- Do not select APIs whose relevance depends on invented intermediate steps.\n"
            "- Avoid APIs that only support implementation, storage, interface, orchestration, or unrelated side functions.\n"
            f"- Return up to {shortlist_n} api_id values.\n\n"
            f"User query:\n{user_query}\n\n"
            f"Subtask:\n{json.dumps(subtask, ensure_ascii=False)}\n\n"
            f"Candidates:\n{json.dumps(shortlist_payload, ensure_ascii=False)}\n\n"
            'Return JSON only:\n{"selected": [{"api_id": "...", "reason": "short reason"}]}'
        )
        shortlist_selected, picked1 = _single_pass_select(
            llm_call=llm_call,
            prompt=pass1_prompt,
            top_n=shortlist_n,
            by_id=by_id,
            baseline=fallback_order[:shortlist_n],
            topsis_scores=topsis_scores,
            debug_prompt_path=_derive_debug_path(debug_prompt_path, "pass1"),
            debug_raw_path=_derive_debug_path(debug_raw_path, "pass1"),
            debug_parsed_path=_derive_debug_path(debug_parsed_path, "pass1"),
        )
        shortlist_ids = [str(x.get("api_id")) for x in shortlist_selected if x.get("api_id")]
        shortlist_candidates = [by_id[api_id] for api_id in shortlist_ids if api_id in by_id]
        shortlist_baseline = sorted(
            shortlist_candidates,
            key=lambda x: (
                0 if str(x.get("api_id")) in topsis_scores else 1,
                -(float(topsis_scores.get(str(x.get("api_id")), -1.0))),
                0 if x.get("rank") is not None else 1,
                int(x.get("rank") or 10**9),
                -(float(x.get("rag_score") or 0.0)),
            ),
        )
        shortlist_payload2 = []
        for c in shortlist_baseline:
            row = dict(c)
            api_id = str(row.get("api_id"))
            if api_id in topsis_scores:
                row["topsis_score"] = topsis_scores[api_id]
            shortlist_payload2.append(
                _slim_candidate(
                    row,
                    include_qos=True,
                    include_topsis=(qos_candidates >= min_qos_candidates),
                    qos_labels_by_id=qos_labels_by_id,
                    deemphasize_rank=True,
                )
            )
        template = Path(prompt_path).read_text(encoding="utf-8")
        prompt2 = (
            template
            .replace("{user_query}", user_query)
            .replace("{subtask_json}", json.dumps(subtask, ensure_ascii=False))
            .replace("{top_n}", str(top_n))
            .replace("{candidates_json}", json.dumps(shortlist_payload2, ensure_ascii=False))
        )
        selected, picked2 = _single_pass_select(
            llm_call=llm_call,
            prompt=prompt2,
            top_n=top_n,
            by_id={str(c.get("api_id")): c for c in shortlist_candidates},
            baseline=shortlist_baseline[:top_n],
            topsis_scores=topsis_scores,
            debug_prompt_path=debug_prompt_path,
            debug_raw_path=debug_raw_path,
            debug_parsed_path=debug_parsed_path,
        )
        trace["selected_by"] = "llm_selector_two_pass_then_fallback"
        trace["shortlist_n"] = shortlist_n
        trace["pass1_picked_count"] = len(picked1)
        trace["picked_count"] = len(picked2)
        return selected[:top_n], trace

    selector_payload: List[Dict[str, Any]] = []
    for c in fallback_order:
        row = dict(c)
        api_id = str(row.get("api_id"))
        if api_id in topsis_scores:
            row["topsis_score"] = topsis_scores[api_id]
        selector_payload.append(
            _slim_candidate(
                row,
                include_qos=include_qos,
                include_topsis=(include_topsis and qos_candidates >= min_qos_candidates),
                qos_labels_by_id=qos_labels_by_id,
                deemphasize_rank=(mode == "PURE_LLM"),
            )
        )

    template = Path(prompt_path).read_text(encoding="utf-8")
    prompt = (
        template
        .replace("{user_query}", user_query)
        .replace("{subtask_json}", json.dumps(subtask, ensure_ascii=False))
        .replace("{top_n}", str(top_n))
        .replace("{candidates_json}", json.dumps(selector_payload, ensure_ascii=False))
    )
    selected, picked = _single_pass_select(
        llm_call=llm_call,
        prompt=prompt,
        top_n=top_n,
        by_id=by_id,
        baseline=baseline,
        topsis_scores=topsis_scores,
        debug_prompt_path=debug_prompt_path,
        debug_raw_path=debug_raw_path,
        debug_parsed_path=debug_parsed_path,
    )
    trace["selected_by"] = "llm_selector_then_fallback"
    trace["prompt_path"] = prompt_path
    trace["picked_count"] = len(picked)
    return selected[:top_n], trace
