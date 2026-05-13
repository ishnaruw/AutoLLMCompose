# src/agents/evaluator.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from src.eval.audit_api_duplicates import collect_duplicate_audit_for_run
from src.eval.audit_api_hallucinations import collect_hallucination_audit_for_run
from src.eval.composition_qos_eval import evaluate_composition_qos
from src.eval.functional_match_eval import evaluate_query, evaluate_retrieval_functional_match
from src.eval.mode_anomaly_report import collect_ranking_anomaly_audit_for_run, write_mode_anomaly_excel


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass
class EvaluationAgent:
    """
    Deterministic evaluation agent adapter.

    This adapter does not let an LLM decide metrics, labels, audits, or report
    structure. It preserves the existing evaluation functions and gives the
    evaluation stage an agent-shaped interface for the AutoGen pipeline.
    """
    catalog_no_qos_path: Path
    name: str = "evaluation_agent"
    description: str = "Deterministic MAOF evaluation and audit agent"

    def evaluate_retrieval_functional_match(
        self,
        *,
        query_dir: Path,
        query_id: Optional[str],
        provider: str,
        model: Optional[str],
        output_dir: Path,
        cache_path: Path,
    ) -> Path:
        return evaluate_retrieval_functional_match(
            query_dir=query_dir,
            query_id=query_id,
            provider=provider,
            model=model,
            output_dir=output_dir,
            cache_path=cache_path,
        )

    def build_evaluation_outputs(
        self,
        *,
        query_dir: Path,
        query_id: Optional[str],
        provider: str,
        model: Optional[str],
        output_dir: Path,
        cache_path: Path,
        retrieval_functional_match_rows_path: Path | None,
    ) -> Dict[str, Any]:
        candidate_api_rankings_excel = evaluate_query(
            query_dir=query_dir,
            query_id=query_id,
            provider=provider,
            model=model,
            output_dir=output_dir,
            cache_path=cache_path,
        )
        candidate_api_rankings_rows_path = output_dir / f"query_{query_id}_candidate_api_rankings_rows.json"

        duplicate_audit = collect_duplicate_audit_for_run(query_dir)
        duplicate_audit_json = output_dir / f"query_{query_id}_duplicate_audit.json"
        _write_json(duplicate_audit_json, duplicate_audit)

        hallucination_audit = collect_hallucination_audit_for_run(query_dir, self.catalog_no_qos_path)
        hallucination_audit_json = output_dir / f"query_{query_id}_hallucination_audit.json"
        _write_json(hallucination_audit_json, hallucination_audit)

        ranking_anomaly_audit = collect_ranking_anomaly_audit_for_run(query_dir, query_id=query_id)
        ranking_anomaly_audit_json = output_dir / f"query_{query_id}_ranking_anomaly_audit.json"
        _write_json(ranking_anomaly_audit_json, ranking_anomaly_audit)

        mode_anomaly_excel = output_dir / f"query_{query_id}_mode_anomalies.xlsx"
        write_mode_anomaly_excel(duplicate_audit, hallucination_audit, mode_anomaly_excel, ranking_anomaly_audit)

        return {
            "evaluation_dir": output_dir,
            "candidate_api_rankings_excel": candidate_api_rankings_excel,
            "candidate_api_rankings_rows_json": candidate_api_rankings_rows_path,
            "retrieval_functional_match_rows_json": retrieval_functional_match_rows_path,
            "duplicate_audit_json": duplicate_audit_json,
            "hallucination_audit_json": hallucination_audit_json,
            "ranking_anomaly_audit_json": ranking_anomaly_audit_json,
            "mode_anomaly_excel": mode_anomaly_excel,
            "cache_path": cache_path,
        }

    def evaluate_composition_qos(
        self,
        *,
        query_dir: Path,
        query_id: Optional[str],
        output_dir: Path,
    ) -> Dict[str, Any]:
        return evaluate_composition_qos(query_dir=query_dir, query_id=query_id, output_dir=output_dir)
