# src/agents/functional_refiner.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.eval.functional_match_eval import evaluate_retrieval_functional_match


@dataclass
class FunctionalRefinerAgent:
    """
    The Functional Refiner Agent assigns binary functional suitability labels
    to retrieved API candidates. It does not rank or select APIs; the main
    pipeline consumes these labels as functional gates for QoS-aware ranking
    and selection, and also writes them into post-hoc evaluation artifacts.
    """

    name: str = "functional_refiner_agent"
    description: str = "LLM-assisted functional suitability labeling for retrieved API candidates"
    stage_name: str = "functional_refinement"
    legacy_stage_name: str = "retrieval_functional_match"

    def refine_candidates(
        self,
        *,
        query_dir: Path,
        query_id: Optional[str],
        provider: str,
        model: Optional[str],
        output_dir: Path,
        cache_path: Path,
    ) -> Path:
        """
        Label retrieved APIs with Functional Match values and return the rows JSON path.

        Existing output filenames are preserved for compatibility with the
        ranking, selection, planner, and evaluation code that already consumes
        retrieval_functional_match rows.
        """
        return evaluate_retrieval_functional_match(
            query_dir=query_dir,
            query_id=query_id,
            provider=provider,
            model=model,
            output_dir=output_dir,
            cache_path=cache_path,
            stage_name=self.stage_name,
            progress_filename="functional_refinement_progress.json",
            summary_metadata={
                "agent": self.__class__.__name__,
                "agent_name": self.name,
                "stage": self.stage_name,
                "legacy_stage": self.legacy_stage_name,
                "agent_mode": "llm_binary_functional_labeling",
            },
            write_functional_refinement_summary=True,
        )
