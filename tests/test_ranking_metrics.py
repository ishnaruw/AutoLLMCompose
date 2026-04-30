from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.eval.ranking_metrics import (
    FALLBACK_K,
    MODE_ORDER,
    PAIRWISE_AVAILABLE,
    STRICT_ALL_MODES,
    aggregate_matrices_with_counts,
    average_overlap,
    build_ranking_cases,
    jaccard_similarity,
    load_report_rows,
    rbo_score,
    spearman_full,
)


def _case_rows(relevant_count: int = 2, candidate_count: int = 5) -> pd.DataFrame:
    rows = []
    for mode in MODE_ORDER:
        for idx in range(1, candidate_count + 1):
            rows.append(
                {
                    "run_dir": "run/q01_example",
                    "report_path": "run/q01_example/evaluation/query_q01_candidate_api_rankings.xlsx",
                    "query_id": "q01",
                    "mode": mode,
                    "subtask_id": "1",
                    "api_id": f"api_{idx}",
                    "mode_rank": idx,
                    "functional_match_label": 1 if idx <= relevant_count else 0,
                    "exclude_from_ranking_eval": 0,
                    "failure_flag": 0,
                }
            )
    return pd.DataFrame(rows)


class RankingMetricTests(unittest.TestCase):
    def test_spearman_full_uses_complete_shared_ranking(self) -> None:
        self.assertEqual(spearman_full(["a", "b", "c"], ["a", "b", "c"]), 1.0)
        self.assertEqual(spearman_full(["a", "b", "c"], ["c", "b", "a"]), -1.0)

    def test_spearman_full_rejects_different_candidate_sets(self) -> None:
        with self.assertRaises(ValueError):
            spearman_full(["a", "b", "c"], ["d", "e", "f"])

    def test_top_k_metrics_cover_overlap_edge_cases(self) -> None:
        self.assertEqual(average_overlap(["a", "b", "c"], ["d", "e", "f"], k=3), 0.0)
        self.assertEqual(jaccard_similarity(["a", "b", "c"], ["d", "e", "f"], k=3), 0.0)
        self.assertEqual(rbo_score(["a", "b", "c"], ["a", "b", "c"], k=3, p=0.9), 1.0)
        self.assertGreater(rbo_score(["a", "b", "c"], ["c", "b", "a"], k=3, p=0.9), 0.8)

    def test_build_cases_uses_qos_hybrid_k_and_fallback(self) -> None:
        cases, warnings = build_ranking_cases(_case_rows(relevant_count=0, candidate_count=FALLBACK_K))

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].k, FALLBACK_K)
        self.assertTrue(cases[0].k_fallback_used)
        self.assertTrue(any("fallback" in warning.lower() for warning in warnings))

    def test_pairwise_available_keeps_valid_pairs_when_one_mode_is_invalid(self) -> None:
        df = _case_rows(relevant_count=2, candidate_count=5)
        invalid = df["mode"] == "no_qos"
        df.loc[invalid, "exclude_from_ranking_eval"] = 1
        df.loc[invalid, "failure_flag"] = 1
        df.loc[invalid, "failure_stage"] = "llm_ranking"
        df.loc[invalid, "failure_reason"] = "duplicate_ranked_apis_after_retries"

        cases, warnings = build_ranking_cases(df, inclusion_policy=PAIRWISE_AVAILABLE)
        matrices, counts = aggregate_matrices_with_counts(cases)

        self.assertEqual(len(cases), 1)
        self.assertNotIn("no_qos", cases[0].valid_modes)
        self.assertEqual(int(counts["jaccard"].loc["qos_pure_llm", "qos_hybrid"]), 1)
        self.assertEqual(int(counts["jaccard"].loc["no_qos", "qos_hybrid"]), 0)
        self.assertTrue(pd.isna(matrices["jaccard"].loc["no_qos", "qos_hybrid"]))
        self.assertTrue(any("pairwise_available" in warning for warning in warnings))

        strict_cases, strict_warnings = build_ranking_cases(df, inclusion_policy=STRICT_ALL_MODES)
        self.assertEqual(strict_cases, [])
        self.assertTrue(any("strict_all_modes" in warning for warning in strict_warnings))

    def test_load_report_rows_normalizes_excel_columns(self) -> None:
        rows = []
        for mode in MODE_ORDER:
            for idx in range(1, 3):
                rows.append(
                    {
                        "Query_ID": "q01",
                        "Mode": mode,
                        "Sub Task": 1,
                        "Selected_API": f"api_{idx}",
                        "Mode Rank": idx,
                        "Functional Match (0/1)": 1 if idx == 1 else 0,
                    }
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "query_q01_candidate_api_rankings.xlsx"
            pd.DataFrame(rows).to_excel(path, sheet_name="Ranked APIs", index=False)
            loaded = load_report_rows(path)

        self.assertEqual(set(["query_id", "mode", "subtask_id", "api_id", "mode_rank"]).issubset(loaded.columns), True)
        self.assertEqual(len(loaded), len(rows))
        self.assertEqual(set(loaded["mode"]), set(MODE_ORDER))


if __name__ == "__main__":
    unittest.main()
