from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.ui.composition_visualization_helpers import (
    build_dataflow_cards_html,
    build_bottleneck_replacement_simulations,
    build_replacement_simulation_dot,
    build_winner_heatmap,
    api_health_status,
    compute_sensitivity_scores,
    detect_invalid_workflow_issues,
    enrich_workflow_for_selection,
    get_recommended_mode,
    group_bottlenecks_by_api,
    normalize_workflow_api_qos_scores_by_subtask_candidates,
    normalize_api_qos_scores,
    recommended_summary_rows,
    selected_by_other_modes,
    simulation_metric_rows,
)


class CompositionVisualizationRecommendationTests(unittest.TestCase):
    def test_selects_highest_valid_qos_adjusted_score(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "Query_ID": "q1",
                    "Mode": "qos_hybrid",
                    "Composition_Validity": 1,
                    "QoS_Adjusted_Composition_Score": 0.756,
                    "Functional_Coverage": 1.0,
                },
                {
                    "Query_ID": "q1",
                    "Mode": "qos_topsis",
                    "Composition_Validity": 1,
                    "QoS_Adjusted_Composition_Score": 0.925,
                    "Functional_Coverage": 0.75,
                },
            ]
        )

        recommendation = get_recommended_mode(df, "q1")

        self.assertEqual(recommendation["status"], "recommended")
        self.assertEqual(recommendation["mode"], "qos_topsis")
        self.assertIn("qos_hybrid has higher functional coverage", recommendation["tradeoff_message"])

    def test_uses_tie_breakers_after_score(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "Query_ID": "q1",
                    "Mode": "qos_pure_llm",
                    "Composition_Validity": 1,
                    "QoS_Adjusted_Composition_Score": 0.8,
                    "Functional_Coverage": 0.75,
                    "Composition_Completeness": 1.0,
                    "Normalized_QoS_Score": 0.7,
                    "Total_Response_Time": 15,
                },
                {
                    "Query_ID": "q1",
                    "Mode": "qos_topsis",
                    "Composition_Validity": 1,
                    "QoS_Adjusted_Composition_Score": 0.8,
                    "Functional_Coverage": 0.75,
                    "Composition_Completeness": 1.0,
                    "Normalized_QoS_Score": 0.7,
                    "Total_Response_Time": 10,
                },
            ]
        )

        recommendation = get_recommended_mode(df, "q1")

        self.assertEqual(recommendation["mode"], "qos_topsis")

    def test_missing_scores_are_unavailable(self) -> None:
        df = pd.DataFrame(
            [
                {"Query_ID": "q1", "Mode": "qos_hybrid", "Composition_Validity": 1},
                {"Query_ID": "q1", "Mode": "qos_topsis", "Composition_Validity": 1},
            ]
        )

        recommendation = get_recommended_mode(df, "q1")

        self.assertEqual(recommendation["status"], "unavailable")
        self.assertEqual(recommendation["mode"], "N/A")
        self.assertIn("scores are missing", recommendation["warning"])

    def test_invalid_only_rows_are_diagnostic(self) -> None:
        df = pd.DataFrame(
            [
                {"Query_ID": "q1", "Mode": "no_qos", "Composition_Validity": 0, "QoS_Adjusted_Composition_Score": 0.2},
                {"Query_ID": "q1", "Mode": "qos_hybrid", "Composition_Validity": 0, "QoS_Adjusted_Composition_Score": 0.4},
            ]
        )

        recommendation = get_recommended_mode(df, "q1")

        self.assertEqual(recommendation["status"], "diagnostic")
        self.assertEqual(recommendation["mode"], "qos_hybrid")

    def test_api_health_uses_local_normalized_qos_scores(self) -> None:
        workflow = normalize_api_qos_scores(
            pd.DataFrame(
                [
                    {"Functional_Match": 1, "rt_ms": 0.4, "tp_rps": 20, "availability": 0.99},
                    {"Functional_Match": 1, "rt_ms": 0.8, "tp_rps": 10, "availability": 0.95},
                    {"Functional_Match": 1, "rt_ms": 1.2, "tp_rps": 1, "availability": 0.90},
                ]
            )
        )

        strong = api_health_status(workflow.iloc[0])
        moderate = api_health_status(workflow.iloc[1])
        weak = api_health_status(workflow.iloc[2])

        self.assertEqual(strong[0], "green")
        self.assertEqual(moderate[0], "orange")
        self.assertEqual(weak[0], "red")
        self.assertAlmostEqual(workflow.iloc[0]["API_Selection_Health"], 1.0)

    def test_api_health_handles_single_api_and_missing_qos(self) -> None:
        single = normalize_api_qos_scores(pd.DataFrame([{"Functional_Match": 1, "rt_ms": 0.5, "tp_rps": 7, "availability": 0.98}]))
        missing = normalize_api_qos_scores(pd.DataFrame([{"Functional_Match": 1}]))

        self.assertEqual(single.iloc[0]["API_QoS_Health"], 1.0)
        self.assertEqual(single.iloc[0]["API_Risk_Label"], "Low")
        self.assertTrue(pd.isna(missing.iloc[0]["API_QoS_Health"]))
        self.assertEqual(missing.iloc[0]["API_Risk_Label"], "Unknown")

    def test_enriched_workflow_health_uses_same_subtask_candidate_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "evaluation").mkdir(parents=True)
            candidate_rows = [
                {"Query_ID": "q1", "Mode": "qos_hybrid", "Sub Task": "1", "Selected_API": "geo_selected", "Functional Match (0/1)": 1, "QoS_RT": 10.0, "QoS_TP": 1.0, "QoS Availability": 0.8},
                {"Query_ID": "q1", "Mode": "qos_hybrid", "Sub Task": "1", "Selected_API": "geo_alt", "Functional Match (0/1)": 1, "QoS_RT": 1.0, "QoS_TP": 10.0, "QoS Availability": 0.99},
                {"Query_ID": "q1", "Mode": "qos_hybrid", "Sub Task": "2", "Selected_API": "sms_selected", "Functional Match (0/1)": 1, "QoS_RT": 100.0, "QoS_TP": 1.0, "QoS Availability": 0.9},
                {"Query_ID": "q1", "Mode": "qos_hybrid", "Sub Task": "2", "Selected_API": "sms_alt", "Functional Match (0/1)": 1, "QoS_RT": 200.0, "QoS_TP": 1.0, "QoS Availability": 0.9},
            ]
            (run_dir / "evaluation" / "query_q1_candidate_api_rankings_rows.json").write_text(json.dumps(candidate_rows), encoding="utf-8")
            workflow_df = pd.DataFrame(
                [
                    {"Query_ID": "q1", "Mode": "qos_hybrid", "Step": 1, "Subtask_ID": "1", "API_ID": "geo_selected"},
                    {"Query_ID": "q1", "Mode": "qos_hybrid", "Step": 2, "Subtask_ID": "2", "API_ID": "sms_selected"},
                ]
            )

            workflow, _ = enrich_workflow_for_selection(
                workflow_df,
                query_id="q1",
                run_dir=run_dir,
                run_name="run",
                mode="qos_hybrid",
            )

            health_by_api = workflow.set_index("API_ID")["API_QoS_Health"].to_dict()
            self.assertAlmostEqual(health_by_api["geo_selected"], 0.0)
            self.assertAlmostEqual(health_by_api["sms_selected"], 1.0)
            self.assertEqual(set(workflow["API_QoS_Health_Source"]), {"Subtask candidate pool"})

    def test_subtask_health_fallback_does_not_cross_normalize_without_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows = pd.DataFrame(
                [
                    {"Query_ID": "q1", "Mode": "qos_hybrid", "Subtask_ID": "1", "API_ID": "api_a", "Functional_Match": 1, "TOPSIS_Score": 0.55},
                    {"Query_ID": "q1", "Mode": "qos_hybrid", "Subtask_ID": "2", "API_ID": "api_b", "Functional_Match": 1, "rt_ms": 1.0, "tp_rps": 10.0, "availability": 0.99},
                ]
            )

            scored = normalize_workflow_api_qos_scores_by_subtask_candidates(
                rows,
                run_dir=Path(tmp),
                query_id="q1",
                mode="qos_hybrid",
            )

            self.assertAlmostEqual(scored.iloc[0]["API_QoS_Health"], 0.55)
            self.assertEqual(scored.iloc[0]["API_QoS_Health_Source"], "Fallback: TOPSIS score")
            self.assertEqual(scored.iloc[0]["API_QoS_Health_Warning"], "Subtask-level candidate pool unavailable; API QoS Health could not be computed reliably.")
            self.assertTrue(pd.isna(scored.iloc[1]["API_QoS_Health"]))
            self.assertEqual(scored.iloc[1]["API_Health_Status"], "gray")

    def test_workflow_availability_summary_uses_product_fallback(self) -> None:
        workflow = pd.DataFrame(
            [
                {"Subtask_ID": "1", "availability": 0.9},
                {"Subtask_ID": "2", "availability": 0.8},
            ]
        )

        rows = recommended_summary_rows(
            {"Composition_Validity": 1, "Total_Response_Time": 0.3},
            workflow,
            pd.DataFrame(),
            mode="qos_topsis",
        )
        summary = {row["Metric"]: row["Value"] for row in rows}

        self.assertEqual(summary["Workflow Availability"], "0.720")
        self.assertEqual(summary["Total Response Time (s)"], "0.300 s")

    def test_groups_bottlenecks_by_api_with_severity(self) -> None:
        workflow = pd.DataFrame(
            [
                {"Subtask_ID": "1", "API_ID": "api_a", "API_Name": "API A", "rt_ms": 0.1, "tp_rps": 10.0, "availability": 0.99},
                {"Subtask_ID": "2", "API_ID": "api_b", "API_Name": "API B", "rt_ms": 0.9, "tp_rps": 2.0, "availability": 0.90},
            ]
        )
        bottlenecks = pd.DataFrame(
            [
                {"Bottleneck_Type": "Latency", "API": "API B", "API_ID": "api_b", "Subtask_ID": "2", "Reason": "Highest response time"},
                {"Bottleneck_Type": "Throughput", "API": "API B", "API_ID": "api_b", "Subtask_ID": "2", "Reason": "Lowest throughput"},
                {"Bottleneck_Type": "Availability", "API": "API B", "API_ID": "api_b", "Subtask_ID": "2", "Reason": "Lowest availability"},
            ]
        )

        groups = group_bottlenecks_by_api(workflow, bottlenecks)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["api_name"], "API B")
        self.assertEqual(groups[0]["dimensions"], ["Latency", "Throughput", "Availability"])
        self.assertAlmostEqual(groups[0]["severity"]["latency_contribution_pct"], 90.0)
        self.assertAlmostEqual(groups[0]["severity"]["throughput_gap_pct"], 80.0)
        self.assertAlmostEqual(groups[0]["severity"]["availability_gap_pct"], 9.090909, places=5)

    def test_bottleneck_replacement_simulation_uses_same_subtask_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "evaluation").mkdir(parents=True)
            (run_dir / "qos_hybrid").mkdir()
            candidate_rows = [
                {
                    "Query_ID": "q1",
                    "Mode": "qos_hybrid",
                    "Sub Task": "2",
                    "Selected_API": "api_b",
                    "Functional Match (0/1)": 1,
                    "Mode Rank": 2,
                    "QoS_RT": 0.9,
                    "QoS_TP": 2.0,
                    "QoS Availability": 0.9,
                },
                {
                    "Query_ID": "q1",
                    "Mode": "qos_hybrid",
                    "Sub Task": "2",
                    "Selected_API": "api_c",
                    "Functional Match (0/1)": 1,
                    "Mode Rank": 1,
                    "QoS_RT": 0.2,
                    "QoS_TP": 9.0,
                    "QoS Availability": 0.99,
                },
            ]
            (run_dir / "evaluation" / "query_q1_candidate_api_rankings_rows.json").write_text(json.dumps(candidate_rows), encoding="utf-8")
            ranked_rows = [
                {"api_id": "api_c", "mode_rank": 1, "service": {"name": "Replacement API", "tool_name": "Weather Tool", "qos": {"rt_ms": 0.2, "tp_rps": 9.0, "availability": 0.99}}},
            ]
            (run_dir / "qos_hybrid" / "2_ranked_s2.json").write_text(json.dumps(ranked_rows), encoding="utf-8")
            workflow = normalize_api_qos_scores(
                pd.DataFrame(
                    [
                        {"Query_ID": "q1", "Mode": "qos_hybrid", "Step": 1, "Subtask_ID": "1", "API_ID": "api_a", "API_Name": "API A", "Functional_Match": 1, "rt_ms": 0.1, "tp_rps": 10.0, "availability": 0.99},
                        {"Query_ID": "q1", "Mode": "qos_hybrid", "Step": 2, "Subtask_ID": "2", "API_ID": "api_b", "API_Name": "API B", "Functional_Match": 1, "rt_ms": 0.9, "tp_rps": 2.0, "availability": 0.90},
                    ]
                )
            )
            bottlenecks = pd.DataFrame(
                [
                    {"Bottleneck_Type": "Latency", "API": "API B", "API_ID": "api_b", "Subtask_ID": "2", "Reason": "Highest response time"},
                    {"Bottleneck_Type": "Throughput", "API": "API B", "API_ID": "api_b", "Subtask_ID": "2", "Reason": "Lowest throughput"},
                ]
            )

            simulations = build_bottleneck_replacement_simulations(
                workflow=workflow,
                bottlenecks=bottlenecks,
                eval_row={"Composition_Validity": 1, "Composition_Completeness": 1.0},
                run_dir=run_dir,
                query_id="q1",
                mode="qos_hybrid",
            )

            self.assertEqual(simulations[0]["status"], "ok")
            self.assertEqual(simulations[0]["replacement_row"]["API_ID"], "api_c")
            self.assertLess(simulations[0]["simulated_metrics"]["Total_Response_Time"], simulations[0]["current_metrics"]["Total_Response_Time"])
            self.assertGreater(simulations[0]["simulated_metrics"]["Bottleneck_Throughput"], simulations[0]["current_metrics"]["Bottleneck_Throughput"])

    def test_what_if_replacement_labels_avoid_recommendation_language(self) -> None:
        rows = simulation_metric_rows(
            {"Total_Response_Time": 2.0, "Bottleneck_Throughput": 1.0, "Workflow_Availability": 0.8, "Functional_Coverage": 1.0},
            {"Total_Response_Time": 1.0, "Bottleneck_Throughput": 2.0, "Workflow_Availability": 0.9, "Functional_Coverage": 1.0},
        )
        self.assertIn("Official Planner Workflow", rows[0])
        self.assertIn("What-If Replacement Workflow", rows[0])
        self.assertNotIn("Current Workflow", rows[0])
        self.assertNotIn("Simulated Replacement", rows[0])

        workflow = pd.DataFrame(
            [
                {"Subtask_ID": "1", "API_ID": "api_a", "API_Name": "Official API"},
                {"Subtask_ID": "2", "API_ID": "api_b", "API_Name": "Other API"},
            ]
        )
        what_if_workflow = pd.DataFrame(
            [
                {"Subtask_ID": "1", "API_ID": "api_c", "API_Name": "Candidate API"},
                {"Subtask_ID": "2", "API_ID": "api_b", "API_Name": "Other API"},
            ]
        )
        dot = build_replacement_simulation_dot(
            workflow,
            what_if_workflow,
            {"Subtask_ID": "1", "API_ID": "api_a"},
            {"Subtask_ID": "1", "API_ID": "api_c"},
        )

        self.assertIn("Official Planner Workflow", dot)
        self.assertIn("What-If Replacement Workflow", dot)
        self.assertIn("Current bottleneck", dot)
        self.assertIn("Candidate replacement", dot)
        self.assertNotIn("Simulated Workflow", dot)
        self.assertNotIn("Suggested Replacement", dot)

    def test_selected_by_other_modes_detects_same_subtask_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "qos_hybrid").mkdir()
            (run_dir / "qos_topsis").mkdir()
            (run_dir / "qos_topsis" / "3_selected_s1.json").write_text(
                json.dumps([{"subtask_id": "1", "api_id": "api_c"}]),
                encoding="utf-8",
            )

            modes = selected_by_other_modes(run_dir, current_mode="qos_hybrid", subtask_id="1", api_id="api_c")

            self.assertEqual(modes, ["qos_topsis"])

    def test_winner_heatmap_prefers_valid_workflows_and_marks_invalid_only(self) -> None:
        rows = pd.DataFrame(
            [
                {"Query_ID": "q1", "Mode": "qos_hybrid", "Composition_Validity": 1, "QoS_Adjusted_Composition_Score": 0.7, "Total_Response_Time": 2.0},
                {"Query_ID": "q1", "Mode": "qos_topsis", "Composition_Validity": 1, "QoS_Adjusted_Composition_Score": 0.9, "Total_Response_Time": 3.0},
                {"Query_ID": "q2", "Mode": "qos_hybrid", "Composition_Validity": 0, "QoS_Adjusted_Composition_Score": 1.0, "Total_Response_Time": 0.1},
            ]
        )

        result = build_winner_heatmap(rows)
        winners = result["winners"].set_index("Query")

        self.assertEqual(winners.loc["q1", "Best QoS-adjusted score"], "qos_topsis")
        self.assertEqual(winners.loc["q1", "Lowest Total Response Time (s)"], "qos_hybrid")
        self.assertEqual(winners.loc["q2", "Best QoS-adjusted score"], "No valid mode")

    def test_sensitivity_scores_are_visualization_only(self) -> None:
        rows = pd.DataFrame(
            [
                {"Mode": "qos_hybrid", "Normalized_QoS_Score": 0.2, "Functional_Coverage": 1.0, "Composition_Completeness": 1.0, "Composition_Validity": 1, "QoS_Adjusted_Composition_Score": 0.9},
                {"Mode": "qos_topsis", "Normalized_QoS_Score": 1.0, "Functional_Coverage": 0.5, "Composition_Completeness": 1.0, "Composition_Validity": 1, "QoS_Adjusted_Composition_Score": 0.6},
            ]
        )

        scored = compute_sensitivity_scores(
            rows,
            {
                "QoS weight": 1.0,
                "Functional Coverage weight": 0.0,
                "Composition Completeness weight": 0.0,
                "Composition Validity weight": 0.0,
            },
        )

        self.assertEqual(scored.iloc[0]["Mode"], "qos_topsis")
        self.assertIn("Original_QoS_Adjusted_Composition_Score", scored.columns)
        self.assertEqual(rows.iloc[0]["QoS_Adjusted_Composition_Score"], 0.9)

    def test_dataflow_cards_use_planner_io_fields(self) -> None:
        workflow = pd.DataFrame(
            [
                {
                    "Step": 1,
                    "Subtask_ID": "1",
                    "API_ID": "geo",
                    "API_Name": "Geo API",
                    "Subtask": "Resolve the user location",
                    "Input_From_Previous_Step": "User location",
                    "Output_To_Next_Step": "Latitude and longitude",
                    "Action": "Resolve location",
                    "Why": "Long rationale should stay out of the graph cards.",
                }
            ]
        )

        html = build_dataflow_cards_html(query_context={"query_id": "q1", "goal": "Find local weather"}, workflow=workflow, mode="qos_hybrid")

        self.assertIn("maof-dataflow-wrap", html)
        self.assertIn("max-width: 850px", html)
        self.assertIn("Find local weather", html)
        self.assertIn("Geo API", html)
        self.assertIn("User location", html)
        self.assertIn("Latitude and longitude", html)
        self.assertIn("Final Planned Workflow", html)
        self.assertIn("↓", html)
        self.assertNotIn("Resolve location", html)
        self.assertNotIn("Long rationale", html)
        self.assertNotIn("digraph", html)

        detailed_html = build_dataflow_cards_html(
            query_context={"query_id": "q1", "goal": "Find local weather"},
            workflow=workflow,
            mode="qos_hybrid",
            detailed=True,
        )
        self.assertIn("Resolve the user location", detailed_html)
        self.assertIn("Resolve location", detailed_html)
        self.assertNotIn("Long rationale", detailed_html)

    def test_invalid_workflow_diagnostics_reports_missing_reason_and_step_warnings(self) -> None:
        workflow = pd.DataFrame(
            [
                {"Step": 1, "Subtask_ID": "1", "API_ID": "", "API_Name": "", "Functional_Match": 1},
                {"Step": 2, "Subtask_ID": "1", "API_ID": "api_b", "API_Name": "API B", "Functional_Match": 0},
            ]
        )

        issues = detect_invalid_workflow_issues(eval_row={"Composition_Validity": 0}, workflow=workflow)
        issue_names = {issue["Issue"] for issue in issues}

        self.assertIn("Workflow marked invalid, but detailed reason is unavailable.", issue_names)
        self.assertIn("missing_selected_api", issue_names)
        self.assertIn("duplicate_subtask", issue_names)
        self.assertIn("functional_mismatch", issue_names)


if __name__ == "__main__":
    unittest.main()
