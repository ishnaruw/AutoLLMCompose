from __future__ import annotations

import json
from pathlib import Path

from src.ui import thesis_figure_generators as tg

import matplotlib.pyplot as plt
import pandas as pd


OFFICIAL_Q01_Q15_COMPOSITION = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "summaries"
    / "RUNS_MAY_31_NEW_5"
    / "fireworks_gpt-oss-120b"
    / "q01_q15_official"
    / "all_15_query_composition_results.csv"
)


def test_normalize_mode_order_prefers_research_order_and_keeps_unknown_modes() -> None:
    modes = ["experimental", "qos_topsis", "no_qos", "qos_hybrid", "qos_pure_llm"]

    assert tg.normalize_mode_order(modes) == [
        "no_qos",
        "qos_pure_llm",
        "qos_topsis",
        "qos_hybrid",
        "experimental",
    ]


def test_compute_unique_and_tied_best_counts_ties_with_tolerance() -> None:
    df = pd.DataFrame(
        [
            {"Query_ID": "q01", "Mode": "no_qos", "QoS_Adjusted_Composition_Score": 0.6},
            {"Query_ID": "q01", "Mode": "qos_pure_llm", "QoS_Adjusted_Composition_Score": 0.9},
            {"Query_ID": "q01", "Mode": "qos_topsis", "QoS_Adjusted_Composition_Score": 0.9000000005},
            {"Query_ID": "q02", "Mode": "no_qos", "QoS_Adjusted_Composition_Score": 0.8},
            {"Query_ID": "q02", "Mode": "qos_pure_llm", "QoS_Adjusted_Composition_Score": 0.7},
            {"Query_ID": "q02", "Mode": "qos_topsis", "QoS_Adjusted_Composition_Score": 0.6},
        ]
    )

    best_df, counts_df = tg.compute_unique_and_tied_best(
        df,
        "QoS_Adjusted_Composition_Score",
        tolerance=1e-9,
    )

    q01 = best_df[best_df["Query_ID"] == "q01"].iloc[0]
    assert q01["Best_Status"] == "tied best"
    assert q01["Best_Modes"] == "qos_pure_llm, qos_topsis"

    counts = counts_df.set_index("Mode")
    assert counts.loc["no_qos", "unique_best_count"] == 1
    assert counts.loc["qos_pure_llm", "tied_best_count"] == 1
    assert counts.loc["qos_topsis", "tied_best_count"] == 1


def test_recompute_score_sensitivity_uses_documented_formula() -> None:
    df = pd.DataFrame(
        [
            {
                "Query_ID": "q01",
                "Mode": "no_qos",
                "Composition_Completeness": 0.5,
                "Functional_Coverage": 0.8,
                "Normalized_QoS_Score": 0.4,
            }
        ]
    )

    scored = tg.recompute_score_sensitivity(df, alpha=0.75, beta=0.25)

    assert abs(scored.loc[0, "score_alpha_0.75_beta_0.25"] - 0.35) < 1e-12


def test_candidate_top10_sensitivity_deduplicates_and_counts_invalid_candidates() -> None:
    rows = pd.DataFrame(
        [
            {
                "query_id": "q01",
                "subtask_id": "1",
                "mode": "no_qos",
                "api_id": "api_a",
                "functional_match_label": 1,
                "QoS_RT_s": 10.0,
                "QoS_TP_kbps": 100.0,
                "QoS Availability": 0.9,
            },
            {
                "query_id": "q01",
                "subtask_id": "1",
                "mode": "qos_topsis",
                "api_id": "api_b",
                "functional_match_label": 0,
                "QoS_RT_s": 1.0,
                "QoS_TP_kbps": 10.0,
                "QoS Availability": 0.95,
            },
            {
                "query_id": "q01",
                "subtask_id": "1",
                "mode": "qos_hybrid",
                "api_id": "api_a",
                "functional_match_label": 1,
                "QoS_RT_s": 10.0,
                "QoS_TP_kbps": 100.0,
                "QoS Availability": 0.9,
            },
        ]
    )

    table, warnings = tg.build_candidate_top10_sensitivity(rows, ["q01"])

    assert not warnings
    assert not table.empty
    assert table["top10_candidate_count"].max() == 2
    assert table["invalid_top10_candidate_count"].max() == 1


def test_aggregate_figure_exports_nonempty_static_bytes() -> None:
    table = pd.DataFrame(
        [
            {
                "Mode": "no_qos",
                "Average_QoS_Adjusted_Composition_Score": 0.8,
                "Std_QoS_Adjusted_Composition_Score": 0.1,
            },
            {
                "Mode": "qos_pure_llm",
                "Average_QoS_Adjusted_Composition_Score": 0.9,
                "Std_QoS_Adjusted_Composition_Score": 0.05,
            },
            {
                "Mode": "qos_hybrid",
                "Average_QoS_Adjusted_Composition_Score": 0.99,
                "Std_QoS_Adjusted_Composition_Score": 0.08,
            },
        ]
    )

    fig = tg.plot_aggregate_score_by_mode(table)
    try:
        ax = fig.axes[0]
        assert [text.get_text() for text in ax.texts] == ["0.800", "0.900", "0.990"]
        assert [round(text.xy[1], 3) for text in ax.texts] == [0.9, 0.95, 1.07]
        assert all(text.get_va() == "bottom" for text in ax.texts)
        assert ax.get_ylim()[1] >= 1.13
        assert len(tg.figure_to_bytes(fig, "pdf")) > 1000
        assert len(tg.figure_to_bytes(fig, "png", dpi=300)) > 1000
    finally:
        plt.close(fig)


def test_figure_5_2_source_table_uses_official_order_and_values() -> None:
    aggregate = pd.DataFrame(
        [
            {
                "Mode": "no_qos",
                "Query_Count": 15,
                "Average_QoS_Adjusted_Composition_Score": 0.826604133333,
                "Std_QoS_Adjusted_Composition_Score": 0.0852521393321,
            },
            {
                "Mode": "qos_hybrid",
                "Query_Count": 15,
                "Average_QoS_Adjusted_Composition_Score": 0.990123533333,
                "Std_QoS_Adjusted_Composition_Score": 0.0233444382604,
            },
            {
                "Mode": "qos_topsis",
                "Query_Count": 15,
                "Average_QoS_Adjusted_Composition_Score": 0.404499,
                "Std_QoS_Adjusted_Composition_Score": 0.214329785633,
            },
            {
                "Mode": "qos_pure_llm",
                "Query_Count": 15,
                "Average_QoS_Adjusted_Composition_Score": 0.946672466667,
                "Std_QoS_Adjusted_Composition_Score": 0.0576958690931,
            },
        ]
    )

    table, warnings = tg.build_figure_5_2_average_score_table(aggregate)

    assert warnings == []
    assert table["Mode"].tolist() == ["qos_hybrid", "qos_pure_llm", "no_qos", "qos_topsis"]
    assert [f"{value:.3f}" for value in table["Average_QoS_Adjusted_Composition_Score"]] == [
        "0.990",
        "0.947",
        "0.827",
        "0.404",
    ]


def test_figure_5_2_plot_has_required_axis_range_error_bars_and_labels() -> None:
    table = pd.DataFrame(
        [
            {
                "Mode": "qos_hybrid",
                "Query_Count": 15,
                "Average_QoS_Adjusted_Composition_Score": 0.990123533333,
                "Std_QoS_Adjusted_Composition_Score": 0.0233444382604,
            },
            {
                "Mode": "qos_pure_llm",
                "Query_Count": 15,
                "Average_QoS_Adjusted_Composition_Score": 0.946672466667,
                "Std_QoS_Adjusted_Composition_Score": 0.0576958690931,
            },
            {
                "Mode": "no_qos",
                "Query_Count": 15,
                "Average_QoS_Adjusted_Composition_Score": 0.826604133333,
                "Std_QoS_Adjusted_Composition_Score": 0.0852521393321,
            },
            {
                "Mode": "qos_topsis",
                "Query_Count": 15,
                "Average_QoS_Adjusted_Composition_Score": 0.404499,
                "Std_QoS_Adjusted_Composition_Score": 0.214329785633,
            },
        ]
    )

    fig = tg.plot_figure_5_2_average_score_by_mode(table)
    try:
        ax = fig.axes[0]
        assert ax.get_title() == ""
        assert ax.get_ylim() == (0.0, 1.0)
        assert ax.get_xlabel() == "Evaluation Mode"
        assert ax.get_ylabel() == "Average QoS-Adjusted Composition Score"
        assert [text.get_text() for text in ax.texts] == ["0.990", "0.947", "0.827", "0.404"]
        assert ax.texts[0].get_position()[1] < 0.99
        assert ax.texts[0].get_color() == "white"
        assert len(ax.containers) >= 1
        assert len(tg.figure_to_bytes(fig, "png", dpi=300)) > 1000
    finally:
        plt.close(fig)


def test_figure_5_7_source_table_uses_official_points() -> None:
    aggregate = pd.DataFrame(
        [
            {
                "Mode": "qos_hybrid",
                "Query_Count": 15,
                "Average_Functional_Coverage": 1.000000,
                "Average_Normalized_QoS_Score": 0.967079,
            },
            {
                "Mode": "qos_topsis",
                "Query_Count": 15,
                "Average_Functional_Coverage": 0.238889,
                "Average_Normalized_QoS_Score": 0.790922,
            },
            {
                "Mode": "no_qos",
                "Query_Count": 15,
                "Average_Functional_Coverage": 0.955556,
                "Average_Normalized_QoS_Score": 0.525717,
            },
            {
                "Mode": "qos_pure_llm",
                "Query_Count": 15,
                "Average_Functional_Coverage": 1.000000,
                "Average_Normalized_QoS_Score": 0.822242,
            },
        ]
    )

    table, warnings = tg.build_figure_5_7_functional_vs_qos_table(aggregate)

    assert warnings == []
    assert table["Mode"].tolist() == ["no_qos", "qos_pure_llm", "qos_topsis", "qos_hybrid"]
    assert [f"{value:.6f}" for value in table["Average_Functional_Coverage"]] == [
        "0.955556",
        "1.000000",
        "0.238889",
        "1.000000",
    ]
    assert [f"{value:.6f}" for value in table["Average_Normalized_QoS_Score"]] == [
        "0.525717",
        "0.822242",
        "0.790922",
        "0.967079",
    ]


def test_figure_5_7_plot_has_fixed_axes_labeled_points_and_no_title() -> None:
    table = pd.DataFrame(
        [
            {
                "Mode": "no_qos",
                "Query_Count": 15,
                "Average_Functional_Coverage": 0.955556,
                "Average_Normalized_QoS_Score": 0.525717,
            },
            {
                "Mode": "qos_pure_llm",
                "Query_Count": 15,
                "Average_Functional_Coverage": 1.000000,
                "Average_Normalized_QoS_Score": 0.822242,
            },
            {
                "Mode": "qos_topsis",
                "Query_Count": 15,
                "Average_Functional_Coverage": 0.238889,
                "Average_Normalized_QoS_Score": 0.790922,
            },
            {
                "Mode": "qos_hybrid",
                "Query_Count": 15,
                "Average_Functional_Coverage": 1.000000,
                "Average_Normalized_QoS_Score": 0.967079,
            },
        ]
    )

    fig = tg.plot_figure_5_7_functional_vs_qos(table)
    try:
        ax = fig.axes[0]
        assert ax.get_title() == ""
        assert ax.get_xlim() == (0.0, 1.0)
        assert ax.get_ylim() == (0.0, 1.0)
        assert ax.get_xlabel() == "Average Functional Coverage"
        assert ax.get_ylabel() == "Average Normalized QoS Score"
        assert {text.get_text() for text in ax.texts} == set(tg.FIGURE_5_7_MODE_ORDER)
        assert len(ax.collections) == 4
        assert len(tg.figure_to_bytes(fig, "png", dpi=300)) > 1000
    finally:
        plt.close(fig)


def test_figure_5_8_source_matrices_use_required_metric_and_mode_order() -> None:
    modes = ["qos_hybrid", "qos_topsis", "qos_pure_llm", "no_qos"]
    base = pd.DataFrame(
        [
            [1.0, 0.4, 0.8, 0.2],
            [0.4, 1.0, 0.3, 0.5],
            [0.8, 0.3, 1.0, 0.6],
            [0.2, 0.5, 0.6, 1.0],
        ],
        index=modes,
        columns=modes,
    )
    matrices = {metric: base.copy() for metric in tg.FIGURE_5_8_METRIC_ORDER}

    ordered, source, warnings = tg.build_figure_5_8_ranking_similarity_matrices(matrices)

    assert warnings == []
    assert list(ordered) == ["spearman", "average_overlap", "rbo", "jaccard"]
    for matrix in ordered.values():
        assert matrix.index.tolist() == tg.FIGURE_5_8_MODE_ORDER
        assert matrix.columns.tolist() == tg.FIGURE_5_8_MODE_ORDER
    assert len(source) == 64
    assert source["Source_File"].drop_duplicates().tolist() == [
        "spearman_matrix.csv",
        "average_overlap_matrix.csv",
        "rbo_matrix.csv",
        "jaccard_matrix.csv",
    ]


def test_figure_5_8_plot_has_four_panel_shared_scale_values_and_no_suptitle() -> None:
    modes = tg.FIGURE_5_8_MODE_ORDER
    matrices = {}
    for offset, metric in enumerate(tg.FIGURE_5_8_METRIC_ORDER):
        matrix = pd.DataFrame(
            [
                [1.0, 0.111 + offset * 0.01, 0.222, 0.333],
                [0.111 + offset * 0.01, 1.0, 0.444, 0.555],
                [0.222, 0.444, 1.0, 0.666],
                [0.333, 0.555, 0.666, 1.0],
            ],
            index=modes,
            columns=modes,
        )
        matrices[metric] = matrix

    fig = tg.plot_figure_5_8_ranking_similarity_panel(matrices)
    try:
        heatmap_axes = fig.axes[:4]
        assert fig._suptitle is None
        assert [ax.get_title() for ax in heatmap_axes] == [
            "Spearman",
            "Average Overlap@K",
            "RBO",
            "Jaccard@K",
        ]
        for ax in heatmap_axes:
            assert ax.images[0].get_clim() == (0.0, 1.0)
            assert [tick.get_text() for tick in ax.get_xticklabels()] == modes
            assert [tick.get_text() for tick in ax.get_yticklabels()] == modes
            assert len(ax.texts) == 16
        assert "0.111" in {text.get_text() for text in heatmap_axes[0].texts}
        assert len(fig.axes) == 5
        assert fig.axes[-1].get_ylabel() == "Similarity"
        assert len(tg.figure_to_bytes(fig, "png", dpi=300)) > 1000
    finally:
        plt.close(fig)


def test_figure_5_10_hybrid_evidence_flow_is_conceptual_diagram_without_caption() -> None:
    fig = tg.plot_figure_5_10_hybrid_evidence_flow()
    try:
        ax = fig.axes[0]
        text = "\n".join(item.get_text() for item in ax.texts)
        normalized_text = " ".join(text.split())
        assert tg.FIGURE_5_10_TITLE in text
        assert "Observed mode behavior" in text
        assert "Design implication" in text
        assert "Hybrid selection rule" in text
        assert "Outcome" in text
        assert "Functional Match Label = 1 APIs" in normalized_text
        assert "TOPSIS over response time" in normalized_text
        assert "QoS-aware refinement" in normalized_text
        assert "Figure 5." not in normalized_text
        assert "caption" not in normalized_text.lower()
        assert fig._suptitle is None
        assert len(ax.patches) >= 11
        assert len(tg.figure_to_bytes(fig, "png", dpi=300)) > 1000
    finally:
        plt.close(fig)


def test_section_511_official_query_score_matrix_matches_requested_winner_pattern() -> None:
    df = pd.read_csv(OFFICIAL_Q01_Q15_COMPOSITION)

    matrix, best_df, warnings = tg.build_query_score_matrix(
        df,
        tg.SECTION_511_QUERY_IDS,
        tg.SECTION_511_MODE_ORDER,
    )

    assert warnings == []
    assert matrix["Query_ID"].tolist() == tg.SECTION_511_QUERY_IDS
    assert [col for col in matrix.columns if col in tg.SECTION_511_MODE_ORDER] == tg.SECTION_511_MODE_ORDER
    assert len(matrix) == 15

    tied = best_df[best_df["Best_Status"] == "tied best"].set_index("Query_ID")["Best_Modes"].to_dict()
    assert tied == {
        "q06": "qos_pure_llm, qos_hybrid",
        "q08": "qos_pure_llm, qos_hybrid",
        "q11": "qos_pure_llm, qos_hybrid",
        "q14": "qos_pure_llm, qos_hybrid",
    }

    unique_best = best_df[best_df["Best_Status"] == "unique best"].set_index("Query_ID")["Best_Modes"].to_dict()
    assert unique_best == {
        "q01": "qos_hybrid",
        "q02": "qos_hybrid",
        "q03": "qos_hybrid",
        "q04": "qos_hybrid",
        "q05": "qos_hybrid",
        "q07": "qos_hybrid",
        "q09": "qos_hybrid",
        "q10": "qos_hybrid",
        "q12": "qos_hybrid",
        "q13": "qos_hybrid",
        "q15": "qos_hybrid",
    }
    assert "no_qos" not in best_df["Best_Modes"].str.cat(sep=", ")
    assert "qos_topsis" not in best_df["Best_Modes"].str.cat(sep=", ")


def test_section_511_grouped_score_plot_has_required_axes_legend_and_no_caption_text() -> None:
    df = pd.read_csv(OFFICIAL_Q01_Q15_COMPOSITION)

    fig = tg.plot_query_grouped_scores(df, tg.SECTION_511_QUERY_IDS, tg.SECTION_511_MODE_ORDER)
    try:
        ax = fig.axes[0]
        assert ax.get_title() == ""
        assert fig._suptitle is None
        assert ax.get_ylim() == (0.0, 1.0)
        assert ax.get_ylabel() == "QoS-Adjusted Composition Score"
        assert [tick.get_text() for tick in ax.get_xticklabels()] == tg.SECTION_511_QUERY_IDS
        assert [item.get_text() for item in ax.get_legend().get_texts()] == tg.SECTION_511_MODE_ORDER
        assert len(ax.patches) == 60
        assert all("Figure 5." not in text.get_text() for text in ax.texts)
        assert len(tg.figure_to_bytes(fig, "png", dpi=300)) > 1000
    finally:
        plt.close(fig)


def test_section_511_winner_tie_heatmap_labels_best_and_tie_without_caption_text() -> None:
    df = pd.read_csv(OFFICIAL_Q01_Q15_COMPOSITION)
    _matrix, best_df, warnings = tg.build_query_score_matrix(df, tg.SECTION_511_QUERY_IDS, tg.SECTION_511_MODE_ORDER)
    assert warnings == []

    fig = tg.plot_winner_tie_heatmap(best_df, tg.SECTION_511_MODE_ORDER)
    try:
        ax = fig.axes[0]
        assert ax.get_title() == ""
        assert fig._suptitle is None
        assert [tick.get_text() for tick in ax.get_xticklabels()] == tg.SECTION_511_MODE_ORDER
        assert [tick.get_text() for tick in ax.get_yticklabels()] == tg.SECTION_511_QUERY_IDS

        label_by_cell = {}
        for text in ax.texts:
            label = text.get_text()
            if not label:
                continue
            col_idx = int(round(text.get_position()[0]))
            row_idx = int(round(text.get_position()[1]))
            label_by_cell[(tg.SECTION_511_QUERY_IDS[row_idx], tg.SECTION_511_MODE_ORDER[col_idx])] = label

        tied_queries = {"q06", "q08", "q11", "q14"}
        unique_queries = set(tg.SECTION_511_QUERY_IDS).difference(tied_queries)
        for query_id in unique_queries:
            assert label_by_cell[(query_id, "qos_hybrid")] == "Best"
        for query_id in tied_queries:
            assert label_by_cell[(query_id, "qos_pure_llm")] == "Tie"
            assert label_by_cell[(query_id, "qos_hybrid")] == "Tie"
        assert all(mode not in {"no_qos", "qos_topsis"} for _query_id, mode in label_by_cell)
        assert "Figure 5." not in " ".join(text.get_text() for text in ax.texts)
        assert len(tg.figure_to_bytes(fig, "png", dpi=300)) > 1000
    finally:
        plt.close(fig)


def test_build_selected_api_path_comparison_uses_planner_paths_and_csv_scores(tmp_path) -> None:
    modes = ["no_qos", "qos_pure_llm", "qos_topsis", "qos_hybrid"]
    composition_rows = []
    loaded_rows = []
    for mode_index, mode in enumerate(modes, start=1):
        api_ids = [f"{mode}_api_{step}" for step in range(1, 4)]
        planner_path = tmp_path / "q02_run" / mode / "4_planner.json"
        planner_path.parent.mkdir(parents=True, exist_ok=True)
        planner_path.write_text(
            json.dumps(
                {
                    "selected_api_ids": api_ids,
                    "execution_workflow": {
                        "steps": [
                            {
                                "step": step,
                                "subtask_id": str(step),
                                "api_id": api_id,
                                "method": "GET",
                                "url": f"https://example.test/{api_id}",
                            }
                            for step, api_id in enumerate(api_ids, start=1)
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )
        composition_rows.append(
            {
                "Query_ID": "q02",
                "Mode": mode,
                "Run_Folder": "q02_run",
                "Planner_Output_File": f"{mode}/4_planner.json",
                "Composition_Completeness": 1.0,
                "Functional_Coverage": 1.0 if mode != "qos_topsis" else 1 / 3,
                "Total_Response_Time_s": 0.1 * mode_index,
                "Bottleneck_Throughput_kbps": 10.0 * mode_index,
                "Average_Workflow_Availability": 0.99,
                "Normalized_QoS_Score": 0.8 + 0.01 * mode_index,
                "QoS_Adjusted_Composition_Score": 0.9 + 0.01 * mode_index,
            }
        )
        for step, api_id in enumerate(api_ids, start=1):
            loaded_rows.append(
                {
                    "query_id": "q02",
                    "mode": mode,
                    "subtask_id": str(step),
                    "api_id": api_id,
                    "functional_match_label": 0 if mode == "qos_topsis" and step > 1 else 1,
                    "selected_for_planner": 1,
                    "mode_rank": step,
                }
            )

    figure_df, score_trace_df, detail_df, warnings = tg.build_selected_api_path_comparison(
        tmp_path,
        pd.DataFrame(composition_rows),
        pd.DataFrame(loaded_rows),
        query_id="q02",
        selected_modes=modes,
        expected_steps=3,
    )

    assert warnings == []
    assert figure_df.columns.tolist() == tg.SECTION_55_FIGURE_COLUMNS
    assert figure_df["Mode"].tolist() == modes
    assert figure_df.loc[0, "Subtask 1"] == "no_qos_api_1\nFM=1"
    assert figure_df.loc[2, "Subtask 2"] == "qos_topsis_api_2\nFM=0"
    assert figure_df.loc[2, "Functional Coverage"] == "0.333333"
    assert [round(value, 2) for value in score_trace_df["QoS_Adjusted_Composition_Score"].tolist()] == [
        0.91,
        0.92,
        0.93,
        0.94,
    ]
    assert detail_df["Source_Planner_Path"].str.endswith("4_planner.json").all()
