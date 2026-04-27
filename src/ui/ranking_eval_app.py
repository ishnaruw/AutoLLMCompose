from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.ranking_metrics import (  # noqa: E402
    DEFAULT_RBO_P,
    METRIC_NAMES,
    MODE_ORDER,
    aggregate_matrices,
    cases_to_frame,
    compute_case_matrices,
    evaluate_parent_runs,
    matrices_to_pairwise_table,
    overlap_by_depth,
    top_lists_to_wide_frame,
)

DEFAULT_PARENT = (
    PROJECT_ROOT
    / "results/logs/RUNS_Fireworks_AI/fireworks_accounts/fireworks/models/deepseek-v3p1"
)

METRIC_LABELS = {
    "spearman": "Spearman",
    "average_overlap": "Average Overlap",
    "rbo": "RBO",
    "jaccard": "Jaccard",
}


@st.cache_data(show_spinner=False)
def _load_bundle(parent_dir: str, rbo_p: float):
    return evaluate_parent_runs(parent_dir, p=rbo_p)


def _heatmap(matrix: pd.DataFrame, title: str, value_range: tuple[float, float]):
    fig = px.imshow(
        matrix.astype(float),
        x=matrix.columns,
        y=matrix.index,
        text_auto=".3f",
        color_continuous_scale="RdYlGn",
        zmin=value_range[0],
        zmax=value_range[1],
        aspect="auto",
        title=title,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), height=410)
    fig.update_xaxes(side="top")
    return fig


def _case_label(row: pd.Series) -> str:
    fallback = " fallback K" if bool(row.get("k_fallback_used")) else ""
    run_name = Path(str(row["run_dir"])).name
    return f"{row['query_id']} / subtask {row['subtask_id']} / K={row['k']}{fallback} / {run_name}"


def _filter_pairwise(pairwise: pd.DataFrame, metrics: list[str], modes: list[str]) -> pd.DataFrame:
    if pairwise.empty:
        return pairwise
    return pairwise[
        pairwise["metric"].isin(metrics)
        & pairwise["mode_a"].isin(modes)
        & pairwise["mode_b"].isin(modes)
    ].copy()


def main() -> None:
    st.set_page_config(page_title="MAOF Ranking Evaluation", layout="wide")
    st.title("MAOF Ranking Evaluation")

    with st.sidebar:
        st.header("Input")
        parent_dir = st.text_input("Parent runs directory", value=str(DEFAULT_PARENT))
        rbo_p = st.slider("RBO p", min_value=0.5, max_value=0.99, value=DEFAULT_RBO_P, step=0.01)
        if st.button("Reload reports"):
            _load_bundle.clear()

    bundle = _load_bundle(parent_dir, rbo_p)
    cases_df = cases_to_frame(bundle.cases)

    if bundle.warnings:
        with st.expander(f"Warnings ({len(bundle.warnings)})", expanded=False):
            for warning in bundle.warnings:
                st.warning(warning)

    if not bundle.cases:
        st.error("No complete query/subtask cases were available for ranking evaluation.")
        st.caption("Check that the selected parent directory contains q* run folders with Excel reports.")
        return

    with st.sidebar:
        st.header("Filters")
        query_options = sorted(cases_df["query_id"].unique().tolist())
        selected_queries = st.multiselect("Query ID", query_options, default=query_options)

        visible_cases_df = cases_df[cases_df["query_id"].isin(selected_queries)]
        subtask_options = sorted(visible_cases_df["subtask_id"].unique().tolist(), key=lambda value: str(value))
        selected_subtasks = st.multiselect("Subtask ID", subtask_options, default=subtask_options)

        selected_metrics = st.multiselect(
            "Metric",
            METRIC_NAMES,
            default=METRIC_NAMES,
            format_func=lambda value: METRIC_LABELS.get(value, value),
        )
        selected_modes = st.multiselect("Modes", MODE_ORDER, default=MODE_ORDER)

    if not selected_metrics:
        st.warning("Select at least one metric.")
        return
    if len(selected_modes) < 2:
        st.warning("Select at least two modes.")
        return

    filtered_ids = set(
        cases_df[
            cases_df["query_id"].isin(selected_queries)
            & cases_df["subtask_id"].isin(selected_subtasks)
        ]["case_id"]
    )
    filtered_cases = [case for case in bundle.cases if case.case_id in filtered_ids]
    if not filtered_cases:
        st.error("No cases match the selected filters.")
        return

    filtered_matrices = aggregate_matrices(filtered_cases, p=rbo_p)
    filtered_pairwise = matrices_to_pairwise_table(filtered_matrices)

    fallback_count = sum(case.k_fallback_used for case in filtered_cases)
    stat_cols = st.columns(5)
    stat_cols[0].metric("Included cases", len(filtered_cases))
    stat_cols[1].metric("Discovered runs", len(bundle.discovered_run_dirs))
    stat_cols[2].metric("Loaded reports", len(bundle.loaded_report_paths))
    stat_cols[3].metric("Fallback K cases", fallback_count)
    stat_cols[4].metric("Rows loaded", len(bundle.raw_rows))

    st.subheader("Overall Mode Similarity")
    for metric in selected_metrics:
        matrix = filtered_matrices[metric].loc[selected_modes, selected_modes]
        value_range = (-1.0, 1.0) if metric == "spearman" else (0.0, 1.0)
        st.plotly_chart(
            _heatmap(matrix, METRIC_LABELS.get(metric, metric), value_range),
            use_container_width=True,
        )

    st.subheader("Pairwise Scores")
    pairwise_view = _filter_pairwise(filtered_pairwise, selected_metrics, selected_modes)
    st.dataframe(pairwise_view.round({"score": 4}), use_container_width=True, hide_index=True)

    if not pairwise_view.empty:
        pairwise_plot = pairwise_view.copy()
        pairwise_plot["metric_label"] = pairwise_plot["metric"].map(METRIC_LABELS)
        pairwise_plot["mode_pair"] = pairwise_plot["mode_a"] + " vs " + pairwise_plot["mode_b"]
        fig = px.bar(
            pairwise_plot,
            x="mode_pair",
            y="score",
            color="metric_label",
            barmode="group",
            range_y=[-1, 1] if "spearman" in selected_metrics else [0, 1],
            labels={"mode_pair": "Mode pair", "score": "Score", "metric_label": "Metric"},
        )
        fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Selected Query/Subtask Details")
    detail_df = cases_to_frame(filtered_cases)
    detail_df["label"] = detail_df.apply(_case_label, axis=1)
    selected_label = st.selectbox("Query/subtask", detail_df["label"].tolist())
    selected_case_id = detail_df.loc[detail_df["label"] == selected_label, "case_id"].iloc[0]
    selected_case = next(case for case in filtered_cases if case.case_id == selected_case_id)

    k_note = (
        "fallback K used because qos_hybrid had 0 functional matches"
        if selected_case.k_fallback_used
        else "K from qos_hybrid Functional Match Label count"
    )
    st.caption(f"K = {selected_case.k} ({k_note})")
    st.dataframe(top_lists_to_wide_frame(selected_case), use_container_width=True, hide_index=True)

    st.subheader("Case-Level Metrics")
    case_matrices = compute_case_matrices(selected_case, p=rbo_p)
    selected_case_metric = st.selectbox(
        "Case metric",
        METRIC_NAMES,
        format_func=lambda value: METRIC_LABELS.get(value, value),
    )
    value_range = (-1.0, 1.0) if selected_case_metric == "spearman" else (0.0, 1.0)
    st.plotly_chart(
        _heatmap(
            case_matrices[selected_case_metric].loc[selected_modes, selected_modes],
            f"{METRIC_LABELS.get(selected_case_metric, selected_case_metric)} for selected case",
            value_range,
        ),
        use_container_width=True,
    )

    st.subheader("Average Overlap by Depth")
    pair_cols = st.columns(2)
    left_mode = pair_cols[0].selectbox("Mode A", selected_modes, index=0)
    default_right = selected_modes.index("qos_hybrid") if "qos_hybrid" in selected_modes else len(selected_modes) - 1
    right_mode = pair_cols[1].selectbox("Mode B", selected_modes, index=default_right)
    ao_depth = overlap_by_depth(
        selected_case.top_lists[left_mode],
        selected_case.top_lists[right_mode],
        selected_case.k,
    )
    fig = px.line(
        ao_depth,
        x="depth",
        y="overlap_ratio",
        markers=True,
        range_y=[0, 1],
        labels={"depth": "Depth", "overlap_ratio": "Overlap ratio"},
    )
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=320)
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
