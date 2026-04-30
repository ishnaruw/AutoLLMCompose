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
    aggregate_matrices_with_counts,
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
    "spearman": "Spearman (All Candidates)",
    "average_overlap": "Top-K Average Overlap",
    "rbo": "Top-K RBO",
    "jaccard": "Top-K Jaccard",
}

METRIC_HELP = {
    "spearman": "Standard Spearman correlation over the complete shared ranked candidate list for each mode.",
    "average_overlap": "Top-K mean overlap across depths 1..K.",
    "rbo": "Top-K RBO with p=0.9 by default; emphasizes highly ranked overlap and is less sensitive to exact position changes when lists contain similar APIs.",
    "jaccard": "Top-K set overlap only; it ignores within-set order.",
}

K_HELP = (
    "Top-K metrics use a shared K per query/subtask. K is derived from the qos_hybrid functional-match "
    "count because qos_hybrid is the functional-refinement mode used to define the meaningful valid "
    "selection depth. This keeps AO, RBO, and Jaccard comparisons at the same cutoff across all modes."
)


@st.cache_data(show_spinner=False)
def _load_bundle(parent_dir: str, rbo_p: float, selected_modes: tuple[str, ...]):
    return evaluate_parent_runs(parent_dir, p=rbo_p, selected_modes=list(selected_modes))


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
        rbo_p = st.slider(
            "RBO p",
            min_value=0.5,
            max_value=0.99,
            value=DEFAULT_RBO_P,
            step=0.01,
            help=METRIC_HELP["rbo"],
        )
        selected_modes = st.multiselect(
            "Modes",
            MODE_ORDER,
            default=MODE_ORDER,
            help="Evaluation includes only query/subtask cases where all selected modes have usable outputs for the metric being computed.",
        )
        st.caption(K_HELP)
        if st.button("Reload reports"):
            _load_bundle.clear()

    if len(selected_modes) < 2:
        st.warning("Select at least two modes.")
        return

    bundle = _load_bundle(parent_dir, rbo_p, tuple(selected_modes))
    cases_df = cases_to_frame(bundle.cases)

    if bundle.warnings:
        with st.expander(f"Warnings ({len(bundle.warnings)})", expanded=False):
            for warning in bundle.warnings:
                st.warning(warning)

    if not bundle.cases:
        st.error("No query/subtask cases were available for the selected modes.")
        st.caption("Check that the selected parent directory contains q* run folders with Excel reports.")
        if not bundle.invalid_cases.empty:
            with st.expander(f"Invalid mode/subtask cases ({len(bundle.invalid_cases)})", expanded=True):
                st.dataframe(bundle.invalid_cases, width="stretch", hide_index=True)
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

    if not selected_metrics:
        st.warning("Select at least one metric.")
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

    filtered_matrices, filtered_counts = aggregate_matrices_with_counts(filtered_cases, p=rbo_p)
    filtered_pairwise = matrices_to_pairwise_table(filtered_matrices, filtered_counts)

    fallback_count = sum(case.k_fallback_used for case in filtered_cases)
    included_pairwise = (
        int(filtered_pairwise["included_cases"].sum())
        if "included_cases" in filtered_pairwise.columns and not filtered_pairwise.empty
        else 0
    )
    stat_cols = st.columns(7)
    stat_cols[0].metric("Included cases", len(filtered_cases))
    stat_cols[1].metric("Discovered runs", len(bundle.discovered_run_dirs))
    stat_cols[2].metric("Loaded reports", len(bundle.loaded_report_paths))
    stat_cols[3].metric("Fallback K cases", fallback_count)
    stat_cols[4].metric("Metric comparisons", included_pairwise)
    stat_cols[5].metric("Invalid cases", len(bundle.invalid_cases))
    stat_cols[6].metric("Rows loaded", len(bundle.raw_rows))
    st.caption("Evaluation rule: strict selected-mode evaluation. Final reporting should use all four modes selected.")

    if not bundle.invalid_cases.empty:
        st.subheader("Invalid Evaluation Cases")
        invalid_cols = st.columns(3)
        invalid_cols[0].dataframe(
            bundle.invalid_cases.groupby("mode", dropna=False).size().reset_index(name="count"),
            width="stretch",
            hide_index=True,
        )
        invalid_cols[1].dataframe(
            bundle.invalid_cases.groupby("failure_reason", dropna=False).size().reset_index(name="count"),
            width="stretch",
            hide_index=True,
        )
        invalid_cols[2].dataframe(
            bundle.invalid_cases.groupby(["query_id", "subtask_id"], dropna=False).size().reset_index(name="count"),
            width="stretch",
            hide_index=True,
        )
        with st.expander("Excluded invalid mode/subtask rows", expanded=False):
            st.dataframe(bundle.invalid_cases, width="stretch", hide_index=True)

    st.subheader("Overall Mode Similarity")
    with st.expander("Metric notes", expanded=False):
        for metric in METRIC_NAMES:
            st.markdown(f"**{METRIC_LABELS[metric]}**: {METRIC_HELP[metric]}")
        st.markdown(f"**K**: {K_HELP}")
    for metric in selected_metrics:
        matrix = filtered_matrices[metric].loc[selected_modes, selected_modes]
        value_range = (-1.0, 1.0) if metric == "spearman" else (0.0, 1.0)
        st.plotly_chart(
            _heatmap(matrix, METRIC_LABELS.get(metric, metric), value_range),
            width="stretch",
        )

    st.subheader("Pairwise Scores")
    pairwise_view = _filter_pairwise(filtered_pairwise, selected_metrics, selected_modes)
    pairwise_table = pairwise_view.copy()
    if not pairwise_table.empty:
        pairwise_table["metric"] = pairwise_table["metric"].map(METRIC_LABELS)
    st.dataframe(pairwise_table.round({"score": 4}), width="stretch", hide_index=True)

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
        st.plotly_chart(fig, width="stretch")

    st.subheader("Selected Query/Subtask Details")
    detail_df = cases_to_frame(filtered_cases)
    detail_df["label"] = detail_df.apply(_case_label, axis=1)
    selected_label = st.selectbox("Query/subtask", detail_df["label"].tolist())
    selected_case_id = detail_df.loc[detail_df["label"] == selected_label, "case_id"].iloc[0]
    selected_case = next(case for case in filtered_cases if case.case_id == selected_case_id)

    k_note = (
        "fallback K used because qos_hybrid had 0 functional matches"
        if selected_case.k_fallback_used
        else "K from qos_hybrid functional-match count"
    )
    ranked_count = min((len(selected_case.ranked_lists.get(mode, [])) for mode in selected_case.valid_modes), default=0)
    st.caption(f"K = {selected_case.k} ({k_note}); Spearman uses all {ranked_count} ranked candidates.")
    st.dataframe(top_lists_to_wide_frame(selected_case), width="stretch", hide_index=True)

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
        width="stretch",
    )

    st.subheader("Average Overlap by Depth")
    pair_cols = st.columns(2)
    left_mode = pair_cols[0].selectbox("Mode A", selected_modes, index=0)
    default_right = selected_modes.index("qos_hybrid") if "qos_hybrid" in selected_modes else len(selected_modes) - 1
    right_mode = pair_cols[1].selectbox("Mode B", selected_modes, index=default_right)
    if left_mode not in selected_case.top_lists or right_mode not in selected_case.top_lists:
        st.warning("The selected pair is not available for this query/subtask under the current inclusion policy.")
        return
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
    st.plotly_chart(fig, width="stretch")


if __name__ == "__main__":
    main()
