from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Callable

from src.core.runtime_bootstrap import harden_scientific_runtime

harden_scientific_runtime()

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui import thesis_figure_generators as tg


@dataclass
class GeneratedArtifact:
    label: str
    stem: str
    kind: str
    dataframe: pd.DataFrame | None = None
    figure: plt.Figure | None = None
    caption: str | None = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = PROJECT_ROOT / "results" / "logs" / "RUNS_MAY_31_NEW_5" / "fireworks_gpt-oss-120b"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "thesis_figures" / "results_chapter"
DEFAULT_PER_QUERY_METRICS = [
    "QoS_Adjusted_Composition_Score",
    "Functional_Coverage",
    "Normalized_QoS_Score",
    "Composition_Validity",
    "Composition_Completeness",
    "Total_Response_Time_s",
    "Bottleneck_Throughput_kbps",
    "Average_Workflow_Availability",
]
SECTION_511_GROUPED_CAPTION = (
    "Figure 5.x: Query-level QoS-Adjusted Composition Score by mode across the official q01 to q15 evaluation set."
)
SECTION_511_HEATMAP_CAPTION = (
    "Figure 5.x: Unique-best and tied-best mode outcomes across the official q01 to q15 evaluation set."
)


@st.cache_data(show_spinner=False)
def _load_artifacts_cached(project_root_text: str, run_dir_text: str, summary_dir_text: str, ranking_dir_text: str):
    return tg.load_thesis_artifacts(
        Path(project_root_text),
        tg.resolve_path(run_dir_text, Path(project_root_text)),
        tg.resolve_path(summary_dir_text, Path(project_root_text)),
        tg.resolve_path(ranking_dir_text, Path(project_root_text)),
    )


def _resolve_text_path(text: str) -> Path:
    return tg.resolve_path(text, PROJECT_ROOT)


def _sync_path_state(key: str, suggested: Path) -> None:
    default_key = f"{key}_suggested"
    if key not in st.session_state:
        st.session_state[key] = str(suggested)
        st.session_state[default_key] = str(suggested)


def _figure_format_specs(selected_formats: list[str]) -> list[tuple[str, int, str]]:
    specs: list[tuple[str, int, str]] = []
    if "PDF" in selected_formats:
        specs.append(("pdf", 300, "application/pdf"))
    if "PNG 300 dpi" in selected_formats:
        specs.append(("png", 300, "image/png"))
    if "PNG 600 dpi" in selected_formats:
        specs.append(("png", 600, "image/png"))
    return specs


def _format_suffix(fmt: str, dpi: int) -> str:
    if fmt == "pdf":
        return "pdf"
    return f"png_{dpi}dpi"


def _render_table(title: str, df: pd.DataFrame, stem: str, selected_formats: list[str], key_prefix: str) -> None:
    st.subheader(title)
    if df.empty:
        st.warning(f"{title}: no rows available from the loaded artifacts.")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)
    cols = st.columns(2)
    if "CSV" in selected_formats:
        cols[0].download_button(
            "Download CSV",
            data=tg.table_to_csv_bytes(df),
            file_name=f"{stem}.csv",
            mime="text/csv",
            key=f"{key_prefix}_{stem}_csv",
        )
    if "PDF" in selected_formats:
        table_fig = tg.plot_table_pdf(df, title)
        cols[1].download_button(
            "Download PDF table",
            data=tg.figure_to_bytes(table_fig, "pdf"),
            file_name=f"{stem}.pdf",
            mime="application/pdf",
            key=f"{key_prefix}_{stem}_pdf",
        )
        plt.close(table_fig)


def _render_figure(
    title: str,
    fig: plt.Figure | None,
    stem: str,
    selected_formats: list[str],
    key_prefix: str,
    caption: str | None = None,
) -> None:
    st.subheader(title)
    if fig is None:
        st.warning(f"{title}: figure could not be generated from the loaded artifacts.")
        return
    st.pyplot(fig, clear_figure=False)
    if caption:
        st.caption(caption)
    specs = _figure_format_specs(selected_formats)
    if not specs:
        return
    cols = st.columns(min(3, len(specs)))
    for idx, (fmt, dpi, mime) in enumerate(specs):
        suffix = _format_suffix(fmt, dpi)
        filename = f"{stem}.{fmt}" if fmt == "pdf" else f"{stem}_{dpi}dpi.png"
        cols[idx % len(cols)].download_button(
            f"Download {suffix.upper()}",
            data=tg.figure_to_bytes(fig, fmt, dpi=dpi),
            file_name=filename,
            mime=mime,
            key=f"{key_prefix}_{stem}_{suffix}",
        )


def _plotly_export_config(filename: str) -> dict:
    return {
        "displaylogo": False,
        "modeBarButtonsToRemove": ["select2d", "lasso2d"],
        "toImageButtonOptions": {
            "format": "png",
            "filename": filename,
            "width": 2200,
            "height": 1150,
            "scale": 2,
        },
    }


def _split_q02_path_cell(value: object) -> tuple[str, str]:
    lines = [line.strip() for line in str(value or "").splitlines() if line.strip()]
    api_id = lines[0] if lines else ""
    functional_match = next((line for line in lines[1:] if line.startswith("FM=")), "FM=n/a")
    return api_id, functional_match


def _wrap_api_id(api_id: str, width: int = 24, max_lines: int = 3) -> str:
    if not api_id:
        return ""
    parts = api_id.split("_")
    lines: list[str] = []
    current = ""
    for part in parts:
        candidate = part if not current else f"{current}_{part}"
        if current and len(candidate) > width:
            lines.append(current)
            current = part
        else:
            current = candidate
    if current:
        lines.append(current)
    if len(lines) > max_lines:
        clipped = lines[: max_lines - 1]
        clipped.append(lines[max_lines - 1][: max(0, width - 3)] + "...")
        lines = clipped
    return "<br>".join(escape(line) for line in lines)


def _functional_match_style(functional_match: str) -> dict[str, str]:
    if functional_match == "FM=1":
        return {"fill": "#E8F5EE", "border": "#2E7D5B", "text": "#14532D"}
    if functional_match == "FM=0":
        return {"fill": "#FFF1E6", "border": "#C2410C", "text": "#7C2D12"}
    return {"fill": "#F8FAFC", "border": "#64748B", "text": "#334155"}


def _score_value(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))


def _score_label(value: object) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def _build_q02_path_plotly_diagram(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    modes = df["Mode"].astype(str).tolist()
    y_positions = {mode: len(modes) - idx for idx, mode in enumerate(modes)}
    node_x = {"Subtask 1": 2.05, "Subtask 2": 4.55, "Subtask 3": 7.05}
    node_width = 1.95
    node_height = 0.56
    score_x = 9.65
    score_width = 1.45
    score_height = 0.095
    lane_left = 0.18
    lane_right = 11.55

    for idx, mode in enumerate(modes):
        y = y_positions[mode]
        lane_fill = "#FFFFFF" if idx % 2 == 0 else "#F8FAFC"
        fig.add_shape(
            type="rect",
            x0=lane_left,
            y0=y - 0.43,
            x1=lane_right,
            y1=y + 0.43,
            line={"width": 0},
            fillcolor=lane_fill,
            layer="below",
        )
        fig.add_annotation(
            x=0.25,
            y=y,
            text=f"<b>{escape(mode)}</b>",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font={"size": 13, "color": "#111827"},
        )

    for label, x in node_x.items():
        fig.add_annotation(
            x=x,
            y=len(modes) + 0.72,
            text=f"<b>{escape(label)}</b>",
            showarrow=False,
            yanchor="middle",
            font={"size": 13, "color": "#334155"},
        )
        fig.add_shape(
            type="line",
            x0=x,
            y0=0.55,
            x1=x,
            y1=len(modes) + 0.45,
            line={"color": "#E2E8F0", "width": 1},
            layer="below",
        )

    fig.add_annotation(
        x=score_x + 0.28,
        y=len(modes) + 0.72,
        text="<b>Workflow scores</b>",
        showarrow=False,
        yanchor="middle",
        font={"size": 13, "color": "#334155"},
    )

    for _, row in df.iterrows():
        mode = str(row["Mode"])
        y = y_positions[mode]
        previous_x: float | None = None
        for subtask, x in node_x.items():
            api_id, functional_match = _split_q02_path_cell(row.get(subtask, ""))
            style = _functional_match_style(functional_match)
            fig.add_shape(
                type="rect",
                x0=x - node_width / 2,
                y0=y - node_height / 2,
                x1=x + node_width / 2,
                y1=y + node_height / 2,
                line={"color": style["border"], "width": 1.8},
                fillcolor=style["fill"],
            )
            fig.add_annotation(
                x=x,
                y=y + 0.055,
                text=f"<b>{_wrap_api_id(api_id)}</b>",
                showarrow=False,
                yanchor="middle",
                font={"size": 10.5, "color": "#111827"},
            )
            fig.add_annotation(
                x=x,
                y=y - 0.205,
                text=escape(functional_match),
                showarrow=False,
                yanchor="middle",
                font={"size": 10.5, "color": style["text"]},
            )
            if previous_x is not None:
                fig.add_annotation(
                    x=x - node_width / 2 - 0.06,
                    y=y,
                    ax=previous_x + node_width / 2 + 0.06,
                    ay=y,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.0,
                    arrowwidth=1.5,
                    arrowcolor="#64748B",
                    text="",
                )
            previous_x = x

        score_specs = [
            ("FC", row.get("Functional Coverage"), "#2563EB", 0.19),
            ("QoS", row.get("Normalized QoS"), "#059669", 0.0),
            ("Score", row.get("Final Score"), "#7C3AED", -0.19),
        ]
        for label, value, color, y_offset in score_specs:
            bar_y = y + y_offset
            fig.add_annotation(
                x=8.62,
                y=bar_y,
                text=label,
                showarrow=False,
                xanchor="right",
                yanchor="middle",
                font={"size": 10.5, "color": "#475569"},
            )
            fig.add_shape(
                type="rect",
                x0=8.72,
                y0=bar_y - score_height / 2,
                x1=8.72 + score_width,
                y1=bar_y + score_height / 2,
                line={"color": "#CBD5E1", "width": 1},
                fillcolor="#F1F5F9",
            )
            fig.add_shape(
                type="rect",
                x0=8.72,
                y0=bar_y - score_height / 2,
                x1=8.72 + score_width * _score_value(value),
                y1=bar_y + score_height / 2,
                line={"width": 0},
                fillcolor=color,
            )
            fig.add_annotation(
                x=10.35,
                y=bar_y,
                text=_score_label(value),
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font={"size": 10.5, "color": "#111827"},
            )

    fig.add_annotation(
        x=0.45,
        y=len(modes) + 1.08,
        text="<b>q02 Selected API Path Comparison Across Modes</b>",
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        font={"size": 19, "color": "#111827"},
    )
    fig.add_annotation(
        x=0.45,
        y=0.18,
        text=(
            "Node fill indicates Functional Match label for the assigned subtask. "
            "Score bars are rounded visual summaries; exact values are reported in Table 5.x."
        ),
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        font={"size": 10.5, "color": "#64748B"},
    )
    fig.update_xaxes(range=[-0.35, 11.95], visible=False, fixedrange=True)
    fig.update_yaxes(range=[-0.25, len(modes) + 1.45], visible=False, fixedrange=True)
    fig.update_layout(
        margin={"l": 48, "r": 36, "t": 44, "b": 64},
        height=625,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        showlegend=False,
    )
    return fig


def _render_section_55_q02_path_dashboard(
    artifacts: tg.LoadedThesisArtifacts,
    composition: pd.DataFrame,
) -> None:
    st.subheader("Figure 5.x: q02 Selected API Path Comparison Across Modes")
    figure_df, score_trace_df, path_detail_df, warnings = tg.build_selected_api_path_comparison(
        artifacts.run_dir,
        composition,
        artifacts.loaded_rows,
        query_id=tg.SECTION_55_QUERY_ID,
        selected_modes=tg.PREFERRED_MODE_ORDER,
        expected_steps=tg.SECTION_55_EXPECTED_STEPS,
    )
    _display_warnings(warnings, "Section 5.5 q02 path warnings")
    if figure_df.empty:
        st.warning("The q02 selected API path comparison could not be built from the loaded artifacts.")
        return

    st.plotly_chart(
        _build_q02_path_plotly_diagram(figure_df),
        use_container_width=True,
        config=_plotly_export_config("figure_5_x_q02_selected_api_path_comparison"),
        key="section_55_q02_selected_api_path_comparison",
    )
    st.caption(
        "Figure 5.x: Dashboard-generated q02 selected API path comparison across the four evaluation modes. "
        "Each row shows the planned API sequence produced by one mode across the decomposed q02 subtasks. "
        "The Functional Match labels indicate whether the selected APIs were functionally suitable for their assigned subtasks, "
        "while the summary values report the workflow-level Functional Coverage, Normalized QoS Score, and QoS-Adjusted Composition Score."
    )

    st.download_button(
        "Download figure source CSV",
        data=tg.table_to_csv_bytes(figure_df),
        file_name="figure_5_x_q02_selected_api_path_comparison_source.csv",
        mime="text/csv",
        key="section_55_q02_path_source_csv",
    )

    with st.expander("Figure source table", expanded=False):
        st.dataframe(figure_df, use_container_width=True, hide_index=True)

    st.subheader("Table 5.x: q02 Workflow Score Trace")
    if score_trace_df.empty:
        st.warning("The q02 score trace table could not be built from the loaded composition summary.")
    else:
        st.dataframe(score_trace_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download score trace CSV",
            data=tg.table_to_csv_bytes(score_trace_df),
            file_name="table_5_x_q02_workflow_score_trace.csv",
            mime="text/csv",
            key="section_55_q02_score_trace_csv",
        )

    with st.expander("q02 selected path source rows", expanded=False):
        if path_detail_df.empty:
            st.info("No selected path source rows were available.")
        else:
            st.dataframe(path_detail_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download path detail CSV",
                data=tg.table_to_csv_bytes(path_detail_df),
                file_name="figure_5_x_q02_selected_api_path_detail.csv",
                mime="text/csv",
                key="section_55_q02_path_detail_csv",
            )


def _save_generated_artifacts(artifacts: list[GeneratedArtifact], output_dir: Path, selected_formats: list[str]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for artifact in artifacts:
        if artifact.kind == "table" and artifact.dataframe is not None and not artifact.dataframe.empty:
            if "CSV" in selected_formats:
                path = output_dir / f"{artifact.stem}.csv"
                artifact.dataframe.to_csv(path, index=False)
                saved.append(path)
            if "PDF" in selected_formats:
                fig = tg.plot_table_pdf(artifact.dataframe, artifact.label)
                path = output_dir / f"{artifact.stem}.pdf"
                path.write_bytes(tg.figure_to_bytes(fig, "pdf"))
                saved.append(path)
                plt.close(fig)
        if artifact.kind == "figure" and artifact.figure is not None:
            for fmt, dpi, _mime in _figure_format_specs(selected_formats):
                if fmt == "pdf":
                    path = output_dir / f"{artifact.stem}.pdf"
                else:
                    path = output_dir / f"{artifact.stem}_{dpi}dpi.png"
                path.write_bytes(tg.figure_to_bytes(artifact.figure, fmt, dpi=dpi))
                saved.append(path)
    return saved


def _artifact_by_stem(artifacts: list[GeneratedArtifact]) -> dict[str, GeneratedArtifact]:
    return {f"{artifact.label} ({artifact.stem})": artifact for artifact in artifacts}


def _selected_query_text(composition_df: pd.DataFrame, query_id: str) -> str:
    if composition_df.empty or "Query_ID" not in composition_df or "Query_Text" not in composition_df:
        return ""
    rows = composition_df[composition_df["Query_ID"].astype(str) == str(query_id)]
    if rows.empty:
        return ""
    values = rows["Query_Text"].dropna().astype(str)
    return values.iloc[0] if not values.empty else ""


def _build_recommended_artifacts(
    artifacts: tg.LoadedThesisArtifacts,
    selected_query_ids: list[str],
    selected_modes: list[str],
    selected_query: str,
    trace_query: str,
    per_query_metrics: list[str],
    top_n: int,
) -> tuple[list[GeneratedArtifact], list[str], dict[str, pd.DataFrame]]:
    warnings: list[str] = []
    generated: list[GeneratedArtifact] = []
    tables: dict[str, pd.DataFrame] = {}

    composition = artifacts.composition_results
    official_scope = tg.build_official_scope_table(artifacts)
    generated.append(GeneratedArtifact("Official Result Scope and Integrity Summary", "table_official_scope_integrity", "table", official_scope))
    tables["official_scope"] = official_scope

    sensitivity_long, sensitivity_summary, sensitivity_warnings = tg.build_weight_sensitivity_tables(
        composition, selected_query_ids, selected_modes
    )
    warnings.extend(sensitivity_warnings)
    if not sensitivity_summary.empty:
        generated.append(
            GeneratedArtifact(
                "Workflow-Level Score Sensitivity Across Weight Settings",
                "table_weight_sensitivity_workflow",
                "table",
                sensitivity_summary,
            )
        )
        generated.append(
            GeneratedArtifact(
                "Average score by mode across weight settings",
                "figure_weight_sensitivity_by_mode",
                "figure",
                figure=tg.plot_weight_sensitivity_by_mode(sensitivity_summary, selected_modes),
            )
        )
        generated.append(
            GeneratedArtifact(
                "Mode ranking stability across weight settings",
                "figure_weight_sensitivity_ranking_stability",
                "figure",
                figure=tg.plot_mode_ranking_stability(sensitivity_summary, selected_modes),
            )
        )
        selected_query_long = sensitivity_long[sensitivity_long["Query_ID"].astype(str) == str(selected_query)].copy()
        generated.append(
            GeneratedArtifact(
                f"Selected query score sensitivity by mode ({selected_query})",
                f"table_weight_sensitivity_{selected_query}",
                "table",
                selected_query_long,
            )
        )
        generated.append(
            GeneratedArtifact(
                f"Selected query score sensitivity by mode ({selected_query})",
                f"figure_weight_sensitivity_{selected_query}",
                "figure",
                figure=tg.plot_selected_query_sensitivity(sensitivity_long, selected_query, selected_modes),
            )
        )
    tables["sensitivity_summary"] = sensitivity_summary
    tables["sensitivity_long"] = sensitivity_long

    candidate_table, candidate_warnings = tg.build_candidate_top10_sensitivity(artifacts.loaded_rows, selected_query_ids)
    warnings.extend(candidate_warnings)
    if not candidate_table.empty:
        generated.append(
            GeneratedArtifact(
                "Candidate-Level Top-10 Sensitivity Summary",
                "table_candidate_top10_sensitivity",
                "table",
                candidate_table,
            )
        )
        generated.append(
            GeneratedArtifact(
                "Invalid Top-10 candidate count by weight setting",
                "figure_candidate_invalid_top10_by_weight",
                "figure",
                figure=tg.plot_candidate_invalid_top10(candidate_table),
            )
        )
    tables["candidate_sensitivity"] = candidate_table

    figure_5_2_table, figure_5_2_warnings = tg.build_figure_5_2_average_score_table(artifacts.aggregate_scores)
    warnings.extend(figure_5_2_warnings)
    if not figure_5_2_table.empty:
        generated.append(
            GeneratedArtifact(
                "Figure 5.2: Average QoS-Adjusted Composition Score by Mode",
                "figure_5_2_average_qos_adjusted_composition_score_by_mode",
                "figure",
                figure=tg.plot_figure_5_2_average_score_by_mode(figure_5_2_table),
            )
        )
        generated.append(
            GeneratedArtifact(
                "Figure 5.2 Source Values",
                "figure_5_2_average_qos_adjusted_composition_score_source",
                "table",
                figure_5_2_table,
            )
        )
    tables["figure_5_2_source"] = figure_5_2_table

    figure_5_7_table, figure_5_7_warnings = tg.build_figure_5_7_functional_vs_qos_table(artifacts.aggregate_scores)
    warnings.extend(figure_5_7_warnings)
    if not figure_5_7_table.empty:
        generated.append(
            GeneratedArtifact(
                "Figure 5.7: Functional Coverage versus Normalized QoS by Mode",
                "figure_5_7_functional_coverage_versus_normalized_qos_by_mode",
                "figure",
                figure=tg.plot_figure_5_7_functional_vs_qos(figure_5_7_table),
            )
        )
        generated.append(
            GeneratedArtifact(
                "Figure 5.7 Source Values",
                "figure_5_7_functional_coverage_versus_normalized_qos_source",
                "table",
                figure_5_7_table,
            )
        )
    tables["figure_5_7_source"] = figure_5_7_table

    aggregate_table, best_df, aggregate_warnings = tg.build_aggregate_performance_table(
        composition, selected_query_ids, selected_modes
    )
    warnings.extend(aggregate_warnings)
    if not aggregate_table.empty:
        generated.append(
            GeneratedArtifact(
                "Aggregate Workflow-Level Performance by Mode",
                "table_aggregate_mode_performance",
                "table",
                aggregate_table,
            )
        )
        generated.extend(
            [
                GeneratedArtifact(
                    "Total Response Time by Mode",
                    "figure_qos_response_time_by_mode",
                    "figure",
                    figure=tg.plot_metric_by_mode(
                        aggregate_table,
                        "Average_Total_Response_Time_s",
                        "Total response time (seconds)",
                        "Total Response Time by Mode",
                    ),
                ),
                GeneratedArtifact(
                    "Bottleneck Throughput by Mode",
                    "figure_qos_bottleneck_throughput_by_mode",
                    "figure",
                    figure=tg.plot_metric_by_mode(
                        aggregate_table,
                        "Average_Bottleneck_Throughput_kbps",
                        "Bottleneck throughput (kbps)",
                        "Bottleneck Throughput by Mode",
                    ),
                ),
                GeneratedArtifact(
                    "Average Workflow Availability by Mode",
                    "figure_qos_availability_by_mode",
                    "figure",
                    figure=tg.plot_metric_by_mode(
                        aggregate_table,
                        "Average_Average_Workflow_Availability",
                        "Average workflow availability",
                        "Average Workflow Availability by Mode",
                    ),
                ),
            ]
        )
    tables["aggregate"] = aggregate_table
    tables["best_queries"] = best_df

    pairwise_table, pairwise_warnings = tg.build_pairwise_similarity_table(artifacts.pairwise_scores)
    warnings.extend(pairwise_warnings)
    if not pairwise_table.empty:
        generated.append(
            GeneratedArtifact(
                "Pairwise Ranking Similarity Interpretation",
                "table_pairwise_ranking_similarity",
                "table",
                pairwise_table,
            )
        )
    figure_5_8_matrices, figure_5_8_source, figure_5_8_warnings = tg.build_figure_5_8_ranking_similarity_matrices(
        artifacts.matrices
    )
    warnings.extend(figure_5_8_warnings)
    if figure_5_8_matrices:
        generated.append(
            GeneratedArtifact(
                "Figure 5.8: Overall Ranking Similarity Across Modes",
                "figure_5_8_overall_ranking_similarity_across_modes",
                "figure",
                figure=tg.plot_figure_5_8_ranking_similarity_panel(figure_5_8_matrices),
            )
        )
        generated.append(
            GeneratedArtifact(
                "Figure 5.8 Source Matrix Values",
                "figure_5_8_overall_ranking_similarity_source",
                "table",
                figure_5_8_source,
            )
        )
    tables["pairwise"] = pairwise_table
    tables["figure_5_8_source"] = figure_5_8_source

    generated.append(
        GeneratedArtifact(
            "Figure 5.10: Hybrid Mode Rationale and Evidence Flow",
            "figure_5_10_hybrid_mode_rationale_and_evidence_flow",
            "figure",
            figure=tg.plot_figure_5_10_hybrid_evidence_flow(),
        )
    )

    agreement_table, agreement_warnings = tg.compute_llm_topsis_agreement(artifacts.loaded_rows, selected_query_ids, top_n)
    warnings.extend(agreement_warnings)
    if not agreement_table.empty:
        generated.append(
            GeneratedArtifact(
                "LLM-TOPSIS Agreement and Disagreement by Query/Subtask",
                "table_llm_topsis_agreement",
                "table",
                agreement_table,
            )
        )
        generated.append(
            GeneratedArtifact(
                "Agreement rate by query",
                "figure_llm_topsis_agreement_rate_by_query",
                "figure",
                figure=tg.plot_agreement_rate_by_query(agreement_table),
            )
        )
    tables["agreement"] = agreement_table

    query_matrix, query_best, query_warnings = tg.build_query_score_matrix(
        composition,
        selected_query_ids,
        selected_modes,
    )
    warnings.extend(query_warnings)
    if not query_matrix.empty:
        generated.append(
            GeneratedArtifact("Query-Level Final Score Matrix", "table_query_level_score_matrix", "table", query_matrix)
        )
        generated.append(
            GeneratedArtifact(
                "Figure 5.x: Grouped Query-Level Final Score by Mode",
                "figure_query_level_grouped_scores",
                "figure",
                figure=tg.plot_query_grouped_scores(composition, selected_query_ids, selected_modes),
                caption=SECTION_511_GROUPED_CAPTION,
            )
        )
        generated.append(
            GeneratedArtifact(
                "Figure 5.x: Winner and Tied-Winner Heatmap",
                "figure_query_winner_tie_heatmap",
                "figure",
                figure=tg.plot_winner_tie_heatmap(query_best, selected_modes),
                caption=SECTION_511_HEATMAP_CAPTION,
            )
        )
    tables["query_matrix"] = query_matrix
    tables["query_best"] = query_best

    for query_id in selected_query_ids:
        table, per_query_warnings = tg.build_per_query_metrics_table(composition, query_id, selected_modes, per_query_metrics)
        warnings.extend(per_query_warnings)
        if table.empty:
            continue
        generated.append(
            GeneratedArtifact(
                f"Per-query metrics ({query_id})",
                f"table_per_query_{query_id}_metrics",
                "table",
                table,
            )
        )
        if "QoS_Adjusted_Composition_Score" in table:
            generated.append(
                GeneratedArtifact(
                    f"Final Score by Mode for {query_id}",
                    f"figure_per_query_{query_id}_score_by_mode",
                    "figure",
                    figure=tg.plot_per_query_score_by_mode(table, query_id),
                )
            )
        if {"Functional_Coverage", "Normalized_QoS_Score"}.issubset(table.columns):
            generated.append(
                GeneratedArtifact(
                    f"Functional Coverage and Normalized QoS by Mode for {query_id}",
                    f"figure_per_query_{query_id}_components",
                    "figure",
                    figure=tg.plot_per_query_components(table, query_id),
                )
            )
        if query_id == selected_query:
            tables["selected_query"] = table

    trace_table, trace_warnings = tg.parse_planner_trace(artifacts.run_dir, composition, trace_query, selected_modes)
    warnings.extend(trace_warnings)
    if not trace_table.empty:
        generated.append(
            GeneratedArtifact(
                f"Selected Query Workflow-Level Mode Comparison ({trace_query})",
                f"table_{trace_query}_workflow_trace",
                "table",
                trace_table,
            )
        )
    tables["workflow_trace"] = trace_table

    diagnostic_table, diagnostic_warnings = tg.build_diagnostic_exceptions_table(
        composition, artifacts.loaded_rows, best_df, selected_query_ids, selected_modes
    )
    warnings.extend(diagnostic_warnings)
    if not diagnostic_table.empty:
        generated.append(
            GeneratedArtifact(
                "Validity, Completeness, Ties, and Diagnostic Exceptions",
                "table_diagnostic_exceptions",
                "table",
                diagnostic_table,
            )
        )
    tables["diagnostics"] = diagnostic_table

    return generated, warnings, tables


def _display_warnings(warnings: list[str], title: str = "Warnings") -> None:
    cleaned = [warning for warning in dict.fromkeys(warnings) if warning]
    if not cleaned:
        return
    with st.expander(f"{title} ({len(cleaned)})", expanded=False):
        for warning in cleaned:
            st.warning(warning)


def _render_validation_panel(
    artifacts: tg.LoadedThesisArtifacts,
    selected_query_ids: list[str],
    selected_modes: list[str],
) -> None:
    st.header("Data Validation")
    validation = tg.validation_summary(artifacts, selected_query_ids, selected_modes)
    cols = st.columns(2)
    cols[0].dataframe(validation["summary"], use_container_width=True, hide_index=True)
    cols[1].dataframe(validation["file_status"], use_container_width=True, hide_index=True)

    detail_tabs = st.tabs(
        [
            "Missing Mode Entries",
            "Duplicate Query-Mode Rows",
            "Malformed or Missing Files",
            "Ranking Included Cases",
            "Ranking Invalid Cases",
            "Ranking Warnings",
        ]
    )
    keys = [
        "missing_mode_entries",
        "duplicate_query_mode_rows",
        "malformed_or_missing",
        "included_cases",
        "invalid_cases",
        "ranking_warnings",
    ]
    for tab, key in zip(detail_tabs, keys):
        with tab:
            df = validation[key]
            if df.empty:
                st.info("None found.")
            else:
                st.dataframe(df, use_container_width=True, hide_index=True)


def render_thesis_results_figure_generator(
    directory_selector: Callable[[str, Path, str], str] | None = None,
    default_run_dir: Path | None = None,
) -> None:
    st.title("Thesis Results Figure Generator")
    st.caption(
        "Generates static thesis figures and CSV/PDF tables from official CSV/JSON artifacts. "
        "This page reads result artifacts only; it does not modify evaluator, ranking, planner, prompt, or experiment outputs."
    )

    default_run_dir = default_run_dir or DEFAULT_RUN_DIR
    with st.sidebar:
        st.header("Results Inputs")
        if directory_selector is not None:
            run_dir_text = directory_selector("Logs run directory", default_run_dir, key="thesis_results_run_dir")
        else:
            run_dir_text = st.text_input("Logs run directory", value=str(default_run_dir), key="thesis_results_run_dir_path")
        run_dir = _resolve_text_path(run_dir_text)

        suggested_summary = tg.resolve_summary_dir(PROJECT_ROOT, run_dir)
        _sync_path_state("thesis_results_summary_dir", suggested_summary)
        st.caption(f"Suggested summary directory: `{suggested_summary}`")
        if st.button("Use suggested summary directory", key="thesis_use_suggested_summary"):
            st.session_state["thesis_results_summary_dir"] = str(suggested_summary)
        summary_dir_text = st.text_input("Summary/result directory", key="thesis_results_summary_dir")
        summary_dir = _resolve_text_path(summary_dir_text)

        report_payload, _warning = tg.load_json_if_exists(summary_dir / "consolidation_report.json")
        suggested_ranking = tg.resolve_ranking_eval_dir(
            PROJECT_ROOT,
            run_dir,
            report_payload if isinstance(report_payload, dict) else {},
        )
        _sync_path_state("thesis_results_ranking_dir", suggested_ranking)
        st.caption(f"Suggested ranking_eval directory: `{suggested_ranking}`")
        if st.button("Use suggested ranking directory", key="thesis_use_suggested_ranking"):
            st.session_state["thesis_results_ranking_dir"] = str(suggested_ranking)
        ranking_dir_text = st.text_input("Ranking eval directory", key="thesis_results_ranking_dir")
        ranking_dir = _resolve_text_path(ranking_dir_text)

        if st.button("Reload thesis artifacts", key="thesis_reload_artifacts"):
            _load_artifacts_cached.clear()
            st.rerun()

    artifacts = _load_artifacts_cached(str(PROJECT_ROOT), str(run_dir), str(summary_dir), str(ranking_dir))
    composition = artifacts.composition_results
    if composition.empty:
        st.error("Composition summary rows were not loaded. Check the summary/result directory.")
        _display_warnings(artifacts.warnings, "Load warnings")
        return

    query_options = tg.sorted_query_ids(composition["Query_ID"].dropna().astype(str)) if "Query_ID" in composition else []
    mode_options = tg.normalize_mode_order(composition["Mode"].dropna().astype(str)) if "Mode" in composition else []
    official_queries = [query for query in tg.official_query_ids(artifacts.consolidation_report, composition) if query in query_options]
    if not official_queries:
        official_queries = query_options

    with st.sidebar:
        st.header("Scope")
        scope_source = st.radio(
            "Official scope",
            ["From consolidation_report.json", "Manual override"],
            index=0,
            key="thesis_scope_source",
        )
        default_queries = official_queries if scope_source == "From consolidation_report.json" else query_options
        selected_query_ids = st.multiselect("Query IDs", query_options, default=default_queries, key="thesis_query_ids")
        selected_modes = st.multiselect("Modes", mode_options, default=mode_options, key="thesis_modes")

        st.header("Exports")
        selected_formats = st.multiselect(
            "Output formats",
            ["PDF", "PNG 300 dpi", "PNG 600 dpi", "CSV"],
            default=["PDF", "PNG 300 dpi", "CSV"],
            key="thesis_output_formats",
        )
        _sync_path_state("thesis_output_dir", DEFAULT_OUTPUT_DIR)
        output_dir_text = st.text_input("Output folder", key="thesis_output_dir")
        output_dir = _resolve_text_path(output_dir_text)

    if not selected_query_ids:
        st.warning("Select at least one query ID.")
        return
    if not selected_modes:
        st.warning("Select at least one mode.")
        return

    selected_query_default = "q02" if "q02" in selected_query_ids else selected_query_ids[0]
    with st.sidebar:
        st.header("Panel Controls")
        selected_query = st.selectbox(
            "Cross-section sensitivity query",
            selected_query_ids,
            index=selected_query_ids.index(selected_query_default),
            key="thesis_selected_query",
        )
        trace_query = st.selectbox(
            "Representative workflow trace query",
            selected_query_ids,
            index=selected_query_ids.index(selected_query_default),
            key="thesis_trace_query",
        )
        available_metric_cols = [col for col in DEFAULT_PER_QUERY_METRICS if col in composition.columns]
        per_query_metrics = st.multiselect(
            "Per-query metrics",
            available_metric_cols,
            default=available_metric_cols,
            key="thesis_per_query_metrics",
        )
        top_n = int(st.number_input("Top-N ranking depth", min_value=1, max_value=100, value=10, step=1))

    generated, generation_warnings, tables = _build_recommended_artifacts(
        artifacts,
        selected_query_ids,
        selected_modes,
        selected_query,
        trace_query,
        per_query_metrics,
        top_n,
    )
    artifact_map = _artifact_by_stem(generated)

    with st.sidebar:
        selected_artifact_label = st.selectbox(
            "Selected figure/table",
            list(artifact_map.keys()),
            key="thesis_selected_artifact",
        )
        generate_selected = st.button("Generate selected figure/table", key="thesis_generate_selected")
        generate_all = st.button("Generate all recommended thesis figures", key="thesis_generate_all")
        save_all = st.button("Save all generated files to output folder", key="thesis_save_all")

    if generate_selected:
        selected_artifact = artifact_map[selected_artifact_label]
        st.success(f"Generated: {selected_artifact.label} -> {selected_artifact.stem}")
        with st.expander("Selected artifact preview", expanded=True):
            if selected_artifact.kind == "table" and selected_artifact.dataframe is not None:
                _render_table(
                    selected_artifact.label,
                    selected_artifact.dataframe,
                    selected_artifact.stem,
                    selected_formats,
                    "selected_preview",
                )
            elif selected_artifact.kind == "figure":
                _render_figure(
                    selected_artifact.label,
                    selected_artifact.figure,
                    selected_artifact.stem,
                    selected_formats,
                    "selected_preview",
                    selected_artifact.caption,
                )
    if generate_all:
        st.success(f"Generated {len(generated)} deterministic figure/table artifacts for the selected scope.")
    if save_all:
        saved = _save_generated_artifacts(generated, output_dir, selected_formats)
        if saved:
            st.success(f"Saved {len(saved)} files to `{output_dir}`.")
            with st.expander("Saved files", expanded=False):
                for path in saved:
                    st.write(f"`{path}`")
        else:
            st.warning("No files were saved. Select at least one export format and ensure artifacts are available.")

    _display_warnings(artifacts.warnings + generation_warnings, "Artifact and generation warnings")
    _render_validation_panel(artifacts, selected_query_ids, selected_modes)

    st.header("Figure and Table Generation")

    panel1, panel2, panel3, panel4 = st.tabs(
        [
            "Scope and Sensitivity",
            "Performance and Components",
            "Ranking and Query Details",
            "Section 5.5 q02 Path Figure",
        ]
    )

    with panel1:
        _render_table(
            "Official Result Scope and Integrity Summary",
            tables.get("official_scope", pd.DataFrame()),
            "table_official_scope_integrity",
            selected_formats,
            "panel1",
        )
        _render_table(
            "Workflow-Level Score Sensitivity Across Weight Settings",
            tables.get("sensitivity_summary", pd.DataFrame()),
            "table_weight_sensitivity_workflow",
            selected_formats,
            "panel2",
        )
        for artifact in generated:
            if artifact.stem in {
                "figure_weight_sensitivity_by_mode",
                "figure_weight_sensitivity_ranking_stability",
                f"figure_weight_sensitivity_{selected_query}",
                "figure_candidate_invalid_top10_by_weight",
            }:
                _render_figure(artifact.label, artifact.figure, artifact.stem, selected_formats, "sensitivity", artifact.caption)
        if not tables.get("candidate_sensitivity", pd.DataFrame()).empty:
            _render_table(
                "Candidate-Level Top-10 Sensitivity Summary",
                tables["candidate_sensitivity"],
                "table_candidate_top10_sensitivity",
                selected_formats,
                "candidate",
            )

    with panel2:
        figure_5_2 = next(
            (item for item in generated if item.stem == "figure_5_2_average_qos_adjusted_composition_score_by_mode"),
            None,
        )
        if figure_5_2 is not None:
            _render_figure(
                figure_5_2.label,
                figure_5_2.figure,
                figure_5_2.stem,
                selected_formats,
                "figure_5_2",
                figure_5_2.caption,
            )
            with st.expander("Figure 5.2 official aggregate source values", expanded=False):
                _render_table(
                    "Figure 5.2 Source Values",
                    tables.get("figure_5_2_source", pd.DataFrame()),
                    "figure_5_2_average_qos_adjusted_composition_score_source",
                    selected_formats,
                    "figure_5_2_source",
                )

        figure_5_7 = next(
            (item for item in generated if item.stem == "figure_5_7_functional_coverage_versus_normalized_qos_by_mode"),
            None,
        )
        if figure_5_7 is not None:
            _render_figure(
                figure_5_7.label,
                figure_5_7.figure,
                figure_5_7.stem,
                selected_formats,
                "figure_5_7",
                figure_5_7.caption,
            )
            with st.expander("Figure 5.7 official aggregate source values", expanded=False):
                _render_table(
                    "Figure 5.7 Source Values",
                    tables.get("figure_5_7_source", pd.DataFrame()),
                    "figure_5_7_functional_coverage_versus_normalized_qos_source",
                    selected_formats,
                    "figure_5_7_source",
                )

        aggregate = tables.get("aggregate", pd.DataFrame())
        _render_table(
            "Aggregate Workflow-Level Performance by Mode",
            aggregate,
            "table_aggregate_mode_performance",
            selected_formats,
            "aggregate",
        )
        if not aggregate.empty and {"qos_pure_llm", "no_qos"}.issubset(set(aggregate["Mode"].astype(str))):
            pure = aggregate.loc[
                aggregate["Mode"].astype(str) == "qos_pure_llm", "Average_QoS_Adjusted_Composition_Score"
            ].iloc[0]
            baseline = aggregate.loc[
                aggregate["Mode"].astype(str) == "no_qos", "Average_QoS_Adjusted_Composition_Score"
            ].iloc[0]
            st.info(f"qos_pure_llm minus no_qos average score difference in selected scope: {pure - baseline:.6f}.")
        for stem in [
            "figure_qos_response_time_by_mode",
            "figure_qos_bottleneck_throughput_by_mode",
            "figure_qos_availability_by_mode",
        ]:
            artifact = next((item for item in generated if item.stem == stem), None)
            if artifact is not None:
                _render_figure(artifact.label, artifact.figure, artifact.stem, selected_formats, "performance", artifact.caption)

        _render_table(
            "Validity, Completeness, Ties, and Diagnostic Exceptions",
            tables.get("diagnostics", pd.DataFrame()),
            "table_diagnostic_exceptions",
            selected_formats,
            "diagnostics",
        )

    with panel3:
        figure_5_8 = next(
            (item for item in generated if item.stem == "figure_5_8_overall_ranking_similarity_across_modes"),
            None,
        )
        if figure_5_8 is not None:
            _render_figure(
                figure_5_8.label,
                figure_5_8.figure,
                figure_5_8.stem,
                selected_formats,
                "figure_5_8",
                figure_5_8.caption,
            )
            with st.expander("Figure 5.8 ranking matrix source values", expanded=False):
                _render_table(
                    "Figure 5.8 Source Matrix Values",
                    tables.get("figure_5_8_source", pd.DataFrame()),
                    "figure_5_8_overall_ranking_similarity_source",
                    selected_formats,
                    "figure_5_8_source",
                )
        figure_5_10 = next(
            (item for item in generated if item.stem == "figure_5_10_hybrid_mode_rationale_and_evidence_flow"),
            None,
        )
        if figure_5_10 is not None:
            _render_figure(
                figure_5_10.label,
                figure_5_10.figure,
                figure_5_10.stem,
                selected_formats,
                "figure_5_10",
                figure_5_10.caption,
            )
        _render_table(
            "Pairwise Ranking Similarity Interpretation",
            tables.get("pairwise", pd.DataFrame()),
            "table_pairwise_ranking_similarity",
            selected_formats,
            "pairwise",
        )

        st.subheader("Selected Query/Subtask Ranking Case")
        if artifacts.loaded_rows.empty:
            st.warning("ranking_eval/loaded_rows.csv is unavailable, so case-level ranking tables cannot be shown.")
        else:
            loaded = artifacts.loaded_rows
            case_queries = [query for query in selected_query_ids if query in set(loaded["query_id"].astype(str))]
            if not case_queries:
                st.warning("No loaded ranking rows match the selected query IDs.")
            else:
                case_cols = st.columns(4)
                case_query = case_cols[0].selectbox("Query ID", case_queries, key="case_query")
                subtask_options = sorted(
                    loaded.loc[loaded["query_id"].astype(str) == case_query, "subtask_id"].dropna().astype(str).unique(),
                    key=lambda value: (len(value), value),
                )
                case_subtask = case_cols[1].selectbox("Subtask ID", subtask_options, key="case_subtask")
                metric = case_cols[2].selectbox(
                    "Ranking metric",
                    list(tg.RANKING_METRIC_LABELS.keys()),
                    format_func=lambda value: tg.RANKING_METRIC_LABELS[value],
                    key="case_metric",
                )
                mode_pair_options = [
                    (left, right)
                    for idx, left in enumerate(selected_modes)
                    for right in selected_modes[idx + 1 :]
                ]
                mode_pair_labels = [f"{left} vs {right}" for left, right in mode_pair_options]
                selected_pair_label = case_cols[3].selectbox("Mode pair", mode_pair_labels, key="case_pair")
                left_mode, right_mode = mode_pair_options[mode_pair_labels.index(selected_pair_label)]
                case_df, topn_wide, selected_rows = tg.case_mode_ranking_table(
                    loaded, case_query, case_subtask, selected_modes, top_n
                )
                st.dataframe(case_df, use_container_width=True, hide_index=True)
                st.caption(f"Top-{top_n} APIs by selected modes")
                st.dataframe(topn_wide, use_container_width=True, hide_index=True)
                if selected_rows.empty:
                    st.warning("No selected_for_planner rows are available for this selected case.")
                else:
                    st.caption("Rows marked selected_for_planner")
                    st.dataframe(selected_rows, use_container_width=True, hide_index=True)

                if not case_df.empty:
                    matrix = tg.compute_case_similarity_matrix(case_df, selected_modes, metric, top_n)
                    fig = tg.plot_case_similarity_heatmap(matrix, tg.RANKING_METRIC_LABELS[metric])
                    _render_figure(
                        "Case-level mode similarity heatmap",
                        fig,
                        f"figure_case_{case_query}_subtask_{case_subtask}_{metric}_heatmap",
                        selected_formats,
                        "case_heatmap",
                    )
                    lists = tg.ranked_api_lists(case_df, [left_mode, right_mode], top_n=None)
                    if left_mode in lists and right_mode in lists:
                        depth_df = tg.average_overlap_by_depth(lists[left_mode], lists[right_mode], top_n)
                        fig = tg.plot_overlap_by_depth(depth_df, left_mode, right_mode)
                        _render_figure(
                            "Average overlap by depth for selected mode pair",
                            fig,
                            f"figure_case_{case_query}_subtask_{case_subtask}_{left_mode}_vs_{right_mode}_overlap_by_depth",
                            selected_formats,
                            "case_overlap",
                        )

        _render_table(
            "LLM-TOPSIS Agreement and Disagreement by Query/Subtask",
            tables.get("agreement", pd.DataFrame()),
            "table_llm_topsis_agreement",
            selected_formats,
            "agreement",
        )
        artifact = next((item for item in generated if item.stem == "figure_llm_topsis_agreement_rate_by_query"), None)
        if artifact:
            _render_figure(artifact.label, artifact.figure, artifact.stem, selected_formats, "agreement_fig", artifact.caption)

        _render_table(
            "Query-Level Final Score Matrix",
            tables.get("query_matrix", pd.DataFrame()),
            "table_query_level_score_matrix",
            selected_formats,
            "query_matrix",
        )
        for stem in ["figure_query_level_grouped_scores", "figure_query_winner_tie_heatmap"]:
            artifact = next((item for item in generated if item.stem == stem), None)
            if artifact:
                _render_figure(artifact.label, artifact.figure, artifact.stem, selected_formats, "query_fig", artifact.caption)

        st.subheader("Detailed Per-Query Result Analysis")
        query_text = _selected_query_text(composition, selected_query)
        if query_text:
            st.caption(f"{selected_query}: {query_text}")
        selected_query_table = tables.get("selected_query", pd.DataFrame())
        _render_table(
            f"Per-query metrics ({selected_query})",
            selected_query_table,
            f"table_per_query_{selected_query}_metrics",
            selected_formats,
            "per_query",
        )
        for stem in [f"figure_per_query_{selected_query}_score_by_mode", f"figure_per_query_{selected_query}_components"]:
            artifact = next((item for item in generated if item.stem == stem), None)
            if artifact:
                _render_figure(artifact.label, artifact.figure, artifact.stem, selected_formats, "per_query_fig", artifact.caption)

        st.subheader("Representative Workflow Trace")
        _render_table(
            f"Selected Query Workflow-Level Mode Comparison ({trace_query})",
            tables.get("workflow_trace", pd.DataFrame()),
            f"table_{trace_query}_workflow_trace",
            selected_formats,
            "trace",
        )

        extended_ids = [query for query in ["q16", "q17", "q18"] if query in query_options]
        if extended_ids:
            with st.expander("Optional Extended Tie-Case Analysis", expanded=False):
                selected_extended = [query for query in extended_ids if query in selected_query_ids]
                if selected_extended:
                    st.warning(
                        "q16-q18 are included in the current manual scope. Figures and tables are recomputed from the selected query IDs."
                    )
                else:
                    st.info("q16-q18 exist in the loaded rows but are outside the selected official scope.")
                extended_rows = composition[composition["Query_ID"].astype(str).isin(extended_ids)]
                st.dataframe(extended_rows, use_container_width=True, hide_index=True)

    with panel4:
        _render_section_55_q02_path_dashboard(artifacts, composition)
