from __future__ import annotations

import pandas as pd

from src.ui import thesis_results_page as page


def test_q02_path_diagram_uses_shapes_and_arrows_not_table_trace() -> None:
    figure_df = pd.DataFrame(
        [
            {
                "Mode": "no_qos",
                "Subtask 1": "api_search\nFM=1",
                "Subtask 2": "api_reviews\nFM=1",
                "Subtask 3": "api_email\nFM=1",
                "Functional Coverage": "1",
                "Normalized QoS": "0.887325",
                "Final Score": "0.966197",
            },
            {
                "Mode": "qos_topsis",
                "Subtask 1": "api_bad_search\nFM=0",
                "Subtask 2": "api_reviews\nFM=1",
                "Subtask 3": "api_bad_email\nFM=0",
                "Functional Coverage": "0.333333",
                "Normalized QoS": "0.849805",
                "Final Score": "0.488275",
            },
        ]
    )

    fig = page._build_q02_path_plotly_diagram(figure_df)

    assert all(trace.type != "table" for trace in fig.data)
    assert len(fig.layout.shapes) >= 20
    assert sum(1 for annotation in fig.layout.annotations if annotation.showarrow) == 4
    assert any("api_search" in str(annotation.text) for annotation in fig.layout.annotations)
