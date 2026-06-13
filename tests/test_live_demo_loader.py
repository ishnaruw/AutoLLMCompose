from __future__ import annotations

import json
from pathlib import Path

from src.ui import live_demo_catalog as catalog
from src.ui import live_demo_loader as loader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT_ROOT / "results/logs/RUNS_MAY_31_NEW_5/fireworks_gpt-oss-120b"


def test_scan_run_folder_discovers_dynamic_queries() -> None:
    scan = loader.scan_run_folder(str(RUN_DIR))

    query_ids = {row["query_id"] for row in scan["queries"]}

    assert "q07" in query_ids
    assert "q14" in query_ids
    assert all("query_dir" in row for row in scan["queries"])


def test_live_demo_query_loads_scores_and_selected_paths() -> None:
    bundle = loader.load_live_demo_query(str(RUN_DIR), "q07")

    rows = {row["Mode"]: row for row in bundle["composition_rows"]}
    paths = bundle["selected_paths"]

    assert bundle["available"] is True
    assert float(rows["qos_pure_llm"]["QoS_Adjusted_Composition_Score"]) > float(rows["no_qos"]["QoS_Adjusted_Composition_Score"])
    assert paths["no_qos"]
    assert paths["qos_pure_llm"]
    assert any(row["selected_for_planner"] == "Yes" for row in bundle["ranking_rows"] if row["mode"] == "qos_pure_llm")
    assert any(row["subtask_id"] == "1" for row in bundle["retrieval_rows"])


def test_live_demo_query_enriches_retrieval_and_ranking_rows() -> None:
    bundle = loader.load_live_demo_query(str(RUN_DIR), "q07")

    retrieval_row = next(row for row in bundle["retrieval_rows"] if row["api_id"] == "newscatcher_v1_aggregation")
    ranking_row = next(
        row
        for row in bundle["ranking_rows"]
        if row["mode"] == "qos_pure_llm" and row["api_id"] == "newsnow_news_powered_by_duck_duck_go"
    )
    selected_retrieval_row = next(row for row in bundle["retrieval_rows"] if row["api_id"] == "newsnow_news_powered_by_duck_duck_go")

    assert retrieval_row["display_name"]
    assert retrieval_row["category"] == "News_Media"
    assert ranking_row["selected_for_planner"] == "Yes"
    assert ranking_row["category"] == "News_Media"
    assert "QoS-Pure-LLM" in selected_retrieval_row["selected_by_modes"]
    assert bundle["query_category"] == "News and Media"
    assert bundle["query_domain"] == "News summarization and SMS briefing"


def test_official_query_aggregate_averages_scores_across_loaded_queries(tmp_path: Path) -> None:
    q01_eval = tmp_path / "q01_20260531T000001" / "evaluation"
    q02_eval = tmp_path / "q02_20260531T000002" / "evaluation"
    q01_eval.mkdir(parents=True)
    q02_eval.mkdir(parents=True)
    (q01_eval / "query_q01_composition_qos_eval_rows.json").write_text(
        json.dumps(
            [
                {"Query_ID": "q01", "Mode": "no_qos", "QoS_Adjusted_Composition_Score": 0.2, "Functional_Coverage": 0.5},
                {"Query_ID": "q01", "Mode": "qos_pure_llm", "QoS_Adjusted_Composition_Score": 0.8, "Functional_Coverage": 1.0},
            ]
        ),
        encoding="utf-8",
    )
    (q02_eval / "query_q02_composition_qos_eval_rows.json").write_text(
        json.dumps(
            [
                {"Query_ID": "q02", "Mode": "no_qos", "QoS_Adjusted_Composition_Score": 0.4, "Functional_Coverage": 0.7},
            ]
        ),
        encoding="utf-8",
    )

    aggregate = loader.load_official_query_aggregate(str(tmp_path))
    rows = {row["Mode"]: row for row in aggregate["score_rows"]}
    components = {row["Mode"]: row for row in aggregate["component_rows"]}

    assert aggregate["found_query_ids"] == ["q01", "q02"]
    assert rows["no_qos"]["Number_of_queries_included"] == 2
    assert rows["no_qos"]["Missing_query_count"] == 13
    assert abs(rows["no_qos"]["Average_QoS_Adjusted_Composition_Score"] - 0.3) < 1e-12
    assert rows["qos_pure_llm"]["Number_of_queries_included"] == 1
    assert rows["qos_pure_llm"]["Missing_query_count"] == 14
    assert rows["qos_pure_llm"]["Average_QoS_Adjusted_Composition_Score"] == 0.8
    assert abs(components["no_qos"]["Average_Functional_Coverage"] - 0.6) < 1e-12


def test_catalog_helper_loads_inspector_metadata() -> None:
    loaded = catalog.load_api_catalog()
    metadata = catalog.find_api_metadata(loaded, "smsto_send_campaign_message-2")

    assert metadata is not None
    assert metadata["tool_name"] == "SMSto"
    assert metadata["category"] == "SMS"
    assert metadata["http_method"] == "POST"
    assert metadata["source_file"].endswith("api_repo.enriched.jsonl")
