# Research Figure Scripts

These scripts generate the Chapter 5 research figures from a completed experiment run folder. They do not use thesis figure numbers or hard-coded result values. Every plotted value is read from the `qXX_*` query folders or from `ranking_eval` inside the run folder.

Example run folder:

```bash
AutoLLMCompose/results/logs/RUNS_MAY_31_NEW_5/fireworks_gpt-oss-120b
```

All generated PNG files are written to:

```bash
<run-folder>/figures
```

## Generate Everything

```bash
python AutoLLMCompose/scripts/generate_research_figures.py \
  AutoLLMCompose/results/logs/RUNS_MAY_31_NEW_5/fireworks_gpt-oss-120b \
  --path-query q02 \
  --panel-queries q01,q02,q03,q04,q05 \
  --panel-queries q06,q07,q08,q09,q10 \
  --panel-queries q11,q12,q13,q14,q15
```

If `--path-query` is omitted, the first discovered query folder is used. If `--panel-queries` is omitted, panels are generated in groups of five discovered queries.

## Individual Figures

Selected API path comparison for one query:

```bash
python AutoLLMCompose/scripts/plot_api_path_comparison.py <run-folder> --query q02
```

Aggregate mode summaries:

```bash
python AutoLLMCompose/scripts/plot_mode_summary.py <run-folder>
```

This writes:

- `average_composition_by_mode.png`
- `coverage_qos_tradeoff.png`
- `functional_coverage_normalized_qos.png`
- `qos_component_metrics.png`

Ranking similarity heatmaps:

```bash
python AutoLLMCompose/scripts/plot_ranking_similarity.py <run-folder>
```

Query-level grouped scores and winner heatmap:

```bash
python AutoLLMCompose/scripts/plot_query_score_overview.py <run-folder>
```

Query score panels for any comma-separated query set:

```bash
python AutoLLMCompose/scripts/plot_query_score_panels.py <run-folder> --queries q01,q02,q03,q04,q05
```

Query IDs may be written as `q01`, `01`, or `1`.

## Output Names

The scripts use readable functional names instead of thesis figure numbers:

- `selected_api_path_qXX.png`
- `average_composition_by_mode.png`
- `coverage_qos_tradeoff.png`
- `ranking_similarity_heatmaps.png`
- `query_composition_scores.png`
- `winner_status_heatmap.png`
- `query_score_panels_qXX_qYY.png`
- `functional_coverage_normalized_qos.png`
- `qos_component_metrics.png`

## Data Requirements

Each query folder must contain:

- `0_decomposer.json`
- `<mode>/4_planner.json`
- `evaluation/query_qXX_composition_qos_eval_summary.json`
- `evaluation/query_qXX_candidate_api_rankings_rows.json` for functional-match labels in the API path figure

The ranking similarity script also requires:

- `ranking_eval/spearman_matrix.csv`
- `ranking_eval/average_overlap_matrix.csv`
- `ranking_eval/rbo_matrix.csv`
- `ranking_eval/jaccard_matrix.csv`

If a required artifact is missing or malformed, the scripts fail with a source-file-specific error instead of inventing values.
