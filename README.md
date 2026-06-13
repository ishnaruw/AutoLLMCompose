# AutoLLMCompose

AutoLLMCompose is a research codebase for multi-agent API discovery, ranking, composition, and evaluation. Given a user goal, the pipeline decomposes the goal into API-retrieval subtasks, retrieves candidate APIs from a FAISS-backed catalog index, ranks them under multiple QoS and non-QoS modes, generates executable composition plans, and writes evaluation artifacts for thesis analysis.

## What AutoLLMCompose Does

The current pipeline supports:

- Query decomposition into ordered API subtasks.
- Shared semantic retrieval from the local API catalog index.
- Candidate ranking across four modes: `no_qos`, `qos_pure_llm`, `qos_topsis`, and `qos_hybrid`.
- Deterministic TOPSIS scoring from QoS metrics.
- LLM-based planning over selected APIs.
- Functional-match, hallucination, duplicate, ranking-anomaly, and composition-QoS evaluation outputs.
- A Streamlit dashboard for running experiments and inspecting completed runs.

## Repository Layout

```text
AutoLLMCompose/
|-- data/
|   |-- processed/api_catalog_sample_balanced/
|   |   |-- api_repo.tooldesc.jsonl        # Functional catalog without QoS
|   |   |-- api_repo.enriched.jsonl        # ToolBench-enriched runtime catalog
|   |   |-- api_qos.jsonl                  # QoS overlay keyed by api_id
|   |   |-- enrichment_manifest.json       # Catalog generation provenance
|   |   `-- README.md                      # Catalog layout notes
|   |-- queries/
|   |   |-- all_user_query.jsonl           # Main batch query set
|   |-- index/faiss_no_qos/
|   |                                      # Committed default FAISS index
|   |-- data_gen/                          # ARCHIVAL notebooks, not runtime setup
|   |-- raw/wsdream/                       # ARCHIVAL raw matrices
|   `-- results/api_inventory/             # ARCHIVAL inventory reports
|-- prompts/                               # LLM prompt templates
|-- src/
|   |-- agents/                            # Decomposer, retriever, ranker, planner, evaluator
|   |-- config/pipeline_config.py          # Central pipeline defaults
|   |-- core/                              # Schemas, parsing, retry, logging helpers
|   |-- driver/run_autogen_pipeline.py     # Main experiment runner
|   |-- eval/                              # Evaluation and audit scripts
|   |-- llm/                               # Provider backends and AutoGen gateway
|   |-- rag/                               # FAISS index build and retrieval
|   |-- tools/                             # Catalog build/backfill utilities
|   `-- ui/ranking_eval_app.py             # Streamlit dashboard
|-- tests/                                 # Unit tests
|-- requirements.txt
`-- README.md
```

## Requirements

- Python 3.12 is used in the current local environment. Python 3.10+ should work for most code, but use 3.12 if you want to match this setup closely.
- An LLM provider key or a local LM Studio server.

## Installation

Run all commands from the repository root.

```bash
cd AutoLLMCompose
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The retrieval layer requires FAISS. If your environment does not already provide it, install:

```bash
python -m pip install faiss-cpu
```

On Apple Silicon or other platforms where FAISS wheels are unavailable, install FAISS through conda:

```bash
conda install -c conda-forge faiss-cpu
```

## Environment Variables

Create a local `.env` file in the repository root. The code loads this file
automatically. Do not commit it.

Set `LLM_PROVIDER` to your default provider, or pass `--provider` on the command line.

```bash
# Provider selection
LLM_PROVIDER=groq

# Azure OpenAI
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-05-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o-dspy

# Mistral
MISTRAL_API_KEY=
MISTRAL_MODEL=mistral-large-latest

# Groq
GROQ_API_KEY=
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_MULTI_MODEL_MODE=false
GROQ_MULTI_MODELS=llama-3.3-70b-versatile,openai/gpt-oss-120b,openai/gpt-oss-20b,llama-3.1-8b-instant,qwen/qwen3-32b

# Fireworks AI
FIREWORKS_API_KEY=
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
FIREWORKS_MODEL=accounts/fireworks/models/gpt-oss-120b

# Together AI
TOGETHER_API_KEY=
TOGETHER_BASE_URL=https://api.together.xyz/v1
TOGETHER_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo

# Gemini
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash

# Azure AI Foundry
AZURE_FOUNDRY_API_KEY=
AZURE_FOUNDRY_ENDPOINT=https://your-resource.services.ai.azure.com/models
AZURE_FOUNDRY_API_VERSION=2024-05-01-preview
AZURE_FOUNDRY_MODEL=DeepSeek-R1-0528
AZURE_FOUNDRY_TIMEOUT_SECONDS=300

# LM Studio, OpenAI-compatible local server
LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1
LMSTUDIO_MODEL=meta-llama-3.1-8b-instruct
LMSTUDIO_API_KEY=lmstudio

# LM Studio native chat endpoint used by lmstudio_qwen
LMSTUDIO_QWEN_CHAT_URL=http://localhost:1234/api/v1/chat
LMSTUDIO_QWEN_MODEL=qwen2.5-3b-instruct.gguf
```

Supported provider names are:

```text
azure, mistral, groq, fireworks, together, gemini, azure_foundry, lmstudio, lmstudio_qwen
```

## Data And Index Setup

The tracked catalog files live in:

```text
data/processed/api_catalog_sample_balanced/
```

Runtime loading uses:

- `api_repo.enriched.jsonl` for functional API evidence.
- `api_qos.jsonl` for QoS values.
- `api_repo.tooldesc.jsonl` as the base functional catalog.

Runtime code should use the canonical catalog files above.

The default FAISS index is committed at `data/index/faiss_no_qos/` so a fresh clone can run the main pipeline without rebuilding retrieval artifacts first.

If you change the catalog or want to regenerate the index, run:

```bash
python -m src.rag.index_build \
  --index_dir data/index/faiss_no_qos \
  --embed_model sentence-transformers/all-MiniLM-L6-v2
```

This creates or refreshes:

```text
data/index/faiss_no_qos/
|-- faiss.index
|-- meta.jsonl
`-- config.json
```

Commit the refreshed index files when the catalog and embedding configuration are intentionally updated.

Normal users do not need the external ToolBench source tree. Only regenerate the
enriched catalog when intentionally changing the committed catalog snapshot:

```bash
python -m src.tools.build_enriched_catalog \
  --toolbench-root /path/to/ToolBench/data/toolenv/tools
```

Regeneration can change catalog evidence and derived QoS overlays, so commit the
new catalog, manifest, and rebuilt FAISS index together.

## Running The Pipeline

Run one or more query IDs non-interactively:

```bash
python -m src.driver.run_autogen_pipeline \
  --query-ids q01 \
  --provider groq \
  --model llama-3.3-70b-versatile \
  --run-tag DEV_RUN
```

Run multiple queries:

```bash
python -m src.driver.run_autogen_pipeline \
  --query-ids q01,q02,q03 \
  --provider fireworks \
  --model deepseek-v3p1 \
  --run-tag FIREWORKS_DEV
```

You can also pass repeated query IDs:

```bash
python -m src.driver.run_autogen_pipeline \
  --query-id q01 \
  --query-id q05 \
  --provider mistral
```

If you omit `--query-ids` and `--query-id`, the script opens an interactive query and provider selector:

```bash
python -m src.driver.run_autogen_pipeline
```

The default query file is `data/queries/all_user_query.jsonl`. To use a custom
query set, create a JSONL file with the same `id`, `title`, and `goal` fields,
then pass it with `--queries-path`.

## Pipeline Stages

For each selected query, the driver performs:

1. Decomposition: writes `0_decomposer.json`.
2. Shared retrieval: writes `1_retriever_s<subtask>.json`.
3. Functional candidate labeling: writes retrieval functional-match evaluation rows.
4. Ranking for each mode:
   - `no_qos`
   - `qos_pure_llm`
   - `qos_topsis`
   - `qos_hybrid`
5. Planner input selection and planning for each mode.
6. Evaluation output generation.
7. Composition-QoS evaluation when planning is enabled.

The mode behavior is:

| Mode | Ranking Signal | Description |
| --- | --- | --- |
| `no_qos` | Functional evidence only | LLM ranks retrieved APIs without QoS fields. |
| `qos_pure_llm` | Functional gate + LLM QoS scoring | Uses functional-refinement labels as a gate, then ranks functionally suitable APIs with QoS-aware LLM prompt context. |
| `qos_topsis` | Deterministic TOPSIS | Ranks by computed QoS closeness score. |
| `qos_hybrid` | Functional match + TOPSIS | Places functionally matching APIs first, ordered by TOPSIS. |

## Outputs

Runs are written under:

```text
results/logs/<run_tag>/<provider_model>/<query_id_timestamp>/
```

Important files in each query run include:

```text
meta.json                         # Run status, timings, provider/model metadata
run_config.json                   # Pipeline config snapshot
run.log                           # Stage-level run log
model_usage.json                  # Model usage and failover metadata
0_decomposer.json                 # Decomposed subtasks
1_retriever_s<id>.json            # Retrieved candidates per subtask
evaluation_result.json            # Pointers to evaluation outputs
<mode>/2_ranked_s<id>.json        # Ranked APIs per mode/subtask
<mode>/3_selected_s<id>.json      # Planner input selection per mode/subtask
<mode>/4_planner.json             # Planner output per mode
evaluation/                       # Excel, JSON, audit, and composition-QoS reports
```

The evaluation folder can contain:

- `query_<id>_candidate_api_rankings.xlsx`
- `query_<id>_candidate_api_rankings_rows.json`
- duplicate and hallucination audit JSON
- mode anomaly reports
- planner selection-K summaries
- composition-QoS rows, summary, and workbook outputs

## Dashboard

Launch the Streamlit dashboard:

```bash
python -m streamlit run src/ui/ranking_eval_app.py
```

The dashboard includes pages for:

- Live Demo Deep Dive for dynamic defense walkthroughs of any query in a selected run folder.
- Ranking evaluation.
- Composition visualizations.
- Launching experiment runs.
- Browsing completed runs.

Experiment runs launched from the UI write logs under `results/logs/streamlit_launches/` and run outputs under the selected run tag.

## Defense Live Demo: Dynamic Query Deep Dive

Use the `Live Demo Deep Dive` dashboard page for the thesis defense walkthrough. Select a run folder from the
sidebar, then select any discovered query from that run folder. The page renders the same live visualization layout
from that query's actual artifacts; it is not hardcoded to q07 or q14. If q07 exists in the selected run folder, the
query selector defaults to q07; otherwise it defaults to the first discovered qXX timestamped query folder.
Query category/domain labels are read from `data/queries/all_user_query.jsonl` when a run artifact does not record
them.

Main goal: Show that QoS-Pure-LLM beats No-QoS. Explain QoS-Hybrid as a bonus functional-first QoS refinement.
QoS-TOPSIS is shown as a diagnostic mode for cases where QoS-only ranking can lower Functional Coverage.

Run command:

```bash
python -m streamlit run src/ui/ranking_eval_app.py
```

The page order is mechanism-first:

1. Query Context
2. Decomposed Subtasks
3. RAG Retrieval Snapshot with a catalog-backed candidate inspector
4. Re-ranking Motivation
5. Ranking by Mode
6. Ranking Difference Visualization with pairwise mode similarity diagnostics
7. Selected Composition Path with selected-API and QoS figures
8. Score Comparison
9. Formula Proof
10. Hypothesis Proof
11. Dynamic Query Takeaway
12. Raw Artifacts, shown only when enabled from the sidebar

Sidebar controls include run folder, dynamic query selector, focused subtask selector, Top-K, show-all-40 toggle
enabled by default, exact-value toggle, raw-artifact toggle, and optional mode visibility. Main-view metrics are loaded from result
artifacts without rewriting official experiment scores; derived rank-similarity and score-component views are labeled
as diagnostics.

Timing:

- 0:00-0:30: Select run folder and q07 from the sidebar. Explain that the page is dynamic.
- 0:30-3:00: q07 deep dive. Show QoS-Pure-LLM improving over No-QoS by improving Normalized QoS while preserving Functional Coverage.
- 3:00-5:15: q14 deep dive. Show QoS-Pure-LLM beating No-QoS and tying QoS-Hybrid.
- 5:15-6:00: Optional q03 or q13, or close with the scoring formula and limitations.

Optional query purposes:

- q03: technical/security query where QoS-Pure-LLM improves and QoS-Hybrid reaches 1.0.
- q13: niche/non-popular query with strong QoS-Pure-LLM improvement and near-perfect Hybrid.
- q04: popular travel query that shows why QoS-TOPSIS alone can fail despite high QoS.

## Evaluation Scripts

Evaluate ranking agreement across a parent run directory:

```bash
python -m src.eval.run_ranking_eval \
  results/logs/DEV_RUN/groq_llama-3.3-70b-versatile \
  --output-dir results/logs/DEV_RUN/ranking_eval
```

Run composition-QoS evaluation for one query run:

```bash
python -m src.eval.composition_qos_eval \
  results/logs/DEV_RUN/groq_llama-3.3-70b-versatile/q01_YYYYMMDDTHHMMSS \
  --query-id q01 \
  --output-dir results/logs/DEV_RUN/groq_llama-3.3-70b-versatile/q01_YYYYMMDDTHHMMSS/evaluation
```

Backfill selected reports when needed:

```bash
python -m src.tools.backfill_candidate_api_rankings_reports --help
python -m src.tools.backfill_mode_anomaly_reports --help
```

## Testing

Run the full unit test suite:

```bash
python -m unittest discover -s tests
```

Run focused tests while changing a subsystem:

```bash
python -m unittest tests.test_json_parsing tests.test_output_schemas
python -m unittest tests.test_ranking_metrics
python -m unittest tests.test_composition_qos_eval
```

For a quick syntax check:

```bash
python -m compileall src tests
```

## Configuration

Pipeline defaults are defined in `src/config/pipeline_config.py`. Common values:

- `run_tag`: default output folder under `results/logs/`.
- `shared_index_dir`: FAISS index path.
- `catalog_enriched_path`: runtime functional catalog path.
- `api_qos_path`: QoS overlay path.
- `rag_top_k`: candidates retrieved per subtask.
- `ranker_max_candidates`: ranker candidate cap.
- `selector_top_n`: fallback number of APIs selected for planner input.
- `planner_enabled`: enables planner generation.
- `planner_temperature`: temperature used only for planner LLM calls.
- `composition_qos_eval_enabled`: enables composition-level QoS evaluation.
- `llm_validation_max_retries`: bounded retries for structurally invalid LLM outputs.

Prefer changing these defaults in code only when you want a persistent project-wide behavior change. For one-off experiments, use CLI flags such as `--provider`, `--model`, `--query-ids`, `--queries-path`, and `--run-tag`.

## Troubleshooting

`RuntimeError: faiss is required`

Install `faiss-cpu` with pip or conda, then rebuild the index if needed.

`FileNotFoundError` for `data/index/faiss_no_qos/faiss.index`

The default index should be included in the repository. If it is missing or stale, rebuild it:

```bash
python -m src.rag.index_build --index_dir data/index/faiss_no_qos
```

Provider key errors such as `GROQ_API_KEY missing`

Add the required key to `.env` or pass a different provider with `--provider`.

LM Studio timeouts or connection errors

Start the LM Studio local server and verify the configured URL:

```bash
LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1
LMSTUDIO_QWEN_CHAT_URL=http://localhost:1234/api/v1/chat
```

No services loaded from catalog

Check that `data/processed/api_catalog_sample_balanced/api_repo.enriched.jsonl`
exists and that the repository data files were pulled correctly. Catalog
regeneration is a maintainer workflow, not a normal setup step.

## Notes For New Contributors

- Keep generated outputs under `results/`; this directory is ignored by git.
- Keep local secrets in `.env`; it is ignored by git.
- Rebuild the FAISS index after changing the functional catalog.
- Use the JSON sidecars as the detailed source of truth for evaluation. Excel workbooks are user-facing reports.
- Do not overwrite historical run artifacts unless the task explicitly asks for regeneration.

## Citation

If you use this framework or datasets, please cite:

```bibtex
@research{Subramanian2025AutoLLMCompose,
  title={AutoLLMCompose: Multi-Agent LLM Framework for Service Discovery and Composition},
  author={Ishwarya Narayana Subramanian and Eyhab Al-Masri},
  year={2025},
  institution={University of Washington Tacoma}
}
```

## License

MIT License (c) 2025 Ishwarya Narayana Subramanian.
See [LICENSE](LICENSE) for details.

## Acknowledgments

- **Prof. Eyhab Al-Masri**, University of Washington Tacoma - ealmasri@uw.edu
- Supported by the University of Washington Master's in Computer Science & Systems program

**Researcher:** Ishwarya Narayana Subramanian, University of Washington Tacoma  
Contact: ishnaruw@uw.edu
