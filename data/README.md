# Data Directory

AutoLLMCompose uses the committed processed catalog, query set, and FAISS index
for normal runs.

## Runtime Data

- `queries/all_user_query.jsonl`: query set used by the pipeline driver.
- `processed/api_catalog_sample_balanced/`: committed API catalog snapshot and
  QoS overlay.
- `index/AutoLLMCompose_v3/shared_no_qos/`: committed FAISS retrieval index.

## Archival Data

The following directories are kept for thesis transparency and reproducibility,
but they are not part of the normal runtime path:

- `data_gen/`: historical notebooks for exploratory catalog generation.
- `raw/wsdream/`: original WS-DREAM matrix files used by older generation work.
- `results/api_inventory/`: ToolBench endpoint count reports produced by the
  archival notebooks.

Do not rerun archival notebooks or regenerate catalog files unless you are
intentionally changing the project dataset.
