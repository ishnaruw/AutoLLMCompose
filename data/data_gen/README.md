# Archival Data Generation Notes

ARCHIVAL: this directory is not required for normal setup or runtime use. The
current API catalog, QoS overlay, and FAISS index are committed in the repo.

These notebooks are kept only as historical/reproducibility material for how
earlier catalog snapshots were explored or generated:

- `api_counts_by_category.ipynb`: exploratory ToolBench category counts.
- `endpoint_inventory.ipynb`: older catalog sampling and ToolBench/WS-DREAM
  processing workflow.

Running these notebooks requires external local data paths such as ToolBench and
WS-DREAM and may produce a different catalog than the committed snapshot. Treat
them as archival research material, not part of the standard pipeline.

For the maintained catalog-refresh path, prefer:

```bash
python -m src.tools.build_enriched_catalog --toolbench-root /path/to/ToolBench/data/toolenv/tools
python -m src.rag.index_build --index_dir data/index/faiss_no_qos
```

Only refresh and commit generated catalog/index files when intentionally changing
the project dataset.
