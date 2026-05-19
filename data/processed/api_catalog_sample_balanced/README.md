# API Catalog Files

This directory contains the committed API catalog snapshot used by
AutoLLMCompose at runtime.

## Runtime Files

- `api_repo.enriched.jsonl`: primary functional API catalog loaded by retrieval, ranking, planning, and evaluation code.
- `api_qos.jsonl`: QoS overlay keyed by `api_id`; merged into catalog rows when `with_qos=True`.
- `api_repo.tooldesc.jsonl`: base functional catalog without QoS; kept as the stable source for rebuilding the enriched catalog.
- `enrichment_manifest.json`: provenance for the current generated snapshot.

Normal project runs should not need to regenerate these files.

## Loader Behavior

Use `src.tools.fetch_services.fetch_services()` or
`src.tools.fetch_services.load_catalog_map()` instead of reading these files
directly.

- `with_qos=False`: loads functional rows from `api_repo.enriched.jsonl` and removes any QoS field.
- `with_qos=True`: loads `api_repo.enriched.jsonl` and merges `api_qos.jsonl` by `api_id`.

## Misc Files

`misc/` stores legacy generation inputs, fallback catalogs, and one-off reports:

- `api_repo.balanced_counts.csv`
- `api_repo.no_qos.tooldesc.jsonl`
- `api_repo.with_qos.tooldesc.jsonl`
- `deprecated_api_repo.no_qos.jsonl`
- `deprecated_api_repo.with_qos.jsonl`

These files are retained for reproducibility and fallback behavior only. New
runtime code should use the canonical files listed above.
