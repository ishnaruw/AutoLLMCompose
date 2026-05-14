# API Catalog Layout

This directory now uses a three-file canonical runtime layout:

- `api_repo.tooldesc.jsonl`: base functional API catalog without QoS fields.
- `api_repo.enriched.jsonl`: ToolBench-enriched functional catalog used by runtime ranking and evaluation prompts.
- `api_qos.jsonl`: compact QoS overlay keyed by `api_id`.

Runtime loading should use `src.tools.fetch_services.fetch_services()` or
`src.tools.fetch_services.load_catalog_map()`. When `with_qos=False`, the loader
returns functional catalog rows without QoS. When `with_qos=True`, it merges
`api_qos.jsonl` into the functional catalog by `api_id`.

## Deprecated Legacy Files

These files are retained only as legacy generation inputs, fallbacks, or
historical references:

- `api_repo.no_qos.tooldesc.jsonl`
- `api_repo.with_qos.tooldesc.jsonl`
- `deprecated_api_repo.no_qos.jsonl`
- `deprecated_api_repo.with_qos.jsonl`

Do not use the deprecated split catalogs for new runtime code. Regenerate the
canonical files with:

```bash
python -m src.tools.build_enriched_catalog
```

