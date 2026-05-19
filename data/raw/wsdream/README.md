# Archival WS-DREAM Data

ARCHIVAL: these files are not read by the current AutoLLMCompose runtime
pipeline.

`dataset1/` contains the original WS-DREAM response-time and throughput matrix
files that were used by older catalog-generation experiments:

- `rtMatrix.txt` and `rtMatrix.csv`: response-time matrix.
- `tpMatrix.txt`: throughput matrix.
- `userlist.txt`: service-user IDs.
- `wslist.txt`: web-service IDs.
- `readme.txt`: original dataset notes.

The committed runtime QoS overlay is
`data/processed/api_catalog_sample_balanced/api_qos.jsonl`. New users should use
that committed overlay instead of regenerating QoS from these raw matrices.

Keep this directory for thesis reproducibility only.
