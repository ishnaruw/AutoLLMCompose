from pathlib import Path

# Query input file used by the main pipeline driver.
# Update this path when switching between query sets.
QUERIES_PATH = Path("data/queries/user_queries.jsonl")

# Prefix each run directory with the query id for easy identification.
# Example: q07_20260310T145117
PREFIX_RUN_DIR_WITH_QUERY_ID = True
