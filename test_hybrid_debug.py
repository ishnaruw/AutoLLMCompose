#!/usr/bin/env python3
"""Debug script to test hybrid ranking logic"""

import json
from pathlib import Path

# Load the relevancy cache
cache_path = Path("results/relevancy_eval/relevancy_cache.json")
cache_data = json.loads(cache_path.read_text())

# Test ORIGINAL parsing (WRONG - loads all queries)
print("=== ORIGINAL PARSING (WRONG) ===")
relevancy_map_old = {}
for cache_key, cache_val in cache_data.items():
    parts = cache_key.split("_", 2)
    if len(parts) >= 3:
        sub_id = parts[1]
        api_id = "_".join(parts[2:])
        relevancy_map_old[(sub_id, api_id)] = cache_val

relevant_count_old = sum(1 for (sid, aid), v in relevancy_map_old.items() if sid == "1" and v.get("relevant") == 1)
print(f"With OLD parsing: {len(relevancy_map_old)} total entries")
print(f"  Relevant for subtask 1 (all queries): {relevant_count_old}")

# Test NEW parsing (CORRECT - loads only q01)
print("\n=== NEW PARSING (CORRECT) ===")
query_id = "q01"
relevancy_map_new = {}
for cache_key, cache_val in cache_data.items():
    parts = cache_key.split("_", 2)
    if len(parts) >= 3:
        cache_query_id = parts[0]
        sub_id = parts[1]
        api_id = "_".join(parts[2:])
        if cache_query_id == query_id:
            relevancy_map_new[(sub_id, api_id)] = cache_val

relevant_count_new = sum(1 for (sid, aid), v in relevancy_map_new.items() if sid == "1" and v.get("relevant") == 1)
print(f"With NEW parsing for {query_id}: {len(relevancy_map_new)} total entries")
print(f"  Relevant for subtask 1: {relevant_count_new}")

# Show the difference
print("\n=== COMPARISON ===")
test_key = ("1", "ip_address_tracker_free_auto_ip_lookup")
print(f"API: {test_key[1]}")
print(f"  In OLD map: {test_key in relevancy_map_old} -> {relevancy_map_old.get(test_key, {}).get('relevant')}")
print(f"  In NEW map: {test_key in relevancy_map_new} -> {relevancy_map_new.get(test_key, {}).get('relevant')}")

# Check what's in the cache for this API
for query_prefix in ["q01", "q14"]:
    cache_key = f"{query_prefix}_1_ip_address_tracker_free_auto_ip_lookup"
    if cache_key in cache_data:
        print(f"  Cache[{cache_key}] = {cache_data[cache_key].get('relevant')}")

