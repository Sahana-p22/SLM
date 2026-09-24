"""1.15 Database Size vs Speed Test: the same 8 real chat questions run
against the actual live backend, restarted 3 times pointed at 3 different
database sizes: current (real production data, whatever size it is
today), medium (3M synthetic rows), large (31.5M synthetic rows, the
scale-test collection gen_scale_collection.py builds). Each restart
points db.py's DB_NAME/collection at a different pre-populated database -
the real production collection is used as-is, never modified, and the
backend is always restarted back onto it before this script exits (even
on failure - see bench_config.restart_backend's own guarantee).

Requires gen_scale_collection.py to have been run first (or FQC_SCALE_TEST_DB
already populated) for the medium/large rows to exist; falls back to
just the "current" size otherwise.
"""
import json
import sys
import time

import requests
from pymongo import MongoClient

from bench_config import REPO_ROOT, MONGO_URI, SCALE_TEST_DB, CHAT_URL, restart_backend
sys.path.insert(0, REPO_ROOT)

QUESTIONS = [
    "how many alerts today", "how many hand touch alerts this week",
    "what's the average inspection time", "break down alerts by type this month",
    "which day had the most fast inspection alerts", "how many alerts happened between 2pm and 4pm",
    "how many missing cleaning alerts yesterday", "how many alerts in the last 30 days",
]

SIZES = [
    ("current", "slm_safety", "alerts"),
    ("medium_3M", SCALE_TEST_DB, "alerts_3m"),
    ("large_31_5M", SCALE_TEST_DB, "alerts_no_index"),
]


def ensure_3m_collection():
    """Builds a 3M-row copy by sampling from the already-generated
    scale-test collection - far cheaper than a fresh generation pass."""
    client = MongoClient(MONGO_URI)
    src = client[SCALE_TEST_DB]["alerts_no_index"]
    if src.estimated_document_count() == 0:
        print(f"  WARNING: {SCALE_TEST_DB}.alerts_no_index is empty - run "
              "gen_scale_collection.py first. Skipping medium/large sizes.")
        return False
    dst = client[SCALE_TEST_DB]["alerts_3m"]
    if dst.estimated_document_count() >= 3_000_000:
        print("  3M collection already exists")
        return True
    print("  building 3M-row collection via $sample...")
    dst.drop()
    client[SCALE_TEST_DB].command("aggregate", "alerts_no_index", pipeline=[
        {"$sample": {"size": 3_000_000}},
        {"$out": "alerts_3m"},
    ], allowDiskUse=True, cursor={})
    dst.create_index([("alert_type", 1), ("timestamp", 1)])
    dst.create_index([("alert_type", 1), ("hour", 1)])
    print(f"  built: {dst.estimated_document_count():,} rows")
    return True


have_scale_data = ensure_3m_collection()
sizes_to_run = SIZES if have_scale_data else SIZES[:1]

results = {}
for label, db_name, coll_name in sizes_to_run:
    print(f"\n=== {label} ({db_name}.{coll_name}) ===")
    ok = restart_backend(db_name=db_name, collection_name=coll_name)
    if not ok:
        print("  FAILED to restart")
        results[label] = {"error": "restart failed"}
        continue
    latencies = []
    for q in QUESTIONS * 2:  # 2 repeats per question
        t0 = time.perf_counter()
        try:
            requests.post(CHAT_URL, json={"question": q, "history": []}, timeout=120)
            latencies.append(time.perf_counter() - t0)
        except Exception:
            pass
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p99 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.99))]
    results[label] = {"p50_s": p50, "p99_s": p99, "n": len(latencies)}
    print(f"  p50={p50:.2f}s p99={p99:.2f}s (n={len(latencies)})")

    # the specific fast-path regression the source report flagged: an
    # unindexed/badly-indexed collection can turn a normally-instant
    # fast-path answer into a multi-second one as the row count grows.
    t0 = time.perf_counter()
    requests.post(CHAT_URL, json={"question": "which day had the most hand touch alerts", "history": []}, timeout=120)
    results[label]["fastpath_superlative_s"] = time.perf_counter() - t0
    print(f"  fast-path superlative question: {results[label]['fastpath_superlative_s']:.3f}s")

print("\n" + "=" * 70)
for label, r in results.items():
    print(f"  {label:20s} {r}")

json.dump(results, open("/tmp/db_size_vs_speed_results.json", "w"), indent=2)

# restart pointed back at real production data before leaving, regardless
# of how the loop above went
restart_backend(db_name="slm_safety", collection_name="alerts")
print("\nBackend restored to production data.")
