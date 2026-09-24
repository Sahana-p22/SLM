"""Scale-test: pure query-plan timing at 10-year (31,557,600-row) volume,
comparing no index -> two separate single-field indexes (the ACTUAL
pre-fix state of this deployment) -> the compound index (the fix already
applied to production earlier this session). Reuses the one generated
31.5M-row collection, changing only its indexes between timing runs
(avoids regenerating 31.5M rows three times).
"""
import json
import time
from pymongo import MongoClient, ASCENDING

from bench_config import MONGO_URI, SCALE_TEST_DB
client = MongoClient(MONGO_URI)
coll = client[SCALE_TEST_DB]["alerts_no_index"]

NOW_ISO = None  # filled from the collection's own max timestamp below
from datetime import timedelta

latest = coll.find().sort("timestamp", -1).limit(1)[0]["timestamp"]
today = latest.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

QUERIES = {
    "count today": lambda: coll.count_documents(
        {"alert_type": {"$ne": "NORMAL_OPERATION"}, "timestamp": {"$gte": today - timedelta(days=1), "$lt": today}}),
    "count type last 7d": lambda: coll.count_documents(
        {"alert_type": "HAND_TOUCH", "timestamp": {"$gte": today - timedelta(days=7), "$lt": today}}),
    "breakdown this month": lambda: list(coll.aggregate([
        {"$match": {"alert_type": {"$ne": "NORMAL_OPERATION"},
                     "timestamp": {"$gte": today.replace(day=1), "$lt": today}}},
        {"$group": {"_id": "$alert_type", "count": {"$sum": 1}}},
    ])),
    "superlative day most hand touch": lambda: list(coll.aggregate([
        {"$match": {"alert_type": "HAND_TOUCH", "timestamp": {"$gte": today - timedelta(days=30), "$lt": today}}},
        {"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}}, "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}, {"$limit": 1},
    ])),
    "avg inspection time fast": lambda: list(coll.aggregate([
        {"$match": {"alert_type": "FAST_INSPECTION"}},
        {"$group": {"_id": None, "avg": {"$avg": "$inspection_time"}}},
    ])),
    "hour filter (materialized field)": lambda: coll.count_documents(
        {"alert_type": {"$ne": "NORMAL_OPERATION"}, "hour": {"$gte": 14, "$lt": 16}}),
    "full count": lambda: coll.count_documents({}),
}


def time_query(fn, reps=3):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return min(times)  # best-of-N, avoids first-run cache-cold penalty skewing the comparison


def run_stage(label):
    print(f"\n=== {label} ===")
    results = {}
    for name, fn in QUERIES.items():
        ms = time_query(fn)
        results[name] = ms
        print(f"  {name:35s} {ms:10.2f} ms")
    return results


print(f"Collection: {coll.estimated_document_count():,} rows, "
      f"{client['slm_safety_scale'].command('collstats', 'alerts_no_index')['size']/1e9:.2f} GB")

# Stage 1: no index at all (baseline, as generated)
coll.drop_indexes()
no_index = run_stage("STAGE 1: no index")

# Stage 2: two SEPARATE single-field indexes - the actual pre-fix state
# this deployment was running in before the compound-index fix applied
# earlier this session.
coll.create_index("alert_type")
coll.create_index("timestamp")
coll.create_index("hour")
two_index = run_stage("STAGE 2: separate single-field indexes (pre-fix state)")

# Stage 3: the compound index fix, as actually applied to the real
# production collection earlier this session.
coll.drop_indexes()
coll.create_index([("alert_type", ASCENDING), ("timestamp", ASCENDING)])
coll.create_index([("alert_type", ASCENDING), ("hour", ASCENDING)])
compound = run_stage("STAGE 3: compound index (the applied fix)")

print("\n" + "=" * 70)
print(f"{'Query':35s} {'No index':>12s} {'2 indexes':>12s} {'Compound':>12s} {'Speedup':>10s}")
for name in QUERIES:
    a, b, c = no_index[name], two_index[name], compound[name]
    speedup = f"{a/c:.0f}x" if c > 0.01 else "—"
    print(f"{name:35s} {a:12.2f} {b:12.2f} {c:12.2f} {speedup:>10s}")

json.dump({"no_index": no_index, "two_index": two_index, "compound": compound},
          open("/tmp/scale_index_bench_results.json", "w"), indent=2)
