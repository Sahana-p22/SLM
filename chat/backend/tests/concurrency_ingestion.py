"""1.9 Multiple-People-At-Once Test + 1.10 Live Data Writing Test.

Ramps 1->2->4->8 simultaneous real chat requests against :8002, measuring
throughput and p50/p99 latency at each level. While the 8-user level runs,
a background writer inserts one row every 10s into an ISOLATED test
collection (never the real production slm_safety.alerts), watching for
write errors/contention - MongoDB has no single-writer-lock the way
SQLite does, but the same read/write-under-load scenario is reproduced
for comparability.
"""
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests
from pymongo import MongoClient

from bench_config import REPO_ROOT, CHAT_URL
sys.path.insert(0, REPO_ROOT)
from chat.backend.seed_data import ZONES, OBJECT_POOLS

API = CHAT_URL
ingest_client = MongoClient("mongodb://localhost:27017")
ingest_coll = ingest_client["slm_safety_bench"]["ingestion_test"]

QUESTIONS = [
    "how many alerts today", "how many hand touch alerts this week",
    "what's the average inspection time", "break down alerts by type this month",
    "how many missing cleaning alerts happened yesterday",
]

ingest_errors = []
ingest_count = [0]
stop_ingest = threading.Event()


def ingest_loop():
    while not stop_ingest.is_set():
        try:
            ingest_coll.insert_one({
                "alert_type": "HAND_TOUCH", "inspection_time": 5.0,
                "objects_present": OBJECT_POOLS["HAND_TOUCH"][0],
                "cloth_detected": False, "zone": ZONES[0],
                "timestamp": datetime.now(timezone.utc),
            })
            ingest_count[0] += 1
        except Exception as e:
            ingest_errors.append(str(e))
        stop_ingest.wait(10)


def one_request(i):
    q = QUESTIONS[i % len(QUESTIONS)]
    t0 = time.perf_counter()
    try:
        r = requests.post(API, json={"question": q, "history": []}, timeout=120)
        dt = time.perf_counter() - t0
        return dt, r.status_code == 200
    except Exception:
        return time.perf_counter() - t0, False


results = {}
for n_users in (1, 2, 4, 8):
    n_requests = n_users * 3  # 3 rounds per level, mirrors ramping real traffic
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_users) as ex:
        outcomes = list(ex.map(one_request, range(n_requests)))
    wall = time.perf_counter() - t_start
    lat = sorted(dt for dt, ok in outcomes)
    errors = sum(1 for _, ok in outcomes if not ok)
    throughput = n_requests / wall
    p50 = lat[len(lat) // 2]
    p99 = lat[min(len(lat) - 1, int(len(lat) * 0.99))]
    results[n_users] = {"throughput_rps": throughput, "p50_s": p50, "p99_s": p99, "errors": errors}
    print(f"  {n_users} users: {throughput:.3f} req/s, p50={p50:.2f}s, p99={p99:.2f}s, errors={errors}")

    if n_users == 8:
        print("  [starting background ingestion during 8-user load]")
        ingest_thread = threading.Thread(target=ingest_loop, daemon=True)
        ingest_thread.start()
        # run one more round of 8-user load while ingestion is active
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(one_request, range(8)))
        time.sleep(15)
        stop_ingest.set()
        ingest_thread.join(timeout=5)

print(f"\nIngestion during load: {ingest_count[0]} rows inserted, {len(ingest_errors)} errors")
if ingest_errors:
    print("  sample error:", ingest_errors[0])

json.dump({"concurrency": results, "ingestion": {"inserted": ingest_count[0], "errors": ingest_errors}},
          open("/tmp/concurrency_ingestion_results.json", "w"), indent=2)
