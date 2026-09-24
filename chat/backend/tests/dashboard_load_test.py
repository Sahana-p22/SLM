"""1.21 Dashboard Load Test: YCSB Workload C (100% read, uniform random
selection) against the app's own MongoDB-backed dashboard endpoints
(/stats/summary, /dashboard/fqc, /alerts/recent, /health) - separate
from the LLM-backed /chat path entirely. Ramped 1 -> 50 simultaneous
simulated users.
"""
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from bench_config import API_BASE as BASE
ENDPOINTS = [
    ("GET", "/stats/summary"),
    ("GET", "/dashboard/fqc"),
    ("GET", "/alerts/recent"),
    ("GET", "/health"),
]


def one_request(_):
    method, path = random.choice(ENDPOINTS)
    t0 = time.perf_counter()
    try:
        r = requests.get(f"{BASE}{path}", timeout=30)
        return time.perf_counter() - t0, r.status_code == 200
    except Exception:
        return time.perf_counter() - t0, False


results = {}
for n_users in (1, 5, 10, 25, 50):
    n_requests = n_users * 4
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_users) as ex:
        outcomes = list(ex.map(one_request, range(n_requests)))
    wall = time.perf_counter() - t0
    lat = sorted(dt for dt, ok in outcomes)
    errors = sum(1 for _, ok in outcomes if not ok)
    throughput = n_requests / wall
    p50 = lat[len(lat) // 2]
    p99 = lat[min(len(lat) - 1, int(len(lat) * 0.99))]
    results[n_users] = {"throughput_rps": throughput, "p50_s": p50, "p99_s": p99, "errors": errors}
    print(f"  {n_users:3d} users: {throughput:7.1f} req/s  p50={p50*1000:.1f}ms  p99={p99*1000:.1f}ms  errors={errors}")

# push higher to find the saturation point, mirroring "105-107 req/s before latency climbs"
print("\n  -- pushing further to find the saturation point --")
for n_users in (75, 100, 150):
    n_requests = n_users * 4
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_users) as ex:
        outcomes = list(ex.map(one_request, range(n_requests)))
    wall = time.perf_counter() - t0
    lat = sorted(dt for dt, ok in outcomes)
    errors = sum(1 for _, ok in outcomes if not ok)
    throughput = n_requests / wall
    p50 = lat[len(lat) // 2]
    p99 = lat[min(len(lat) - 1, int(len(lat) * 0.99))]
    results[n_users] = {"throughput_rps": throughput, "p50_s": p50, "p99_s": p99, "errors": errors}
    print(f"  {n_users:3d} users: {throughput:7.1f} req/s  p50={p50*1000:.1f}ms  p99={p99*1000:.1f}ms  errors={errors}")

json.dump(results, open("/tmp/dashboard_load_results.json", "w"), indent=2)
