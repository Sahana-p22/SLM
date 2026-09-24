"""1.11 Long-Run Stability Test: continuous real chat traffic against
:8002, sampling the backend process's RSS memory every 10s, comparing
average latency in the first 20% of requests vs the last 20% (drift).

Shortened from the source report's 15 minutes to 6, at the user's
explicit request to finish the overall benchmark run faster - noted
honestly in the report rather than silently kept at 15 (a shorter window
gives less confidence ruling out a slow leak, same caveat the source
report already carried at 15 minutes, just stronger)."""
import json
import subprocess
import time

import requests

from bench_config import CHAT_URL

API = CHAT_URL
DURATION_S = 6 * 60
SAMPLE_EVERY_S = 10

QUESTIONS = [
    "how many alerts today", "how many hand touch alerts this week",
    "what's the average inspection time", "break down alerts by type this month",
    "how many missing cleaning alerts happened yesterday",
    "which day had the most fast inspection alerts", "compare this week vs last week",
]


def backend_rss_mb():
    out = subprocess.run(
        ["bash", "-c", "ps aux | grep 'uvicorn.*8002' | grep -v grep | awk '{print $6}'"],
        capture_output=True, text=True,
    ).stdout.strip()
    try:
        return int(out.split()[0]) / 1024
    except (ValueError, IndexError):
        return None


t_start = time.time()
last_sample = 0
mem_samples = []
latencies = []
errors = 0
i = 0

print(f"Running {DURATION_S/60:.0f} minutes of continuous load...")
while time.time() - t_start < DURATION_S:
    q = QUESTIONS[i % len(QUESTIONS)]
    i += 1
    t0 = time.perf_counter()
    try:
        r = requests.post(API, json={"question": q, "history": []}, timeout=60)
        latencies.append(time.perf_counter() - t0)
        if r.status_code != 200:
            errors += 1
    except Exception:
        errors += 1
        latencies.append(time.perf_counter() - t0)

    now = time.time() - t_start
    if now - last_sample >= SAMPLE_EVERY_S:
        rss = backend_rss_mb()
        mem_samples.append((round(now, 1), rss))
        last_sample = now
        print(f"  t={now:.0f}s  requests={i}  errors={errors}  rss={rss}MB")

n = len(latencies)
first_20 = latencies[: max(1, n // 5)]
last_20 = latencies[-max(1, n // 5):]
drift = ((sum(last_20) / len(last_20)) - (sum(first_20) / len(first_20))) / (sum(first_20) / len(first_20)) * 100

mem_growth = (mem_samples[-1][1] - mem_samples[0][1]) if len(mem_samples) >= 2 and mem_samples[0][1] and mem_samples[-1][1] else None

print(f"\nTotal requests: {n}, errors: {errors}")
print(f"Memory: start={mem_samples[0][1]}MB end={mem_samples[-1][1]}MB growth={mem_growth}MB")
print(f"Latency drift (first 20% vs last 20%): {drift:.2f}%")

json.dump({
    "requests": n, "errors": errors, "duration_s": DURATION_S,
    "mem_samples": mem_samples, "mem_growth_mb": mem_growth,
    "latency_drift_pct": drift,
    "first_20_mean_s": sum(first_20) / len(first_20),
    "last_20_mean_s": sum(last_20) / len(last_20),
}, open("/tmp/soak_test_results.json", "w"), indent=2)
