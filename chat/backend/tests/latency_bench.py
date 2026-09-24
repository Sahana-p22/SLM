"""Latency/throughput benchmark: real wall-clock timing per question,
broken down by the stage timings the backend already reports (query
generation, DB execution, answer generation), across a representative
mix of question complexity (fast-path, single LLM call, report/facet,
follow-up). Run against the live :8002 backend.
"""
import json
import statistics as st
import time
import requests

from bench_config import CHAT_URL

API = CHAT_URL

QUESTIONS = [
    # fast-path (no LLM call at all)
    "how many alerts today",
    "how many alerts yesterday",
    "how many alerts in the last 7 days",
    "how many alerts between 2pm and 4pm",
    "how many distinct alert types are there",
    # single LLM call, simple
    "how many hand touch alerts happened this week",
    "what's the average inspection time",
    "which day had the most missing cleaning alerts",
    "how many alerts happened at FQC Station 1 today",
    "how many inspections took over 30 seconds",
    # breakdown / grouping
    "break down alerts by type for the last 7 days",
    "break down alerts by type this month",
    "which hour is busiest this week",
    "compare this week vs last week",
    "compare average inspection time by type",
    # reports (facet, heaviest shape)
    "give me a quarterly report",
    "give me a weekly report",
    "give me a report for June",
    "day wise report for this week",
]

results = []
for q in QUESTIONS:
    t0 = time.perf_counter()
    r = requests.post(API, json={"question": q, "history": []}, timeout=180)
    dt = time.perf_counter() - t0
    d = r.json()
    stages = {s["name"]: s["duration_ms"] for s in d.get("stages", [])}
    results.append({
        "question": q,
        "total_ms": round(dt * 1000, 1),
        "query_gen_ms": stages.get("Query generation", 0),
        "db_exec_ms": stages.get("Database execution", 0),
        "answer_gen_ms": stages.get("Answer generation", 0),
        "self_correction_ms": sum(v for k, v in stages.items() if "Self-correction" in k),
        "fast_path": "Question matched a simple" in json.dumps(d.get("stages", [])),
        "row_count": d.get("row_count"),
    })
    print(f"{dt*1000:8.1f}ms  {'[fast-path]' if results[-1]['fast_path'] else '           '}  {q}")

print("\n" + "=" * 80)
totals = [r["total_ms"] for r in results]
db_times = [r["db_exec_ms"] for r in results]
fast_path_times = [r["total_ms"] for r in results if r["fast_path"]]
llm_times = [r["total_ms"] for r in results if not r["fast_path"]]

print(f"Total questions: {len(results)}")
print(f"Overall latency  - mean: {st.mean(totals):.0f}ms  median: {st.median(totals):.0f}ms  "
      f"min: {min(totals):.0f}ms  max: {max(totals):.0f}ms")
if fast_path_times:
    print(f"Fast-path only   - mean: {st.mean(fast_path_times):.0f}ms  "
          f"median: {st.median(fast_path_times):.0f}ms  n={len(fast_path_times)}")
if llm_times:
    print(f"LLM-call only    - mean: {st.mean(llm_times):.0f}ms  "
          f"median: {st.median(llm_times):.0f}ms  n={len(llm_times)}")
print(f"DB execution     - mean: {st.mean(db_times):.1f}ms  median: {st.median(db_times):.1f}ms  "
      f"max: {max(db_times):.1f}ms")

self_corrections = sum(1 for r in results if r["self_correction_ms"] > 0)
print(f"Self-corrections triggered: {self_corrections}/{len(results)}")

json.dump(results, open("/tmp/latency_bench_results.json", "w"), indent=2)
print("\nRaw results written to /tmp/latency_bench_results.json")
