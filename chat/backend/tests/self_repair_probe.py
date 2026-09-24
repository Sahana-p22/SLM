"""Item 9 equivalent - the self-fixing/auto-repair path
(_regenerate_after_error: feed the actual Mongo error back to the model
and let it self-correct once) has plenty of INDIRECT exercise from every
other fix in this session (a deterministic repair usually catches things
before they ever reach Mongo), but had no DELIBERATE, direct test
confirming it fires on a genuine Mongo error and produces a working
corrected pipeline. This calls execute_pipeline() with a deliberately
broken RAW pipeline (bypassing every deterministic repair) to get a real
Mongo error, then _regenerate_after_error() with that exact error text -
exactly the shape production hits when a bad pipeline reaches Mongo.

Per instructions: report findings, do not change the repair logic itself
unless a real bug turns up.

Run from the repo root, with LD_LIBRARY_PATH set (needs the GGUF model
loaded for the retry's own LLM call) - takes a while, ~15 real LLM calls.
"""
import sys, time, traceback
from bench_config import REPO_ROOT
sys.path.insert(0, REPO_ROOT)
from chat.backend import llm_query as L

CASES = [
    # (label, question, deliberately-broken pipeline)
    ("nonexistent field reference", "how many alerts mention the widget field",
     [{"$match": {"widget_status": "broken"}}, {"$count": "total"}]),

    ("hallucinated operator name", "what's the average inspection time",
     [{"$match": {}}, {"$group": {"_id": None, "avg": {"$weekOfYear": "$inspection_time"}}}]),

    ("negative $slice third arg", "give me a quarterly report",
     [{"$match": {}},
      {"$facet": {"by_type": [{"$group": {"_id": "$alert_type", "count": {"$sum": 1}}}]}},
      {"$project": {"by_type": {"$slice": ["$by_type", 0, -1]}}}]),

    ("$group with no _id", "compare this week vs last week",
     [{"$match": {}}, {"$group": {"total": {"$sum": 1}}}]),

    ("$sum given a bare array literal", "total alerts for the last 7 days",
     [{"$match": {}}, {"$addFields": {"t": {"$sum": ["notafield"]}}}, {"$count": "total"}]),

    ("$elemMatch inside $project", "give me a weekly report",
     [{"$match": {}},
      {"$facet": {"by_type": [{"$group": {"_id": "$alert_type", "count": {"$sum": 1}}}]}},
      {"$project": {"by_type": {"$elemMatch": {"count": {"$gt": 0}}}}}]),

    ("unsupported operator entirely", "how many hand touch alerts today",
     [{"$match": {"alert_type": "HAND_TOUCH"}}, {"$count_distinct": "alert_type"}]),

    ("type mismatch: $hour on a string field", "alerts by hour today",
     [{"$match": {}}, {"$group": {"_id": {"$hour": "$alert_type"}, "count": {"$sum": 1}}}]),

    ("dangling $expr with missing operand", "hand touch alerts between 2pm and 4pm",
     [{"$match": {"$expr": {"$gte": [{"$hour": "$timestamp"}]}}}, {"$count": "total"}]),

    ("$replaceRoot onto a non-object", "give me a monthly report",
     [{"$match": {}},
      {"$facet": {"by_type": [{"$group": {"_id": "$alert_type", "count": {"$sum": 1}}}]}},
      {"$replaceRoot": {"newRoot": "$by_type"}}]),

    ("$divide by a non-numeric field", "average inspection time as a percentage",
     [{"$match": {}}, {"$group": {"_id": None, "pct": {"$avg": {"$divide": ["$inspection_time", "$zone"]}}}}]),

    ("$dateToString on a non-date field", "breakdown of alerts by month",
     [{"$match": {}}, {"$group": {"_id": {"$dateToString": {"format": "%Y-%m", "date": "$alert_type"}},
                                   "count": {"$sum": 1}}}]),

    ("malformed $bucket (missing boundaries)", "histogram of inspection times",
     [{"$match": {}}, {"$bucket": {"groupBy": "$inspection_time"}}]),

    ("$sortByCount on a nonexistent field", "most common zone",
     [{"$match": {}}, {"$sortByCount": "$nonexistent_field_xyz"}]),

    ("$unwind a field that is never an array", "break down cloth detections",
     [{"$match": {}}, {"$unwind": "$cloth_detected"}, {"$count": "total"}]),
]

results = []
for label, question, broken_pipeline in CASES:
    print(f"\n{'=' * 70}\n{label}: {question!r}")
    print(f"  broken pipeline: {broken_pipeline}")

    exec_result = L.execute_pipeline("data_query", broken_pipeline)
    if exec_result.get("intent") != "error":
        print(f"  SKIP - this pipeline did not actually fail against Mongo "
              f"(intent={exec_result.get('intent')!r}); not a valid test case, "
              f"picking a different broken shape needed.")
        results.append((label, "skip", "pipeline did not error"))
        continue

    error_msg = exec_result.get("error", "")
    print(f"  Mongo error: {error_msg[:200]}")

    t0 = time.time()
    try:
        retried = L._regenerate_after_error(question, broken_pipeline, error_msg, history=None)
    except Exception:
        print("  RETRY RAISED AN EXCEPTION:")
        traceback.print_exc()
        results.append((label, "fail", "exception during retry"))
        continue
    dt = time.time() - t0

    if retried is None:
        print(f"  RETRY RETURNED None ({dt:.1f}s) - self-correction did not fire/produce anything")
        results.append((label, "fail", "retry returned None"))
        continue
    if retried.get("intent") != "data_query":
        print(f"  RETRY intent={retried.get('intent')!r} ({dt:.1f}s) - gave up rather than fixed")
        results.append((label, "fail", f"intent={retried.get('intent')}"))
        continue

    fixed_pipeline = retried.get("pipeline")
    print(f"  retried pipeline ({dt:.1f}s): {fixed_pipeline}")
    retried_exec = L.execute_pipeline("data_query", fixed_pipeline)
    if retried_exec.get("intent") == "error":
        print(f"  STILL FAILS after retry: {retried_exec.get('error', '')[:200]}")
        results.append((label, "fail", f"still errors: {retried_exec.get('error', '')[:150]}"))
    else:
        print(f"  FIXED - retried pipeline ran cleanly, {len(retried_exec.get('rows', []))} row(s)")
        results.append((label, "pass", f"{len(retried_exec.get('rows', []))} rows"))

print("\n" + "=" * 70)
passed = sum(1 for _, s, _ in results if s == "pass")
failed = sum(1 for _, s, _ in results if s == "fail")
skipped = sum(1 for _, s, _ in results if s == "skip")
print(f"{passed} fired-and-fixed, {failed} failed, {skipped} skipped (of {len(results)} cases)")
for label, status, detail in results:
    print(f"  [{status.upper():4s}] {label}: {detail}")
