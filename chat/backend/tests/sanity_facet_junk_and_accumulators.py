"""Unit sanity checks for fixes D (post-$facet junk $project) and E
(accumulator hoisted to stage level). No server, no model."""
from chat.backend import llm_query as L

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -> ' + detail) if detail and not cond else ''}")
    if not cond:
        fails.append(name)


FACET = {"$facet": {"by_type": [{"$group": {"_id": "$alert_type", "count": {"$sum": 1}}}]}}

print("FIX D - post-$facet junk $project (all observed variants)")
for op, proj in [
    ("$concatArrays", {"by_type": {"$concatArrays": ["$by_type.count"]}}),
    ("$slice (negative arg - hard MongoDB error)", {"by_type": {"$slice": ["$by_type", 0, -1]}}),
    ("$concat", {"by_type": {"$concat": ["$by_type.count"]}}),
    ("$arrayElemAt", {"by_type": {"$arrayElemAt": ["$by_type.count", 0]}}),
]:
    out = L._strip_lossy_facet_projection([FACET, {"$project": proj}])
    check(f"{op} dropped", len(out) == 1 and "$facet" in out[0], str(out))

check("benign passthrough $project kept",
      len(L._strip_lossy_facet_projection([FACET, {"$project": {"by_type": 1}}])) == 2)
check("$slice without a $facet left alone",
      len(L._strip_lossy_facet_projection(
          [{"$match": {}}, {"$project": {"x": {"$slice": ["$a", 0, -1]}}}])) == 2)

print("\nFIX E - accumulator hoisted to stage level")
broken = [{"$match": {"alert_type": "HAND_TOUCH"}, "$expr": {"$avg": "$inspection_time"}}]
out = L._recover_hoisted_accumulator_stage(broken)
check("split into 2 stages", len(out) == 2, str(out))
check("$match preserved", out and "$match" in out[0] and out[0]["$match"]["alert_type"] == "HAND_TOUCH", str(out))
check("$group avg synthesised",
      len(out) == 2 and out[1] == {"$group": {"_id": None, "avg": {"$avg": "$inspection_time"}}}, str(out))
check("result passes _validate_pipeline", L._validate_pipeline(out)[0], str(L._validate_pipeline(out)))

bare = [{"$match": {"a": 1}, "$sum": "$x"}]
check("bare accumulator (no $expr) also recovered",
      len(L._recover_hoisted_accumulator_stage(bare)) == 2,
      str(L._recover_hoisted_accumulator_stage(bare)))

# negatives - must not disturb healthy pipelines
healthy = [{"$match": {"a": 1}}, {"$group": {"_id": None, "n": {"$sum": 1}}}, {"$limit": 5}]
check("healthy pipeline untouched", L._recover_hoisted_accumulator_stage(healthy) == healthy)

merged_valid = [{"$group": {"_id": "$t", "c": {"$sum": 1}}, "$sort": {"c": -1}}]
check("all-valid merged stages left for _split_merged_stages",
      L._recover_hoisted_accumulator_stage(merged_valid) == merged_valid,
      str(L._recover_hoisted_accumulator_stage(merged_valid)))

print("\nEnd-to-end: broken stage survives the whole repair chain")
final = L._finalize_pipeline(
    {"intent": "data_query",
     "pipeline": [{"$match": {"alert_type": "HAND_TOUCH"}, "$expr": {"$avg": "$inspection_time"}}],
     "explanation": ""},
    "What's the average inspection time for hand touch alerts in the last 7 days?")
check("no longer 'unsupported'", final.get("intent") == "data_query", str(final)[:200])
check("pipeline has a $group", any("$group" in s for s in (final.get("pipeline") or [])), str(final)[:250])

print("\n" + "=" * 60)
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {', '.join(fails)}")
raise SystemExit(1 if fails else 0)
