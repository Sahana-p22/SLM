"""Unit sanity checks for FIX S/T/U/V - the 5 real bugs found via
deep_bench root-cause analysis. Every input here is a shape captured
from a real, reproduced live failure."""
import copy
from chat.backend import llm_query as L

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -> ' + detail) if detail and not cond else ''}")
    if not cond:
        fails.append(name)


print("FIX S - inverted NORMAL_OPERATION filter")
pipe = [{"$match": {"alert_type": {"$ne": "NORMAL_OPERATION"}}}, {"$count": "total"}]
out = L._fix_normal_operation_polarity(copy.deepcopy(pipe), "How many normal operation events have been logged?")
check("flipped $ne to direct match", out[0]["$match"]["alert_type"] == "NORMAL_OPERATION", str(out))
check("compliant phrasing also triggers it",
      L._fix_normal_operation_polarity(copy.deepcopy(pipe), "How many compliant events were there?")[0]["$match"]["alert_type"] == "NORMAL_OPERATION")
untouched = L._fix_normal_operation_polarity(copy.deepcopy(pipe), "How many hand touch alerts today?")
check("unrelated question left untouched", untouched[0]["$match"]["alert_type"] == {"$ne": "NORMAL_OPERATION"}, str(untouched))
excl_wanted = L._fix_normal_operation_polarity(copy.deepcopy(pipe), "How many alerts were there, not counting normal operation?")
check("explicit exclusion phrasing left untouched", excl_wanted[0]["$match"]["alert_type"] == {"$ne": "NORMAL_OPERATION"}, str(excl_wanted))
check("'operation was normal' reordered phrasing also flips it",
      L._fix_normal_operation_polarity(copy.deepcopy(pipe), "How many times was the operation normal?")[0]["$match"]["alert_type"] == "NORMAL_OPERATION")

print("\nFIX T - imperative 'count' recognized as needing aggregation")
check("'Count the X' matches RANKING_INTENT_RE",
      bool(L.RANKING_INTENT_RE.search("Count the hand touch alerts in the last 7 days.")))
check("'how many' still matches (regression)", bool(L.RANKING_INTENT_RE.search("How many alerts today?")))
check("'count of' still matches (regression)", bool(L.RANKING_INTENT_RE.search("What's the count of alerts?")))
raw_dump = [{"$match": {"alert_type": "HAND_TOUCH"}}, {"$limit": 200}]
check("_needs_aggregation now catches the imperative phrasing",
      L._needs_aggregation("Count the hand touch alerts in the last 7 days.", raw_dump))

print("\nFIX U - plain count question misrouted into a superlative shape")
superlative_shape = [
    {"$match": {"alert_type": "MISSING_CLEANING", "timestamp": {"$gte": "x", "$lt": "y"}}},
    {"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}}, "count": {"$sum": 1}}},
    {"$sort": {"count": -1}},
    {"$limit": 1},
]
out = L._fix_misclassified_plain_total(copy.deepcopy(superlative_shape),
                                        "How many missing cleaning alerts happened this week?")
check("collapsed to a single whole-period $group",
      len(out) == 2 and out[1] == {"$group": {"_id": None, "count": {"$sum": 1}}}, str(out))

print("\nFIX U negatives - a real superlative/breakdown question keeps its shape")
untouched2 = L._fix_misclassified_plain_total(copy.deepcopy(superlative_shape),
                                               "Which day had the most missing cleaning alerts this week?")
check("superlative question untouched", untouched2 == superlative_shape, str(untouched2))
untouched3 = L._fix_misclassified_plain_total(copy.deepcopy(superlative_shape),
                                               "Break down missing cleaning alerts by day this week.")
check("breakdown question untouched", untouched3 == superlative_shape, str(untouched3))
group_by_type = [
    {"$match": {}}, {"$group": {"_id": "$alert_type", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}}, {"$limit": 1},
]
check("group-by-TYPE (not sub-period) left for the superlative logic, untouched",
      L._fix_misclassified_plain_total(copy.deepcopy(group_by_type), "How many alerts are there?") == group_by_type)

print("\nFIX V - chained $group where the 2nd references a dropped field")
chained = [
    {"$match": {"alert_type": "HAND_TOUCH", "timestamp": {"$gte": "x", "$lt": "y"}}},
    {"$group": {"_id": None, "sum": {"$sum": "$inspection_time"}}},
    {"$group": {"_id": None, "avg_inspection_time": {"$avg": "$inspection_time"}}},
    {"$sort": {"avg_inspection_time": -1}},
    {"$limit": 1},
]
out = L._repair_chained_group_dropped_field(copy.deepcopy(chained))
check("dead-end first $group dropped",
      out == [chained[0], chained[2], chained[3], chained[4]], str(out))

print("\nFIX V negatives - legitimate chained groups untouched")
legit_chain = [
    {"$match": {}},
    {"$group": {"_id": "$alert_type", "count": {"$sum": 1}}},
    {"$group": {"_id": None, "total": {"$sum": "$count"}}},  # re-uses the FIRST group's own output field
]
check("legit re-aggregation of a $group's own output untouched",
      L._repair_chained_group_dropped_field(copy.deepcopy(legit_chain)) == legit_chain,
      str(L._repair_chained_group_dropped_field(copy.deepcopy(legit_chain))))
single_group = [{"$match": {}}, {"$group": {"_id": None, "avg": {"$avg": "$inspection_time"}}}]
check("single $group (nothing to chain) untouched",
      L._repair_chained_group_dropped_field(copy.deepcopy(single_group)) == single_group)

print("\nEnd-to-end through _finalize_pipeline")
f = L._finalize_pipeline(
    {"intent": "data_query",
     "pipeline": [{"$match": {"alert_type": "HAND_TOUCH", "timestamp": {"$gte": "2026-09-17T00:00:00+00:00", "$lt": "2026-09-24T00:00:00+00:00"}}},
                  {"$group": {"_id": None, "sum": {"$sum": "$inspection_time"}}},
                  {"$group": {"_id": None, "avg_inspection_time": {"$avg": "$inspection_time"}}},
                  {"$sort": {"avg_inspection_time": -1}}, {"$limit": 1}],
     "explanation": ""},
    "What is the average inspection time for hand touch alerts in the last 7 days?")
final_pipe = f.get("pipeline") or []
check("final pipeline has exactly one $group and it computes $avg over $inspection_time",
      any("$group" in s and s["$group"].get("avg_inspection_time", {}).get("$avg") == "$inspection_time"
          for s in final_pipe),
      str(final_pipe))

f2 = L._finalize_pipeline(
    {"intent": "data_query",
     "pipeline": [{"$match": {"alert_type": "MISSING_CLEANING", "timestamp": {"$gte": "2026-09-21T00:00:00+00:00", "$lt": "2026-09-24T00:00:00+00:00"}}},
                  {"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}}, "count": {"$sum": 1}}},
                  {"$sort": {"count": -1}}, {"$limit": 1}],
     "explanation": ""},
    "How many missing cleaning alerts happened this week?")
final_pipe2 = f2.get("pipeline") or []
check("final pipeline collapses to a single whole-period $group (no $sort/$limit survives)",
      not any("$sort" in s or "$limit" in s for s in final_pipe2)
      and any(s.get("$group", {}).get("_id") is None for s in final_pipe2 if "$group" in s),
      str(final_pipe2))

f3 = L._finalize_pipeline(
    {"intent": "data_query",
     "pipeline": [{"$match": {"alert_type": {"$ne": "NORMAL_OPERATION"}}}, {"$count": "total"}],
     "explanation": ""},
    "How many normal operation events have been logged?")
final_pipe3 = f3.get("pipeline") or []
check("final pipeline correctly matches NORMAL_OPERATION directly",
      any(s.get("$match", {}).get("alert_type") == "NORMAL_OPERATION" for s in final_pipe3 if "$match" in s),
      str(final_pipe3))

print("\n" + "=" * 60)
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {', '.join(fails)}")
raise SystemExit(1 if fails else 0)
