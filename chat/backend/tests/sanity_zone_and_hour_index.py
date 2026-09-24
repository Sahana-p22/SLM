"""Unit sanity checks for FIX P - indexed hour-range filter (item 5's
Mongo equivalent) and FIX N - zone-value normalization (item 3's Mongo
equivalent)."""
from chat.backend import llm_query as L

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -> ' + detail) if detail and not cond else ''}")
    if not cond:
        fails.append(name)


print("FIX P - $expr/$hour filter rewritten to the indexed field")
expr_shape = [
    {"$match": {"alert_type": {"$ne": "NORMAL_OPERATION"}}},
    {"$match": {"$expr": {"$and": [
        {"$gte": [{"$hour": "$timestamp"}, 14]},
        {"$lt": [{"$hour": "$timestamp"}, 16]},
    ]}}},
    {"$count": "total"},
]
out = L._use_indexed_hour_filter([dict(s) for s in expr_shape])
check("rewritten to a plain 'hour' field match",
      out[1] == {"$match": {"hour": {"$gte": 14, "$lt": 16}}}, str(out))
check("other stages untouched", out[0] == expr_shape[0] and out[2] == expr_shape[2])

print("\nFIX P negatives - must not touch unrelated shapes")
group_shape = [{"$group": {"_id": {"hour": {"$hour": "$timestamp"}}, "count": {"$sum": 1}}}]
check("$group $hour (breakdown, not a filter) left untouched",
      L._use_indexed_hour_filter([dict(s) for s in group_shape]) == group_shape, str(group_shape))

other_expr = [{"$match": {"$expr": {"$eq": ["$alert_type", "HAND_TOUCH"]}}}]
check("unrelated \\$expr left untouched",
      L._use_indexed_hour_filter([dict(s) for s in other_expr]) == other_expr)

three_clause = [{"$match": {"$expr": {"$and": [
    {"$gte": [{"$hour": "$timestamp"}, 14]},
    {"$lt": [{"$hour": "$timestamp"}, 16]},
    {"$eq": ["$alert_type", "HAND_TOUCH"]},
]}}}]
check("a 3-clause \\$and (not the plain 2-clause hour shape) left untouched",
      L._use_indexed_hour_filter([dict(s) for s in three_clause]) == three_clause,
      str(L._use_indexed_hour_filter([dict(s) for s in three_clause])))

print("\nFIX P - end to end through _finalize_pipeline")
f = L._finalize_pipeline(
    {"intent": "data_query",
     "pipeline": [
         {"$match": {"alert_type": {"$ne": "NORMAL_OPERATION"}}},
         {"$match": {"$expr": {"$and": [
             {"$gte": [{"$hour": "$timestamp"}, 9]},
             {"$lt": [{"$hour": "$timestamp"}, 11]},
         ]}}},
         {"$count": "total"},
     ], "explanation": ""},
    "how many alerts between 9am and 11am today")
check("final pipeline uses the indexed field, not \\$expr",
      "$expr" not in str(f.get("pipeline")), str(f.get("pipeline")))
check("hour bounds preserved correctly",
      any(isinstance(s, dict) and s.get("$match", {}).get("hour") == {"$gte": 9, "$lt": 11}
          for s in (f.get("pipeline") or [])),
      str(f.get("pipeline")))

print("\nFIX N - zone value normalization")
n = L._normalize_zone_value
check("exact match", n("FQC Station 1") == "FQC Station 1")
check("lowercase", n("fqc station 1") == "FQC Station 1")
check("underscores", n("fqc_station_1") == "FQC Station 1")
check("hyphens", n("FQC-Station-1") == "FQC Station 1")
check("extra whitespace", n("FQC   Station  1") == "FQC Station 1")
check("leading/trailing whitespace", n("  fqc station 1  ") == "FQC Station 1")
check("unrelated string -> None (not silently guessed)", n("Station 2") is None)
check("non-string -> None", n(123) is None)

repaired = L._repair_zone_filter({"$match": {"zone": "fqc_station_1", "alert_type": "HAND_TOUCH"}})
check("repair normalizes a zone value inside a real pipeline stage",
      repaired == {"$match": {"zone": "FQC Station 1", "alert_type": "HAND_TOUCH"}}, str(repaired))

repaired_in = L._repair_zone_filter({"$match": {"zone": {"$in": ["fqc station 1", "FQC-Station-1"]}}})
check("repair normalizes every value inside a \\$in list",
      repaired_in == {"$match": {"zone": {"$in": ["FQC Station 1", "FQC Station 1"]}}}, str(repaired_in))

unrelated_zone = L._repair_zone_filter({"$match": {"zone": "Loading Dock"}})
check("an unrecognized zone name is left as-is, not guessed at",
      unrelated_zone == {"$match": {"zone": "Loading Dock"}}, str(unrelated_zone))

print("\n" + "=" * 60)
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {', '.join(fails)}")
raise SystemExit(1 if fails else 0)
