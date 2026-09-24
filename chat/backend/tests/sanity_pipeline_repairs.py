"""Unit sanity checks for fixes H (missing $group _id), I (unary
accumulator given an array) and J (date operator on a non-date field),
plus the relaxed G. Every input is a shape captured from a live run."""
from chat.backend import llm_query as L

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -> ' + detail) if detail and not cond else ''}")
    if not cond:
        fails.append(name)


print("FIX H - $group with no _id")
broken = [{"$group": {"total_this_week": {"$sum": "$this_week.count"},
                      "total_last_week": {"$sum": "$last_week.count"}}}]
out = L._repair_group_missing_id(broken)
check("_id: None inserted", out[0]["$group"]["_id"] is None, str(out))
check("accumulators preserved",
      out[0]["$group"]["total_this_week"] == {"$sum": "$this_week.count"}, str(out))
healthy = [{"$group": {"_id": "$alert_type", "count": {"$sum": 1}}}]
check("healthy $group untouched", L._repair_group_missing_id(healthy) == healthy)
check("existing _id: None untouched",
      L._repair_group_missing_id([{"$group": {"_id": None, "n": {"$sum": 1}}}])
      == [{"$group": {"_id": None, "n": {"$sum": 1}}}])

print("\nFIX I - unary accumulator handed a single-element array")
u = L._unwrap_unary_accumulator_args
check("$sum ['x'] unwrapped",
      u({"$addFields": {"t": {"$sum": ["this_week.count"]}}})
      == {"$addFields": {"t": {"$sum": "this_week.count"}}})
check("nested inside a $group unwrapped",
      u([{"$group": {"_id": None, "s": {"$sum": ["$a"]}}}])
      == [{"$group": {"_id": None, "s": {"$sum": "$a"}}}])
check("$avg / $min / $max also unwrapped",
      u({"a": {"$avg": ["$x"]}, "b": {"$min": ["$y"]}, "c": {"$max": ["$z"]}})
      == {"a": {"$avg": "$x"}, "b": {"$min": "$y"}, "c": {"$max": "$z"}})
check("genuine multi-arg $sum left alone",
      u({"$sum": ["$a", "$b"]}) == {"$sum": ["$a", "$b"]})
check("$sum: 1 left alone", u({"$sum": 1}) == {"$sum": 1})
check("$sum: '$field' left alone", u({"$sum": "$f"}) == {"$sum": "$f"})
check("empty array left alone", u({"$sum": []}) == {"$sum": []})

print("\nFIX J - date operator aimed at a non-date field")
r = L._retarget_date_operators
check("the observed $dayOfWeek: '$dayOfWeek'",
      r({"$group": {"_id": {"dayOfWeek": {"$dayOfWeek": "$dayOfWeek"}}, "count": {"$sum": 1}}})
      == {"$group": {"_id": {"dayOfWeek": {"$dayOfWeek": "$timestamp"}}, "count": {"$sum": 1}}})
check("$hour: '$hour' retargeted", r({"$hour": "$hour"}) == {"$hour": "$timestamp"})
check("$dateToString date retargeted",
      r({"$dateToString": {"format": "%Y-%m", "date": "$month"}})
      == {"$dateToString": {"format": "%Y-%m", "date": "$timestamp"}})
check("already-correct $dayOfWeek untouched",
      r({"$dayOfWeek": "$timestamp"}) == {"$dayOfWeek": "$timestamp"})
check("already-correct $dateToString untouched",
      r({"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}})
      == {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}})
check("non-date operators untouched",
      r({"$sum": "$inspection_time", "$toUpper": "$zone"})
      == {"$sum": "$inspection_time", "$toUpper": "$zone"})
check("a field literally named timestamp in $match untouched",
      r({"$match": {"timestamp": {"$gte": "2026-01-01"}}})
      == {"$match": {"timestamp": {"$gte": "2026-01-01"}}})

print("\nFIX G relaxed - degenerate post-$facet stages")
strip = L._strip_junk_post_facet_stages
FACET = {"$facet": {"this_week": [{"$count": "n"}], "last_week": [{"$count": "n"}]}}
observed = [FACET,
            {"$addFields": {"total_this_week": {"$sum": ["this_week.count"]}}},
            {"$project": {"last_week": 0}},
            {"$group": {"_id": None, "sum": {"$sum": ["total_this_week"]}}},
            {"$limit": 1}]
out = strip(observed)
check("$addFields and $group both dropped",
      not any(k in s for s in out for k in ("$addFields", "$group")), str(out)[:250])
check("$facet, $project and $limit kept", len(out) == 3, str(out)[:250])

legit = [FACET, {"$unwind": "$this_week"}, {"$unwind": "$last_week"},
         {"$addFields": {"week": {"$add": ["$this_week.count", "$last_week.count"]}}},
         {"$group": {"_id": "$w", "total": {"$sum": "$week"}}}, {"$limit": 1}]
check("post-$unwind $addFields/$group still kept", strip(legit) == legit, str(strip(legit))[:250])

print("\nEnd-to-end through _finalize_pipeline")
f = L._finalize_pipeline(
    {"intent": "data_query",
     "pipeline": [{"$match": {"alert_type": "HAND_TOUCH"}},
                  {"$group": {"_id": {"dayOfWeek": {"$dayOfWeek": "$dayOfWeek"}},
                              "count": {"$sum": 1}}},
                  {"$sort": {"count": -1}}, {"$limit": 1}],
     "explanation": ""},
    "which day of the week has the most hand touch alerts?")
check("date operator repaired end to end",
      "$timestamp" in str(f.get("pipeline")), str(f.get("pipeline"))[:250])

print("\n" + "=" * 60)
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {', '.join(fails)}")
raise SystemExit(1 if fails else 0)
