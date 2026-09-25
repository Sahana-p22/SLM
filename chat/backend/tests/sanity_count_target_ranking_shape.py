"""Unit sanity checks for FIX (count-target date search misclassified as a
busiest-day ranking pipeline). Found live, with zero conversation history:
"which days did we get 328 alerts" built the exact $group -> $sort{count:-1}
-> $limit:1 shape that's correct for "which day had the most alerts", and
used it here regardless of the number 328 in the question - answered "416
alerts on 2025-03-21" (the real all-time busiest day) no matter what count
was actually asked for. Companion to `_is_date_dimension_ranking` /
`_DATE_COUNT_SEARCH_RE` (sanity_date_count_search_carryover.py), which stops
this question shape from inheriting a stale date range; this fix targets
the pipeline SHAPE itself once the range is right."""
from chat.backend import llm_query as L

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -> ' + detail) if detail and not cond else ''}")
    if not cond:
        fails.append(name)


def busiest_day_shape():
    return [
        {"$match": {"alert_type": {"$ne": "NORMAL_OPERATION"}}},
        {"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
                     "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 1},
    ]


print("FIX - a count-target question rewrites the busiest-day tail into a $match on the target count")
pipeline = L._fix_misclassified_count_target_ranking(busiest_day_shape(), "which days did we get 328 alerts")
check("no $sort stage left", not any(isinstance(s, dict) and "$sort" in s for s in pipeline), f"got {pipeline!r}")
check("no $limit stage left", not any(isinstance(s, dict) and "$limit" in s for s in pipeline), f"got {pipeline!r}")
match_stages = [s["$match"] for s in pipeline if isinstance(s, dict) and "$match" in s and "count" in s.get("$match", {})]
check("a $match on count:328 was added", match_stages == [{"count": 328}], f"got {pipeline!r}")

print("\nFIX - works for 'which day had 250 alerts' phrasing too")
pipeline2 = L._fix_misclassified_count_target_ranking(busiest_day_shape(), "which day had 250 alerts")
match_stages2 = [s["$match"] for s in pipeline2 if isinstance(s, dict) and "$match" in s and "count" in s.get("$match", {})]
check("a $match on count:250 was added", match_stages2 == [{"count": 250}], f"got {pipeline2!r}")

print("\nRegression - a genuine superlative question's pipeline is left untouched")
original = busiest_day_shape()
pipeline3 = L._fix_misclassified_count_target_ranking(busiest_day_shape(), "which day had the most alerts")
check("superlative pipeline unchanged", pipeline3 == original, f"got {pipeline3!r}")

print("\nFIX - a blended shape (model already added its own $match on the count, but kept the "
      "redundant $sort/$limit tail out of habit) has the leftover tail stripped, without duplicating "
      "the $match")
blended = [
    {"$match": {"alert_type": {"$ne": "NORMAL_OPERATION"}}},
    {"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
                 "count": {"$sum": 1}}},
    {"$match": {"count": 328}},
    {"$sort": {"count": -1}},
    {"$limit": 1},
]
pipeline5 = L._fix_misclassified_count_target_ranking(list(blended), "which days did we get 328 alerts")
check("no $sort stage left", not any(isinstance(s, dict) and "$sort" in s for s in pipeline5), f"got {pipeline5!r}")
check("no $limit stage left", not any(isinstance(s, dict) and "$limit" in s for s in pipeline5), f"got {pipeline5!r}")
match_stages5 = [s["$match"] for s in pipeline5 if isinstance(s, dict) and "$match" in s and "count" in s.get("$match", {})]
check("exactly one $match on count:328 (not duplicated)", match_stages5 == [{"count": 328}], f"got {pipeline5!r}")

print("\nRegression - a pipeline with no group/sort/limit ranking tail is left untouched")
plain = [
    {"$match": {"alert_type": {"$ne": "NORMAL_OPERATION"}}},
    {"$count": "total"},
]
pipeline4 = L._fix_misclassified_count_target_ranking(list(plain), "which days did we get 328 alerts")
check("non-ranking pipeline unchanged", pipeline4 == plain, f"got {pipeline4!r}")

print("\nEnd-to-end - _is_date_dimension_ranking still true for both wordings (companion fix unaffected)")
check("'which days did we get 328 alerts' still detected as date-dimension search",
      L._is_date_dimension_ranking("which days did we get 328 alerts"))
check("'which day had the most alerts' still detected as date-dimension ranking",
      L._is_date_dimension_ranking("which day had the most alerts"))

print("\n" + "=" * 60)
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {', '.join(fails)}")
raise SystemExit(1 if fails else 0)
