"""Unit sanity checks for FIX G - post-$facet $group re-collecting sections."""
from chat.backend import llm_query as L

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -> ' + detail) if detail and not cond else ''}")
    if not cond:
        fails.append(name)


strip = L._strip_junk_post_facet_stages

FACET = {"$facet": {
    "total_alerts": [{"$count": "total"}],
    "by_type": [{"$group": {"_id": "$alert_type", "count": {"$sum": 1}}}],
    "by_month": [{"$group": {"_id": "$m", "count": {"$sum": 1}}}],
    "by_day": [{"$group": {"_id": "$d", "count": {"$sum": 1}}}],
    "avg_inspection_time": [{"$group": {"_id": None, "avg": {"$avg": "$t"}}}],
}}

# Verbatim from the failing quarterly_report run.
JUNK_GROUP = {"$group": {
    "_id": None,
    "total_alerts": {"$sum": "$total_alerts"},
    "by_type": {"$push": {"count": "$by_type.count"}},
    "by_month": {"$push": {"count": "$by_month.count"}},
    "by_day": {"$push": {"count": "$by_day.count"}},
    "avg_inspection_time": {"$sum": "$avg_inspection_time"},
}}

print("The observed quarterly_report pipeline")
out = strip([{"$match": {"a": 1}}, FACET, {"$limit": 1}, JUNK_GROUP])
check("junk $group dropped", len(out) == 3 and "$group" not in out[-1], str(out)[:200])
check("$facet and $limit preserved", "$facet" in out[1] and "$limit" in out[2])

print("\nNegative - a $group doing real work after $unwind is kept")
# Verbatim shape from busiest_week_this_quarter, which passes today.
legit = [
    {"$facet": {"this_week": [{"$count": "n"}], "last_week": [{"$count": "n"}]}},
    {"$unwind": "$this_week"},
    {"$unwind": "$last_week"},
    {"$addFields": {"week": {"$add": ["$this_week.count", "$last_week.count"]}}},
    {"$group": {"_id": "$w", "total_alerts": {"$sum": "$week"}}},
    {"$limit": 1},
]
check("post-$unwind $group kept", strip(legit) == legit, str(strip(legit))[:250])

# Superseded: the section-name guard this used to assert was widened once
# the smoke run turned up two more post-$facet $group/$addFields shapes
# that named no section and were fatal anyway. With no $unwind there is a
# single document in play, so ANY $group after the $facet is degenerate.
print("\nWidened rule - any post-$facet $group without an $unwind is dropped")
unrelated = [FACET, {"$group": {"_id": "$something_else", "n": {"$sum": 1}}}]
check("$group naming no facet section dropped too",
      len(strip(unrelated)) == 1, str(strip(unrelated))[:200])
check("post-$facet $addFields dropped",
      len(strip([FACET, {"$addFields": {"t": {"$sum": ["this_week.count"]}}}])) == 1)

print("\nNegative - no $facet at all")
nofacet = [{"$match": {}}, {"$group": {"_id": None, "by_type": {"$push": "$x"}}}]
check("pipeline without a $facet untouched", strip(nofacet) == nofacet)

print("\nEarlier fixes still hold")
check("junk $project still dropped",
      len(strip([FACET, {"$project": {"by_type": {"$concatArrays": ["$by_type.count"]}}}])) == 1)
check("no-op $sort still dropped", len(strip([FACET, {"$sort": {"total_alerts": -1}}])) == 1)
check("benign $project still kept", len(strip([FACET, {"$project": {"by_type": 1}}])) == 2)

print("\n" + "=" * 60)
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {', '.join(fails)}")
raise SystemExit(1 if fails else 0)
