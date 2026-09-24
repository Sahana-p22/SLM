"""Unit sanity checks for the generalised post-$facet junk-stage guard.
No server, no model. Every input here is a stage shape observed in a real run."""
from chat.backend import llm_query as L

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -> ' + detail) if detail and not cond else ''}")
    if not cond:
        fails.append(name)


FACET = {"$facet": {
    "by_type": [{"$group": {"_id": "$alert_type", "count": {"$sum": 1}}}],
    "by_month": [{"$group": {"_id": "$month", "count": {"$sum": 1}}}],
}}
strip = L._strip_junk_post_facet_stages

print("Junk $project variants (all observed live)")
for label, proj in [
    ("$concatArrays", {"by_type": {"$concatArrays": ["$by_type.count"]}}),
    ("$slice negative arg", {"by_type": {"$slice": ["$by_type", 0, -1]}}),
    ("$concat", {"by_type": {"$concat": ["$by_type.count"]}}),
    ("$arrayElemAt", {"by_type": {"$arrayElemAt": ["$by_type", 0]}}),
    ("$elemMatch", {"by_type": {"$elemMatch": {"count": {"$gt": 0}}}}),
    ("$ifNull", {"by_type": {"$ifNull": ["$by_type.count", 0]}}),
    ("$map", {"by_type": {"$map": {"input": "$by_type", "in": "$$this.count"}}}),
    ("$reduce", {"t": {"$reduce": {"input": "$by_type", "initialValue": 0, "in": 1}}}),
    ("$filter", {"by_type": {"$filter": {"input": "$by_type", "cond": True}}}),
]:
    out = strip([FACET, {"$project": proj}])
    check(f"{label} dropped", len(out) == 1 and "$facet" in out[0], str(out))

print("\n$replaceRoot / $replaceWith onto an array section")
check("$replaceRoot newRoot: $by_type dropped",
      len(strip([FACET, {"$replaceRoot": {"newRoot": "$by_type"}}])) == 1)
check("$replaceWith $by_month dropped",
      len(strip([FACET, {"$replaceWith": "$by_month"}])) == 1)

print("\nTrailing $sort over parallel arrays")
bad_sort = {"$sort": {"total_alerts": -1, "by_type.count": -1, "by_month.count": -1}}
check("parallel-array $sort dropped", len(strip([FACET, bad_sort])) == 1)
check("full observed chain reduces to the bare $facet",
      strip([{"$match": {"a": 1}}, FACET,
             {"$project": {"by_type": {"$ifNull": ["$by_type.count", 0]}}},
             bad_sort, {"$limit": 1}]) ==
      [{"$match": {"a": 1}}, FACET, {"$limit": 1}],
      str(strip([{"$match": {"a": 1}}, FACET,
                 {"$project": {"by_type": {"$ifNull": ["$by_type.count", 0]}}},
                 bad_sort, {"$limit": 1}])))

print("\nNegatives - must not touch anything healthy")
check("benign passthrough $project kept",
      len(strip([FACET, {"$project": {"by_type": 1, "by_month": 1}}])) == 2)
check("$limit after $facet kept", len(strip([FACET, {"$limit": 1}])) == 2)
check("$unwind after $facet kept", len(strip([FACET, {"$unwind": "$by_type"}])) == 2)

nofacet = [{"$match": {}},
           {"$group": {"_id": "$alert_type", "count": {"$sum": 1}}},
           {"$sort": {"count": -1}},
           {"$project": {"x": {"$ifNull": ["$a", 0]}}},
           {"$limit": 10}]
check("pipeline with no $facet left completely alone", strip(nofacet) == nofacet,
      str(strip(nofacet)))

pre = [{"$sort": {"timestamp": -1}},
       {"$project": {"x": {"$slice": ["$a", 0, 2]}}},
       FACET]
check("stages BEFORE the $facet left alone", strip(pre) == pre, str(strip(pre)))

print("\n" + "=" * 60)
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {', '.join(fails)}")
raise SystemExit(1 if fails else 0)
