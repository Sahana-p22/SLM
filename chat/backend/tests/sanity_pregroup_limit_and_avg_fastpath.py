"""Unit sanity checks for FIX L (pre-group $limit truncation), found by
the 157-prompt phrasing battery.
(The average-inspection-time fast path this file used to also cover was
removed along with every other fast path - see FIX HH in llm_query.py.)"""
from chat.backend import llm_query as L

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -> ' + detail) if detail and not cond else ''}")
    if not cond:
        fails.append(name)


print("FIX L - $limit dropped when it precedes the $group/$count it feeds")
observed = [
    {"$match": {"timestamp": {"$gte": "x", "$lt": "y"}, "alert_type": {"$ne": "NORMAL_OPERATION"}}},
    {"$match": {"alert_type": {"$ne": "NORMAL_OPERATION"}}},
    {"$limit": 200},
    {"$group": {"_id": None, "count": {"$sum": 1}}},
]
out = L._strip_pregroup_limit(observed)
check("pre-group $limit dropped", not any("$limit" in s for s in out), str(out))
check("everything else kept", len(out) == 3, str(out))

before_count = [{"$match": {"a": 1}}, {"$limit": 200}, {"$count": "total"}]
check("pre-$count $limit dropped too", not any("$limit" in s for s in L._strip_pregroup_limit(before_count)))

print("\nFIX L negatives - legitimate post-group $limit untouched")
legit = [
    {"$group": {"_id": "$alert_type", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}},
    {"$limit": 1},
]
check("$limit AFTER $group kept (superlative pattern)", L._strip_pregroup_limit(legit) == legit, str(L._strip_pregroup_limit(legit)))

no_group = [{"$match": {"a": 1}}, {"$limit": 25}]
check("no $group/$count at all -> $limit untouched", L._strip_pregroup_limit(no_group) == no_group)

facet_limit = [{"$facet": {"a": [{"$count": "n"}]}}, {"$limit": 1}]
check("$facet is not a $group - trailing $limit untouched",
      L._strip_pregroup_limit(facet_limit) == facet_limit, str(L._strip_pregroup_limit(facet_limit)))

print("\nEnd-to-end through _finalize_pipeline")
f = L._finalize_pipeline(
    {"intent": "data_query",
     "pipeline": [{"$match": {"alert_type": {"$ne": "NORMAL_OPERATION"}}},
                  {"$limit": 200},
                  {"$group": {"_id": None, "count": {"$sum": 1}}}],
     "explanation": ""},
    "total alerts, last 7 days")
check("no $limit survives before the $group in the final pipeline",
      not any(isinstance(s, dict) and "$limit" in s and "$group" not in s
              for s in (f.get("pipeline") or [])[:-1]),
      str(f.get("pipeline")))

print("\n" + "=" * 60)
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {', '.join(fails)}")
raise SystemExit(1 if fails else 0)
