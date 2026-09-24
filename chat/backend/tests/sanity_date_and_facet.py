"""Unit-level sanity checks for the three accuracy fixes. No server, no model."""
from datetime import datetime, timezone
from chat.backend import llm_query as L

NOW = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)
fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -> ' + detail) if detail and not cond else ''}")
    if not cond:
        fails.append(name)


print("FIX A - bare month name -> date range")
r = L._extract_relative_date_range("give me a report for June", NOW)
check("'report for June' resolves", r is not None, f"got {r}")
if r:
    check("June starts 2026-06-01", r[0] == datetime(2026, 6, 1, tzinfo=timezone.utc), str(r[0]))
    check("June ends 2026-07-01", r[1] == datetime(2026, 7, 1, tzinfo=timezone.utc), str(r[1]))

r2 = L._extract_relative_date_range("give me a report for December", NOW)
check("future month rolls to last year (Dec 2025)",
      r2 and r2[0] == datetime(2025, 12, 1, tzinfo=timezone.utc), str(r2))

r3 = L._extract_relative_date_range("give me a report for July 2026", NOW)
check("explicit month+year still wins",
      r3 and r3[0] == datetime(2026, 7, 1, tzinfo=timezone.utc), str(r3))

# "may" must not hijack unrelated questions
r4 = L._extract_relative_date_range("how many alerts may have been missed today", NOW)
check("modal 'may' not treated as month",
      r4 is not None and r4[0].month != 5, str(r4))

# regression: existing relative phrasing untouched
r5 = L._extract_relative_date_range("how many alerts in the last 7 days", NOW)
check("'last 7 days' still works", r5 is not None and (r5[1] - r5[0]).days == 7, str(r5))

print("\nFIX B - lossy post-$facet $project dropped")
lossy = [
    {"$match": {"alert_type": {"$ne": "NORMAL_OPERATION"}}},
    {"$facet": {"by_type": [{"$group": {"_id": "$alert_type", "count": {"$sum": 1}}}]}},
    {"$project": {"by_type": {"$concatArrays": ["$by_type.count"]}}},
]
out = L._strip_lossy_facet_projection(lossy)
check("lossy $project removed", len(out) == 2 and "$facet" in out[-1], str(out))

keep = [
    {"$facet": {"by_type": [{"$group": {"_id": "$alert_type", "count": {"$sum": 1}}}]}},
    {"$project": {"by_type": 1}},
]
check("benign $project after $facet kept", len(L._strip_lossy_facet_projection(keep)) == 2)

nofacet = [{"$match": {}}, {"$project": {"x": {"$concatArrays": ["$a.b"]}}}]
check("$project without a $facet left alone", len(L._strip_lossy_facet_projection(nofacet)) == 2)

print("\nFIX C - truncating $limit on breakdowns dropped")
bd = [
    {"$match": {}},
    {"$group": {"_id": "$alert_type", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}},
    {"$limit": 1},
]
out = L._strip_truncating_breakdown_limit(bd, "Break down this month's alerts by type")
check("breakdown $limit 1 removed", all("$limit" not in s for s in out), str(out))

out = L._strip_truncating_breakdown_limit(bd, "Which day had the most hand touch alerts?")
check("superlative keeps its $limit 1", any("$limit" in s for s in out), str(out))

out = L._strip_truncating_breakdown_limit(
    [{"$group": {"_id": None, "n": {"$sum": 1}}}, {"$limit": 1}], "break down by type")
check("$group _id:null (scalar) keeps limit", any("$limit" in s for s in out), str(out))

out = L._strip_truncating_breakdown_limit(bd, "how many alerts happened today")
check("non-breakdown question untouched", any("$limit" in s for s in out), str(out))

big = [{"$group": {"_id": "$alert_type", "count": {"$sum": 1}}}, {"$limit": 200}]
check("large cap (200) preserved",
      any("$limit" in s for s in L._strip_truncating_breakdown_limit(big, "break down by type")))

print("\n" + "=" * 60)
print(f"{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
raise SystemExit(1 if fails else 0)
