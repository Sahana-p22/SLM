"""Unit sanity checks for FIX X/Y - report $facet robustness when the
model's own facet deviates from the fixed 5-section contract."""
from datetime import datetime, timezone
from chat.backend import llm_query as L

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -> ' + detail) if detail and not cond else ''}")
    if not cond:
        fails.append(name)


print("FIX Y - inject missing mandatory report sections")
match_stage = {"$match": {"alert_type": {"$ne": "NORMAL_OPERATION"},
                          "timestamp": {"$gte": datetime(2026, 9, 1, tzinfo=timezone.utc),
                                        "$lt": datetime(2026, 9, 25, tzinfo=timezone.utc)}}}
by_day_only = [match_stage, {"$facet": {"by_day": [
    {"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}}, "count": {"$sum": 1}}},
    {"$sort": {"_id": 1}},
]}}]
out = L._ensure_report_facet_sections([dict(s) for s in by_day_only])
facet_spec = out[1]["$facet"]
check("all 5 mandatory sections present after injection",
      set(facet_spec.keys()) == set(L._MANDATORY_REPORT_SECTIONS), str(set(facet_spec.keys())))
check("the model's own by_day sub-pipeline untouched",
      facet_spec["by_day"] == by_day_only[1]["$facet"]["by_day"])

print("\nFIX Y negatives - must not touch a non-report facet")
compare_facet = [match_stage, {"$facet": {"this_week": [{"$count": "n"}], "last_week": [{"$count": "n"}]}}]
check("comparison facet (this_week/last_week) left alone",
      L._ensure_report_facet_sections([dict(s) for s in compare_facet]) == compare_facet)
already_complete = [match_stage, {"$facet": {s: [{"$count": "n"}] for s in L._MANDATORY_REPORT_SECTIONS}}]
check("already-complete report facet left alone",
      L._ensure_report_facet_sections([dict(s) for s in already_complete]) == already_complete)
no_facet = [match_stage, {"$count": "total"}]
check("no facet at all -> untouched", L._ensure_report_facet_sections([dict(s) for s in no_facet]) == no_facet)

print("\nFIX X - by_month-only facet still derives by_week via direct fetch")
by_month_only_rows = [{
    "total_alerts": [{"total": 4940}],
    "by_type": [{"_id": "FAST_INSPECTION", "count": 2282}],
    "by_month": [{"_id": "2026-06", "count": 4940}],
    "avg_inspection_time": [{"_id": None, "avg": 10.43}],
}]
june_pipeline = [{"$match": {"timestamp": {"$gte": datetime(2026, 6, 1, tzinfo=timezone.utc),
                                           "$lt": datetime(2026, 7, 1, tzinfo=timezone.utc)}}}]
out2 = L._restructure_report(by_month_only_rows, june_pipeline)
check("by_week present after restructuring a by_month-only facet",
      "by_week" in out2[0], str(list(out2[0].keys())) if out2 and isinstance(out2[0], dict) else str(out2))
check("by_month no longer present (superseded by by_week for month scope)",
      "by_month" not in out2[0], str(out2))

print("\nFIX X negatives - a genuinely empty range still returns rows unchanged")
far_future = [{"$match": {"timestamp": {"$gte": datetime(3000, 1, 1, tzinfo=timezone.utc),
                                        "$lt": datetime(3000, 2, 1, tzinfo=timezone.utc)}}}]
empty_rows = [{"total_alerts": [], "by_type": [], "avg_inspection_time": []}]
check("no by_day, no data anywhere -> returns rows unchanged, no crash",
      L._restructure_report(empty_rows, far_future) == empty_rows)

print("\n" + "=" * 60)
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {', '.join(fails)}")
raise SystemExit(1 if fails else 0)
