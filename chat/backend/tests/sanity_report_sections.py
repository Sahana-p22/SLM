"""Unit sanity checks for chat.backend.report — this project's deterministic
report builder, ported from the Retail deployment's report.py and reshaped
to slm-llama3b's own 5-section report contract (total_alerts, by_type,
by_month/by_week/by_day depending on scope, avg_inspection_time — see
sanity_report_sections.py / sanity_fqc_report_sections.py in the original
Mongo project for the equivalent pipeline-shape checks this ports the
INTENT of, not the mechanism, since there is no $facet here at all).

Runs real read-only SQL against this deployment's own fqc.db — no model,
no HTTP server, but not a pure-Python unit test either; numbers are
independently re-derived here (a small local oracle) rather than trusted
from report.build() itself, so a real regression in the report's SQL
would be caught, not just a crash."""
from datetime import datetime

from chat.backend.db import run_readonly
from chat.backend.report import build, is_report_request

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -> ' + detail) if detail and not cond else ''}")
    if not cond:
        fails.append(name)


NOW = datetime(2026, 9, 25, 12, 0, 0)


def oracle_total(a, b):
    row = run_readonly(
        "SELECT COUNT(*) AS n FROM alerts WHERE alert_type != 'NORMAL_OPERATION' "
        "AND timestamp >= ? AND timestamp < ?", (a, b))
    return row[0]["n"]


def oracle_type_count(a, b, t):
    row = run_readonly(
        "SELECT COUNT(*) AS n FROM alerts WHERE alert_type = ? "
        "AND timestamp >= ? AND timestamp < ?", (t, a, b))
    return row[0]["n"]


print("is_report_request classification")
check("'give me a quarterly report' is a report request", is_report_request("give me a quarterly report"))
check("'weekly summary' is a report request", is_report_request("weekly summary"))
check("'how many alerts today' is NOT a report request", not is_report_request("how many alerts today"))
check("'day wise report' is a report request", is_report_request("day wise report for this month"))

print("\nquarterly report - every mandatory section present, numbers oracle-verified")
rep = build("give me a quarterly report", NOW)
check("build() returns a result", rep is not None)
facet = rep["result"][0]
for section in ("total_alerts", "by_type", "by_month", "by_week", "avg_inspection_time"):
    check(f"section '{section}' is present", section in facet)
a, b = datetime(2026, 7, 1), datetime(2026, 10, 1)  # Q3 2026, calendar-aligned
check("total_alerts matches an independent oracle count",
      facet["total_alerts"] == oracle_total(a.isoformat(sep=" "), b.isoformat(sep=" ")),
      f"report said {facet['total_alerts']}")
by_type_map = {r["alert_type"]: r["count"] for r in facet["by_type"]}
for t in ("FAST_INSPECTION", "HAND_TOUCH", "MISSING_CLEANING"):
    oracle = oracle_type_count(a.isoformat(sep=" "), b.isoformat(sep=" "), t)
    check(f"by_type[{t}] matches an independent oracle count",
          by_type_map.get(t) == oracle, f"report said {by_type_map.get(t)}, oracle says {oracle}")
check("by_type counts sum to total_alerts",
      sum(by_type_map.values()) == facet["total_alerts"])
check("answer sentence mentions the total figure",
      f"{facet['total_alerts']:,}" in rep["answer"])
check("answer sentence is non-empty and doesn't crash on formatting",
      len(rep["answer"]) > 20)

print("\nweekly report - by_day section present instead of by_month/by_week")
rep_w = build("weekly report", NOW)
facet_w = rep_w["result"][0]
check("by_day is populated for a week-scope report", len(facet_w["by_day"]) > 0)
check("by_day day-counts sum to the reported total",
      sum(r["count"] for r in facet_w["by_day"]) == facet_w["total_alerts"])

print("\nmonthly report - by_week (type-split) section present")
rep_m = build("monthly report for september 2026", NOW)
facet_m = rep_m["result"][0]
check("by_week is populated for a month-scope report", len(facet_m["by_week"]) > 0)
check("by_week rows carry a 'type' field (per-week type split, matching the "
      "original Mongo report's by_week-for-month-scope shape)",
      all("type" in r for r in facet_m["by_week"]))

print("\nreport with an explicit period name uses THAT period, not the default")
rep_june = build("give me a report for june 2026", NOW)
check("explicit month name in the question is honored",
      "June 2026" in rep_june["answer"])

print("\n" + "=" * 60)
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {', '.join(fails)}")
raise SystemExit(1 if fails else 0)
