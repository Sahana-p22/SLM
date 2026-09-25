"""Unit sanity checks for chat.backend.repair's pre-execution diagnose()
layer — this project's replacement for slm-llama3b's post-hoc Mongo pipeline
repair functions. Covers the ported equivalents of several of that project's
historically-fixed bug classes: sanity_count_target_ranking_shape.py
(misclassified_count_target), sanity_date_count_search_carryover.py's
underlying date-scope-inheritance concern (unrequested_date_filter /
strip_unrequested_date_filter), and general type/aggregation-filter
omissions this project's own diagnose() was built to catch pre-execution
rather than post-hoc. No server, no model."""
from chat.backend.repair import (diagnose, enforce_single_range,
                                 missing_normal_operation_exclusion,
                                 missing_type_filter,
                                 misclassified_count_target,
                                 needs_aggregation,
                                 strip_unrequested_date_filter,
                                 unrequested_date_filter, validate_sql,
                                 ValidationError)

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -> ' + detail) if detail and not cond else ''}")
    if not cond:
        fails.append(name)


print("misclassified_count_target - the exact bug fixed in slm-llama3b this session")
bad_sql = ("SELECT date(timestamp) AS d, COUNT(*) AS c FROM alerts "
           "WHERE alert_type != 'NORMAL_OPERATION' GROUP BY d ORDER BY c DESC LIMIT 1")
problem = misclassified_count_target("which days did we get 328 alerts", bad_sql)
check("flags a superlative-shaped query for a count-target question", problem is not None, str(problem))

good_sql = ("SELECT date(timestamp) AS d, COUNT(*) AS c FROM alerts "
            "WHERE alert_type != 'NORMAL_OPERATION' GROUP BY d HAVING c = 328")
check("does NOT flag a correctly-shaped HAVING COUNT(*) = N query",
      misclassified_count_target("which days did we get 328 alerts", good_sql) is None)

check("genuine superlative question ('which day had the MOST alerts') is left alone",
      misclassified_count_target("which day had the most alerts", bad_sql) is None)

check("a question naming no specific count target doesn't fire at all",
      misclassified_count_target("which day had the fewest alerts", bad_sql) is None)

print("\nmissing_type_filter - question names a specific alert type the SQL ignores")
check("'hand touch' named but SQL has no HAND_TOUCH filter -> flagged",
      missing_type_filter("how many hand touch alerts today",
                          "SELECT COUNT(*) FROM alerts WHERE timestamp >= '2026-09-25'") == "HAND_TOUCH")
check("SQL that already filters the named type is not flagged",
      missing_type_filter("how many hand touch alerts today",
                          "SELECT COUNT(*) FROM alerts WHERE alert_type = 'HAND_TOUCH'") is None)

print("\nmissing_normal_operation_exclusion - generic 'alerts' must exclude NORMAL_OPERATION")
check("generic alerts question with no exclusion -> flagged",
      missing_normal_operation_exclusion("how many alerts happened today",
                                        "SELECT COUNT(*) FROM alerts WHERE timestamp >= '2026-09-25'"))
check("SQL that already excludes NORMAL_OPERATION is not flagged",
      not missing_normal_operation_exclusion(
          "how many alerts happened today",
          "SELECT COUNT(*) FROM alerts WHERE alert_type != 'NORMAL_OPERATION'"))
check("a question specifically about normal operation is not flagged",
      not missing_normal_operation_exclusion(
          "how many normal operation events were logged",
          "SELECT COUNT(*) FROM alerts WHERE alert_type = 'NORMAL_OPERATION'"))

print("\nneeds_aggregation - a computed-figure question that only filters rows")
check("'how many' with no COUNT/GROUP BY -> flagged",
      needs_aggregation("how many alerts today", "SELECT * FROM alerts WHERE timestamp >= '2026-09-25'"))
check("'list all' with no aggregation is explicitly exempted",
      not needs_aggregation("list all alerts today", "SELECT * FROM alerts WHERE timestamp >= '2026-09-25'"))
check("a question that already has COUNT is not flagged",
      not needs_aggregation("how many alerts today", "SELECT COUNT(*) FROM alerts WHERE timestamp >= '2026-09-25'"))

print("\nunrequested_date_filter / strip_unrequested_date_filter (force-fix)")
spurious = "SELECT COUNT(*) FROM alerts WHERE alert_type = 'HAND_TOUCH' AND timestamp >= '2026-09-25'"
check("date-less question with a spurious date filter -> flagged",
      unrequested_date_filter("how many hand touch alerts in total", spurious, has_range=False) is not None)
stripped = strip_unrequested_date_filter(spurious)
check("force-fix actually removes the filter",
      "timestamp" not in stripped.lower())
check("force-fix leaves the real filter (alert_type) intact",
      "hand_touch" in stripped.lower())
check("a question that DOES name a period is never flagged",
      unrequested_date_filter("how many hand touch alerts this week", spurious, has_range=True) is None)

print("\nenforce_single_range (date-boundary drift force-fix)")
drifted = "SELECT COUNT(*) FROM alerts WHERE timestamp >= '2024-03-01 00:00:00' AND timestamp < '2024-04-02 00:00:00'"
fixed, changed = enforce_single_range(drifted, "2024-03-01 00:00:00", "2024-04-01 00:00:00")
check("drifted end boundary is corrected", "2024-04-01 00:00:00" in fixed)
check("reports that a change was made", changed)
same, unchanged = enforce_single_range(
    "SELECT COUNT(*) FROM alerts WHERE timestamp >= '2024-03-01 00:00:00' AND timestamp < '2024-04-01 00:00:00'",
    "2024-03-01 00:00:00", "2024-04-01 00:00:00")
check("already-correct bounds are reported as unchanged", not unchanged)

print("\nvalidate_sql - read-only enforcement (belt-and-suspenders under SQLite's own query_only pragma)")
for bad in ["DELETE FROM alerts", "DROP TABLE alerts", "INSERT INTO alerts VALUES (1)",
           "SELECT * FROM alerts; DROP TABLE alerts", "SELECT * FROM sqlite_master"]:
    try:
        validate_sql(bad)
        check(f"rejects: {bad!r}", False, "did not raise")
    except ValidationError:
        check(f"rejects: {bad!r}", True)
check("a plain SELECT against alerts passes", validate_sql("SELECT COUNT(*) FROM alerts").startswith("SELECT"))

print("\ndiagnose() - the combined pre-execution check used by pipeline._plan/_resolve")
check("catches a count-target-ranking misclassification end to end",
      diagnose("which days did we get 328 alerts", bad_sql, has_range=False) is not None)
check("a fully correct, in-scope query is not flagged",
      diagnose("how many hand touch alerts today",
               "SELECT COUNT(*) FROM alerts WHERE alert_type = 'HAND_TOUCH' "
               "AND timestamp >= '2026-09-25 00:00:00' AND timestamp < '2026-09-26 00:00:00'",
               has_range=True) is None)

print("\n" + "=" * 60)
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {', '.join(fails)}")
raise SystemExit(1 if fails else 0)
