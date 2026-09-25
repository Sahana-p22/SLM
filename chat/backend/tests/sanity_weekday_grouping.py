"""Unit sanity checks for the day-of-the-week grouping diagnose-check +
force-fix, added to this port after a live probe found a real bug: 'which
day of the week has the most hand touch alerts?' generated
`GROUP BY date(timestamp)` -- a single specific calendar date -- instead of
aggregating per weekday (Monday/Tuesday/etc). The question was asking about
a 7-value weekday category across the whole dataset, not one specific date.
No server, no model."""
from chat.backend.repair import enforce_weekday_grouping, wrong_weekday_grouping

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -> ' + detail) if detail and not cond else ''}")
    if not cond:
        fails.append(name)


BROKEN = "SELECT date(timestamp) AS day, COUNT(*) AS c FROM alerts WHERE alert_type = 'HAND_TOUCH' GROUP BY date(timestamp) ORDER BY c DESC LIMIT 1"
FIXED = ("SELECT CASE strftime('%w', timestamp) WHEN '0' THEN 'Sunday' WHEN '1' THEN 'Monday' "
        "WHEN '2' THEN 'Tuesday' WHEN '3' THEN 'Wednesday' WHEN '4' THEN 'Thursday' "
        "WHEN '5' THEN 'Friday' ELSE 'Saturday' END FROM alerts GROUP BY strftime('%w', timestamp)")

print("wrong_weekday_grouping - the exact live bug (diagnose-level check)")
check("flags 'GROUP BY date(timestamp)' for a 'day of the week' question",
      wrong_weekday_grouping("which day of the week has the most hand touch alerts?", BROKEN) is not None)
check("flags the 'weekday' phrasing variant too",
      wrong_weekday_grouping("what weekday has the most fast inspection alerts?", BROKEN) is not None)
check("flags 'days of the week' (plural) phrasing",
      wrong_weekday_grouping("on which days of the week do we get missing cleaning alerts?", BROKEN) is not None)
check("does NOT flag an already-correct strftime('%w', ...) grouping",
      wrong_weekday_grouping("which day of the week has the most alerts?",
                             "SELECT strftime('%w', timestamp) FROM alerts GROUP BY strftime('%w', timestamp)") is None)
check("does NOT flag a genuine 'which day' (specific-date) question -- must not regress this",
      wrong_weekday_grouping("which day had the most hand touch alerts?", BROKEN) is None)
check("does NOT flag an unrelated question with no weekday/date wording",
      wrong_weekday_grouping("how many hand touch alerts happened today?",
                             "SELECT COUNT(*) FROM alerts WHERE alert_type = 'HAND_TOUCH'") is None)

print("\nenforce_weekday_grouping - deterministic force-fix")
fixed, changed = enforce_weekday_grouping("which day of the week has the most hand touch alerts?", BROKEN)
check("rewrites 'date(timestamp)' into a weekday-name CASE expression",
      "strftime('%w', timestamp)" in fixed and "'Monday'" in fixed, fixed)
check("reports a change was made", changed)
check("leaves the rest of the SQL untouched",
      "alert_type = 'HAND_TOUCH'" in fixed and "ORDER BY c DESC LIMIT 1" in fixed)
check("rewrites BOTH the SELECT expression and the GROUP BY clause consistently",
      fixed.count("strftime('%w', timestamp)") >= 2, fixed)

fixed2, changed2 = enforce_weekday_grouping(
    "which day had the most hand touch alerts?", BROKEN)
check("a genuine 'which day' (specific-date) question is left completely untouched",
      not changed2 and fixed2 == BROKEN)

fixed3, changed3 = enforce_weekday_grouping(
    "how many hand touch alerts happened today?",
    "SELECT COUNT(*) FROM alerts WHERE alert_type = 'HAND_TOUCH'")
check("no date(timestamp) grouping at all -> no-op even for weekday-less questions",
      not changed3)

fixed4, changed4 = enforce_weekday_grouping(
    "what weekday has the most alerts?",
    "SELECT strftime('%w', timestamp) AS wd, COUNT(*) FROM alerts GROUP BY strftime('%w', timestamp)")
check("an already-correct weekday grouping is left alone (no date(timestamp) to rewrite)",
      not changed4)

print("\n" + "=" * 60)
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {', '.join(fails)}")
raise SystemExit(1 if fails else 0)
