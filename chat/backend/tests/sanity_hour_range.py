"""Unit sanity checks for the hour-of-day range extraction + force-fix,
added to this port after regression_suite_sql.py's hour_range_2pm_4pm_last_30d
case found a real, live bug: the model wrote 'hour = 14' for 'between 2pm and
4pm', collapsing the range to a single hour and silently dropping the '4pm'
half. Ported concept (not mechanism) from slm-llama3b's FIX Z /
sanity_hour_range_phrasing.py. No server, no model."""
from chat.backend.dates import extract_hour_range
from chat.backend.repair import enforce_hour_range, wrong_hour_range

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -> ' + detail) if detail and not cond else ''}")
    if not cond:
        fails.append(name)


print("extract_hour_range - bare 'X to/and Y' phrasings")
for phrase, expected in [
    ("2pm to 4pm", (14, 16)), ("2pm and 4pm", (14, 16)),
    ("9am to 11am", (9, 11)), ("6pm to 8pm", (18, 20)),
    ("10am to 12pm", (10, 12)), ("1am to 3am", (1, 3)),
    ("11pm to 1am", (23, 1)),  # overnight wraparound
]:
    q = f"how many alerts happened between {phrase}"
    got = extract_hour_range(q)
    check(f"{q!r} -> {expected}", got == expected, str(got))

print("\nextract_hour_range - qualitative time-of-day words")
for word, expected in [("morning", (6, 12)), ("afternoon", (12, 17)),
                       ("evening", (17, 21)), ("night", (21, 24))]:
    q = f"how many alerts happened yesterday {word}"
    check(f"{q!r} -> {expected}", extract_hour_range(q) == expected)

print("\nextract_hour_range - negatives")
check("bare 'between 3 and 5' with no am/pm -> None (ambiguous)",
      extract_hour_range("how many alerts between 3 and 5") is None)
check("'last 3 to 5 days' (not an hour range) -> None",
      extract_hour_range("how many alerts in the last 3 to 5 days") is None)
check("no time phrase at all -> None",
      extract_hour_range("how many alerts happened today") is None)

print("\nwrong_hour_range - the exact live bug (diagnose-level check)")
check("flags 'hour = 14' when the question named a 2pm-4pm range",
      wrong_hour_range((14, 16), "SELECT COUNT(*) FROM alerts WHERE hour = 14") is not None)
check("does NOT flag a correctly-shaped range filter",
      wrong_hour_range((14, 16), "SELECT COUNT(*) FROM alerts WHERE hour >= 14 AND hour < 16") is None)
check("flags a missing hour filter entirely when a range was named",
      wrong_hour_range((14, 16), "SELECT COUNT(*) FROM alerts WHERE alert_type = 'HAND_TOUCH'") is not None)
check("no range named -> never flagged",
      wrong_hour_range(None, "SELECT COUNT(*) FROM alerts WHERE hour = 14") is None)

print("\nwrong_hour_range / enforce_hour_range - the BETWEEN-inclusive variant")
check("flags 'hour BETWEEN 14 AND 16' (inclusive, over-includes hour 16) for a 2pm-4pm range",
      wrong_hour_range((14, 16), "SELECT COUNT(*) FROM alerts WHERE hour BETWEEN 14 AND 16") is not None)
check("does NOT flag the correctly-adjusted 'hour BETWEEN 14 AND 15'",
      wrong_hour_range((14, 16), "SELECT COUNT(*) FROM alerts WHERE hour BETWEEN 14 AND 15") is None)
fixed_b, changed_b = enforce_hour_range("SELECT COUNT(*) FROM alerts WHERE hour BETWEEN 14 AND 16", 14, 16)
check("force-fixes the inclusive BETWEEN into the correct half-open range",
      "hour >= 14 AND hour < 16" in fixed_b, fixed_b)
check("reports a change was made for the BETWEEN case", changed_b)
fixed_b2, changed_b2 = enforce_hour_range("SELECT COUNT(*) FROM alerts WHERE hour BETWEEN 14 AND 15", 14, 16)
check("an already-correct BETWEEN (14 AND 15) is left alone", not changed_b2)

print("\nenforce_hour_range - deterministic force-fix")
fixed, changed = enforce_hour_range("SELECT COUNT(*) FROM alerts WHERE hour = 14 AND alert_type != 'X'", 14, 16)
check("rewrites 'hour = 14' into a proper range", "hour >= 14 AND hour < 16" in fixed, fixed)
check("reports a change was made", changed)
check("leaves the rest of the SQL untouched", "alert_type != 'X'" in fixed)

fixed2, changed2 = enforce_hour_range(
    "SELECT COUNT(*) FROM alerts WHERE hour >= 23 OR hour < 1", 23, 1)
check("an existing correct wraparound (2 comparisons) is left alone",
      not changed2 and fixed2 == "SELECT COUNT(*) FROM alerts WHERE hour >= 23 OR hour < 1")

fixed3, changed3 = enforce_hour_range("SELECT COUNT(*) FROM alerts WHERE hour = 23", 23, 1)
check("overnight wraparound rewrite uses OR, not a backwards AND range",
      "hour >= 23 OR hour < 1" in fixed3, fixed3)

print("\n" + "=" * 60)
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {', '.join(fails)}")
raise SystemExit(1 if fails else 0)
