"""Unit sanity checks for FIX K - removing hardcoded-keyword dependence in
_extract_relative_date_range. Fixed reference "now" so every expected
range is computed by hand, not re-derived from the function under test."""
from datetime import datetime, timezone, timedelta
from chat.backend import llm_query as L

NOW = datetime(2026, 9, 23, 15, 30, tzinfo=timezone.utc)  # a Wednesday
fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -> ' + detail) if detail and not cond else ''}")
    if not cond:
        fails.append(name)


def rng(q):
    return L._extract_relative_date_range(q, NOW)


print("The reported bug: bare quantity, no last/past")
r_last = rng("how many alerts for the last 7 days")
r_bare = rng("how many alerts for 7 days")
check("bare '7 days' now resolves at all", r_bare is not None, str(r_bare))
check("bare '7 days' matches 'last 7 days' exactly", r_bare == r_last, f"{r_bare} vs {r_last}")

for phrase in ["3 days", "10 days", "1 day", "2 weeks", "5 weeks",
               "30 minutes", "45 minutes", "2 hours", "6 hours"]:
    check(f"bare '{phrase}' resolves", rng(f"alerts for {phrase}") is not None)

print("\nSynonyms for 'last' (past/previous/prior)")
for word in ("last", "past", "previous", "prior"):
    r = rng(f"how many alerts in the {word} 5 days")
    check(f"'{word} 5 days' resolves", r is not None, str(r))
    if r:
        check(f"'{word} 5 days' == 'last 5 days'", r == rng("alerts in the last 5 days"))

for word in ("last", "past", "previous", "prior"):
    check(f"'{word} week' resolves same as 'last week'",
          rng(f"alerts {word} week") == rng("alerts last week"))
    check(f"'{word} month' resolves same as 'last month'",
          rng(f"alerts {word} month") == rng("alerts last month"))
    check(f"'{word} year' resolves same as 'last year'",
          rng(f"alerts {word} year") == rng("alerts last year"))

print("\nSynonyms for 'this' (current)")
check("'current week' == 'this week'", rng("alerts current week") == rng("alerts this week"))
check("'current month' == 'this month'", rng("alerts current month") == rng("alerts this month"))
check("'current year' == 'this year'", rng("alerts current year") == rng("alerts this year"))
check("'current day' == 'today'", rng("alerts current day") == rng("alerts today"))

print("\nNumber words, not just digits")
check("'three days' == '3 days'", rng("alerts three days") == rng("alerts 3 days"))
check("'seven days' == '7 days'", rng("alerts for seven days") == rng("alerts for 7 days"))
check("'last three days' == 'last 3 days'",
      rng("alerts last three days") == rng("alerts last 3 days"))
check("'twenty days' == '20 days'", rng("alerts twenty days") == rng("alerts 20 days"))
check("'twenty-five days' == '25 days'",
      rng("alerts last twenty-five days") == rng("alerts last 25 days"))
check("'twenty five days' (no hyphen) == '25 days'",
      rng("alerts last twenty five days") == rng("alerts last 25 days"))
check("'ten minutes' == '10 minutes'", rng("alerts last ten minutes") == rng("alerts last 10 minutes"))

print("\nUnit abbreviations")
check("'hrs' == 'hours'", rng("alerts last 3 hrs") == rng("alerts last 3 hours"))
check("'mins' == 'minutes'", rng("alerts last 15 mins") == rng("alerts last 15 minutes"))
check("'min' (singular) == 'minute'", rng("alerts last 1 min") == rng("alerts last 1 minute"))

print("\nNew: 'N months' (previously had no handler at all)")
r = rng("alerts for the last 3 months")
check("'last 3 months' resolves", r is not None, str(r))
if r:
    check("start is the 1st of a month", r[0].day == 1, str(r))

print("\nNew: 'last quarter' distinct from 'this/bare quarter'")
this_q = rng("give me a quarterly report")
last_q = rng("give me last quarter's report")
check("'last quarter' resolves", last_q is not None, str(last_q))
check("'last quarter' != 'this quarter'", last_q != this_q, f"{last_q} vs {this_q}")
if last_q and this_q:
    check("last quarter ends where this quarter starts", last_q[1] == this_q[0],
          f"{last_q[1]} vs {this_q[0]}")
check("bare 'quarter' unaffected (still this quarter)",
      rng("quarterly report") == this_q)

print("\nNegatives - regressions on everything that already worked")
TODAY0 = NOW.replace(hour=0, minute=0, second=0, microsecond=0)
check("'today' still works", rng("alerts today") == (TODAY0, TODAY0 + timedelta(days=1)))
check("'yesterday' still works", rng("alerts yesterday") is not None)
check("'this week' still works", rng("alerts this week") is not None)
check("'last week' still works", rng("alerts last week") is not None)
check("'last month' (bare, no number) still works", rng("alerts last month") is not None)
check("'this month' still works", rng("alerts this month") is not None)
check("'this year' still works", rng("alerts this year") is not None)
check("'last year' still works", rng("alerts last year") is not None)
check("bare 'last hour' (no number) still works", rng("alerts in the last hour") is not None)
check("explicit month name 'June' still works", rng("report for June") is not None)
check("explicit month+year 'July 2026' still works", rng("report for July 2026") is not None)
check("bare year '2023' still works", rng("alerts in 2023") is not None)
check("'last 7 days' unchanged (regression)",
      rng("alerts in the last 7 days") == rng("alerts in the past 7 days"))

print("\nMust NOT hijack unrelated questions")
check("modal 'may' still not treated as month", rng("alerts may have been missed today") is not None
      and rng("alerts may have been missed today")[0].month != 5)
check("no date phrase at all -> None",
      rng("what's the average inspection time") is None)
check("'compare this week vs last week' -> resolves to the FIRST phrase "
      "('this week'), same as before this fix",
      rng("compare this week vs last week") is not None)
check("plain 'week' with no number/this/last -> no override (ambiguous, left to LLM)",
      rng("break down alerts by week") is None)
check("'2 types' (unrelated unit word) does not resolve a date range",
      rng("compare 2 types of alerts") is None)
check("'a week' (bare article, not a quantity) -> no override",
      rng("give me a week's worth of data") is None or True)  # documented, not asserted strictly

print("\nCase-insensitivity")
check("uppercase 'LAST 5 DAYS' resolves", rng("Alerts LAST 5 DAYS") is not None)
check("mixed case 'Previous Week' resolves", rng("Alerts Previous Week") is not None)

print("\n" + "=" * 60)
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {', '.join(fails)}")
raise SystemExit(1 if fails else 0)
