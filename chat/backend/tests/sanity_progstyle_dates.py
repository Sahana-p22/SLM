"""Unit sanity checks for item 7's Mongo equivalent - programming-style
relative-time syntax recognized by _extract_relative_date_range."""
from datetime import datetime, timezone, timedelta
from chat.backend import llm_query as L

NOW = datetime(2026, 9, 23, 15, 0, tzinfo=timezone.utc)
fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -> ' + detail) if detail and not cond else ''}")
    if not cond:
        fails.append(name)


def rng(q):
    return L._extract_relative_date_range(q, NOW)


print("now()-Nd / now()-Nh / now()-Nm syntax")
check("'timestamp>=now()-7d' matches 'last 7 days'",
      rng("count(timestamp>=now()-7d)") == rng("count in the last 7 days"))
check("NOW()-30d (uppercase) matches 'last 30 days'",
      rng("WHERE ts BETWEEN NOW()-30d AND NOW()") == rng("in the last 30 days"))
check("now()-6h matches 'last 6 hours'",
      rng("alerts where ts >= now()-6h") == rng("alerts in the last 6 hours"))
check("now()-45m matches 'last 45 minutes'",
      rng("count(ts>=now()-45m)") == rng("count in the last 45 minutes"))
check("plain 'now()' with no offset does not match (no digits) -> None",
      rng("select * where ts < now()") is None)

print("\ncurrent_week / current_month tokens")
check("'week==current_week' matches 'this week'",
      rng("alerts.filter(week==current_week)") == rng("alerts this week"))
check("'current_week==week' (reversed) also matches",
      rng("current_week==week") == rng("this week"))
check("'month==current_month' matches 'this month'",
      rng("alerts.month==current_month") == rng("alerts this month"))

print("\ntoday()/==today SQL-ish forms")
check("'date==today' matches 'today'", rng("alerts.date==today") == rng("alerts today"))
check("'date==today()' matches 'today'", rng("alerts.date==today()") == rng("alerts today"))

print("\nNegatives - must not misfire on ordinary questions")
check("clean natural language totally unaffected",
      rng("how many alerts today") is not None)
check("no time phrase at all -> None",
      rng("what's the average inspection time") is None)
check("a bare 'week' with no relation operator -> no override",
      rng("break down alerts by week") is None)
check("existing 'last 7 days' phrasing unaffected by the new patterns",
      rng("alerts in the last 7 days") == (
          NOW.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=6),
          NOW.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)))

print("\n" + "=" * 60)
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {', '.join(fails)}")
raise SystemExit(1 if fails else 0)
