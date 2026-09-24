"""Unit sanity checks for FIX BB (tomorrow never resolved to a date
range at all) and FIX CC (a bare day+month with no year silently
dropped the day and matched the whole month instead). Both found live
by the user testing the deployed chatbot directly."""
from datetime import datetime, timedelta, timezone
from chat.backend import llm_query as L

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -> ' + detail) if detail and not cond else ''}")
    if not cond:
        fails.append(name)


NOW = datetime(2026, 9, 24, 10, 0, tzinfo=timezone.utc)
TODAY0 = NOW.replace(hour=0, minute=0, second=0, microsecond=0)
L._now_for_absolute_date = lambda: NOW  # pin "now" for deterministic checks

print("FIX BB - 'tomorrow' resolves to a real one-day future window")
check("'how many alerts tomorrow'",
      L._extract_relative_date_range("how many alerts tomorrow", NOW)
      == (TODAY0 + timedelta(days=1), TODAY0 + timedelta(days=2)))
check("'alerts happened tomorrow?'",
      L._extract_relative_date_range("alerts happened tomorrow?", NOW)
      == (TODAY0 + timedelta(days=1), TODAY0 + timedelta(days=2)))

print("\nFIX BB negatives - regression on today/yesterday")
check("'today' still works",
      L._extract_relative_date_range("how many alerts today", NOW) == (TODAY0, TODAY0 + timedelta(days=1)))
check("'yesterday' still works",
      L._extract_relative_date_range("how many alerts yesterday", NOW)
      == (TODAY0 - timedelta(days=1), TODAY0))

print("\nFIX CC - bare day+month (no year) resolves to that exact single day")
check("'september 24th' (month day, no year) -> this year, since it's today",
      L._extract_absolute_date("how many alerts for september 24th") == datetime(2026, 9, 24, tzinfo=timezone.utc))
check("'24th of september' (day month, no year)",
      L._extract_absolute_date("how many alerts for the 24th of september") == datetime(2026, 9, 24, tzinfo=timezone.utc))
check("'september 30th' (future this year) -> rolls back to last year",
      L._extract_absolute_date("how many alerts for september 30th") == datetime(2025, 9, 30, tzinfo=timezone.utc))
check("'january 5th' (already passed this year) -> stays this year",
      L._extract_absolute_date("how many alerts for january 5th") == datetime(2026, 1, 5, tzinfo=timezone.utc))

print("\nFIX CC negatives - a real year still wins outright, no ambiguity")
check("'september 24th 2026' still resolves via the year-bearing pattern",
      L._extract_absolute_date("how many alerts for september 24th 2026") == datetime(2026, 9, 24, tzinfo=timezone.utc))
check("'september 24 2020' (a different, explicit year) is honored exactly",
      L._extract_absolute_date("how many alerts for september 24 2020") == datetime(2020, 9, 24, tzinfo=timezone.utc))

print("\nFIX CC negatives - bare month alone (no day) still means the WHOLE month")
r = L._extract_relative_date_range("how many alerts for september", NOW)
check("'for september' (no day at all) spans the whole month",
      r == (datetime(2026, 9, 1, tzinfo=timezone.utc), datetime(2026, 10, 1, tzinfo=timezone.utc)), str(r))
r2 = L._extract_relative_date_range("give me a report for july 2026", NOW)
check("'report for july 2026' (month+year, no day) still spans the whole month, not a false day match",
      r2 == (datetime(2026, 7, 1, tzinfo=timezone.utc), datetime(2026, 8, 1, tzinfo=timezone.utc)), str(r2))

print("\n" + "=" * 60)
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {', '.join(fails)}")
raise SystemExit(1 if fails else 0)
