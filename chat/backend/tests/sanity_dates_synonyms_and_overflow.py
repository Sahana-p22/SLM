"""Unit sanity checks for chat.backend.dates.extract_range, ported from
slm-llama3b's Mongo-side FIX AA (unbounded quantity overflow), FIX BB
("tomorrow" never resolved), and FIX CC (bare day+month with no year
silently matched the whole month) — plus this.dates.py's own this/last/next
synonym coverage. Genuine question-answering behavior, not a Mongo-pipeline
mechanic, so it ported cleanly onto the SQL/SQLite side; no server, no
model."""
from datetime import datetime

from chat.backend.dates import extract_range

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -> ' + detail) if detail and not cond else ''}")
    if not cond:
        fails.append(name)


NOW = datetime(2026, 9, 25, 12, 0, 0)


def rng(q):
    return extract_range(q, NOW)


print("FIX AA equivalent - huge quantities no longer raise, and clamp to a sane ceiling")
for q in [
    "the last 999999999 days",
    "the last 10000000000000 minutes",
    "the last 999999999999 hours",
    "the last 999999999 weeks",
    "the last 999999999 months",
]:
    try:
        got = rng(q)
    except OverflowError as e:
        check(f"no crash: {q!r}", False, f"raised OverflowError: {e}")
        continue
    check(f"no crash, returns a real range: {q!r}", got is not None and got[0] < got[1], str(got))
    if got:
        check(f"clamped start stays within datetime's valid range: {q!r}",
              datetime(1, 1, 1) < got[0] < NOW)

print("\nFIX AA negatives - ordinary small quantities unaffected")
check("'last 7 days' still gives a 7-8 day window",
      (lambda r: r is not None and 6 <= (r[1] - r[0]).days <= 8)(rng("the last 7 days")))
check("'last 3 months' still resolves to a real range", rng("the last 3 months") is not None)

print("\nFIX BB equivalent - 'tomorrow'")
check("'tomorrow' resolves to a real single-day range",
      rng("tomorrow") == (datetime(2026, 9, 26), datetime(2026, 9, 27)))

print("\nFIX CC equivalent - bare day+month, no year, must be THAT DAY not the whole month")
check("'march 3rd' -> March 3rd only, not the whole month",
      rng("what happened on march 3rd") == (datetime(2026, 3, 3), datetime(2026, 3, 4)))
check("'3rd of march' (day-first, no year) -> March 3rd only",
      rng("3rd of march") == (datetime(2026, 3, 3), datetime(2026, 3, 4)))
check("a real invalid date (feb 30) with no year falls through gracefully, not a crash",
      rng("what happened on february 30th") is not None)  # falls through to month-only match

print("\nthis/last/next synonyms (week/month/quarter/year)")
check("'this week' starts on a Monday <= now",
      (lambda r: r is not None and r[0].weekday() == 0 and r[0] <= NOW)(rng("this week")))
check("'next week' starts after this week",
      rng("next week")[0] > rng("this week")[0])
check("'last month' is the calendar month before now's",
      rng("last month") == (datetime(2026, 8, 1), datetime(2026, 9, 1)))
check("'next month' is the calendar month after now's",
      rng("next month") == (datetime(2026, 10, 1), datetime(2026, 11, 1)))
check("'this quarter' is Jul-Oct 2026 (Q3)",
      rng("this quarter") == (datetime(2026, 7, 1), datetime(2026, 10, 1)))
check("'last quarter' is Apr-Jul 2026 (Q2)",
      rng("last quarter") == (datetime(2026, 4, 1), datetime(2026, 7, 1)))
check("'next year' is calendar year 2027",
      rng("next year") == (datetime(2027, 1, 1), datetime(2028, 1, 1)))

print("\nExisting exact-date and month+year cases unaffected by this pass's changes")
check("'september 19th 2023' unaffected",
      rng("september 19th 2023") == (datetime(2023, 9, 19), datetime(2023, 9, 20)))
check("'march 2024' (month+year, no day) still means the whole month",
      rng("march 2024") == (datetime(2024, 3, 1), datetime(2024, 4, 1)))
check("no time phrase at all -> None", rng("what's the average inspection time") is None)

print("\n" + "=" * 60)
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {', '.join(fails)}")
raise SystemExit(1 if fails else 0)
