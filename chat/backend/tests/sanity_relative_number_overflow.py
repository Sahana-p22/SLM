"""Unit sanity checks for FIX AA - unbounded quantity phrases ("last
999999999 days") no longer overflow datetime arithmetic and crash with a
500. Found live via the edge_cases battery."""
from datetime import datetime, timezone
from chat.backend import llm_query as L

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -> ' + detail) if detail and not cond else ''}")
    if not cond:
        fails.append(name)


NOW = datetime(2026, 9, 24, 10, 0, tzinfo=timezone.utc)

print("FIX AA - huge quantities no longer raise, and clamp to a sane ceiling")
for q, unit_word in [
    ("how many alerts happened in the last 999999999 days", "day"),
    ("how many alerts happened in the last 10000000000000 minutes", "minute"),
    ("how many alerts in the last 999999999999 hours", "hour"),
    ("how many alerts in the last 999999999 weeks", "week"),
    ("how many alerts in the last 999999999 months", "month"),
    ("how many alerts in the last 999999999 years", "year"),
]:
    try:
        got = L._extract_relative_date_range(q, NOW)
    except (OverflowError, ValueError) as e:
        check(f"no crash: {q!r}", False, f"raised {type(e).__name__}: {e}")
        continue
    check(f"no crash, returns a real range: {q!r}", got is not None and got[0] < got[1], str(got))
    if got:
        check(f"clamped start stays within datetime's valid range: {q!r}",
              datetime(1, 1, 1, tzinfo=timezone.utc) < got[0] < NOW)

print("\nFIX AA negatives - ordinary small quantities unaffected")
check("'last 7 days' still gives a 7-8 day window",
      (lambda r: r is not None and 6 <= (r[1] - r[0]).days <= 8)(
          L._extract_relative_date_range("how many alerts in the last 7 days", NOW)))
check("'last 3 months' still resolves to a real range",
      L._extract_relative_date_range("how many alerts in the last 3 months", NOW) is not None)

print("\nFIX AA - clamp helper directly")
check("day ceiling applied", L._clamp_rel_number(999_999_999, "day") == 36_500)
check("small value passes through unclamped", L._clamp_rel_number(5, "day") == 5)
check("unknown unit passes through unclamped", L._clamp_rel_number(999_999_999, "quarter") == 999_999_999)

print("\n" + "=" * 60)
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {', '.join(fails)}")
raise SystemExit(1 if fails else 0)
