"""Unit sanity checks for FIX F - unmentioned multi-row $facet sections."""
from chat.backend import llm_query as L

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -> ' + detail) if detail and not cond else ''}")
    if not cond:
        fails.append(name)


FACET = {
    "total_alerts": [{"total": 18714}],
    "by_type": [{"_id": "FAST_INSPECTION", "count": 8682},
                {"_id": "HAND_TOUCH", "count": 7280},
                {"_id": "MISSING_CLEANING", "count": 2752}],
    "by_month": [{"_id": "2026-06", "count": 4940}, {"_id": "2026-07", "count": 4983}],
    "by_week": [{"_id": "2026-W23", "count": 1200},
                {"_id": "2026-W28", "count": 1420},
                {"_id": "2026-W30", "count": 1100}],
}
app = L._append_missing_multi_row_sections

# The real answer from the failing run: covers type and month, never says "week".
REAL = ("The total number of alerts recorded this quarter is 18,714. The breakdown by "
        "type shows fast inspection alerts accounted for 8682. The busiest month was "
        "July, with 4,983 alerts.")

print("The observed failure")
out = app(REAL, FACET)
check("by_week now mentioned", "week" in out.lower(), out)
check("peak week is the correct one (W28/1420)", "2026-W28" in out and "1,420" in out, out)
check("original prose preserved verbatim", out.startswith(REAL), out)
check("by_type not re-appended", out.lower().count("the busiest was") == 1, out)

print("\nNegatives - a section already covered is left alone")
covered = REAL + " Week 28 was the busiest week of the quarter."
check("no addition when every section is mentioned", app(covered, FACET) == covered,
      app(covered, FACET))
check("month mentioned via 'July' counts (keyword 'month' present)",
      "2026-07" not in app(covered, FACET))

print("\nNegatives - shapes that must not be touched")
check("single-row sections ignored",
      app("Total 5.", {"total_alerts": [{"total": 5}]}) == "Total 5.")
check("empty section ignored", app("Nothing.", {"by_week": []}) == "Nothing.")
check("non-numeric section ignored",
      app("Hi.", {"by_week": [{"_id": "a", "label": "x"}, {"_id": "b", "label": "y"}]}) == "Hi.")
check("no facet sections at all -> unchanged", app("Hi.", {}) == "Hi.")

print("\nPunctuation")
check("missing full stop is added",
      app("Total was 18,714", {"by_week": FACET["by_week"]}).startswith("Total was 18,714."),
      app("Total was 18,714", {"by_week": FACET["by_week"]}))

print("\n_section_is_mentioned directly")
flat = [L._flatten_row(r) for r in FACET["by_week"]]
check("'weekly' counts as a week mention",
      L._section_is_mentioned("A weekly view follows.", "by_week", flat))
check("'daily' counts as a day mention",
      L._section_is_mentioned("Daily counts held steady.", "by_day", flat))
check("unrelated prose does not count",
      not L._section_is_mentioned("Totals were high.", "by_week", flat))

print("\n" + "=" * 60)
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {', '.join(fails)}")
raise SystemExit(1 if fails else 0)
