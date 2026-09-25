"""Unit sanity checks for FIX (date-dimension COUNT search wrongly
inheriting the previous turn's narrow date scope). Found live: "how many
alerts on september 19th 2023" (179) followed by "which days did we get
250 alerts" inherited the single Sept-19/20 window from history, so the
search across all days collapsed to that one day and answered with its
own unrelated count (179) again - a stale, silently-repeated answer
rather than an actual search for a day matching 250. Companion to the
existing `_is_date_dimension_ranking` superlative case ("which day had
the most alerts")."""
from chat.backend import llm_query as L

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -> ' + detail) if detail and not cond else ''}")
    if not cond:
        fails.append(name)


print("FIX - count-target date-dimension searches are recognized (inheritance must be suppressed)")
positives = [
    "which days did we get 250 alerts",
    "on which days did we get 250 alerts",
    "which day had 250 alerts",
    "what day saw 250 alerts",
    "which days recorded over 250 alerts",
    "any day that hit 300 alerts",
    "which week had 1000 alerts",
    "which months had at least 500 alerts",
]
for q in positives:
    check(f"detected: {q!r}", L._is_date_dimension_ranking(q))

print("\nNegatives - ordinary questions must NOT be misclassified as date-dimension searches")
negatives = [
    "how many alerts on september 19th 2023",
    "how many alerts did we get last week",
    "give me a quarterly report",
    "how many hand touch alerts on 19th september 2023",
    "what is the average inspection time",
    "how many alerts happened in 2025",
    "break that down by type",
]
for q in negatives:
    check(f"not misclassified: {q!r}", not L._is_date_dimension_ranking(q))

print("\nRegression - existing superlative ranking case still detected")
check("'which day had the most alerts'", L._is_date_dimension_ranking("which day had the most alerts"))
check("'which day had the most hand touch alerts?'",
      L._is_date_dimension_ranking("which day had the most hand touch alerts?"))

print("\nEnd-to-end - inherited range is None (not the previous single-day window) for the count-search follow-up")
history = [{
    "question": "how many alerts on september 19th 2023",
    "answer": "There were 179 alerts.",
    "pipeline": [
        {"$match": {"alert_type": {"$ne": "NORMAL_OPERATION"},
                     "timestamp": {"$gte": "2023-09-19T00:00:00", "$lt": "2023-09-20T00:00:00"}}},
        {"$count": "total"},
    ],
}]
range_, source = L._resolve_date_range_with_source(
    "which days did we get 250 alerts", history, L._now_for_absolute_date()
)
check("no inherited range for the count-search follow-up", range_ is None, f"got {range_!r}")
check("source is not 'inherited'", source != "inherited", f"got {source!r}")

print("\nRegression - a genuine follow-up naming no time period of its own still inherits as before")
range2, source2 = L._resolve_date_range_with_source(
    "break that down by type", history, L._now_for_absolute_date()
)
check("plain follow-up still inherits the previous single-day window", source2 == "inherited", f"got {source2!r}")

print("\n" + "=" * 60)
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {', '.join(fails)}")
raise SystemExit(1 if fails else 0)
