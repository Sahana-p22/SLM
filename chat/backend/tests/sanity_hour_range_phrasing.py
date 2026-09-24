"""Unit sanity checks for FIX Z - hour-range phrasing without "between"/
"from", and qualitative time-of-day words. Every positive case here is
either the user's own reported bug or a direct rephrasing of it."""
from chat.backend import llm_query as L

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -> ' + detail) if detail and not cond else ''}")
    if not cond:
        fails.append(name)


extract = L._extract_hour_range_from_question

print("The user's exact reported bugs")
check("'yesterday between 1pm and 3pm' (already worked) still works",
      extract("how many alerts happened yesterday between 1pm and 3pm") == (13, 15))
check("'yesterday 1pm to 3pm' (no lead-in word - was broken) now works",
      extract("how many alerts happened yesterday 1pm to 3pm") == (13, 15),
      str(extract("how many alerts happened yesterday 1pm to 3pm")))
check("'yesterday afternoon' (qualitative - was broken) now resolves to a range",
      extract("how many alerts happened yesterday afternoon") == (12, 17),
      str(extract("how many alerts happened yesterday afternoon")))

print("\nBare 'X to Y' / 'X and Y' hour ranges, many phrasings")
cases = [
    ("2pm to 4pm", (14, 16)), ("2pm and 4pm", (14, 16)),
    ("9am to 11am", (9, 11)), ("9am and 11am", (9, 11)),
    ("6pm to 8pm", (18, 20)), ("10am to 12pm", (10, 12)),
    ("1am to 3am", (1, 3)), ("11pm to 1am", (23, 1)),
]
for phrase, expected in cases:
    for template in ["how many alerts happened {p}", "how many alerts {p} today",
                      "how many alerts today {p}", "alerts {p} yesterday"]:
        q = template.format(p=phrase)
        got = extract(q)
        check(f"{q!r}", got == expected, str(got))

print("\nQualitative time-of-day words, many phrasings")
tod_cases = [
    ("morning", (6, 12)), ("afternoon", (12, 17)), ("evening", (17, 21)),
    ("night", (21, 24)), ("early morning", (5, 8)), ("late night", (21, 24)),
]
for word, expected in tod_cases:
    for template in ["how many alerts happened yesterday {w}", "how many alerts this {w}",
                      "how many alerts {w} today", "alerts in the {w}"]:
        q = template.format(w=word)
        got = extract(q)
        check(f"{q!r}", got == expected, str(got))

print("\nNegatives - must not misfire")
check("bare 'between 3 and 5' with no am/pm anywhere -> None (genuinely ambiguous)",
      extract("how many alerts between 3 and 5") is None)
check("'last 3 to 5 days' (not an hour range at all) -> None",
      extract("how many alerts in the last 3 to 5 days") is None)
check("no hour/time-of-day phrase at all -> None",
      extract("how many alerts happened today") is None)
check("a single am/pm mention with no range -> None",
      extract("how many alerts happened at 3pm") is None)

print("\nEnd-to-end through _fix_hour_range_filter")
pipe = [{"$match": {"alert_type": {"$ne": "NORMAL_OPERATION"}, "timestamp": {"$gte": "x", "$lt": "y"}}},
        {"$count": "total"}]
out = L._fix_hour_range_filter("how many alerts happened yesterday 1pm to 3pm", [dict(s) for s in pipe])
check("hour filter injected for the bare-range phrasing",
      any("$expr" in s.get("$match", {}) for s in out if isinstance(s, dict) and "$match" in s), str(out))
out2 = L._fix_hour_range_filter("how many alerts happened yesterday afternoon", [dict(s) for s in pipe])
check("hour filter injected for the qualitative 'afternoon' phrasing",
      any("$expr" in s.get("$match", {}) for s in out2 if isinstance(s, dict) and "$match" in s), str(out2))

print("\n" + "=" * 60)
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {', '.join(fails)}")
raise SystemExit(1 if fails else 0)
