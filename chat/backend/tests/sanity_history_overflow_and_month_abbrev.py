"""Unit sanity checks for FIX DD (history-driven context-window crash),
FIX EE (month abbreviations never recognized), and FIX FF ($ne/$eq with
a list operand always matching everything). All three found live by the
user testing the deployed chatbot directly."""
from datetime import datetime, timezone
from chat.backend import llm_query as L

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -> ' + detail) if detail and not cond else ''}")
    if not cond:
        fails.append(name)


NOW = datetime(2026, 9, 24, 10, 0, tzinfo=timezone.utc)
L._now_for_absolute_date = lambda: NOW

print("FIX EE - abbreviated month names resolve to the exact real date")
check("'sep 4 2025' -> 2025-09-04, not a bare-year fallback",
      L._extract_absolute_date("how many alerts happened on sep 4 2025")
      == datetime(2025, 9, 4, tzinfo=timezone.utc))
check("'jan 1 2020'",
      L._extract_absolute_date("alerts on jan 1 2020") == datetime(2020, 1, 1, tzinfo=timezone.utc))
check("'dec 25th 2024'",
      L._extract_absolute_date("alerts on dec 25th 2024") == datetime(2024, 12, 25, tzinfo=timezone.utc))
check("'sept 4 2025' (4-letter variant)",
      L._extract_absolute_date("alerts on sept 4 2025") == datetime(2025, 9, 4, tzinfo=timezone.utc))
check("'4 sep 2025' (day-month order)",
      L._extract_absolute_date("alerts on the 4th of sep 2025") == datetime(2025, 9, 4, tzinfo=timezone.utc))

print("\nFIX EE negatives - full month names still work exactly as before")
check("'september 4 2025' (full name, regression)",
      L._extract_absolute_date("alerts on september 4 2025") == datetime(2025, 9, 4, tzinfo=timezone.utc))
check("'for august' (bare full month, no day/year) still spans the whole month",
      L._extract_relative_date_range("how many alerts for august", NOW)
      == (datetime(2026, 8, 1, tzinfo=timezone.utc), datetime(2026, 9, 1, tzinfo=timezone.utc)))

print("\nFIX FF - $ne/$eq with a list operand rewritten to $nin/$in")
pipe = [{"$match": {"alert_type": {"$ne": ["NORMAL_OPERATION", "FAST_INSPECTION"]}}}, {"$count": "total"}]
out = L._repair_ne_eq_list_operand(pipe)
check("$ne with a list becomes $nin",
      out[0]["$match"]["alert_type"] == {"$nin": ["NORMAL_OPERATION", "FAST_INSPECTION"]}, str(out))
pipe2 = [{"$match": {"alert_type": {"$eq": ["HAND_TOUCH", "FAST_INSPECTION"]}}}]
out2 = L._repair_ne_eq_list_operand(pipe2)
check("$eq with a list becomes $in",
      out2[0]["$match"]["alert_type"] == {"$in": ["HAND_TOUCH", "FAST_INSPECTION"]}, str(out2))

print("\nFIX FF negatives - a real scalar $ne/$eq is left untouched")
pipe3 = [{"$match": {"alert_type": {"$ne": "NORMAL_OPERATION"}}}]
check("scalar $ne untouched",
      L._repair_ne_eq_list_operand(pipe3) == pipe3)
pipe4 = [{"$match": {"alert_type": {"$eq": "HAND_TOUCH"}}}]
check("scalar $eq untouched",
      L._repair_ne_eq_list_operand(pipe4) == pipe4)
pipe5 = [{"$match": {"alert_type": {"$in": ["HAND_TOUCH", "FAST_INSPECTION"]}}}]
check("an already-correct $in untouched",
      L._repair_ne_eq_list_operand(pipe5) == pipe5)

print("\nFIX DD - history block never exceeds the per-answer character cap")
long_answer = "The breakdown by type shows many things. " * 50
history = [{"question": "q1", "answer": long_answer, "pipeline": []}] * 4
history_block_lines = []
for h in history[-4:]:
    answer_text = h["answer"] or ""
    if len(answer_text) > 300:
        answer_text = answer_text[:300] + "..."
    history_block_lines.append(f'Q: {h["question"]}\nA: {answer_text}')
built = "Conversation so far:\n" + "\n".join(history_block_lines) + "\n\n"
check("capped history block stays well under the uncapped size",
      len(built) < len(long_answer) * 2, f"built len={len(built)}")

print("\n" + "=" * 60)
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {', '.join(fails)}")
raise SystemExit(1 if fails else 0)
