"""Unit sanity checks for FIX W - numeric-string comparison values."""
from chat.backend import llm_query as L

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -> ' + detail) if detail and not cond else ''}")
    if not cond:
        fails.append(name)


print("FIX W - numeric-string comparison coercion")
pipe = [{"$match": {"inspection_time": {"$lt": "5"}, "alert_type": {"$ne": "NORMAL_OPERATION"}}}, {"$count": "total"}]
out = L._coerce_numeric_comparison_strings(pipe)
check("'$lt': '5' coerced to a real int", out[0]["$match"]["inspection_time"]["$lt"] == 5
      and isinstance(out[0]["$match"]["inspection_time"]["$lt"], int), str(out))

for op in ("$lt", "$lte", "$gt", "$gte"):
    r = L._coerce_numeric_comparison_strings({op: "12.5"})
    check(f"{op} coerces a float string", r == {op: 12.5}, str(r))

check("negative number string coerced",
      L._coerce_numeric_comparison_strings({"$gt": "-3"}) == {"$gt": -3})

print("\nFIX W negatives - must not touch dates or non-comparison strings")
date_pipe = {"$gte": "2026-09-17T00:00:00+00:00"}
check("a date string under $gte is left alone (handled separately by _convert_date_strings)",
      L._coerce_numeric_comparison_strings(date_pipe) == date_pipe, str(L._coerce_numeric_comparison_strings(date_pipe)))
check("a real number is already fine, untouched",
      L._coerce_numeric_comparison_strings({"$lt": 5}) == {"$lt": 5})
check("a non-numeric string under $lt is left alone (not silently guessed)",
      L._coerce_numeric_comparison_strings({"$lt": "unknown"}) == {"$lt": "unknown"})
check("$eq with a numeric string untouched (not a range comparison, not in scope)",
      L._coerce_numeric_comparison_strings({"$eq": "5"}) == {"$eq": "5"})
check("alert_type equality string untouched",
      L._coerce_numeric_comparison_strings({"alert_type": "HAND_TOUCH"}) == {"alert_type": "HAND_TOUCH"})

print("\nEnd-to-end through _finalize_pipeline")
f = L._finalize_pipeline(
    {"intent": "data_query",
     "pipeline": [{"$match": {"inspection_time": {"$lt": "5"}, "alert_type": {"$ne": "NORMAL_OPERATION"}}},
                  {"$count": "total"}],
     "explanation": ""},
    "How many inspections took under 5 seconds?")
final_pipe = f.get("pipeline") or []
check("final pipeline has a real numeric $lt",
      any(isinstance(s.get("$match", {}).get("inspection_time", {}), dict)
          and s["$match"]["inspection_time"].get("$lt") == 5
          for s in final_pipe if "$match" in s),
      str(final_pipe))

print("\n" + "=" * 60)
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {', '.join(fails)}")
raise SystemExit(1 if fails else 0)
