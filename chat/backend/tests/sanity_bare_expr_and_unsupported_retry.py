"""Unit sanity checks for FIX JJ (a bare $expr pipeline stage gets
wrapped in $match instead of being rejected as a disallowed stage) and
FIX II (a plausible real question, greedy-refused as unsupported, gets
one sampled retry before the refusal is accepted). Both found live after
FIX HH removed every fast path, exposing the raw model's own generation
weaknesses on hour-range questions for the first time."""
from chat.backend import llm_query as L

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -> ' + detail) if detail and not cond else ''}")
    if not cond:
        fails.append(name)


print("FIX JJ - a bare $expr stage is wrapped in $match")
pipe = [
    {"$match": {"alert_type": {"$ne": "NORMAL_OPERATION"}}},
    {"$expr": {"$and": [{"$gte": [{"$hour": "$timestamp"}, 14]}, {"$lt": [{"$hour": "$timestamp"}, 16]}]}},
]
out = L._wrap_bare_expr_stage(pipe)
check("bare $expr became {'$match': {'$expr': ...}}",
      out[1] == {"$match": {"$expr": pipe[1]["$expr"]}}, str(out))
check("the real $match stage before it is untouched", out[0] == pipe[0])

print("\nFIX JJ negatives - a real $match with $expr already inside is untouched")
already_wrapped = [{"$match": {"$expr": {"$eq": ["$alert_type", "HAND_TOUCH"]}}}]
check("already-correct $match/$expr shape untouched",
      L._wrap_bare_expr_stage(already_wrapped) == already_wrapped)
multi_key = [{"$expr": {"$eq": [1, 1]}, "$other": "x"}]
check("a dict with $expr AND other keys (not a bare single-key stage) untouched",
      L._wrap_bare_expr_stage(multi_key) == multi_key)

print("\nFIX JJ - end to end through _finalize_pipeline, the exact live-reported shape")
final = L._finalize_pipeline(
    {"intent": "data_query",
     "pipeline": [
         {"$match": {"alert_type": {"$ne": "NORMAL_OPERATION"},
                     "timestamp": {"$gte": "2026-09-24T12:00:00", "$lt": "2026-10-24T00:00:00"}}},
         {"$expr": {"$and": [{"$gte": [{"$hour": "$timestamp"}, 14]},
                              {"$lt": [{"$hour": "$timestamp"}, 16]}]}},
     ], "explanation": "Alerts between 2pm and 4pm in the last 30 days"},
    "How many alerts happened between 2pm and 4pm in the last 30 days?")
check("no longer rejected as unsupported",
      final.get("intent") == "data_query", str(final))
check("no bare $expr stage survives",
      not any(isinstance(s, dict) and list(s.keys()) == ["$expr"] for s in (final.get("pipeline") or [])),
      str(final))

print("\nFIX II - the retry regex matches real questions, not gibberish")
check("'alerts' present -> eligible for a retry",
      bool(L._LIKELY_REAL_QUESTION_RE.search("How many alerts happened between 2pm and 4pm?")))
check("'inspection' present -> eligible for a retry",
      bool(L._LIKELY_REAL_QUESTION_RE.search("what's the average inspection time")))
check("genuinely off-topic text -> NOT eligible (a real refusal isn't retried)",
      not L._LIKELY_REAL_QUESTION_RE.search("what's the weather like today"))

print("\n" + "=" * 60)
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {', '.join(fails)}")
raise SystemExit(1 if fails else 0)
