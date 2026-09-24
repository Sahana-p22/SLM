"""Smoke test over the live 8002 backend: every question must return a
200, a non-empty answer, and no leaked error text."""
import json
import urllib.request

from bench_config import API_BASE as API
QUESTIONS = [
    "hi",
    "how many alerts today?",
    "how many hand touch alerts yesterday?",
    "what's the average inspection time?",
    "break down alerts by type this month",
    "which hour is busiest this week?",
    "which day of the week has the most hand touch alerts?",
    "give me a quarterly report",
    "give me a report for June",
    "give me a weekly report",
    "day wise report for this week",
    "compare this week vs last week",
    "how many alerts in the last 15 minutes?",
    "what was the longest inspection time?",
    "how many inspections took over 30 seconds?",
    "asdkjhaskdjh",
]
BAD = ("traceback", "planexecutor", "cannot sort", "$elemmatch", "internal server error",
       "exception", "failed to", "didn't run cleanly")

fails = []
for q in QUESTIONS:
    body = json.dumps({"question": q, "history": []}).encode()
    req = urllib.request.Request(f"{API}/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.loads(r.read())
    except Exception as e:
        print(f"  FAIL  {q!r}  -> {type(e).__name__}: {e}")
        fails.append(q)
        continue

    answer = (d.get("answer") or "").strip()
    leaked = [b for b in BAD if b in answer.lower()]
    if not answer:
        print(f"  FAIL  {q!r}  -> empty answer")
        fails.append(q)
    elif leaked:
        print(f"  FAIL  {q!r}  -> leaked {leaked}: {answer[:120]}")
        fails.append(q)
    else:
        print(f"  PASS  {q!r}\n          {answer[:150].replace(chr(10), ' ')}")

print("\n" + "=" * 60)
print(f"{len(QUESTIONS) - len(fails)}/{len(QUESTIONS)} smoke questions OK"
      + ("" if not fails else f"  FAILED: {fails}"))
raise SystemExit(1 if fails else 0)
