"""Smoke test over the live approach2 backend (port 8006): every question
must return a 200, a non-empty answer, and no leaked error text. Ported
directly from slm-llama3b's smoke_suite.py — this is a pure end-to-end
sanity check, not tied to Mongo or SQL specifics, so it carries over as-is
apart from the API base and one question ("how many inspections took over
30 seconds?") kept even though the original project's own benchmark docs
flag it as hitting a separate, still-open $div/`$divide`-style bug there —
worth having here too as an early warning if this port ever regresses onto
something similar."""
import json
import os
import urllib.request

API = os.environ.get("APPROACH2_API_BASE", "http://127.0.0.1:8006")
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
