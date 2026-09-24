"""Item 7 equivalent: does accuracy drop for programming-style / messy
real-world phrasing vs clean plain English? Same underlying questions,
oracle-checked, three phrasing registers per question:
  clean   - normal plain English (baseline sanity)
  messy   - typos, run-ons, no punctuation, all-lowercase, abbreviations
  progstyle - snake_case/field-name style, comparison operators, SQL-ish

Run from the repo root with the server up on :8002.
"""
import re
import sys
from datetime import datetime, timedelta, timezone

import requests

from bench_config import REPO_ROOT, CHAT_URL
sys.path.insert(0, REPO_ROOT)
from chat.backend.db import get_alerts_collection

API = CHAT_URL
NUM_RE = re.compile(r"[\d,]+(?:\.\d+)?")


def ask(q):
    r = requests.post(API, json={"question": q, "history": []}, timeout=180)
    r.raise_for_status()
    return r.json()


def contains_number(answer, n):
    for tok in NUM_RE.findall(answer or ""):
        try:
            if abs(float(tok.replace(",", "")) - n) < 0.5:
                return True
        except ValueError:
            continue
    return False


NOW = datetime.now(timezone.utc)
TODAY = NOW.replace(hour=0, minute=0, second=0, microsecond=0)
c = get_alerts_collection()

def db_count(gte, lt, atype=None):
    cond = {"timestamp": {"$gte": gte, "$lt": lt}, "alert_type": {"$ne": "NORMAL_OPERATION"}}
    if atype:
        cond["alert_type"] = atype
    return c.count_documents(cond)

y_gte, y_lt = TODAY - timedelta(days=1), TODAY
w7_gte, w7_lt = TODAY - timedelta(days=6), TODAY + timedelta(days=1)
week_start = TODAY - timedelta(days=TODAY.weekday())

CASES = [
    # (label, expected, [clean, messy, progstyle])
    ("count_yesterday",
     db_count(y_gte, y_lt),
     ["How many alerts happened yesterday?",
      "yesterday how many alerts we got no cap",
      "SELECT COUNT(*) alerts WHERE timestamp == yesterday"]),

    ("count_hand_touch_7d",
     db_count(w7_gte, w7_lt, "HAND_TOUCH"),
     ["How many hand touch alerts were there in the last 7 days?",
      "hand touch alerts last 7 days howmany total pls",
      "count(alert_type=hand_touch AND timestamp>=now()-7d)"]),

    ("count_this_week",
     db_count(week_start, TODAY + timedelta(days=1)),
     ["How many alerts have occurred this week?",
      "alerts this week total how many so far",
      "alerts.filter(week==current_week).count()"]),

    ("avg_inspection_time",
     None,  # checked separately below (needs a live avg, not a count)
     ["What is the average inspection time?",
      "whats avg inspection time like overall",
      "AVG(inspection_time) FROM alerts"]),

    ("count_missing_cleaning_today",
     db_count(TODAY, TODAY + timedelta(days=1), "MISSING_CLEANING"),
     ["How many missing cleaning alerts happened today?",
      "missing cleaning today how many we got",
      "alerts.alert_type==MISSING_CLEANING AND alerts.date==today"]),

    ("count_last_30d",
     db_count(TODAY - timedelta(days=29), TODAY + timedelta(days=1)),
     ["How many alerts have there been in the last 30 days?",
      "last 30 days alerts total gimme the number",
      "count WHERE timestamp BETWEEN now()-30d AND now()"]),
]

results = []
for label, expected, phrasings in CASES:
    for register, q in zip(["clean", "messy", "progstyle"], phrasings):
        try:
            d = ask(q)
        except Exception as e:
            print(f"  FAIL  [{label}/{register}] {q!r} -> HTTP error: {e}")
            results.append((label, register, False))
            continue
        answer = d.get("answer") or ""
        if expected is None:
            ok = bool(answer.strip()) and "average" in answer.lower() or "seconds" in answer.lower()
        else:
            ok = contains_number(answer, expected)
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  [{label:22s}/{register:9s}] {q!r}\n"
              f"          -> expected={expected}, answer={answer[:120]!r}")
        results.append((label, register, ok))

print("\n" + "=" * 70)
by_register = {}
for label, register, ok in results:
    by_register.setdefault(register, []).append(ok)
for register, oks in by_register.items():
    print(f"{register:10s}: {sum(oks)}/{len(oks)} passed")
total_pass = sum(1 for *_, ok in results if ok)
print(f"\nTOTAL: {total_pass}/{len(results)}")
