"""Large rephrasing battery specifically targeting the 6 bug patterns
just fixed (S/T/U/V/W), to confirm each fix generalizes across many
wordings rather than only working for the exact original failing
question. Oracle-verified against live MongoDB, not against what the
pipeline claims.
"""
import re
import sys
from datetime import datetime, timedelta, timezone

import requests

from bench_config import REPO_ROOT, CHAT_URL
sys.path.insert(0, REPO_ROOT)
from chat.backend.db import get_alerts_collection

API = CHAT_URL
c = get_alerts_collection()
NOW = datetime.now(timezone.utc)
TODAY = NOW.replace(hour=0, minute=0, second=0, microsecond=0)
week_start = TODAY - timedelta(days=TODAY.weekday())
NUM_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")


def ask(q):
    r = requests.post(API, json={"question": q, "history": []}, timeout=180)
    r.raise_for_status()
    return r.json()


def extract_numbers(text):
    out = []
    for t in NUM_RE.findall(text or ""):
        try:
            out.append(float(t.replace(",", "")))
        except ValueError:
            continue
    return out


def contains_number(answer, n, tol=0.5):
    return any(abs(x - n) < tol for x in extract_numbers(answer))


TYPE_PHRASES = {"FAST_INSPECTION": "fast inspection", "HAND_TOUCH": "hand touch", "MISSING_CLEANING": "missing cleaning"}
WINDOWS = [
    ("today", TODAY, TODAY + timedelta(days=1)),
    ("this week", week_start, TODAY + timedelta(days=1)),
    ("in the last 7 days", TODAY - timedelta(days=6), TODAY + timedelta(days=1)),
    ("in the last 14 days", TODAY - timedelta(days=13), TODAY + timedelta(days=1)),
    ("in the last 3 days", TODAY - timedelta(days=2), TODAY + timedelta(days=1)),
    ("this month", TODAY.replace(day=1), TODAY + timedelta(days=1)),
]

groups = []

# --- Group S: NORMAL_OPERATION polarity, many phrasings ---------------
normal_op_count = c.count_documents({"alert_type": "NORMAL_OPERATION"})
groups.append(("FIX S - normal operation polarity", normal_op_count, [
    "How many normal operation events have been logged?",
    "How many normal operation alerts were there?",
    "Count the normal operation events.",
    "What's the total number of normal operation events?",
    "How many compliant events were there?",
    "How many compliant inspections have been recorded?",
    "Give me the count of normal operation events.",
    "Tell me how many normal operation events happened.",
    "What is the number of normal operation events logged?",
    "How many times was the operation normal?",
]))

# --- Group T: imperative "count" phrasing, many alert types/windows ---
for atype, aphrase in TYPE_PHRASES.items():
    wphrase, wgte, wlt = WINDOWS[list(TYPE_PHRASES).index(atype) % len(WINDOWS)]
    expected = c.count_documents({"alert_type": atype, "timestamp": {"$gte": wgte, "$lt": wlt}})
    groups.append((f"FIX T - imperative count, {aphrase} {wphrase}", expected, [
        f"Count the {aphrase} alerts {wphrase}.",
        f"Count {aphrase} alerts {wphrase}.",
        f"Give me a count of {aphrase} alerts {wphrase}.",
        f"Tally the {aphrase} alerts {wphrase}.",
        f"Total up the {aphrase} alerts {wphrase}.",
    ]))

# --- Group U: plain total misrouted into a superlative shape ----------
for atype, aphrase in TYPE_PHRASES.items():
    wphrase, wgte, wlt = WINDOWS[(list(TYPE_PHRASES).index(atype) + 1) % len(WINDOWS)]
    expected = c.count_documents({"alert_type": atype, "timestamp": {"$gte": wgte, "$lt": wlt}})
    groups.append((f"FIX U - plain total, {aphrase} {wphrase}", expected, [
        f"How many {aphrase} alerts happened {wphrase}?",
        f"How many {aphrase} alerts were there {wphrase}?",
        f"What's the total number of {aphrase} alerts {wphrase}?",
        f"How many {aphrase} alerts occurred {wphrase}?",
    ]))
# a couple of the exact bug-triggering windows/types, several more phrasings
_mc_week_expected = c.count_documents(
    {"alert_type": "MISSING_CLEANING", "timestamp": {"$gte": week_start, "$lt": NOW}})
groups.append(("FIX U - the original bug case, more phrasings", _mc_week_expected, [
    "How many missing cleaning alerts happened this week?",
    "How many missing cleaning alerts were recorded this week?",
    "What's the total missing cleaning alert count this week?",
    "How many missing cleaning incidents occurred this week?",
]))

# --- Group V: chained-group average, many types/windows ---------------
for atype, aphrase in TYPE_PHRASES.items():
    for wphrase, wgte, wlt in WINDOWS:
        rows = list(c.aggregate([
            {"$match": {"alert_type": atype, "timestamp": {"$gte": wgte, "$lt": wlt}}},
            {"$group": {"_id": None, "avg": {"$avg": "$inspection_time"}}},
        ]))
        if not rows:
            continue
        expected = round(rows[0]["avg"], 2)
        groups.append((f"FIX V - avg, {aphrase} {wphrase}", expected, [
            f"What is the average inspection time for {aphrase} alerts {wphrase}?",
            f"What's the average inspection time for {aphrase} alerts {wphrase}?",
        ]))

# --- Group W: numeric threshold as a string, many phrasings -----------
for threshold, cmp_word, mongo_op in [
    (5, "under", "$lt"), (5, "less than", "$lt"), (5, "below", "$lt"),
    (20, "over", "$gt"), (20, "more than", "$gt"), (20, "above", "$gt"),
    (10, "at least", "$gte"), (10, "at most", "$lte"),
    (15, "under", "$lt"), (25, "over", "$gt"),
]:
    expected = c.count_documents({"alert_type": {"$ne": "NORMAL_OPERATION"}, "inspection_time": {mongo_op: threshold}})
    groups.append((f"FIX W - threshold {cmp_word} {threshold}s", expected, [
        f"How many inspections took {cmp_word} {threshold} seconds?",
        f"How many alerts had an inspection time {cmp_word} {threshold} seconds?",
        f"Count the alerts where inspection time was {cmp_word} {threshold} seconds.",
    ]))

# ---------------------------------------------------------------------
total = 0
failures = []
for label, expected, phrasings in groups:
    for q in phrasings:
        total += 1
        try:
            d = ask(q)
        except Exception as e:
            failures.append((label, q, f"HTTP error: {e}"))
            print(f"  FAIL  [{label:45s}] {q!r} -> {e}")
            continue
        answer = d.get("answer") or ""
        ok = contains_number(answer, expected)
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  [{label:45s}] {q!r}\n"
              f"          -> expected {expected}, answer={answer[:100]!r}")
        if not ok:
            failures.append((label, q, f"expected {expected}, got: {answer[:150]}"))

print("\n" + "=" * 70)
print(f"{total - len(failures)}/{total} rephrasing-battery prompts passed")
if failures:
    print(f"\n{len(failures)} FAILURES:")
    for label, q, detail in failures:
        print(f"  [{label}] {q!r} -> {detail}")
raise SystemExit(1 if failures else 0)
