"""Large oracle-verified battery specifically for hour-range/time-of-day
phrasing - built after a real user-reported bug showed this whole
category had never been tested at scale the way date-phrase rephrasing
was earlier in this session. Every group's oracle count is computed
directly against the DB using the materialized `hour` field, independent
of what the pipeline itself claims.
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
YESTERDAY = TODAY - timedelta(days=1)
NUM_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")


def ask(q):
    r = requests.post(API, json={"question": q, "history": []}, timeout=180)
    r.raise_for_status()
    return r.json()


def contains_number(answer, n, tol=0.5):
    for t in NUM_RE.findall(answer or ""):
        try:
            if abs(float(t.replace(",", "")) - n) < tol:
                return True
        except ValueError:
            continue
    return False


def oracle(day_start, h1, h2):
    return c.count_documents({
        "timestamp": {"$gte": day_start, "$lt": day_start + timedelta(days=1)},
        "alert_type": {"$ne": "NORMAL_OPERATION"},
        "hour": {"$gte": h1, "$lt": h2},
    })


DAYS = [("today", TODAY), ("yesterday", YESTERDAY)]

HOUR_RANGES_EXPLICIT = [
    (9, 11, "9am", "11am"), (2, 4, "2pm", "4pm"), (14, 16, "2pm", "4pm"),
    (10, 12, "10am", "12pm"), (18, 20, "6pm", "8pm"), (13, 15, "1pm", "3pm"),
]

TIME_OF_DAY = [
    ("morning", 6, 12), ("afternoon", 12, 17), ("evening", 17, 21), ("night", 21, 24),
]

groups = []

# --- explicit hour ranges x every phrasing style x day -----------------
for h1_disp, h2_disp, h1s, h2s in [(2, 4, "2pm", "4pm"), (9, 11, "9am", "11am"),
                                    (1, 3, "1pm", "3pm"), (6, 8, "6pm", "8pm"),
                                    (10, 12, "10am", "12pm"), (3, 5, "3pm", "5pm")]:
    def to24(s):
        h = int(s[:-2]); mer = s[-2:]
        return 0 if (mer == "am" and h == 12) else (h if mer == "am" else (12 if h == 12 else h + 12))
    h1, h2 = to24(h1s), to24(h2s)
    for day_label, day_start in DAYS:
        expected = oracle(day_start, h1, h2)
        groups.append((f"{day_label} {h1s}-{h2s}", expected, [
            f"how many alerts happened {day_label} between {h1s} and {h2s}",
            f"how many alerts happened {day_label} from {h1s} to {h2s}",
            f"how many alerts happened {day_label} {h1s} to {h2s}",
            f"how many alerts happened {day_label} {h1s} and {h2s}",
            f"how many alerts {h1s} to {h2s} {day_label}",
            f"how many alerts between {h1s} and {h2s} {day_label}",
        ]))

# --- qualitative time-of-day words x every phrasing style x day --------
for word, h1, h2 in TIME_OF_DAY:
    for day_label, day_start in DAYS:
        expected = oracle(day_start, h1, h2)
        groups.append((f"{day_label} {word}", expected, [
            f"how many alerts happened {day_label} {word}",
            f"how many alerts {word} {day_label}",
            f"how many alerts happened this {word}" if day_label == "today" else f"how many alerts {day_label} in the {word}",
            f"alerts {day_label} {word}?",
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
            print(f"  FAIL  [{label:20s}] {q!r} -> {e}")
            continue
        answer = d.get("answer") or ""
        if expected == 0:
            ok = "no matching" in answer.lower() or "0" in answer or "zero" in answer.lower()
        else:
            ok = contains_number(answer, expected)
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  [{label:20s}] {q!r}\n          -> expected {expected}, answer={answer[:100]!r}")
        if not ok:
            failures.append((label, q, f"expected {expected}, got: {answer[:150]}"))

print("\n" + "=" * 70)
print(f"{total - len(failures)}/{total} hour-phrasing battery prompts passed")
if failures:
    print(f"\n{len(failures)} FAILURES:")
    for label, q, detail in failures:
        print(f"  [{label}] {q!r} -> {detail}")
raise SystemExit(1 if failures else 0)
