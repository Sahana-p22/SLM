"""150-200 prompt battery: the same underlying question asked with many
different, none-hardcoded-word-dependent phrasings, oracle-checked against
live MongoDB (not against what the pipeline claims). Built specifically to
catch the class of bug reported live: "how many alerts for 7 days" (no
"last") returning a different, wrong number than "how many alerts for the
last 7 days".

Run from the repo root, with the server already up on :8002:
    python /tmp/phrasing_battery.py
"""
import json
import re
import sys
from datetime import datetime, timedelta, timezone

import requests

from bench_config import REPO_ROOT, CHAT_URL as API
sys.path.insert(0, REPO_ROOT)
from chat.backend.db import get_alerts_collection  # noqa: E402
NO_DATA_RE = re.compile(r"no (matching )?data|no results|nothing (was )?found", re.IGNORECASE)
NUM_RE = re.compile(r"[\d,]+(?:\.\d+)?")


def ask(question):
    r = requests.post(API, json={"question": question, "history": []}, timeout=180)
    r.raise_for_status()
    return r.json()


def answer_contains_number(answer, n):
    n_int_str = f"{int(round(n)):,}"
    n_plain_str = str(int(round(n)))
    for tok in NUM_RE.findall(answer or ""):
        cleaned = tok.replace(",", "")
        try:
            val = float(cleaned)
        except ValueError:
            continue
        if abs(val - n) < 0.5:
            return True
    return n_int_str in (answer or "") or n_plain_str in (answer or "")


def db_count(gte, lt, alert_type=None):
    cond = {"timestamp": {"$gte": gte, "$lt": lt}, "alert_type": {"$ne": "NORMAL_OPERATION"}}
    if alert_type:
        cond["alert_type"] = alert_type
    return get_alerts_collection().count_documents(cond)


# ---------------------------------------------------------------------
# Build the prompt list: groups of phrasings that must all resolve to the
# SAME oracle count, keyed by a (gte, lt) window computed independently
# here (not by calling llm_query's own date-range function, to avoid
# testing the fix against itself).
# ---------------------------------------------------------------------
NOW = datetime.now(timezone.utc)
TODAY = NOW.replace(hour=0, minute=0, second=0, microsecond=0)


def days_window(n):
    return TODAY - timedelta(days=n - 1), TODAY + timedelta(days=1)


def weeks_window(n):
    return TODAY - timedelta(days=n * 7 - 1), TODAY + timedelta(days=1)


def hours_window(n):
    return NOW - timedelta(hours=n), NOW


def minutes_window(n):
    return NOW - timedelta(minutes=n), NOW


groups = []

# --- N days, every phrasing variant -----------------------------------
for n, words in [(3, "three"), (5, "five"), (7, "seven"), (10, "ten"), (14, "fourteen")]:
    gte, lt = days_window(n)
    phrasings = [
        f"how many alerts for {n} days",
        f"how many alerts for the last {n} days",
        f"how many alerts in the past {n} days",
        f"how many alerts over the last {n} days",
        f"how many alerts in the previous {n} days",
        f"how many alerts in the prior {n} days",
        f"how many alerts happened in {n} days",
        f"how many alerts for {words} days",
        f"how many alerts for the last {words} days",
        f"count of alerts in the last {n} days",
        f"total alerts, last {n} days",
        f"alerts over the past {n} days?",
        f"give me the alert count for {n} days",
    ]
    groups.append((f"{n}-day window", gte, lt, phrasings))

# --- N weeks -------------------------------------------------------------
for n in (2, 3, 4):
    gte, lt = weeks_window(n)
    phrasings = [
        f"how many alerts in the last {n} weeks",
        f"how many alerts for {n} weeks",
        f"how many alerts in the past {n} weeks",
        f"how many alerts over {n} weeks",
        f"how many alerts in the previous {n} weeks",
    ]
    groups.append((f"{n}-week window", gte, lt, phrasings))

# --- N hours ---------------------------------------------------------
for n in (2, 3, 6, 12):
    gte, lt = hours_window(n)
    phrasings = [
        f"how many alerts in the last {n} hours",
        f"how many alerts for {n} hours",
        f"how many alerts in the past {n} hours",
        f"how many alerts over the last {n} hours",
    ]
    groups.append((f"{n}-hour window", gte, lt, phrasings))

# --- N minutes ---------------------------------------------------------
for n in (15, 30, 45):
    gte, lt = minutes_window(n)
    phrasings = [
        f"how many alerts in the last {n} minutes",
        f"how many alerts for {n} minutes",
        f"how many alerts in the past {n} minutes",
    ]
    groups.append((f"{n}-minute window", gte, lt, phrasings))

# --- today / yesterday synonyms --------------------------------------
gte, lt = TODAY, TODAY + timedelta(days=1)
groups.append(("today", gte, lt, [
    "how many alerts today",
    "how many alerts happened today",
    "how many alerts current day",
    "how many alerts so far today",
    "alert count for today",
]))
y_gte, y_lt = TODAY - timedelta(days=1), TODAY
groups.append(("yesterday", y_gte, y_lt, [
    "how many alerts yesterday",
    "how many alerts happened yesterday",
    "alert count for yesterday",
]))

# --- this/current week, last/past/previous/prior week -------------------
this_week_start = TODAY - timedelta(days=TODAY.weekday())
groups.append(("this week", this_week_start, TODAY + timedelta(days=1), [
    "how many alerts this week",
    "how many alerts current week",
    "alert count for this week",
]))
last_week_start = this_week_start - timedelta(days=7)
groups.append(("last week", last_week_start, this_week_start, [
    "how many alerts last week",
    "how many alerts past week",
    "how many alerts previous week",
    "how many alerts prior week",
]))

# --- this/current month, last/past/previous/prior month -----------------
this_month_start = TODAY.replace(day=1)
groups.append(("this month", this_month_start, TODAY + timedelta(days=1), [
    "how many alerts this month",
    "how many alerts current month",
    "alert count for this month",
]))
lm_end = this_month_start
lm_start = (lm_end.replace(day=1) - timedelta(days=1)).replace(day=1)
groups.append(("last month", lm_start, lm_end, [
    "how many alerts last month",
    "how many alerts past month",
    "how many alerts previous month",
    "how many alerts prior month",
]))

# --- this/current year, last/past/previous/prior year -------------------
this_year_start = TODAY.replace(month=1, day=1)
groups.append(("this year", this_year_start, this_year_start.replace(year=this_year_start.year + 1), [
    "how many alerts this year",
    "how many alerts current year",
]))
ly_start = this_year_start.replace(year=this_year_start.year - 1)
groups.append(("last year", ly_start, this_year_start, [
    "how many alerts last year",
    "how many alerts past year",
    "how many alerts previous year",
    "how many alerts prior year",
]))

# --- with a type filter layered on top (phrasing must still resolve) ----
for n, words in [(5, "five"), (10, "ten")]:
    gte, lt = days_window(n)
    for atype, phrase_type in [("HAND_TOUCH", "hand touch"), ("FAST_INSPECTION", "fast inspection")]:
        expected = db_count(gte, lt, atype)
        phrasings = [
            f"how many {phrase_type} alerts for {n} days",
            f"how many {phrase_type} alerts in the last {n} days",
            f"how many {phrase_type} alerts for {words} days",
            f"how many {phrase_type} alerts in the previous {n} days",
        ]
        groups.append((f"{n}-day {atype} window", None, None, phrasings, expected))

# --- avg / breakdown phrasing robustness (separate check shape) --------
avg_groups = []
for n, words in [(7, "seven"), (14, "fourteen")]:
    gte, lt = days_window(n)
    rows = list(get_alerts_collection().aggregate([
        {"$match": {"timestamp": {"$gte": gte, "$lt": lt}, "alert_type": {"$ne": "NORMAL_OPERATION"}}},
        {"$group": {"_id": None, "avg": {"$avg": "$inspection_time"}}},
    ]))
    if rows:
        avg_groups.append((round(rows[0]["avg"], 2), [
            f"what's the average inspection time for {n} days",
            f"what's the average inspection time for the last {n} days",
            f"average inspection time for {words} days",
            f"average inspection time over the past {n} days",
        ]))

# ---------------------------------------------------------------------
results = []
failures = []
total = 0

for group in groups:
    if len(group) == 4:
        label, gte, lt, phrasings = group
        # Sub-day windows (hours/minutes) are only ~68 alerts wide over the
        # whole window, so real new data arriving during the several
        # minutes it takes to work through 150+ sequential LLM calls can
        # shift the true count by 1-2 between when this loop started and
        # when a given request actually lands. Recomputed fresh right
        # before EACH request in that case, instead of once for the whole
        # group, so the oracle and the live pipeline are compared against
        # the same moment in time rather than racing a live-growing
        # dataset.
        recompute_live = label.endswith("-hour window") or label.endswith("-minute window")
        expected = db_count(gte, lt) if (gte is not None and not recompute_live) else None
    else:
        label, gte, lt, phrasings, expected = group
        recompute_live = False

    for q in phrasings:
        total += 1
        if recompute_live:
            n = int(label.split("-")[0])
            live_now = datetime.now(timezone.utc)
            window = timedelta(hours=n) if "hour" in label else timedelta(minutes=n)
            gte, lt = live_now - window, live_now
            expected = db_count(gte, lt)
        try:
            d = ask(q)
        except Exception as e:
            failures.append((label, q, f"HTTP error: {e}"))
            print(f"  FAIL  [{label:20s}] {q!r} -> {e}")
            continue
        answer = (d.get("answer") or "")
        ok = False
        detail = ""
        if expected == 0:
            got_zero_or_nodata = NO_DATA_RE.search(answer) or answer.strip().startswith("0")
            ok = bool(got_zero_or_nodata)
            detail = f"expected 0, answer={answer[:80]!r}"
        elif expected is None:
            ok = bool(answer.strip()) and not NO_DATA_RE.search(answer)
            detail = f"answer={answer[:80]!r}"
        else:
            # Sub-day windows: allow the answer to be within 2 of the
            # oracle, since the LLM call itself takes a few real seconds
            # between when this script computes "now" and when the server
            # builds its own pipeline against ITS "now" - a live, still-
            # growing dataset can genuinely gain a row or two in that gap.
            # This is slack for the test harness's own clock skew, not for
            # the pipeline being allowed to be wrong.
            tolerance = 2 if recompute_live else 0
            ok = answer_contains_number(answer, expected) or any(
                answer_contains_number(answer, expected + d)
                for d in range(-tolerance, tolerance + 1)
            )
            detail = f"expected {expected} (+/-{tolerance}), answer={answer[:100]!r}"
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  [{label:20s}] {q!r}  -> {detail}")
        if not ok:
            failures.append((label, q, detail))
        results.append((label, q, ok))

for expected, phrasings in avg_groups:
    for q in phrasings:
        total += 1
        try:
            d = ask(q)
        except Exception as e:
            failures.append(("avg", q, f"HTTP error: {e}"))
            print(f"  FAIL  [avg] {q!r} -> {e}")
            continue
        answer = d.get("answer") or ""
        ok = answer_contains_number(answer, expected) or answer_contains_number(answer, round(expected, 1))
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  [avg                 ] {q!r}  -> expected ~{expected}, answer={answer[:100]!r}")
        if not ok:
            failures.append(("avg", q, answer[:150]))
        results.append(("avg", q, ok))

passed = sum(1 for *_, ok in results if ok)
print("\n" + "=" * 70)
print(f"{passed}/{len(results)} phrasing-battery prompts passed")
if failures:
    print(f"\n{len(failures)} FAILURES:")
    for label, q, detail in failures:
        print(f"  [{label}] {q!r} -> {detail}")

json.dump(
    {"total": len(results), "passed": passed, "failed": len(failures),
     "failures": [{"group": g, "question": q, "detail": d} for g, q, d in failures]},
    open("/tmp/phrasing_battery_report.json", "w"), indent=2, default=str,
)
raise SystemExit(1 if failures else 0)
