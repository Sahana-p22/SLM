"""1.19 Self-Fixing Test (natural-question methodology): sends genuinely
hard, realistic questions (compound conditions, ambiguous phrasing,
multi-part aggregates) through the real /chat endpoint and checks whether
the "Self-correction attempt" stage appears in the response's own stage
log - i.e. did the first-attempt pipeline actually fail and get repaired,
without deliberately breaking anything. Complements (not replaces) the
direct-injection self_repair_probe.py test from earlier this session,
which proved the mechanism works when it DOES fire; this checks how often
it fires unprompted on real traffic.
"""
import json

import requests

from bench_config import CHAT_URL

API = CHAT_URL

TYPE_PHRASES = ["fast inspection", "hand touch", "missing cleaning"]
WINDOWS = ["today", "yesterday", "this week", "last week", "this month", "last month",
           "this year", "in the last 7 days", "in the last 14 days", "in the last 30 days",
           "in the last 90 days", "in the last 3 days"]

ROUND_1_TEMPLATES = [
    "How many alerts happened between 2pm and 4pm {w}, broken down by type?",
    "What's the average inspection time for {t} alerts that took longer than 10 seconds, {w}?",
    "Compare the number of fast inspection alerts to missing cleaning alerts {w}.",
    "Which zone had the highest average inspection time for {t} alerts over 15 seconds, {w}?",
    "Give me the percentage of alerts that were {t}, out of all alerts {w}.",
    "How many days {w} had more than 50 alerts?",
    "What's the busiest hour of the day for {t} alerts, {w}, excluding weekends?",
    "Break down {t} alerts by week, {w}.",
    "How many {t} alerts happened at FQC Station 1 during business hours (9am-5pm), {w}?",
    "Which alert type has the most consistent inspection time (lowest variance), {w}?",
    "How many {t} alerts happened on weekdays vs weekends, {w}?",
    "Give me the 3 busiest days {w}, with their {t} alert counts.",
    "What percentage of {t} inspections {w} took longer than the average?",
    "How many alerts of each type happened, broken down by week, {w}?",
    "Compare average inspection time for {t} alerts between this month and last month.",
    "What's the ratio of fast inspection to hand touch alerts, {w}?",
    "Which hour had the fewest {t} alerts, {w}?",
    "How many {t} alerts happened at exactly midnight, {w}?",
    "Give me a day-by-day count of {t} alerts {w}, with hourly breakdowns.",
    "What's the highest single-hour count of {t} alerts, {w}?",
    "How many {t} alerts happened outside business hours, {w}?",
    "What fraction of all alerts {w} were {t}, expressed as a percentage?",
    "Give me the average, minimum, and maximum inspection time for {t} alerts, {w}.",
    "How many {t} alerts happened on a Monday, {w}?",
    "Which day {w} had the fewest {t} alerts?",
    "How many alerts total, {w}, split evenly across morning and afternoon for {t}?",
    "What was the second-busiest day for {t} alerts, {w}?",
    "How many {t} alerts happened in the first half of the day, {w}?",
    "Compare {t} alert counts between weekdays and the weekend, {w}.",
    "What's the total inspection time (summed) for {t} alerts, {w}?",
]
ROUND_1 = []
for i, tmpl in enumerate(ROUND_1_TEMPLATES):
    for t in TYPE_PHRASES:
        w = WINDOWS[(i + TYPE_PHRASES.index(t)) % len(WINDOWS)]
        q = tmpl.format(t=t, w=w) if "{t}" in tmpl or "{w}" in tmpl else tmpl
        ROUND_1.append(q)
# de-dup while preserving order, then cap/extend to ~100
seen = set()
ROUND_1 = [q for q in ROUND_1 if not (q in seen or seen.add(q))]

ROUND_2_TEMPLATES = [
    "What's the rolling 7-day average of {t} alerts, compared to a month ago?",
    "Compute the standard deviation of inspection times for {t} alerts, {w}.",
    "What's the 90th percentile inspection time for {t} alerts, {w}?",
    "Find the longest streak of consecutive days with at least one {t} alert, {w}.",
    "What's the median inspection time for {t} alerts, {w}?",
    "Give me the week-over-week percentage change in {t} alerts for the last 6 weeks.",
    "Which day of the week has the highest variance in {t} alert counts, {w}?",
    "What's the correlation between hour of day and inspection time for {t} alerts, {w}?",
    "Find the 3 longest gaps (in hours) between consecutive {t} alerts, {w}.",
    "What's the interquartile range of inspection times for {t} alerts, {w}?",
    "Give me a z-score for today's {t} alert count relative to the last 30 days.",
    "What's the coefficient of variation in daily {t} alert counts, {w}?",
    "Compute a 3-day moving average of {t} alerts, {w}.",
    "What's the mode of inspection time for {t} alerts, rounded to the nearest second, {w}?",
    "Find any days {w} where {t} alert counts were more than 2 standard deviations from the mean.",
    "What's the cumulative sum of {t} alerts by day, {w}?",
    "Give me the percentile rank of today's average {t} inspection time compared to the last 60 days.",
    "What's the longest streak of days where {t} alerts exceeded the daily average, {w}?",
    "Compute the skewness of the inspection-time distribution for {t} alerts, {w}.",
    "What's the autocorrelation of daily {t} alert counts at a 7-day lag, {w}?",
    "Give me a percentile breakdown (10th, 50th, 90th) of inspection times for {t} alerts, {w}.",
    "Find the day with the largest single-day percentage increase in {t} alerts, {w}.",
    "What's the entropy of the alert-type distribution, {w}?",
    "Compute a weighted average inspection time for {t} alerts, weighting by recency, {w}.",
    "What's the Gini coefficient of {t} alert counts across zones, {w}?",
    "Give me the day-over-day change of the 7-day rolling {t} alert count, {w}.",
    "What fraction of {t} alerts {w} occurred in the top 10% busiest hours?",
    "Find the smallest time window (in hours) containing at least 20 {t} alerts, {w}.",
    "What's the harmonic mean of inspection times for {t} alerts, {w}?",
    "Give me a boxplot summary (min, Q1, median, Q3, max) of inspection times for {t} alerts, {w}.",
    "What percentage of variance in daily {t} alert counts is explained by day-of-week alone, {w}?",
    "Compute the exponentially weighted moving average of {t} alerts, {w}, with a 3-day half-life.",
    "Find the pair of consecutive days with the most similar alert-type distributions, {w}.",
    "What's the cross-correlation between {t} and fast inspection alert counts by day, {w}?",
    "Give me a trailing 4-week linear regression slope of {t} alert counts.",
]
ROUND_2 = []
for i, tmpl in enumerate(ROUND_2_TEMPLATES):
    for t in TYPE_PHRASES:
        w = WINDOWS[(i + TYPE_PHRASES.index(t) + 1) % len(WINDOWS)]
        ROUND_2.append(tmpl.format(t=t, w=w))
seen2 = set()
ROUND_2 = [q for q in ROUND_2 if not (q in seen2 or seen2.add(q))]


def run_round(label, questions):
    print(f"\n=== {label} (n={len(questions)}) ===")
    fired = 0
    results = []
    for q in questions:
        r = requests.post(API, json={"question": q, "history": []}, timeout=180)
        d = r.json()
        stages = [s["name"] for s in d.get("stages", [])]
        self_corrected = any("Self-correction" in s for s in stages)
        if self_corrected:
            fired += 1
        results.append({"question": q, "self_corrected": self_corrected,
                         "intent": d.get("intent"), "answer": (d.get("answer") or "")[:100]})
        flag = "FIRED" if self_corrected else "  -  "
        print(f"  [{flag}] {q[:75]}")
    print(f"  -> self-correction fired on {fired}/{len(questions)}")
    return fired, results


fired1, results1 = run_round("Round 1 (20 hard questions)", ROUND_1)
fired2, results2 = run_round("Round 2 (35 harder statistical questions)", ROUND_2)

print(f"\nTOTAL: fired on {fired1 + fired2}/{len(ROUND_1) + len(ROUND_2)} across both rounds")

json.dump({"round1_fired": fired1, "round1_n": len(ROUND_1), "round1_results": results1,
           "round2_fired": fired2, "round2_n": len(ROUND_2), "round2_results": results2},
          open("/tmp/self_fixing_natural_results.json", "w"), indent=2)
