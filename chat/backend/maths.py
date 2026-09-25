"""Deterministic post-query maths layer — ported from the Retail deployment's
retail_llm/maths.py.

SQL aggregates (SUM/COUNT/AVG) run inside SQLite's exact arithmetic engine and
are trusted. But a *derived* figure — a percentage, a share of a total, a
growth rate, a per-unit average — is where a small model goes wrong, whether
it tries to hand-write the division in SQL or (worse) just states a number in
its answer prose. This module recognises those question shapes from the raw
component numbers already in the result rows and computes the derived value
itself, in plain Python float arithmetic.

When it fires, its ready-made sentence is used verbatim and the LLM
answer-phrasing call is skipped entirely for that turn.
"""
import re

from . import phrase

_PCT_RE = re.compile(r"\b(percent(age)?|%|share|proportion)\b|\bout of (?:the )?total\b", re.I)
_GROWTH_RE = re.compile(
    r"\b(growth|grew|grown|increase[d]?|decrease[d]?|change|compared?\s+to|vs\.?|versus|"
    r"gone up|gone down|dropped|risen|fallen|higher or lower)\b", re.I)
_AVG_RE = re.compile(r"\baverage\b|\bavg\b|\bper\s+(day|hour|zone|type)\b", re.I)

_COUNTISH_RE = re.compile(r"count|alerts?$|days?$", re.I)
_ALREADY_DERIVED_RE = re.compile(r"avg|average|per_|_pct|ratio", re.I)


def _nums(row):
    return [(k, v) for k, v in row.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)]


def _share(rows, question):
    if not (_PCT_RE.search(question) and len(rows) == 1):
        return None
    nums = _nums(rows[0])
    if len(nums) != 2:
        return None
    (k1, v1), (k2, v2) = nums
    if abs(v1) <= abs(v2):
        part_k, part_v, whole_k, whole_v = k1, v1, k2, v2
    else:
        part_k, part_v, whole_k, whole_v = k2, v2, k1, v1
    if not whole_v:
        return None
    pct = part_v / whole_v * 100
    sentence = (f"{phrase._humankey(part_k)} is {pct:.1f}% of {phrase._humankey(whole_k)} "
                f"({phrase._fmt(part_k, part_v)} of {phrase._fmt(whole_k, whole_v)}).")
    note = (f"share computed as ({part_k} / {whole_k}) * 100 = {pct:.1f}% "
            f"— arithmetic done in code, not asked of the model.")
    return {"sentence": sentence, "note": note}


def _growth(rows, question):
    if not (_GROWTH_RE.search(question) and len(rows) == 2):
        return None
    shared = [k for k in set(rows[0]) & set(rows[1])
              if isinstance(rows[0][k], (int, float)) and isinstance(rows[1][k], (int, float))
              and not isinstance(rows[0][k], bool) and not isinstance(rows[1][k], bool)]
    for k in shared:
        cur, prev = rows[0][k], rows[1][k]
        if not prev:
            continue
        delta = (cur - prev) / prev * 100
        direction = "up" if cur >= prev else "down"
        sentence = (f"{phrase._humankey(k)} is {phrase._fmt(k, cur)}, {direction} "
                    f"{abs(delta):.0f}% from {phrase._fmt(k, prev)}.")
        note = (f"change computed as (({k} row1 - {k} row2) / {k} row2) * 100 = {delta:+.1f}% "
                f"— arithmetic done in code, not asked of the model.")
        return {"sentence": sentence, "note": note}
    return None


def _average(rows, question):
    if not (_AVG_RE.search(question) and len(rows) == 1):
        return None
    nums = _nums(rows[0])
    if len(nums) != 2 or any(_ALREADY_DERIVED_RE.search(k) for k, _ in nums):
        return None
    count_pair = next(((k, v) for k, v in nums if _COUNTISH_RE.search(k)), None)
    if not count_pair:
        return None
    count_k, count_v = count_pair
    sum_k, sum_v = next((k, v) for k, v in nums if k != count_k)
    if not count_v:
        return None
    avg = sum_v / count_v
    unit = phrase._humankey(count_k)
    if unit.endswith("s"):
        unit = unit[:-1]
    sentence = (f"Average {phrase._humankey(sum_k)} per {unit} is {phrase._fmt(sum_k, avg)} "
                f"(from {phrase._fmt(sum_k, sum_v)} over {count_v}).")
    note = (f"average computed as {sum_k} / {count_k} = {avg:.2f} "
            f"— arithmetic done in code, not asked of the model.")
    return {"sentence": sentence, "note": note}


def augment(rows, question):
    """-> (rows, computed). `computed` is None, or a dict with a ready-made
    'sentence' (used verbatim as the answer, skipping the LLM phrasing call)
    and an audit 'note' for the stage log."""
    if not rows or not question:
        return rows, None
    for fn in (_share, _growth, _average):
        computed = fn(rows, question)
        if computed:
            return rows, computed
    return rows, None
