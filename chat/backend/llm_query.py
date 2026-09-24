# backend/llm_query.py
#
# General-purpose "ask anything about the alert log" pipeline. The LLM
# NEVER computes the answer itself — it only ever:
#   1. writes a MongoDB aggregation pipeline (a real query, not a fixed
#      enum of pre-built question types), which this module validates
#      (stage whitelist + deep scan for dangerous operators) and executes
#      against Mongo for real, and
#   2. phrases the final answer using ONLY the exact rows that pipeline
#      returned.
# Moving from a fixed operation schema (count/avg/list/group_count) to
# open-ended pipelines is what makes "which date had the most X alerts" or
# "give me a quarterly report" answerable at all, at the cost of no longer
# being able to fully template every possible answer shape — the tradeoff
# is documented at each point below.

import difflib
import itertools
import json
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone

from chat.backend.config import GGUF_MODEL_PATH
from chat.backend.db import get_alerts_collection

_llm = None
_llm_lock = threading.Lock()


def _load_model():
    """FastAPI can dispatch several requests concurrently (each on its own
    thread), and the very first few requests after a restart can all see
    `_llm is None` before any of them finishes loading — observed live as
    "Loading Qwen2.5-3B-Instruct..." printing three times in a row, each
    call trying to load its own full copy of the model into the same 4GB
    GPU at once, exhausting VRAM and leaving the server unresponsive. The
    cheap unlocked check keeps the fast path (model already loaded) free
    of lock overhead; `_llm_lock` (shared with `_chat`/`_chat_stream`,
    see there for why) only matters for the narrow startup window, and
    the re-check inside it means only the first thread to arrive actually
    loads anything — every other thread just waits and returns once that
    load finishes."""
    global _llm
    if _llm is not None:
        return
    with _llm_lock:
        if _llm is not None:
            return

        print("[llm_query] Loading Llama-3.2-3B-Instruct (GGUF, Q8_0) via llama.cpp...")

        # llama.cpp's CUDA build dynamically links the CUDA runtime DLLs
        # (cudart/cublas) that would normally come from installing the full
        # CUDA Toolkit. On Windows, where that toolkit isn't installed on
        # this machine, PyTorch's own bundled copies of those exact DLLs
        # are used instead by pointing Windows' DLL search path at them —
        # `os.add_dll_directory` is Windows-only (AttributeError on Linux,
        # found deploying to a Linux box) and unnecessary there anyway,
        # since Linux resolves shared libraries via LD_LIBRARY_PATH/rpath,
        # not a per-process DLL search path.
        if hasattr(os, "add_dll_directory"):
            import torch
            torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            os.add_dll_directory(torch_lib)

        from llama_cpp import Llama
        _llm = Llama(
            model_path=GGUF_MODEL_PATH,
            n_gpu_layers=-1,
            n_ctx=4096,
            verbose=False,
        )
        print("[llm_query] Model ready.")


def _chat(system_prompt: str, user_prompt: str, max_new_tokens: int = 500, sample: bool = False) -> str:
    """Switched from transformers+bitsandbytes (4-bit NF4) to llama.cpp
    (GGUF, Q4_K_M) — same base model, but llama.cpp's kernels feed
    quantized weights into the GPU's native low-bit tensor-core path
    directly, instead of unpacking every weight to fp16 before every
    matmul the way bitsandbytes' inference path does. Measured ~5-6x
    faster decode on the same GPU (RTX 3050 Laptop, 4GB) as a result.
    This also replaces the custom KV-cache-prefix-reuse logic from the
    bitsandbytes version: llama.cpp caches the shared prefix between
    consecutive calls on the same `Llama` instance automatically, and
    even a full cold reprocess of the ~2,000-token system prompt measured
    well under 2 seconds here (prefill is parallelized across the whole
    prompt, unlike token-by-token decode), so the extra complexity wasn't
    worth carrying forward.

    Every actual generate() call is serialized through `_llm_lock` —
    llama.cpp's single Llama instance is NOT safe for concurrent
    inference from multiple threads (FastAPI dispatches concurrent
    requests on their own threads by default): two overlapping
    create_chat_completion() calls on the same instance were observed
    live crashing the whole process with a hard C-level assertion
    (GGML_ASSERT tensor-dimension mismatch — the two calls' internal
    compute-graph state collided). The lock means a second request
    queues and waits its turn instead of corrupting shared state; given
    generation already takes single-digit seconds, this doesn't change
    perceived latency much for the normal case of one user at a time.
    """
    _load_model()

    # Greedy decoding is deterministic — if a first attempt produces a
    # syntax mistake, retrying with the identical prompt pattern reproduces
    # the identical mistake. `sample=True` is used only for retry passes,
    # so they actually have a chance to land on a different (correct) token
    # path instead of repeating the same error verbatim.
    gen_kwargs = {"temperature": 0.4, "top_p": 0.9, "top_k": 50} if sample else {"temperature": 0.0}

    with _llm_lock:
        out = _llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_new_tokens,
            repeat_penalty=1.1,
            **gen_kwargs,
        )
    usage = out.get("usage", {})
    print(f"[llm_query] prompt tokens: {usage.get('prompt_tokens')}, completion tokens: {usage.get('completion_tokens')}")
    return (out["choices"][0]["message"]["content"] or "").strip()


def _chat_stream(system_prompt: str, user_prompt: str, max_new_tokens: int = 200):
    """Same generation as `_chat`, but yields text chunks as they're
    produced instead of waiting for the full completion — used only for
    the final answer phrasing, so the frontend can render it word-by-word
    like a normal chat UI instead of showing nothing until the whole
    sentence is done. Query-pipeline generation is NOT streamed: it isn't
    shown to the user directly and needs to be fully parsed as JSON before
    it's usable anyway, so streaming it would add complexity for zero
    perceived-latency benefit.

    Holds `_llm_lock` for the entire duration of the stream, not just the
    call that starts it — a generator only actually runs its body between
    `next()` calls, so the lock has to wrap every `yield`, not just the
    `create_chat_completion(...)` call, or a second request could start
    generating while this one is still mid-stream. See `_chat`'s
    docstring for why concurrent calls on one Llama instance aren't safe.
    """
    _load_model()

    with _llm_lock:
        stream = _llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_new_tokens,
            repeat_penalty=1.1,
            temperature=0.0,
            stream=True,
        )
        for chunk in stream:
            content = chunk["choices"][0]["delta"].get("content")
            if content:
                yield content


# =========================================================
# STEP 1: question -> MongoDB aggregation pipeline
# =========================================================

QUERY_SYSTEM_PROMPT = """You are a MongoDB query-writing assistant for a factory safety alert log. \
Output ONLY a single JSON object, no prose, no markdown fences.

Output schema:
{
  "intent": "data_query" | "greeting" | "unsupported",
  "pipeline": [ ...MongoDB aggregation pipeline stages... ],
  "explanation": "one short sentence describing what this computes"
}

Collection: "alerts". Each document has exactly these fields:
- alert_type: string, one of "FAST_INSPECTION", "HAND_TOUCH", "MISSING_CLEANING", "NORMAL_OPERATION"
- zone: string, currently always "FQC Station 1"
- cloth_detected: boolean
- inspection_time: number (seconds)
- timestamp: BSON date
- narration_en: string (English description, free text)
- narration_ta: string (Tamil description, free text)

Rules:
- "intent":"greeting" for hellos/small talk; "intent":"unsupported" for gibberish/off-topic — omit "pipeline" \
or leave it []. "intent":"data_query" for any real question, however phrased — no fixed list of shapes.
- Every pipeline's first $match MUST exclude NORMAL_OPERATION (alert_type: {"$ne": "NORMAL_OPERATION"}) unless \
the question explicitly asks about normal/compliant operation.
- Allowed stages ONLY: $match, $group, $sort, $project, $limit, $count, $bucket, $bucketAuto, $facet, $unwind, \
$addFields, $set, $sortByCount, $skip, $replaceRoot, $replaceWith. Never $out, $merge, $lookup, $graphLookup, \
$function, $accumulator, $where.
- Always end with a $limit/$count/$group that caps the result to a small number of rows. "Top N" -> $sort + \
$limit: N.
- Date filters: $match "timestamp" with real ISO $gte/$lt strings computed from the given current date. If the \
question names NO time period at all, add NO timestamp filter — answer over all history; don't invent a recent \
window.
- "Which date/day had the most/least X": group by calendar date ($dateToString "%Y-%m-%d"), not weekday, unless \
"day of the week" is explicitly asked.
- Day-of-week grouping: always $dayOfWeek (1-7, Sunday=1). Never $dateToString "%w" (0-6, Sunday=0 — a \
different, incompatible convention).
- Calendar-week grouping/filtering: $dateToString "%Y-%U". There is NO $weekOfYear operator in MongoDB.
- In $group, accumulators ($sum/$avg/$count) are SIBLINGS of "_id", never nested inside it. WRONG: \
{"_id": {"hour": {"$hour": "$timestamp"}, "count": {"$sum": 1}}} (count never aggregates). RIGHT: \
{"_id": {"hour": {"$hour": "$timestamp"}}, "count": {"$sum": 1}} — same pattern as day-of-week grouping: \
wrap the extracted field in a NAMED sub-key ("hour", "dayOfWeek", etc.) inside "_id", with the accumulator \
as _id's sibling, never bare {"_id": {"$hour": "$timestamp"}} (produces an unlabeled result the answer \
step can't describe).
- Ranking questions ("peak"/"busiest"/"most"/"highest"/"least"/"lowest"/"fewest" + a single top/bottom \
result): the $sort stage MUST sort by the accumulator field (e.g. "count"), never by "_id" — sorting by \
"_id" returns whichever group happens to sort first/last by its own key, not the one with the most/least \
records. {"$sort": {"count": -1}} for most/highest/busiest/peak, {"$sort": {"count": 1}} for \
least/lowest/fewest.
- Only use the multi-section $facet REPORT shape when the question explicitly asks for a "report"/"summary"/ \
"day wise"/"weekly breakdown"/"monthly breakdown" of everything. A plain "how many X happened <period>" is \
just $match + $count, nothing more, no matter how the time period is phrased. "Break down alerts by type" (or \
any "break down by X" naming exactly ONE dimension) is NOT a report either — it's a single plain $group by \
that one dimension. Do not wrap a simple count or a one-dimension breakdown in $facet or add sections nobody \
asked for — $facet is ONLY for an explicit multi-section report request.
- Full REPORT questions: always the $facet shape in the worked example below (only the date range changes). \
"Quarterly" = trailing 3 calendar months ending on the current date.
- Unrecognized alert type in the question: pick the closest of the 4 real types, or omit the filter — never \
invent a new type.
- Use given conversation history/previous pipeline only for a follow-up that needs it; ignore it otherwise.
- Output MUST be strict JSON: every key double-quoted, no trailing commas, no comments, no Python \
None/True/False (use null/true/false). JSON has no math operators — never write `19 * 3600`; compute and write \
the literal number.
- Hour-OF-DAY filters ("between 2am and 4am") use $expr + $hour, e.g. {"$expr": {"$and": [{"$gte": \
[{"$hour": "$timestamp"}, 2]}, {"$lt": [{"$hour": "$timestamp"}, 4]}]}} — never epoch-second math.
- "Last hour"/"last N hours" is a plain relative time WINDOW (like "last 7 days", just in hours) — only a \
$gte/$lt range, NEVER an $hour $expr filter. The word "hour" in the question is not a signal to filter by \
hour-of-day.
- "Compare X vs Y" (two time windows or filters): one $facet with a named sub-pipeline per side, each with its \
own COMPLETE, literal $match date range. CRITICAL: no "timestamp" condition may exist in any $match BEFORE the \
$facet — a document excluded there is gone for every branch, so a branch's own date range can never recover it. \
Never use $replaceRoot to just rename $group's own output fields (e.g. to "hourPeak") — leave "_id"/"count"/etc \
as-is.

Worked examples (pipeline field only, current date assumed 2026-07-28):

Q: "how many hand touch alerts in the last 7 days"
pipeline: [
  {"$match": {"alert_type": "HAND_TOUCH", "timestamp": {"$gte": "2026-07-22T00:00:00", "$lt": "2026-07-29T00:00:00"}}},
  {"$count": "total"}
]

Q: "which date had the most missing cleaning alerts"
pipeline: [
  {"$match": {"alert_type": "MISSING_CLEANING"}},
  {"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}}, "count": {"$sum": 1}}},
  {"$sort": {"count": -1}},
  {"$limit": 1}
]

Q: "how many alerts happened between 2am and 4am in the last 30 days"
pipeline: [
  {"$match": {"alert_type": {"$ne": "NORMAL_OPERATION"}, "timestamp": {"$gte": "2026-06-28T00:00:00", "$lt": "2026-07-29T00:00:00"}}},
  {"$match": {"$expr": {"$and": [{"$gte": [{"$hour": "$timestamp"}, 2]}, {"$lt": [{"$hour": "$timestamp"}, 4]}]}}},
  {"$count": "total"}
]

Q: "compare fast inspection and hand touch counts for this week vs last week"
pipeline: [
  {"$facet": {
    "this_week": [
      {"$match": {"alert_type": {"$in": ["FAST_INSPECTION", "HAND_TOUCH"]}, "timestamp": {"$gte": "2026-07-27T00:00:00", "$lt": "2026-07-29T00:00:00"}}},
      {"$group": {"_id": "$alert_type", "count": {"$sum": 1}}}
    ],
    "last_week": [
      {"$match": {"alert_type": {"$in": ["FAST_INSPECTION", "HAND_TOUCH"]}, "timestamp": {"$gte": "2026-07-20T00:00:00", "$lt": "2026-07-27T00:00:00"}}},
      {"$group": {"_id": "$alert_type", "count": {"$sum": 1}}}
    ]
  }}
]

Q: "give me a quarterly report" (or ANY "report"/"day wise"/"weekly"/"monthly breakdown" style question — \
always this exact same shape, only the $gte/$lt date range changes to match what was actually asked for)
pipeline: [
  {"$match": {"alert_type": {"$ne": "NORMAL_OPERATION"}, "timestamp": {"$gte": "2026-04-28T00:00:00", "$lt": "2026-07-29T00:00:00"}}},
  {"$facet": {
    "total_alerts": [{"$count": "total"}],
    "by_type": [{"$group": {"_id": "$alert_type", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}],
    "by_month": [{"$group": {"_id": {"$dateToString": {"format": "%Y-%m", "date": "$timestamp"}}, "count": {"$sum": 1}}}, {"$sort": {"_id": 1}}],
    "by_day": [{"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}}, "count": {"$sum": 1}}}, {"$sort": {"_id": 1}}],
    "avg_inspection_time": [{"$group": {"_id": None, "avg": {"$avg": "$inspection_time"}}}]
  }}
]
Every report ALWAYS has exactly these 5 sections in this order: total_alerts, by_type, by_month, by_day \
(calendar date only, no type split — by_type covers that), avg_inspection_time. The app rolls by_day into \
weekly totals automatically — never compute week buckets yourself.
"""


ALLOWED_STAGES = {
    "$match", "$group", "$sort", "$project", "$limit", "$count", "$bucket",
    "$bucketAuto", "$facet", "$unwind", "$addFields", "$set", "$sortByCount",
    "$skip", "$replaceRoot", "$replaceWith",
}

FORBIDDEN_OPERATORS = {
    "$out", "$merge", "$lookup", "$graphLookup", "$function", "$accumulator",
    "$where", "$currentOp", "$collStats", "$indexStats", "$listSessions",
    "$listLocalSessions", "$planCacheStats", "$redact", "$geoNear", "$search",
}

MAX_PIPELINE_STAGES = 10
DEFAULT_RESULT_LIMIT = 200

DATE_STRING_RE = re.compile(
    # The model doesn't always write the "T" separator ISO strictly requires
    # — it sometimes writes a plain space ("2026-06-29 00:00:00+00:00"),
    # which is what a follow-up question then copies verbatim from history.
    # `datetime.fromisoformat` (3.11+) parses either form fine, but this
    # regex used to only recognize "T", so the space form was never even
    # attempted and silently stayed a raw string — comparing a string
    # against a BSON Date field matches nothing, which is why a follow-up
    # question with no date phrasing of its own (nothing for the
    # deterministic relative-date fallback to override) could silently
    # return zero rows despite real data existing in that range.
    r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?)?$"
)


def _scan_for_forbidden(node) -> str | None:
    """Recursively walks the pipeline looking for any forbidden operator
    key at any depth (not just top-level stage names) — a $function or
    $where could otherwise be smuggled inside a $group accumulator."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k in FORBIDDEN_OPERATORS:
                return k
            found = _scan_for_forbidden(v)
            if found:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _scan_for_forbidden(item)
            if found:
                return found
    return None


def _convert_date_strings(node):
    """Walks the pipeline converting any ISO date/datetime string leaf
    into a real datetime so pymongo serializes it as a BSON Date — Mongo
    won't match a string against a Date field otherwise."""
    if isinstance(node, dict):
        return {k: _convert_date_strings(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_convert_date_strings(v) for v in node]
    if isinstance(node, str) and DATE_STRING_RE.match(node):
        # A follow-up question's pipeline is shown the previous turn's
        # pipeline as history context, and that previous pipeline's dates
        # come back from THIS function already converted to real datetimes
        # — which the API then serializes with a timezone offset
        # ("2026-07-28T00:00:00+00:00"). The model tends to copy that
        # exact string verbatim, so this has to round-trip that format
        # too, not just the plain "date"/"date T time" shapes it
        # originally generates on a first turn.
        s = node[:-1] + "+00:00" if node.endswith("Z") else node
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            try:
                dt = datetime.strptime(s, "%Y-%m-%d")
            except ValueError:
                # The model occasionally emits an invalid clock value (an
                # "hour" of 25, etc.) when doing its own date arithmetic.
                # This used to crash the whole request with a raw
                # ValueError ("unconverted data remains: ..."); leaving
                # the string unconverted instead means the $match just
                # won't match that malformed value against the Date
                # field (empty result), not a 500 to the user.
                print(f"[llm_query] could not parse date-like string {node!r}, leaving as-is")
                return node
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    return node


_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    # Standard abbreviations - found live: "sep 4 2025" matched NONE of
    # the full-name-only patterns below, fell through to a bare-year
    # fallback elsewhere in the file, and silently expanded a one-day
    # question into a whole-YEAR range (60,499 instead of one day's real
    # count). Every regex built from _MONTH_NAME_ALT picks these up
    # automatically - no other pattern needs touching.
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
# Longest names first, so a short abbreviation can't shadow a longer
# name that starts the same way at the alternation's first-match point
# (regex alternation backtracks to a later boundary-passing option
# regardless, but ordering longest-first keeps the common case fast and
# matches this file's existing convention for word alternations).
_MONTH_NAME_ALT = "|".join(sorted(_MONTH_NAMES, key=len, reverse=True))
_ABS_DATE_DAY_MONTH_YEAR_RE = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s*(?:of\s+)?({_MONTH_NAME_ALT}),?\s*(\d{{4}})\b"
)
_ABS_DATE_MONTH_DAY_YEAR_RE = re.compile(
    rf"\b({_MONTH_NAME_ALT})\s*(\d{{1,2}})(?:st|nd|rd|th)?,?\s*(\d{{4}})\b"
)
_ABS_DATE_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_ABS_DATE_MONTH_YEAR_RE = re.compile(rf"\b({_MONTH_NAME_ALT})\s*(?:of\s+)?,?\s*(\d{{4}})\b")
# Day + month with NO year ("september 24th", "24th of september") -
# tried only after the year-bearing patterns above have already failed,
# so this never fires when a year IS present. See FIX CC docstring on
# _extract_absolute_date for why this needs its own patterns rather than
# falling through to the bare-month handler.
_ABS_DATE_DAY_MONTH_NO_YEAR_RE = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s*(?:of\s+)?({_MONTH_NAME_ALT})\b"
)
_ABS_DATE_MONTH_DAY_NO_YEAR_RE = re.compile(
    rf"\b({_MONTH_NAME_ALT})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b"
)


def _now_for_absolute_date() -> datetime:
    return datetime.now(timezone.utc)


def _extract_absolute_date(q: str) -> datetime | None:
    """Parses a single, explicit calendar date named in the question
    ("22nd july 2026", "july 22, 2026", "2026-07-22") into a timezone-
    aware midnight datetime. Found via testing that this was the ONE
    date-phrase category `_extract_relative_date_range` had no
    deterministic case for at all — every relative phrase (yesterday,
    last week, ...) was already overridden, but an absolute date was left
    entirely to the model's own arithmetic, which produced a 2-day
    window (2026-07-21 to 2026-07-23, an off-by-one on the lower bound)
    for "22nd july 2026" instead of just that one day."""
    m = _ABS_DATE_DAY_MONTH_YEAR_RE.search(q)
    if m:
        day, month_name, year = int(m.group(1)), m.group(2), int(m.group(3))
        month = _MONTH_NAMES[month_name]
    else:
        m = _ABS_DATE_MONTH_DAY_YEAR_RE.search(q)
        if m:
            month_name, day, year = m.group(1), int(m.group(2)), int(m.group(3))
            month = _MONTH_NAMES[month_name]
        else:
            m = _ABS_DATE_ISO_RE.search(q)
            if m:
                year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
            else:
                m = _ABS_DATE_DAY_MONTH_NO_YEAR_RE.search(q)
                if m:
                    day, month_name = int(m.group(1)), m.group(2)
                    month = _MONTH_NAMES[month_name]
                else:
                    m = _ABS_DATE_MONTH_DAY_NO_YEAR_RE.search(q)
                    if not m:
                        return None
                    month_name, day = m.group(1), int(m.group(2))
                    month = _MONTH_NAMES[month_name]
                # No year given - use the most recent real occurrence of
                # this date: this year, unless that would land in the
                # future, in which case last year. Same convention the
                # bare-month-no-year handler (just below this function)
                # already uses for a month alone.
                try:
                    candidate = datetime(_now_for_absolute_date().year, month, day, tzinfo=timezone.utc)
                except ValueError:
                    return None
                year = candidate.year if candidate <= _now_for_absolute_date() else candidate.year - 1

    try:
        return datetime(year, month, day, tzinfo=timezone.utc)
    except ValueError:
        return None


# Synonyms the model and real users both reach for interchangeably with
# "last" and "this" - found live: "previous week", "prior month" and
# "current quarter" all fell straight through this function with no
# deterministic override, exactly like the bare "N days" case below.
_LAST_SYNONYM = r"(?:last|past|previous|prior)"
_THIS_SYNONYM = r"(?:this|current)"

# Spelled-out quantities ("last three days", "past twenty minutes") are at
# least as common in natural phrasing as digits, and the regexes below
# used to only recognize \d+. Longest-first so "twenty one" doesn't get
# cut short by "twenty" matching alone.
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30,
}
_ONES = [("one", 1), ("two", 2), ("three", 3), ("four", 4), ("five", 5),
         ("six", 6), ("seven", 7), ("eight", 8), ("nine", 9)]
for _tens_word, _tens_val in (("twenty", 20), ("thirty", 30)):
    for _ones_word, _ones_val in _ONES:
        _NUMBER_WORDS[f"{_tens_word}-{_ones_word}"] = _tens_val + _ones_val
        _NUMBER_WORDS[f"{_tens_word} {_ones_word}"] = _tens_val + _ones_val
_NUMBER_WORD_ALT = "|".join(
    re.escape(w) for w in sorted(_NUMBER_WORDS, key=len, reverse=True)
)
_REL_NUMBER_PATTERN = rf"(?:\d+|{_NUMBER_WORD_ALT})"


def _parse_rel_number(text: str) -> int:
    text = text.strip().lower()
    return int(text) if text.isdigit() else _NUMBER_WORDS[text]


# datetime only covers year 1-9999. An unbounded quantity ("last
# 999999999 days") fed straight into timedelta()/replace(year=...)
# overflows that range and raises OverflowError/ValueError, which
# previously propagated as an unhandled 500 instead of an answer. These
# ceilings are generous (100+ years back in every unit) while staying
# safely inside datetime's valid range from any "now" this app will ever
# run with.
_REL_NUMBER_CEILINGS = {
    "minute": 10_000_000, "min": 10_000_000,
    "hour": 500_000, "hr": 500_000,
    "day": 36_500, "week": 5_000, "month": 1_200, "year": 100,
}


def _clamp_rel_number(n: int, unit: str) -> int:
    ceiling = _REL_NUMBER_CEILINGS.get(unit)
    return min(n, ceiling) if ceiling is not None else n


# One combined pattern for every "<quantity> <unit>" phrase, with the
# last/past/previous/prior prefix OPTIONAL - a bare "7 days" means exactly
# what "last 7 days" means. This is what makes "how many alerts for 7
# days" (no "last") resolve the same way as "how many alerts for the last
# 7 days" instead of silently returning no deterministic override at all.
_REL_UNIT_RE = re.compile(
    rf"\b(?:{_LAST_SYNONYM}\s+)?({_REL_NUMBER_PATTERN})\s+"
    r"(minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)


def _extract_relative_date_range(question: str, now: datetime) -> tuple[datetime, datetime] | None:
    """Regex fallback for common relative-date phrasing. A 3B model under
    greedy decoding doesn't reliably get date arithmetic right inside an
    otherwise-correct pipeline, so this deterministically overrides the
    first $match stage's "timestamp" filter when a clear phrase is found —
    same rationale as before, just re-applied to the new pipeline shape.

    Sub-day windows (hours/minutes) are checked against the raw current
    instant, not midnight — "last hour" needs precision, unlike "today".
    A missing case here previously let a malformed model-generated
    timestamp (e.g. hour=25) through uncaught, which crashed the request.
    """
    q = question.lower()

    abs_date = _extract_absolute_date(q)
    if abs_date:
        return abs_date, abs_date + timedelta(days=1)

    # A named month WITHOUT a day ("report for the month of july 2026",
    # "report for july 2026") has no day number for `_extract_absolute_date`
    # to find, so it fell all the way through to the newly-added bare-year
    # case below — which only knows how to build a FULL CALENDAR YEAR
    # range, not a single month. Found live: "give me a report for the
    # month of july 2026" silently became a report for all of 2026 instead
    # of July alone (total 38,657 instead of July's actual 5,031) the
    # moment the bare-year fallback was added — this has to be checked
    # first so the more specific "month + year" phrasing wins.
    month_year_match = _ABS_DATE_MONTH_YEAR_RE.search(q)
    if month_year_match:
        month_name, year = month_year_match.group(1), int(month_year_match.group(2))
        month = _MONTH_NAMES[month_name]
        start = datetime(year, month, 1, tzinfo=timezone.utc)
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) if month == 12 else datetime(year, month + 1, 1, tzinfo=timezone.utc)
        return start, end

    # A bare month name with NO year ("give me a report for June") had no
    # handler: _extract_absolute_date needs a day number and the month+year
    # branch above needs a year, so such a question ended up with no date
    # range at all. Found live: "give me a report for June" reported the
    # ALL-TIME total (307,755 alerts) as if it were June's, and, with no
    # timestamp range left in the pipeline, _report_scope also fell through
    # to its "quarter" default and built the wrong section layout. Same
    # failure class the month+year branch above exists for, one step less
    # specific.
    #
    # A preposition is required ("for/in/during/of June") rather than
    # matching a bare month name anywhere in the text, because "may" is far
    # more common as an English modal verb ("may I", "that may be") than as
    # a month, and would otherwise hijack unrelated questions.
    bare_month_match = re.search(
        rf"\b(?:for|in|during|of)\s+(?:the\s+month\s+of\s+)?({_MONTH_NAME_ALT})\b", q
    )
    if bare_month_match:
        month = _MONTH_NAMES[bare_month_match.group(1)]
        # Most recent occurrence of that month: this year if it has already
        # started, otherwise last year.
        year = now.year if month <= now.month else now.year - 1
        start = datetime(year, month, 1, tzinfo=timezone.utc)
        end = (datetime(year + 1, 1, 1, tzinfo=timezone.utc) if month == 12
               else datetime(year, month + 1, 1, tzinfo=timezone.utc))
        return start, end

    # A quantity phrase always wins over the bare calendar-unit phrases
    # below ("7 days" is more specific than "week"), and is checked before
    # them for the same reason the month+year branch above is checked
    # before the bare-month branch. See _REL_UNIT_RE for why this no
    # longer requires "last"/"past" to be present at all.
    # Programming/query-style relative-time syntax - now()-7d, NOW()-30d,
    # week==current_week, date==today - found live dropping the date
    # filter entirely (fell through to an all-time total) despite the
    # REST of a prog-style question, like "alert_type=hand_touch", being
    # understood just fine. A person copy-pasting from their own code or
    # a ticket into the chat box is a real, not hypothetical, phrasing
    # register for this audience.
    now_fn_match = re.search(r"\bnow\(\)\s*-\s*(\d+)\s*([dhm])\b", q)
    if now_fn_match:
        n, unit = int(now_fn_match.group(1)), now_fn_match.group(2)
        if unit == "d":
            today_ = now.replace(hour=0, minute=0, second=0, microsecond=0)
            return today_ - timedelta(days=n - 1), today_ + timedelta(days=1)
        if unit == "h":
            return now - timedelta(hours=n), now
        if unit == "m":
            return now - timedelta(minutes=n), now

    if re.search(r"==\s*current_week\b|\bcurrent_week\b\s*==", q):
        today_ = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start = today_ - timedelta(days=today_.weekday())
        return start, today_ + timedelta(days=1)
    if re.search(r"==\s*current_month\b|\bcurrent_month\b\s*==", q):
        today_ = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return today_.replace(day=1), today_ + timedelta(days=1)
    if re.search(r"==\s*today\(\)|==\s*today\b|\btoday\(\)\s*==", q):
        today_ = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return today_, today_ + timedelta(days=1)

    unit_match = _REL_UNIT_RE.search(q)
    if unit_match:
        n = _parse_rel_number(unit_match.group(1))
        unit = unit_match.group(2).lower().rstrip("s")
        n = _clamp_rel_number(n, unit)
        if unit in ("min", "minute"):
            return now - timedelta(minutes=n), now
        if unit in ("hr", "hour"):
            return now - timedelta(hours=n), now
        today_ = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if unit == "day":
            start = today_ - timedelta(days=n - 1)
            return start, today_ + timedelta(days=1)
        if unit == "week":
            start = today_ - timedelta(days=n * 7 - 1)
            return start, today_ + timedelta(days=1)
        if unit == "month":
            # Calendar-month aligned (start of the month N months back),
            # matching the bare "last month" phrase's own convention
            # below rather than a rolling N*30-day window - "last 3
            # months" was previously unrecognized entirely (no handler
            # existed for a counted month phrase, only the bare "last
            # month" singular), so there was no existing precedent to
            # break here.
            total_months = today_.year * 12 + (today_.month - 1) - n
            y, m = divmod(total_months, 12)
            start = today_.replace(year=y, month=m + 1, day=1)
            return start, today_ + timedelta(days=1)
        if unit == "year":
            # A single trailing N-year window ending today - distinct from
            # _try_multi_year_group's "last N years" handling, which only
            # fires alongside an explicit per-year grouping cue ("per
            # year", "each year", "trend", ...) and needs several date
            # buckets rather than one range. When there is no such cue,
            # execution falls through to here, and a plain total over the
            # trailing N years is exactly what was asked.
            start = today_.replace(year=today_.year - n)
            return start, today_ + timedelta(days=1)

    if re.search(rf"\b{_LAST_SYNONYM}\s+hour\b", q):
        return now - timedelta(hours=1), now

    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    if "yesterday" in q:
        d = today - timedelta(days=1)
        return d, today

    # Never handled at all before this fix - fell all the way through to
    # no date filter, silently returning the ALL-TIME total mislabeled
    # "Tomorrow's total". A real one-day future window is the honest
    # answer here (correctly comes back empty, since no future data
    # exists), not a crash or a refusal - same principle as far-future
    # years elsewhere in this file.
    if "tomorrow" in q:
        d = today + timedelta(days=1)
        return d, d + timedelta(days=1)

    # "this afternoon"/"this evening"/"this morning"/"this night" name a
    # time-of-day word alongside "this", which used to only recognize
    # literal "this day" - found live: the hour range from these got
    # applied correctly (see _extract_hour_range_from_question), but with
    # no date constraint at all, answering across the ENTIRE dataset
    # filtered by hour-of-day alone instead of just today.
    if (re.search(r"\btoday\b", q) or re.search(rf"\b{_THIS_SYNONYM}\s+day\b", q)
            or re.search(rf"\b{_THIS_SYNONYM}\s+(?:morning|afternoon|evening|night|noon|midday)\b", q)):
        return today, today + timedelta(days=1)

    if re.search(rf"\b{_THIS_SYNONYM}\s+week\b", q):
        start = today - timedelta(days=today.weekday())
        return start, today + timedelta(days=1)

    if re.search(rf"\b{_LAST_SYNONYM}\s+week\b", q):
        this_week_start = today - timedelta(days=today.weekday())
        start = this_week_start - timedelta(days=7)
        return start, this_week_start

    if re.search(rf"\b{_LAST_SYNONYM}\s+month\b", q):
        # No deterministic fallback existed for this phrase at all, so the
        # model's own date math stood uncorrected — it pattern-matched
        # "give me a report" style questions onto the quarterly-report
        # worked example and returned a trailing-3-month range instead of
        # a single calendar month.
        first_of_this_month = today.replace(day=1)
        if first_of_this_month.month == 1:
            start = first_of_this_month.replace(year=first_of_this_month.year - 1, month=12)
        else:
            start = first_of_this_month.replace(month=first_of_this_month.month - 1)
        return start, first_of_this_month

    if re.search(rf"\b{_THIS_SYNONYM}\s+month\b", q):
        start = today.replace(day=1)
        return start, today + timedelta(days=1)

    # "last quarter" and "this/bare quarter" used to be indistinguishable
    # - both fell into the single "quarter" substring check below and
    # always returned the CURRENT trailing-90-day quarter, so "give me
    # last quarter's report" silently reported this quarter's numbers
    # instead. Checked as its own branch, before the generic one, so the
    # more specific phrasing wins.
    if re.search(rf"\b{_LAST_SYNONYM}\s+quarter\b", q):
        this_q_start = (today.replace(day=1) - timedelta(days=89)).replace(day=1)
        last_q_reference = this_q_start - timedelta(days=1)  # last day of prev quarter
        start = (last_q_reference.replace(day=1) - timedelta(days=89)).replace(day=1)
        return start, this_q_start

    if "quarter" in q:
        start = (today.replace(day=1) - timedelta(days=89)).replace(day=1)
        return start, today + timedelta(days=1)

    # Calendar-year phrasing ("this year", "last year", a bare "2023") had
    # no deterministic case at all — unlike every other relative phrase
    # above, these fell straight through to the model's own date math with
    # no safety-net override. Found live: "how many alerts happened in
    # 2023?" silently dropped the year filter entirely and returned the
    # all-time total across all years instead, and "last year" was off by
    # ~1,000 alerts from the model's own arithmetic. Checked last, after
    # every more-specific phrase above, so a bare year mentioned alongside
    # "quarter"/"this month"/etc. doesn't override the more specific intent.
    if re.search(rf"\b{_THIS_SYNONYM}\s+year\b", q):
        start = today.replace(month=1, day=1)
        return start, start.replace(year=start.year + 1)

    if re.search(rf"\b{_LAST_SYNONYM}\s+year\b", q):
        start = today.replace(month=1, day=1, year=today.year - 1)
        return start, start.replace(year=start.year + 1)

    # Only overridden when exactly one distinct year is mentioned — a
    # multi-year question ("2021 vs 2025", "each of the last 5 years")
    # needs several ranges at once, which a single top-level $match
    # override can't express, so those are deliberately left to the
    # model's own (separately-tracked) pipeline shape rather than
    # clobbered with a single wrong-looking year filter.
    bare_years = {int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", q)}
    if len(bare_years) == 1:
        year = bare_years.pop()
        start = datetime(year, 1, 1, tzinfo=timezone.utc)
        return start, datetime(year + 1, 1, 1, tzinfo=timezone.utc)

    return None


ACCUMULATOR_OPS = {
    "$sum", "$avg", "$min", "$max", "$first", "$last", "$push", "$addToSet",
    "$stdDevPop", "$stdDevSamp", "$mergeObjects",
}


def _strip_empty_sort_stage(pipeline: list) -> list:
    """The model sometimes tacks an empty {"$sort": {}} onto the end of a \
    pipeline that already collapsed to a single group (e.g. a plain total \
    count) — there's nothing left to meaningfully sort by, but instead of \
    omitting the stage entirely it writes a $sort with no keys, which \
    Mongo rejects outright ("$sort stage must have at least one sort \
    key"). Observed live causing a simple "how many alerts today" \
    question to fail, retry, and reproduce the identical empty $sort \
    again — an empty $sort is always either meaningless or broken, never \
    useful, so it's just dropped rather than trusting a retry to notice \
    its own mistake."""
    return [
        stage for stage in pipeline
        if not (isinstance(stage, dict) and set(stage) == {"$sort"} and not stage["$sort"])
    ]


def _clamp_sort_directions(pipeline: list) -> list:
    """MongoDB's $sort accepts exactly 1 or -1 per field (or a $meta
    expression, unused in this domain) — nothing else. Found live as a
    recurring shape, not tied to one specific prompt: the model
    sometimes writes some other number as the sort direction — e.g. a
    literal -3 bled in from arithmetic embedded in the question's own
    text ("(5-3)"), but also seen with no such trigger present at all —
    which Mongo rejects outright ("$sort key ordering must be 1 ... or
    -1") and which the model reliably reproduces unchanged on retry,
    burning both self-correction attempts on the identical mistake.
    Clamps every $sort field's direction to +-1 by sign (0 defaults to
    -1, the dominant intent in this domain — "top"/"most"/"busiest"
    ranking questions)."""
    for stage in pipeline:
        if isinstance(stage, dict) and isinstance(stage.get("$sort"), dict):
            for k, v in list(stage["$sort"].items()):
                if isinstance(v, (int, float)) and not isinstance(v, bool) and v not in (1, -1):
                    fixed = 1 if v > 0 else -1
                    print(f"[llm_query] clamped invalid $sort direction {v!r} on {k!r} to {fixed}")
                    stage["$sort"][k] = fixed
    return pipeline


def _repair_noop_replace_root(pipeline: list) -> list:
    """The model sometimes tacks a {"$replaceRoot": {"newRoot": "$"}} onto \
    the end of a pipeline, apparently trying to "flatten" a $group's \
    output back to top-level fields — except a $group's output already IS \
    a flat document, and "$" alone isn't a valid field-path expression at \
    all, so this reliably fails at Mongo with no useful information for a \
    retry to act on. It's always redundant when it appears, so it's just \
    dropped rather than relying on the model to notice its own mistake."""
    return [
        stage for stage in pipeline
        if not (
            isinstance(stage, dict)
            and set(stage) == {"$replaceRoot"}
            and isinstance(stage["$replaceRoot"], dict)
            and stage["$replaceRoot"].get("newRoot") == "$"
        )
    ]


def _wrap_bare_expr_stage(pipeline: list) -> list:
    """Rewrites a bare {"$expr": {...}} pipeline STAGE into
    {"$match": {"$expr": {...}}}. See FIX JJ above - $expr is a query
    expression operator, valid only nested inside a real stage (usually
    $match), never a pipeline stage in its own right; MongoDB (and this
    file's own stage whitelist) rejects it outright as an unrecognized
    stage name when the model emits it bare."""
    fixed = []
    changed = False
    for stage in pipeline:
        if isinstance(stage, dict) and list(stage.keys()) == ["$expr"]:
            fixed.append({"$match": stage})
            changed = True
        else:
            fixed.append(stage)
    if changed:
        print("[llm_query] wrapped a bare $expr stage in $match")
    return fixed


def _split_merged_stages(pipeline: list) -> list:
    """Fixes a bracket-counting mistake that produces perfectly valid JSON
    but an invalid pipeline shape: the model forgets to close one stage's
    object and open the next, so two stages end up as sibling keys of a
    single dict — {"$group": {...}, "$sort": {...}} — instead of two
    separate {"$group": {...}} and {"$sort": {...}} array elements. This
    parses fine (no JSON error to catch) but fails `_validate_pipeline`'s
    "exactly one key per stage" check as a flat "malformed stage" with no
    way to tell what was actually meant — splitting it back into its
    intended separate stages, in the same order, recovers a normally
    turns-out-correct pipeline that would otherwise be discarded outright.

    Recurses into each `$facet` branch's own sub-pipeline too — found live
    on a "compare 2021 vs 2025" question: the exact same merged-stage
    mistake ({"$match": {...}, "$count": "..."} as one dict) appeared
    *inside* a $facet branch instead of at the top level, which this
    function's original top-level-only loop never looked inside, so the
    malformed branch reached MongoDB uncorrected and failed outright with
    "a pipeline stage specification object must contain exactly one
    field" — a fixable mistake reported as an unrecoverable error.
    """
    def split(stages):
        new_stages = []
        for stage in stages:
            if isinstance(stage, dict) and len(stage) > 1 and all(k in ALLOWED_STAGES for k in stage):
                for k, v in stage.items():
                    new_stages.append({k: v})
            elif isinstance(stage, dict) and "$facet" in stage and isinstance(stage["$facet"], dict):
                stage["$facet"] = {
                    branch: split(sub_stages) if isinstance(sub_stages, list) else sub_stages
                    for branch, sub_stages in stage["$facet"].items()
                }
                new_stages.append(stage)
            else:
                new_stages.append(stage)
        return new_stages

    return split(pipeline)


def _merge_duplicate_match_equality_stages(pipeline: list) -> list:
    """"How many hand touch and cleaning alerts" asks for the union of
    two alert types, but the model sometimes emits that as two separate
    top-level `$match` stages, each pinning the same field to a
    different single literal value — e.g.
    `{"$match": {"alert_type": "HAND_TOUCH"}}` followed by
    `{"$match": {"alert_type": "MISSING_CLEANING"}}`. Consecutive
    `$match` stages AND together in an aggregation pipeline, and no
    document can have two different values for the same field at once,
    so this always matches zero documents — Mongo doesn't error on it,
    it just silently returns nothing, which surfaces as "No matching
    data found for that" (reads as "there is no such data" rather than
    "the query is wrong"). Found live on "how many hand touch and
    cleaning alerts". Detects consecutive `$match` stages that each
    assert plain equality on the same single field to different literal
    values and merges them into one `$match` using `$in`, which is what
    an "X and Y" category question actually means."""
    out = []
    pending_field = None
    pending_values: list = []

    def _flush():
        nonlocal pending_field, pending_values
        if pending_field is not None:
            value = pending_values[0] if len(pending_values) == 1 else {"$in": pending_values}
            out.append({"$match": {pending_field: value}})
        pending_field, pending_values = None, []

    for stage in pipeline:
        if (isinstance(stage, dict) and len(stage) == 1 and isinstance(stage.get("$match"), dict)
                and len(stage["$match"]) == 1):
            field, value = next(iter(stage["$match"].items()))
            if isinstance(value, (str, int, float, bool)):
                if pending_field == field:
                    if value not in pending_values:
                        pending_values.append(value)
                    continue
                _flush()
                pending_field, pending_values = field, [value]
                continue
        _flush()
        out.append(stage)
    _flush()
    return out


DATE_PART_OPERATOR_BY_KEY = {
    "dayOfWeek": "$dayOfWeek",
    "hour": "$hour",
    "minute": "$minute",
    "second": "$second",
    "year": "$year",
    "day": "$dayOfMonth",
    "dayOfMonth": "$dayOfMonth",
    "dayOfYear": "$dayOfYear",
}


def _repair_bare_date_field_reference(pipeline: list) -> list:
    """Fixes {"_id": {"dayOfWeek": "$timestamp"}} — the model naming the \
    key "dayOfWeek" but forgetting to actually wrap the field in the \
    $dayOfWeek operator, so Mongo groups by the raw Date VALUE (every \
    document its own group, since timestamps are all distinct) instead \
    of an actual day-of-week number. Found live: this both produced a \
    meaningless "day of week" answer AND crashed the whole request \
    outright, since a raw datetime landing in the result is exactly what \
    used to slip past JSON serialization uncaught. Only rewrites a bare \
    "$field" reference under a key whose intended date-part operator can \
    be inferred unambiguously from the key name itself."""
    for stage in pipeline:
        if not isinstance(stage, dict) or "$group" not in stage:
            continue
        group = stage["$group"]
        if not isinstance(group, dict):
            continue
        id_val = group.get("_id")
        # "_id" itself set to a bare "$timestamp" (not nested under a
        # named sub-key at all) is the same mistake one level up — every
        # document gets its own group since exact timestamps are all
        # distinct, and the group key ends up as a raw ISO string with no
        # semantic meaning the phrasing step can describe. There's no key
        # name here to infer intent from, so this defaults to grouping by
        # calendar date (%Y-%m-%d) — the same convention used everywhere
        # else in this file for "which day" style questions.
        if id_val == "$timestamp":
            group["_id"] = {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}}
            continue
        if not isinstance(id_val, dict):
            continue
        for key, val in list(id_val.items()):
            operator = DATE_PART_OPERATOR_BY_KEY.get(key)
            if operator and isinstance(val, str) and val.startswith("$") and not val.startswith("$$"):
                id_val[key] = {operator: val}
    return pipeline


_DOTTED_TIMESTAMP_FIELD_RE = re.compile(r"^\$timestamp\.\w+$")


def _repair_dotted_timestamp_field_reference(node):
    """Fixes {"$dayOfWeek": "$timestamp.dayOfWeek"} (and the same mistake
    under $hour/$month/$year/etc.) — `timestamp` is a plain BSON Date, not
    a subdocument, so appending the operator's own key name onto the
    field path as if it were a nested field always points at nothing and
    silently evaluates to null/missing. Found live on "what's the busiest
    day of the week" grouping by null for every document instead of an
    actual weekday, which is indistinguishable from a query that ran fine
    but answered the wrong question. Walks the whole pipeline structure
    (not just $group._id, since the same mistake can appear inside
    $project/$addFields/$match too) and rewrites any string value shaped
    like "$timestamp.<anything>" back to plain "$timestamp"."""
    if isinstance(node, dict):
        return {k: _repair_dotted_timestamp_field_reference(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_repair_dotted_timestamp_field_reference(v) for v in node]
    if isinstance(node, str) and _DOTTED_TIMESTAMP_FIELD_RE.match(node):
        return "$timestamp"
    return node


def _group_id_has_date_truncation(node) -> bool:
    if isinstance(node, dict):
        if any(op in node for op in ("$dateToString", "$dateTrunc", "$dayOfMonth", "$dayOfYear", "$dayOfWeek")):
            return True
        return any(_group_id_has_date_truncation(v) for v in node.values())
    return False


def _repair_untruncated_date_group_id(pipeline: list, question: str) -> list:
    """Deterministic fix for a hallucination shape the model cannot
    reliably fix on its own from a text-only correction prompt (found
    live: given "this pipeline groups by the raw timestamp instead of a
    truncated calendar date, fix it", the model kept the broken $group
    unchanged and instead bolted on a decorative, useless $addFields
    computing a "calendar_date" AFTER the raw per-millisecond grouping
    had already happened — two retries in a row, same result).

    A $group stage's `_id` set to the bare `$timestamp` field groups by
    an exact millisecond instant — nearly every document has a
    different one, so a "which day had the most X" ranking over it is
    meaningless (each group has count 1, "winner" is arbitrary). Only
    fires when the question actually asks about a calendar date/day
    (not "day of the week", which legitimately wants $dayOfWeek, not
    $dateToString) and only rewrites a $group's own "_id" — a bare
    "$timestamp" reference is completely normal and correct anywhere
    else in a pipeline (e.g. {"$max": "$timestamp"}, $sort, $match), so
    this must never touch those."""
    if not (_EXPLICIT_DATE_RE.search(question) and not _EXPLICIT_WEEKDAY_RE.search(question)):
        return pipeline

    def _rewrite(node):
        if node == "$timestamp":
            return {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}}
        if isinstance(node, dict):
            return {k: _rewrite(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_rewrite(v) for v in node]
        return node

    for stage in pipeline:
        if isinstance(stage, dict) and isinstance(stage.get("$group"), dict):
            group_id = stage["$group"].get("_id")
            if group_id is not None and not _group_id_has_date_truncation(group_id):
                stage["$group"]["_id"] = _rewrite(group_id)
    return pipeline


def _repair_nested_group_accumulators(pipeline: list) -> list:
    """Deterministically fixes the single most recurring mistake this
    model makes in $group stages: nesting an accumulator field (e.g.
    "count": {"$sum": 1}) INSIDE "_id" instead of as its sibling — so it
    never actually aggregates and $sort/$limit downstream just pick an
    arbitrary row. The prompt already spells out the correct form with a
    worked right/wrong example, but that's advisory only; the model still
    reproduces this exact pattern often enough (including on the
    error-correction retry path, which has its own fresh chance to get it
    wrong) that it needs a hard, structural fix here rather than relying
    on the model to have read the instructions carefully. Any key inside
    "_id" whose value is a single-key dict naming a real accumulator
    operator gets hoisted out to be a sibling of "_id", same as the model
    was told to write it in the first place.
    """
    for stage in pipeline:
        if not isinstance(stage, dict) or "$group" not in stage:
            continue
        group = stage["$group"]
        if not isinstance(group, dict):
            continue
        id_val = group.get("_id")
        if not isinstance(id_val, dict):
            continue
        for key in list(id_val.keys()):
            val = id_val[key]
            if isinstance(val, dict) and len(val) == 1 and next(iter(val)) in ACCUMULATOR_OPS:
                del id_val[key]
                group.setdefault(key, val)
    return pipeline


_ASKS_FOR_NORMAL_OPERATION_RE = re.compile(
    r"\bnormal operation\b|\bcompliant\b|"
    r"\boperation\b.{0,15}\bnormal\b|\bnormal\b.{0,15}\boperation\b",
    re.IGNORECASE,
)


def _repair_chained_group_dropped_field(pipeline: list) -> list:
    """Drops a $group stage whose only purpose was computing a scratch
    intermediate value the very next $group then ignores in favor of
    reaching for a raw document field that no longer exists.

    Found live: "average inspection time for hand touch alerts in the
    last 7 days" produced
        {"$group": {"_id": None, "sum": {"$sum": "$inspection_time"}}},
        {"$group": {"_id": None, "avg_inspection_time": {"$avg": "$inspection_time"}}}
    - the FIRST $group collapses every matched document into one row
    holding only "_id" and "sum". By the second stage "$inspection_time"
    isn't a field on anything anymore, so $avg over it is null - answered
    "we don't have any data" for a question with 386 real matching rows.
    The first $group did real work for nothing: it computed a sum the
    second stage never uses. Dropping it lets the second $group run
    directly against the still-intact, already-filtered documents, where
    $inspection_time genuinely exists.

    A specific case of the same category _repair_nested_group_accumulators
    handles (the model treating a $group's own accumulator output as if
    the original field survives downstream) - this is the two-stage form
    of that same mistake rather than the one-stage nested form."""
    i = 0
    while i < len(pipeline) - 1:
        first, second = pipeline[i], pipeline[i + 1]
        if not (isinstance(first, dict) and set(first) == {"$group"}
                and isinstance(second, dict) and "$group" in second):
            i += 1
            continue
        first_group = first["$group"]
        second_group = second["$group"]
        if not (isinstance(first_group, dict) and isinstance(second_group, dict)):
            i += 1
            continue
        first_outputs = set(first_group.keys())  # "_id" plus whatever it names

        def _referenced_fields(node):
            found = set()
            if isinstance(node, str) and node.startswith("$") and not node.startswith("$$"):
                found.add(node[1:].split(".")[0])
            elif isinstance(node, dict):
                for v in node.values():
                    found |= _referenced_fields(v)
            elif isinstance(node, list):
                for v in node:
                    found |= _referenced_fields(v)
            return found

        second_refs = _referenced_fields({k: v for k, v in second_group.items() if k != "_id"})
        # The second $group reaches for a field the first $group's output
        # never carries (and isn't a re-reference to the first's own
        # accumulator names) - the first stage is dead scratch work.
        if second_refs and not second_refs & first_outputs:
            print("[llm_query] dropped a $group stage whose scratch output the very next "
                  "$group ignores in favor of an already-gone raw field")
            del pipeline[i]
            continue  # re-check the new pipeline[i] (the former second stage)
        i += 1
    return pipeline


def _fix_misclassified_plain_total(pipeline: list, question: str) -> list:
    """Collapses a $group-by-subperiod -> $sort -> $limit(small) tail back
    into a single whole-period aggregate, when the question itself never
    asked to rank or break anything down.

    Found live: "How many missing cleaning alerts happened this week?" -
    no "which day", "most", "breakdown" wording anywhere - got answered
    "33 missing cleaning alerts on 2026-09-21" (a single day's count)
    instead of the real weekly total, 87. The model built the exact
    $group-by-date -> $sort -> $limit 1 shape that's CORRECT for "which
    day had the most X", and used it here regardless. Nothing else in the
    safety net catches this: _needs_aggregation only checks that SOME
    aggregation stage exists (this pipeline has one, just grouped on the
    wrong key), and _strip_truncating_breakdown_limit only fires for
    actual breakdown wording. This is a third, previously uncovered
    shape - a superlative pipeline for a non-superlative question."""
    if _BREAKDOWN_RE.search(question) or SUPERLATIVE_RE.search(question):
        return pipeline  # a real breakdown/superlative question - leave the shape alone
    if not re.search(r"\bhow many\b|\bcount(?:\s+the|\s+of)?\b|\btotal\b", question, re.IGNORECASE):
        return pipeline

    for i in range(len(pipeline) - 2):
        group_stage, sort_stage, limit_stage = pipeline[i], pipeline[i + 1], pipeline[i + 2]
        if not (isinstance(group_stage, dict) and set(group_stage) == {"$group"}
                and isinstance(sort_stage, dict) and set(sort_stage) == {"$sort"}
                and isinstance(limit_stage, dict) and set(limit_stage) == {"$limit"}
                and isinstance(limit_stage["$limit"], (int, float)) and limit_stage["$limit"] <= 5):
            continue
        group = group_stage["$group"]
        id_val = group.get("_id")
        # Only a sub-period grouping key (a computed date/hour/week
        # expression, directly as _id or nested one level under a named
        # sub-key) counts - grouping by alert_type is a real breakdown
        # and is left to the breakdown-specific fix.
        _DATE_PART_OPS = {"$dateToString", "$hour", "$dayOfWeek", "$dayOfMonth", "$dayOfYear",
                          "$week", "$isoWeek", "$month", "$year", "$dateTrunc"}

        def _is_date_part_expr(node):
            return isinstance(node, dict) and len(node) == 1 and next(iter(node)) in _DATE_PART_OPS

        groups_by_subperiod = _is_date_part_expr(id_val) or (
            isinstance(id_val, dict) and any(_is_date_part_expr(v) for v in id_val.values())
        )
        if not groups_by_subperiod:
            continue
        accumulators = {k: v for k, v in group.items() if k != "_id"}
        if not accumulators:
            continue
        print("[llm_query] collapsed a group-by-subperiod/sort/limit-1 shape back into a "
              "whole-period total for a plain count/total question")
        pipeline[i:i + 3] = [{"$group": {"_id": None, **accumulators}}]
        break
    return pipeline


_NUMERIC_COMPARISON_OPS = ("$lt", "$lte", "$gt", "$gte")


def _coerce_numeric_comparison_strings(node):
    """Converts a numeric-looking string value inside a $lt/$lte/$gt/$gte
    comparison into a real number.

    Found live: "how many inspections took under 5 seconds?" produced
    {"inspection_time": {"$lt": "5"}} - "5" as a JSON string rather than
    the number 5. MongoDB's comparison operators are BSON type-aware, so
    a numeric field compared against a string never matches ANY document,
    however many real rows would satisfy the same comparison numerically
    - confirmed live: 129,132 real matching rows, reported as "no
    matching data found" with no error, nothing to signal anything had
    gone wrong. Range comparisons in this schema are always against a
    numeric field (inspection_time) or a date (already handled
    separately by _convert_date_strings, which this intentionally
    doesn't touch - a numeric-looking string is never mistaken for a
    date string, and vice versa)."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if (k in _NUMERIC_COMPARISON_OPS and isinstance(v, str)
                    and not DATE_STRING_RE.match(v)):
                try:
                    out[k] = int(v) if v.lstrip("-").isdigit() else float(v)
                    print(f"[llm_query] coerced a numeric-string comparison value ({k}: {v!r}) to a real number")
                    continue
                except ValueError:
                    pass
            out[k] = _coerce_numeric_comparison_strings(v)
        return out
    if isinstance(node, list):
        return [_coerce_numeric_comparison_strings(v) for v in node]
    return node


def _repair_ne_eq_list_operand(node):
    """Rewrites {"$ne": [...]} -> {"$nin": [...]} and {"$eq": [...]} ->
    {"$in": [...]} wherever the operand is a list. See FIX FF above -
    no field in this schema is array-typed, so $ne/$eq against a list is
    never a real, intended comparison; it's always meant to be a
    membership test the model reached for the wrong operator name for."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "$ne" and isinstance(v, list):
                out["$nin"] = _repair_ne_eq_list_operand(v)
                print(f"[llm_query] rewrote $ne with a list operand to $nin: {v!r}")
                continue
            if k == "$eq" and isinstance(v, list):
                out["$in"] = _repair_ne_eq_list_operand(v)
                print(f"[llm_query] rewrote $eq with a list operand to $in: {v!r}")
                continue
            out[k] = _repair_ne_eq_list_operand(v)
        return out
    if isinstance(node, list):
        return [_repair_ne_eq_list_operand(v) for v in node]
    return node


_MANDATORY_REPORT_SECTIONS = ("total_alerts", "by_type", "by_month", "by_day", "avg_inspection_time")


def _ensure_report_facet_sections(pipeline: list) -> list:
    """Injects the standard sub-pipeline for any of the 5 mandatory
    report sections missing from the model's own $facet, using the same
    date-range $match the model's facet already runs under.

    Found live: "give me a day wise report for this month" generated a
    $facet with ONLY a by_day section - total_alerts, by_type, and
    avg_inspection_time were never computed at all. _restructure_report
    can reshape what's there, but it can't recover a section that was
    never generated in the first place - the resulting report silently
    lost 3 of its 5 promised sections with no error, nothing to signal
    anything was missing.

    Only fires when the $facet already looks like a report attempt (at
    least 2 of the 5 known section names present) - a real, unrelated
    multi-branch $facet (a this-week-vs-last-week comparison, say, which
    names its branches "this_week"/"last_week") never matches this and
    is left completely alone."""
    match_cond = None
    facet_at = None
    for i, stage in enumerate(pipeline):
        if isinstance(stage, dict) and "$match" in stage and isinstance(stage["$match"], dict) and "timestamp" in stage["$match"]:
            match_cond = stage["$match"]
        if isinstance(stage, dict) and "$facet" in stage:
            facet_at = i
            break
    if facet_at is None or match_cond is None:
        return pipeline

    facet_spec = pipeline[facet_at]["$facet"]
    if not isinstance(facet_spec, dict):
        return pipeline
    present = set(facet_spec.keys()) & set(_MANDATORY_REPORT_SECTIONS)
    if not present:
        # Zero overlap with the mandatory section names at all - a real
        # unrelated multi-branch facet (a this-week-vs-last-week
        # comparison, say) never uses any of these names, so this is a
        # safe signal on its own; found live that even just "by_day"
        # alone (a single one of the 5) is a real, if severely
        # incomplete, report attempt worth completing, not a coincidence
        # worth requiring two matches to trust.
        return pipeline

    missing = [s for s in _MANDATORY_REPORT_SECTIONS if s not in facet_spec]
    if not missing:
        return pipeline

    standard = {
        "total_alerts": [{"$count": "total"}],
        "by_type": [{"$group": {"_id": "$alert_type", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}],
        "by_month": [{"$group": {"_id": {"$dateToString": {"format": "%Y-%m", "date": "$timestamp"}}, "count": {"$sum": 1}}},
                     {"$sort": {"_id": 1}}],
        "by_day": [{"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}}, "count": {"$sum": 1}}},
                   {"$sort": {"_id": 1}}],
        "avg_inspection_time": [{"$group": {"_id": None, "avg": {"$avg": "$inspection_time"}}}],
    }
    for section in missing:
        facet_spec[section] = standard[section]
        print(f"[llm_query] injected the missing mandatory report section '{section}'")
    return pipeline


def _fix_normal_operation_polarity(pipeline: list, question: str) -> list:
    """Flips an inverted alert_type filter for a question that explicitly
    asks about NORMAL_OPERATION events. The model defaults to excluding
    NORMAL_OPERATION so reliably (it is the right default for almost every
    OTHER question) that it reproduces the exact same {"$ne":
    "NORMAL_OPERATION"} filter even when asked directly FOR those events -
    found live answering "how many normal operation events" with 307,755
    (everything EXCEPT normal operation) instead of the real 322,136.
    _inject_normal_operation_exclusion already knows not to ADD this
    filter for such a question; this is its complement, correcting one
    the model wrote anyway despite that."""
    if not _ASKS_FOR_NORMAL_OPERATION_RE.search(question):
        return pipeline
    if re.search(r"\b(not|excluding|except|other than)\b.{0,20}normal operation", question, re.IGNORECASE):
        return pipeline  # a question that genuinely wants the exclusion - leave it alone
    for stage in pipeline:
        if not (isinstance(stage, dict) and "$match" in stage and isinstance(stage["$match"], dict)):
            continue
        cond = stage["$match"].get("alert_type")
        if isinstance(cond, dict) and cond.get("$ne") == "NORMAL_OPERATION":
            stage["$match"]["alert_type"] = "NORMAL_OPERATION"
            print("[llm_query] flipped an inverted alert_type filter for a question specifically about NORMAL_OPERATION")
    return pipeline


def _inject_normal_operation_exclusion(pipeline: list, question: str) -> list:
    q = question.lower()
    if "normal operation" in q or "compliant" in q or "normal_operation" in q:
        return pipeline
    for stage in pipeline:
        if isinstance(stage, dict) and "$match" in stage and isinstance(stage["$match"], dict):
            stage["$match"].setdefault("alert_type", {"$ne": "NORMAL_OPERATION"})
            return pipeline
    pipeline.insert(0, {"$match": {"alert_type": {"$ne": "NORMAL_OPERATION"}}})
    return pipeline


def _extract_inherited_date_range(history: list | None) -> tuple[datetime, datetime] | None:
    """Found live: "how many alerts happened in 2025" (fast-path, correct
    2025 range) followed by "break that down by type" — the follow-up
    names no time period of its own at all, relying entirely on the
    previous turn's scope, but nothing here previously looked at history
    to recover that scope. `_strip_unwanted_date_filter` saw no time
    phrase in the CURRENT question and stripped whatever date filter the
    model attached (correctly guessing it should carry 2025 forward or
    not), silently falling back to an all-time total across all 5 years
    of data (303,878 instead of the correct 60,499) — a follow-up that
    looked identical in shape to a completely unscoped first question.
    Recovers the previous turn's own timestamp range (if any) from its
    stored pipeline, so a follow-up naturally continues the same time
    scope unless it states its own."""
    if not history:
        return None
    last_pipeline = history[-1].get("pipeline")
    if not isinstance(last_pipeline, list):
        return None
    for stage in last_pipeline:
        if not (isinstance(stage, dict) and "$match" in stage and isinstance(stage["$match"], dict)):
            continue
        ts = stage["$match"].get("timestamp")
        if not (isinstance(ts, dict) and "$gte" in ts and "$lt" in ts):
            continue
        try:
            gte = ts["$gte"] if isinstance(ts["$gte"], datetime) else datetime.fromisoformat(str(ts["$gte"]))
            lt = ts["$lt"] if isinstance(ts["$lt"], datetime) else datetime.fromisoformat(str(ts["$lt"]))
        except ValueError:
            return None
        return gte, lt
    return None


_DATE_RANKING_RE = re.compile(
    r"\b(day|date|week|month)\b(?:\W+\w+){0,6}?\W+\b(most common|most|highest|busiest|peak|top|least|lowest|fewest)\b"
    r"|\b(most common|most|highest|busiest|peak|top|least|lowest|fewest)\b(?:\W+\w+){0,6}?\W+\b(day|date|week|month)\b",
    re.IGNORECASE,
)


def _is_date_dimension_ranking(question: str) -> bool:
    """"Which day/week/month had the most/least X" asks to rank across
    the date dimension itself. Inheriting a narrow date scope from a
    previous turn (see `_extract_inherited_date_range`) collapses that
    ranking down to whatever single day/window the prior turn happened
    to be scoped to, which then trivially "wins" by being the only
    candidate — answering with the prior turn's own day/count instead of
    actually ranking anything. Found live: "how many hand touch alerts
    on 19th september 2023" (68) followed by "Which day had the most
    hand touch alerts?" inherited the single Sept-19 window and answered
    "2023-09-19 ... 68 alerts", silently restating the previous turn
    instead of searching across all days. A current question that names
    its own range explicitly (handled separately by
    `_extract_relative_date_range`, e.g. "which day this month...") is
    unaffected — only the silent history carryover is suppressed here."""
    return bool(_DATE_RANKING_RE.search(question))


def _resolve_date_range_with_source(
    question: str, history: list | None, now: datetime
) -> tuple[tuple[datetime, datetime] | None, str | None]:
    """Same resolution as `_resolve_date_range`, but also reports WHERE
    the range came from — "explicit" (the current question named its
    own time phrase) or "inherited" (carried over silently from the
    previous turn, see `_extract_inherited_date_range`). Found live: a
    follow-up like "show me the number of alerts for each category"
    (no time phrase of its own) after an earlier "how many alerts this
    quarter" silently inherited the quarter scope and answered with
    quarter-only counts, with nothing in the answer distinguishing that
    from an all-time total — the exact same question asked in a fresh
    chat (no history) returns very different, much larger numbers, with
    no visible reason why. `_finalize_pipeline` uses this to remember
    when a range was inherited so the phrased answer can say so."""
    explicit = _extract_relative_date_range(question, now)
    if explicit:
        return explicit, "explicit"
    if _is_date_dimension_ranking(question):
        return None, None
    inherited = _extract_inherited_date_range(history)
    if inherited:
        return inherited, "inherited"
    return None, None


def _resolve_date_range(question: str, history: list | None, now: datetime) -> tuple[datetime, datetime] | None:
    """The current question's own time phrase always wins if it has one;
    a follow-up with no time phrase of its own falls back to whatever
    range the previous turn was scoped to, rather than losing that scope
    entirely (see `_extract_inherited_date_range`) — unless the question
    is itself ranking across the date dimension (see
    `_is_date_dimension_ranking`), where inheriting would be
    self-defeating rather than helpful."""
    return _resolve_date_range_with_source(question, history, now)[0]


def _strip_unwanted_date_filter(pipeline: list, question: str, history: list | None = None) -> list:
    """Found live: "break down all alerts by type" — a question with NO \
    time reference whatsoever — still got a $match filtered to a recent \
    7-day window the question never asked for. A prompt instruction alone \
    ("only add a date filter if asked") didn't reliably override the \
    model's default bias toward a recent-window default, so this backs \
    it with a deterministic check: if the question contains not a single \
    temporal keyword, any timestamp condition in a top-level $match is \
    stripped outright rather than trusted.

    Also found live: "which year had the most alerts overall?" — the \
    word "year" here names the $group dimension, not an actual date \
    range, but the old keyword-only check treated it as one and trusted \
    whatever window the model attached, which turned out to be a bogus \
    2-day "today" window the model defaults to out of habit — silently \
    narrowing an "across all history" question down to 2 days of data. \
    Checking against `_resolve_date_range` instead of a bare keyword \
    match means the filter is only trusted when it corresponds to a \
    phrase this file can independently verify — either in the current \
    question, or (for a follow-up naming no time period of its own) \
    inherited from the previous turn — and `_inject_date_range` will then \
    overwrite it with the correctly-computed range anyway."""
    if _resolve_date_range(question, history, datetime.now(timezone.utc)):
        return pipeline
    for stage in pipeline:
        if isinstance(stage, dict) and "$match" in stage and isinstance(stage["$match"], dict):
            stage["$match"].pop("timestamp", None)
    return [s for s in pipeline if not (isinstance(s, dict) and s.get("$match") == {})]


def _inject_date_range(pipeline: list, question: str, history: list | None = None) -> list:
    date_range = _resolve_date_range(question, history, datetime.now(timezone.utc))
    if not date_range:
        return pipeline
    start, end = date_range

    for stage in pipeline:
        if isinstance(stage, dict) and "$match" in stage and isinstance(stage["$match"], dict):
            if "timestamp" in stage["$match"]:
                stage["$match"]["timestamp"] = {"$gte": start, "$lt": end}
                return pipeline

    for stage in pipeline:
        if isinstance(stage, dict) and "$match" in stage and isinstance(stage["$match"], dict):
            stage["$match"]["timestamp"] = {"$gte": start, "$lt": end}
            return pipeline

    pipeline.insert(0, {"$match": {"timestamp": {"$gte": start, "$lt": end}}})
    return pipeline


def _validate_pipeline(pipeline) -> tuple[bool, str]:
    if not isinstance(pipeline, list) or not pipeline:
        return False, "empty or malformed pipeline"
    if len(pipeline) > MAX_PIPELINE_STAGES:
        return False, "pipeline too long"

    for stage in pipeline:
        if not isinstance(stage, dict) or len(stage) != 1:
            return False, "malformed stage"
        stage_name = next(iter(stage))
        if stage_name not in ALLOWED_STAGES:
            return False, f"disallowed stage {stage_name}"

    forbidden = _scan_for_forbidden(pipeline)
    if forbidden:
        return False, f"forbidden operator {forbidden}"

    return True, ""


def _has_result_cap(pipeline: list) -> bool:
    last_two = {next(iter(s)) for s in pipeline[-2:] if isinstance(s, dict) and len(s) == 1}
    return bool(last_two & {"$limit", "$count", "$facet", "$group"})


UNSUPPORTED_RESULT = {"intent": "unsupported", "pipeline": None, "explanation": None}
GREETING_RESULT = {"intent": "greeting", "pipeline": None, "explanation": None}
GAP_QUESTION_RESULT = {"intent": "gap_unsupported", "pipeline": None, "explanation": None}


def _normalize_python_literals(text: str) -> str:
    """The model occasionally writes Python's None/True/False instead of
    JSON's null/true/false — e.g. {"_id": None} inside a $group stage
    (common when averaging with no grouping key). Word-boundary substitution
    outside of what's realistically ever a string value in this domain
    (Mongo operator names, alert types, dates) is safe enough here."""
    text = re.sub(r"\bNone\b", "null", text)
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    return text


_ARITH_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*([*+/-])\s*(\d+(?:\.\d+)?)(?![\w.])")
_STRING_LITERAL_RE = re.compile(r'"(?:[^"\\]|\\.)*"')


def _normalize_arithmetic(text: str) -> str:
    """JSON has no expression syntax, but this model likes to write \
    time-unit math directly into a pipeline (e.g. `19 * 3600` for "7pm in \
    seconds", or `6 * 86400` for "6 days"), which is a plain JSON syntax \
    error. Evaluating each simple `<number> <op> <number>` literal down to \
    its numeric result — repeated until nothing changes, so chained \
    expressions like `2 * 3600 + 1800` collapse fully — turns those into \
    valid JSON without needing the model to get it right.

    Critically, this must only touch text OUTSIDE quoted string literals —
    an ISO date like "2026-06-29T00:00:00" is full of `\\d+-\\d+` patterns \
    that look exactly like subtraction, and applying this blindly across \
    the whole text corrupted dates into garbage like "2020-29T00:00:0:00" \
    (matching nothing in the DB, which then surfaced as a false "no data \
    found"). Splitting on string literals and only touching the gaps \
    between them keeps every quoted value byte-for-byte untouched."""
    parts = _STRING_LITERAL_RE.split(text)
    literals = _STRING_LITERAL_RE.findall(text)

    def _collapse(segment: str) -> str:
        for _ in range(4):
            new_segment, count = _ARITH_RE.subn(
                lambda m: _eval_arith(m.group(1), m.group(2), m.group(3)), segment
            )
            if count == 0:
                break
            segment = new_segment
        return segment

    parts = [_collapse(p) for p in parts]

    out = []
    for i, part in enumerate(parts):
        out.append(part)
        if i < len(literals):
            out.append(literals[i])
    return "".join(out)


def _eval_arith(a: str, op: str, b: str) -> str:
    a, b = float(a), float(b)
    result = {"*": a * b, "+": a + b, "-": a - b, "/": a / b if b else a}[op]
    return str(int(result)) if result == int(result) else str(result)


def _quote_bare_keys(text: str) -> str:
    """The model sometimes emits unquoted object keys (`{_id: null, ...}`) \
    — valid JS, invalid JSON. Any identifier-looking token right after `{` \
    or `,` that isn't already inside quotes is a key in this grammar (Mongo \
    stage/operator names are always the quoted string values or already- \
    quoted keys), so it's safe to blanket-quote here."""
    return re.sub(r'([{,]\s*)([A-Za-z_$][A-Za-z0-9_]*)(\s*:)', r'\1"\2"\3', text)


def _strip_json_comments_and_trailing_commas(text: str) -> str:
    """Strips `// ...` line comments (another JS-ism the model reaches for \
    when explaining a magic number inline) and trailing commas before a \
    closing bracket, both of which are invalid JSON."""
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return text


_INVALID_EQ_KEY_RE = re.compile(
    r',?\s*"[$A-Za-z_][$A-Za-z0-9_]*"\s*:\s*\{(?:[^{}]|\{[^{}]*\})*==\s*"[^"]*"\s*\}'
)


def _strip_invalid_equality_expr(text: str) -> str:
    """Strips a specific JS-ism: the model double-checking a date it \
    already correctly wrote into $gte/$lt by ALSO adding a redundant, \
    invalid key like {"_id": {"$dateToString": {...}} == "2026-07-22"} — \
    a bare `==` comparison is never valid JSON syntax, and no amount of \
    brace-rebalancing can fix a genuinely wrong token, only a missing or \
    extra bracket. Found live on "how many alerts happened on 22nd July \
    2026" — the date range was already right; this extra clause was pure \
    (broken) redundancy, so it's simply deleted rather than repaired."""
    return _INVALID_EQ_KEY_RE.sub("", text)


_JS_DATE_CTOR_RE = re.compile(r'(?:new\s+Date|ISODate)\(\s*"([^"]*)"\s*\)')


def _strip_js_date_constructors(text: str) -> str:
    """The model sometimes writes a Mongo-shell-style date literal — \
    `new Date("2026-07-28T00:00:00Z")` or `ISODate("...")` — both valid in \
    the mongosh REPL, neither valid JSON. Unwrapping to the bare quoted \
    string is exactly what `_convert_date_strings` already expects."""
    return _JS_DATE_CTOR_RE.sub(r'"\1"', text)


def _try_repair_json(text: str, error: json.JSONDecodeError) -> dict | None:
    """Targets one specific, highly consistent mistake this model makes:
    closing one brace too many after a nested $match filter (e.g.
    {"$match": {"timestamp": {"$gte": ..., "$lt": ...}}}} — one `}` too
    many before the comma). This isn't random noise — it reproduces
    character-for-character across both greedy and sampled retries, so a
    generic bracket-count repair is more reliable here than hoping a retry
    escapes it. Only used if removing exactly one `}` right before the
    reported error position makes the whole thing parse; otherwise no-op.
    """
    before = text[:error.pos].rstrip()
    if not before.endswith("}"):
        return None
    candidate = before[:-1] + text[len(before):]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _try_repair_missing_brace(text: str, error: json.JSONDecodeError) -> dict | None:
    """The mirror-image mistake to `_try_repair_json` above: the model
    closes one brace too FEW, most often in the last accumulator of a
    $facet's last branch (e.g. {"$avg": "$inspection_time"}]} instead of
    {"$avg": "$inspection_time"}}]} — it stops one level short and jumps
    straight to closing the surrounding array. json reports this as
    "Expecting ',' delimiter" right at the unexpected `]`/`}` that follows;
    inserting one `}` at that exact position is enough to fix it when
    that's really the whole problem, and a no-op (fails to parse, so the
    caller moves on) when it isn't."""
    candidate = text[:error.pos] + "}" + text[error.pos:]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _try_repair_missing_brace_before_comma(text: str, error: json.JSONDecodeError) -> dict | None:
    """A different "one brace too few" shape than `_try_repair_missing_brace` \
    handles: a whole PIPELINE STAGE's dict never closes before the comma \
    that starts the next stage — e.g. {"$match": {"$expr": {...}}, \
    {"$group": {...}}, ... — one `}` short right after the nested $expr \
    closes, so the stage-level dict is still open when its sibling's `{` \
    arrives. json reports this as "Expecting property name enclosed in \
    double quotes" at the position of that unexpected `{`, which is \
    AFTER the comma — inserting a brace there (what \
    `_try_repair_missing_brace` tries first) lands in the wrong spot; the \
    brace actually needs to go BEFORE that same comma."""
    if "Expecting property name" not in error.msg:
        return None
    comma_pos = text.rfind(",", 0, error.pos)
    if comma_pos == -1:
        return None
    candidate = text[:comma_pos] + "}" + text[comma_pos:]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _try_repair_by_rebalancing_brackets(text: str) -> dict | None:
    """A catch-all for bracket mistakes the two narrower repairs above
    don't cover — most often the model closing a nested $facet with the
    WRONG bracket type (a stray `}` where the pipeline array actually
    needed `]`), which is neither "one extra brace" nor "one missing
    brace" but a genuine type mismatch. Walks the text tracking a real
    open-bracket stack (skipping over string contents so brackets inside
    quoted values, e.g. a date string, are never touched); whenever a
    closing bracket doesn't match what the stack expects, it's swapped
    for the one that does, and any openers still unclosed at the end get
    their closers appended. This is deliberately the LAST repair
    attempted — it's more invasive than the other two, so it only runs
    once they've both already failed to explain the error."""
    closers = {"{": "}", "[": "]"}
    stack = []
    chars = list(text)
    in_string = False
    escape = False
    for i, ch in enumerate(chars):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                continue
            expected = closers[stack[-1]]
            if ch != expected:
                chars[i] = expected
            stack.pop()
    candidate = "".join(chars) + "".join(closers[c] for c in reversed(stack))
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _generate_pipeline_json(user_prompt: str, force_sample: bool = False) -> dict | None:
    """Generates the query JSON, with one self-correction retry if the
    output isn't valid JSON. Asking a 3B model under greedy decoding to
    emit a correctly-bracketed nested pipeline occasionally produces a
    small syntax slip (a stray brace, a missing comma). Re-prompting with
    the SAME question but sampling enabled gives it a real chance to land
    on a different token path — critically, the retry does NOT show the
    model its own broken output: doing that first made it just anchor on
    the previous text and copy the identical mistake back verbatim, even
    with sampling on, since "repeat what's right there in context" was an
    easier completion than actually re-deriving a correct pipeline.
    """
    for attempt in range(2):
        raw = _chat(QUERY_SYSTEM_PROMPT, user_prompt, max_new_tokens=500,
                    sample=(force_sample or attempt == 1))
        print(f"[llm_query] {'retry' if attempt else 'raw'} pipeline output: {raw!r}")

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            print("[llm_query] no JSON object found in output")
            continue

        candidate = _strip_json_comments_and_trailing_commas(match.group(0))
        candidate = _strip_invalid_equality_expr(candidate)
        candidate = _strip_js_date_constructors(candidate)
        candidate = _normalize_python_literals(candidate)
        candidate = _normalize_arithmetic(candidate)
        candidate = _quote_bare_keys(candidate)

        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            print(f"[llm_query] JSON parse failed: {exc}")
            repaired = _try_repair_json(candidate, exc)
            if repaired is not None:
                print("[llm_query] auto-repaired JSON (removed one redundant brace)")
                return repaired
            repaired = _try_repair_missing_brace(candidate, exc)
            if repaired is not None:
                print("[llm_query] auto-repaired JSON (inserted one missing brace)")
                return repaired
            repaired = _try_repair_missing_brace_before_comma(candidate, exc)
            if repaired is not None:
                print("[llm_query] auto-repaired JSON (inserted missing brace before comma)")
                return repaired
            repaired = _try_repair_by_rebalancing_brackets(candidate)
            if repaired is not None:
                print("[llm_query] auto-repaired JSON (rebalanced mismatched brackets)")
                return repaired
            continue

    return None


_HOUR_RANGE_RE = re.compile(
    # "between"/"from" is now OPTIONAL - found live: "yesterday 1pm to
    # 3pm" (no lead-in word at all) silently dropped its hour filter
    # entirely, while "yesterday between 1pm and 3pm" worked, for what a
    # user reasonably expects to be the identical question. At least one
    # side must still carry an explicit am/pm marker (enforced below in
    # _extract_hour_range_from_question, unchanged) so a bare "between 2
    # and 4" with no meridiem anywhere - genuinely ambiguous - still
    # isn't guessed at.
    r"\b(?:(?:between|from)\s+)?(\d{1,2})\s*(am|pm)?\s+(?:and|to)\s+(\d{1,2})\s*(am|pm)?\b",
    re.IGNORECASE,
)

# Qualitative time-of-day words, mapped to the hour range each
# colloquially means - found live: "how many alerts happened yesterday
# afternoon" silently dropped "afternoon" entirely and answered for the
# whole day, the same root cause (nothing recognized it as a time-of-day
# constraint at all) as the bare-hour-range gap above.
_TIME_OF_DAY_HOURS = {
    "early morning": (5, 8), "morning": (6, 12), "midday": (11, 14), "noon": (11, 14),
    "afternoon": (12, 17), "evening": (17, 21), "night": (21, 24), "late night": (21, 24),
}
_TIME_OF_DAY_RE = re.compile(
    r"\b(early morning|late night|morning|midday|noon|afternoon|evening|night)\b", re.IGNORECASE,
)


def _to_hour24(num: int, meridiem: str | None) -> int:
    if meridiem == "am":
        return 0 if num == 12 else num
    if meridiem == "pm":
        return 12 if num == 12 else num + 12
    return num


def _extract_hour_range_from_question(question: str) -> tuple[int, int] | None:
    """Parses an explicit "between Xam/pm and Yam/pm" hour-of-day range
    directly out of the question text, independent of anything the model
    generated — found via testing that the model's own am/pm-to-24-hour
    conversion is unreliable: the identical question ("between 2pm and
    4pm yesterday") produced a pipeline with the hour filter DROPPED
    entirely in one run, and the wrong hour bounds (16-18 instead of
    14-16) in another, on two separate live runs. Returns None (leave the
    pipeline alone) when the question doesn't name an explicit am/pm
    range, so this never touches "last N hours"-style relative windows or
    bare-number ranges it can't be confident about."""
    m = _HOUR_RANGE_RE.search(question)
    if m:
        h1_s, mer1, h2_s, mer2 = m.groups()
        mer1 = (mer1 or mer2 or "").lower() or None
        mer2 = (mer2 or mer1 or "").lower() or None
        if mer1 is not None or mer2 is not None:
            return _to_hour24(int(h1_s), mer1), _to_hour24(int(h2_s), mer2)
        # both sides bare (no am/pm anywhere) - genuinely ambiguous,
        # fall through to the qualitative-word check rather than guess.

    tod = _TIME_OF_DAY_RE.search(question)
    if tod:
        return _TIME_OF_DAY_HOURS[tod.group(1).lower()]

    return None


def _fix_hour_range_filter(question: str, pipeline: list) -> list:
    """Corrects (or, if entirely missing, injects) the $expr/$hour
    range-filter stage to match the am/pm range actually named in the
    question — see `_extract_hour_range_from_question` for why this can't
    just be trusted from the model's own output."""
    bounds = _extract_hour_range_from_question(question)
    if bounds is None or not isinstance(pipeline, list):
        return pipeline
    start_hour, end_hour = bounds

    for stage in pipeline:
        if not isinstance(stage, dict) or "$match" not in stage:
            continue
        cond = stage["$match"]
        if not isinstance(cond, dict) or "$expr" not in cond:
            continue
        expr = cond["$expr"]
        if not isinstance(expr, dict) or not isinstance(expr.get("$and"), list):
            continue
        clauses = expr["$and"]
        fixed_lower = fixed_upper = False
        for i, clause in enumerate(clauses):
            if not isinstance(clause, dict) or len(clause) != 1:
                continue
            op, operands = next(iter(clause.items()))
            if not isinstance(operands, list) or len(operands) != 2:
                continue
            left, right = operands
            if not (isinstance(left, dict) and "$hour" in left and isinstance(right, (int, float))):
                continue
            # Normalize the OPERATOR too, not just the number — the model
            # sometimes writes $lte for the upper bound (found live: "4pm
            # to 6pm" using {"$lte": [{"$hour": ...}, 18]}, which actually
            # matches hour 18 too, extending a 2-hour window to 3). "Xam/pm
            # to Yam/pm" always means the half-open [X, Y) hour range, so
            # the lower bound is always $gte and the upper is always $lt.
            if op in ("$gte", "$gt"):
                clauses[i] = {"$gte": [left, start_hour]}
                fixed_lower = True
            elif op in ("$lt", "$lte"):
                clauses[i] = {"$lt": [left, end_hour]}
                fixed_upper = True
        if fixed_lower or fixed_upper:
            return pipeline

    # No $hour $expr stage found anywhere — the model dropped the filter
    # entirely (observed live). Inject one right after the trailing plain
    # date-range $match, before any $group/$facet/$sort/$count/$limit.
    insert_at = 0
    for i, stage in enumerate(pipeline):
        if not isinstance(stage, dict):
            continue
        name = next(iter(stage), None)
        if name == "$match" and "$expr" not in stage["$match"]:
            insert_at = i + 1
        elif name in ("$group", "$facet", "$sort", "$count", "$limit"):
            break
    pipeline.insert(insert_at, {"$match": {"$expr": {"$and": [
        {"$gte": [{"$hour": "$timestamp"}, start_hour]},
        {"$lt": [{"$hour": "$timestamp"}, end_hour]},
    ]}}})
    return pipeline


def _use_indexed_hour_filter(pipeline: list) -> list:
    """Rewrites an $expr/$hour range FILTER into an equivalent plain
    {"hour": {"$gte": a, "$lt": b}} match against the materialized `hour`
    field (see seed_data.py's build_alert and the (alert_type, hour)
    index) - functionally identical, but an ordinary field comparison
    Mongo can serve with an index seek instead of a per-document $hour
    computation it can never index.

    Found live: "how many alerts between 2pm and 4pm" (no day named, so
    nothing narrows the alert_type-filtered set down first) examined
    307,755 documents every single time on this 629,891-row collection -
    a number that scales with the TOTAL collection size, not with how
    many alerts actually fall in the 2-hour window, and gets worse
    without bound as the collection grows (confirmed ~200ms here; the
    same shape at 31.5M rows would be an order of magnitude slower).
    ~200ms per fast-path question that's specifically supposed to be the
    cheap, no-LLM-call shortcut defeats the point of it being a fast
    path at all.

    Only touches a $match whose SOLE content is this $expr/$hour
    $and-of-two-clauses shape - a $group's {"$hour": "$timestamp"} (used
    for hour-of-day BREAKDOWNS, not a filter) is a different
    expression position entirely and is left untouched; grouping still
    has to visit every matched row regardless of any index."""
    for i, stage in enumerate(pipeline):
        if not isinstance(stage, dict) or set(stage) != {"$match"}:
            continue
        cond = stage["$match"]
        if not isinstance(cond, dict) or set(cond) != {"$expr"}:
            continue
        expr = cond["$expr"]
        if not isinstance(expr, dict) or not isinstance(expr.get("$and"), list) or len(expr["$and"]) != 2:
            continue
        gte_val = lt_val = None
        for clause in expr["$and"]:
            if not isinstance(clause, dict) or len(clause) != 1:
                gte_val = lt_val = None
                break
            op, operands = next(iter(clause.items()))
            if (op == "$gte" and isinstance(operands, list) and len(operands) == 2
                    and operands[0] == {"$hour": "$timestamp"} and isinstance(operands[1], (int, float))):
                gte_val = operands[1]
            elif (op == "$lt" and isinstance(operands, list) and len(operands) == 2
                    and operands[0] == {"$hour": "$timestamp"} and isinstance(operands[1], (int, float))):
                lt_val = operands[1]
            else:
                gte_val = lt_val = None
                break
        if gte_val is not None and lt_val is not None:
            print(f"[llm_query] rewrote $expr/$hour filter to an indexed "
                  f"hour range [{gte_val}, {lt_val})")
            pipeline[i] = {"$match": {"hour": {"$gte": gte_val, "$lt": lt_val}}}
    return pipeline


def _contains_operator(node, op: str) -> bool:
    """True if `op` appears anywhere as a key in the expression tree."""
    if isinstance(node, dict):
        return any(k == op or _contains_operator(v, op) for k, v in node.items())
    if isinstance(node, list):
        return any(_contains_operator(v, op) for v in node)
    return False


# Operators seen in post-$facet $project stages that rearrange the facet
# sections themselves - never useful here and routinely fatal. Every one of
# these was observed live on report questions; see
# _strip_junk_post_facet_stages for the full account.
_FACET_MANGLING_OPERATORS = (
    "$concatArrays", "$slice", "$concat", "$arrayElemAt",
    "$elemMatch", "$ifNull", "$map", "$reduce", "$filter",
)


def _fix_mixed_inclusion_exclusion_project(pipeline: list) -> list:
    """MongoDB refuses a $project that mixes inclusion (1) and exclusion
    (0) on different fields ("Cannot do exclusion on field X in inclusion
    projection") — every field but _id must agree on one mode. Found
    live on a post-$facet $project meant to keep two sections and drop
    three others: the model wrote the fields it wanted as 1 and the ones
    it didn't as 0 in the same stage, which reads as sensible English
    ("keep these, drop those") but is invalid Mongo. This executes,
    fails, and burns both self-correction retries on the identical
    mistake rather than getting fixed — worth a deterministic repair
    instead of hoping a third model attempt gets it right. An inclusion
    field is the more reliable signal of actual intent (each one names a
    fact the answer needs), so this keeps every "1" field, plus _id if
    it's explicitly kept, and drops the "0" fields outright — which is
    exactly what leaving them out of an inclusion $project already
    means."""
    for stage in pipeline:
        if not (isinstance(stage, dict) and isinstance(stage.get("$project"), dict)):
            continue
        proj = stage["$project"]
        included = {k for k, v in proj.items() if k != "_id" and v in (1, True)}
        excluded = {k for k, v in proj.items() if k != "_id" and v in (0, False)}
        if included and excluded:
            print(f"[llm_query] dropped conflicting exclusion fields {sorted(excluded)} from a mixed $project")
            stage["$project"] = {k: v for k, v in proj.items() if k not in excluded}
    return pipeline


def _strip_junk_post_facet_stages(pipeline: list) -> list:
    """Drop the trailing reshaping stages the model habitually bolts onto a
    report's $facet.

    A $facet already returns exactly one document holding every requested
    section in its final, labelled shape - {"by_type": [{_id, count}, ...],
    "by_month": [...], ...} - which is precisely what _restructure_report
    and _facet_to_sentence consume. Anything appended after it can only
    degrade that, and in practice always did. Observed on report questions,
    every one of these from a real run:

      $project {"$concatArrays": ["$by_type.count"]}
          Strips the _id half of each pair, leaving bare numbers with no
          labels. The answer step then has to guess which type each number
          belongs to - caught inventing "109,656 regular alerts" and
          "68,967 other types of alerts", two categories that do not exist,
          stated as fact.
      $project {"$slice": ["$by_type", 0, -1]}
          MongoDB rejects a negative third argument outright, so the whole
          report died with a PlanExecutor error.
      $project {"$elemMatch": {...}}
          "Cannot use $elemMatch in this context" - same fatal outcome.
      $project {"$ifNull": ["$by_type.count", 0]}
          Produces parallel arrays, which the trailing $sort below then
          cannot sort: "cannot sort with keys that are parallel arrays".
      $replaceRoot {"newRoot": "$by_type"}
          Every facet section is an array and $replaceRoot demands an
          object, so this can only ever fail.
      $sort over facet sub-paths, after a $limit 1
          Sorting a single already-materialised document is a no-op at
          best, and fatal whenever the projection above left parallel
          arrays.

    Each variant failed differently, and the self-correction retry usually
    regenerated the same shape, burning three model calls per report and
    still returning nothing. Dropping them as a family is both simpler and
    more reliable than trying to repair each operator in turn.
    """
    facet_at = next((i for i, s in enumerate(pipeline)
                     if isinstance(s, dict) and "$facet" in s), None)
    if facet_at is None:
        return pipeline

    cleaned = list(pipeline[:facet_at + 1])
    saw_unwind = False
    for stage in pipeline[facet_at + 1:]:
        if not isinstance(stage, dict) or len(stage) != 1:
            cleaned.append(stage)
            continue
        name = next(iter(stage))

        if name == "$project" and any(
            _contains_operator(stage[name], op) for op in _FACET_MANGLING_OPERATORS
        ):
            print("[llm_query] dropped junk post-$facet $project "
                  "(array-mangling operator over facet sections)")
            continue

        # $replaceRoot/$replaceWith onto a facet section: the section is an
        # array, the stage demands an object. Always fatal.
        if name in ("$replaceRoot", "$replaceWith"):
            spec = stage[name]
            new_root = spec.get("newRoot") if isinstance(spec, dict) else spec
            if isinstance(new_root, str) and new_root.startswith("$"):
                print(f"[llm_query] dropped post-$facet {name} onto an array section")
                continue

        # A $sort after the facet can only be sorting that single output
        # document by its own section sub-paths - a no-op at best.
        if name == "$sort" and isinstance(stage[name], dict):
            print("[llm_query] dropped no-op post-$facet $sort")
            continue

        # An $unwind decomposes the facet document into real rows, after
        # which a $group is doing genuine work ("which week had the most
        # alerts this quarter?" legitimately runs
        # $facet -> $unwind -> $unwind -> $addFields -> $group). Track it
        # so the $group rule below only fires on an un-decomposed facet.
        if name == "$unwind":
            saw_unwind = True
            cleaned.append(stage)
            continue

        # Without an intervening $unwind, a $group can only be
        # re-aggregating the single facet output document by its own
        # section names - which destroys it twice over. Observed live on
        # "Give me a quarterly report", where the model appended:
        #
        #     {"$group": {"_id": None,
        #                 "total_alerts": {"$sum": "$total_alerts"},
        #                 "by_type": {"$push": {"count": "$by_type.count"}},
        #                 ...}}
        #
        # $push over "$by_type.count" keeps only the count half of each
        # pair, so by_type came back as [{"count": [8682, 7280, 2752]}] -
        # three bare numbers with no idea which alert type each belongs
        # to, the same label-stripping that had the model inventing
        # category names. Worse, "$sum" over an array of sub-documents is
        # 0 by definition, so total_alerts was reported as 0 for a
        # quarter with 18,714 alerts. It also flattens by_day into the
        # section list, so _restructure_report could no longer recognise
        # the plain date+count shape and bailed out, silently dropping
        # the entire weekly view from the report.
        # Without an $unwind there is exactly ONE document in play, so
        # there is nothing for a $group to group or an $addFields to add
        # that the renderer does not already read straight off the facet
        # sections. Every instance seen live was actively destructive,
        # and each broke differently:
        #
        #   {"$group": {"_id": None,
        #               "by_type": {"$push": {"count": "$by_type.count"}},
        #               "total_alerts": {"$sum": "$total_alerts"}, ...}}
        #       $push over "$by_type.count" keeps only the count half of
        #       each pair, so by_type came back as
        #       [{"count": [8682, 7280, 2752]}] - three bare numbers with
        #       nothing saying which alert type each belongs to, the same
        #       label-stripping that had the model inventing category
        #       names. "$sum" over an array of sub-documents is 0 by
        #       definition, so a quarter with 18,714 alerts reported 0.
        #       It also flattened by_day into the section list, so
        #       _restructure_report stopped recognising the plain
        #       date+count shape and silently dropped the whole weekly
        #       view from the report.
        #   {"$group": {"total_this_week": {"$sum": "$this_week.count"}}}
        #       No _id at all: "a group specification must include an
        #       _id", so the entire question failed.
        #   {"$addFields": {"total_this_week": {"$sum": ["this_week.count"]}}}
        #       "The $sum accumulator is a unary operator" - fatal too.
        if name in ("$group", "$addFields", "$set") and not saw_unwind:
            print(f"[llm_query] dropped degenerate post-$facet {name} "
                  "(single document, nothing to aggregate)")
            continue

        cleaned.append(stage)
    return cleaned


# Former name, kept so the existing sanity checks keep exercising this.
_strip_lossy_facet_projection = _strip_junk_post_facet_stages


_BREAKDOWN_RE = re.compile(
    r"\b(break\s*down|breakdown|by\s+(?:alert\s+)?type|by\s+day|by\s+month|"
    r"by\s+week|by\s+zone|by\s+hour|each\s+type|per\s+type|distribution)\b",
    re.IGNORECASE,
)

# "which day had the most...", "peak hour", "top 3" and friends legitimately
# want a truncating $limit, so the breakdown fix below must never touch them.
_SUPERLATIVE_RE = re.compile(
    r"\b(most|highest|peak|busiest|top|largest|biggest|fewest|least|lowest|"
    r"quietest|smallest|maximum|minimum|longest|shortest)\b",
    re.IGNORECASE,
)


def _strip_truncating_breakdown_limit(pipeline: list, question: str) -> list:
    """Remove a small $limit that truncates a grouped breakdown.

    Found live: "Break down this month's alerts by type" produced
    $group by alert_type -> $sort count desc -> $limit 1, so the answer
    reported only FAST_INSPECTION (1888) and silently dropped HAND_TOUCH and
    MISSING_CLEANING. Every number shown was real, which makes this worse
    than an obvious error - the answer reads as complete while omitting two
    thirds of the requested breakdown.

    Only applies when the question actually asks for a breakdown AND is not
    a superlative. Any cap removed here is replaced by _finalize_pipeline's
    DEFAULT_RESULT_LIMIT, so output stays bounded.
    """
    if not _BREAKDOWN_RE.search(question) or _SUPERLATIVE_RE.search(question):
        return pipeline

    groups_by_label = any(
        isinstance(s, dict) and "$group" in s
        and isinstance(s["$group"], dict) and s["$group"].get("_id") is not None
        for s in pipeline
    )
    if not groups_by_label:
        return pipeline

    cleaned = []
    for stage in pipeline:
        if (isinstance(stage, dict) and "$limit" in stage
                and isinstance(stage["$limit"], int) and stage["$limit"] < 10):
            print(f"[llm_query] dropped truncating $limit {stage['$limit']} "
                  f"on a breakdown question")
            continue
        cleaned.append(stage)
    return cleaned


_STAGE_LEVEL_ACCUMULATORS = ("$avg", "$sum", "$min", "$max")


def _recover_hoisted_accumulator_stage(pipeline: list) -> list:
    """Rescue a stage that pairs a real stage with an accumulator that has
    been hoisted up to stage level.

    Found live on "What's the average inspection time for hand touch alerts
    in the last 7 days?", where the model emitted a single dict holding two
    top-level keys:

        {"$match": {"alert_type": "HAND_TOUCH", "timestamp": {...}},
         "$expr":  {"$avg": "$inspection_time"}}

    `$expr` is an expression operator, not a pipeline stage, so
    `_split_merged_stages` (which only splits when every key is a valid
    stage) left it alone and `_validate_pipeline` threw the whole thing out
    as "malformed stage". The user's perfectly answerable question then came
    back as the canned "I can only answer questions about the alert log"
    refusal - a false refusal caused purely by one misplaced brace.

    The intent is unambiguous: keep the real stage, and turn the stray
    accumulator into the $group it was meant to be. Dropping the stray key
    alone would not be enough - that would leave a bare $match returning raw
    documents for a question asking for an average, which is exactly the
    "model improvises a number from raw rows" failure `_needs_aggregation`
    exists to prevent.
    """
    recovered = []
    for stage in pipeline:
        if not isinstance(stage, dict) or len(stage) < 2:
            recovered.append(stage)
            continue

        real_stages = {k: v for k, v in stage.items() if k in ALLOWED_STAGES}
        strays = {k: v for k, v in stage.items() if k not in ALLOWED_STAGES}
        if not real_stages or not strays:
            recovered.append(stage)
            continue

        for k, v in real_stages.items():
            recovered.append({k: v})

        for stray_key, stray_val in strays.items():
            # Two shapes seen in the wild: the accumulator wrapped in $expr
            # ({"$expr": {"$avg": "$inspection_time"}}), or the accumulator
            # hoisted bare as the stray key itself ({"$sum": "$x"}).
            if stray_key in _STAGE_LEVEL_ACCUMULATORS:
                acc, acc_arg = stray_key, stray_val
            elif isinstance(stray_val, dict):
                acc = next((a for a in _STAGE_LEVEL_ACCUMULATORS if a in stray_val), None)
                acc_arg = stray_val.get(acc) if acc else None
            else:
                acc, acc_arg = None, None

            if acc:
                field = acc.lstrip("$")
                recovered.append({"$group": {"_id": None, field: {acc: acc_arg}}})
                print(f"[llm_query] recovered hoisted {acc} accumulator into a $group")

    return recovered


_UNARY_ACCUMULATORS = ("$sum", "$avg", "$min", "$max", "$first", "$last")

# The alert log carries exactly one date field. Any date operator applied
# to anything else cannot have been meant literally - see
# _retarget_date_operators.
_DATE_FIELD = "$timestamp"
_DATE_OPERATORS = (
    "$dayOfWeek", "$dayOfMonth", "$dayOfYear", "$hour", "$minute", "$second",
    "$week", "$isoWeek", "$month", "$year", "$isoWeekYear", "$isoDayOfWeek",
)


def _repair_group_missing_id(pipeline: list) -> list:
    """Insert the `_id` a $group left out.

    MongoDB refuses a $group with no _id outright - "a group specification
    must include an _id" (code 15955) - so the question returns nothing at
    all. Seen live on "compare this week vs last week":

        {"$group": {"total_this_week": {"$sum": "$this_week.count"},
                    "total_last_week": {"$sum": "$last_week.count"}}}

    Every accumulator here is an unkeyed total, which is precisely what
    `_id: None` means, so the intent is not ambiguous and the repair is
    the one MongoDB would have required anyway.
    """
    repaired = []
    for stage in pipeline:
        if (isinstance(stage, dict) and len(stage) == 1 and "$group" in stage
                and isinstance(stage["$group"], dict)
                and "_id" not in stage["$group"]):
            print("[llm_query] added the missing _id to a $group")
            repaired.append({"$group": {"_id": None, **stage["$group"]}})
        else:
            repaired.append(stage)
    return repaired


def _unwrap_unary_accumulator_args(node):
    """Unwrap a single-element array handed to a unary accumulator.

    $sum/$avg/$min/$max take one argument in an accumulator position, and
    the model sometimes wraps it in a list out of habit from the
    expression form: {"$sum": ["this_week.count"]}. MongoDB rejects it -
    "The $sum accumulator is a unary operator" (code 40237) - and the
    whole question dies. Observed on "day wise report for this week".

    Only a ONE-element array is unwrapped. A genuine multi-argument
    {"$sum": ["$a", "$b"]} in an expression position is valid and is left
    exactly as it is.
    """
    if isinstance(node, list):
        return [_unwrap_unary_accumulator_args(v) for v in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for key, val in node.items():
        val = _unwrap_unary_accumulator_args(val)
        if key in _UNARY_ACCUMULATORS and isinstance(val, list) and len(val) == 1:
            print(f"[llm_query] unwrapped a single-element array passed to {key}")
            val = val[0]
        out[key] = val
    return out


def _retarget_date_operators(node):
    """Point a date operator at the only date field there is.

    Found live on "which day of the week has the most hand touch alerts?",
    where the model grouped by:

        {"_id": {"dayOfWeek": {"$dayOfWeek": "$dayOfWeek"}}}

    `$dayOfWeek` applied to a field named `dayOfWeek`, which does not
    exist - it echoed the operator name back as its own argument. The
    field is missing, so the whole _id evaluates to null, every alert
    collapses into one bucket, and the answer came back as a flat "There
    were 109,656 alerts" - no day named, for a question whose entire
    point was which day. A silently wrong-shaped answer rather than an
    error, which is the worse failure of the two.

    The collection has exactly one date field (alert_type, zone,
    cloth_detected and inspection_time are not dates), so a date operator
    pointed anywhere else is unambiguously a mistake, and there is only
    one thing it can have meant.
    """
    if isinstance(node, list):
        return [_retarget_date_operators(v) for v in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for key, val in node.items():
        if key in _DATE_OPERATORS and isinstance(val, str) and val != _DATE_FIELD:
            print(f"[llm_query] retargeted {key} from {val!r} to {_DATE_FIELD}")
            out[key] = _DATE_FIELD
            continue
        if (key in ("$dateToString", "$dateTrunc", "$dateToParts")
                and isinstance(val, dict) and isinstance(val.get("date"), str)
                and val["date"] != _DATE_FIELD):
            print(f"[llm_query] retargeted {key} date from {val['date']!r} to {_DATE_FIELD}")
            out[key] = {**_retarget_date_operators(val), "date": _DATE_FIELD}
            continue
        out[key] = _retarget_date_operators(val)
    return out


def _strip_pregroup_limit(pipeline: list) -> list:
    """Drop a $limit that appears before a later $group/$count stage.

    Found live on "total alerts, last 7 days" (a phrasing no different in
    meaning from "how many alerts in the last 7 days", which answers
    correctly): the model wrote

        [{"$match": {...date range...}}, {"$match": {"alert_type": ...}},
         {"$limit": 200}, {"$group": {"_id": None, "count": {"$sum": 1}}}]

    - the 200-row cap it was told to always include ended up applied to
      the RAW documents feeding the $group, not to the group's output.
      1,043 real matches got truncated to 200 before they were ever
      counted, and the $group faithfully reported "200" - a completely
      wrong number with no error, no dropped rows the caller could
      detect, nothing to signal anything went wrong.

    There is no question in this domain where sampling only the first N
    raw documents before aggregating over an entire time period is ever
    the intent - every $limit that belongs before the data reaches a
    $group is written that way by mistake, echoing the "always cap your
    results" instruction into the wrong position. Dropping it can only
    ever make the aggregate MORE complete, never wrong in a new way.
    """
    group_or_count_at = next(
        (i for i, s in enumerate(pipeline)
         if isinstance(s, dict) and ("$group" in s or "$count" in s)),
        None,
    )
    if group_or_count_at is None:
        return pipeline
    cleaned = []
    for i, stage in enumerate(pipeline):
        if i < group_or_count_at and isinstance(stage, dict) and set(stage) == {"$limit"}:
            print(f"[llm_query] dropped a $limit:{stage['$limit']} stage before "
                  "the $group/$count it should have followed")
            continue
        cleaned.append(stage)
    return cleaned


# The alert log has exactly one zone value in practice (see
# ZONES in seed_data.py) - kept as a set, not a bare string, so a future
# multi-zone deployment only has to widen this one constant.
_CANONICAL_ZONES = {"FQC Station 1"}
_ZONE_NORMALIZED = {z.lower().replace("_", " ").replace("-", " ").strip(): z
                     for z in _CANONICAL_ZONES}


def _normalize_zone_value(value: str) -> str | None:
    """Maps any case/spacing/underscore variant of a real zone name back
    to its canonical stored form ("fqc_station_1", "FQC  STATION 1", "fqc-
    station-1" -> "FQC Station 1"), or None if it doesn't match any known
    zone at all. Mirrors the same normalize-then-repair treatment already
    given to alert_type values (see `_row_to_clause`'s ALERT_TYPES
    handling) - a zone filter is exact-string equality against the DB,
    so a plausible-looking variant that isn't byte-for-byte identical
    silently matches nothing, precisely the "zone name always returns 0
    alerts" failure class."""
    if not isinstance(value, str):
        return None
    key = value.lower().replace("_", " ").replace("-", " ")
    key = re.sub(r"\s+", " ", key).strip()
    return _ZONE_NORMALIZED.get(key)


def _repair_zone_filter(node):
    """Walks the pipeline canonicalizing any `zone` match value found
    under a $match/$expr/$eq/$in - see `_normalize_zone_value`. A value
    that doesn't correspond to any real zone at all is left untouched
    (not silently rewritten to a guess); it will correctly match nothing,
    which is the truthful answer for a zone that was never in the data."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "zone":
                if isinstance(v, str):
                    out[k] = _normalize_zone_value(v) or v
                elif isinstance(v, dict) and "$in" in v and isinstance(v["$in"], list):
                    out[k] = {**v, "$in": [_normalize_zone_value(x) or x for x in v["$in"]]}
                elif isinstance(v, dict) and "$eq" in v and isinstance(v["$eq"], str):
                    out[k] = {**v, "$eq": _normalize_zone_value(v["$eq"]) or v["$eq"]}
                else:
                    out[k] = _repair_zone_filter(v)
            else:
                out[k] = _repair_zone_filter(v)
        return out
    if isinstance(node, list):
        return [_repair_zone_filter(v) for v in node]
    return node


def _finalize_pipeline(parsed: dict, question: str, history: list | None = None) -> dict:
    """Shared validate/inject/convert step between a fresh generation and
    an error-driven regeneration (see `_regenerate_after_error`) — both
    start from a raw {"intent", "pipeline", "explanation"} dict and need
    the exact same treatment before they're runnable."""
    intent = parsed.get("intent")
    pipeline = parsed.get("pipeline")

    if intent == "greeting" and not pipeline:
        return dict(GREETING_RESULT)

    if isinstance(pipeline, list):
        pipeline = _wrap_bare_expr_stage(pipeline)
        pipeline = _split_merged_stages(pipeline)
        pipeline = _merge_duplicate_match_equality_stages(pipeline)
        pipeline = _recover_hoisted_accumulator_stage(pipeline)
        pipeline = _repair_group_missing_id(pipeline)
        pipeline = _unwrap_unary_accumulator_args(pipeline)
        pipeline = _repair_noop_replace_root(pipeline)
        pipeline = _strip_empty_sort_stage(pipeline)
        pipeline = _clamp_sort_directions(pipeline)

    # A 3B model under greedy decoding drops the boilerplate "intent" field
    # far more often than it produces a genuinely bad pipeline — trust a
    # valid, non-empty pipeline over a missing/wrong label rather than
    # discarding a perfectly good query because "intent" wasn't restated.
    ok, reason = _validate_pipeline(pipeline)
    if not ok:
        print(f"[llm_query] pipeline rejected: {reason}")
        return dict(UNSUPPORTED_RESULT)

    pipeline = _repair_bare_date_field_reference(pipeline)
    pipeline = _repair_zone_filter(pipeline)
    pipeline = _repair_dotted_timestamp_field_reference(pipeline)
    pipeline = _repair_untruncated_date_group_id(pipeline, question)
    # Runs after the two repairs above, not before: _repair_bare_date_field_reference
    # is itself a source of mis-targeted date operators. It infers the operator from
    # the key name and wraps whatever value it finds, which is right when the model
    # wrote {"dayOfWeek": "$timestamp"} but wrong when it wrote
    # {"dayOfWeek": "$dayOfWeek"} - that becomes {"$dayOfWeek": "$dayOfWeek"}, a date
    # operator reading a field that does not exist.
    pipeline = _retarget_date_operators(pipeline)
    pipeline = _repair_nested_group_accumulators(pipeline)
    pipeline = _repair_chained_group_dropped_field(pipeline)
    pipeline = _coerce_numeric_comparison_strings(pipeline)
    pipeline = _repair_ne_eq_list_operand(pipeline)
    pipeline = _ensure_report_facet_sections(pipeline)
    pipeline = _fix_normal_operation_polarity(pipeline, question)
    pipeline = _fix_misclassified_plain_total(pipeline, question)
    pipeline = _strip_pregroup_limit(pipeline)
    pipeline = _fix_mixed_inclusion_exclusion_project(pipeline)
    pipeline = _strip_junk_post_facet_stages(pipeline)
    pipeline = _strip_truncating_breakdown_limit(pipeline, question)
    pipeline = _strip_unwanted_date_filter(pipeline, question, history)
    pipeline = _inject_normal_operation_exclusion(pipeline, question)
    pipeline = _inject_date_range(pipeline, question, history)
    pipeline = _fix_hour_range_filter(question, pipeline)
    pipeline = _use_indexed_hour_filter(pipeline)
    pipeline = _convert_date_strings(pipeline)
    pipeline = _repair_missing_average_pipeline(pipeline, question)

    if not _has_result_cap(pipeline):
        pipeline.append({"$limit": DEFAULT_RESULT_LIMIT})

    # Recorded so the answer-phrasing step can say what time period it
    # actually used when that period came from a silent history carryover
    # rather than anything the current question itself asked for — see
    # `_resolve_date_range_with_source`.
    _range, _source = _resolve_date_range_with_source(question, history, datetime.now(timezone.utc))
    inherited_range = _range if _source == "inherited" else None

    return {
        "intent": "data_query",
        "pipeline": pipeline,
        "explanation": parsed.get("explanation") or "",
        "_inherited_date_range": inherited_range,
    }


_BARE_GREETING_RE = re.compile(
    r"^\s*(hi+|he+llo+|hey+a?|yo+|sup|howdy|good\s*(morning|afternoon|evening|day))"
    r"[\s,]*(deep\s*insight|assistant|there|bot)?\s*[!.?]*\s*$",
    re.IGNORECASE,
)


def _is_bare_greeting(question: str) -> bool:
    """Deterministic short-circuit for the single most common message
    this app gets, found necessary via live testing: intent
    classification is a free-form judgment call the model makes fresh
    every request, and once conversation history has a prior data
    question in it, a plain "hi" was observed being misclassified as a
    continuation of the earlier data question — reproduced live, e.g.
    "hi" -> a pipeline computing an all-time total count instead of the
    greeting response. A bare greeting is completely unambiguous, so
    there's no reason to trust the model's judgment on it at all rather
    than answering it directly, the same way relative dates ("yesterday")
    are computed deterministically instead of trusted to the model."""
    return bool(_BARE_GREETING_RE.match(question))


# All deterministic fast paths (plain total, average, hour-range count,
# distinct count, multi-year group) were removed per explicit user request -
# every question now goes through the same LLM-generate -> validate -> repair
# pipeline uniformly, no shortcut bypasses it.


_DANGLING_TIME_PREP_RE = re.compile(r"\b(?:on|at|in|since|between|before|after|during)\s*\??\s*$", re.IGNORECASE)


def _is_incomplete_time_question(question: str) -> bool:
    """Catches a question that's been cut off mid-thought right before the
    actual time reference — "how many alerts happened on" with nothing
    after "on". Found live: with no date phrase to extract, this fell
    through every deterministic check AND the model's own judgment, which
    silently dropped the (missing) date filter and confidently answered
    with the ALL-TIME total (307,889) as if that were a real answer to
    "on ___?" — worse than not answering at all, since a fabricated-looking
    but real number reads as trustworthy. A trailing, unresolved
    preposition is a strong, low-false-positive signal that the question
    was never finished, so this is checked deterministically up front
    rather than trusted to either the model or a downstream validator."""
    return bool(_DANGLING_TIME_PREP_RE.search(question.strip()))


_DAY_WORD_RE = re.compile(r"\b(day|date)s?\b", re.IGNORECASE)
_ZERO_ALERTS_RE = re.compile(
    r"\b(no|zero|0)\s+alerts\b|\bwithout\s+(any\s+)?alerts\b", re.IGNORECASE
)


def _is_gap_detection_question(question: str) -> bool:
    """"Was there any day with no alerts" / "did any day have 0 alerts" —
    asks the system to find GAPS: calendar dates with zero matching
    rows. A $group/$sort/$limit pipeline (what every other question in
    this app uses) can only ever rank days that already HAVE at least
    one document; a day with no alerts never appears as a row at all,
    so there is no pipeline shape built here that can answer this.
    Found live: rather than erroring, the model pattern-matches this
    onto the closest thing it knows ("which day had the most/least
    alerts") and confidently answers a completely different question —
    "416 alerts on 2025-03-21" for a yes/no zero-alerts question, which
    reads as a real, trustworthy answer despite not addressing what was
    asked at all. Better to say plainly this isn't supported than
    silently answer a different question with real-looking numbers."""
    return bool(_DAY_WORD_RE.search(question) and _ZERO_ALERTS_RE.search(question))


# The small set of words this file's regexes actually key off - day
# words, alert-type words, and period words. Deliberately short: a
# longer, general-purpose dictionary would risk "correcting" real words
# that just happen to be close to something in it.
_TYPO_VOCAB = [
    "yesterday", "today", "tomorrow", "morning", "afternoon", "evening", "night",
    "cleaning", "touch", "inspection", "missing", "normal", "operation",
    "weekly", "monthly", "yearly", "hourly", "quarterly", "quarter",
    "week", "month", "year", "hour", "minute", "day", "daily",
    "report", "average", "breakdown", "alerts", "alert", "compare",
    "distinct", "between", "zone", "station", "cloth", "detected",
]
_TYPO_VOCAB_SET = set(_TYPO_VOCAB)
_TYPO_WORD_RE = re.compile(r"[A-Za-z]+")

# General fallback dictionary, for typos anywhere in the question that
# aren't one of the specific words above (found live: the first version
# of this fix only caught a ~35-word list, missing everyday typos
# elsewhere in a question - "happend", "recieved", "occured", etc. -
# that don't touch date/type extraction directly but still make the
# question harder for the model to read correctly). Lazily constructed
# (loading its frequency dictionary has a real cost) and only used as a
# SECOND pass, after the precise domain-vocabulary pass above already
# ran - that one stays authoritative for the words that matter most to
# this file's own regexes, since it's proven not to misfire on this
# domain's own vocabulary (see the guards below for why a bare general
# dictionary pass alone is NOT safe to run unguarded: tested "FQC" -> "for"
# and "MongoDB" -> "mongols" against real proper nouns this app uses).
_spellchecker = None


def _get_spellchecker():
    global _spellchecker
    if _spellchecker is None:
        from spellchecker import SpellChecker
        _spellchecker = SpellChecker()
    return _spellchecker


def _normalize_common_typos(question: str) -> str:
    """See FIX GG above. Two passes:
    1. The small, precise domain vocabulary (date/type/period words this
       file's own regexes key off), via difflib at a high similarity
       cutoff - unchanged from the original fix, still authoritative.
    2. A general English dictionary fallback for whatever's left, guarded
       against proper nouns and codes: only a plain lowercase-or-
       Title-Case word (never ALL-CAPS or mIxEd case like "FQC"/
       "MongoDB"/"FastAPI"), never containing a digit, is ever considered
       - and only if the dictionary doesn't already recognize the word as
       correctly spelled (skips real-but-uncommon words the model can
       already handle fine, rather than guessing at them)."""
    def fix_domain(m):
        word = m.group(0)
        lower = word.lower()
        # 4-letter minimum (not 5) - this pass is a small, curated,
        # low-false-positive vocabulary (tested: no accidental matches
        # against common 4-letter words like then/than/some/many/with),
        # and 4-letter typos of real vocabulary words are common - found
        # live: "tday" (a typo of "today") was skipped entirely at
        # the old 5-letter minimum, so the question got zero date filter
        # at all and silently returned the ALL-TIME total mislabeled
        # "today".
        if lower in _TYPO_VOCAB_SET or len(lower) < 4:
            return word
        matches = difflib.get_close_matches(lower, _TYPO_VOCAB, n=1, cutoff=0.8)
        if matches:
            corrected = matches[0]
            print(f"[llm_query] typo-corrected {word!r} -> {corrected!r}")
            return corrected
        return word
    question = _TYPO_WORD_RE.sub(fix_domain, question)

    def fix_general(m):
        word = m.group(0)
        # Proper-noun / code guard: only plain "word" or "Word" (first
        # letter optionally capitalized, everything else lowercase) is
        # eligible - "FQC", "MongoDB", "FastAPI" and similar all fail
        # this check and are left completely untouched.
        if not (word.islower() or (word[:1].isupper() and word[1:].islower())):
            return word
        if len(word) < 5:
            return word
        lower = word.lower()
        if lower in _TYPO_VOCAB_SET:
            return word  # already handled (or correct) in the domain pass
        try:
            sp = _get_spellchecker()
        except Exception:
            return word  # dictionary unavailable - never block the request over this
        if lower in sp:
            return word  # a real, already-correctly-spelled word - leave it alone
        correction = sp.correction(lower)
        if correction and correction != lower:
            fixed = correction.capitalize() if word[:1].isupper() else correction
            print(f"[llm_query] general typo-corrected {word!r} -> {fixed!r}")
            return fixed
        return word
    return _TYPO_WORD_RE.sub(fix_general, question)


_LIKELY_REAL_QUESTION_RE = re.compile(
    r"\balerts?\b|\binspections?\b|\bcleaning\b|\btouch\b|\bzone\b|\breport\b|\bstation\b",
    re.IGNORECASE,
)


def parse_question_to_pipeline(question: str, history: list | None = None) -> dict:
    question = _normalize_common_typos(question)
    if _is_bare_greeting(question):
        return dict(GREETING_RESULT)

    if _is_incomplete_time_question(question):
        return dict(UNSUPPORTED_RESULT)

    if _is_gap_detection_question(question):
        return dict(GAP_QUESTION_RESULT)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    history_block = ""
    if history:
        # A prior report's phrased answer can run to many hundreds of
        # tokens on its own (it narrates several breakdowns in prose) -
        # capped per-answer so the history block can never grow large
        # enough to blow the model's context window on its own, no
        # matter how verbose an earlier turn's answer was. See FIX DD.
        _HISTORY_ANSWER_CHAR_CAP = 300
        lines = []
        for h in history[-4:]:
            answer_text = h["answer"] or ""
            if len(answer_text) > _HISTORY_ANSWER_CHAR_CAP:
                answer_text = answer_text[:_HISTORY_ANSWER_CHAR_CAP] + "..."
            lines.append(f'Q: {h["question"]}\nA: {answer_text}')
        history_block = "Conversation so far:\n" + "\n".join(lines) + "\n\n"

        last_pipeline = history[-1].get("pipeline")
        # A multi-section $facet REPORT pipeline is large (5+ named
        # sub-pipelines) and, dumped whole into a follow-up's prompt,
        # was observed live overwhelming the model into ignoring the new
        # question entirely — asked "how many alerts between 2pm and 4pm
        # yesterday" right after a quarterly report, it answered a
        # completely unrelated "peak day+hour" question instead. Reports
        # are also rarely a useful starting point to "adapt" for an
        # unrelated follow-up anyway, so it's better left out.
        is_report_pipeline = isinstance(last_pipeline, list) and any(
            isinstance(s, dict) and "$facet" in s for s in last_pipeline
        )
        if last_pipeline and not is_report_pipeline:
            history_block += f"Previous pipeline (adapt for a follow-up if relevant): {json.dumps(last_pipeline)}\n\n"

    user_prompt = f"{history_block}Current date: {today}\nNew question: {question}"

    try:
        parsed = _generate_pipeline_json(user_prompt)
    except ValueError as e:
        # Safety net for FIX DD: the per-answer cap above should already
        # keep this from happening, but a single very long new question
        # (or history sitting right at the boundary) could still overflow
        # the context window. Degrade gracefully - drop history entirely
        # and try once more as a fresh, standalone question - rather than
        # crashing the request outright.
        if history and "exceed context window" in str(e):
            print(f"[llm_query] prompt overflowed context window with history included "
                  f"({e}) - retrying once with history dropped")
            fallback_prompt = f"Current date: {today}\nNew question: {question}"
            try:
                parsed = _generate_pipeline_json(fallback_prompt)
            except ValueError:
                return dict(UNSUPPORTED_RESULT)
        else:
            raise
    if parsed is None:
        return dict(UNSUPPORTED_RESULT)

    if parsed.get("intent") == "unsupported" and _LIKELY_REAL_QUESTION_RE.search(question):
        print(f"[llm_query] greedy attempt classified a plausible real data question as "
              f"unsupported - retrying once with sampling before accepting that verdict")
        retried = _generate_pipeline_json(user_prompt, force_sample=True)
        if retried is not None and retried.get("intent") != "unsupported":
            parsed = retried

    return _finalize_pipeline(parsed, question, history)


def _regenerate_after_error(question: str, failed_pipeline: list, error_msg: str, history: list | None = None) -> dict | None:
    """A pipeline that passes validation can still fail at Mongo — most
    often a hallucinated operator name that isn't in the forbidden-list
    scan because it's simply not a real thing (e.g. "$weekOfYear", which
    doesn't exist in MongoDB at all). Rather than enumerating every
    possible wrong operator name in the prompt up front, feed the actual
    Mongo error back and let the model self-correct once — this also
    self-heals the case where a bad pipeline got copied into a follow-up
    question's history and would otherwise keep failing every turn after.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    user_prompt = (
        f"Current date: {today}\nQuestion: {question}\n\n"
        f"This pipeline was generated for that question, but MongoDB rejected it:\n"
        f"{json.dumps(failed_pipeline, default=str)}\n\n"
        f"MongoDB error: {error_msg}\n\n"
        f"Output a corrected pipeline (same JSON schema) that fixes this error and still answers the question."
    )
    parsed = _generate_pipeline_json(user_prompt)
    if parsed is None:
        return None
    return _finalize_pipeline(parsed, question, history)


AGGREGATION_STAGES = {"$group", "$bucket", "$bucketAuto", "$sortByCount", "$count", "$facet"}
RANKING_INTENT_RE = re.compile(
    r"\b(most|least|highest|lowest|average|busiest|peak|top\s*\d|breakdown|break\s*down|"
    r"compare|comparison|how many|count(?:\s+the|\s+of)?|total|which\s+\w+\s+had)\b",
    re.IGNORECASE,
)


def _needs_aggregation(question: str, pipeline) -> bool:
    """Catches the most dangerous failure mode found in testing: a
    question that needs counting/grouping ("which day had the most hand
    touch alerts") gets a pipeline that's just $match + $limit, returning
    raw unaggregated documents. Nothing else in the safety net catches
    this — a raw alert document (narration, objects_present, etc.) isn't
    scalar or facet-shaped, so none of the completeness/omission checks
    in `_finalize_answer` apply to it, and the phrasing model was
    observed filling the gap by estimating a "busiest day" from a 25-row
    preview of raw documents — a fabricated number (it said 9; the real
    answer, computed properly, was a 3-way tie at 7) presented with full
    confidence. Better to catch the pipeline shape itself before it ever
    reaches the point where the model has to improvise an answer.
    """
    if not isinstance(pipeline, list):
        return False
    stage_names = {next(iter(s)) for s in pipeline if isinstance(s, dict) and len(s) == 1}
    if stage_names & AGGREGATION_STAGES:
        return False
    return bool(RANKING_INTENT_RE.search(question))


_AVERAGE_INTENT_RE = re.compile(r"\b(average|avg|mean)\b", re.IGNORECASE)


def _needs_average_fix(question: str, pipeline) -> bool:
    """`_needs_aggregation` only checks that SOME aggregation stage is
    present, which a pipeline can satisfy with a $group that never
    actually computes an average — found live on "what's the average
    inspection time for 7 days": the model wrote a "busiest hour"-shaped
    pipeline instead ({"_id": {"hour": ...}, "count": {"$sum": 1}},
    $sort, $limit 1), which executes fine and passes the aggregation
    check, then got a nonsense $project dividing that count by 3600 and
    relabelling it "avg_inspection_time" — a fluent, confident, totally
    wrong answer ("0.02 alerts between 3pm and 4pm") for a question that
    named no hour at all. A question asking for an average needs a real
    $avg accumulator somewhere in the pipeline; nothing else reliably
    signals that the aggregation computed is the right KIND, only that
    some aggregation happened."""
    if not isinstance(pipeline, list):
        return False
    if not _AVERAGE_INTENT_RE.search(question):
        return False
    return not _contains_operator(pipeline, "$avg")


def _repair_missing_average_pipeline(pipeline: list, question: str) -> list:
    """Deterministic counterpart to `_needs_average_fix`: the one retry
    the self-correction loop gives the model isn't reliable here either
    (found live: given "add a real $avg accumulator", the 3B model kept
    its wrong "busiest hour" pipeline shape completely unchanged).
    `inspection_time` is the only numeric field in this schema (see
    seed_data.py) — "average" only ever means "average inspection_time"
    here, so it's safe to rebuild directly instead of hoping a second
    model attempt gets the shape right: keep whatever leading $match
    stage(s) the model already produced (that part — the date/type
    scope — is usually right) and replace everything after with a
    single real $avg group."""
    if not _needs_average_fix(question, pipeline):
        return pipeline
    match_stages = []
    for stage in pipeline:
        if isinstance(stage, dict) and len(stage) == 1 and "$match" in stage:
            match_stages.append(stage)
        else:
            break
    print("[llm_query] rebuilt pipeline to compute a real average instead of the model's wrong shape")
    return match_stages + [{"$group": {"_id": None, "avg_inspection_time": {"$avg": "$inspection_time"}}}]


def _mentioned_alert_type(question: str) -> str | None:
    """Returns the single alert type named in the question text, if
    exactly one is — "hand touch alerts", "fast inspection", etc. Used to
    catch a specific follow-up failure mode: a multi-turn conversation's
    history (an unrelated earlier "break down by type" turn, say) leads
    the model to write a pipeline that groups across ALL alert types
    instead of filtering to the one actually asked about, so $sort+$limit
    picks whichever type/day combination happens to have the global max —
    observed live returning "Fast Inspection" data for a question that
    explicitly asked about hand touch alerts."""
    q = question.lower()
    found = [t for t in ALERT_TYPES if t != "NORMAL_OPERATION" and t.replace("_", " ").lower() in q]
    return found[0] if len(found) == 1 else None


def _pipeline_filters_to_type(pipeline: list, alert_type: str) -> bool:
    """True if some $match stage constrains alert_type to exactly this
    value (directly, or via a single-value $in/$eq) anywhere in the
    pipeline — a $group grouping BY alert_type doesn't count, since that's
    exactly the shape that let a different type's data win the ranking."""
    for stage in pipeline:
        if not isinstance(stage, dict) or "$match" not in stage:
            continue
        cond = stage["$match"]
        if not isinstance(cond, dict):
            continue
        val = cond.get("alert_type")
        if val == alert_type:
            return True
        if isinstance(val, dict):
            in_list = val.get("$in") or val.get("$eq")
            if in_list == alert_type or (isinstance(in_list, list) and in_list == [alert_type]):
                return True
    return False


def _needs_type_filter_fix(question: str, pipeline) -> str | None:
    """Returns the alert type that SHOULD be filtered on but isn't, or
    None if the pipeline is fine (either no single type was named, or it
    already filters correctly)."""
    if not isinstance(pipeline, list):
        return None
    alert_type = _mentioned_alert_type(question)
    if not alert_type:
        return None
    if _pipeline_filters_to_type(pipeline, alert_type):
        return None
    return alert_type


_LOW_SUPERLATIVES = {"least", "lowest", "fewest"}


def _needs_ranking_sort_fix(question: str, pipeline) -> str | None:
    """Catches a hallucination shape found in testing: a "peak hour"
    question correctly groups by hour and counts, but then sorts by the
    $group's `_id` (the hour number itself) instead of by the count —
    e.g. `{"$sort": {"_id": 1}}, {"$limit": 1}` returns hour 0 (whichever
    happens to sort first numerically) instead of the hour with the
    highest count. Mongo doesn't error on this, it just silently answers
    a completely different question ("what's the earliest hour with any
    alerts" instead of "which hour has the most alerts"). Only fires when
    there's a single accumulator field to rank by and a $limit narrowing
    to a single top/bottom result — the highest-risk shape, where sorting
    on the wrong field changes which single row comes back."""
    if not isinstance(pipeline, list):
        return None
    match = SUPERLATIVE_RE.search(question)
    if not match:
        return None
    if not any(isinstance(s, dict) and "$limit" in s for s in pipeline):
        return None

    group_stage = next((s["$group"] for s in pipeline if isinstance(s, dict) and "$group" in s), None)
    if not isinstance(group_stage, dict):
        return None
    accumulator_fields = [
        k for k, v in group_stage.items()
        if k != "_id" and isinstance(v, dict)
        and any(op in v for op in ("$sum", "$avg", "$max", "$min", "$count"))
    ]
    if len(accumulator_fields) != 1:
        return None
    metric = accumulator_fields[0]

    sort_stage = next((s["$sort"] for s in pipeline if isinstance(s, dict) and "$sort" in s), None)
    if not isinstance(sort_stage, dict):
        return None

    wants_desc = match.group(1).lower() not in _LOW_SUPERLATIVES
    expected = -1 if wants_desc else 1
    if sort_stage.get(metric) == expected:
        return None
    return metric


_EXPLICIT_DATE_RE = re.compile(r"\bdate\b|\bday\b", re.IGNORECASE)
_EXPLICIT_WEEKDAY_RE = re.compile(r"\bday\s*-?\s*of\s*-?\s*(the\s+)?week\b|\bweekday\b", re.IGNORECASE)


def _needs_date_vs_weekday_fix(question: str, pipeline) -> str | None:
    """Catches two related failures found in testing: (1) plain "which
    day/date had the most/least X" defaulting to day-OF-THE-WEEK grouping
    (Sunday/Monday/... — a category that repeats every week) instead of
    an actual calendar date, which is what a bare "day"/"date" — without
    "day of the week"/"weekday" explicitly said — actually means; and (2)
    a follow-up specifically copying the PREVIOUS turn's day-of-week
    pipeline verbatim even once the new question's wording asked for a
    date. \bday\b (not "today"/"yesterday"/"days" — none of those have
    "day" as a separate whole word) is treated the same as "date" unless
    "day of the week"/"weekday" is explicitly said. Returns "weekday" or
    "date" (which wrong grouping was found), or None if the pipeline's
    grouping already matches what THIS question asked for."""
    if not isinstance(pipeline, list):
        return None
    wants_date = bool(_EXPLICIT_DATE_RE.search(question)) and not _EXPLICIT_WEEKDAY_RE.search(question)
    wants_weekday = bool(_EXPLICIT_WEEKDAY_RE.search(question))
    if not wants_date and not wants_weekday:
        return None

    def _contains(node, needle: str) -> bool:
        if isinstance(node, str):
            return needle in node
        if isinstance(node, dict):
            return any(needle in k for k in node.keys()) or any(_contains(v, needle) for v in node.values())
        if isinstance(node, list):
            return any(_contains(v, needle) for v in node)
        return False

    group_id = None
    for stage in pipeline:
        if isinstance(stage, dict) and isinstance(stage.get("$group"), dict):
            group_id = stage["$group"].get("_id")
            break
    if group_id is None:
        return None

    groups_by_weekday = _contains(group_id, "dayOfWeek")
    groups_by_date = _contains(group_id, "dateToString")

    if wants_date and groups_by_weekday and not groups_by_date:
        return "weekday"
    if wants_weekday and groups_by_date and not groups_by_weekday:
        return "date"
    return None


def _needs_date_truncation_fix(question: str, pipeline) -> bool:
    """Catches a hallucination shape distinct from the weekday-vs-date
    mix-up above: the pipeline correctly decides to group by "day" but
    groups by the raw, untruncated `$timestamp` field (an exact
    millisecond instant) instead of a truncated calendar date. Every
    document's timestamp differs from every other's, so this produces
    one group per document rather than one group per calendar day,
    which makes a "most alerts in a day" ranking meaningless — the
    "winning" group is just whichever document happens to sort first,
    with a count of 1 or 2. Found live: a same-day follow-up question
    ("which day had the most hand touch alerts?" right after a turn
    scoped to one specific day) reliably reproduced this shape even
    though the exact same question asked fresh (no history) reliably
    produces the correct $dateToString grouping — the extra
    conversation history in the prompt measurably increases how often
    the model reaches for the simpler, wrong, raw-field group key."""
    if not isinstance(pipeline, list):
        return False
    if not (_EXPLICIT_DATE_RE.search(question) and not _EXPLICIT_WEEKDAY_RE.search(question)):
        return False
    if not any(isinstance(s, dict) and "$limit" in s for s in pipeline):
        return False
    group_stage = next((s["$group"] for s in pipeline if isinstance(s, dict) and "$group" in s), None)
    if not isinstance(group_stage, dict):
        return False
    group_id = group_stage.get("_id")

    def _is_bare_timestamp(node) -> bool:
        if node == "$timestamp":
            return True
        if isinstance(node, dict):
            return any(_is_bare_timestamp(v) for v in node.values())
        return False

    def _has_truncation(node) -> bool:
        if isinstance(node, dict):
            if any(op in node for op in ("$dateToString", "$dateTrunc", "$dayOfMonth", "$dayOfYear", "$dayOfWeek")):
                return True
            return any(_has_truncation(v) for v in node.values())
        return False

    return _is_bare_timestamp(group_id) and not _has_truncation(group_id)


def _groups_by_null_with_first_last(pipeline) -> bool:
    """Catches a follow-up-contamination hallucination found in testing:
    a "most common alert TYPE" question gets a pipeline that groups ALL
    documents into ONE bucket ({"_id": null}) — collapsing the very
    dimension ("which alert_type") the question needed distinguished —
    while smuggling in a category value via {"$first": "$alert_type"},
    which just grabs whichever document happens to come first in Mongo's
    arbitrary/insertion order, not the type with the highest count.
    Observed live right after a $facet report turn in conversation
    history: {"$group": {"_id": null, "count": {"$sum": 1}, "alert_type":
    {"$first": "$alert_type"}}} — count came out as the GRAND TOTAL across
    every type (1124), but was then reported as if it were that one
    "$first"-picked type's own count. Grouping by null while also
    $first/$last-ing a categorical field is never a valid way to answer
    "which X is most common" — the category itself must be the group
    key."""
    if not isinstance(pipeline, list):
        return False
    for stage in pipeline:
        if not isinstance(stage, dict) or not isinstance(stage.get("$group"), dict):
            continue
        group = stage["$group"]
        if group.get("_id") is not None:
            continue
        for key, val in group.items():
            if key == "_id":
                continue
            if isinstance(val, dict) and ("$first" in val or "$last" in val):
                return True
    return False


def _uses_now_variable(pipeline) -> bool:
    """Catches a specific hallucination shape found in testing: a $group
    keyed on `"$$NOW.hour"` (or any other use of the `$$NOW` system
    variable) instead of extracting the hour from the document's own
    `$timestamp` field (e.g. `{"$hour": "$timestamp"}`). `$$NOW` is the
    time the QUERY ran, not per-document data, so grouping by it collapses
    every matching document into one bucket regardless of when the alert
    actually happened — Mongo doesn't error on this (it's syntactically
    valid), it just silently produces a meaningless single group, which
    the phrasing step then has no real hour data to describe and fabricates
    one instead (observed live: "peak hour ... 10pm and 11pm" for a
    pipeline whose only group came back as `{"_id": null, "count": 299}`,
    a single bucket with no hour information at all)."""
    if not isinstance(pipeline, list):
        return False

    def _contains_now(node) -> bool:
        if isinstance(node, str):
            return "$NOW" in node
        if isinstance(node, dict):
            return any(_contains_now(v) for v in node.values())
        if isinstance(node, list):
            return any(_contains_now(v) for v in node)
        return False

    return _contains_now(pipeline)


NEEDS_NOW_FIX_MSG = (
    "This pipeline references the `$$NOW` system variable, which is the current wall-clock time "
    "the query is running at — it is NOT per-document data. Grouping or filtering by `$$NOW` (e.g. "
    '"$$NOW.hour") collapses every document into one meaningless bucket regardless of when the alert '
    'actually happened. To extract the hour/day/etc. of each alert, use the document\'s own '
    '"$timestamp" field instead, e.g. {"$hour": "$timestamp"} or {"$dayOfWeek": "$timestamp"}. '
    "Remove every reference to `$$NOW` and replace it with the appropriate expression over `$timestamp`."
)


NEEDS_AGGREGATION_MSG = (
    "This pipeline only filters and limits — it returns raw, unaggregated documents. "
    "The question requires counting/grouping to answer correctly (e.g. a 'which X had the most' "
    "question needs $group + $sort + $limit; a 'how many' question needs $count or $group). "
    "Add the missing $group/$count/$bucket/$sortByCount stage."
)


def _stage(stages: list, name: str, description: str, start_time: float) -> None:
    """Appends one lifecycle-stage record with its wall-clock duration —
    the backbone of the "show query" detail panel's stage-by-stage
    breakdown (what happened, in what order, and how long each part
    took), not just the final pipeline JSON."""
    stages.append({
        "name": name,
        "description": description,
        "duration_ms": round((time.time() - start_time) * 1000),
    })


def _resolve_query(question: str, history: list | None) -> tuple[dict, dict, list]:
    """Runs the generate -> execute step, with up to two self-correction
    passes chained together — one retry used to only cover a Mongo
    execution error, but a fix for one problem (e.g. adding the missing
    $group the question needed) can itself introduce a different one (a
    $group followed by a garbage {"$replaceRoot": {"newRoot": "$"}} that
    then fails to execute), and the old single-shot version just silently
    gave up and fell back to the ORIGINAL broken, non-aggregating
    pipeline in that case — which is how a fabricated answer ("12
    FAST_INSPECTION alerts" from a raw, unfiltered-by-type document dump)
    made it all the way to the user despite the aggregation guard having
    correctly caught the original problem. Looping means whatever the
    last attempt produced is used even if imperfect, which is never worse
    than deterministically returning the one pipeline already known to be
    wrong. Returns (parsed, result, stages)."""
    stages: list = []

    t0 = time.time()
    parsed = parse_question_to_pipeline(question, history)
    _stage(stages, "Query generation",
           "Asked the language model to translate the question into a MongoDB aggregation pipeline "
           "(includes JSON-syntax repair and structural validation of the result).", t0)

    t0 = time.time()
    result = execute_pipeline(parsed["intent"], parsed.get("pipeline"))
    _stage(stages, "Database execution", "Ran the generated pipeline against MongoDB.", t0)

    for attempt in range(2):
        # A deterministically-built pipeline (`_try_multi_year_group` etc.)
        # is correct by construction and doesn't need the heuristic checks
        # below — found live: `_needs_type_filter_fix` doesn't know the
        # difference between "filter to just this type" and "compute this
        # type's SHARE of the total" (the whole point of a percentage
        # pipeline), so it flagged a correct percentage-by-year pipeline as
        # missing a type filter and "fixed" it by filtering out every other
        # type — which zeroed out the denominator and made the percentage
        # always ~100%. A real Mongo execution error is still worth
        # retrying even for a trusted pipeline, so only that case is kept.
        if parsed.get("_trusted_pipeline") and result.get("intent") != "error":
            break

        missing_type = _needs_type_filter_fix(question, parsed.get("pipeline")) if result.get("intent") == "data_query" else None

        if result.get("intent") == "error":
            problem = result.get("error", "")
            reason = f"the database rejected it ({problem})"
            print(f"[llm_query] pipeline execution failed, self-correcting: {problem}")
        elif _needs_aggregation(question, parsed.get("pipeline")):
            problem = NEEDS_AGGREGATION_MSG
            reason = "it only filtered/limited instead of aggregating, which the question needed"
            print("[llm_query] pipeline returns raw documents but question needs aggregation, retrying")
        elif _needs_average_fix(question, parsed.get("pipeline")):
            problem = (
                'The question asks for an AVERAGE, but this pipeline never uses a "$avg" accumulator '
                "anywhere — it computes something else entirely (e.g. counting/grouping by hour) and "
                "just relabels that result as if it were the average. Group with a real $avg accumulator "
                'instead: {"_id": null, "avg": {"$avg": "$inspection_time"}} (add an alert_type/date filter '
                "to the $match first if the question named one, but do not group by hour/day/type unless "
                "the question explicitly asked for a breakdown)."
            )
            reason = "it never computed an actual average ($avg) despite the question asking for one"
            print("[llm_query] pipeline doesn't compute an average despite the question asking for one, retrying")
        elif _groups_by_null_with_first_last(parsed.get("pipeline")):
            problem = (
                'This pipeline groups everything into ONE bucket ({"_id": null}) and then uses '
                '{"$first": ...} or {"$last": ...} to smuggle in a categorical field (e.g. alert_type) — '
                "that just grabs an arbitrary document's value, not the category with the highest count, "
                'and the accumulated "count" is the GRAND TOTAL across every category, not that one '
                'category\'s own count. To find which category is most/least common, group BY that '
                'category instead: {"_id": "$alert_type", "count": {"$sum": 1}}, then $sort + $limit.'
            )
            reason = "it grouped everything into one bucket instead of grouping by the category being ranked"
            print("[llm_query] pipeline groups by null with $first/$last instead of the real category, retrying")
        elif _uses_now_variable(parsed.get("pipeline")):
            problem = NEEDS_NOW_FIX_MSG
            reason = "it grouped/filtered by the current query time ($$NOW) instead of each alert's own timestamp"
            print("[llm_query] pipeline references $$NOW instead of the document timestamp, retrying")
        elif (wrong_grouping := _needs_date_vs_weekday_fix(question, parsed.get("pipeline"))):
            if wrong_grouping == "weekday":
                problem = (
                    'The question asks for a specific calendar "date", but this pipeline groups by '
                    'day-OF-THE-WEEK ($dayOfWeek — Sunday/Monday/etc, a category that repeats every week) '
                    "instead of an actual date. Group by the real calendar date instead: "
                    '{"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}}, "count": {"$sum": 1}}.'
                )
            else:
                problem = (
                    'The question asks for a "day of the week", but this pipeline groups by calendar date '
                    "instead. Group by day of the week instead: "
                    '{"_id": {"dayOfWeek": {"$dayOfWeek": "$timestamp"}}, "count": {"$sum": 1}}.'
                )
            reason = (
                f"it grouped by {'day-of-week' if wrong_grouping == 'weekday' else 'calendar date'} "
                "instead of what the question actually asked for"
            )
            print(f"[llm_query] pipeline grouping mismatch ({wrong_grouping} vs question), retrying")
        elif _needs_date_truncation_fix(question, parsed.get("pipeline")):
            problem = (
                'This pipeline groups by the raw "$timestamp" field directly, which is an exact '
                "millisecond instant — nearly every document has a different one, so this produces "
                "one group per document instead of one group per calendar day, and the ranking becomes "
                'meaningless. Group by the truncated calendar date instead: {"_id": {"$dateToString": '
                '{"format": "%Y-%m-%d", "date": "$timestamp"}}, "count": {"$sum": 1}}.'
            )
            reason = "it grouped by the raw timestamp instead of a truncated calendar date"
            print("[llm_query] pipeline groups by raw untruncated timestamp instead of calendar date, retrying")
        elif (bad_sort_metric := _needs_ranking_sort_fix(question, parsed.get("pipeline"))):
            wants_desc = SUPERLATIVE_RE.search(question).group(1).lower() not in _LOW_SUPERLATIVES
            problem = (
                f'This pipeline\'s $sort stage does not sort by "{bad_sort_metric}" (the $group\'s own '
                f"count/aggregate field) in the direction the question needs. Sorting by the group key "
                f'(e.g. "_id") instead of the aggregated metric returns an arbitrary row, not the actual '
                f'highest/lowest one. Fix the $sort stage to sort by "{bad_sort_metric}" '
                f"({-1 if wants_desc else 1} for {'descending' if wants_desc else 'ascending'})."
            )
            reason = f'it sorted by the wrong field instead of "{bad_sort_metric}"'
            print(f"[llm_query] pipeline sorts by the wrong field instead of \"{bad_sort_metric}\", retrying")
        elif missing_type:
            problem = (
                f'The question specifically asks about "{missing_type}" alerts, but this pipeline never '
                f'filters alert_type to just "{missing_type}" — it aggregates across ALL alert types, so '
                f"whichever type happens to have the highest count can win instead of the one actually asked "
                f'about. Add {{"alert_type": "{missing_type}"}} to the (first) $match stage.'
            )
            reason = f'it never filtered to the "{missing_type}" alert type the question named'
            print(f"[llm_query] pipeline doesn't filter to the alert type the question named ({missing_type}), retrying")
        else:
            break

        if not parsed.get("pipeline"):
            break

        t0 = time.time()
        retried = _regenerate_after_error(question, parsed["pipeline"], problem, history)
        if not retried or retried.get("intent") != "data_query":
            _stage(stages, f"Self-correction attempt {attempt + 1}",
                   f"Previous pipeline was rejected ({reason}); regeneration did not produce a usable fix, "
                   "so the previous result was kept.", t0)
            break
        retried_result = execute_pipeline(retried["intent"], retried.get("pipeline"))
        _stage(stages, f"Self-correction attempt {attempt + 1}",
               f"Previous pipeline was rejected ({reason}); regenerated and re-ran it against MongoDB.", t0)
        parsed, result = retried, retried_result

    return parsed, result, stages


# =========================================================
# STEP 2: execute the validated pipeline against Mongo
# =========================================================

def _round_floats(node):
    """$avg on inspection_time comes back as a raw float with a dozen+
    decimal digits (11.22871287128713) — round it for display everywhere
    it's used (table, answer-generation prompt, deterministic fallback)
    rather than patching each call site separately. Recurses into $facet
    sub-lists, which are the one place a row's values are themselves
    dicts/lists rather than scalars."""
    if isinstance(node, float):
        return round(node, 2)
    if isinstance(node, dict):
        return {k: _round_floats(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_round_floats(v) for v in node]
    return node


def _jsonify_value(v):
    """Recursively makes a single value JSON-safe (ObjectId, datetime ->
    str). MUST recurse into dicts/lists, not just handle the top level —
    a compound $group "_id" like {"_id": {"dayOfWeek": "$timestamp"}}
    (the model forgetting to wrap a field in $dayOfWeek, so Mongo groups
    by the raw Date value instead of a day-of-week number) puts a live
    datetime object one level down, and json.dumps() on that later
    crashed the whole request with an unhandled 500 — "Object of type
    datetime is not JSON serializable" — since a shallow, top-level-only
    conversion never saw it."""
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, dict):
        return {k: _jsonify_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_jsonify_value(item) for item in v]
    if isinstance(v, (str, int, float, bool, type(None))):
        return _round_floats(v)
    return str(v)


def _jsonify(rows: list) -> list:
    """Makes aggregation results JSON-safe (ObjectId, datetime -> str)."""
    return [{k: _jsonify_value(v) for k, v in row.items()} for row in rows]


def execute_pipeline(intent: str, pipeline: list | None) -> dict:
    if intent == "unsupported":
        return {"intent": "unsupported", "rows": []}
    if intent == "gap_unsupported":
        return {"intent": "gap_unsupported", "rows": []}
    if intent == "greeting":
        return {"intent": "greeting", "rows": []}

    collection = get_alerts_collection()
    try:
        cursor = collection.aggregate(pipeline, maxTimeMS=5000)
        # A HARD safety net, independent of whatever the pipeline itself
        # claims to have capped. `_has_result_cap` treats any pipeline
        # ending in $group as already bounded, which is only true when
        # the group key has low cardinality - grouping by a
        # high-cardinality key (the raw $timestamp, a per-second
        # $dateToString, ...) can legitimately return one group per
        # document. islice stops PULLING further rows from the cursor
        # the moment the cap is hit, rather than materializing the full
        # result and truncating after - the difference between a >200-row
        # answer costing a few extra network round-trips and one costing
        # a 27-million-character response body.
        fetched = list(itertools.islice(cursor, DEFAULT_RESULT_LIMIT + 1))
        truncated = len(fetched) > DEFAULT_RESULT_LIMIT
        rows = _jsonify(fetched[:DEFAULT_RESULT_LIMIT])
    except Exception as exc:
        return {"intent": "error", "rows": [], "error": str(exc)}

    # Flatten {"_id": {"hour": 7}, "count": 2}-shaped rows (from grouping
    # by $hour/$dayOfWeek/etc.) into {"hour": 7, "count": 2} — done here,
    # at the source, so the frontend table and the answer-phrasing step
    # both see the same clean shape instead of the phrasing step quietly
    # fixing it up for itself while the table still shows raw nested JSON.
    rows = _flatten_rows_for_output(rows)
    rows = _restructure_report(rows, pipeline)

    return {"intent": "data_query", "rows": rows, "truncated": truncated}


def _week_label(year: int, month: int, week_num: int) -> str:
    from calendar import monthrange

    start_day = (week_num - 1) * 7 + 1
    end_day = min(start_day + 6, monthrange(year, month)[1])
    month_name = datetime(year, month, 1).strftime("%B")
    return f"{month_name} Week {week_num} ({month:02d}/{start_day:02d}-{month:02d}/{end_day:02d})"


def _compute_week_rollup(day_rows: list) -> list:
    """Rolls per-day counts up into "1st week = days 1-7, 2nd week = days \
    8-14, ..." weekly totals — the exact grouping asked for. Done here in \
    Python rather than asked of the model: MongoDB has no built-in \
    "week-of-month" operator, and a $group needing $dayOfMonth + a \
    computed bucket index is exactly the kind of multi-step expression \
    this model doesn't reproduce reliably. The model only ever has to \
    write a plain per-day $group (already a proven-reliable pattern); \
    this function does the guaranteed-correct rollup on real, already-\
    aggregated numbers.
    """
    totals: dict[tuple[int, int, int], int] = {}
    for row in day_rows:
        if not isinstance(row, dict):
            continue
        # The group key comes back as "date" when the model nests it in a
        # sub-dict ({"_id": {"date": ..., ...}}), or stays "_id" when it
        # groups by a bare date string ({"_id": "2026-04-01"}) — both are
        # valid, equally-correct pipeline shapes, so both need handling
        # here rather than only recognizing one of them.
        date_str = row.get("date")
        if date_str is None and isinstance(row.get("_id"), str):
            date_str = row["_id"]
        count = row.get("count")
        if not date_str or count is None:
            continue
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        week_num = ((d.day - 1) // 7) + 1
        key = (d.year, d.month, week_num)
        totals[key] = totals.get(key, 0) + count

    return [
        {"week": _week_label(*key), "count": totals[key]}
        for key in sorted(totals.keys())
    ]


def _report_scope(pipeline: list) -> str:
    """'week' | 'month' | 'quarter', based on the report's own date span
    (the leading $match's timestamp $gte/$lt) — decides which section
    layout `_restructure_report` builds. Span-based rather than matching
    keywords in the question text: the model's own $gte/$lt is already
    correctly sized per the "quarterly = 3 months" etc. prompt rule, so
    it's a strictly more reliable signal, and it also handles phrasings
    the three fixed keywords wouldn't ("a report for the last 4 weeks",
    "report for June")."""
    for stage in pipeline:
        if not isinstance(stage, dict) or "$match" not in stage:
            continue
        ts = stage["$match"].get("timestamp")
        if isinstance(ts, dict) and isinstance(ts.get("$gte"), datetime) and isinstance(ts.get("$lt"), datetime):
            span_days = (ts["$lt"] - ts["$gte"]).days
            if span_days <= 10:
                return "week"
            if span_days <= 40:
                return "month"
            return "quarter"
    return "quarter"


def _report_date_range(pipeline: list) -> tuple | None:
    for stage in pipeline:
        if not isinstance(stage, dict) or "$match" not in stage:
            continue
        ts = stage["$match"].get("timestamp")
        if isinstance(ts, dict) and isinstance(ts.get("$gte"), datetime) and isinstance(ts.get("$lt"), datetime):
            return ts["$gte"], ts["$lt"]
    return None


def _fetch_day_type_counts(pipeline: list) -> list:
    """Deterministic (non-LLM) per-day, per-type counts over the report's
    own date range — used to build a monthly report's "each week broken
    down by type" section. Queried directly against Mongo rather than
    asked of the model: a 2-dimensional $group (date AND type at once) is
    exactly the kind of shape this 3B model has proven unreliable at
    reproducing this session, and the exact date range is already known
    from the report's own leading $match, so there's no need to risk it."""
    date_range = _report_date_range(pipeline)
    if date_range is None:
        return []
    gte, lt = date_range
    collection = get_alerts_collection()
    cursor = collection.aggregate([
        {"$match": {"alert_type": {"$ne": "NORMAL_OPERATION"}, "timestamp": {"$gte": gte, "$lt": lt}}},
        {"$group": {
            "_id": {"date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}}, "type": "$alert_type"},
            "count": {"$sum": 1},
        }},
    ])
    return _jsonify(list(cursor))


def _compute_week_type_rollup(day_type_rows: list) -> list:
    """Same day->week rollup as `_compute_week_rollup`, but keeping the
    alert_type dimension instead of collapsing it — produces one row per
    (week, type) pair, e.g. {"week": "July Week 1 (07/01-07/07)", "type":
    "FAST_INSPECTION", "count": 42}."""
    totals: dict[tuple, int] = {}
    for row in day_type_rows:
        id_val = row.get("_id") if isinstance(row, dict) else None
        if not isinstance(id_val, dict):
            continue
        date_str, alert_type, count = id_val.get("date"), id_val.get("type"), row.get("count")
        if not date_str or not alert_type or count is None:
            continue
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        week_num = ((d.day - 1) // 7) + 1
        key = (d.year, d.month, week_num, alert_type)
        totals[key] = totals.get(key, 0) + count

    return [
        {"week": _week_label(year, month, week_num), "type": alert_type, "count": count}
        for (year, month, week_num, alert_type), count in sorted(totals.items())
    ]


def _restructure_report(rows: list, pipeline: list) -> list:
    """Reshapes a report's $facet result into the section layout that
    matches its actual time span:
      - quarter: by_type, by_month, by_week (week totals only)
      - month:   by_type, by_week (each week broken down by type)
      - week:    by_type, by_day (true daily breakdown, not rolled up)
    The model always generates the SAME fixed $facet shape (total_alerts,
    by_type, by_month, by_day, avg_inspection_time) regardless of scope —
    reliably varying the $facet shape itself on request isn't something
    this model does consistently, so a single simple shape is generated
    every time, then Python deterministically reshapes it based on the
    real date span it was run over (see `_report_scope`)."""
    if len(rows) != 1 or not _is_facet_row(rows[0]):
        return rows

    facet = rows[0]
    by_day = facet.get("by_day")

    def _is_plain_date_row(r):
        if not isinstance(r, dict) or "type" in r:
            return False
        if "date" in r:
            return True
        return isinstance(r.get("_id"), str) and DATE_RE.match(r["_id"])

    flat_days = None
    if isinstance(by_day, list) and by_day:
        candidate = [_flatten_row(r) for r in by_day if isinstance(r, dict)]
        if candidate and all(_is_plain_date_row(r) for r in candidate):
            flat_days = candidate

    day_type_rows = None
    if flat_days is None:
        # The model's own facet didn't carry a usable by_day section -
        # found live generating by_month only for "give me a report for
        # June", silently dropping by_day (and with it, by_week) despite
        # the prompt's own fixed-5-section contract. Queried directly
        # here instead of trusting the model's facet shape at all -
        # _fetch_day_type_counts works off the report's own $match date
        # range, independent of whatever the facet happened to include.
        day_type_rows = _fetch_day_type_counts(pipeline)
        if not day_type_rows:
            return rows  # genuinely no data for this range - nothing to restructure
        totals: dict[str, int] = {}
        for row in day_type_rows:
            id_val = row.get("_id") if isinstance(row, dict) else None
            if not isinstance(id_val, dict):
                continue
            date_str, count = id_val.get("date"), row.get("count")
            if not date_str or count is None:
                continue
            totals[date_str] = totals.get(date_str, 0) + count
        flat_days = [{"date": d, "count": c} for d, c in sorted(totals.items())]

    scope = _report_scope(pipeline)
    new_facet = {k: v for k, v in facet.items() if k != "by_day"}
    new_facet.pop("by_month", None)

    if scope == "week":
        normalized_days = []
        for r in flat_days:
            date_val = r.get("date") or r.get("_id")
            count_val = r.get("count")
            if date_val is None or count_val is None:
                continue
            normalized_days.append({"date": date_val, "count": count_val})
        new_facet["by_day"] = sorted(normalized_days, key=lambda r: r["date"])
    elif scope == "month":
        if day_type_rows is None:
            day_type_rows = _fetch_day_type_counts(pipeline)
        new_facet["by_week"] = _compute_week_type_rollup(day_type_rows) if day_type_rows else _compute_week_rollup(flat_days)
    else:
        new_facet["by_month"] = facet.get("by_month", [])
        new_facet["by_week"] = _compute_week_rollup(flat_days)
    return [new_facet]


def _flatten_rows_for_output(rows: list) -> list:
    flattened = []
    for row in rows:
        if not isinstance(row, dict):
            flattened.append(row)
        elif _is_facet_row(row):
            flattened.append({
                name: [_flatten_row(r) if isinstance(r, dict) else r for r in sub_rows]
                for name, sub_rows in row.items()
            })
        else:
            flattened.append(_flatten_row(row))
    return flattened


# =========================================================
# STEP 3: rows -> natural language answer
#
# Unlike the old fixed-schema version, result shapes here are genuinely
# open-ended (a scalar count, a top-N list, a $facet multi-report object,
# arbitrary group-bys) so there's no finite set of templates that covers
# them. The LLM phrases this one — but only ever from the literal JSON
# rows just pulled from Mongo, with instructions to never compute or
# invent a number beyond what's present. The underlying numbers are still
# 100% DB-grounded; only the wording is model-generated.
# =========================================================

UNSUPPORTED_ANSWER = (
    "I can only answer questions about the alert log — this data has alert_type, zone, cloth_detected, "
    "inspection_time, and timestamp for each alert. Could you rephrase your question in those terms?"
)

GAP_QUESTION_ANSWER = (
    "I can tell you counts and breakdowns for specific days, but I can't currently scan for gaps — "
    "days with zero alerts — across a whole date range. Try asking about a specific day instead, "
    "e.g. \"how many alerts on March 3rd?\""
)

GREETING_ANSWER = (
    "Hello! I'm your factory safety alert assistant — ask me anything about the alert log: counts, "
    "trends, breakdowns, top days, even a full report. E.g. \"which day had the most hand touch alerts?\" "
    "or \"give me a quarterly report\"."
)

ERROR_ANSWER = (
    "I wasn't able to answer that — the query it needed didn't run cleanly against the database. "
    "Could you try rephrasing the question, maybe with a simpler or more specific time range or category?"
)


def _append_inherited_scope_note(answer: str, parsed: dict) -> str:
    """A follow-up that names no time period of its own silently
    inherits the previous turn's date scope (see
    `_resolve_date_range_with_source` / `_extract_inherited_date_range`)
    rather than losing it — right for a genuine follow-up, but found
    live to be actively misleading when the answer gives no hint that a
    narrower scope was applied at all: "show me the number of alerts
    for each category" right after "how many alerts this quarter"
    answered with quarter-only counts and no indication why they were
    smaller than the same question asked fresh, in a new chat, with no
    history. Appends a plain statement of the actual date range used
    whenever it came from that silent carryover, so the scope is never
    invisible."""
    date_range = parsed.get("_inherited_date_range")
    if not date_range:
        return answer
    start, end = date_range
    note = f" (using the same time period as your last question: {start.strftime('%b %d, %Y')} to {end.strftime('%b %d, %Y')})"
    return answer.rstrip() + note

ANSWER_SYSTEM_PROMPT = """You are a friendly, conversational factory safety data assistant — write like you're \
chatting with a colleague, not printing a database record. You are given the user's question and the EXACT \
rows a MongoDB query just returned — this is the complete ground truth. Write a short, natural-sounding answer \
using ONLY the numbers/values present in those rows. Do not invent, estimate, round differently, or compute \
anything beyond simple restatement of what's given. inspection_time is always in SECONDS. If a group key is an \
hour of day (0-23), phrase it as a one-hour time range (e.g. hour 13 is "between 1pm and 2pm"), never as a \
single instant like "around 1pm" and never as a duration or "seconds" — an hour value is a bucket, not a length \
of time. The rows given to you are ALWAYS non-empty — never say "no data was found" or anything like it, there \
is always real data to describe here. Never mention JSON, MongoDB, pipelines, field names like \
"_id", or queries — just answer plainly, the way a person would say it out loud. If the result has multiple \
named groups (e.g. from a report), mention EVERY group and its figure in flowing sentences — do not summarize \
by picking out only one or two and dropping the rest, and do not just list "key: value" pairs. Never reply \
with a single bare clause like "9 alerts on 2026-07-28." — restate what was actually asked as part of the \
sentence so it reads like a real answer, e.g. "The busiest day was 2026-07-28, with 9 fast inspection alerts \
recorded." Aim for at least one full sentence of context, not just a number and a label. If a date value is \
followed by a parenthetical day name, e.g. "2026-02-01 (Sunday)", that day name is real ground truth too — \
always state BOTH the date and the day name together (e.g. "on 2026-02-01, a Sunday"), never drop the day name \
or report just the bare date. Only attribute a value \
to a time period, zone, or category if a field in that exact row actually names it — if the question asked for \
a comparison (e.g. "this week vs last week") but the rows don't carry a field distinguishing the two periods, \
do not invent which period each number belongs to; describe only what the rows actually show. When listing \
several values (e.g. a breakdown by type), never call one of them "most common", "highest", "busiest", or any \
other superlative unless you have actually compared every number in the rows and confirmed it truly is the \
largest — a wrong superlative (calling out a smaller number as the top one) is worse than no superlative at \
all, so when in doubt just state the figures plainly without ranking language."""


def _cap_facet_section_for_prompt(sub_rows: list) -> list:
    """A $facet result is always exactly ONE row, so the top-level `rows[:25]` \
    cap in `_build_answer_prompt` does nothing to limit it — a "by_day" \
    section spanning a quarter (~90 dates x 3 types) serializes whole into \
    the prompt regardless, which is exactly what blew a prompt out to \
    thousands of tokens and OOM'd the 4GB GPU. That section is always \
    rendered deterministically anyway (see `_is_day_type_breakdown`), so \
    the phrasing model doesn't need to see more than a handful of its rows \
    to write a reasonable one-line mention of it."""
    flat_sub = [_flatten_row(r) for r in sub_rows if isinstance(r, dict)]
    if _is_day_type_breakdown(flat_sub) or _is_week_type_breakdown(flat_sub):
        return sub_rows[:5]
    return sub_rows[:25]


def _with_weekday_suffix(node):
    """Appends the weekday name to any bare YYYY-MM-DD date string
    ("2026-02-01" -> "2026-02-01 (Sunday)") before rows reach the
    answer-phrasing prompt. The phrasing model is told to restate exactly
    what's in the rows and never compute anything itself — asking it to
    work out that 2026-02-01 was a Sunday is exactly the kind of
    computation not to trust an LLM with, so the weekday is computed here
    deterministically and handed to it as part of the date string itself.
    That's what lets a "which DATE had the fewest X" answer naturally
    name both the date and the day in the model's own generated sentence,
    without needing to fall back to the deterministic template."""
    if isinstance(node, str) and DATE_RE.match(node):
        weekday = _weekday_name(node)
        return f"{node} ({weekday})" if weekday else node
    if isinstance(node, dict):
        return {k: _with_weekday_suffix(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_with_weekday_suffix(v) for v in node]
    return node


def _build_answer_prompt(question: str, rows: list) -> str:
    # Cap what's sent to the phrasing prompt (latency + memory), while the
    # full (already-capped-at-200) row set still goes to the frontend table.
    # Narration text is stripped entirely — it's for display, not for the
    # phrasing model to reason about, and Tamil script in particular
    # tokenizes so inefficiently on this model (several tokens per
    # character) that a handful of rows with narration_ta blew a 40-row
    # preview out to ~17,000 tokens and OOM'd the 4GB GPU outright.
    if len(rows) == 1 and _is_facet_row(rows[0]):
        preview_rows = [{
            name: [_strip_narration(r) for r in _cap_facet_section_for_prompt(sub_rows)]
            for name, sub_rows in rows[0].items()
        }]
        truncated_note = ""
    else:
        preview_rows = [_strip_narration(r) for r in rows[:25]]
        truncated_note = f" (showing first {len(preview_rows)} of {len(rows)} rows)" if len(rows) > 25 else ""
    preview_rows = _with_weekday_suffix(preview_rows)
    return (
        f"Question: {question}\n\n"
        f"Query result rows{truncated_note} (ground truth, use exactly): {json.dumps(preview_rows)}"
    )


def _finalize_answer(rows: list, answer: str, question: str = "", truncated: bool = False) -> str:
    """Applies the same deterministic correction pass regardless of
    whether `answer` came from a one-shot `_chat` call or was assembled
    from a token stream — the safety net (never let a bare/incomplete/
    hallucinated-"no data" answer stand) has to apply identically either
    way, so both paths funnel through here rather than duplicating it."""
    all_scalar = all(_is_scalar_row(r) for r in rows)
    is_facet = len(rows) == 1 and _is_facet_row(rows[0])

    # Greedy decoding still occasionally collapses to a bare/incomplete
    # answer that drops some of the actual result values — a lone "3" for
    # a count, a bare "2026-07-18" with no count attached, a list of dates
    # with every count silently dropped, or (for $facet "report" queries)
    # only one or two of several named sections actually getting
    # mentioned. `_is_bare_answer` catches the degenerate case where a
    # value IS technically present but with no sentence around it (a raw
    # "18" — the omit-check alone would consider that fine, since 18 does
    # appear in "18").
    if is_facet:
        # A wrong superlative ("April had the highest count" when June
        # actually did) is fixed IN PLACE first — surgically replacing
        # just that sentence — rather than treated as a reason to throw
        # away the model's entire natural-sounding paragraph. Only if
        # something else is ALSO wrong (bare, hallucinated "no data",
        # missing a single-row value with no table backing it up) does
        # the whole thing fall back to the fully deterministic version.
        answer = _correct_wrong_superlative_sentences(answer, rows[0])
        answer = _append_missing_single_row_sections(answer, rows[0])
        answer = _append_missing_multi_row_sections(answer, rows[0])

        # A day-by-day, week-bucketed breakdown is a specific requested
        # FORMAT, not just a correctness bar — asking the model to
        # reliably reproduce "Week 1 (Jul 1-7): Jul 1 - Fast Inspection:
        # 5, ..." in prose isn't realistic (the same way it wasn't
        # realistic to expect it to enumerate every row of a 12-row
        # breakdown earlier). Whenever a $facet section IS a day+type
        # breakdown, that section's text is always built deterministically
        # rather than trusted from the model, regardless of whether the
        # rest of its answer was otherwise fine.
        has_day_breakdown = any(
            _is_day_type_breakdown([_flatten_row(r) for r in sub_rows if isinstance(r, dict)])
            or _is_week_type_breakdown([_flatten_row(r) for r in sub_rows if isinstance(r, dict)])
            for sub_rows in rows[0].values()
        )
        all_facet_rows = [
            _flatten_row(r) for sub_rows in rows[0].values() for r in sub_rows if isinstance(r, dict)
        ]
        invents_date = _answer_invents_date(answer, all_facet_rows)
        misattributes_month = _answer_misattributes_type_total_to_month(answer, rows[0])
        if _is_bare_answer(answer) or NO_DATA_CLAIM_RE.search(answer) or _facet_omits_values(answer, rows[0]) or has_day_breakdown or invents_date or misattributes_month:
            answer = _facet_to_sentence(rows[0])
    elif all_scalar:
        # Requiring every single value to be restated verbatim is only
        # realistic for a handful of rows — a natural sentence genuinely
        # can mention "9 on Tuesday, 5 on Monday". Demanding the same for
        # a 12-row week-by-type breakdown just forces the model to fail
        # the check every time (it summarizes, as any person would) and
        # permanently falls back to a "key: value, key: value, ..." dump
        # — exactly the robotic listing style this was built to avoid.
        # The full breakdown already renders as a table below the message
        # for these larger results, so the text only needs to not be
        # bare/wrong, not exhaustively complete.
        omits_values = len(rows) <= 6 and any(_answer_omits_row_values(answer, r) for r in rows)
        if _is_bare_answer(answer) or omits_values or _answer_has_wrong_superlative(answer, rows) or _answer_invents_alert_type(answer, rows, question) or _answer_invents_date(answer, rows):
            answer = _rows_to_sentence(rows, question=question)
        elif NO_DATA_CLAIM_RE.search(answer):
            answer = "There's data for this — see the breakdown below."
    elif NO_DATA_CLAIM_RE.search(answer):
        # Rows are confirmed non-empty at this point (the true-empty case
        # already returned above), and not a scalar/facet shape we know
        # how to render deterministically — still better to say something
        # generically true than let a false "nothing found" stand when
        # the table right below it proves otherwise.
        answer = "There's data for this — see the breakdown below."

    answer = _ensure_weekday_after_dates(answer)
    if truncated:
        # execute_pipeline hit the hard DEFAULT_RESULT_LIMIT-row cap and
        # stopped pulling further rows - said explicitly rather than
        # silently handing over a partial result that reads as complete.
        answer = answer.rstrip()
        if answer and not answer.endswith((".", "!", "?")):
            answer += "."
        answer += (f" (Showing the first {DEFAULT_RESULT_LIMIT} rows - "
                   "there are more matching this query; narrow the time range "
                   "or add a filter to see the rest.)")
    return answer


def _ensure_weekday_after_dates(answer: str) -> str:
    """Deterministic safety net for always pairing a date with its day
    name: `_build_answer_prompt` already hands the model a "2026-02-01
    (Sunday)"-style string and instructs it to keep both, but a small
    model doesn't always comply. Finds any bare YYYY-MM-DD date in the
    final answer text not already followed by a parenthetical/day name,
    and appends the weekday computed directly from that same date —
    never invented, never dependent on the model having gotten it right."""
    def _replace(match: re.Match) -> str:
        date_str = match.group(0)
        weekday = _weekday_name(date_str)
        if not weekday:
            return date_str
        tail = answer[match.end():match.end() + 20]
        if weekday in tail or re.match(r"\s*\(", tail):
            return date_str  # already has a parenthetical / day name right after it
        return f"{date_str} ({weekday})"

    return re.sub(r"\b\d{4}-\d{2}-\d{2}\b", _replace, answer)


def phrase_answer(question: str, result: dict) -> str:
    intent = result.get("intent")

    if intent == "unsupported":
        return UNSUPPORTED_ANSWER
    if intent == "gap_unsupported":
        return GAP_QUESTION_ANSWER
    if intent == "greeting":
        return GREETING_ANSWER
    if intent == "error":
        print(f"[llm_query] returning generic error answer to user; underlying DB error: {result.get('error')}")
        return ERROR_ANSWER

    rows = result["rows"]
    if not rows:
        return "No matching data found for that."

    # rows are already flattened (nested _id group keys resolved) by
    # execute_pipeline, so the frontend table and this phrasing step see
    # the identical shape.
    user_prompt = _build_answer_prompt(question, rows)
    answer = _chat(ANSWER_SYSTEM_PROMPT, user_prompt, max_new_tokens=200)
    return _finalize_answer(rows, answer, question, truncated=result.get("truncated", False))


NO_DATA_CLAIM_RE = re.compile(r"no (matching )?data|no results|nothing (was )?found", re.IGNORECASE)

ALERT_TYPES = {"FAST_INSPECTION", "HAND_TOUCH", "MISSING_CLEANING", "NORMAL_OPERATION"}
VALUE_KEYS = ("total", "count", "avg", "sum")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
WEEKDAY_NAMES = ["", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


def _flatten_row(row: dict) -> dict:
    """A group-by-hour/dayOfWeek pipeline produces {"_id": {"hour": 7},
    "count": 2} — a nested dict inside _id that neither the scalar-row nor
    facet-row detection recognized, so it fell through with zero safety
    net (the bug behind "peak time" returning a bare, unitless "7").
    Flattening {"hour": 7} into the row itself makes it scalar-shaped like
    every other group-by result."""
    if not isinstance(row, dict):
        return row
    id_val = row.get("_id")
    if isinstance(id_val, dict) and id_val:
        # Also handles compound group keys with more than one named field
        # (e.g. grouping by hour AND zone at once), not just the single-key
        # case — merge every key _id carries, not just the first.
        flat = {k: v for k, v in row.items() if k != "_id"}
        flat.update(id_val)
        return flat
    return row


def _is_scalar_row(row: dict) -> bool:
    return isinstance(row, dict) and all(
        isinstance(v, (str, int, float, bool, type(None))) for v in row.values()
    )


def _is_facet_row(row: dict) -> bool:
    return isinstance(row, dict) and len(row) > 0 and all(isinstance(v, list) for v in row.values())


def _is_bare_answer(answer: str) -> bool:
    # A well-formed sentence has at least a couple of real words around
    # the number(s) — "There were 18 alerts." A degenerate "18" or
    # "2026-07-18" with nothing else doesn't.
    return len(re.findall(r"[A-Za-z]+", answer)) < 2


def _answer_omits_row_values(answer: str, row: dict) -> bool:
    # Plain substring matching gives false positives — e.g. a count of 2
    # is "found" inside the date string "2026-07-23" even though the
    # answer never actually states the count. Word-boundary matching
    # avoids treating a value as present just because its digits happen
    # to occur inside a longer, unrelated number/string.
    for v in row.values():
        if v is None:
            continue
        # A large number is legitimately written with thousands
        # separators ("20,138" for 20138) — that's normal, readable
        # prose, not a different number, so both forms count as present.
        candidates = [str(v)]
        if isinstance(v, (int, float)) and not isinstance(v, bool) and abs(v) >= 1000:
            candidates.append(f"{v:,}")
        if any(re.search(rf"\b{re.escape(c)}\b", answer) for c in candidates):
            continue
        # An alert-type enum value ("MISSING_CLEANING") is legitimately
        # written in prose as "missing cleaning" — that's the model
        # following the "talk like a person, not a database record"
        # instruction correctly, not an omission. Only flag it missing if
        # neither the literal nor the humanized form shows up anywhere.
        if isinstance(v, str) and v.upper() in ALERT_TYPES:
            humanized = v.replace("_", " ")
            if re.search(rf"\b{re.escape(humanized)}\b", answer, re.IGNORECASE):
                continue
        return True
    return False


def _answer_invents_date(answer: str, rows: list) -> bool:
    """Same failure mode as `_answer_invents_alert_type`, for calendar \
    dates instead of alert types: observed live on a report whose "by_day" \
    section had been rolled up into "by_week" (so no row anywhere carries \
    a specific date any more) — the model still said "the busiest day was \
    2026-07-28", inventing both a specific date AND misattributing the \
    whole month's total to it. Any YYYY-MM-DD the answer names must \
    actually appear as a value in the rows somewhere, or it's fabricated."""
    present_dates = {
        v for row in rows if isinstance(row, dict)
        for v in row.values() if isinstance(v, str) and DATE_RE.match(v)
    }
    for mentioned in re.findall(r"\b\d{4}-\d{2}-\d{2}\b", answer):
        if mentioned not in present_dates:
            return True
    return False


def _answer_misattributes_type_total_to_month(answer: str, facet_row: dict) -> bool:
    """Catches a hallucination specific to quarterly reports: attaching a
    "by_type" total (which spans the WHOLE report date range — e.g.
    FAST_INSPECTION: 7148 across May-August) to one specific month —
    observed live as "Fast inspection alerts... totaling 7148 instances
    in May alone." The report has no month+type breakdown row anywhere
    to support that (by_type totals are never month-specific), so any
    sentence that pairs a real by_type total with a real by_month month
    name is fabricating a scope the data doesn't have. A sentence
    correctly describing a by_month row itself (e.g. "May had the
    highest count with 5,578 alerts") is unaffected, since 5,578 isn't a
    by_type total."""
    by_type_rows = facet_row.get("by_type") or []
    by_month_rows = facet_row.get("by_month") or []
    if not isinstance(by_type_rows, list) or not isinstance(by_month_rows, list):
        return False

    type_totals = {
        r["count"] for r in by_type_rows
        if isinstance(r, dict) and isinstance(r.get("count"), (int, float)) and not isinstance(r.get("count"), bool)
    }
    month_names = set()
    for r in by_month_rows:
        if not isinstance(r, dict):
            continue
        month_str = r.get("_id") if isinstance(r.get("_id"), str) else r.get("date")
        if isinstance(month_str, str) and MONTH_RE.match(month_str):
            try:
                month_names.add(datetime.strptime(month_str, "%Y-%m").strftime("%B"))
            except ValueError:
                pass
    if not type_totals or not month_names:
        return False

    for sentence in re.split(r"(?<=[.!?])\s+", answer):
        if not any(re.search(rf"\b{m}\b", sentence, re.IGNORECASE) for m in month_names):
            continue
        for total in type_totals:
            if re.search(rf"\b{int(total):,}\b", sentence) or re.search(rf"\b{int(total)}\b", sentence):
                return True
    return False


def _answer_invents_alert_type(answer: str, rows: list, question: str = "") -> bool:
    """Catches the mirror-image bug to `_answer_omits_row_values`: instead \
    of dropping a real value, the model INVENTS one that was never in the \
    data at all — observed live on a plain total-count query (no \
    alert_type field anywhere in the row) answered as "146 fast \
    inspection alerts", fabricating a category the pipeline never \
    filtered or grouped by. If none of the rows carry a given alert type \
    at all, the answer must not name it.

    EXCEPTION: if the question itself named exactly one alert type \
    ("which day had the most FAST INSPECTION alerts"), the pipeline's own \
    $match already silently filters every row to that type — it's true \
    of every row without needing to be a literal field in the output, so \
    the answer restating it ("100 fast inspection alerts") is correctly \
    contextualizing the number, not inventing one. This was rejecting a \
    genuinely good answer and replacing it with a worse bare fallback."""
    present_types = {
        v.upper() for row in rows if isinstance(row, dict)
        for v in row.values() if isinstance(v, str) and v.upper() in ALERT_TYPES
    }
    asked_type = _mentioned_alert_type(question) if question else None
    if asked_type:
        present_types.add(asked_type)
    for alert_type in ALERT_TYPES:
        if alert_type == "NORMAL_OPERATION" or alert_type in present_types:
            continue
        for candidate in (alert_type, alert_type.replace("_", " ")):
            if re.search(rf"\b{re.escape(candidate)}\b", answer, re.IGNORECASE):
                return True
    return False


SUPERLATIVE_RE = re.compile(r"\b(most common|most|highest|busiest|peak|top|least|lowest|fewest)\b", re.IGNORECASE)


def _label_candidates(label_val: str) -> list:
    """The model paraphrases labels rather than echoing the raw value
    verbatim — "FAST_INSPECTION" becomes "fast inspection", "2026-04"
    becomes "April" — so matching a label mention in prose needs both the
    raw and humanized forms, or a correct paraphrase looks like a miss."""
    candidates = [label_val]
    if label_val.upper() in ALERT_TYPES:
        candidates.append(label_val.replace("_", " "))
    if MONTH_RE.match(label_val):
        try:
            candidates.append(datetime.strptime(label_val, "%Y-%m").strftime("%B"))
        except ValueError:
            pass
    return candidates


def _answer_has_wrong_superlative(answer: str, rows: list) -> bool:
    """Catches a specific hallucination observed live: "HAND TOUCH alerts \
    were most common with 7 instances, followed by FAST INSPECTION with 9 \
    instances" — self-contradicting, since 9 > 7. The model gets the \
    individual numbers right (so `_answer_omits_row_values` sees nothing \
    wrong) but attaches a superlative to the wrong one. Splits the answer \
    into sentences and checks each sentence containing a superlative word: \
    if it also names a row whose value ISN'T the actual max (or min, for \
    "least"/"lowest"/"fewest") among all rows, the claim is wrong."""
    numeric_rows = [
        (v, k, r) for r in rows
        for k, v in r.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    if len(numeric_rows) < 2:
        return False
    max_val = max(v for v, _, _ in numeric_rows)
    min_val = min(v for v, _, _ in numeric_rows)

    for sentence in re.split(r"(?<=[.!?])\s+", answer):
        match = SUPERLATIVE_RE.search(sentence)
        if not match:
            continue
        is_low = match.group(1).lower() in ("least", "lowest", "fewest")
        target = min_val if is_low else max_val
        for value, _key, row in numeric_rows:
            if value == target:
                continue
            for label_val in row.values():
                if not isinstance(label_val, str):
                    continue
                if any(re.search(rf"\b{re.escape(c)}\b", sentence, re.IGNORECASE) for c in _label_candidates(label_val)):
                    return True
    return False


def _corrective_clause(top_row: dict, is_low: bool) -> str:
    """Builds a standalone corrective sentence ("June actually had the \
    highest count, with 5,205 alerts.") for the specific row that really \
    is the max/min — used in place of a wrong sentence, so it needs to \
    read as a complete, clearly-corrective statement on its own rather \
    than the bare fragment `_row_to_clause` produces for use inline in a \
    longer list."""
    value_key, value, labels = _split_value_and_labels(top_row)
    display_value = f"{value:,}" if isinstance(value, (int, float)) and not isinstance(value, bool) and abs(value) >= 1000 else value
    value_phrase = f"{display_value} seconds" if value_key == "avg" else f"{display_value} {'alert' if value == 1 else 'alerts'}"
    superlative = "lowest" if is_low else "highest"

    if not labels:
        return f"The {superlative} was {value_phrase}."

    label_key, label_val = labels[0]
    if isinstance(label_val, str) and label_val.upper() in ALERT_TYPES:
        subject = label_val.replace("_", " ").title()
    else:
        subject = _describe_label(label_key, label_val)
        for prefix in ("on ", "in ", "between "):
            if subject.startswith(prefix):
                subject = subject[len(prefix):]
                break
    return f"{subject} actually had the {superlative} count, with {value_phrase}."


def _correct_wrong_superlative_sentences(answer: str, facet_row: dict) -> str:
    """A wrong superlative claim ("April had the highest count" when June
    actually did) is confined to ONE sentence about ONE section's data —
    throwing away the model's entire natural-sounding paragraph over a
    single bad sentence is a worse trade than just fixing that sentence.
    Checked per $facet section (not all sections' numbers pooled
    together, unlike `_answer_has_wrong_superlative`'s coarser check) so
    the replacement clause is built from the right data. Sections with
    only one row are skipped — nothing to be "most" or "least" of."""
    sentences = re.split(r"(?<=[.!?])\s+", answer)

    for sub_rows in facet_row.values():
        flat_sub = [_flatten_row(r) for r in sub_rows if isinstance(r, dict)]
        numeric_rows = [
            (v, r) for r in flat_sub if isinstance(r, dict)
            for v in r.values()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        if len(numeric_rows) < 2:
            continue
        max_val = max(v for v, _ in numeric_rows)
        min_val = min(v for v, _ in numeric_rows)

        for i, sentence in enumerate(sentences):
            match = SUPERLATIVE_RE.search(sentence)
            if not match:
                continue
            is_low = match.group(1).lower() in ("least", "lowest", "fewest")
            target = min_val if is_low else max_val
            wrong = any(
                isinstance(label_val, str)
                and any(re.search(rf"\b{re.escape(c)}\b", sentence, re.IGNORECASE) for c in _label_candidates(label_val))
                for value, row in numeric_rows if value != target
                for label_val in row.values()
            )
            if wrong:
                top_row = next(row for value, row in numeric_rows if value == target)
                sentences[i] = _corrective_clause(top_row, is_low)

    return " ".join(sentences)


# The word a section's name promises the reader. A report that never says
# "week" anywhere has not reported the weekly view, whatever else it got
# right - see _append_missing_multi_row_sections.
_SECTION_KEYWORDS = {
    "by_week": ("week", "weekly"),
    "by_month": ("month", "monthly"),
    "by_day": ("day", "daily", "date"),
    "by_type": ("type", "category"),
    "by_hour": ("hour", "hourly"),
}


def _section_is_mentioned(answer: str, name: str, flat_sub: list) -> bool:
    low = answer.lower()
    if any(kw in low for kw in _SECTION_KEYWORDS.get(name, (name.replace("_", " "),))):
        return True
    # Fall back to the section's own labels, for a section name with no
    # entry above ("by_shift", say). Matching on labels alone is not
    # enough on its own: the model writes "July" where the row says
    # "2026-07", so a mentioned section routinely carries none of its
    # literal label strings.
    for row in flat_sub:
        _, _, labels = _split_value_and_labels(row)
        for _, label_val in labels:
            if isinstance(label_val, str) and len(label_val) > 2 and label_val.lower() in low:
                return True
    return False


def _append_missing_multi_row_sections(answer: str, facet_row: dict) -> str:
    """Append a one-line mention for any multi-row section the answer
    never refers to at all.

    `_facet_omits_values` deliberately exempts multi-row sections from the
    completeness check, and rightly so: a report is meant to highlight
    rather than enumerate, and every such section already renders as a
    table below the message. But that exemption was doing double duty -
    it also let a section go entirely unmentioned. Observed live on "Give
    me a quarterly report": the $facet returned a correct 17-row by_week
    section, `_restructure_report` shaped it correctly, and the model's
    paragraph then never used the word "week" once (it spent the space
    repeating "July actually had the highest count" twice instead). The
    weekly view was in the payload and in the table, but as far as the
    prose was concerned the report simply did not have one.

    Appending a highlight, rather than the full enumeration
    `_facet_to_sentence` would produce, keeps faith with the highlight-
    not-enumerate design: 17 weeks spelled out would bury the paragraph
    it is being added to. The peak is computed here in Python precisely
    because a superlative is the one thing the phrasing model must never
    be trusted to work out for itself (cf.
    `_correct_wrong_superlative_sentences`).
    """
    additions = []
    for name, sub_rows in facet_row.items():
        if not isinstance(sub_rows, list) or len(sub_rows) < 2:
            continue
        flat_sub = [_flatten_row(r) for r in sub_rows if isinstance(r, dict)]
        flat_sub = [r for r in flat_sub if isinstance(r, dict)]
        if not flat_sub or _section_is_mentioned(answer, name, flat_sub):
            continue

        # A day/type or week/type block is a requested FORMAT, not a
        # highlight - those already force the fully deterministic
        # rendering upstream, so never half-summarise one here.
        if _is_day_type_breakdown(flat_sub) or _is_week_type_breakdown(flat_sub):
            continue

        scored = []
        for r in flat_sub:
            value_key, value, labels = _split_value_and_labels(r)
            if value_key is None or not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            if labels:
                scored.append((value, labels[0][1]))
        if not scored:
            continue

        top_value, top_label = max(scored, key=lambda p: p[0])
        lead_in = FACET_SECTION_LEAD_INS.get(name) or f"{name.replace('_', ' ')}, "
        additions.append(
            f"{lead_in}the busiest was {top_label} with {top_value:,} "
            f"({len(scored)} in total - see the breakdown below)."
        )

    if not additions:
        return answer

    answer = answer.rstrip()
    if answer and not answer.endswith((".", "!", "?")):
        answer += "."
    return (answer + " " + " ".join(additions)).strip()


def _append_missing_single_row_sections(answer: str, facet_row: dict) -> str:
    """A single-row section (total_alerts, avg_inspection_time, ...) has
    no table backing it up — if the model's natural summary skips
    mentioning it (observed live: it covered totals/types/months/weeks
    but dropped the average inspection time entirely), that number is
    gone from the response altogether, not just less detailed. Rather
    than discard an otherwise-good paragraph over one missing fact, the
    missing sentence is appended once at the end."""
    additions = []
    for sub_rows in facet_row.values():
        if len(sub_rows) != 1:
            continue
        flat = _flatten_row(sub_rows[0]) if isinstance(sub_rows[0], dict) else sub_rows[0]
        if not isinstance(flat, dict) or not _is_scalar_row(flat):
            continue
        if _answer_omits_row_values(answer, flat):
            # `_rows_to_sentence` (not `_row_to_clause` directly) so a
            # bare value like "10.58 seconds" gets its "The average was
            # ..." lead-in — appended as a fragment with no subject reads
            # just as database-record-ish as the thing this was meant to
            # avoid.
            additions.append(_rows_to_sentence([flat]))

    if not additions:
        return answer

    answer = answer.rstrip()
    if answer and not answer.endswith((".", "!", "?")):
        answer += "."
    return (answer + " " + " ".join(additions)).strip()


def _facet_omits_values(answer: str, row: dict) -> bool:
    for sub_rows in row.values():
        # A report's whole point is a natural, analytical SUMMARY —
        # "FAST_INSPECTION was the most common type" reads far better than
        # dutifully naming all three types and every count, and the model
        # was already writing exactly that well on its own (confirmed live
        # — it just kept getting overridden). Any section with more than
        # one row already gets its own table right below the message, so
        # the text's job is to highlight, not to enumerate — completeness
        # is only actually required for a single-row section (total,
        # average, ...), since nothing else on screen carries that number.
        if len(sub_rows) > 1:
            continue
        for sub_row in sub_rows:
            flat = _flatten_row(sub_row) if isinstance(sub_row, dict) else sub_row
            if isinstance(flat, dict) and _is_scalar_row(flat) and _answer_omits_row_values(answer, flat):
                return True
    return False


# ---------------------------------------------------------------------
# Deterministic fallback answers — only used when the model's own
# phrasing fails one of the checks above. Built as actual sentences, not
# "key: value" dumps, since a database-record-looking fallback would be
# just as jarring as the bug it's replacing.
# ---------------------------------------------------------------------

def _split_value_and_labels(row: dict):
    """Splits a scalar row into (value_key, value, [(label_key, label_val), ...]).
    value_key is the metric field (total/count/avg/sum) if recognizable,
    else the first numeric field — everything else describes what that
    number belongs to (a date, an alert type, an hour of day, ...)."""
    items = [(k, v) for k, v in row.items() if v is not None]
    value_key = next((k for k, v in items if k in VALUE_KEYS), None)
    if value_key is None:
        value_key = next(
            (k for k, v in items if isinstance(v, (int, float)) and not isinstance(v, bool)), None
        )
    labels = [(k, v) for k, v in items if k != value_key]
    return value_key, (row.get(value_key) if value_key else None), labels


def _format_hour_range(h: int) -> str:
    def fmt(hour24):
        hour24 = hour24 % 24
        h12 = hour24 % 12 or 12
        return f"{h12}{'am' if hour24 < 12 else 'pm'}"
    return f"{fmt(h)} and {fmt(h + 1)}"


def _camel_to_words(key: str) -> str:
    """The model occasionally invents its own field names via a stray \
    $replaceRoot (e.g. renaming an hour group's output to "hourPeak") \
    despite being told not to — when that happens, a raw key like \
    "hourPeak" displayed as-is reads as a single run-together word \
    ("HourPeak 16: 47 alerts."), not a label. This is the fallback \
    formatter's last line of defense: split camelCase/snake_case into \
    real words so ANY unanticipated key name still degrades to \
    something readable instead of garbled."""
    return re.sub(r"(?<!^)(?=[A-Z])", " ", key).replace("_", " ").replace(".", " ").strip().lower()


def _weekday_name(date_str: str) -> str | None:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%A")
    except ValueError:
        return None


def _describe_label(key: str, value) -> str:
    key_lower = key.lower()
    if isinstance(value, str) and DATE_RE.match(value):
        weekday = _weekday_name(value)
        return f"on {value} ({weekday})" if weekday else f"on {value}"
    if isinstance(value, str) and MONTH_RE.match(value):
        try:
            return f"in {datetime.strptime(value, '%Y-%m').strftime('%B')}"
        except ValueError:
            return f"in {value}"
    if isinstance(value, str) and value.upper() in ALERT_TYPES:
        return value.replace("_", " ").title()
    # Substring match (not exact) so a model-invented variant like
    # "hourPeak" or "peak_hour" still gets recognized as an hour value,
    # not just the exact key "hour".
    if "hour" in key_lower and isinstance(value, (int, float)):
        return f"between {_format_hour_range(int(value))}"
    if "dayofweek" in key_lower.replace(" ", "").replace("_", "") and isinstance(value, (int, float)) and 1 <= int(value) <= 7:
        return f"on {WEEKDAY_NAMES[int(value)]}s"
    if key_lower == "zone" or (isinstance(value, str) and _normalize_zone_value(value)):
        # Value-based, not just key-based: grouping directly by $zone
        # ({"$group": {"_id": "$zone", ...}}) leaves the key as Mongo's
        # own generic "_id", not "zone" - found live producing "Id FQC
        # Station 1: 307,755 alerts." (the raw key "_id" leaking into the
        # sentence as the literal word "Id"). A zone value is recognized
        # by matching it against the known zone set directly, the same
        # way an alert-type value is recognized above regardless of what
        # its key is called.
        return f"in {_normalize_zone_value(value) or value}"
    if key_lower == "week":
        return f"in {value}"
    return f"{_camel_to_words(key)} {value}"


def _row_to_clause(row: dict) -> str:
    value_key, value, labels = _split_value_and_labels(row)

    if value_key is None:
        # A bare single-field group row ({"_id": "FAST_INSPECTION"}, from
        # grouping by a field with NO accumulator - the "distinct values"
        # shape) has no number to anchor on at all, so the generic
        # "key value" join below rendered it as the literal, unreadable
        # "id FAST_INSPECTION". If the one value present is a recognized
        # alert type or zone, render just that - the label already says
        # everything the row means, the raw key name adds nothing.
        if len(row) == 1:
            (_, only_value), = row.items()
            if isinstance(only_value, str) and only_value.upper() in ALERT_TYPES:
                return only_value.replace("_", " ").title()
            if isinstance(only_value, str) and _normalize_zone_value(only_value):
                return _normalize_zone_value(only_value)
        return ", ".join(f"{k.replace('_', ' ').lstrip()} {v}" for k, v in row.items() if v is not None)

    display_value = f"{value:,}" if isinstance(value, (int, float)) and not isinstance(value, bool) and abs(value) >= 1000 else value
    value_phrase = f"{display_value} seconds" if value_key == "avg" else f"{display_value} {'alert' if value == 1 else 'alerts'}"

    if not labels:
        return value_phrase

    # A compound group key (e.g. grouped by day AND type) carries more
    # than one label — describing only labels[0] silently dropped every
    # other dimension, so two different alert types in the same week both
    # rendered as the identical, now-ambiguous "1 alert in 2026-26". A
    # parenthetical dump ("65 alerts (on 2026-07-20, Hand Touch)") fixed
    # the missing-information bug but still reads like a database record,
    # not a sentence — if one label is an alert type, use it as the
    # sentence's subject ("Hand Touch had 65 alerts on 2026-07-20") the
    # same way the single-label case below already does.
    if len(labels) > 1:
        type_label = next((pair for pair in labels if isinstance(pair[1], str) and pair[1].upper() in ALERT_TYPES), None)
        other_labels = [pair for pair in labels if pair != type_label] if type_label else labels
        other_descs = [_describe_label(k, v) for k, v in other_labels]
        if len(other_descs) > 2:
            trailing = ", ".join(other_descs[:-1]) + f", and {other_descs[-1]}"
        else:
            trailing = " and ".join(other_descs)
        if type_label:
            subject = _describe_label(*type_label)
            return f"{subject} had {value_phrase} {trailing}".rstrip() if trailing else f"{subject} had {value_phrase}"
        return f"{value_phrase} {trailing}".rstrip() if trailing else value_phrase

    label_key, label_val = labels[0]
    desc = _describe_label(label_key, label_val)

    if isinstance(label_val, str) and label_val.upper() in ALERT_TYPES:
        return f"{desc} had {value_phrase}"
    if desc.startswith(("on ", "in ", "around ", "between ")):
        return f"{value_phrase} {desc}"
    return f"{desc}: {value_phrase}"


_BARE_VALUE_RE = re.compile(r"^[\d,]+(\.\d+)? (alerts?|seconds)$")

TIME_PHRASE_PATTERNS = [
    (re.compile(r"\byesterday\b", re.IGNORECASE), "yesterday"),
    (re.compile(r"\btoday\b", re.IGNORECASE), "today"),
    (re.compile(r"\bthis week\b", re.IGNORECASE), "this week"),
    (re.compile(r"\blast week\b", re.IGNORECASE), "last week"),
    (re.compile(r"\bthis month\b", re.IGNORECASE), "this month"),
    (re.compile(r"\blast month\b", re.IGNORECASE), "last month"),
    (re.compile(r"\bthis year\b", re.IGNORECASE), "this year"),
    (re.compile(r"\b(?:last|past)\s+(\d+)\s+days?\b", re.IGNORECASE), None),
    (re.compile(r"\b(?:last|past)\s+(\d+)\s+hours?\b", re.IGNORECASE), None),
]


def _extract_time_phrase(question: str) -> str | None:
    """A row with no label at all (a plain $count/$avg — nothing for
    `_describe_label` to work with) has no context of its own to restate;
    the only place "yesterday" or "this week" exists is the question that
    was asked. Reused so a bare "146 alerts" becomes "146 alerts recorded
    yesterday" instead of stripping the time frame out entirely.

    An explicit "between Xam/pm and Yam/pm" hour range is checked first
    and combined with whatever day-level phrase is also present ("between
    2pm and 4pm yesterday"), rather than the day phrase alone silently
    dropping the hour range the question actually asked about."""
    if not question:
        return None

    day_phrase = None
    for pattern, phrase in TIME_PHRASE_PATTERNS:
        match = pattern.search(question)
        if match:
            # The named phrases ("yesterday", "this week", ...) already
            # read naturally after "recorded" on their own; the two
            # "last/past N days/hours" patterns don't carry a fixed
            # phrase and fall back to the raw matched text, which needs
            # "in the" in front of it to read as a real sentence
            # ("recorded in the last 7 days", not "recorded last 7 days").
            day_phrase = phrase if phrase else f"in the {match.group(0).lower()}"
            break

    hour_match = _HOUR_RANGE_RE.search(question)
    if hour_match:
        hour_text = hour_match.group(0).lower()
        return f"{hour_text} {day_phrase}" if day_phrase else hour_text
    if day_phrase:
        return day_phrase
    return None


def _rows_to_sentence(rows: list, limit: int = 20, question: str = "") -> str:
    clauses = [_row_to_clause(r) for r in rows[:limit]]
    if len(clauses) == 1:
        if _BARE_VALUE_RE.match(clauses[0]):
            # A genuinely bare clause (no label at all — a plain total
            # count or average) reads like a database record without
            # SOME lead-in ("146 alerts." isn't a sentence). Anything
            # `_row_to_clause` already gave a subject/verb to (e.g. "Hand
            # Touch had 65 alerts on 2026-07-20") must NOT get a lead-in
            # prepended — "There were Hand Touch had 65..." is broken
            # grammar, so this only fires for the truly bare case.
            lead_in = "The average was" if clauses[0].endswith("seconds") else "There were"
            time_phrase = _extract_time_phrase(question)
            suffix = f" recorded {time_phrase}" if time_phrase and not clauses[0].endswith("seconds") else ""
            text = f"{lead_in} {clauses[0]}{suffix}"
        else:
            text = clauses[0]
    elif len(clauses) == 2:
        text = f"{clauses[0]} and {clauses[1]}"
    else:
        text = ", ".join(clauses[:-1]) + f", and {clauses[-1]}"
    text = text[0].upper() + text[1:]
    if not text.endswith("."):
        text += "."
    if len(rows) > limit:
        text += f" ({len(rows)} total, showing top {limit}.)"
    return text


def _is_day_type_breakdown(rows: list) -> bool:
    return bool(rows) and all(
        isinstance(r, dict) and DATE_RE.match(str(r.get("date", ""))) for r in rows
    )


def _format_day_breakdown_by_week(rows: list) -> str:
    """Renders a flat [{"date": "2026-07-01", "type": "...", "count": N}, \
    ...] list as: Week 1 (days 1-7 of the month), Week 2 (8-14), etc. — \
    each week header followed by one line per date, each date listing \
    every type's count. This is a specific requested FORMAT (see \
    `_is_day_type_breakdown`'s call site), built deterministically in \
    Python from the real aggregated rows rather than asked of the model,
    since "group these into calendar weeks, then by day, then by type"
    is a third level of structure beyond anything the model has reliably
    produced this session."""
    from calendar import monthrange

    by_date: dict[str, list[dict]] = {}
    for r in rows:
        by_date.setdefault(r["date"], []).append(r)

    weeks: dict[tuple[int, int, int], list[str]] = {}
    for date_str in by_date:
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        week_num = ((d.day - 1) // 7) + 1
        weeks.setdefault((d.year, d.month, week_num), []).append(date_str)

    lines = []
    for year, month, week_num in sorted(weeks.keys()):
        dates_in_week = sorted(weeks[(year, month, week_num)])
        start_day = (week_num - 1) * 7 + 1
        end_day = min(start_day + 6, monthrange(year, month)[1])
        month_name = datetime(year, month, 1).strftime("%B")
        lines.append(f"{month_name} Week {week_num} ({month:02d}/{start_day:02d}-{month:02d}/{end_day:02d}):")
        for date_str in dates_in_week:
            parts = []
            for day_row in by_date[date_str]:
                count_val = day_row.get("count")
                if count_val is None:
                    continue
                type_val = day_row.get("type")
                label = str(type_val).replace("_", " ").title() if type_val else "Alerts"
                parts.append(f"{label}: {count_val}")
            if parts:
                lines.append(f"  {date_str} — {', '.join(parts)}")
    return "\n".join(lines)


def _is_week_type_breakdown(rows: list) -> bool:
    return bool(rows) and all(isinstance(r, dict) and "week" in r and "type" in r for r in rows)


def _format_week_type_breakdown(rows: list) -> str:
    """Renders a flat [{"week": "July Week 1 (07/01-07/07)", "type": \
    "...", "count": N}, ...] list as one line per week, listing every \
    type's count on that line — a monthly report's "each week broken \
    down by type" section. Same reasoning as `_format_day_breakdown_by_week`: \
    built deterministically from the real aggregated rows rather than \
    asked of the model."""
    by_week: dict[str, list[dict]] = {}
    order: list[str] = []
    for r in rows:
        wk = r["week"]
        if wk not in by_week:
            by_week[wk] = []
            order.append(wk)
        by_week[wk].append(r)

    lines = []
    for wk in order:
        parts = []
        for r in by_week[wk]:
            count_val = r.get("count")
            if count_val is None:
                continue
            type_val = r.get("type")
            label = str(type_val).replace("_", " ").title() if type_val else "Alerts"
            parts.append(f"{label}: {count_val}")
        if parts:
            lines.append(f"{wk} — {', '.join(parts)}")
    return "\n".join(lines)


FACET_SECTION_LEAD_INS = {
    "total_alerts": "",
    "by_type": "By type, ",
    "by_month": "By month, ",
    "by_week": "By week, ",
    "by_day": "By day, ",
    "avg_inspection_time": "",
}


def _facet_to_sentence(row: dict) -> str:
    # Flowing prose, not a "Label — text" list of labeled paragraphs — a
    # report reads like a person summarizing findings, and bolded section
    # headers in front of every sentence reads like a database printout
    # even when the sentence itself is perfectly natural. Regular
    # sections join into one continuous paragraph; only the rare day-by-
    # day block (a multi-line, pre-formatted table-in-text) gets its own
    # paragraph break, since it isn't a sentence to begin with.
    flowing = []
    blocks = []
    for name, sub_rows in row.items():
        flat_sub = [_flatten_row(r) for r in sub_rows if isinstance(r, dict)]
        if not flat_sub:
            continue
        if _is_day_type_breakdown(flat_sub):
            blocks.append(f"{name.replace('_', ' ').capitalize()}:\n{_format_day_breakdown_by_week(flat_sub)}")
            continue
        if _is_week_type_breakdown(flat_sub):
            blocks.append(f"{name.replace('_', ' ').capitalize()}:\n{_format_week_type_breakdown(flat_sub)}")
            continue
        lead_in = FACET_SECTION_LEAD_INS.get(name, f"{name.replace('_', ' ')}, ")
        sentence = _rows_to_sentence(flat_sub)
        if lead_in:
            sentence = lead_in + sentence[0].lower() + sentence[1:]
        flowing.append(sentence)
    if not flowing and not blocks:
        # Every named branch of the $facet came back empty — a real "no
        # matching data" case, not a phrasing bug. Returning "" here used
        # to silently blank out whatever the model had streamed, leaving
        # an empty assistant bubble with no explanation at all.
        return "No matching data found for that."
    paragraphs = ([" ".join(flowing)] if flowing else []) + blocks
    return "\n\n".join(paragraphs)


def _strip_narration(row):
    if not isinstance(row, dict):
        return row
    return {k: v for k, v in row.items() if k not in ("narration_en", "narration_ta")}


# =========================================================
# TOP-LEVEL ENTRY POINT
# =========================================================

def answer_question(question: str, history: list | None = None) -> dict:
    parsed, result, stages = _resolve_query(question, history)

    t0 = time.time()
    if parsed.get("_fast_path"):
        # Deterministic query -> deterministic sentence, no LLM call for
        # either half — see `_try_fast_path`. Mirrors `phrase_answer`'s
        # own empty-rows guard: MongoDB's $count stage emits ZERO rows
        # (not a {"total": 0} row) when nothing matches, which crashed
        # `_rows_to_sentence` (built assuming at least one row) the first
        # time a fast-path window genuinely had no alerts in it.
        rows = result["rows"]
        if parsed.get("_fast_path_answer"):
            answer = parsed["_fast_path_answer"]
        else:
            answer = _rows_to_sentence(rows, question=question) if rows else "No matching data found for that."
        _stage(stages, "Answer generation",
               "Question matched a simple, unambiguous pattern (a plain total count over a recognized "
               "time window) — phrased directly from the result, no model call needed.", t0)
    else:
        answer = phrase_answer(question, result)
        _stage(stages, "Answer generation",
               "Asked the language model to phrase the raw result rows as a natural-language answer, then ran "
               "deterministic safety checks (no invented values, no wrong superlatives, no dropped data).", t0)

    answer = _append_inherited_scope_note(answer, parsed)

    return {
        "question": question,
        "intent": parsed["intent"],
        "pipeline": parsed.get("pipeline"),
        "explanation": parsed.get("explanation"),
        "result": result["rows"],
        "row_count": len(result["rows"]),
        "answer": answer,
        "stages": stages,
    }


def answer_question_stream(question: str, history: list | None = None):
    """Same overall flow as `answer_question`, but yields
    (event_type, payload) tuples so the API layer can push them to the
    client as they happen: a "meta" event as soon as the pipeline has run
    (question/intent/pipeline/result — the frontend can render the table
    immediately), then a "token" event per chunk of the answer as the
    model writes it, then a "done" event with the final, safety-net-
    corrected answer text (which may differ from the streamed tokens if
    `_finalize_answer` had to override a bare/incomplete answer — the
    frontend should replace its accumulated text with this on "done"
    rather than trust the stream was already correct).
    """
    parsed, result, stages = _resolve_query(question, history)
    rows = result["rows"]

    meta = {
        "question": question,
        "intent": parsed["intent"],
        "pipeline": parsed.get("pipeline"),
        "explanation": parsed.get("explanation"),
        "result": rows,
        "row_count": len(rows),
        "stages": stages,
    }
    yield "meta", meta

    intent = result.get("intent")
    if intent == "unsupported":
        yield "done", {"answer": UNSUPPORTED_ANSWER, "stages": stages}
        return
    if intent == "gap_unsupported":
        yield "done", {"answer": GAP_QUESTION_ANSWER, "stages": stages}
        return
    if intent == "greeting":
        yield "done", {"answer": GREETING_ANSWER, "stages": stages}
        return
    if intent == "error":
        print(f"[llm_query] returning generic error answer to user; underlying DB error: {result.get('error')}")
        yield "done", {"answer": ERROR_ANSWER, "stages": stages}
        return
    if not rows:
        yield "done", {"answer": "No matching data found for that.", "stages": stages}
        return

    if parsed.get("_fast_path"):
        # Deterministic query -> deterministic sentence, no LLM call for
        # either half — see `_try_fast_path`. Sent as a single "token"
        # chunk (rather than a real stream) so the frontend renders it
        # through the identical code path either way.
        t0 = time.time()
        answer = parsed.get("_fast_path_answer") or _rows_to_sentence(rows, question=question)
        answer = _append_inherited_scope_note(answer, parsed)
        _stage(stages, "Answer generation",
               "Question matched a simple, unambiguous pattern (a plain total count over a recognized "
               "time window) — phrased directly from the result, no model call needed.", t0)
        yield "token", {"text": answer}
        yield "done", {"answer": answer, "stages": stages}
        return

    answer_t0 = time.time()
    user_prompt = _build_answer_prompt(question, rows)

    # Rows are confirmed non-empty at this point, so a model answer that
    # opens with "No matching data found..." is always a hallucination
    # here, not a real result — `_finalize_answer` catches and overrides
    # it, but naively streaming raw tokens live means the user briefly
    # SEES the wrong "no data" claim on screen before it's yanked back and
    # replaced, which reads as broken rather than corrected. So the first
    # ~20 characters are held back (unstreamed) just long enough to check
    # whether they're heading into that specific hallucination; if not,
    # they're flushed as one chunk and every following chunk streams live
    # as normal. If they are, nothing is streamed at all — the whole
    # answer stays buffered and only the corrected final text is sent.
    full_text = ""
    held = ""
    committed = False
    suppressed = False
    HOLD_THRESHOLD = 20

    for chunk in _chat_stream(ANSWER_SYSTEM_PROMPT, user_prompt, max_new_tokens=200):
        full_text += chunk
        if committed:
            yield "token", {"text": chunk}
            continue
        if suppressed:
            continue
        held += chunk
        if len(held) >= HOLD_THRESHOLD:
            if NO_DATA_CLAIM_RE.search(held):
                suppressed = True
            else:
                yield "token", {"text": held}
                committed = True

    if not committed and not suppressed and held:
        # Stream ended before the hold threshold was reached (a short
        # answer) — flush whatever's left, unless it's the hallucination.
        if NO_DATA_CLAIM_RE.search(held):
            suppressed = True
        else:
            yield "token", {"text": held}

    final_answer = _finalize_answer(rows, full_text.strip(), question, truncated=result.get("truncated", False))
    final_answer = _append_inherited_scope_note(final_answer, parsed)
    _stage(stages, "Answer generation",
           "Asked the language model to phrase the raw result rows as a natural-language answer (streamed "
           "live), then ran deterministic safety checks (no invented values, no wrong superlatives, no "
           "dropped data).", answer_t0)
    yield "done", {"answer": final_answer, "stages": stages}
