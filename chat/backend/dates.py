"""Deterministic relative-date parsing — ported from the Retail deployment's
retail_llm/dates.py (same technique, domain-agnostic).

Overrides the model's own (unreliable) date arithmetic: we compute the range
in code and both (a) feed it into the query prompt and (b) re-check it after.

Returns a half-open [start, end) tuple of datetimes, or None if unrecognized.
"""
import re
from datetime import datetime, timedelta

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})

# Caps unbounded "last/past N <unit>" phrases (e.g. "the last 999999999
# days") from overflowing datetime arithmetic (OverflowError: date value
# out of range). Found while porting this project's own historically-fixed
# bug list (FIX AA in slm-llama3b's llm_query.py covers the identical
# MongoDB-side case) — this SQLite/SQL port had the same unguarded
# timedelta(days=n) call and reproduced the identical crash. Set far larger
# than this dataset's real ~5-year span, so no genuine question is affected.
_MAX_RANGE_DAYS = 20000


def _day_start(d: datetime) -> datetime:
    return d.replace(hour=0, minute=0, second=0, microsecond=0)


def _week_start(d: datetime) -> datetime:
    return _day_start(d) - timedelta(days=d.weekday())  # Monday


def _month_start(d: datetime) -> datetime:
    return _day_start(d).replace(day=1)


def _add_month(d: datetime, n: int) -> datetime:
    m = d.month - 1 + n
    return d.replace(year=d.year + m // 12, month=m % 12 + 1, day=1)


_MONTH_ALT = "|".join(MONTHS)


def _specific_date(t: str, now: datetime):
    """A full specific calendar date ('september 19th 2023', 'sep 19, 2023',
    '19 september 2023', '2023-09-19') — NOT covered by Retail's original
    dates.py, whose domain (recent retail sales) rarely needs an exact
    historical day. This FQC alert log spans 2021-2026 and "how many alerts
    on <date>" is a common, exact-day question here, so this case is added
    on top of the ported logic.

    Also covers a bare day+month with NO year ("what happened on march
    3rd", "3rd of march") — found to silently fall through to the
    month-only regex further down in extract_range() and match the WHOLE
    month instead of just that one day (the identical bug class as
    slm-llama3b's FIX CC). Defaults the year to `now`'s year when absent,
    same convention already used for the month-only case below."""
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            s = datetime(y, mo, d)
            return (s, s + timedelta(days=1))
        except ValueError:
            pass
    m = re.search(r"\b(" + _MONTH_ALT + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})\b", t)
    if m:
        mo, d, y = MONTHS[m.group(1)], int(m.group(2)), int(m.group(3))
        try:
            s = datetime(y, mo, d)
            return (s, s + timedelta(days=1))
        except ValueError:
            pass
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + _MONTH_ALT + r")\.?,?\s*(\d{4})\b", t)
    if m:
        d, mo, y = int(m.group(1)), MONTHS[m.group(2)], int(m.group(3))
        try:
            s = datetime(y, mo, d)
            return (s, s + timedelta(days=1))
        except ValueError:
            pass
    # Bare day+month, no year, either order: "march 3rd" / "3rd of march" / "3 march".
    m = re.search(r"\b(" + _MONTH_ALT + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b", t)
    if m:
        mo, d = MONTHS[m.group(1)], int(m.group(2))
        try:
            s = datetime(now.year, mo, d)
            return (s, s + timedelta(days=1))
        except ValueError:
            pass
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(" + _MONTH_ALT + r")\.?\b", t)
    if m:
        d, mo = int(m.group(1)), MONTHS[m.group(2)]
        try:
            s = datetime(now.year, mo, d)
            return (s, s + timedelta(days=1))
        except ValueError:
            pass
    return None


def extract_range(text: str, now: datetime):
    t = text.lower().strip()

    specific = _specific_date(t, now)
    if specific:
        return specific

    if re.search(r"\btoday\b|\bso far today\b", t):
        s = _day_start(now)
        return (s, s + timedelta(days=1))
    if re.search(r"\byesterday\b", t):
        s = _day_start(now) - timedelta(days=1)
        return (s, s + timedelta(days=1))
    if re.search(r"\btomorrow\b", t):
        s = _day_start(now) + timedelta(days=1)
        return (s, s + timedelta(days=1))

    m = re.search(r"\b(this|last|previous|past|next)\s+(week|month|year|quarter)\b", t)
    if m:
        which, unit = m.group(1), m.group(2)
        direction = -1 if which in ("last", "previous", "past") else (1 if which == "next" else 0)
        if unit == "week":
            s = _week_start(now) + timedelta(days=7 * direction)
            return (s, s + timedelta(days=7))
        if unit == "month":
            s = _add_month(_month_start(now), direction)
            return (s, _add_month(s, 1))
        if unit == "year":
            y = now.year + direction
            return (datetime(y, 1, 1), datetime(y + 1, 1, 1))
        if unit == "quarter":
            q = (now.month - 1) // 3
            s = datetime(now.year, q * 3 + 1, 1)
            s = _add_month(s, 3 * direction)
            return (s, _add_month(s, 3))

    m = re.search(r"\b(?:last|past|previous)\s+(\d+)\s*(minute|hour|day|week|month)s?\b", t)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        if unit == "minute":
            n = min(n, _MAX_RANGE_DAYS * 24 * 60)
            return (now - timedelta(minutes=n), now)
        if unit == "hour":
            n = min(n, _MAX_RANGE_DAYS * 24)
            return (now - timedelta(hours=n), now)
        if unit == "day":
            n = min(n, _MAX_RANGE_DAYS)
            return (_day_start(now) - timedelta(days=n - 1), _day_start(now) + timedelta(days=1))
        if unit == "week":
            n = min(n, _MAX_RANGE_DAYS // 7)
            return (_day_start(now) - timedelta(weeks=n), _day_start(now) + timedelta(days=1))
        if unit == "month":
            n = min(n, _MAX_RANGE_DAYS // 30)
            return (_add_month(_month_start(now), -n), _add_month(_month_start(now), 1))

    m = re.search(r"\b(" + "|".join(MONTHS) + r")\.?\s*(\d{4})?\b", t)
    if m and m.group(1) in MONTHS:
        mon = MONTHS[m.group(1)]
        yr = int(m.group(2)) if m.group(2) else now.year
        s = datetime(yr, mon, 1)
        return (s, _add_month(s, 1))

    m = re.search(r"\b(19\d{2}|20\d{2})\b", t)
    if m:
        y = int(m.group(1))
        return (datetime(y, 1, 1), datetime(y + 1, 1, 1))

    return None


def label(rng) -> str:
    a, b = rng
    return f"{a.isoformat(sep=' ')} .. {b.isoformat(sep=' ')}"


_TOD_RANGES = {
    "early morning": (5, 8), "late night": (21, 24),
    "morning": (6, 12), "afternoon": (12, 17), "evening": (17, 21), "night": (21, 24),
}
_HOUR_WORD = r"(\d{1,2})\s*(am|pm)"
_HOUR_RANGE_RE = re.compile(_HOUR_WORD + r"\s*(?:to|and|-)\s*" + _HOUR_WORD, re.I)


def _to_24h(h: str, ampm: str) -> int:
    h = int(h)
    if ampm.lower() == "am":
        return 0 if h == 12 else h
    return 12 if h == 12 else h + 12


def extract_hour_range(text: str):
    """Hour-of-day RANGE, e.g. 'between 2pm and 4pm' -> (14, 16) (half-open,
    matching extract_range's convention: hour >= start AND hour < end). Also
    a small set of qualitative time-of-day words. Returns None if the
    question names no hour-of-day range at all, or only a bare number range
    with no am/pm anywhere (genuinely ambiguous - '3 to 5' could mean
    anything). A start > end pair (e.g. '11pm to 1am' -> (23, 1)) is a real
    overnight wraparound, left for the caller to interpret as
    `hour >= start OR hour < end`.

    Ported concept (not the original's Mongo-pipeline mechanism) from
    slm-llama3b's FIX Z / sanity_hour_range_phrasing.py, after finding this
    port's own SQL generation reproduces the identical underlying bug: a
    named hour range ('between 2pm and 4pm') collapsed to a single-hour
    equality (`hour = 14`), silently dropping the '4pm' half of the range.
    """
    t = text.lower()
    for phrase in ("early morning", "late night", "morning", "afternoon", "evening", "night"):
        if re.search(rf"\b{phrase}\b", t):
            return _TOD_RANGES[phrase]
    m = _HOUR_RANGE_RE.search(t)
    if m:
        return (_to_24h(m.group(1), m.group(2)), _to_24h(m.group(3), m.group(4)))
    return None
