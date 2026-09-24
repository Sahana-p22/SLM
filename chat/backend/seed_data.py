# chat/backend/seed_data.py
#
# Backfills the `alerts` collection with a simulated history so the chatbot
# has real data to answer questions about. There's no live detection
# pipeline feeding Mongo yet, so this stands in until one exists.
#
# Reuses the narration templates from model/dataset_builder — the one
# deliberate coupling point between the model/ and chat/ halves of the
# repo, so demo alerts read the same as real training data would.
#
# Volume/pattern design (1 year, 100-200+ alerts/day, "looks real"):
#   - Hourly shape follows a 3-shift factory floor: quiet overnight,
#     ramp-up at shift start, a lunch dip, an afternoon peak, wind-down
#     at night — not a flat/uniform spread across the day.
#   - Weekdays run busier than weekends (reduced weekend staffing), but
#     every single day is still clamped to at least 100 alerts, per spec.
#   - A slow year-long trend lowers the MISSING_CLEANING share over time
#     (a believable "the corrective-action program is working" story)
#     instead of every month looking statistically identical.
#   - A small fraction of days get an anomaly spike in one alert type
#     (simulating an equipment issue or a rough shift) so the data isn't
#     perfectly smooth.

import random
from datetime import datetime, timedelta, timezone

from chat.backend.db import get_alerts_collection
from model.dataset_builder.narration_generator import generate_narration

random.seed(42)

ZONES = ["FQC Station 1"]

OBJECT_POOLS = {
    "FAST_INSPECTION": [["conveyor_belt", "person"], ["back_panel", "person"]],
    "HAND_TOUCH": [["back_panel", "person"], ["workstation", "person"]],
    "MISSING_CLEANING": [["workstation", "person"], ["panel", "person"]],
    "NORMAL_OPERATION": [["back_panel", "cloth", "gloved_hand"], ["conveyor_belt", "gloved_hand"]],
}

DAYS_OF_HISTORY = 365

# Non-NORMAL_OPERATION alert count per day. Weekends dip toward the low
# end, weekdays lean toward the high end, but neither ever drops below
# the 100 floor.
WEEKDAY_ALERT_RANGE = (150, 200)
WEEKEND_ALERT_RANGE = (100, 150)

# Normal (compliant) operation events logged alongside the real alerts —
# roughly comparable volume to the alerts themselves, same as a real QC
# station would log far more "all good" checks than actual issues.
NORMAL_OPS_RATIO_RANGE = (0.8, 1.3)

# Relative likelihood of an alert landing in each hour (0-23) — a 3-shift
# floor: quiet overnight, shift-start ramp-up, a lunch dip, an afternoon
# peak, evening wind-down. Not a flat distribution.
HOUR_WEIGHTS = [
    3, 2, 2, 2, 3, 5,      # 00-05: night shift, quiet
    9, 14, 16, 17, 16, 15,  # 06-11: morning ramp-up + shift
    10, 15, 18, 19, 18, 15,  # 12-17: lunch dip then afternoon peak
    12, 11, 9, 8, 6, 4,     # 18-23: evening wind-down
]

BASE_TYPE_WEIGHTS = {
    "FAST_INSPECTION": 0.42,
    "HAND_TOUCH": 0.36,
    "MISSING_CLEANING": 0.22,
}

# MISSING_CLEANING share drifts down over the year (start-of-year weight
# vs end-of-year weight) — a slow trend rather than a static ratio.
MISSING_CLEANING_START_WEIGHT = 0.30
MISSING_CLEANING_END_WEIGHT = 0.14

ANOMALY_DAY_CHANCE = 0.05
ANOMALY_MULTIPLIER_RANGE = (1.4, 2.2)


def random_inspection_time(alert_type):
    if alert_type == "FAST_INSPECTION":
        return round(random.uniform(0.5, 4.0), 1)
    return round(random.uniform(5.0, 30.0), 1)


def weighted_hour():
    return random.choices(range(24), weights=HOUR_WEIGHTS, k=1)[0]


def random_timestamp_on(day, elapsed_minutes=None):
    """A timestamp on `day` following the shift-shaped hourly pattern.
    `elapsed_minutes` (only used for "today") caps how far into the day a
    generated timestamp can land, so nothing is placed in the future."""
    hour = weighted_hour()
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    ts = day.replace(hour=hour, minute=minute, second=second, microsecond=0)
    if elapsed_minutes is not None:
        minutes_into_day = hour * 60 + minute
        if minutes_into_day > elapsed_minutes:
            ts = day + timedelta(minutes=random.uniform(0, elapsed_minutes))
    return ts


def random_recent_timestamp(now, max_minutes_ago):
    seconds_ago = random.uniform(5, max_minutes_ago * 60)
    return now - timedelta(seconds=seconds_ago)


def alert_type_weights_for_day(day_progress):
    """`day_progress` is 0.0 at the start of the year, 1.0 at the end —
    interpolates MISSING_CLEANING's share down over that span, matching
    the "the improvement program is working" trend rather than a fixed
    ratio for all 365 days."""
    missing_cleaning = (
        MISSING_CLEANING_START_WEIGHT
        + (MISSING_CLEANING_END_WEIGHT - MISSING_CLEANING_START_WEIGHT) * day_progress
    )
    remaining = 1.0 - missing_cleaning
    base_remaining = BASE_TYPE_WEIGHTS["FAST_INSPECTION"] + BASE_TYPE_WEIGHTS["HAND_TOUCH"]
    return {
        "FAST_INSPECTION": remaining * (BASE_TYPE_WEIGHTS["FAST_INSPECTION"] / base_remaining),
        "HAND_TOUCH": remaining * (BASE_TYPE_WEIGHTS["HAND_TOUCH"] / base_remaining),
        "MISSING_CLEANING": missing_cleaning,
    }


def build_alert(timestamp, alert_type):
    context = {
        "alert_type": alert_type,
        "inspection_time": random_inspection_time(alert_type),
        "objects_present": random.choice(OBJECT_POOLS[alert_type]),
        "cloth_detected": alert_type == "NORMAL_OPERATION",
    }

    narration = generate_narration(context)

    return {
        "alert_type": context["alert_type"],
        "inspection_time": context["inspection_time"],
        "objects_present": context["objects_present"],
        "cloth_detected": context["cloth_detected"],
        "narration_en": narration["english"],
        "narration_ta": narration["tamil"],
        "zone": random.choice(ZONES),
        "timestamp": timestamp,
        # Materialized alongside timestamp (timestamp is always UTC-aware
        # here, matching Mongo's default $hour timezone) so an hour-of-day
        # filter can be a plain indexed field range instead of a $expr
        # computation Mongo can never use an index to serve - see the
        # (alert_type, hour) index and _use_indexed_hour_filter in
        # llm_query.py. Every future insert needs this written at source;
        # a one-time backfill alone would silently go stale the next time
        # seed()/top_up() runs.
        "hour": timestamp.hour,
    }


def alert_count_for_day(day, day_progress):
    is_weekend = day.weekday() >= 5
    low, high = WEEKEND_ALERT_RANGE if is_weekend else WEEKDAY_ALERT_RANGE
    count = random.randint(low, high)

    anomaly_type = None
    if random.random() < ANOMALY_DAY_CHANCE:
        anomaly_type = random.choice(["FAST_INSPECTION", "HAND_TOUCH", "MISSING_CLEANING"])
        count = int(count * random.uniform(*ANOMALY_MULTIPLIER_RANGE))

    return max(count, 100), anomaly_type


def build_day_alerts(day, day_progress, count, anomaly_type, elapsed_minutes=None):
    weights = alert_type_weights_for_day(day_progress)
    types = list(weights.keys())
    type_weights = list(weights.values())

    docs = []
    for _ in range(count):
        alert_type = (
            anomaly_type if anomaly_type and random.random() < 0.5
            else random.choices(types, weights=type_weights, k=1)[0]
        )
        docs.append(build_alert(random_timestamp_on(day, elapsed_minutes), alert_type))

    normal_ops = int(count * random.uniform(*NORMAL_OPS_RATIO_RANGE))
    for _ in range(normal_ops):
        docs.append(build_alert(random_timestamp_on(day, elapsed_minutes), "NORMAL_OPERATION"))

    return docs


def seed(days=DAYS_OF_HISTORY):
    collection = get_alerts_collection()

    existing = collection.count_documents({})
    if existing > 0:
        print(f"[INFO] Collection already has {existing} documents. Dropping before reseeding.")
        collection.drop()

    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    batch = []
    total_inserted = 0
    BATCH_SIZE = 5000

    def flush():
        nonlocal batch, total_inserted
        if not batch:
            return
        collection.insert_many(batch)
        total_inserted += len(batch)
        print(f"[INFO] Inserted {total_inserted} documents so far...")
        batch = []

    # Past full days (offset 1..N-1) — always fully in the past regardless
    # of what time "now" is, so no future-timestamp risk here.
    for offset in range(days, 0, -1):
        day = today_start - timedelta(days=offset)
        day_progress = 1 - (offset / days)  # 0.0 at year start -> ~1.0 near today
        count, anomaly_type = alert_count_for_day(day, day_progress)
        batch.extend(build_day_alerts(day, day_progress, count, anomaly_type))
        if len(batch) >= BATCH_SIZE:
            flush()

    # Today (offset 0): same shift-shaped hourly pattern, but every
    # timestamp is capped to the elapsed portion of the day so nothing
    # lands in the future — this is what keeps the dashboard's short
    # windows (15m/1h/6h) populated right after seeding.
    elapsed_minutes = max(1, int((now - today_start).total_seconds() / 60))
    count, anomaly_type = alert_count_for_day(today_start, 1.0)
    # Scale today's volume down proportionally to how much of the day has
    # actually elapsed, so "today" doesn't already show a full day's worth
    # of alerts at 9am.
    day_fraction = min(1.0, elapsed_minutes / (24 * 60))
    today_count = max(10, int(count * day_fraction))
    batch.extend(build_day_alerts(today_start, 1.0, today_count, anomaly_type, elapsed_minutes))

    # Guaranteed-recent bursts, so the dashboard's 15m/1h/6h presets always
    # have something to show right after seeding, not just whatever
    # happened to land there by chance.
    for _ in range(5):
        batch.append(build_alert(random_recent_timestamp(now, max_minutes_ago=15), random.choice(list(BASE_TYPE_WEIGHTS.keys()))))
    for _ in range(10):
        batch.append(build_alert(random_recent_timestamp(now, max_minutes_ago=60), random.choice(list(BASE_TYPE_WEIGHTS.keys()))))
    for _ in range(25):
        batch.append(build_alert(random_recent_timestamp(now, max_minutes_ago=360), random.choice(list(BASE_TYPE_WEIGHTS.keys()))))

    flush()

    collection.create_index("timestamp")
    collection.create_index("alert_type")
    collection.create_index("zone")

    print(f"[INFO] Done. Inserted {total_inserted} simulated alerts across {days} days "
          f"(including guaranteed-recent alerts for 15m/1h/6h windows).")


def top_up():
    """Non-destructive gap-fill: adds alerts from the newest timestamp already
    in the collection up to "now", leaving every existing document untouched.

    Why this exists: seed() drops and regenerates the whole collection, which
    would destroy a frozen historical dataset (and any benchmark baseline
    measured against it). But a demo box that has been idle for weeks answers
    "how many alerts happened today?" with 0 -- not because the pipeline is
    wrong, but because the data simply stops before today. That reads as a
    correctness bug in every relative-date question ("today", "this week",
    "this month"), so it is worth fixing properly rather than reseeding.
    """
    collection = get_alerts_collection()

    newest = list(collection.find({}, {"timestamp": 1, "_id": 0})
                  .sort("timestamp", -1).limit(1))
    if not newest:
        print("[INFO] Collection is empty -- run a full seed() instead.")
        return

    last_ts = newest[0]["timestamp"]
    if last_ts.tzinfo is None:  # pymongo hands back naive UTC datetimes
        last_ts = last_ts.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    last_day_start = last_ts.replace(hour=0, minute=0, second=0, microsecond=0)

    if last_day_start >= today_start:
        print(f"[INFO] Already current (newest alert {last_ts.isoformat()}); nothing to top up.")
        return

    batch = []
    total_inserted = 0
    BATCH_SIZE = 5000

    def flush():
        nonlocal batch, total_inserted
        if not batch:
            return
        collection.insert_many(batch)
        total_inserted += len(batch)
        batch = []

    # Full days strictly between the last seeded day and today. The day the
    # data currently ends on is skipped: it already holds a partial day's
    # alerts, and topping it up would double-count that day.
    day = last_day_start + timedelta(days=1)
    n_days = 0
    while day < today_start:
        count, anomaly_type = alert_count_for_day(day, 1.0)
        batch.extend(build_day_alerts(day, 1.0, count, anomaly_type))
        n_days += 1
        if len(batch) >= BATCH_SIZE:
            flush()
        day += timedelta(days=1)

    # Today, capped to the elapsed part of the day so nothing lands in the future.
    elapsed_minutes = max(1, int((now - today_start).total_seconds() / 60))
    count, anomaly_type = alert_count_for_day(today_start, 1.0)
    day_fraction = min(1.0, elapsed_minutes / (24 * 60))
    batch.extend(build_day_alerts(today_start, 1.0, max(10, int(count * day_fraction)),
                                  anomaly_type, elapsed_minutes))

    # Same guaranteed-recent bursts seed() adds, so the dashboard's 15m/1h/6h
    # presets have something to show immediately after a top-up too.
    for _ in range(5):
        batch.append(build_alert(random_recent_timestamp(now, 15), random.choice(list(BASE_TYPE_WEIGHTS.keys()))))
    for _ in range(10):
        batch.append(build_alert(random_recent_timestamp(now, 60), random.choice(list(BASE_TYPE_WEIGHTS.keys()))))
    for _ in range(25):
        batch.append(build_alert(random_recent_timestamp(now, 360), random.choice(list(BASE_TYPE_WEIGHTS.keys()))))

    flush()

    collection.create_index("timestamp")
    collection.create_index("alert_type")
    collection.create_index("zone")

    print(f"[INFO] Topped up {total_inserted} alerts across {n_days} full day(s) "
          f"plus today ({last_day_start.date()} -> {today_start.date()}).")


if __name__ == "__main__":
    import sys
    if "--top-up" in sys.argv:
        top_up()
    else:
        seed()
