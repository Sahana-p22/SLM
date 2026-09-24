"""Generates the 10-year production-volume scale-test collection:
31,557,600 rows (10 years @ 1 row/10s), matching the real schema
(alert_type, inspection_time, objects_present, cloth_detected,
narration_en/ta, zone, timestamp, hour), in an ISOLATED database
(slm_safety_scale) so it never touches the real production
slm_safety.alerts collection.

Bulk-inserted directly via pymongo (not through the app), matching the
real data's type distribution and time span (10 years back from today,
1 row/10s).
"""
import random
import sys
import time
from datetime import datetime, timedelta, timezone

from pymongo import MongoClient

from bench_config import REPO_ROOT, MONGO_URI, SCALE_TEST_DB
sys.path.insert(0, REPO_ROOT)
from chat.backend.seed_data import ZONES, OBJECT_POOLS

client = MongoClient(MONGO_URI)
db = client[SCALE_TEST_DB]

TOTAL_ROWS = 31_557_600  # 10 years @ 1 row / 10s
BATCH = 50_000
NOW = datetime.now(timezone.utc)
START = NOW - timedelta(seconds=10 * TOTAL_ROWS)

TYPE_WEIGHTS = [("FAST_INSPECTION", 0.42), ("HAND_TOUCH", 0.35),
                ("MISSING_CLEANING", 0.13), ("NORMAL_OPERATION", 0.10)]
TYPES = [t for t, _ in TYPE_WEIGHTS]
WEIGHTS = [w for _, w in TYPE_WEIGHTS]

INSPECTION_TIME_RANGES = {
    "FAST_INSPECTION": (2, 8), "HAND_TOUCH": (5, 20),
    "MISSING_CLEANING": (3, 15), "NORMAL_OPERATION": (1, 5),
}


def build_row(ts, alert_type):
    lo, hi = INSPECTION_TIME_RANGES[alert_type]
    return {
        "alert_type": alert_type,
        "inspection_time": round(random.uniform(lo, hi), 2),
        "objects_present": random.choice(OBJECT_POOLS[alert_type]),
        "cloth_detected": alert_type == "NORMAL_OPERATION",
        "narration_en": "",  # skip narration text at this volume - not queried by any benchmark question
        "narration_ta": "",
        "zone": random.choice(ZONES),
        "timestamp": ts,
        "hour": ts.hour,
    }


def generate(collection_name):
    coll = db[collection_name]
    coll.drop()
    print(f"[{collection_name}] generating {TOTAL_ROWS:,} rows...")
    t0 = time.time()
    ts = START
    step = timedelta(seconds=10)
    inserted = 0
    batch = []
    while inserted < TOTAL_ROWS:
        atype = random.choices(TYPES, weights=WEIGHTS, k=1)[0]
        batch.append(build_row(ts, atype))
        ts += step
        inserted += 1
        if len(batch) >= BATCH:
            coll.insert_many(batch, ordered=False)
            batch = []
            if inserted % 1_000_000 < BATCH:
                elapsed = time.time() - t0
                rate = inserted / elapsed
                eta = (TOTAL_ROWS - inserted) / rate
                print(f"  {inserted:,}/{TOTAL_ROWS:,} ({rate:,.0f} rows/s, "
                      f"eta {eta/60:.1f} min)")
    if batch:
        coll.insert_many(batch, ordered=False)
    dt = time.time() - t0
    print(f"[{collection_name}] done: {coll.estimated_document_count():,} rows in {dt/60:.1f} min")


if __name__ == "__main__":
    generate("alerts_no_index")
    print("Collection ready with NO index for baseline timing.")
