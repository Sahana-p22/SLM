# chat/backend/verify_mongo_sqlite_parity.py
#
# Re-runnable parity check between MongoDB's slm_safety.alerts collection
# (source of truth) and the SQLite mirror (chat/backend/alerts.sqlite3).
# Exits non-zero and prints every mismatch found if anything doesn't match —
# meant to be run after every migration/sync change, not just once.
import os
import sys
import json
import random
import sqlite3

import pymongo

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
SQLITE_PATH = os.environ.get("ALERTS_SQLITE_PATH", os.path.join(os.path.dirname(__file__), "alerts.sqlite3"))
SAMPLE_SIZE = int(os.environ.get("PARITY_SAMPLE_SIZE", "200"))

FIELDS = ["id", "alert_type", "timestamp", "inspection_time", "zone",
          "cloth_detected", "objects_present", "narration_en", "narration_ta", "hour"]


def doc_to_expected_row(doc: dict) -> dict:
    return {
        "id": str(doc["_id"]),
        "alert_type": doc["alert_type"],
        "timestamp": doc["timestamp"].isoformat(),
        "inspection_time": doc.get("inspection_time"),
        "zone": doc.get("zone"),
        "cloth_detected": int(bool(doc.get("cloth_detected"))) if doc.get("cloth_detected") is not None else None,
        "objects_present": json.dumps(doc.get("objects_present")) if doc.get("objects_present") is not None else None,
        "narration_en": doc.get("narration_en"),
        "narration_ta": doc.get("narration_ta"),
        "hour": doc.get("hour"),
    }


def main() -> int:
    client = pymongo.MongoClient(MONGO_URI)
    coll = client["slm_safety"]["alerts"]
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row

    errors = []

    # 1. Total count
    mongo_count = coll.count_documents({})
    sqlite_count = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    print(f"[parity] total count: mongo={mongo_count} sqlite={sqlite_count}")
    if mongo_count != sqlite_count:
        errors.append(f"TOTAL COUNT MISMATCH: mongo={mongo_count} sqlite={sqlite_count}")

    # 2. Count by alert_type
    mongo_by_type = {
        r["_id"]: r["c"] for r in coll.aggregate([
            {"$group": {"_id": "$alert_type", "c": {"$sum": 1}}}
        ])
    }
    sqlite_by_type = {
        r["alert_type"]: r["c"] for r in
        conn.execute("SELECT alert_type, COUNT(*) as c FROM alerts GROUP BY alert_type")
    }
    print(f"[parity] by-type: mongo={mongo_by_type} sqlite={sqlite_by_type}")
    if mongo_by_type != sqlite_by_type:
        errors.append(f"BY-TYPE COUNT MISMATCH: mongo={mongo_by_type} sqlite={sqlite_by_type}")

    # 3. Min/max timestamp
    mongo_min = coll.find_one(sort=[("timestamp", 1)])["timestamp"].isoformat()
    mongo_max = coll.find_one(sort=[("timestamp", -1)])["timestamp"].isoformat()
    sqlite_min = conn.execute("SELECT MIN(timestamp) FROM alerts").fetchone()[0]
    sqlite_max = conn.execute("SELECT MAX(timestamp) FROM alerts").fetchone()[0]
    print(f"[parity] timestamp range: mongo=({mongo_min}, {mongo_max}) sqlite=({sqlite_min}, {sqlite_max})")
    if mongo_min != sqlite_min or mongo_max != sqlite_max:
        errors.append(f"TIMESTAMP RANGE MISMATCH: mongo=({mongo_min},{mongo_max}) sqlite=({sqlite_min},{sqlite_max})")

    # 4. Field-by-field spot check on a random sample spread across the whole collection
    ids = [d["_id"] for d in coll.find({}, {"_id": 1})]
    sample_ids = random.sample(ids, min(SAMPLE_SIZE, len(ids)))
    mismatches = 0
    for oid in sample_ids:
        doc = coll.find_one({"_id": oid})
        expected = doc_to_expected_row(doc)
        row = conn.execute("SELECT * FROM alerts WHERE id = ?", (str(oid),)).fetchone()
        if row is None:
            errors.append(f"MISSING IN SQLITE: id={oid}")
            mismatches += 1
            continue
        row_dict = dict(row)
        for field in FIELDS:
            if row_dict.get(field) != expected.get(field):
                errors.append(
                    f"FIELD MISMATCH id={oid} field={field}: mongo={expected.get(field)!r} sqlite={row_dict.get(field)!r}"
                )
                mismatches += 1
    print(f"[parity] spot-checked {len(sample_ids)} random documents field-by-field, {mismatches} mismatches")

    if errors:
        print(f"\n[parity] FAILED with {len(errors)} error(s):")
        for e in errors[:50]:
            print("  -", e)
        return 1
    print("\n[parity] ALL CHECKS PASSED — SQLite mirror is a lossless copy of MongoDB.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
