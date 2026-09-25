# chat/backend/migrate_mongo_to_sqlite.py
#
# One-time full migration of every document in MongoDB's slm_safety.alerts
# collection into a local SQLite mirror, losslessly. Every field the source
# documents actually have (_id, alert_type, inspection_time, objects_present,
# cloth_detected, narration_en, narration_ta, zone, timestamp, hour) is
# represented in the SQLite schema — nothing is dropped or approximated.
#
# Encoding conventions (documented here since they're the contract the
# sync daemon and the verification script both depend on):
#   _id             -> TEXT, the ObjectId's 24-char hex string (str(oid)).
#                      This is the SQLite PRIMARY KEY, which is what makes
#                      re-running this script (or the sync daemon) an
#                      idempotent "INSERT OR IGNORE" rather than a duplicate.
#   timestamp       -> TEXT, datetime.isoformat() exactly as Mongo returns
#                      it (naive, no timezone suffix — matches source).
#   cloth_detected  -> INTEGER 0/1 (SQLite has no native bool).
#   objects_present -> TEXT, a JSON array (json.dumps of the Python list),
#                      so the exact list of strings round-trips exactly.
#   everything else -> stored as-is (TEXT/REAL/INTEGER matching its Python type).
import os
import sys
import time
import json
import sqlite3

import pymongo

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
SQLITE_PATH = os.environ.get("ALERTS_SQLITE_PATH", os.path.join(os.path.dirname(__file__), "alerts.sqlite3"))
BATCH_SIZE = 5000

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    alert_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    inspection_time REAL,
    zone TEXT,
    cloth_detected INTEGER,
    objects_present TEXT,
    narration_en TEXT,
    narration_ta TEXT,
    hour INTEGER
);
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(alert_type)",
    "CREATE INDEX IF NOT EXISTS idx_alerts_zone ON alerts(zone)",
    "CREATE INDEX IF NOT EXISTS idx_alerts_type_timestamp ON alerts(alert_type, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_alerts_type_hour ON alerts(alert_type, hour)",
]


def doc_to_row(doc: dict) -> tuple:
    return (
        str(doc["_id"]),
        doc["alert_type"],
        doc["timestamp"].isoformat(),
        doc.get("inspection_time"),
        doc.get("zone"),
        int(bool(doc.get("cloth_detected"))) if doc.get("cloth_detected") is not None else None,
        json.dumps(doc.get("objects_present")) if doc.get("objects_present") is not None else None,
        doc.get("narration_en"),
        doc.get("narration_ta"),
        doc.get("hour"),
    )


def migrate(fresh: bool = True) -> int:
    client = pymongo.MongoClient(MONGO_URI)
    collection = client["slm_safety"]["alerts"]
    total = collection.count_documents({})
    print(f"[migrate] source has {total} documents")

    conn = sqlite3.connect(SQLITE_PATH)
    if fresh:
        conn.execute("DROP TABLE IF EXISTS alerts")
    conn.execute(SCHEMA)
    conn.commit()

    t0 = time.time()
    n = 0
    batch = []
    cursor = collection.find({}, no_cursor_timeout=True).sort("_id", 1).batch_size(BATCH_SIZE)
    try:
        for doc in cursor:
            batch.append(doc_to_row(doc))
            if len(batch) >= BATCH_SIZE:
                conn.executemany(
                    "INSERT OR IGNORE INTO alerts VALUES (?,?,?,?,?,?,?,?,?,?)", batch
                )
                conn.commit()
                n += len(batch)
                batch = []
                if n % 100000 == 0:
                    print(f"[migrate] {n}/{total} ({time.time()-t0:.1f}s)")
        if batch:
            conn.executemany("INSERT OR IGNORE INTO alerts VALUES (?,?,?,?,?,?,?,?,?,?)", batch)
            conn.commit()
            n += len(batch)
    finally:
        cursor.close()

    for idx_sql in INDEXES:
        conn.execute(idx_sql)
    conn.commit()

    sqlite_count = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    conn.close()
    print(f"[migrate] inserted {n} rows in {time.time()-t0:.1f}s; sqlite table now has {sqlite_count} rows")
    return sqlite_count


if __name__ == "__main__":
    fresh = "--incremental" not in sys.argv
    count = migrate(fresh=fresh)
    if count == 0:
        print("[migrate] ERROR: zero rows migrated", file=sys.stderr)
        sys.exit(1)
