# chat/backend/mongo_sqlite_sync.py
#
# Keeps the SQLite mirror (alerts.sqlite3) up to date with MongoDB forever,
# for as long as this process runs. MongoDB stays the write path (nothing
# here ever writes to Mongo) — this only reads newly-inserted documents and
# copies them into SQLite.
#
# Mechanism: this MongoDB is a standalone mongod (confirmed via `hello` —
# no replica set), so native Change Streams aren't available without
# reconfiguring the live instance, which risks the already-running app on
# port 8002/8005. Polling is lower-risk and perfectly adequate here — this
# is factory-alert-log-scale write volume (at most a few inserts/sec), not
# a high-frequency feed. Polls for any document with _id greater than the
# highest _id synced so far (ObjectIds are monotonically increasing by
# creation time in this deployment, so "greater than the last one we saw"
# reliably means "created after"), inserts them into SQLite with
# INSERT OR IGNORE keyed on that same id, so running this twice, or after
# a crash/restart, never double-inserts anything.
import os
import sys
import time
import json
import signal
import sqlite3

import pymongo

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
SQLITE_PATH = os.environ.get("ALERTS_SQLITE_PATH", os.path.join(os.path.dirname(__file__), "alerts.sqlite3"))
POLL_INTERVAL_SECONDS = float(os.environ.get("SYNC_POLL_INTERVAL", "3"))

_running = True


def _handle_stop(signum, frame):
    global _running
    _running = False


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


def _last_synced_object_id(conn: sqlite3.Connection):
    """The highest id currently in SQLite, as a real ObjectId (so it can be
    used directly in a Mongo query), or None if the table is empty."""
    from bson import ObjectId
    row = conn.execute("SELECT id FROM alerts ORDER BY id DESC LIMIT 1").fetchone()
    if row is None:
        return None
    try:
        return ObjectId(row[0])
    except Exception:
        return None


def sync_once(conn: sqlite3.Connection, collection) -> int:
    """Runs one poll cycle: finds every Mongo doc newer than what SQLite
    already has, inserts it. Returns the number of new rows inserted."""
    last_id = _last_synced_object_id(conn)
    query = {"_id": {"$gt": last_id}} if last_id is not None else {}
    cursor = collection.find(query).sort("_id", 1)
    batch = [doc_to_row(doc) for doc in cursor]
    if not batch:
        return 0
    conn.executemany("INSERT OR IGNORE INTO alerts VALUES (?,?,?,?,?,?,?,?,?,?)", batch)
    conn.commit()
    return len(batch)


def run_forever():
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    client = pymongo.MongoClient(MONGO_URI)
    collection = client["slm_safety"]["alerts"]
    conn = sqlite3.connect(SQLITE_PATH)

    print(f"[sync] watching {MONGO_URI}/slm_safety.alerts -> {SQLITE_PATH}, polling every {POLL_INTERVAL_SECONDS}s")
    while _running:
        try:
            n = sync_once(conn, collection)
            if n:
                print(f"[sync] synced {n} new row(s)")
        except Exception as exc:
            print(f"[sync] ERROR during sync cycle: {exc}", file=sys.stderr)
        for _ in range(int(POLL_INTERVAL_SECONDS * 10)):
            if not _running:
                break
            time.sleep(0.1)
    conn.close()
    print("[sync] stopped.")


if __name__ == "__main__":
    if "--once" in sys.argv:
        _conn = sqlite3.connect(SQLITE_PATH)
        _client = pymongo.MongoClient(MONGO_URI)
        _n = sync_once(_conn, _client["slm_safety"]["alerts"])
        print(f"[sync] one-shot: synced {_n} new row(s)")
        _conn.close()
    else:
        run_forever()
