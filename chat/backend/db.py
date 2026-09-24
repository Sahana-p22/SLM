# backend/db.py

from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "slm_safety"
ALERTS_COLLECTION = "alerts"

_client = MongoClient(MONGO_URI)
_db = _client[DB_NAME]
_indexes_ensured = False


def _ensure_indexes(collection) -> None:
    """Declares the indexes the query patterns in llm_query.py actually
    need, as code rather than a one-off shell command against the live
    server - so a fresh deployment (a new environment, a restored backup)
    gets them automatically instead of silently running unindexed until
    someone notices.

    (alert_type, timestamp) compound is the one that matters: almost
    every generated pipeline's $match filters BOTH fields at once (an
    alert-type constraint - even the default "$ne NORMAL_OPERATION" - and
    a date range), and single-field indexes on each separately only let
    Mongo use ONE of them as an index scan, fetching and then filtering
    the other in memory. Confirmed live: the same query examined 30,500
    documents to return 14,837 with only timestamp_1 available, and
    exactly 14,837 (no wasted fetches) once this compound index existed.
    `create_index` is idempotent - safe to call on every startup, and a
    no-op once the index already exists."""
    global _indexes_ensured
    if _indexes_ensured:
        return
    collection.create_index([("alert_type", 1), ("timestamp", 1)])
    _indexes_ensured = True


def get_alerts_collection():
    collection = _db[ALERTS_COLLECTION]
    _ensure_indexes(collection)
    return collection
