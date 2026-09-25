# chat/backend/db_sqlite.py
#
# The shared SQLite connection to the alerts mirror (chat/backend/
# alerts.sqlite3, built by migrate_mongo_to_sqlite.py and kept live by
# mongo_sqlite_sync.py). Every read path in this app — the chat/SQL
# pipeline in llm_query_sql.py, and the deterministic dashboard/stats/
# alerts-recent endpoints in main.py — goes through this one connection,
# so there's exactly one place that opens the file and one place that
# knows its path.
import os
import sqlite3
import threading

SQLITE_PATH = os.environ.get(
    "ALERTS_SQLITE_PATH",
    os.path.join(os.path.dirname(__file__), "alerts.sqlite3"),
)

_lock = threading.Lock()
_conn = None


def get_connection() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        with _lock:
            if _conn is None:
                _conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
                _conn.row_factory = sqlite3.Row
    return _conn
