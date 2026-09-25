# backend/db.py — SQLite backend for the approach2 experiment.
#
# Replaces slm-llama3b's pymongo client. The LLM now writes SQL against
# this database instead of MongoDB aggregation pipelines. Read-only by
# design: run_readonly() is the only way pipeline code touches the DB.

import sqlite3
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = _REPO_ROOT / "chat" / "backend" / "fqc.db"

_conn = None


def _get_conn():
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
    return _conn


def run_readonly(sql: str, params=()):
    """Execute a SELECT and return a list of plain dicts. Uses SQLite's own
    query_only pragma as a second, engine-level enforcement layer underneath
    repair.validate_sql()'s text-level check — belt and suspenders."""
    conn = _get_conn()
    conn.execute("PRAGMA query_only = ON")
    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return rows


def run_write(sql: str, params=()):
    """Used only by the migration script — never by pipeline/query code."""
    conn = _get_conn()
    conn.execute("PRAGMA query_only = OFF")
    conn.execute(sql, params)
    conn.commit()
    conn.execute("PRAGMA query_only = ON")
