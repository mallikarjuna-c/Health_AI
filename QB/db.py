"""
Level 2: SQLite storage with hand-written SQL (no ORM).
"""

import os
import json
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions.db")


def get_connection():
    # new connection per request: sqlite3 connections should not be shared across threads
    return sqlite3.connect(DB_PATH)


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at       TEXT    NOT NULL,
                input_json       TEXT    NOT NULL,
                risk_probability REAL    NOT NULL,
                risk_level       TEXT    NOT NULL
            )
        """)


def save_prediction(patient_dict, prob, level):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO predictions (created_at, input_json, risk_probability, risk_level) "
            "VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), json.dumps(patient_dict), prob, level),
        )


def get_stats():
    with get_connection() as conn:
        total, avg_risk, high_share = conn.execute("""
            SELECT COUNT(*),
                   AVG(risk_probability),
                   AVG(CASE WHEN risk_level = 'High risk' THEN 1.0 ELSE 0.0 END)
            FROM predictions
        """).fetchone()

    # AVG returns NULL when the table is empty
    return {
        "total_requests": total,
        "average_risk": round(avg_risk, 3) if avg_risk is not None else 0.0,
        "high_risk_share": round(high_share, 3) if high_share is not None else 0.0,
    }
