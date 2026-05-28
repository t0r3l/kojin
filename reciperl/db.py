"""SQLite persistence layer for Kōjin user profiles.

Tables
------
users       — identity + nutrition profile
meal_slots  — per-user configurable meal plan (ordered slots with kcal fraction)
user_rl     — RL state: interaction history + personalised user embedding
"""

from __future__ import annotations

import os
import pickle
import re
import sqlite3
from contextlib import contextmanager
from typing import Generator

import numpy as np

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "kojin.db")
_VALID_USERNAME = re.compile(r"^[a-zA-Z0-9_\-]{1,32}$")

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS users (
    username        TEXT PRIMARY KEY,
    gender          TEXT    NOT NULL DEFAULT 'Homme',
    age             INTEGER NOT NULL DEFAULT 25,
    weight          REAL    NOT NULL DEFAULT 75.0,
    height          REAL    NOT NULL DEFAULT 175.0,
    daily_activity  TEXT    NOT NULL DEFAULT 'Sédentaire (bureau, peu de marche)',
    sport           TEXT    NOT NULL DEFAULT 'Aucun',
    goal            TEXT    NOT NULL DEFAULT 'balanced',
    regime          TEXT    NOT NULL DEFAULT '',
    days_completed  INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS meal_slots (
    username    TEXT    NOT NULL REFERENCES users(username) ON DELETE CASCADE,
    slot_order  INTEGER NOT NULL,
    label       TEXT    NOT NULL,
    category    TEXT    NOT NULL,
    fraction    REAL    NOT NULL,
    PRIMARY KEY (username, slot_order)
);

CREATE TABLE IF NOT EXISTS user_rl (
    username         TEXT PRIMARY KEY REFERENCES users(username) ON DELETE CASCADE,
    history          BLOB,
    history_ratings  BLOB,
    user_embedding   BLOB,
    updated_at       TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_DEFAULT_MEAL_SLOTS = [
    {"slot_order": 0, "label": "Petit-déjeuner", "category": "petit_dej", "fraction": 0.25},
    {"slot_order": 1, "label": "Déjeuner",       "category": "plat",      "fraction": 0.35},
    {"slot_order": 2, "label": "Goûter",          "category": "snack",     "fraction": 0.10},
    {"slot_order": 3, "label": "Dîner",           "category": "plat",      "fraction": 0.30},
]


def is_valid_username(username: str) -> bool:
    return bool(_VALID_USERNAME.match(username))


@contextmanager
def _connect() -> Generator[sqlite3.Connection, None, None]:
    os.makedirs(os.path.dirname(os.path.abspath(_DB_PATH)), exist_ok=True)
    con = sqlite3.connect(_DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db() -> None:
    with _connect() as con:
        con.executescript(_SCHEMA)


def user_exists(username: str) -> bool:
    with _connect() as con:
        row = con.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone()
    return row is not None


def create_user(username: str) -> None:
    with _connect() as con:
        con.execute("INSERT OR IGNORE INTO users (username) VALUES (?)", (username,))
        existing = con.execute(
            "SELECT COUNT(*) FROM meal_slots WHERE username=?", (username,)
        ).fetchone()[0]
        if existing == 0:
            con.executemany(
                "INSERT INTO meal_slots (username, slot_order, label, category, fraction) "
                "VALUES (?, ?, ?, ?, ?)",
                [(username, s["slot_order"], s["label"], s["category"], s["fraction"])
                 for s in _DEFAULT_MEAL_SLOTS],
            )


def get_user(username: str) -> dict | None:
    with _connect() as con:
        row = con.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    return dict(row) if row else None


def save_user(username: str, **fields) -> None:
    """Upsert user nutrition profile fields."""
    allowed = {"gender", "age", "weight", "height", "daily_activity", "sport", "goal", "regime", "days_completed"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return
    set_clause = ", ".join(f"{k}=?" for k in updates)
    values = list(updates.values()) + [username]
    with _connect() as con:
        con.execute(
            f"INSERT OR IGNORE INTO users (username) VALUES (?)", (username,)
        )
        con.execute(
            f"UPDATE users SET {set_clause}, updated_at=datetime('now') WHERE username=?",
            values,
        )


def get_meal_slots(username: str) -> list[dict]:
    with _connect() as con:
        rows = con.execute(
            "SELECT slot_order, label, category, fraction FROM meal_slots "
            "WHERE username=? ORDER BY slot_order",
            (username,),
        ).fetchall()
    return [dict(r) for r in rows] if rows else list(_DEFAULT_MEAL_SLOTS)


def save_meal_slots(username: str, slots: list[dict]) -> None:
    """Replace all meal slots for a user."""
    with _connect() as con:
        con.execute("DELETE FROM meal_slots WHERE username=?", (username,))
        con.executemany(
            "INSERT INTO meal_slots (username, slot_order, label, category, fraction) "
            "VALUES (?, ?, ?, ?, ?)",
            [(username, i, s["label"], s["category"], s["fraction"])
             for i, s in enumerate(slots)],
        )


def get_rl_state(username: str) -> dict | None:
    with _connect() as con:
        row = con.execute("SELECT * FROM user_rl WHERE username=?", (username,)).fetchone()
    if row is None:
        return None
    result: dict = {}
    if row["history"] is not None:
        result["history"] = np.frombuffer(row["history"], dtype=np.int64).copy()
    if row["history_ratings"] is not None:
        result["history_ratings"] = np.frombuffer(row["history_ratings"], dtype=np.float32).copy()
    if row["user_embedding"] is not None:
        result["user_embedding"] = np.frombuffer(row["user_embedding"], dtype=np.float32).copy()
    return result or None


def save_rl_state(
    username: str,
    history: np.ndarray,
    history_ratings: np.ndarray,
    days_completed: int,
    user_embedding: np.ndarray | None = None,
) -> None:
    history_blob = history.astype(np.int64).tobytes()
    ratings_blob = history_ratings.astype(np.float32).tobytes()
    emb_blob = user_embedding.astype(np.float32).tobytes() if user_embedding is not None else None
    with _connect() as con:
        con.execute("INSERT OR IGNORE INTO users (username) VALUES (?)", (username,))
        con.execute(
            "UPDATE users SET days_completed=?, updated_at=datetime('now') WHERE username=?",
            (days_completed, username),
        )
        con.execute(
            "INSERT INTO user_rl (username, history, history_ratings, user_embedding, updated_at) "
            "VALUES (?, ?, ?, ?, datetime('now')) "
            "ON CONFLICT(username) DO UPDATE SET "
            "history=excluded.history, history_ratings=excluded.history_ratings, "
            "user_embedding=excluded.user_embedding, updated_at=excluded.updated_at",
            (username, history_blob, ratings_blob, emb_blob),
        )


def migrate_from_json() -> int:
    """Migrate existing JSON profiles to SQLite. Returns number migrated."""
    import json

    profiles_dir = os.path.join(os.path.dirname(__file__), "..", "data", "profiles")
    if not os.path.isdir(profiles_dir):
        return 0
    migrated = 0
    for fname in os.listdir(profiles_dir):
        if not fname.endswith(".json"):
            continue
        username = fname[:-5]
        if not is_valid_username(username):
            continue
        if user_exists(username):
            continue
        try:
            with open(os.path.join(profiles_dir, fname)) as f:
                data = json.load(f)
            create_user(username)
            history = np.array(data.get("history", []), dtype=np.int64)
            ratings = np.array(data.get("history_ratings", []), dtype=np.float32)
            days = int(data.get("days_completed", 0))
            emb_list = data.get("user_embedding")
            emb = np.array(emb_list, dtype=np.float32) if emb_list else None
            save_rl_state(username, history, ratings, days, user_embedding=emb)
            migrated += 1
        except Exception:
            continue
    return migrated
