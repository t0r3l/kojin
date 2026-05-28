"""User profile persistence.

Profiles are stored as JSON files in ``data/profiles/<username>.json``.
Each file contains the user's interaction history (last window_k accepted
recipe IDs + their NCF ratings) and the number of days completed, so the
RL policy can resume from the user's actual recent choices on the next login.

The ``data/profiles/`` directory is git-ignored (``data/`` is excluded).
"""

from __future__ import annotations

import json
import os
import re

import numpy as np

_PROFILES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "profiles")
_VALID_USERNAME = re.compile(r"^[a-zA-Z0-9_\-]{1,32}$")


def _profile_path(username: str) -> str:
    return os.path.join(_PROFILES_DIR, f"{username}.json")


def is_valid_username(username: str) -> bool:
    return bool(_VALID_USERNAME.match(username))


def profile_exists(username: str) -> bool:
    return is_valid_username(username) and os.path.exists(_profile_path(username))


def load_profile(username: str) -> dict | None:
    """Return profile dict or None if it doesn't exist."""
    if not profile_exists(username):
        return None
    with open(_profile_path(username)) as f:
        return json.load(f)


def save_profile(
    username: str,
    history: np.ndarray,
    history_ratings: np.ndarray,
    days_completed: int,
    user_embedding: np.ndarray | None = None,
) -> None:
    """Persist the user's interaction history and personalised embedding to disk."""
    if not is_valid_username(username):
        return
    os.makedirs(_PROFILES_DIR, exist_ok=True)
    data: dict = {
        "history": history.tolist(),
        "history_ratings": history_ratings.tolist(),
        "days_completed": days_completed,
    }
    if user_embedding is not None:
        data["user_embedding"] = user_embedding.tolist()
    with open(_profile_path(username), "w") as f:
        json.dump(data, f)


def list_profiles() -> list[str]:
    if not os.path.exists(_PROFILES_DIR):
        return []
    return [
        f[:-5]
        for f in os.listdir(_PROFILES_DIR)
        if f.endswith(".json") and is_valid_username(f[:-5])
    ]
