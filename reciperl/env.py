"""Interactive food recommendation environment — §3.2 of Liu et al. (2024).

A gym-like wrapper around the pre-trained NCF simulator. The environment
exposes the *fused* user state described in §3.3 to the agent and emits
rewards combining empirical user feedback with a sequential bonus
(Eq. 6).
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from .config import RecipeRLConfig
from .data import RecipeRLDataset
from .state import FusedState


class RecipeEnv:
    """Single-user, multi-step recommendation environment.

    The agent observes the current fused state ``sₜ`` and outputs an
    action ``aₜ ∈ {0, …, n_items-1}``. The environment then:

    1. Looks up the simulated user feedback ``fᵢⱼ ∈ [0, 5]``.
    2. Updates the per-item recommendation counter (used by the
       forgetting score λ_r).
    3. Maintains the sliding window of the last ``k`` (item, rating) pairs.
    4. Returns the new state and reward
       ``r = (fᵢⱼ - 2.5)/2.5 + α (c_p - c_n)`` (Eq. 6).
    """

    def __init__(
        self,
        dataset: RecipeRLDataset,
        rating_matrix: np.ndarray,
        state_module: FusedState,
        cfg: RecipeRLConfig,
        device: torch.device,
    ):
        self.cfg = cfg
        self.device = device
        self.state_module = state_module
        self.dataset = dataset
        self.rating_matrix = rating_matrix
        self.n_items = dataset.n_items
        self.k = cfg.window_k

        self._user: int = 0
        self._history: deque[tuple[int, float]] = deque(maxlen=self.k)
        self._rec_counts: np.ndarray = np.zeros(self.n_items, dtype=np.int64)
        self._cp = 0  # consecutive positives
        self._cn = 0  # consecutive negatives
        self._step = 0

    # ── lifecycle ──────────────────────────────────────────────────────
    def reset(self, user: int | None = None, rng: np.random.Generator | None = None) -> torch.Tensor:
        rng = rng or np.random.default_rng()
        self._user = int(rng.integers(self.dataset.n_users)) if user is None else int(user)
        self._rec_counts[:] = 0
        self._cp = self._cn = 0
        self._step = 0

        # Seed the history with the user's strongest known interactions, to
        # provide a non-trivial cold-start state (mirrors §4.1).
        self._history.clear()
        user_rows = self.dataset.ratings[self.dataset.ratings[:, 0] == self._user]
        if user_rows.size:
            top = user_rows[np.argsort(-user_rows[:, 2])[: self.k]]
            for item_id, rating in zip(top[:, 1].astype(int), top[:, 2]):
                self._history.append((int(item_id), float(rating)))
        # Pad with item 0 / rating 0 if necessary.
        while len(self._history) < self.k:
            self._history.appendleft((0, 0.0))

        return self._observe()

    # ── core step ──────────────────────────────────────────────────────
    def step(self, action: int) -> tuple[torch.Tensor, float, bool, dict]:
        rating = float(self.rating_matrix[self._user, action])
        self._rec_counts[action] += 1
        reward = self._reward(rating)
        self._history.append((int(action), rating))
        self._step += 1
        done = self._step >= self.cfg.horizon_T
        info = {"rating": rating, "hit": rating >= self.cfg.positive_threshold}
        return self._observe(), reward, done, info

    def _reward(self, rating: float) -> float:
        # Empirical reward normalised to [-1, 1] (§3.2).
        empirical = (rating - 2.5) / 2.5
        # Sequential bonus tracks streaks of positive / negative feedback.
        if rating >= self.cfg.positive_threshold:
            self._cp += 1
            self._cn = 0
        else:
            self._cn += 1
            self._cp = 0
        sequential = self._cp - self._cn
        return float(empirical + self.cfg.alpha_sequential * sequential)

    # ── observation building ───────────────────────────────────────────
    def _observe(self) -> torch.Tensor:
        items = torch.as_tensor([i for i, _ in self._history], dtype=torch.long, device=self.device)
        ratings = torch.as_tensor([r for _, r in self._history], dtype=torch.float32, device=self.device)
        # Per-window recommendation counts (paper sliding window context).
        counts = torch.as_tensor(
            [self._rec_counts[i] for i, _ in self._history],
            dtype=torch.float32,
            device=self.device,
        )
        last_item = items[-1]
        last_cat = torch.as_tensor(
            self.dataset.item_category[int(last_item)], dtype=torch.long, device=self.device
        )
        user = torch.as_tensor(self._user, dtype=torch.long, device=self.device)
        with torch.no_grad():
            state = self.state_module(
                user.unsqueeze(0),
                items.unsqueeze(0),
                ratings.unsqueeze(0),
                counts.unsqueeze(0),
                last_item.unsqueeze(0),
                last_cat.unsqueeze(0),
            )
        return state.squeeze(0)
