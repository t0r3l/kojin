"""Evaluation metrics — §4.2 of Liu et al. (2024).

* Reward@T  (Eq. 18)
* Precision@T  (Eq. 19)
* NDCG@T  (Eqs. 20-21) — binary relevance, threshold ``positive_threshold``.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .config import RecipeRLConfig
from .env import RecipeEnv
from .policy import ActorCritic


def _dcg(rels: list[int]) -> float:
    return sum(r / math.log2(t + 2) for t, r in enumerate(rels))


@torch.no_grad()
def eval_topk(
    env: RecipeEnv,
    policy: ActorCritic,
    cfg: RecipeRLConfig,
    rng: np.random.Generator,
    n_episodes: int = 32,
) -> dict[str, float]:
    """Run ``n_episodes`` greedy rollouts of length T and return the metrics."""
    rewards: list[float] = []
    hits_at_T: list[int] = []
    ndcgs: list[float] = []

    for _ in range(n_episodes):
        state = env.reset(rng=rng)
        ep_rewards: list[float] = []
        ep_rels: list[int] = []
        done = False
        while not done:
            dist, _ = policy(state.unsqueeze(0))
            action = int(dist.probs.argmax(dim=-1).item())  # greedy
            state, reward, done, info = env.step(action)
            ep_rewards.append(reward)
            ep_rels.append(int(info["hit"]))
        rewards.append(float(np.mean(ep_rewards)))
        hits_at_T.append(int(np.sum(ep_rels)))
        ideal = sorted(ep_rels, reverse=True)
        idcg = _dcg(ideal)
        ndcgs.append(_dcg(ep_rels) / idcg if idcg > 0 else 0.0)

    T = cfg.horizon_T
    return {
        "reward@T": float(np.mean(rewards)),
        "precision@T": float(np.mean(hits_at_T)) / T,
        "ndcg@T": float(np.mean(ndcgs)),
    }
