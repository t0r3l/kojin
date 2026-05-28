"""Default hyperparameters for RecipeRL.

The values follow §4.4 of Liu et al. (2024) where applicable, and otherwise
default to widely-used PPO settings.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RecipeRLConfig:
    # ── Catalogue mode ──────────────────────────────────────────────────
    # use_foodcom=True  ⇒ Food.com Kaggle dataset (real ratings, primary mode).
    # use_foodcom=False ⇒ hand-seeded data/recipes.json + Dirichlet ratings
    #                     (dev fallback, no external data required).
    use_recipes: bool = True          # kept for checkpoint compatibility
    use_foodcom: bool = False
    foodcom_zip_path: str = "data/foodcom-recipes-and-reviews.zip"
    checkpoint_path: str = "data/reciperl.pt"

    # ── Data ────────────────────────────────────────────────────────────
    max_recipes: int = 2000           # cap on Food.com recipes loaded for training
    num_synthetic_users: int = 512    # synthetic users for the dev-fallback mode
    ratings_per_user: int = 40        # ratings sampled per synthetic user
    test_user_frac: float = 0.2

    # ── Neural CF simulator (§3.2) ──────────────────────────────────────
    cf_embed_dim: int = 32
    cf_hidden: tuple[int, ...] = (64, 32, 16)
    cf_epochs: int = 5
    cf_batch_size: int = 1024
    cf_lr: float = 1e-3

    # ── State representation (§3.3) ─────────────────────────────────────
    window_k: int = 10
    attn_heads: int = 4
    gaussian_noise_std: float = 0.05

    # ── Reward (§3.2, Eq. 6) ────────────────────────────────────────────
    alpha_sequential: float = 0.1   # α ∈ {0.0, 0.1, 0.2}
    positive_threshold: float = 3.0  # rating ≥ 3 ⇒ positive

    # ── PPO (§3.4) ──────────────────────────────────────────────────────
    horizon_T: int = 32             # episode length, paper uses T = 32
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ppo_epochs: int = 4
    ppo_batch_size: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # ── Training schedule ───────────────────────────────────────────────
    rollout_episodes: int = 16
    total_updates: int = 50
    seed: int = 42
    device: str = "cpu"

    # ── Evaluation (§4.2) ───────────────────────────────────────────────
    top_k: int = 10
