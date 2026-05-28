"""End-to-end RecipeRL training CLI.

Steps:
1. Load catalogue & ratings (Food.com or hand-seeded fallback).
2. Train the Neural CF simulator (§3.2).
3. Build the fused state module (§3.3) warm-started from the NCF embeddings.
4. Train a residual PPO actor-critic on the simulated environment (§3.4).
5. Report Reward@T, Precision@T, NDCG@T on held-out users (§4.2).
6. Persist a checkpoint to ``cfg.checkpoint_path`` so the Streamlit app can
   load the trained policy without retraining.

Usage::

    python -m reciperl.train --foodcom --steps 200
    python -m reciperl.train --steps 50   # hand-seeded dev fallback
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .config import RecipeRLConfig
from .data import load_dataset
from .env import RecipeEnv
from .evaluate import eval_topk
from .ncf import precompute_rating_matrix, train_ncf
from .policy import ActorCritic
from .ppo import PPOTrainer
from .state import FusedState


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RecipeRL on the Kōjin catalogue.")
    p.add_argument("--steps", type=int, default=None,
                   help="Override config.total_updates (number of PPO updates).")
    p.add_argument("--top-k", type=int, default=None,
                   help="Override config.top_k for evaluation.")
    p.add_argument("--device", default=None,
                   help="Override config.device (e.g. 'cuda').")
    p.add_argument("--max-recipes", type=int, default=None,
                   help="Override config.max_recipes (cap on Food.com recipes loaded).")
    p.add_argument("--foodcom", action="store_true",
                   help="Train on the Food.com Kaggle dataset with real user ratings.")
    p.add_argument("--foodcom-zip", default=None,
                   help="Path to the Food.com zip (default: data/foodcom-recipes-and-reviews.zip).")
    p.add_argument("--checkpoint", default=None,
                   help="Override config.checkpoint_path.")
    return p.parse_args()


def save_checkpoint(
    path: str,
    cfg: RecipeRLConfig,
    state_module: torch.nn.Module,
    policy: ActorCritic,
    rating_matrix: np.ndarray,
    item_names: list[str],
    recipe_catalogue=None,
) -> None:
    """Persist everything the app needs to reload the policy and simulator."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    blob: dict = {
        "config": cfg.__dict__,
        "state_module": state_module.state_dict(),
        "policy": policy.state_dict(),
        "rating_matrix": rating_matrix,
        "item_names": item_names,
    }
    if recipe_catalogue is not None:
        from .foodcom_data import catalogue_to_blob
        blob["recipe_catalogue"] = catalogue_to_blob(recipe_catalogue)
    torch.save(blob, path)
    print(f"[checkpoint] saved to {path}")


def main() -> None:
    args = parse_args()
    cfg = RecipeRLConfig()
    if args.steps is not None:
        cfg.total_updates = args.steps
    if args.top_k is not None:
        cfg.top_k = args.top_k
    if args.device is not None:
        cfg.device = args.device
    if args.max_recipes is not None:
        cfg.max_recipes = args.max_recipes
    if args.foodcom:
        cfg.use_foodcom = True
    if args.foodcom_zip is not None:
        cfg.foodcom_zip_path = args.foodcom_zip
    if args.checkpoint is not None:
        cfg.checkpoint_path = args.checkpoint

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device)

    mode = "foodcom" if cfg.use_foodcom else "recipes (hand-seeded fallback)"
    print(f"[1/4] Loading catalogue ({mode})…")

    foodcom_catalogue = None
    if cfg.use_foodcom:
        from .foodcom_data import load_foodcom_dataset
        dataset, foodcom_catalogue = load_foodcom_dataset(
            cfg.foodcom_zip_path, max_recipes=cfg.max_recipes
        )
    else:
        dataset = load_dataset(cfg)
    print(f"      users={dataset.n_users}  items={dataset.n_items}  "
          f"categories={dataset.n_categories}  ratings={len(dataset.ratings)}")

    print("[2/4] Training Neural Collaborative Filtering simulator (§3.2)…")
    ncf = train_ncf(dataset, cfg, device)
    rating_matrix = precompute_rating_matrix(ncf, dataset.n_users, dataset.n_items, device)

    print("[3/4] Building fused state representation (§3.3)…")
    state_module = FusedState(dataset.n_users, dataset.n_items, dataset.n_categories, cfg).to(device)
    state_module.warmstart_from_ncf(ncf.user_emb, ncf.item_emb)

    env = RecipeEnv(dataset, rating_matrix, state_module, cfg, device)
    policy = ActorCritic(state_module.out_dim, dataset.n_items).to(device)

    print("[4/4] Training Residual PPO (§3.4)…")
    trainer = PPOTrainer(env, state_module, policy, cfg, device)
    for update in range(cfg.total_updates):
        roll = trainer.collect(rng)
        stats = trainer.update(roll)
        ep_reward = float(np.mean(roll.rewards))
        print(
            f"  update {update + 1:3d}/{cfg.total_updates}  "
            f"avg_r={ep_reward:+.3f}  "
            f"pi_loss={stats['policy_loss']:+.3f}  "
            f"v_loss={stats['value_loss']:.3f}  "
            f"H={stats['entropy']:.2f}"
        )

    print("\n=== Evaluation (§4.2) ===")
    metrics = eval_topk(env, policy, cfg, rng)
    for name, value in metrics.items():
        print(f"  {name:14s} = {value:.4f}")

    save_checkpoint(
        cfg.checkpoint_path, cfg, state_module, policy, rating_matrix,
        dataset.item_names, recipe_catalogue=foodcom_catalogue,
    )


if __name__ == "__main__":
    main()
