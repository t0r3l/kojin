"""Dataset loader for RecipeRL.

Two catalogue modes are supported:

* **Food.com mode** (primary) — real Food.com Kaggle recipes and real user
  ratings loaded by :mod:`reciperl.foodcom_data`. Requires the Kaggle zip at
  ``cfg.foodcom_zip_path``.
* **Recipe mode** (dev fallback) — the 30 hand-seeded recipes from
  ``data/recipes.json`` with Dirichlet-synthesised ratings. No external data
  needed; useful for quick iteration and CI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RecipeRLConfig
from .recipes import RecipeCatalogue, load_recipes


@dataclass
class RecipeRLDataset:
    """Materialised training arrays.

    Attributes
    ----------
    n_users, n_items, n_categories
        Vocabulary sizes.
    item_category
        ``(n_items,)`` int array — primary category id of every item.
    ratings
        ``(N, 3)`` float array of (user, item, rating in [0, 5]).
    train_mask, test_mask
        Boolean masks selecting train/test rows of ``ratings``.
    item_names
        Recipe name lookup, used for human-readable Top-k output.
    """

    n_users: int
    n_items: int
    n_categories: int
    item_category: np.ndarray
    ratings: np.ndarray
    train_mask: np.ndarray
    test_mask: np.ndarray
    item_names: list[str]


def synthesize_ratings(
    cfg: RecipeRLConfig,
    item_category: np.ndarray,
    n_categories: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample (user, item, rating) triples from a Dirichlet-mixture of categories.

    Used by the hand-seeded dev fallback. Each synthetic user draws a category
    preference vector from Dirichlet(0.5); ratings combine that affinity with
    Gaussian noise, clipped to [0, 5].
    """
    n_items = item_category.shape[0]
    user_prefs = rng.dirichlet(np.full(n_categories, 0.5), size=cfg.num_synthetic_users)

    triples: list[tuple[int, int, float]] = []
    sample_size = min(cfg.ratings_per_user, n_items)
    for u in range(cfg.num_synthetic_users):
        items = rng.choice(n_items, size=sample_size, replace=False)
        cat_aff = user_prefs[u, item_category[items]]
        ratings = 5.0 * cat_aff + rng.normal(0.0, 0.4, size=items.shape)
        ratings = np.clip(ratings, 0.0, 5.0)
        triples.extend((u, int(i), float(r)) for i, r in zip(items, ratings))

    return np.asarray(triples, dtype=np.float64)


def load_dataset(cfg: RecipeRLConfig) -> RecipeRLDataset:
    """Build a :class:`RecipeRLDataset` ready for NCF + PPO training.

    * ``cfg.use_foodcom=True`` → Food.com Kaggle dataset with real user ratings.
    * ``cfg.use_foodcom=False`` → hand-seeded ``data/recipes.json`` + Dirichlet ratings.
    """
    if cfg.use_foodcom:
        from .foodcom_data import load_foodcom_dataset
        dataset, _catalogue = load_foodcom_dataset(
            cfg.foodcom_zip_path, max_recipes=cfg.max_recipes
        )
        return dataset

    rng = np.random.default_rng(cfg.seed)
    catalogue: RecipeCatalogue = load_recipes()
    item_names = catalogue.names
    item_category = catalogue.category_ids
    n_categories = len(catalogue.categories)

    n_items = item_category.shape[0]
    ratings = synthesize_ratings(cfg, item_category, n_categories, rng)

    # Train/test split by user (§4.1).
    users = np.unique(ratings[:, 0].astype(np.int64))
    rng.shuffle(users)
    n_test = max(1, int(cfg.test_user_frac * users.size))
    test_users = set(users[:n_test].tolist())
    test_mask = np.array([int(u) in test_users for u in ratings[:, 0]])
    train_mask = ~test_mask

    return RecipeRLDataset(
        n_users=cfg.num_synthetic_users,
        n_items=n_items,
        n_categories=n_categories,
        item_category=item_category,
        ratings=ratings,
        train_mask=train_mask,
        test_mask=test_mask,
        item_names=item_names,
    )
