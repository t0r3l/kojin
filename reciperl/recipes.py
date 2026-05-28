"""Recipe catalogue loader.

Provides typed access to ``data/recipes.json`` (seeded by hand). Recipes are
treated as RL *actions*: each recipe becomes one item in the catalogue, with
its abstract ingredient list, dietary tags, category/cuisine vocabulary and
per-serving macro targets.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

# Path mirrors the convention in ``kojin_common`` to avoid duplication.
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RECIPES_PATH = os.path.join(DATA_DIR, "recipes.json")


@dataclass
class RecipeIngredient:
    name: str
    approx_g: float
    role: str  # protein | carb | fat | veg | ingredient
    proportion: float = 0.0  # weight fraction of this ingredient in the recipe (0–1); 0 = unknown


@dataclass
class Recipe:
    id: int
    name: str
    category: str          # plat | petit_dej | snack | dessert
    cuisine: str
    tags: list[str]        # halal, vegan, vegetarian, gluten_free, …
    ingredients: list[RecipeIngredient]
    macros_per_serving: dict[str, float]  # kcal, prot, fat, carb


@dataclass
class RecipeCatalogue:
    recipes: list[Recipe]
    categories: list[str]
    cuisines: list[str]

    def __len__(self) -> int:
        return len(self.recipes)

    @property
    def names(self) -> list[str]:
        return [r.name for r in self.recipes]

    @property
    def category_ids(self) -> np.ndarray:
        """Integer index of the recipe's category in ``self.categories``."""
        idx = {c: i for i, c in enumerate(self.categories)}
        return np.asarray([idx[r.category] for r in self.recipes], dtype=np.int64)

    def filter_by_tag(self, tag: str | None) -> list[int]:
        """Return recipe ids compatible with ``tag`` (e.g. 'vegan'). ``None`` = no filter."""
        if not tag:
            return [r.id for r in self.recipes]
        return [r.id for r in self.recipes if tag in r.tags]


def load_recipes(path: str = RECIPES_PATH) -> RecipeCatalogue:
    """Read the JSON seed file and return a :class:`RecipeCatalogue`."""
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)

    recipes: list[Recipe] = []
    for r in blob["recipes"]:
        recipes.append(
            Recipe(
                id=int(r["id"]),
                name=str(r["name"]),
                category=str(r["category"]),
                cuisine=str(r["cuisine"]),
                tags=list(r.get("tags", [])),
                ingredients=[RecipeIngredient(**i) for i in r["ingredients"]],
                macros_per_serving={k: float(v) for k, v in r["macros_per_serving"].items()},
            )
        )
    # Validate id contiguity to keep mapping recipe.id == catalogue index.
    for expected, rec in enumerate(recipes):
        if rec.id != expected:
            raise ValueError(f"Recipe id must be contiguous; got {rec.id} at index {expected}.")
    return RecipeCatalogue(
        recipes=recipes,
        categories=list(blob["categories"]),
        cuisines=list(blob["cuisines"]),
    )
