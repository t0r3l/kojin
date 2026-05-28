"""Food.com / Kaggle dataset loader.

Parses the Kaggle ``foodcom-recipes-and-reviews`` zip (``recipes.csv`` +
``reviews.csv``) into the structures used by the RecipeRL pipeline:

* :class:`~reciperl.data.RecipeRLDataset` — for NCF + PPO training with real
  user ratings (§4.1 of Liu et al., 2024).
* :class:`~reciperl.recipes.RecipeCatalogue` — for inference-time recipe
  selection, with ``proportion`` fields derived from
  ``RecipeIngredientQuantities``.

The zip is extracted lazily on first call; subsequent calls reuse the
already-extracted CSVs.
"""
from __future__ import annotations

import os
import re
import zipfile

import numpy as np
import pandas as pd

from .recipes import Recipe, RecipeCatalogue, RecipeIngredient


# ── Kōjin category vocabulary ────────────────────────────────────────────────

# Desserts are excluded from the catalogue entirely (see _EXCLUDE_CATEGORIES).
_KŌJIN_CATEGORIES: list[str] = ["plat", "petit_dej", "snack"]
_CAT_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(_KŌJIN_CATEGORIES)}

# ── Explicit category sets ────────────────────────────────────────────────────
# Built from the complete list of 311 Food.com RecipeCategory values.
# Exact string matching — no substring ambiguity (e.g. "Savory Pies" ≠ "Pie").

# Recipes in these categories are dropped at load time.
_EXCLUDE_CATEGORIES: frozenset[str] = frozenset({
    # Sweet desserts
    "Dessert", "Pie", "Bar Cookie", "Drop Cookies", "Candy",
    "Frozen Desserts", "Cheesecake", "Tarts", "Gelatin",
    "Chocolate Chip Cookies", "Ice Cream",
    "Key Lime Pie", "Desserts Fruit", "Peanut Butter Pie",
    "Apple Pie", "Coconut Cream Pie", "Lemon Cake", "Bread Pudding",
    "Snacks Sweet",
    # Non-food items
    "Bath/Beauty", "Household Cleaner", "Homeopathy/Remedies",
})

# Recipes in these categories are always mapped to "petit_dej".
_BREAKFAST_CATEGORIES: frozenset[str] = frozenset({
    "Breakfast", "Brunch", "Breads", "Quick Breads", "Yeast Breads",
    "Sourdough Breads", "Scones", "Oatmeal", "Bread Machine",
    "Buttermilk Biscuits", "Wheat Bread", "Breakfast Eggs",
    "Breakfast Casseroles",
})

# Recipes in these categories are always mapped to "snack".
_SNACK_CATEGORIES: frozenset[str] = frozenset({
    "Beverages", "Smoothies", "Punch Beverage", "Shakes",
    "Spreads", "Chutneys", "Jellies",
})

# Keyword fallback for unmapped Food.com categories (e.g. cuisine or
# time-based categories whose name still hints at meal type).
# Positive classification only — no keyword-based exclusion to avoid
# false positives like "Salmon Cake" or "Fish Cake".
_CATEGORY_RULES: list[tuple[str, str]] = [
    ("breakfast", "petit_dej"),
    ("brunch", "petit_dej"),
    ("morning", "petit_dej"),
    ("pancake", "petit_dej"),
    ("waffle", "petit_dej"),
    ("scone", "petit_dej"),
    ("crepe", "petit_dej"),
    ("bread", "petit_dej"),
    ("snack", "snack"),
    ("appetizer", "snack"),
    ("beverage", "snack"),
    ("drink", "snack"),
    ("smoothie", "snack"),
    ("muffin", "snack"),
]


def _map_category(category_raw: str, name_raw: str = "") -> str:
    """Map a Food.com RecipeCategory to a Kōjin category, or "exclude".

    Priority:
    1. Exact exclusion set  → "exclude"
    2. Exact breakfast set  → "petit_dej"
    3. Exact snack set      → "snack"
    4. Keyword on category string (for unmapped categories)
    5. Keyword on recipe name (positive only — avoids "Salmon Cake" false-positives)
    6. Default             → "plat"
    """
    cat = str(category_raw).strip()

    if cat in _EXCLUDE_CATEGORIES:
        return "exclude"
    if cat in _BREAKFAST_CATEGORIES:
        return "petit_dej"
    if cat in _SNACK_CATEGORIES:
        return "snack"

    low = cat.lower()
    for keyword, result in _CATEGORY_RULES:
        if keyword in low:
            return result

    name_low = str(name_raw).lower()
    for keyword, result in _CATEGORY_RULES:
        if keyword in name_low:
            return result

    return "plat"


def recategorize_catalogue(catalogue: RecipeCatalogue, csv_path: str | None = None) -> None:
    """Re-apply category mapping to all recipes in the catalogue (in-place).

    Used to fix stale categories in checkpoints trained with an older mapping.
    Excluded categories are reclassified as "plat" (they should have been
    filtered out during training; keeping them out of dessert slots is enough).
    """
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path, usecols=["Name", "RecipeCategory"], on_bad_lines="skip")
        df = df.dropna(subset=["Name", "RecipeCategory"]).reset_index(drop=True)
        for recipe in catalogue.recipes:
            if recipe.id < len(df):
                row = df.iloc[recipe.id]
                mapped = _map_category(str(row["RecipeCategory"]), str(row["Name"]))
                recipe.category = "plat" if mapped == "exclude" else mapped
    # Name-based pass for any remaining recipes without a clear category.
    for recipe in catalogue.recipes:
        if recipe.category not in _KŌJIN_CATEGORIES:
            recipe.category = "plat"


# ── R-vector parser ──────────────────────────────────────────────────────────

_R_QUOTED_RE = re.compile(r'"([^"]*)"')


def _parse_r_vector(s: object) -> list[str]:
    """Parse an R-style ``c("a", "b", "c")`` string into a Python list."""
    if not isinstance(s, str) or not s.strip():
        return []
    return _R_QUOTED_RE.findall(s)


# ── Quantity parser ──────────────────────────────────────────────────────────

_FRACTION_RE = re.compile(r"(\d+)?\s*(\d+)/(\d+)")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _parse_single_quantity(s: str) -> float:
    """Convert a quantity string (``'1'``, ``'1/2'``, ``'1 1/2'``) to float.

    Returns 1.0 for anything that cannot be parsed (equal-weight fallback).
    """
    s = s.strip()
    if not s:
        return 1.0
    # Try mixed fraction first: "1 1/2"
    m = _FRACTION_RE.search(s)
    if m:
        whole = int(m.group(1)) if m.group(1) else 0
        num = int(m.group(2))
        den = int(m.group(3))
        return whole + (num / den if den else 1.0)
    # Try plain number
    m2 = _NUMBER_RE.search(s)
    if m2:
        try:
            return float(m2.group().replace(",", "."))
        except ValueError:
            pass
    return 1.0


def _parse_quantities(parts: list[str]) -> np.ndarray:
    """Convert a list of quantity strings to a dimensionless float array."""
    if not parts:
        return np.ones(1)
    return np.array([_parse_single_quantity(p) for p in parts], dtype=float)


def _quantities_to_proportions(qtys: np.ndarray) -> np.ndarray:
    """Normalise quantity array to proportions summing to 1."""
    total = qtys.sum()
    if total <= 0.0:
        n = max(len(qtys), 1)
        return np.full(n, 1.0 / n)
    return qtys / total


# ── Tag inference ────────────────────────────────────────────────────────────

_VEGAN_KW = frozenset({"vegan", "plant-based", "plant based"})
_VEG_KW = frozenset({"vegetarian", "veggie", "meatless"})
_HALAL_KW = frozenset({"halal"})
_GF_KW = frozenset({"gluten-free", "gluten free", "gluten_free"})


def _infer_tags(keywords_raw: object, category_raw: object) -> list[str]:
    low = (str(keywords_raw) + " " + str(category_raw)).lower()
    tags: list[str] = []
    if any(k in low for k in _VEGAN_KW):
        tags += ["vegan", "vegetarian"]
    elif any(k in low for k in _VEG_KW):
        tags.append("vegetarian")
    if any(k in low for k in _HALAL_KW):
        tags.append("halal")
    if any(k in low for k in _GF_KW):
        tags.append("gluten_free")
    return tags


# ── ZIP extraction ───────────────────────────────────────────────────────────

def _ensure_extracted(zip_path: str) -> tuple[str, str]:
    """Extract ``recipes.csv`` + ``reviews.csv`` from the zip if needed.

    Returns ``(recipes_csv_path, reviews_csv_path)``.
    """
    dest_dir = os.path.dirname(os.path.abspath(zip_path))
    recipes_csv = os.path.join(dest_dir, "recipes.csv")
    reviews_csv = os.path.join(dest_dir, "reviews.csv")

    if os.path.exists(recipes_csv) and os.path.exists(reviews_csv):
        return recipes_csv, reviews_csv

    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"Food.com zip not found at {zip_path!r}.\n"
            "Download it from https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews "
            "and place it at that path."
        )

    print(f"[foodcom] Extracting {zip_path} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            basename = os.path.basename(name)
            if basename in ("recipes.csv", "reviews.csv"):
                target = os.path.join(dest_dir, basename)
                with zf.open(name) as src, open(target, "wb") as dst:
                    dst.write(src.read())

    return recipes_csv, reviews_csv


# ── Recipe loader ─────────────────────────────────────────────────────────────

_RECIPE_COLS = [
    "RecipeId", "Name", "RecipeCategory", "Keywords",
    "RecipeIngredientParts", "RecipeIngredientQuantities",
    "Calories", "ProteinContent", "FatContent",
    "CarbohydrateContent", "FiberContent",
]


def load_foodcom_recipes(
    zip_path: str,
    max_recipes: int | None = None,
) -> tuple[list[Recipe], list[str]]:
    """Load Food.com recipes and return ``(recipes, categories)``.

    Each :class:`~reciperl.recipes.Recipe` has real ingredient names and
    ``proportion`` fields derived from ``RecipeIngredientQuantities``.
    Recipe ids are contiguous from 0.
    """
    recipes_csv, _ = _ensure_extracted(zip_path)
    df = pd.read_csv(recipes_csv, usecols=_RECIPE_COLS, on_bad_lines="skip")
    df = df.dropna(subset=["Name", "RecipeCategory"])
    if max_recipes is not None:
        df = df.head(max_recipes)

    # Filter excluded categories (desserts, non-food) before assigning IDs
    # so that recipe IDs are contiguous and match the rating matrix columns.
    keep = [
        _map_category(str(row["RecipeCategory"]), str(row["Name"])) != "exclude"
        for _, row in df[["RecipeCategory", "Name"]].iterrows()
    ]
    df = df[keep].reset_index(drop=True)
    print(f"[foodcom] {sum(keep)} / {len(keep)} recipes kept after excluding desserts/non-food.")

    recipes: list[Recipe] = []
    for new_id, (_, row) in enumerate(df.iterrows()):
        ing_parts = _parse_r_vector(row.get("RecipeIngredientParts", ""))
        ing_qtys_raw = _parse_r_vector(row.get("RecipeIngredientQuantities", ""))

        n_ing = len(ing_parts)
        if n_ing == 0:
            ing_parts = ["ingredient"]
            ing_qtys_raw = ["1"]
            n_ing = 1

        # Align length: pad missing quantities with "1", truncate excess.
        while len(ing_qtys_raw) < n_ing:
            ing_qtys_raw.append("1")
        ing_qtys_raw = ing_qtys_raw[:n_ing]

        qtys = _parse_quantities(ing_qtys_raw)
        props = _quantities_to_proportions(qtys)

        ingredients = [
            RecipeIngredient(
                name=ing_parts[i],
                approx_g=float(props[i] * 300),  # placeholder: 300 g serving total
                role="ingredient",
                proportion=float(props[i]),
            )
            for i in range(n_ing)
        ]

        category = _map_category(row.get("RecipeCategory", ""), str(row["Name"]))
        tags = _infer_tags(row.get("Keywords", ""), row.get("RecipeCategory", ""))

        macros = {
            "kcal": float(row.get("Calories") or 0.0),
            "prot": float(row.get("ProteinContent") or 0.0),
            "fat": float(row.get("FatContent") or 0.0),
            "carb": float(row.get("CarbohydrateContent") or 0.0),
        }

        recipes.append(Recipe(
            id=new_id,
            name=str(row["Name"]),
            category=category,
            cuisine="international",
            tags=tags,
            ingredients=ingredients,
            macros_per_serving=macros,
        ))

    return recipes, _KŌJIN_CATEGORIES


# ── Catalogue helpers ─────────────────────────────────────────────────────────

def build_catalogue(recipes: list[Recipe], categories: list[str]) -> RecipeCatalogue:
    """Wrap a recipe list into a :class:`RecipeCatalogue`."""
    return RecipeCatalogue(recipes=recipes, categories=categories, cuisines=["international"])


def catalogue_to_blob(catalogue: RecipeCatalogue) -> list[dict]:
    """Serialise a catalogue to a JSON-safe list of dicts for checkpoint storage."""
    return [
        {
            "id": r.id,
            "name": r.name,
            "category": r.category,
            "cuisine": r.cuisine,
            "tags": r.tags,
            "ingredients": [
                {
                    "name": i.name,
                    "approx_g": i.approx_g,
                    "role": i.role,
                    "proportion": i.proportion,
                }
                for i in r.ingredients
            ],
            "macros_per_serving": r.macros_per_serving,
        }
        for r in catalogue.recipes
    ]


def catalogue_from_blob(
    blob: list[dict],
    categories: list[str] | None = None,
) -> RecipeCatalogue:
    """Reconstruct a :class:`RecipeCatalogue` from a checkpoint blob."""
    if categories is None:
        categories = _KŌJIN_CATEGORIES
    recipes = [
        Recipe(
            id=int(r["id"]),
            name=str(r["name"]),
            category=str(r["category"]),
            cuisine=str(r.get("cuisine", "international")),
            tags=list(r.get("tags", [])),
            ingredients=[RecipeIngredient(**i) for i in r["ingredients"]],
            macros_per_serving={k: float(v) for k, v in r["macros_per_serving"].items()},
        )
        for r in blob
    ]
    return RecipeCatalogue(recipes=recipes, categories=categories, cuisines=["international"])


# ── Dataset builder ───────────────────────────────────────────────────────────

def load_foodcom_dataset(
    zip_path: str,
    min_ratings: int = 20,
    max_recipes: int | None = None,
) -> tuple[object, RecipeCatalogue]:
    """Build a :class:`~reciperl.data.RecipeRLDataset` from real Food.com ratings.

    Also returns the :class:`RecipeCatalogue` so ``train.py`` can serialise it
    into the checkpoint for inference-time use without reloading the CSVs.

    Filtering (§4.1 of Liu et al., 2024):
    - Keep only reviews with rating ≥ 4.
    - Keep only users with ≥ ``min_ratings`` such reviews.
    - Recipes with no retained reviews are kept in the catalogue (the NCF
      simulator is still queried for them via the rating matrix).
    """
    from .data import RecipeRLDataset  # lazy import to avoid circular dependency

    recipes_csv, reviews_csv = _ensure_extracted(zip_path)
    recipes, categories = load_foodcom_recipes(zip_path, max_recipes=max_recipes)

    # Build original RecipeId → new contiguous id mapping.
    # Must apply the same exclusion filter as load_foodcom_recipes so that
    # the mapping aligns with the recipe list (excluded recipes have no new id).
    id_df = pd.read_csv(
        recipes_csv, usecols=["RecipeId", "Name", "RecipeCategory"], on_bad_lines="skip"
    )
    id_df = id_df.dropna(subset=["Name", "RecipeCategory"])
    if max_recipes is not None:
        id_df = id_df.head(max_recipes)
    id_df = id_df[
        id_df.apply(
            lambda r: _map_category(str(r["RecipeCategory"]), str(r["Name"])) != "exclude",
            axis=1,
        )
    ].reset_index(drop=True)
    orig_to_new: dict[int, int] = {
        int(rid): new_id for new_id, rid in enumerate(id_df["RecipeId"])
    }

    # Load and filter reviews.
    reviews_df = pd.read_csv(
        reviews_csv, usecols=["RecipeId", "AuthorId", "Rating"], on_bad_lines="skip"
    )
    reviews_df = reviews_df.dropna()
    reviews_df["Rating"] = pd.to_numeric(reviews_df["Rating"], errors="coerce")
    reviews_df = reviews_df.dropna(subset=["Rating"])
    reviews_df = reviews_df[reviews_df["Rating"] >= 4.0]

    # Map original recipe ids to new contiguous ids; drop unknowns.
    reviews_df["item_id"] = reviews_df["RecipeId"].map(orig_to_new)
    reviews_df = reviews_df.dropna(subset=["item_id"])
    reviews_df["item_id"] = reviews_df["item_id"].astype(int)

    # Keep users with ≥ min_ratings qualifying reviews.
    user_counts = reviews_df["AuthorId"].value_counts()
    valid_users = user_counts[user_counts >= min_ratings].index
    reviews_df = reviews_df[reviews_df["AuthorId"].isin(valid_users)]

    # Re-map AuthorId → contiguous user index.
    unique_users = sorted(reviews_df["AuthorId"].unique())
    user_to_idx: dict[object, int] = {u: i for i, u in enumerate(unique_users)}
    reviews_df["user_id"] = reviews_df["AuthorId"].map(user_to_idx)

    ratings = reviews_df[["user_id", "item_id", "Rating"]].to_numpy(dtype=np.float64)

    n_users = len(unique_users)
    n_items = len(recipes)
    catalogue = build_catalogue(recipes, categories)
    item_category = catalogue.category_ids
    n_categories = len(categories)

    # Train / test split by user (80 / 20, §4.1).
    rng = np.random.default_rng(42)
    all_users = np.arange(n_users)
    rng.shuffle(all_users)
    n_test = max(1, int(0.2 * n_users))
    test_users = set(all_users[:n_test].tolist())
    test_mask = np.array([int(r[0]) in test_users for r in ratings])
    train_mask = ~test_mask

    dataset = RecipeRLDataset(
        n_users=n_users,
        n_items=n_items,
        n_categories=n_categories,
        item_category=item_category,
        ratings=ratings,
        train_mask=train_mask,
        test_mask=test_mask,
        item_names=[r.name for r in recipes],
    )
    return dataset, catalogue
