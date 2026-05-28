"""Day composer — batch and interactive modes.

**Batch mode** (``compose_day``): non-interactive, picks all recipes at once.
Kept for backwards compatibility and headless use.

**Interactive mode** (full-day proposal + per-meal feedback):

1. ``start_session`` generates one proposal for *every* meal slot at once.
2. The user sees the full day; they can click "Changer" on any meal →
   ``change_recipe`` masks that recipe for its slot and immediately picks a
   new one; the other slots are unchanged.
3. "Valider la journée" → ``validate_day`` accepts all current proposals,
   updates the FusedState history window in slot order, and runs ingredient
   resolution for each meal.

API::

    session = start_session(loaded, user_id=None, meal_slots=slots,
                             daily_targets=[2000, 150, 70, 250])
    # session.proposals is a list[Recipe | None], one per slot
    change_recipe(session, loaded, slot_idx=2)   # user dislikes slot 2
    validate_day(session, loaded, resolver=resolver, regime=regime)
    # session.is_done is True; session.completed_meals holds the full day
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from kojin_common import optimize_bento

from .config import RecipeRLConfig
from .policy import ActorCritic
from .recipes import Recipe, RecipeCatalogue, load_recipes
from .state import FusedState


# ── Checkpoint loader ──────────────────────────────────────────────────────


@dataclass
class LoadedPolicy:
    cfg: RecipeRLConfig
    state_module: FusedState
    policy: ActorCritic
    rating_matrix: np.ndarray
    item_names: list[str]
    catalogue: RecipeCatalogue | None = None  # None → load_recipes() on demand

    @property
    def mean_user_vec(self) -> torch.Tensor:
        """Centroid of all learned user embeddings — used for anonymous users."""
        return self.state_module.user_emb.weight.mean(dim=0, keepdim=True).detach()

    @property
    def mean_ratings(self) -> np.ndarray:
        """Per-item mean rating across all training users — cold-start for new users."""
        return self.rating_matrix.mean(axis=0)


def load_policy(path: str, device: str | torch.device = "cpu") -> LoadedPolicy:
    """Restore a checkpoint produced by ``reciperl.train``."""
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = blob["config"]
    cfg = RecipeRLConfig(**cfg_dict)

    rating_matrix = np.asarray(blob["rating_matrix"])
    n_users, n_items = rating_matrix.shape
    if "recipe_catalogue" in blob:
        from .foodcom_data import catalogue_from_blob, recategorize_catalogue
        cat = catalogue_from_blob(blob["recipe_catalogue"])
        # Re-apply category rules in case the checkpoint was built with an older
        # rule set (e.g. "Breads" → "plat" instead of "petit_dej").
        csv_path = os.path.join(os.path.dirname(os.path.abspath(path)), "recipes.csv")
        recategorize_catalogue(cat, csv_path if os.path.exists(csv_path) else None)
        n_categories = len(cat.categories)
    elif cfg.use_recipes:
        cat = load_recipes()
        n_categories = len(cat.categories)
    else:
        cat = None
        n_categories = max(cfg_dict.get("num_categories", 8), 1)

    state_module = FusedState(n_users, n_items, n_categories, cfg).to(device)
    state_module.load_state_dict(blob["state_module"])
    state_module.eval()

    policy = ActorCritic(state_module.out_dim, n_items).to(device)
    policy.load_state_dict(blob["policy"])
    policy.eval()

    return LoadedPolicy(
        cfg=cfg,
        state_module=state_module,
        policy=policy,
        rating_matrix=rating_matrix,
        item_names=list(blob["item_names"]),
        catalogue=cat,
    )


# ── Shared data structures ─────────────────────────────────────────────────


@dataclass
class MealSlot:
    label: str            # human-readable, e.g. "Petit-déjeuner"
    category: str         # must match Recipe.category
    fraction: float       # share of daily kcal (carb, fat) — protein is absolute
    portion_legumes: float = 0.0


@dataclass
class ComposedMeal:
    slot: MealSlot
    recipe: Recipe
    ingredient_sql: list[str]      # SQL queries used for each ingredient slot
    products_df: pl.DataFrame      # candidate products fed into the solver
    solution: pl.DataFrame | None  # output of ``optimize_bento``


@dataclass
class ComposedDay:
    user_id: int
    targets: list[float]           # [kcal, prot, fat, carb] daily
    meals: list[ComposedMeal] = field(default_factory=list)


# ── Internal primitives ────────────────────────────────────────────────────


def _get_catalogue(loaded: LoadedPolicy) -> RecipeCatalogue:
    return loaded.catalogue if loaded.catalogue is not None else load_recipes()


def _category_mask(
    catalogue: RecipeCatalogue, category: str, exclude_ids: Iterable[int]
) -> torch.Tensor:
    excluded = set(int(i) for i in exclude_ids)
    flags = [
        (r.category == category) and (r.id not in excluded)
        for r in catalogue.recipes
    ]
    return torch.tensor(flags, dtype=torch.bool)


def _build_initial_history(
    user_id: int | None,
    rating_matrix: np.ndarray,
    cfg: RecipeRLConfig,
    rng: np.random.Generator,
    liked_ids: list[int] | None = None,
) -> np.ndarray:
    """Build cold-start history.

    ``user_id=None`` (anonymous user) uses the mean rating vector so the policy
    starts from globally popular items rather than a specific user's preferences.
    History then personalises purely through the ``s_ACH`` cross-attention as
    the user interacts.
    """
    ratings = rating_matrix[user_id] if user_id is not None else rating_matrix.mean(axis=0)
    k = min(cfg.window_k, ratings.size)
    top_idx = np.argpartition(ratings, -k)[-k:]
    top_idx = top_idx[np.argsort(-ratings[top_idx])]
    history = top_idx.astype(np.int64)
    if liked_ids:
        valid = [i for i in liked_ids if 0 <= i < ratings.size]
        if valid:
            seed = np.array(valid, dtype=np.int64)
            history = np.concatenate([seed, history])[-cfg.window_k:]
    return history


def _observe_state(
    state_module: FusedState,
    user_id: int | None,
    history: np.ndarray,
    history_ratings: np.ndarray,
    rec_counts_window: np.ndarray,
    last_item: int,
    last_cat: int,
    device: torch.device,
    user_vec: torch.Tensor | None = None,
) -> torch.Tensor:
    """Forward pass through FusedState.

    ``user_id=None`` + ``user_vec`` → anonymous user using the mean embedding.
    """
    u = torch.tensor([0 if user_id is None else user_id], dtype=torch.long, device=device)
    h = torch.tensor(history, dtype=torch.long, device=device).unsqueeze(0)
    hr = torch.tensor(history_ratings, dtype=torch.float32, device=device).unsqueeze(0)
    rc = torch.tensor(rec_counts_window, dtype=torch.float32, device=device).unsqueeze(0)
    li = torch.tensor([last_item], dtype=torch.long, device=device)
    lc = torch.tensor([last_cat], dtype=torch.long, device=device)
    vec = user_vec.to(device) if user_vec is not None else None
    with torch.no_grad():
        s = state_module(u, h, hr, rc, li, lc, user_vec=vec)
    return s


def _resolve_and_optimize(
    recipe: Recipe,
    slot: MealSlot,
    resolver,
    regime: str | None,
    daily_targets: list[float],
) -> tuple[list[str], pl.DataFrame, pl.DataFrame | None]:
    """Resolve abstract ingredients → catalogue products → gram quantities.

    Returns ``(sql_log, products_df, solution)``.
    """
    if resolver is None:
        return [], pl.DataFrame(), None

    resolved = resolver.resolve_recipe(recipe, regime=regime)
    sql_log = [r.sql for r in resolved]

    frames_with_slots: list[pl.DataFrame] = []
    slot_proportions: dict[int, float] = {}
    for slot_idx, r in enumerate(resolved):
        if r.products.height > 0:
            frames_with_slots.append(
                r.products.with_columns(pl.lit(slot_idx).alias("_slot_id"))
            )
            slot_proportions[slot_idx] = r.ingredient.proportion

    if not frames_with_slots:
        return sql_log, pl.DataFrame(), None

    products_df = (
        pl.concat(frames_with_slots, how="diagonal_relaxed")
        .unique(subset=["product_name", "_slot_id"])
    )
    raw_slot_ids = products_df["_slot_id"].to_list()
    active_slots = sorted(slot_proportions.keys())
    slot_remap = {old: new for new, old in enumerate(active_slots)}
    remapped_slot_ids = [slot_remap[s] for s in raw_slot_ids]
    raw_props = np.array([slot_proportions[s] for s in active_slots], dtype=float)
    products_df = products_df.drop("_slot_id")

    if raw_props.sum() > 1e-9:
        final_slot_ids: list[int] | None = remapped_slot_ids
        final_props: list[float] | None = (raw_props / raw_props.sum()).tolist()
    else:
        final_slot_ids = None
        final_props = None

    solution = optimize_bento(
        products_df,
        daily_targets,
        meal_fraction=slot.fraction,
        portion_legumes=slot.portion_legumes,
        ingredient_slot_ids=final_slot_ids,
        ingredient_proportions=final_props,
    )
    return sql_log, products_df, solution


# ── Interactive session (full-day proposal + per-meal feedback) ────────────


@dataclass
class InteractiveSession:
    """Mutable state for the full-day interactive feedback loop.

    Stored in ``st.session_state`` between Streamlit reruns. Does not hold
    any torch modules (those stay in ``@st.cache_resource``), except for
    ``user_vec`` which is the per-user embedding tensor that evolves over time.

    All meal slots are proposed at once. The user can swap individual meals
    (``change_recipe``) and finalise the whole day with ``validate_day``.

    ``user_vec`` starts as the centroid of all learned user embeddings
    (cold-start) and is fine-tuned in-place after each validated day using
    the session's accepted and rejected recipes as training signal. It
    personalises ``s_UI`` and ``s_UC`` across sessions, complementing the
    ``s_ACH`` cross-attention that already adapts through history.

    ``history`` / ``history_ratings`` and ``user_vec`` are all persisted
    in the user's profile and reloaded on next login.
    """
    user_id: int | None
    meal_slots: list[MealSlot]
    daily_targets: list[float]

    user_vec: torch.Tensor              # (1, d) — per-user embedding, updated by adapt

    history: np.ndarray                 # sliding window of accepted recipe IDs
    history_ratings: np.ndarray
    rec_counts_global: np.ndarray       # times each item was shown

    chosen_global: list[int]            # accepted IDs (post-validate)
    rejected_per_slot: list[set[int]]   # rejected IDs per slot index
    proposals: list[Recipe | None]      # current proposal per slot

    completed_meals: list[ComposedMeal] # filled by validate_day

    @property
    def is_done(self) -> bool:
        return len(self.completed_meals) == len(self.meal_slots)


def _adapt_user_embedding(
    user_vec: torch.Tensor,
    accepted_ids: list[int],
    rejected_ids: list[int],
    state_module: FusedState,
    lr: float = 1e-3,
    n_steps: int = 30,
) -> torch.Tensor:
    """Fine-tune the user embedding on one session's explicit feedback.

    Only *user_vec* is updated; all other model weights remain frozen.

    Loss: MSE between NCF-style dot-product prediction and target rating.
      - Accepted recipe → target 5.0  (user explicitly chose it)
      - Rejected recipe → target 1.0  (user saw it and refused)

    With d=64 and typically 4–20 feedback pairs, 30 Adam steps at lr=1e-3
    give stable updates that stay close to the mean-embedding initialisation,
    keeping the input within the distribution seen during training.
    """
    if not accepted_ids and not rejected_ids:
        return user_vec.clone().detach()

    vec = user_vec.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([vec], lr=lr)

    pairs = [(i, 5.0) for i in accepted_ids] + [(i, 1.0) for i in rejected_ids]
    ids = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    targets = torch.tensor([p[1] for p in pairs], dtype=torch.float32)

    for _ in range(n_steps):
        optimizer.zero_grad()
        # NCF-style: pᵤ · (wᵢ qᵢ) — same formula as FusedState._ddr_u
        # item_w → (N,1), item_emb → (N,d): broadcast (N,1)*(N,d) = (N,d)
        weighted = state_module.item_w(ids) * state_module.item_emb(ids)
        preds = (vec * weighted).sum(dim=-1)
        F.mse_loss(preds, targets).backward()
        optimizer.step()

    return vec.detach()


def _select_recipe(
    session: InteractiveSession,
    loaded: LoadedPolicy,
    slot: MealSlot,
    excluded: set[int],
    device: torch.device,
) -> Recipe | None:
    """Run the policy for *slot*, excluding *excluded* IDs. Returns a recipe or None."""
    catalogue = _get_catalogue(loaded)
    cat_ids = catalogue.category_ids

    mask = _category_mask(catalogue, slot.category, excluded).to(device)
    if not mask.any():
        # Relax: only honour cross-slot dedup, ignore per-slot rejections.
        mask = _category_mask(catalogue, slot.category, set(session.chosen_global)).to(device)
        if not mask.any():
            return None

    last_item = int(session.history[-1])
    last_cat = int(cat_ids[last_item])

    state = _observe_state(
        loaded.state_module,
        session.user_id,
        session.history,
        session.history_ratings,
        session.rec_counts_global[session.history],
        last_item,
        last_cat,
        device,
        user_vec=session.user_vec.to(device),
    )
    with torch.no_grad():
        action, _, _ = loaded.policy.act(state, mask=mask.unsqueeze(0))
    recipe_id = int(action.item())
    session.rec_counts_global[recipe_id] += 1
    return catalogue.recipes[recipe_id]


def _generate_all_proposals(
    session: InteractiveSession,
    loaded: LoadedPolicy,
    device: torch.device,
) -> None:
    """Fill session.proposals for all slots (no inter-slot duplicates)."""
    proposed_so_far: set[int] = set()
    for i, slot in enumerate(session.meal_slots):
        excluded = set(session.chosen_global) | session.rejected_per_slot[i] | proposed_so_far
        recipe = _select_recipe(session, loaded, slot, excluded, device)
        session.proposals[i] = recipe
        if recipe is not None:
            proposed_so_far.add(recipe.id)


def start_session(
    loaded: LoadedPolicy,
    *,
    user_id: int | None = None,
    meal_slots: list[MealSlot],
    daily_targets: list[float],
    device: str | torch.device = "cpu",
    rng: np.random.Generator | None = None,
    prev_history: np.ndarray | None = None,
    prev_history_ratings: np.ndarray | None = None,
    user_embedding: np.ndarray | None = None,
) -> InteractiveSession:
    """Initialise a session and immediately propose recipes for *all* slots.

    ``user_embedding``: flat float array of shape ``(d,)`` saved from the
    previous session's ``session.user_vec``. When provided, the session
    starts from the user's personalised embedding instead of the global
    mean. On first visit (no stored embedding) the mean is used as cold-start.
    """
    rng = rng or np.random.default_rng()
    device = torch.device(device)
    catalogue = _get_catalogue(loaded)
    n_slots = len(meal_slots)

    if prev_history is not None and len(prev_history) > 0:
        history = prev_history.astype(np.int64)
        history_ratings = (
            prev_history_ratings.astype(np.float32)
            if prev_history_ratings is not None
            else np.zeros(len(prev_history), dtype=np.float32)
        )
    else:
        history = _build_initial_history(
            user_id, loaded.rating_matrix, loaded.cfg, rng
        )
        ratings_src = (
            loaded.rating_matrix[user_id]
            if user_id is not None
            else loaded.mean_ratings
        )
        history_ratings = ratings_src[history].astype(np.float32)

    if user_embedding is not None:
        user_vec = torch.tensor(user_embedding, dtype=torch.float32).unsqueeze(0)
    else:
        user_vec = loaded.mean_user_vec.clone()

    session = InteractiveSession(
        user_id=user_id,
        meal_slots=meal_slots,
        daily_targets=daily_targets,
        user_vec=user_vec,
        history=history,
        history_ratings=history_ratings,
        rec_counts_global=np.zeros(len(catalogue), dtype=np.float32),
        chosen_global=[],
        rejected_per_slot=[set() for _ in range(n_slots)],
        proposals=[None] * n_slots,
        completed_meals=[],
    )
    _generate_all_proposals(session, loaded, device)
    return session


def change_recipe(
    session: InteractiveSession,
    loaded: LoadedPolicy,
    slot_idx: int,
    device: str | torch.device = "cpu",
) -> None:
    """Reject the current proposal for *slot_idx* and immediately pick a new one.

    The rejected recipe is masked for this slot. All other slots' proposals
    are excluded to avoid cross-slot duplicates in the new pick.
    """
    device = torch.device(device)
    current = session.proposals[slot_idx]
    if current is not None:
        session.rejected_per_slot[slot_idx].add(current.id)

    other_ids = {
        r.id for i, r in enumerate(session.proposals)
        if r is not None and i != slot_idx
    }
    excluded = set(session.chosen_global) | session.rejected_per_slot[slot_idx] | other_ids
    session.proposals[slot_idx] = _select_recipe(
        session, loaded, session.meal_slots[slot_idx], excluded, device
    )


def validate_day(
    session: InteractiveSession,
    loaded: LoadedPolicy,
    *,
    resolver=None,
    regime: str | None = None,
    device: str | torch.device = "cpu",
) -> None:
    """Accept all current proposals, update history, adapt user embedding,
    and run ingredient resolution.

    After accepting the meals, ``session.user_vec`` is fine-tuned on the
    session's explicit feedback (accepted recipes → target 5.0, rejected
    recipes → target 1.0) so future sessions start from an increasingly
    personalised embedding.
    """
    if session.is_done:
        return

    device = torch.device(device)
    cfg = loaded.cfg
    ratings_src = (
        loaded.mean_ratings
        if session.user_id is None
        else loaded.rating_matrix[session.user_id]
    )

    valid_pairs = [
        (slot, recipe)
        for slot, recipe in zip(session.meal_slots, session.proposals)
        if recipe is not None
    ]

    for slot, recipe in valid_pairs:
        session.chosen_global.append(recipe.id)
        new_rating = float(ratings_src[recipe.id])
        if session.history.size >= cfg.window_k:
            session.history = np.concatenate([session.history[1:], [recipe.id]])
            session.history_ratings = np.concatenate([session.history_ratings[1:], [new_rating]])
        else:
            session.history = np.concatenate([session.history, [recipe.id]])
            session.history_ratings = np.concatenate([session.history_ratings, [new_rating]])

    # Resolve all meals in parallel (I/O-bound LLM calls).
    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

    resolution_results: dict[int, tuple] = {}

    def _resolve_meal(idx: int, recipe, slot, resolver, regime, targets):
        return idx, _resolve_and_optimize(recipe, slot, resolver, regime, targets)

    with ThreadPoolExecutor(max_workers=len(valid_pairs) or 1) as pool:
        futures = {
            pool.submit(_resolve_meal, i, recipe, slot, resolver, regime, session.daily_targets): i
            for i, (slot, recipe) in enumerate(valid_pairs)
        }
        for fut in _as_completed(futures):
            idx, result = fut.result()
            resolution_results[idx] = result

    for i, (slot, recipe) in enumerate(valid_pairs):
        sql_log, products_df, solution = resolution_results[i]
        session.completed_meals.append(ComposedMeal(
            slot=slot, recipe=recipe,
            ingredient_sql=sql_log, products_df=products_df, solution=solution,
        ))

    # Fine-tune user embedding on this session's feedback.
    # Accepted recipes signal positive preference; recipes seen and rejected
    # via "Changer" signal negative preference.
    all_rejected = list(
        {rid for slot_set in session.rejected_per_slot for rid in slot_set}
        - set(session.chosen_global)
    )
    session.user_vec = _adapt_user_embedding(
        session.user_vec,
        accepted_ids=session.chosen_global,
        rejected_ids=all_rejected,
        state_module=loaded.state_module,
    )


# ── Batch mode (backwards-compatible) ─────────────────────────────────────


def compose_day(
    loaded: LoadedPolicy,
    *,
    user_id: int,
    daily_targets: list[float],
    meal_slots: list[MealSlot],
    resolver,
    regime: str | None = None,
    rng: np.random.Generator | None = None,
    device: str | torch.device = "cpu",
    exclude_recipe_ids: set[int] | None = None,
    liked_recipe_ids: list[int] | None = None,
) -> ComposedDay:
    """Non-interactive batch composition (all slots picked without feedback).

    ``exclude_recipe_ids`` are masked globally; ``liked_recipe_ids`` seed the
    cold-start history. Use the interactive API (``start_session`` /
    ``accept_recipe`` / ``reject_recipe``) to match the paper's feedback loop.
    """
    rng = rng or np.random.default_rng()
    device = torch.device(device)
    catalogue = _get_catalogue(loaded)
    cat_ids = catalogue.category_ids
    global_exclude = set(exclude_recipe_ids) if exclude_recipe_ids else set()

    history = _build_initial_history(
        user_id, loaded.rating_matrix, loaded.cfg, rng, liked_ids=liked_recipe_ids
    )
    history_ratings = loaded.rating_matrix[user_id, history].astype(np.float32)
    rec_counts_global = np.zeros(len(catalogue), dtype=np.float32)
    chosen: list[int] = []
    day = ComposedDay(user_id=user_id, targets=list(daily_targets))

    for slot in meal_slots:
        mask = _category_mask(catalogue, slot.category, set(chosen) | global_exclude).to(device)
        if not mask.any():
            mask = torch.tensor(
                [r.id not in chosen for r in catalogue.recipes],
                dtype=torch.bool, device=device,
            )
            if not mask.any():
                break

        last_item = int(history[-1])
        last_cat = int(cat_ids[last_item])

        state = _observe_state(
            loaded.state_module, user_id, history, history_ratings,
            rec_counts_global[history], last_item, last_cat, device,
        )

        with torch.no_grad():
            action, _, _ = loaded.policy.act(state, mask=mask.unsqueeze(0))
        recipe_id = int(action.item())
        recipe = catalogue.recipes[recipe_id]
        chosen.append(recipe_id)
        rec_counts_global[recipe_id] += 1

        new_rating = float(loaded.rating_matrix[user_id, recipe_id])
        if history.size >= loaded.cfg.window_k:
            history = np.concatenate([history[1:], [recipe_id]])
            history_ratings = np.concatenate([history_ratings[1:], [new_rating]])
        else:
            history = np.concatenate([history, [recipe_id]])
            history_ratings = np.concatenate([history_ratings, [new_rating]])

        sql_log, products_df, solution = _resolve_and_optimize(
            recipe, slot, resolver, regime, daily_targets
        )
        day.meals.append(ComposedMeal(
            slot=slot, recipe=recipe,
            ingredient_sql=sql_log, products_df=products_df, solution=solution,
        ))

    return day
