"""RL Planner — full-day interactive meal planning (paper §3.4).

Flow:
1. Policy proposes all meals for the day at once (slots defined in user profile).
2. User can click **Changer** on any meal to swap it for a new proposal.
3. **Valider la journée** accepts all current proposals, runs ingredient
   resolution (LLM + optimiser), and saves the interaction history to the
   user's profile so the policy personalises over time.
"""

from __future__ import annotations

import os

import numpy as np
import streamlit as st

from kojin_common import (
    apply_theme,
    compute_targets,
    reference_llm_provider,
)
try:
    from reciperl.compose import (
        MealSlot,
        change_recipe,
        start_session,
        validate_day,
        load_policy,
    )
    _TORCH_OK = True
except ImportError as _torch_err:
    _TORCH_OK = False
from reciperl.db import (
    create_user,
    get_meal_slots,
    get_rl_state,
    get_user,
    init_db,
    is_valid_username,
    save_rl_state,
    user_exists,
    migrate_from_json,
)

apply_theme()

_CHECKPOINT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "reciperl.pt")
)


@st.cache_resource(show_spinner="Chargement de la politique RL…")
def _load_policy():
    return load_policy(_CHECKPOINT)


st.title("RL Planner")
st.markdown(
    '<p class="subtitle" style="font-style:italic;color:#AD9E7B">'
    "Planification journalière interactive — boucle de feedback §3.4"
    "</p>",
    unsafe_allow_html=True,
)

if not _TORCH_OK:
    st.error(
        f"Module `torch` introuvable.\n\n"
        "Installez PyTorch :\n"
        "```bash\npip install torch --index-url https://download.pytorch.org/whl/cpu\n```"
    )
    st.stop()

if not os.path.exists(_CHECKPOINT):
    st.error(
        f"Checkpoint introuvable : `{_CHECKPOINT}`.\n\n"
        "Lancez l'entraînement :\n"
        "```bash\npython -m reciperl.train --foodcom --max-recipes 50000 --steps 50\n```"
    )
    st.stop()

init_db()
migrate_from_json()
loaded = _load_policy()

# ── Login ─────────────────────────────────────────────────────────────────────

username: str | None = st.session_state.get("rl_username")

if username is None:
    with st.sidebar.form("rl_login_form", clear_on_submit=True):
        st.markdown("### Connexion")
        raw = st.text_input(
            "Pseudo",
            placeholder="ex : alice",
            help="Lettres, chiffres, _ et - uniquement (32 car. max).",
        )
        submitted = st.form_submit_button("Se connecter", use_container_width=True)

    if submitted:
        uname = raw.strip().lower()
        if not is_valid_username(uname):
            st.sidebar.error("Pseudo invalide — lettres, chiffres, _ et - uniquement.")
        else:
            create_user(uname)
            st.session_state["rl_username"] = uname

            rl = get_rl_state(uname)
            if rl:
                if "history" in rl:
                    st.session_state["rl_prev_history"] = rl["history"]
                if "history_ratings" in rl:
                    st.session_state["rl_prev_history_ratings"] = rl["history_ratings"]
                if "user_embedding" in rl:
                    st.session_state["rl_user_embedding"] = rl["user_embedding"]

            user_row = get_user(uname)
            st.session_state["rl_days_completed"] = user_row["days_completed"] if user_row else 0
            st.session_state.pop("rl_session", None)
            st.rerun()

    st.sidebar.info("Connectez-vous pour que vos préférences soient conservées d'une journée à l'autre.")
    st.info("Connectez-vous dans la barre latérale pour accéder au planificateur.")
    st.stop()

# ── Load user data from DB ────────────────────────────────────────────────────

user_row  = get_user(username) or {}
db_slots  = get_meal_slots(username)

n_days: int          = user_row.get("days_completed", st.session_state.get("rl_days_completed", 0))
is_new: bool         = not user_exists(username) or n_days == 0
prev_history         = st.session_state.get("rl_prev_history")
prev_history_ratings = st.session_state.get("rl_prev_history_ratings")

# ── Sidebar — user info ───────────────────────────────────────────────────────

st.sidebar.markdown("### Profil")
st.sidebar.markdown(
    f"**{username}**"
    + (f" · {n_days} jour{'s' if n_days > 1 else ''}" if n_days > 0 else " · première journée")
)
if prev_history is not None and n_days > 0:
    st.sidebar.caption(f"Historique : {len(prev_history)} interactions")

if user_row:
    gender         = user_row.get("gender", "Homme")
    age            = int(user_row.get("age", 25))
    weight         = float(user_row.get("weight", 75.0))
    height         = float(user_row.get("height", 175.0))
    daily_activity = user_row.get("daily_activity", "Sédentaire (bureau, peu de marche)")
    sport          = user_row.get("sport", "Aucun")
    goal           = user_row.get("goal", "balanced")
    regime         = user_row.get("regime", "")
else:
    gender, age, weight, height = "Homme", 25, 75.0, 175.0
    daily_activity = "Sédentaire (bureau, peu de marche)"
    sport, goal, regime = "Aucun", "balanced", ""
    st.sidebar.warning(
        "Profil non configuré. Rendez-vous dans **Profil** pour renseigner vos informations."
    )

col_reset, col_logout = st.sidebar.columns(2)
with col_reset:
    if st.button("Réinitialiser", key="rl_reset", help="Efface l'historique"):
        save_rl_state(username, np.array([], dtype=np.int64), np.array([], dtype=np.float32), 0)
        for k in ["rl_prev_history", "rl_prev_history_ratings", "rl_user_embedding", "rl_session"]:
            st.session_state.pop(k, None)
        st.session_state["rl_days_completed"] = 0
        st.rerun()
with col_logout:
    if st.button("Déconnexion", key="rl_logout"):
        for k in ["rl_username", "rl_prev_history", "rl_prev_history_ratings",
                  "rl_days_completed", "rl_session"]:
            st.session_state.pop(k, None)
        st.rerun()

# ── Build MealSlots from DB ───────────────────────────────────────────────────

meal_slots = [
    MealSlot(s["label"], s["category"], s["fraction"])
    for s in db_slots
]

# ── LLM & nutrition targets ───────────────────────────────────────────────────

_llm_ready = reference_llm_provider() != "bedrock" or bool(os.environ.get("BEDROCK_MODEL_ID"))

energy, proteins, fat, carbs, _ = compute_targets(
    gender, age, weight, height, daily_activity, sport, goal
)
daily_targets = [float(energy), float(proteins), float(fat), float(carbs)]

# ── Start new day ─────────────────────────────────────────────────────────────

if st.button("Nouvelle journée", type="primary", use_container_width=True):
    with st.spinner("Génération du plan journalier…"):
        st.session_state["rl_session"] = start_session(
            loaded,
            user_id=None,
            meal_slots=meal_slots,
            daily_targets=daily_targets,
            rng=np.random.default_rng(),
            prev_history=prev_history,
            prev_history_ratings=prev_history_ratings,
            user_embedding=st.session_state.get("rl_user_embedding"),
        )
    st.rerun()

session = st.session_state.get("rl_session")

if session is None:
    st.info(
        f"Bonjour **{username}** ! "
        + ("Votre historique de préférences est chargé. " if n_days > 0 else "Première journée — la politique part des recettes populaires. ")
        + "Cliquez sur **Nouvelle journée** pour commencer."
    )
    st.stop()

# ── Targets banner ────────────────────────────────────────────────────────────

st.markdown(
    f'<div class="targets-box">'
    f"Objectifs — <strong>{energy}</strong> kcal &nbsp;·&nbsp; "
    f"<strong>{proteins}</strong>g prot &nbsp;·&nbsp; "
    f"<strong>{fat}</strong>g lip &nbsp;·&nbsp; "
    f"<strong>{carbs}</strong>g gluc"
    f"</div>",
    unsafe_allow_html=True,
)
st.markdown("")

if not _llm_ready:
    st.info(
        "LLM non configuré — résolution d'ingrédients désactivée. "
        "Placez votre clé dans `.claude_key`."
    )

# ── Day complete — show results ───────────────────────────────────────────────

if session.is_done:
    new_days = n_days + 1
    updated_emb = session.user_vec.squeeze(0).numpy()
    save_rl_state(
        username,
        session.history,
        session.history_ratings,
        new_days,
        user_embedding=updated_emb,
    )
    st.session_state["rl_prev_history"]         = session.history.copy()
    st.session_state["rl_prev_history_ratings"] = session.history_ratings.copy()
    st.session_state["rl_user_embedding"]        = updated_emb
    st.session_state["rl_days_completed"]        = new_days

    st.success(f"Journée {new_days} complète ! Préférences sauvegardées pour **{username}**.")

    for meal in session.completed_meals:
        recipe = meal.recipe
        with st.expander(f"**{meal.slot.label}** — {recipe.name}", expanded=True):
            if recipe.tags:
                st.markdown(f"**Tags :** {', '.join(recipe.tags)}")
            macros = recipe.macros_per_serving
            if any(macros.get(k, 0) for k in ("kcal", "prot", "fat", "carb")):
                st.markdown(
                    f"**Macros :** {macros.get('kcal', 0):.0f} kcal &nbsp;·&nbsp; "
                    f"{macros.get('prot', 0):.1f}g prot &nbsp;·&nbsp; "
                    f"{macros.get('fat', 0):.1f}g lip &nbsp;·&nbsp; "
                    f"{macros.get('carb', 0):.1f}g gluc",
                    unsafe_allow_html=True,
                )
            st.markdown("**Ingrédients :**")
            for ing in recipe.ingredients:
                st.markdown(f"- {ing.name}")

            if meal.solution is not None and len(meal.solution) > 0:
                st.markdown("**Composition Kōjin :**")
                st.dataframe(meal.solution.to_pandas(), use_container_width=True, hide_index=True)
            elif meal.products_df.height == 0 and _llm_ready:
                st.caption("Aucun produit Kōjin trouvé pour cette recette.")

    st.info(
        f"L'historique de {len(session.history)} interactions sera utilisé "
        "pour personnaliser vos prochaines recommandations."
    )
    st.stop()

# ── Full-day meal grid ────────────────────────────────────────────────────────

st.markdown("### Votre journée")

col_left, col_right = st.columns(2)

for i, (slot, recipe) in enumerate(zip(session.meal_slots, session.proposals)):
    col = col_left if i % 2 == 0 else col_right
    with col:
        with st.container(border=True):
            st.markdown(f"##### {slot.label}")
            if recipe is None:
                st.warning(f"Catalogue épuisé pour `{slot.category}`.")
                continue

            st.markdown(f"**{recipe.name}**")
            st.caption(f"`{recipe.category}`" + (f" · {', '.join(recipe.tags)}" if recipe.tags else ""))

            macros = recipe.macros_per_serving
            if any(macros.get(k, 0) for k in ("kcal", "prot", "fat", "carb")):
                st.markdown(
                    f"<small>{macros.get('kcal', 0):.0f} kcal &nbsp;·&nbsp; "
                    f"{macros.get('prot', 0):.1f}g prot &nbsp;·&nbsp; "
                    f"{macros.get('fat', 0):.1f}g lip &nbsp;·&nbsp; "
                    f"{macros.get('carb', 0):.1f}g gluc</small>",
                    unsafe_allow_html=True,
                )

            with st.expander("Ingrédients", expanded=False):
                for ing in recipe.ingredients:
                    st.markdown(f"- {ing.name}")

            n_rej = len(session.rejected_per_slot[i])
            rej_label = f"Changer{f' ({n_rej} rejeté·s)' if n_rej else ''}"
            if st.button(rej_label, key=f"rl_change_{i}", type="secondary", use_container_width=True):
                change_recipe(session, loaded, i)
                st.session_state["rl_session"] = session
                st.rerun()

# ── Validate button ───────────────────────────────────────────────────────────

st.markdown("")
n_valid = sum(1 for r in session.proposals if r is not None)
if n_valid < len(session.meal_slots):
    st.warning(f"{len(session.meal_slots) - n_valid} repas sans proposition disponible.")

if st.button("Valider la journée ✓", type="primary", use_container_width=True, disabled=n_valid == 0):
    resolver = None
    if _llm_ready:
        try:
            from reciperl.recipe_prompts import IngredientResolver
            resolver = IngredientResolver()
        except Exception:
            resolver = None
    with st.spinner("Résolution des ingrédients et optimisation…"):
        validate_day(session, loaded, resolver=resolver, regime=regime)
    st.session_state["rl_session"] = session
    st.rerun()
