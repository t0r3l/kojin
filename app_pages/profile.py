"""Page Profil — gestion du compte utilisateur et du plan de repas.

Onglets
-------
1. Informations    — données personnelles + objectif nutritionnel
2. Plan de repas   — nombre de repas, labels, catégories et proportions kcal
"""

from __future__ import annotations

import streamlit as st

from kojin_common import (
    DAILY_ACTIVITY,
    GOALS,
    REGIME_LABELS,
    SPORT_FREQUENCY,
    apply_theme,
    compute_targets,
)
from reciperl.db import (
    create_user,
    get_meal_slots,
    get_rl_state,
    get_user,
    init_db,
    is_valid_username,
    save_meal_slots,
    save_user,
    user_exists,
    migrate_from_json,
)

apply_theme()

_CATEGORIES = ["plat", "petit_dej", "snack"]
_CATEGORY_LABELS = {"plat": "Plat principal", "petit_dej": "Petit-déjeuner", "snack": "Snack / Goûter"}

_DEFAULT_SLOTS = [
    {"label": "Petit-déjeuner", "category": "petit_dej", "fraction": 0.25},
    {"label": "Déjeuner",       "category": "plat",      "fraction": 0.35},
    {"label": "Goûter",         "category": "snack",     "fraction": 0.10},
    {"label": "Dîner",          "category": "plat",      "fraction": 0.30},
]

st.title("Profil")
st.markdown(
    '<p class="subtitle" style="font-style:italic;color:#AD9E7B">'
    "Vos préférences personnelles et plan alimentaire journalier"
    "</p>",
    unsafe_allow_html=True,
)

init_db()
migrate_from_json()

# ── Login ─────────────────────────────────────────────────────────────────────

username: str | None = st.session_state.get("rl_username")

if username is None:
    with st.container(border=True):
        st.markdown("### Connexion")
        with st.form("profile_login_form", clear_on_submit=True):
            raw = st.text_input(
                "Pseudo",
                placeholder="ex : alice",
                help="Lettres, chiffres, _ et - uniquement (32 car. max). Créé automatiquement.",
            )
            submitted = st.form_submit_button("Se connecter", use_container_width=True)

        if submitted:
            uname = raw.strip().lower()
            if not is_valid_username(uname):
                st.error("Pseudo invalide — lettres, chiffres, _ et - uniquement.")
            else:
                is_new = not user_exists(uname)
                create_user(uname)
                st.session_state["rl_username"] = uname

                rl = get_rl_state(uname)
                if rl:
                    import numpy as np
                    if "history" in rl:
                        st.session_state["rl_prev_history"] = rl["history"]
                    if "history_ratings" in rl:
                        st.session_state["rl_prev_history_ratings"] = rl["history_ratings"]
                    if "user_embedding" in rl:
                        st.session_state["rl_user_embedding"] = rl["user_embedding"]

                user_row = get_user(uname)
                if user_row:
                    st.session_state["rl_days_completed"] = user_row["days_completed"]

                if is_new:
                    st.success(f"Profil **{uname}** créé ! Remplissez vos informations ci-dessous.")
                else:
                    st.success(f"Bienvenue **{uname}** !")
                st.rerun()

    st.info("Créez un compte ou connectez-vous pour accéder à votre profil.")
    st.stop()

# ── Logged-in header ──────────────────────────────────────────────────────────

n_days = st.session_state.get("rl_days_completed", 0)
col_title, col_logout = st.columns([4, 1])
col_title.markdown(
    f"**{username}** · {n_days} jour{'s' if n_days != 1 else ''} complété{'s' if n_days != 1 else ''}"
)
if col_logout.button("Déconnexion", key="profile_logout"):
    for k in ["rl_username", "rl_prev_history", "rl_prev_history_ratings",
              "rl_days_completed", "rl_session", "rl_user_embedding"]:
        st.session_state.pop(k, None)
    st.rerun()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_info, tab_plan = st.tabs(["Informations personnelles", "Plan de repas"])

# ── Tab 1 : informations ──────────────────────────────────────────────────────

with tab_info:
    user_row = get_user(username) or {}

    gender = st.radio(
        "Genre",
        ["Homme", "Femme"],
        index=0 if user_row.get("gender", "Homme") == "Homme" else 1,
        horizontal=True,
    )
    col_a, col_w, col_h = st.columns(3)
    age    = col_a.number_input("Âge",        min_value=14,  max_value=100,   value=int(user_row.get("age", 25)))
    weight = col_w.number_input("Poids (kg)", min_value=30.0, max_value=250.0, value=float(user_row.get("weight", 75.0)), step=0.5)
    height = col_h.number_input("Taille (cm)",min_value=120.0,max_value=230.0, value=float(user_row.get("height", 175.0)), step=0.5)

    st.markdown("#### Activité physique")
    col_act, col_sport = st.columns(2)
    daily_activity_keys = list(DAILY_ACTIVITY.keys())
    saved_activity = user_row.get("daily_activity", daily_activity_keys[0])
    daily_activity = col_act.selectbox(
        "Activité quotidienne",
        daily_activity_keys,
        index=daily_activity_keys.index(saved_activity) if saved_activity in daily_activity_keys else 0,
    )
    sport_keys = list(SPORT_FREQUENCY.keys())
    saved_sport = user_row.get("sport", sport_keys[0])
    sport = col_sport.selectbox(
        "Sport",
        sport_keys,
        index=sport_keys.index(saved_sport) if saved_sport in sport_keys else 0,
    )

    st.markdown("#### Objectif & régime")
    goal_keys   = list(GOALS.keys())
    goal_values = list(GOALS.values())
    saved_goal_val = user_row.get("goal", "balanced")
    goal_idx = goal_values.index(saved_goal_val) if saved_goal_val in goal_values else 0
    goal_label = st.selectbox("Objectif", goal_keys, index=goal_idx)
    goal = GOALS[goal_label]

    regime_keys = list(REGIME_LABELS.keys())
    saved_regime = user_row.get("regime", "")
    regime = st.selectbox(
        "Régime alimentaire",
        regime_keys,
        format_func=lambda k: REGIME_LABELS[k],
        index=regime_keys.index(saved_regime) if saved_regime in regime_keys else 0,
    )

    # Targets preview
    energy, proteins, fat, carbs, _ = compute_targets(
        gender, age, weight, height, daily_activity, sport, goal
    )
    st.markdown(
        f'<div class="targets-box">'
        f"Besoins estimés — <strong>{energy}</strong> kcal &nbsp;·&nbsp; "
        f"<strong>{proteins}</strong>g prot &nbsp;·&nbsp; "
        f"<strong>{fat}</strong>g lip &nbsp;·&nbsp; "
        f"<strong>{carbs}</strong>g gluc"
        f"</div>",
        unsafe_allow_html=True,
    )

    if st.button("Enregistrer le profil", type="primary", use_container_width=True, key="save_info"):
        save_user(
            username,
            gender=gender, age=age, weight=weight, height=height,
            daily_activity=daily_activity, sport=sport, goal=goal, regime=regime,
        )
        st.success("Profil enregistré.")

# ── Tab 2 : meal plan ─────────────────────────────────────────────────────────

with tab_plan:
    st.markdown(
        "Définissez le nombre de repas de votre journée et la proportion de vos "
        "calories totales allouée à chacun. Les proportions doivent totaliser **100 %**."
    )

    db_slots = get_meal_slots(username)

    n_meals = st.slider(
        "Nombre de repas par jour",
        min_value=1,
        max_value=5,
        value=len(db_slots),
        key="profile_n_meals",
    )

    # Build editable slot list: keep existing slots, pad/trim to n_meals
    while len(db_slots) < n_meals:
        templates = [
            {"label": "Petit-déjeuner", "category": "petit_dej", "fraction": 0.25},
            {"label": "Collation matin","category": "snack",     "fraction": 0.10},
            {"label": "Déjeuner",       "category": "plat",      "fraction": 0.35},
            {"label": "Goûter",         "category": "snack",     "fraction": 0.10},
            {"label": "Dîner",          "category": "plat",      "fraction": 0.20},
        ]
        db_slots.append(templates[len(db_slots) % len(templates)])
    db_slots = db_slots[:n_meals]

    # Renormalise fractions to sum=1 after resize
    total_frac = sum(s["fraction"] for s in db_slots)
    if total_frac > 0:
        for s in db_slots:
            s["fraction"] = s["fraction"] / total_frac

    st.markdown("---")

    slot_labels    = [s["label"]    for s in db_slots]
    slot_cats      = [s["category"] for s in db_slots]
    slot_fractions = [s["fraction"] for s in db_slots]

    col_headers = st.columns([2, 2, 3])
    col_headers[0].markdown("**Nom du repas**")
    col_headers[1].markdown("**Catégorie**")
    col_headers[2].markdown("**Proportion kcal (%)**")

    for i in range(n_meals):
        col_label, col_cat, col_frac = st.columns([2, 2, 3])
        slot_labels[i] = col_label.text_input(
            f"Repas {i+1}",
            value=slot_labels[i],
            key=f"plan_label_{i}",
            label_visibility="collapsed",
        )
        cat_idx = _CATEGORIES.index(slot_cats[i]) if slot_cats[i] in _CATEGORIES else 0
        slot_cats[i] = col_cat.selectbox(
            f"Catégorie {i+1}",
            _CATEGORIES,
            index=cat_idx,
            format_func=lambda k: _CATEGORY_LABELS[k],
            key=f"plan_cat_{i}",
            label_visibility="collapsed",
        )
        slot_fractions[i] = col_frac.slider(
            f"Fraction {i+1}",
            min_value=5,
            max_value=80,
            value=max(5, round(slot_fractions[i] * 100)),
            step=5,
            format="%d%%",
            key=f"plan_frac_{i}",
            label_visibility="collapsed",
        )

    total_pct = sum(slot_fractions)
    if total_pct != 100:
        delta = total_pct - 100
        st.warning(
            f"Total : **{total_pct}%** — "
            f"{'réduisez' if delta > 0 else 'augmentez'} de {abs(delta)}% pour atteindre 100%."
        )
    else:
        st.success("Total : 100% ✓")

    if st.button(
        "Enregistrer le plan de repas",
        type="primary",
        use_container_width=True,
        key="save_plan",
        disabled=(total_pct != 100),
    ):
        new_slots = [
            {
                "label":    slot_labels[i],
                "category": slot_cats[i],
                "fraction": slot_fractions[i] / 100.0,
            }
            for i in range(n_meals)
        ]
        save_meal_slots(username, new_slots)
        st.success("Plan de repas enregistré.")
        st.rerun()
