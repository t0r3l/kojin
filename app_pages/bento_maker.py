"""Bento Maker — default landing page.

Composes nutritionally-optimised bentos from the Open Food Facts catalogue
based on the user profile, activity, goal and dietary preferences.
"""

from __future__ import annotations

import os
import traceback

import polars as pl
import streamlit as st

from kojin_common import (
    BENTO_NAMES,
    CSV_PATH,
    DAILY_ACTIVITY,
    GOALS,
    REGIME_LABELS,
    SPORT_FREQUENCY,
    apply_regime,
    apply_theme,
    compute_targets,
    ensure_csv_from_s3,
    exclude_animal_protein,
    load_products,
    optimize_bento,
    run_data_prep,
)

apply_theme()
ensure_csv_from_s3()

st.title("Bento Maker — 弁当")
st.markdown(
    '<p class="subtitle">donnons à votre corps les repas qu&#39;il mérite</p>',
    unsafe_allow_html=True,
)

if not os.path.exists(CSV_PATH):
    st.warning("Le fichier de données n'a pas encore été préparé.")
    if st.button("Lancer la préparation des données"):
        products = run_data_prep()
        st.success(f"Données prêtes — {len(products)} produits.")
        st.rerun()
    st.stop()

products = load_products(CSV_PATH)

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.markdown("### Profil")

gender = st.sidebar.radio("Genre", ["Homme", "Femme"], horizontal=True)
col_a, col_w = st.sidebar.columns(2)
age = col_a.number_input("Âge", min_value=14, max_value=100, value=25)
weight = col_w.number_input("Poids (kg)", min_value=30.0, max_value=250.0, value=75.0, step=0.5)
height = st.sidebar.number_input("Taille (cm)", min_value=120.0, max_value=230.0, value=175.0, step=0.5)

st.sidebar.markdown("### Activité")
daily_activity = st.sidebar.selectbox("Quotidienne", list(DAILY_ACTIVITY.keys()))
sport = st.sidebar.selectbox("Sport", list(SPORT_FREQUENCY.keys()))

st.sidebar.markdown("### Objectif")
goal_label = st.sidebar.selectbox("But", list(GOALS.keys()))
goal = GOALS[goal_label]

energy, proteins, fat, carbs, portion_legumes = compute_targets(
    gender, age, weight, height, daily_activity, sport, goal
)

st.sidebar.markdown("### Régime")
regime = st.sidebar.selectbox(
    "Type", list(REGIME_LABELS.keys()), format_func=lambda k: REGIME_LABELS[k]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 弁当 Bentos")

num_bentos = st.sidebar.slider("Nombre de bentos / jour", 1, 5, 3)

FRAC_STEP = 0.05
FRAC_MIN = 0.05

if "num_bentos_prev" not in st.session_state or st.session_state.num_bentos_prev != num_bentos:
    equal = round(1.0 / num_bentos / FRAC_STEP) * FRAC_STEP
    for i in range(num_bentos):
        st.session_state[f"frac_{i}"] = equal
    leftover = round(1.0 - equal * num_bentos, 2)
    if abs(leftover) >= FRAC_STEP:
        st.session_state["frac_0"] = round(st.session_state["frac_0"] + leftover, 2)
    st.session_state.num_bentos_prev = num_bentos


def _on_fraction_change(changed_idx: int):
    n = st.session_state.num_bentos_prev
    new_val = st.session_state[f"frac_{changed_idx}"]
    others = [i for i in range(n) if i != changed_idx]

    old_others_sum = sum(st.session_state.get(f"_prev_frac_{i}", 1.0 / n) for i in others)
    remaining = round(1.0 - new_val, 2)

    for i in others:
        if old_others_sum > 0:
            proportion = st.session_state.get(f"_prev_frac_{i}", 1.0 / n) / old_others_sum
        else:
            proportion = 1.0 / len(others)
        st.session_state[f"frac_{i}"] = max(
            FRAC_MIN, round(remaining * proportion / FRAC_STEP) * FRAC_STEP
        )

    actual_total = new_val + sum(st.session_state[f"frac_{i}"] for i in others)
    drift = round(1.0 - actual_total, 2)
    if others and abs(drift) > 0.001:
        st.session_state[f"frac_{others[-1]}"] = max(
            FRAC_MIN, round(st.session_state[f"frac_{others[-1]}"] + drift, 2)
        )

    for i in range(n):
        st.session_state[f"_prev_frac_{i}"] = st.session_state[f"frac_{i}"]


for i in range(num_bentos):
    if f"_prev_frac_{i}" not in st.session_state:
        st.session_state[f"_prev_frac_{i}"] = st.session_state.get(
            f"frac_{i}", round(1.0 / num_bentos, 2)
        )

for i in range(num_bentos):
    st.sidebar.slider(
        f"Taille — {BENTO_NAMES[i]}",
        min_value=FRAC_MIN,
        max_value=1.0 - FRAC_MIN * (num_bentos - 1),
        step=FRAC_STEP,
        key=f"frac_{i}",
        on_change=_on_fraction_change,
        args=(i,),
    )

fractions = [st.session_state[f"frac_{i}"] for i in range(num_bentos)]
total_fraction = sum(fractions)
cls = "fraction-ok" if abs(total_fraction - 1.0) < 0.02 else "fraction-over"
st.sidebar.markdown(
    f'<p class="fraction-remaining {cls}">Total : {total_fraction:.0%}</p>',
    unsafe_allow_html=True,
)

protein_bento = st.sidebar.selectbox(
    "Bento avec protéine animale",
    range(num_bentos),
    format_func=lambda i: BENTO_NAMES[i],
    index=0,
)


# ── Main ─────────────────────────────────────────────────────────────────────

col_count, col_targets = st.columns([1, 3])

with col_count:
    st.markdown(f'<div class="product-count">{len(products):,}</div>', unsafe_allow_html=True)
    st.markdown('<div class="product-count-label">produits</div>', unsafe_allow_html=True)

with col_targets:
    st.markdown(
        f'<div class="targets-box">'
        f'Objectifs journaliers &nbsp;—&nbsp; '
        f'<strong>{energy}</strong> kcal &nbsp;&middot;&nbsp; '
        f'<strong>{proteins}</strong>g prot &nbsp;&middot;&nbsp; '
        f'<strong>{fat}</strong>g lip &nbsp;&middot;&nbsp; '
        f'<strong>{carbs}</strong>g gluc &nbsp;&middot;&nbsp; '
        f'<strong>{portion_legumes}</strong>g légumes &nbsp;&middot;&nbsp; '
        f'<strong>{num_bentos}</strong> bentos'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("")

if st.button("Composer les bentos", type="primary", use_container_width=True):
    protein_per_bento = round(proteins / num_bentos)
    base_products = apply_regime(products, regime)

    if len(base_products) == 0:
        st.error("Aucun produit ne correspond au régime sélectionné.")
        st.session_state.pop("bentos", None)
    else:
        used_codes: set[str] = set()
        bentos_state: list[dict] = []

        for idx in range(num_bentos):
            frac = fractions[idx]
            name = BENTO_NAMES[idx]

            is_protein_bento = idx == protein_bento
            if is_protein_bento:
                bento_products = base_products
            else:
                bento_products = exclude_animal_protein(base_products)

            if used_codes:
                bento_products = bento_products.filter(
                    ~pl.col("code").is_in(list(used_codes))
                )

            bento_targets = [energy, protein_per_bento, fat, carbs]

            error_msg: str | None = None
            try:
                bento = optimize_bento(
                    bento_products,
                    bento_targets,
                    frac,
                    portion_legumes,
                    allow_one_animal=is_protein_bento,
                )
            except Exception as exc:
                error_msg = f"Erreur bento {name} : {exc}\n{traceback.format_exc()}"
                bento = None

            if bento is not None and len(bento) > 0:
                selected_names = bento["Aliment"].to_list()
                matched = bento_products.filter(pl.col("product_name").is_in(selected_names))
                used_codes.update(matched["code"].to_list())

            bentos_state.append({
                "name": name,
                "fraction": frac,
                "is_protein": is_protein_bento,
                "protein_per_bento": protein_per_bento,
                "rows": bento.to_dicts() if bento is not None and len(bento) > 0 else [],
                "error": error_msg,
            })

        st.session_state.bentos = bentos_state


if "bentos" in st.session_state:
    for bento in st.session_state.bentos:
        prot_label = f" — {bento['protein_per_bento']}g prot"
        animal_label = " ◆ animal" if bento["is_protein"] else ""
        st.markdown(
            f'<div class="bento-header">弁当 {bento["name"]} '
            f'<span style="font-size:0.8rem;color:#888">'
            f'{bento["fraction"]:.0%}{prot_label}{animal_label}</span></div>',
            unsafe_allow_html=True,
        )

        if bento["error"]:
            st.error(bento["error"].splitlines()[0])
            with st.expander("Détails"):
                st.code(bento["error"])
            continue

        if not bento["rows"]:
            st.info("Aucun aliment trouvé pour ce bento.")
            continue

        st.dataframe(bento["rows"], use_container_width=True, hide_index=True)
