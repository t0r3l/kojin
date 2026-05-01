"""Bento Maker — default landing page.

Composes nutritionally-optimised bentos from the Open Food Facts catalogue
based on the user profile, activity, goal and dietary preferences.
"""

from __future__ import annotations

import io
import math
import os
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
import streamlit as st

from bento_editor import (
    clear_qty_widgets_for_bento,
    clear_streamlit_editor_keys,
    per_bento_macro_targets,
    render_bento_gate_header,
    render_bento_inline_table,
    replace_bento_row_ingredient,
    run_daily_rebalance_after_anchor,
    sync_locked_bentos_from_ingredient_locks,
)
from kojin_common import (
    BENTO_NAMES,
    CSV_PATH,
    DAILY_ACTIVITY,
    GOALS,
    NUTR_DISPLAY,
    REGIME_LABELS,
    SPORT_FREQUENCY,
    align_bento_aliments_to_catalog,
    apply_regime,
    apply_theme,
    compute_targets,
    ensure_csv_from_s3,
    exclude_animal_protein,
    format_product_display_label,
    load_products,
    optimize_bento,
    run_data_prep,
)

apply_theme()
ensure_csv_from_s3()

# Kcal/g conversion factors for the macro breakdown charts
_KCAL_PER_G = {"proteins": 4.0, "fat": 9.0, "carbohydrates": 4.0}
_MACRO_NC    = ["proteins", "fat", "carbohydrates"]
_MACRO_LABEL = {"proteins": "Prot", "fat": "Lip", "carbohydrates": "Gluc"}
# Shades of near-black for segments (light enough to keep white text readable)
_SEG_COLORS  = ["#1a1a1a", "#2f2f2f", "#464646"]
_INK         = "#1a1a1a"
_PAPER       = "#fafaf8"


def _pie_chart(
    values: list[float],
    seg_labels: list[str],
    title: str,
    total_kcal: float,
    *,
    ax: plt.Axes,
) -> None:
    """Renders a single filled pie on *ax* (black segments, white dividers, labels in segments)."""
    total = sum(values)
    effective = [max(v, 0.001 * total) for v in values]  # guard against 0-size wedges

    wedges, _ = ax.pie(
        effective,
        colors=_SEG_COLORS,
        wedgeprops=dict(edgecolor=_PAPER, linewidth=1.8),
        startangle=90,
        labels=None,
    )
    ax.set_facecolor("none")

    for w, lbl in zip(wedges, seg_labels):
        angle = math.radians((w.theta1 + w.theta2) / 2)
        rx, ry = 0.58 * math.cos(angle), 0.58 * math.sin(angle)
        ax.text(
            rx, ry, lbl,
            ha="center", va="center",
            color=_PAPER, fontsize=11, fontweight="bold",
            fontfamily="sans-serif",
        )

    ax.set_title(
        f"{title}  {total_kcal:.0f} kcal",
        color="#AD9E7B", fontsize=13, pad=8, fontweight="bold", fontfamily="sans-serif",
    )


def _render_bento_macro_charts(df_bento: pl.DataFrame, targets: dict[str, float], *, container=None) -> None:
    """Affiche deux camemberts côte à côte sous chaque bento :
    - gauche : répartition kcal cible par macronutriment
    - droite  : répartition kcal obtenus + % de la cible atteinte par macro
    """
    _st = container if container is not None else st
    # Totaux obtenus (en grammes)
    obtained_g = {
        nc: float(df_bento[NUTR_DISPLAY[nc]].sum()) if NUTR_DISPLAY[nc] in df_bento.columns else 0.0
        for nc in _MACRO_NC
    }

    # Conversion en kcal
    target_kcal  = {nc: targets.get(nc, 0.0) * _KCAL_PER_G[nc] for nc in _MACRO_NC}
    obtained_kcal = {nc: obtained_g[nc] * _KCAL_PER_G[nc] for nc in _MACRO_NC}

    total_target_kcal   = sum(target_kcal.values())
    total_obtained_kcal = sum(obtained_kcal.values())

    if total_target_kcal <= 0:
        return

    # ── Labels ──────────────────────────────────────────────────────────────
    labels_tgt = [
        f"{_MACRO_LABEL[nc]}\n{target_kcal[nc]:.0f} kcal"
        for nc in _MACRO_NC
    ]
    labels_obt = [
        f"{_MACRO_LABEL[nc]}\n{obtained_kcal[nc]/target_kcal[nc]*100:.0f}%"
        if target_kcal[nc] > 0 else _MACRO_LABEL[nc]
        for nc in _MACRO_NC
    ]

    # ── Figure ──────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 4.0))
    fig.patch.set_alpha(0.0)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.02, wspace=0.08)

    _pie_chart(
        [target_kcal[nc] for nc in _MACRO_NC],
        seg_labels=labels_tgt,
        title="Cible",
        total_kcal=total_target_kcal,
        ax=ax1,
    )
    _pie_chart(
        [obtained_kcal[nc] for nc in _MACRO_NC],
        seg_labels=labels_obt,
        title="Obtenu",
        total_kcal=total_obtained_kcal,
        ax=ax2,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                transparent=True, edgecolor="none")
    plt.close(fig)
    buf.seek(0)

    import base64
    img_b64 = base64.b64encode(buf.read()).decode()
    _st.markdown(
        f'<div style="display:flex;justify-content:center;align-items:center;margin:0">'
        f'<img src="data:image/png;base64,{img_b64}" '
        f'style="width:100%;max-width:580px;border:none">'
        f'</div>',
        unsafe_allow_html=True,
    )

st.title("Bento Planner")
st.markdown(
    '<p class="subtitle" style="font-style:italic;color:#AD9E7B">Donnons à votre corps les repas qu&#39;il mérite</p>',
    unsafe_allow_html=True,
)

if not os.path.exists(CSV_PATH):
    st.warning("Le fichier de données n'a pas encore été préparé.")
    if st.button("Lancer la préparation des données"):
        products = run_data_prep()
        st.success(f"Données prêtes — {len(products)} ingrédients.")
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
st.sidebar.markdown("### Bentos")

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

st.sidebar.markdown("### Proportion des bentos")
for i in range(num_bentos):
    st.sidebar.slider(
        f"Bento {BENTO_NAMES[i]}",
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
    st.markdown('<div class="product-count-label">ingrédients</div>', unsafe_allow_html=True)

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
        st.error("Aucun ingrédient ne correspond au régime sélectionné.")
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
        st.session_state.bento_frames = [
            pl.DataFrame(b["rows"]) if b["rows"] else None for b in bentos_state
        ]
        st.session_state.ingredient_locks = [
            [False] * len(b["rows"]) if b["rows"] else [] for b in bentos_state
        ]
        sync_locked_bentos_from_ingredient_locks(num_bentos)
        clear_streamlit_editor_keys()


if "bentos" in st.session_state:
    n_comp = len(st.session_state.bentos)
    if "bento_frames" not in st.session_state or len(st.session_state.bento_frames) != n_comp:
        st.session_state.bento_frames = [
            pl.DataFrame(st.session_state.bentos[i]["rows"])
            if st.session_state.bentos[i]["rows"]
            else None
            for i in range(n_comp)
        ]
    if "ingredient_locks" not in st.session_state or len(st.session_state.ingredient_locks) != n_comp:
        st.session_state.ingredient_locks = []
        for ii in range(n_comp):
            fr = (
                st.session_state.bento_frames[ii]
                if ii < len(st.session_state.get("bento_frames", []))
                else None
            )
            nrows = len(fr) if fr is not None else 0
            st.session_state.ingredient_locks.append([False] * nrows)
    if "locked_bentos" not in st.session_state or len(st.session_state.locked_bentos) != n_comp:
        st.session_state.locked_bentos = [False] * n_comp
    sync_locked_bentos_from_ingredient_locks(n_comp)

    # Cache catalog_by_bento : ne recalculer que si regime/protein_bento/n_comp changent
    _cat_cache_key = (regime, protein_bento, n_comp)
    if st.session_state.get("_catalog_cache_key") != _cat_cache_key:
        base_cat = apply_regime(products, regime)
        _no_animal = exclude_animal_protein(base_cat)
        catalog_by_bento = [
            base_cat if j == protein_bento else _no_animal
            for j in range(n_comp)
        ]
        st.session_state["_catalog_by_bento"] = catalog_by_bento
        st.session_state["_catalog_cache_key"] = _cat_cache_key
    else:
        catalog_by_bento = st.session_state["_catalog_by_bento"]
    meal_fracs = [b["fraction"] for b in st.session_state.bentos]
    p_pb = int(
        st.session_state.bentos[0].get("protein_per_bento", round(proteins / max(n_comp, 1)))
    )
    allow_anim = [b.get("is_protein", False) for b in st.session_state.bentos]
    anchor_from_edit: int | None = None
    pending_rep = st.session_state.get("_pending_bento_row_replace")

    for i, bento in enumerate(st.session_state.bentos):
        if pending_rep is not None and int(pending_rep.get("bento_idx", -1)) == i:
            st.session_state.pop("_pending_bento_row_replace", None)
            df0 = (
                st.session_state.bento_frames[i]
                if i < len(st.session_state.get("bento_frames", []))
                else None
            )
            if df0 is not None and len(df0) > 0:
                nu = replace_bento_row_ingredient(
                    df0,
                    int(pending_rep["row_idx"]),
                    str(pending_rep["product_name"]),
                    catalog_by_bento[i],
                )
                clear_qty_widgets_for_bento(i)
                st.session_state.bento_frames[i] = nu
                st.session_state.bentos[i]["rows"] = nu.to_dicts()
                il = st.session_state.setdefault("ingredient_locks", [])
                while len(il) <= i:
                    il.append([])
                if len(il[i]) < len(nu):
                    il[i].extend([False] * (len(nu) - len(il[i])))
                elif len(il[i]) > len(nu):
                    il[i] = il[i][: len(nu)]
                rj = int(pending_rep["row_idx"])
                if 0 <= rj < len(il[i]):
                    il[i][rj] = True
                # Store the replaced row as edited so macro-aware rebalance knows the trigger
                rl = st.session_state.setdefault("_rebalance_locked_rows", {})
                rl[i] = rl.get(i, [])
                if rj not in rl[i]:
                    rl[i].append(rj)
                st.session_state["_rebalance_anchor_bento"] = i
                anchor_from_edit = i
            pending_rep = None

        prot_label = f" — {bento['protein_per_bento']}g prot"
        animal_label = " ◆ animal" if bento["is_protein"] else ""

        df_preview_hdr = (
            st.session_state.bento_frames[i]
            if i < len(st.session_state.get("bento_frames", []))
            else None
        )
        nr_hdr = len(df_preview_hdr) if df_preview_hdr is not None else 0
        gate_title = (
            f'Bento n°{i + 1} '
            f'<span style="font-size:0.8rem;color:#AD9E7B">'
            f'{bento["fraction"]:.0%}{prot_label}{animal_label}</span>'
        )
        render_bento_gate_header(bento_idx=i, title_inner_html=gate_title, meal_n_rows=nr_hdr)

        if bento["error"]:
            st.error(bento["error"].splitlines()[0])
            with st.expander("Détails"):
                st.code(bento["error"])
            continue

        df_view = (
            st.session_state.bento_frames[i]
            if i < len(st.session_state.get("bento_frames", []))
            else None
        )
        if df_view is not None and len(df_view) > 0:
            from datetime import datetime as _dt3
            print(f"[KOJIN {_dt3.now().strftime('%H:%M:%S.%f')[:-3]}] RENDER bento={i} q={df_view['Quantité (g)'].to_list()}", flush=True)
            aligned = align_bento_aliments_to_catalog(df_view, catalog_by_bento[i])
            if aligned is not None and aligned["Aliment"].to_list() != df_view["Aliment"].to_list():
                st.session_state.bento_frames[i] = aligned
                st.session_state.bentos[i]["rows"] = aligned.to_dicts()
                st.rerun()
            df_view = aligned
            _bento_targets = per_bento_macro_targets(
                float(energy), float(fat), float(carbs),
                bento["fraction"], float(p_pb),
            )
            _chart_fn = lambda container, _df=df_view, _tgt=_bento_targets: _render_bento_macro_charts(_df, _tgt, container=container)
            nu = render_bento_inline_table(
                bento_idx=i,
                df_view=df_view,
                catalog=catalog_by_bento[i],
                chart_renderer=_chart_fn,
            )
            if nu is not None:
                rk = st.session_state.pop("_replace_lock_mark", None)
                st.session_state.bento_frames[i] = nu
                st.session_state.bentos[i]["rows"] = nu.to_dicts()
                il = st.session_state.setdefault("ingredient_locks", [])
                while len(il) <= i:
                    il.append([])
                if len(il[i]) < len(nu):
                    il[i].extend([False] * (len(nu) - len(il[i])))
                elif len(il[i]) > len(nu):
                    il[i] = il[i][: len(nu)]
                if rk is not None and rk[0] == i and 0 <= rk[1] < len(il[i]):
                    il[i][rk[1]] = True
                anchor_from_edit = i
        elif not bento["rows"]:
            st.info("Aucun aliment trouvé pour ce bento.")
            continue
        else:
            rows = bento["rows"]
            df_fb = pl.DataFrame(rows)
            if "Aliment" in df_fb.columns and len(df_fb) > 0:
                fmt = format_product_display_label
                df_fb = df_fb.with_columns(
                    pl.col("Aliment").map_elements(
                        lambda x: fmt("" if x is None else str(x)),
                        return_dtype=pl.String,
                    )
                )
            st.dataframe(df_fb.to_pandas(), use_container_width=True, hide_index=True)
            # Affichage camemberts cible vs obtenu sous le tableau (fallback)
            _bento_targets = per_bento_macro_targets(
                float(energy), float(fat), float(carbs),
                bento["fraction"], float(p_pb),
            )
            _render_bento_macro_charts(df_fb, _bento_targets, container=st)

    sync_locked_bentos_from_ingredient_locks(n_comp)
    rebal_idx = st.session_state.pop("_rebalance_anchor_bento", None)
    if rebal_idx is None:
        rebal_idx = anchor_from_edit
    from datetime import datetime as _dt_pre
    print(f"[KOJIN {_dt_pre.now().strftime('%H:%M:%S.%f')[:-3]}] PRE-REBAL: rebal_idx={rebal_idx} anchor_from_edit={anchor_from_edit}", flush=True)
    if rebal_idx is not None and 0 <= rebal_idx < n_comp:
        bf = st.session_state.bento_frames
        if bf[rebal_idx] is not None and len(bf[rebal_idx]) > 0:
            il = st.session_state.setdefault("ingredient_locks", [])
            with st.spinner(
                "Rééquilibrage des autres repas (solveur nutritionnel ; peut prendre quelques secondes)…"
            ):
                new_all = run_daily_rebalance_after_anchor(
                    anchor_bento_idx=rebal_idx,
                    all_frames=bf,
                    ingredient_locks=il,
                    catalog_by_bento=catalog_by_bento,
                    energy=float(energy),
                    proteins_daily=float(proteins),
                    protein_per_bento=float(p_pb),
                    fat=float(fat),
                    carbs=float(carbs),
                    meal_fractions=meal_fracs,
                    allow_one_animal_list=allow_anim,
                    portion_legumes=float(portion_legumes),
                )
            from datetime import datetime as _dt
            for j, fr in enumerate(new_all):
                if j < len(st.session_state.bento_frames):
                    st.session_state.bento_frames[j] = fr
                    if j < len(st.session_state.bentos):
                        st.session_state.bentos[j]["rows"] = (
                            fr.to_dicts() if fr is not None and len(fr) > 0 else []
                        )
                    # Invalidate widget cache for any bento whose quantities changed,
                    # so the next run renders fresh widgets without stale Streamlit state.
                    clear_qty_widgets_for_bento(j)
            _rebal_q = st.session_state.bento_frames[rebal_idx]["Quantité (g)"].to_list() if st.session_state.bento_frames[rebal_idx] is not None else []
            print(f"[KOJIN {_dt.now().strftime('%H:%M:%S.%f')[:-3]}] bento_maker REBAL DONE: rebal_idx={rebal_idx} saved_q={_rebal_q}", flush=True)
            sync_locked_bentos_from_ingredient_locks(n_comp)
            st.rerun()
