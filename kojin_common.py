"""Shared utilities used by every page of the Kōjin Streamlit app.

Centralises:
- Japanese minimalist CSS theme (``apply_theme``)
- Data file paths and S3 hydration (``ensure_csv_from_s3``)
- Domain constants (regimes, goals, bento names…); filtres catalogue produits dans ``data_prep_nutriments``
- Polars data loading helpers (filtres préparation via ``filter_products_catalog``)
- The bento optimiser (``optimize_bento``)
- Provider-agnostic LangChain factory (``get_chat_llm``) : **Amazon Bedrock** ou **Groq**
  (HTTP OpenAI-compatible) selon ``LLM_PROVIDER`` / présence de ``GROQ_API_KEY`` ;
  compare optionnel Bedrock/OpenAI-compat (``get_compare_sql_chat_llm``).
"""

from __future__ import annotations

import os
import re

import numpy as np
import polars as pl
import streamlit as st
from scipy.optimize import lsq_linear, nnls

# ─── Paths ───────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "products_names_with_macro_nutriments.csv")
DUCKDB_PATH = os.path.join(DATA_DIR, "products.duckdb")
DUCKDB_TABLE = "products"


def ensure_csv_from_s3() -> None:
    """Download the products CSV from S3 if ``DATA_S3_URI`` is set and the
    file is not already present locally. No-op in dev without the env var."""
    s3_uri = os.environ.get("DATA_S3_URI")
    if not s3_uri or os.path.exists(CSV_PATH):
        return

    from urllib.parse import urlparse

    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        return

    import boto3

    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    os.makedirs(DATA_DIR, exist_ok=True)
    boto3.client("s3").download_file(bucket, key, CSV_PATH)


# ─── Theme ───────────────────────────────────────────────────────────────────

# Sections: root variables · sidebar · typography · widgets (sliders, selectbox, radio)
#           data tables · buttons · miscellaneous (expander, tabs, scrollbar)
_CUSTOM_CSS = """
<style>
:root {
    --ink: #1a1a1a;
    --paper: #fafaf8;
    --stone: #888;
    --line: #e0e0dc;
    --font-serif: Georgia, 'Times New Roman', serif;

}

/* Remplace la couleur primaire Streamlit (orange) par blanc dans la sidebar */
:root, [data-testid="stSidebar"] {
    --primary: #ffffff !important;
}
/* piste active (gauche du curseur) en blanc, piste inactive (droite) en gris */
[data-testid="stSidebar"] div[data-baseweb="slider"] > div > div > div:first-child {
    background-color: #ffffff !important;
}
[data-testid="stSidebar"] div[data-baseweb="slider"] > div > div > div:last-child {
    background-color: #555 !important;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--paper) !important;
    color: var(--ink) !important;
    font-family: var(--font-serif) !important;
    font-size: 16px;
    font-weight: normal;
    line-height: 1.7;
}

[data-testid="stSidebar"] {
    background-color: var(--ink) !important;
    color: var(--paper) !important;
}
[data-testid="stSidebar"] * {
    color: var(--paper) !important;
}
[data-testid="stSidebar"] label {
    color: var(--paper) !important;
    font-family: var(--font-serif) !important;
    font-weight: 400;
    font-size: 0.82rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] input {
    background-color: #2a2a2a !important;
    border: 1px solid #444 !important;
    color: var(--paper) !important;
}

/* slider thumb (poignée) en blanc */
[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
    background-color: #ffffff !important;
    border-color: #ffffff !important;
}
/* valeur affichée sous le thumb en blanc */
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stSliderThumbValue"] {
    color: var(--paper) !important;
}
/* textes labels et ticks slider en blanc */
[data-testid="stSidebar"] [data-testid="stSlider"] label,
[data-testid="stSidebar"] [data-testid="stSlider"] p,
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: var(--paper) !important;
    text-transform: none !important;
    letter-spacing: normal !important;
}
/* cercle du radio en blanc */
[data-testid="stSidebar"] [data-testid="stRadio"] [data-baseweb="radio"] [role="radio"] {
    border-color: #ffffff !important;
    background-color: transparent !important;
    width: 14px !important;
    height: 14px !important;
    min-width: 14px !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [data-baseweb="radio"] [role="radio"][aria-checked="true"] {
    background-color: #ffffff !important;
    border-color: #ffffff !important;
}
/* textes des options radio en blanc, plus petits */
[data-testid="stSidebar"] [data-testid="stRadio"] label span {
    color: var(--paper) !important;
    font-size: 0.75rem !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    gap: 6px !important;
    align-items: center !important;
}
/* boutons +/- des number inputs toujours visibles */
[data-testid="stSidebar"] [data-testid="stNumberInput"] button {
    opacity: 1 !important;
    visibility: visible !important;
    color: #ffffff !important;
    border-color: #555 !important;
    background-color: #2a2a2a !important;
}

h1 {
    font-family: var(--font-serif) !important;
    font-weight: bold !important;
    font-size: clamp(26px, 4vw, 40px) !important;
    line-height: 1.25 !important;
    letter-spacing: -0.5px !important;
    margin-bottom: 0.1em !important;
}

h2, h3 {
    font-family: var(--font-serif) !important;
    font-size: 18px !important;
    font-weight: normal !important;
    font-style: italic !important;
    line-height: 1.35 !important;
    letter-spacing: normal !important;
}

p {
    font-family: var(--font-serif) !important;
    font-size: 16px !important;
    font-weight: normal !important;
    line-height: 1.7 !important;
}

.subtitle {
    font-family: var(--font-serif);
    font-size: 1rem;
    color: var(--stone);
    letter-spacing: 0.02em;
    margin-bottom: 2rem;
}

.bento-header {
    font-family: var(--font-serif);
    font-weight: normal;
    font-style: italic;
    font-size: 1.4rem;
    letter-spacing: 0.04em;
    border-bottom: 1px solid var(--ink);
    padding-bottom: 0.4rem;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
}

.bento-fraction {
    font-size: 0.78rem;
    color: var(--stone);
    letter-spacing: 0.02em;
    margin-bottom: 0.8rem;
}

.product-count {
    font-family: var(--font-serif);
    font-size: 3rem;
    font-weight: bold;
    line-height: 1;
}
.product-count-label {
    font-size: 0.75rem;
    color: #AD9E7B;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.targets-box {
    background: var(--ink);
    color: var(--paper);
    padding: 1.2rem 1.5rem;
    border-radius: 2px;
    margin: 1rem 0;
    font-size: 0.85rem;
    line-height: 1.8;
    letter-spacing: 0.02em;
}
.targets-box strong { font-weight: 600; }

div[data-testid="stDataFrame"] {
    border: 1px solid var(--line) !important;
    border-radius: 0 !important;
}
div[data-testid="stDataFrame"] th {
    background-color: var(--ink) !important;
    color: var(--paper) !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
}

button[kind="primary"] {
    background-color: var(--ink) !important;
    color: var(--paper) !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: var(--font-serif) !important;
    font-weight: 400 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.7rem 2rem !important;
}
button[kind="primary"]:hover {
    background-color: #333 !important;
}

button[kind="secondary"] {
    background-color: var(--ink) !important;
    color: var(--paper) !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: var(--font-serif) !important;
    font-weight: 400 !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.4rem !important;
}
button[kind="secondary"] p,
button[kind="secondary"] div {
    color: var(--paper) !important;
}
button[kind="secondary"]:hover {
    background-color: #333 !important;
    color: var(--paper) !important;
}

button[kind="tertiary"] {
    text-transform: none !important;
    letter-spacing: normal !important;
    font-family: var(--font-serif) !important;
}

.stDivider { border-color: var(--line) !important; }

[data-testid="stMetricValue"] {
    font-family: var(--font-serif) !important;
    font-weight: bold !important;
}

.fraction-remaining {
    font-size: 0.8rem;
    color: var(--stone);
    letter-spacing: 0.02em;
    padding: 0.3rem 0;
}
.fraction-ok { color: var(--stone); }
.fraction-over { color: #c44; }

.sql-block {
    font-family: 'Menlo', 'Courier New', monospace;
    font-size: 0.78rem;
    color: var(--stone);
    background: #f3f3ee;
    padding: 0.7rem 1rem;
    border-left: 2px solid var(--ink);
    margin: 0.6rem 0;
    white-space: pre-wrap;
}
/* Search / text inputs in main content */
section[data-testid="stMain"] div[data-testid="stTextInput"] input {
    color: #fafaf8 !important;
    background-color: #1a1a1a !important;
    border: 1px solid #444 !important;
}
section[data-testid="stMain"] div[data-testid="stTextInput"] input::placeholder {
    color: #888 !important;
}
/* Number inputs (quantity columns in grams) in main content */
section[data-testid="stMain"] div[data-testid="stNumberInput"] input {
    color: #fafaf8 !important;
    background-color: #1a1a1a !important;
    border: 1px solid #444 !important;
}
</style>
"""


def apply_theme() -> None:
    """Inject the global CSS theme. Safe to call from every page."""
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


# ─── Domain constants ────────────────────────────────────────────────────────

NUTR_COLS = ["energy-kcal", "proteins", "fat", "carbohydrates", "fiber"]
NUTR_DISPLAY = {
    "energy-kcal": "kcal",
    "proteins": "Prot (g)",
    "fat": "Lip (g)",
    "carbohydrates": "Gluc (g)",
    "fiber": "Fibres (g)",
}
TAG_COLS_REGIME = ["halal", "vegan", "bio", "vegetarian", "gluten_free", "kascher", "no_palm_oil"]
ANIMAL_PROTEIN_TAGS = ["meat", "fish", "lait"]

REGIME_LABELS = {
    "": "Aucun",
    "Vegan": "Vegan",
    "Vegetarian": "Végétarien",
    "Halal": "Halal",
    "Casher": "Casher",
    "Sans Gluten": "Sans Gluten",
    "Bio": "Bio",
}

DAILY_ACTIVITY = {
    "Sédentaire (bureau, peu de marche)": 1.2,
    "Légèrement actif (marche, debout)": 1.375,
    "Actif (travail physique)": 1.55,
}

SPORT_FREQUENCY = {
    "Aucun": 0.0,
    "1–2 séances / semaine": 0.1,
    "3–4 séances / semaine": 0.2,
    "5+ séances / semaine": 0.35,
}

GOALS = {
    "Sèche musculaire": "lean",
    "Recomposition": "balanced",
    "Prise de masse": "bulk",
}

GOAL_PARAMS = {
    "lean":     (0.90, 2.2, 0.25, 200),
    "balanced": (1.00, 1.8, 0.30, 150),
    "bulk":     (1.15, 1.6, 0.25, 100),
}

BENTO_NAMES = ["n°1", "n°2", "n°3", "n°4", "n°5"]


OIL_PATTERN = r"(?i)\bhuile\b|\boil\b|\bhuile d|\bhuile de"


# ─── Targets ─────────────────────────────────────────────────────────────────

def compute_targets(gender: str, age: int, weight: float, height: float,
                    daily_activity: str, sport: str, goal: str):
    if gender == "Homme":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    multiplier = min(DAILY_ACTIVITY[daily_activity] + SPORT_FREQUENCY[sport], 1.95)
    tdee = bmr * multiplier

    cal_factor, prot_per_kg, fat_pct, portion_leg = GOAL_PARAMS[goal]
    energy = round(tdee * cal_factor)
    proteins = round(prot_per_kg * weight)
    fat = round((energy * fat_pct) / 9)
    carbs = round((energy - proteins * 4 - fat * 9) / 4)
    carbs = max(carbs, 50)

    return energy, proteins, fat, carbs, portion_leg


def format_product_display_label(s: str | None) -> str:
    """Affichage type « Fraises » : première lettre majuscule, le reste en minuscules."""
    if s is None:
        return ""
    t = str(s).strip()
    if len(t) <= 1:
        return t.upper() if t else t
    return t[0].upper() + t[1:].lower()


def normalize_catalog_product_names(df: pl.DataFrame) -> pl.DataFrame:
    """Uniformise ``product_name`` sur tout le catalogue (chargement CSV, préparation)."""
    if "product_name" not in df.columns or len(df) == 0:
        return df
    # Capitalise la 1re lettre, reste en minuscules — expressions natives Polars
    return df.with_columns(
        (
            pl.col("product_name").fill_null("").str.strip_chars()
            .str.slice(0, 1).str.to_uppercase()
            + pl.col("product_name").fill_null("").str.strip_chars()
            .str.slice(1).str.to_lowercase()
        ).alias("product_name")
    )


def catalog_rows_matching_name(catalog: pl.DataFrame, name: str) -> pl.DataFrame:
    """Filtre le catalogue sur ``product_name`` ; secours insensible à la casse (sessions anciennes)."""
    if not name or catalog is None or len(catalog) == 0 or "product_name" not in catalog.columns:
        return catalog.head(0)
    nm = str(name).strip()
    pm = catalog.filter(pl.col("product_name") == nm)
    if len(pm) > 0:
        return pm
    return catalog.filter(pl.col("product_name").str.to_lowercase() == nm.lower())


def canonical_product_name_map_lower(catalog: pl.DataFrame) -> dict[str, str]:
    """minuscules → nom canonique présent dans le catalogue (pour aligner colonne ``Aliment``)."""
    m: dict[str, str] = {}
    for p in catalog["product_name"].to_list():
        k = str(p).strip().lower()
        if k:
            m.setdefault(k, str(p))
    return m


def align_bento_aliments_to_catalog(df: pl.DataFrame | None, catalog: pl.DataFrame) -> pl.DataFrame | None:
    """Réécrit les ``Aliment`` des bentos avec le libellé du catalogue si équivalent modulo casse."""
    if df is None or len(df) == 0 or catalog is None or len(catalog) == 0:
        return df
    if "Aliment" not in df.columns:
        return df
    m = canonical_product_name_map_lower(catalog)
    names = [str(x) for x in df["Aliment"].to_list()]
    new_names = [m.get(nm.strip().lower(), nm) for nm in names]
    if new_names == names:
        return df
    return df.with_columns(pl.Series(name="Aliment", values=new_names, dtype=pl.String))


def format_exploration_results_for_display(df: pl.DataFrame | None) -> pl.DataFrame | None:
    """Met en forme ``product_name`` dans les tableaux Exploration (requêtes DuckDB)."""
    if df is None or len(df) == 0 or "product_name" not in df.columns:
        return df
    # Capitalise la 1re lettre — expressions natives Polars
    return df.with_columns(
        (
            pl.col("product_name").fill_null("").str.strip_chars()
            .str.slice(0, 1).str.to_uppercase()
            + pl.col("product_name").fill_null("").str.strip_chars()
            .str.slice(1).str.to_lowercase()
        ).alias("product_name")
    )


# ─── Data loading ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Chargement des données…")
def load_products(path: str) -> pl.DataFrame:
    from data_prep_nutriments import filter_products_catalog

    df = pl.read_csv(path, ignore_errors=True, separator=",", truncate_ragged_lines=True)
    df = filter_products_catalog(df)
    return normalize_catalog_product_names(df)


def apply_regime(products: pl.DataFrame, regime: str) -> pl.DataFrame:
    match regime:
        case "Vegan":
            return products.filter(pl.col("vegan") == True)
        case "Vegetarian":
            return products.filter((pl.col("vegetarian") == True) | (pl.col("meat") == False))
        case "Halal":
            return products.filter((pl.col("halal") == True) | (pl.col("vegetarian") == True))
        case "Casher":
            return products.filter(pl.col("kascher") == True)
        case "Sans Gluten":
            return products.filter(pl.col("gluten_free") == True)
        case "Bio":
            return products.filter(pl.col("bio") == True)
        case _:
            return products


def is_animal(row: dict) -> bool:
    return any(row.get(tag) is True for tag in ANIMAL_PROTEIN_TAGS)


def is_oil(row: dict) -> bool:
    name = row.get("product_name", "")
    return bool(re.search(OIL_PATTERN, name))


def exclude_animal_protein(products: pl.DataFrame) -> pl.DataFrame:
    mask = pl.lit(True)
    for tag in ANIMAL_PROTEIN_TAGS:
        if tag in products.columns:
            mask = mask & (pl.col(tag) == False)
    return products.filter(mask)


# ─── Optimisation ────────────────────────────────────────────────────────────

def _run_solver(M, targets, upper_bounds, solveur="hybride"):
    if solveur == "hybride":
        x_init, _ = nnls(M, targets)
        masque_x = x_init > 0
        indices = np.where(masque_x)[0]

        if len(indices) == 0:
            scores = np.sum(M.T, axis=1)
            indices = np.argsort(scores)[-50:]
            masque_x = np.zeros(M.shape[1], dtype=bool)
            masque_x[indices] = True

        M_f = M[:, masque_x]
        ub_f = upper_bounds[masque_x]
        result = lsq_linear(M_f, targets, bounds=(np.zeros(M_f.shape[1]), ub_f), method="bvls")
        return result.x, masque_x, indices
    else:
        x, _ = nnls(M, targets)
        return x, np.ones(M.shape[1], dtype=bool), np.arange(M.shape[1])


def optimize_bento(
    products_df: pl.DataFrame,
    user_targets: list[float],
    meal_fraction: float,
    portion_legumes: float,
    allow_one_animal: bool = False,
    ingredient_slot_ids: list[int] | None = None,
    ingredient_proportions: list[float] | None = None,
    proportion_weight: float = 0.25,
):
    if len(products_df) == 0:
        return None

    raw = np.array(user_targets, dtype=float)
    targets = np.array([
        raw[0] * meal_fraction,
        raw[1],
        raw[2] * meal_fraction,
        raw[3] * meal_fraction,
        25 * meal_fraction,
    ])

    include_legumes = portion_legumes > 0.0
    nutr_cols = list(NUTR_COLS)
    if include_legumes:
        nutr_cols.append("portion_legumes")
        targets = np.concatenate((targets, [portion_legumes * meal_fraction]))

    # Ajout des colonnes manquantes en un seul appel
    missing_cols = [col for col in nutr_cols if col not in products_df.columns]
    if missing_cols:
        products_df = products_df.with_columns([pl.lit(0.0).alias(col) for col in missing_cols])

    if "portion_maximale" not in products_df.columns:
        products_df = products_df.with_columns(pl.lit(200.0).alias("portion_maximale"))

    # fill_nan + fill_null en un seul appel
    products_df = products_df.with_columns([
        pl.col(col).fill_nan(0.0).fill_null(0.0) for col in nutr_cols
    ])

    arr = products_df.select(nutr_cols).to_numpy() / 100.0
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    M = arr.T
    upper_bounds = products_df["portion_maximale"].to_numpy()

    # Soft proportion constraints: guide the solver to respect the recipe's
    # ingredient ratios while keeping macro targets as the primary objective.
    # Each slot s contributes one extra row: sum(x[slot==s]) ≈ T_est * props[s].
    # Scaled by proportion_weight so macro rows dominate.
    if (
        ingredient_slot_ids is not None
        and ingredient_proportions is not None
        and len(ingredient_slot_ids) == len(products_df)
        and len(ingredient_proportions) > 0
    ):
        slot_ids_arr = np.array(ingredient_slot_ids, dtype=int)
        props = np.array(ingredient_proportions, dtype=float)
        prop_sum = props.sum()
        if prop_sum > 1e-9:
            props = props / prop_sum  # ensure they sum to 1
            T_est = max((raw[0] * meal_fraction) / 3.5, 1.0)
            extra_rows = []
            extra_targets = []
            for s in range(len(props)):
                row = (slot_ids_arr == s).astype(float) * proportion_weight
                extra_rows.append(row)
                extra_targets.append(T_est * props[s] * proportion_weight)
            M = np.vstack([M, np.array(extra_rows)])
            targets = np.concatenate([targets, extra_targets])

    x, _masque, sel_indices = _run_solver(M, targets, upper_bounds)

    # Indexation directe plutôt que filter + is_in (plus rapide)
    selected_products = products_df[sel_indices.tolist()]

    quantities = x
    pos_mask = quantities > 1e-6
    indices = np.where(pos_mask)[0]
    qty_list = quantities[pos_mask]

    if len(indices) > 0:
        animal_items = []
        oil_items = []
        other_items = []
        for j, idx in enumerate(indices):
            row = selected_products.row(idx, named=True)
            if is_animal(row):
                animal_items.append((idx, qty_list[j]))
            elif is_oil(row):
                oil_items.append((idx, qty_list[j]))
            else:
                other_items.append((idx, qty_list[j]))

        kept = list(other_items)

        if allow_one_animal and animal_items:
            animal_items.sort(key=lambda t: t[1], reverse=True)
            kept.append(animal_items[0])

        if oil_items:
            oil_items.sort(key=lambda t: t[1], reverse=True)
            kept.append(oil_items[0])

        if kept:
            indices = np.array([t[0] for t in kept])
            qty_list = np.array([t[1] for t in kept])

    rows = []
    for i, q in zip(indices, qty_list):
        row = selected_products.row(i, named=True)
        nutr_per_portion = {}
        for nc in NUTR_COLS:
            val = row.get(nc, 0.0) or 0.0
            nutr_per_portion[nc] = round(val * q / 100.0, 1)
        rows.append({
            "Aliment": row["product_name"],
            "Quantité (g)": round(q, 1),
            **{NUTR_DISPLAY[nc]: nutr_per_portion[nc] for nc in NUTR_COLS},
        })

    if not rows:
        return None

    return pl.DataFrame(rows).sort("Quantité (g)", descending=True)


# ─── Data preparation pipeline ───────────────────────────────────────────────

def run_data_prep():
    from data_prep_nutriments import (
        add_tags,
        clean_categories,
        download_data,
        filter_products_catalog,
        get_nutriments,
    )

    with st.spinner("Téléchargement depuis Hugging Face…"):
        df = download_data(True)
    with st.spinner("Extraction des nutriments…"):
        df = get_nutriments(df)
    with st.spinner("Nettoyage des catégories…"):
        df = clean_categories(df)
    with st.spinner("Ajout des tags…"):
        df = add_tags(df)
    with st.spinner("Filtrage catalogue…"):
        df = filter_products_catalog(df)
    with st.spinner("Sauvegarde…"):
        df = df.collect()
        df = normalize_catalog_product_names(df)
        os.makedirs(DATA_DIR, exist_ok=True)
        df.write_csv(CSV_PATH)
    return df


# ─── LLM — Bedrock OU Groq (LangChain), agnostic AWS vs local ────────────────
#
# Exploration : référence = ``reference_llm_provider()`` → Bedrock (**IAM**), Groq ou OpenAI.
# Définir explicitement ``LLM_PROVIDER`` en prod ECS : ``bedrock``, ``groq`` ou ``openai``.
# Mode ``auto`` : OpenAI si ``OPENAI_API_KEY``, sinon Groq si ``GROQ_API_KEY``, sinon Bedrock.

BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.amazon.nova-pro-v1:0").strip()
# Agent secondaire (compare) — Bedrock uniquement ou ``OPENAI_COMPAT_*``.
BEDROCK_COMPARE_MODEL_ID = os.environ.get("BEDROCK_COMPARE_MODEL_ID", "").strip()

GROQ_OPENAI_BASE = "https://api.groq.com/openai/v1"
GROQ_MODEL_ID = os.environ.get("GROQ_MODEL_ID", "llama-3.1-8b-instant").strip()

OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "gpt-4o").strip()

ANTHROPIC_MODEL_ID = os.environ.get("ANTHROPIC_MODEL_ID", "claude-haiku-4-5-20251001").strip()


def _load_claude_key_file() -> None:
    """Load ANTHROPIC_API_KEY from .claude_key if not already set."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    for path in (".claude_key", os.path.expanduser("~/.claude_key")):
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("CLAUDE_API_KEY="):
                        os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1]
                        return
                    if line.startswith("ANTHROPIC_API_KEY="):
                        os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1]
                        return


_load_claude_key_file()


def _secret_or_env(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is not None and str(raw).strip():
        return str(raw).strip()
    try:
        val = st.secrets.get(name)
        return str(val).strip() if val else None
    except (AttributeError, FileNotFoundError, KeyError):
        return None


def reference_llm_provider() -> str:
    """``anthropic`` | ``openai`` | ``groq`` | ``bedrock`` — résolu au moment de l'appel."""
    mode = (os.environ.get("LLM_PROVIDER") or "auto").strip().lower()
    if mode == "anthropic":
        return "anthropic"
    if mode == "openai":
        return "openai"
    if mode == "groq":
        return "groq"
    if mode == "bedrock":
        return "bedrock"
    # auto: Anthropic > OpenAI > Groq > Bedrock
    if _secret_or_env("ANTHROPIC_API_KEY"):
        return "anthropic"
    if _secret_or_env("OPENAI_API_KEY"):
        return "openai"
    if _secret_or_env("GROQ_API_KEY"):
        return "groq"
    return "bedrock"


def reference_llm_label() -> str:
    """Libellé court pour titres Exploration / logs."""
    p = reference_llm_provider()
    if p == "anthropic":
        return f"Anthropic — {ANTHROPIC_MODEL_ID}"
    if p == "openai":
        return f"OpenAI — {OPENAI_MODEL_ID}"
    if p == "groq":
        return f"Groq — {GROQ_MODEL_ID}"
    return str(BEDROCK_MODEL_ID)


def reference_model_id_for_metrics() -> str:
    p = reference_llm_provider()
    if p == "anthropic":
        return ANTHROPIC_MODEL_ID
    if p == "openai":
        return OPENAI_MODEL_ID
    if p == "groq":
        return GROQ_MODEL_ID
    return BEDROCK_MODEL_ID


def _bedrock_region() -> str:
    return (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "eu-west-1"
    )


def _build_bedrock_chat(model_id: str, temperature: float, max_tokens: int):
    from langchain_aws import ChatBedrockConverse

    return ChatBedrockConverse(
        model_id=model_id,
        region_name=_bedrock_region(),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _build_openai_chat(temperature: float, max_tokens: int):
    from langchain_openai import ChatOpenAI

    key = _secret_or_env("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OpenAI sélectionné (LLM_PROVIDER=openai ou auto avec clé attendue) mais "
            "`OPENAI_API_KEY` est absent — définit la variable d'environnement ou "
            "ajoute `OPENAI_API_KEY` dans `.streamlit/secrets.toml`."
        )
    return ChatOpenAI(
        model=OPENAI_MODEL_ID,
        api_key=key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _build_anthropic_chat(temperature: float, max_tokens: int):
    from langchain_anthropic import ChatAnthropic

    key = _secret_or_env("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "Anthropic sélectionné mais `ANTHROPIC_API_KEY` est absent — "
            "définit la variable d'environnement ou place la clé dans `.claude_key`."
        )
    return ChatAnthropic(
        model=ANTHROPIC_MODEL_ID,
        api_key=key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _build_groq_chat(temperature: float, max_tokens: int):
    from langchain_openai import ChatOpenAI

    key = _secret_or_env("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "Groq sélectionné (LLM_PROVIDER=groq ou auto avec clé attendue) mais "
            "`GROQ_API_KEY` est absent — définit la variable d'environnement ou "
            "ajoute `GROQ_API_KEY` dans `.streamlit/secrets.toml`."
        )
    return ChatOpenAI(
        base_url=GROQ_OPENAI_BASE.rstrip("/"),
        model=GROQ_MODEL_ID,
        api_key=key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


@st.cache_resource(show_spinner=False)
def _cached_ref_llm_anthropic(temperature: float, max_tokens: int):
    return _build_anthropic_chat(temperature, max_tokens)


@st.cache_resource(show_spinner=False)
def _cached_ref_llm_openai(temperature: float, max_tokens: int):
    return _build_openai_chat(temperature, max_tokens)


@st.cache_resource(show_spinner=False)
def _cached_ref_llm_groq(temperature: float, max_tokens: int):
    return _build_groq_chat(temperature, max_tokens)


@st.cache_resource(show_spinner=False)
def _cached_ref_llm_bedrock(model_id: str, temperature: float, max_tokens: int):
    return _build_bedrock_chat(model_id, temperature, max_tokens)


def get_chat_llm(temperature: float = 0.1, max_tokens: int = 800):
    """Référence Exploration : Anthropic, OpenAI, Groq ou Bedrock selon ``LLM_PROVIDER`` / clés présentes."""
    p = reference_llm_provider()
    if p == "anthropic":
        return _cached_ref_llm_anthropic(temperature, max_tokens)
    if p == "openai":
        return _cached_ref_llm_openai(temperature, max_tokens)
    if p == "groq":
        return _cached_ref_llm_groq(temperature, max_tokens)
    return _cached_ref_llm_bedrock(BEDROCK_MODEL_ID, temperature, max_tokens)


def compare_agent_mode() -> str | None:
    """``bedrock_compare`` > ``openai_compat`` > ``None`` (priorité aux deuxième agent Bedrock)."""
    if BEDROCK_COMPARE_MODEL_ID:
        return "bedrock_compare"
    if os.environ.get("OPENAI_COMPAT_BASE_URL", "").strip() and os.environ.get("OPENAI_COMPAT_MODEL", "").strip():
        return "openai_compat"
    return None


def compare_sql_llm_label() -> str | None:
    """Label affiché pour l'agent de comparaison (Bedrock Llama/custom ou compatible OpenAI)."""
    mode = compare_agent_mode()
    if mode == "bedrock_compare":
        return BEDROCK_COMPARE_MODEL_ID
    if mode == "openai_compat":
        return os.environ.get("OPENAI_COMPAT_MODEL", "").strip() or None
    return None


@st.cache_resource(show_spinner=False)
def get_compare_sql_chat_llm(temperature: float = 0.0, max_tokens: int = 600):
    """Second agent Exploration : soit ``BEDROCK_COMPARE_MODEL_ID``, soit compatible OpenAI (``OPENAI_COMPAT_*``)."""
    if BEDROCK_COMPARE_MODEL_ID:
        return _build_bedrock_chat(
            BEDROCK_COMPARE_MODEL_ID, temperature, max_tokens
        )

    base_url = os.environ.get("OPENAI_COMPAT_BASE_URL", "").strip()
    model_name = os.environ.get("OPENAI_COMPAT_MODEL", "").strip()
    if not base_url or not model_name:
        raise RuntimeError(
            "Missing OPENAI_COMPAT_BASE_URL and/or OPENAI_COMPAT_MODEL "
            "(required for compare SQL agent)."
        )

    from langchain_openai import ChatOpenAI

    api_key = os.environ.get("OPENAI_COMPAT_API_KEY", "-")
    url = base_url.rstrip("/")
    return ChatOpenAI(
        base_url=url,
        model=model_name,
        api_key=api_key if api_key else "-",
        temperature=temperature,
        max_tokens=max_tokens,
    )


def append_exploration_metrics_event(event: dict) -> None:
    """Append one JSON line to an NDJSON file when ``KOJIN_EXPLORATION_LOG_JSONL`` is truthy.

    Used to monitor latency, SQL equality, and DuckDB success in CloudWatch Logs
    (ship the file or stdout) or ad-hoc analysis. Never raises into the UI.
    """
    flag = os.environ.get("KOJIN_EXPLORATION_LOG_JSONL", "").lower()
    if flag not in ("1", "true", "yes", "on"):
        return
    import json
    from datetime import datetime, timezone

    path = os.environ.get("KOJIN_EXPLORATION_LOG_PATH", "").strip()
    if not path:
        path = "/tmp/kojin_exploration_metrics.ndjson"

    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass
