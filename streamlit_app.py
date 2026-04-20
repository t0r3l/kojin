import streamlit as st
import polars as pl
import numpy as np
from scipy.optimize import nnls, lsq_linear
import os
import traceback

st.set_page_config(
    page_title="Kōjin — コージン",
    page_icon="◯",
    layout="wide",
)

# ─── Japanese minimalist theme ──────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+JP:wght@200;400;700&family=Inter:wght@300;400;500&display=swap');

:root {
    --ink: #1a1a1a;
    --paper: #fafaf8;
    --stone: #888;
    --line: #e0e0dc;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--paper) !important;
    color: var(--ink) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 300;
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
    font-family: 'Inter', sans-serif !important;
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

h1 {
    font-family: 'Noto Serif JP', serif !important;
    font-weight: 200 !important;
    letter-spacing: 0.12em;
    font-size: 2.8rem !important;
    margin-bottom: 0.1em !important;
}

h2, h3 {
    font-family: 'Noto Serif JP', serif !important;
    font-weight: 400 !important;
    letter-spacing: 0.06em;
}

.subtitle {
    font-size: 0.9rem;
    color: var(--stone);
    letter-spacing: 0.08em;
    margin-bottom: 2rem;
}

.bento-header {
    font-family: 'Noto Serif JP', serif;
    font-weight: 200;
    font-size: 1.4rem;
    letter-spacing: 0.1em;
    border-bottom: 1px solid var(--ink);
    padding-bottom: 0.4rem;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
}

.bento-fraction {
    font-size: 0.78rem;
    color: var(--stone);
    letter-spacing: 0.06em;
    margin-bottom: 0.8rem;
}

.product-count {
    font-family: 'Noto Serif JP', serif;
    font-size: 3rem;
    font-weight: 200;
    line-height: 1;
}
.product-count-label {
    font-size: 0.75rem;
    color: var(--stone);
    text-transform: uppercase;
    letter-spacing: 0.12em;
}

.targets-box {
    background: var(--ink);
    color: var(--paper);
    padding: 1.2rem 1.5rem;
    border-radius: 2px;
    margin: 1rem 0;
    font-size: 0.85rem;
    line-height: 1.8;
    letter-spacing: 0.03em;
}
.targets-box strong { font-weight: 500; }

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
    font-family: 'Noto Serif JP', serif !important;
    font-weight: 400 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.7rem 2rem !important;
}
button[kind="primary"]:hover {
    background-color: #333 !important;
}

.stDivider { border-color: var(--line) !important; }

[data-testid="stMetricValue"] {
    font-family: 'Noto Serif JP', serif !important;
    font-weight: 200 !important;
}

.fraction-remaining {
    font-size: 0.8rem;
    color: var(--stone);
    letter-spacing: 0.04em;
    padding: 0.3rem 0;
}
.fraction-ok { color: var(--stone); }
.fraction-over { color: #c44; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "products_names_with_macro_nutriments.csv")


def _ensure_csv_from_s3() -> None:
    """Download the products CSV from S3 if ``DATA_S3_URI`` is set and
    the file is not already present locally. No-op in dev without the env var.
    """
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


_ensure_csv_from_s3()

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

BENTO_NAMES = ["一 Ichi", "二 Ni", "三 San", "四 Shi", "五 Go"]


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


# ─── Data loading ────────────────────────────────────────────────────────────

EXCLUDED_CATEGORIES = (
    # alcohol
    r"boissons-alcoolisees|bieres|biere|vins,|,vins$|vins-blancs|vins-rouges|"
    r"spiritueux|whisky|rhum|vodka|liqueurs?|cocktail|aperitifs?-alcoolise|"
    r"cidres?|champagnes?|cognac|gin,|,gin$|wine|wines|"
    # drinks
    r"sodas|soft-drinks|energy-drinks|"
    r"jus-de-fruits|jus-de-legumes|jus-d|fruit-juices|vegetable-juices|"
    r"nectars|smoothies|"
    # supplements & powders
    r"proteines-en-poudre|protein-powder|whey|caseine|bcaa|"
    r"complements?-alimentaires|mass-gainer|creatine|isolat-de-proteine|"
    r"protein-shake|gainers|barres-proteinees|protein-bars|energy-bars|"
    r"complements-pour-le-bodybuilding|"
    # sugars & sweets
    r"sucres,|,sucres$|sucre-blanc|sucre-raffine|sucre-en-poudre|sucre-glace|"
    r"bonbons|candies|confiseries|confectionery|sweet-snacks|snacks-sucres|"
    r"sirops|syrups|sirop-de-glucose|sirop-de-fructose|"
    r"caramels|marshmallow|guimauves|reglisse|nougat|pralines|dragees|"
    r"pates-a-tartiner-sucrees|pates-de-fruits|"
    r"chewing-gum|gommes-a-macher|"
    r"cereales-pour-petit-dejeuner|breakfast-cereals|"
    r"pop-tarts|brownie|cookie|biscuits|"
    r"gateaux|cakes|muffins|donuts|beignets|"
    r"glaces|ice-creams|sorbets|desserts|"
    r"pancakes|crepes|gaufres|waffles|viennoiseries|"
    r"sucettes|lollipops|"
    # sauces & condiments
    r"sauces|ketchup|moutardes|mustards|mayonnaises|"
    r"vinaigrettes|salad-dressings|dressings|"
    r"condiments|"
    # snacks & chips
    r"chips-et-frites|chips-and-fries|crisps|potato-crisps|"
    r"snacks-sales|salty-snacks|amuse-gueules|appetizers|"
    r"biscuits-aperitifs|tortillas|nachos|"
    r"barres|bars|cereal-bars|"
    # prepared / cooked foods
    r"plats-prepares|prepared-meals|plats-cuisines|ready-meals|"
    r"pizzas|quiches|tartes-salees|"
    r"sandwiches|sandwichs|wraps|burgers|"
    r"salades-composees|coleslaw|"
    r"plats-a-base-de-pates|plats-a-base-de-riz|"
    r"plats-traiteur|entrees-et-snacks|"
    r"soupes|soups|potages|velout|"
    r"surgeles|frozen-foods"
)

EXCLUDED_NAMES = (
    r"whey|protein.?powder|protéines? en poudre|proteine en poudre|"
    r"casein|caséine|bcaa|mass.?gainer|créatine|creatine|"
    r"isolat|protein.?shake|protein.?bar|barre protéinée|barre proteinee|"
    r"pre.?workout|post.?workout|"
    r"iso.?whey|iso.?protein|isofood|iso.?food|"
    r"meal.?replacement|nutrition.?shake|muscle.?milk|"
    r"mutant|powerbar|musclepharm|muscle.?pharm|orgain|"
    r"candy|bonbon|marshmallow|guimauve|gummy|gummies|"
    r"chewing.?gum|nougat|caramel|praline|dragée|dragee|réglisse|reglisse|"
    r"melting.?heart|tropical.?splash|skittles|haribo|"
    r"pop.?tart|brownie|fudge|cookie|"
    r"energy.?drink|energy.?gel|"
    r"sirop|syrup|"
    r"collag[eè]ne|spiruline|chlorell[ea]|"
    r"g[eé]lule|capsule|comprim[eé]|"
    r"huile essentielle|essential oil|"
    r"m[eé]latonine|ashwagandha|rhodiola|"
    r"charbon v[eé]g[eé]tal|detox|minceur|aminciss|"
    r"huile de foie de morue|cod liver oil|"
    r"superfood|moringa|baobab.?en.?poudre|açaï.?en.?poudre|"
    r"dietary.?supplement|milkshake|"
    # chips & snacks (branded & generic)
    r"chips|crisps|pringles|doritos|nachos|lays|cheetos|"
    r"potato.?chip|tortilla.?chip|corn.?chip|kettle.?chip|"
    r"popcorn|crackers|bretzels?|pretzel|"
    r"snack.?mix|trail.?mix|"
    # alcohol in name
    r"\bwine\b|\bvin blanc\b|\bvin rouge\b|\bvin rosé\b|"
    r"\bbeer\b|\bbière\b|\bale\b|\blager\b|"
    # powders & iso (not spices/cocoa)
    r"\biso\b|protein.?powder|"
    r"meal.?replacement.?powder|nutrition.?powder|"
    # juices & drinks
    r"fruit.?shoot|pur.?jus|\bjus de\b|\bjuice\b|\bnectar\b|"
    r"\bsoda\b|\bcola\b|\bfanta\b|\bsprite\b|"
    # sweets
    r"sucette|lollipop|ice.?cream|crème glacée|glace |sorbet|"
    r"gâteau|gateau|cake|muffin|donut|beignet|"
    r"crêpe|pancake|waffle|gaufre|viennoiserie|"
    # sauces
    r"\bsauce\b|ketchup|moutarde|mustard|"
    r"mayonnaise|\bmayo\b|vinaigrette|dressing|"
    # prepared / cooked foods
    r"pizza|lasagne|quiche|gratin|"
    r"sandwich|burger|wrap |croque.?monsieur|croque.?madame|"
    r"plat.?préparé|plat.?cuisiné|plat.?prepare|plat.?cuisine"
)


@st.cache_data(show_spinner="Chargement des données…")
def load_products(path: str) -> pl.DataFrame:
    df = pl.read_csv(path, ignore_errors=True, separator=",", truncate_ragged_lines=True)
    df = df.filter(
        (pl.col("nova_group").is_null() | (pl.col("nova_group") < 4))
        & ~pl.col("categories").str.to_lowercase().str.contains(EXCLUDED_CATEGORIES)
        & ~pl.col("product_name").str.to_lowercase().str.contains(EXCLUDED_NAMES)
        & ~(
            (pl.col("carbohydrates") > 60)
            & (pl.col("proteins") < 5)
        )
        & (pl.col("energy-kcal") > 0)
        & (pl.col("fiber") <= 40)
        & (pl.col("proteins") <= 85)
        & (pl.col("fat") <= 100)
        & (pl.col("carbohydrates") <= 100)
    )
    return df


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


OIL_PATTERN = r"(?i)\bhuile\b|\boil\b|\bhuile d|\bhuile de"


def is_animal(row: dict) -> bool:
    return any(row.get(tag) is True for tag in ANIMAL_PROTEIN_TAGS)


def is_oil(row: dict) -> bool:
    import re
    name = row.get("product_name", "")
    return bool(re.search(OIL_PATTERN, name))


def exclude_animal_protein(products: pl.DataFrame) -> pl.DataFrame:
    mask = pl.lit(True)
    for tag in ANIMAL_PROTEIN_TAGS:
        if tag in products.columns:
            mask = mask & (pl.col(tag) == False)
    return products.filter(mask)


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
):
    if len(products_df) == 0:
        return None

    # user_targets = [energy, proteins, fat, carbs]
    # protein is already per-bento (equal split), fraction applies to the rest
    raw = np.array(user_targets, dtype=float)
    targets = np.array([
        raw[0] * meal_fraction,   # energy-kcal
        raw[1],                   # proteins (already per-bento)
        raw[2] * meal_fraction,   # fat
        raw[3] * meal_fraction,   # carbohydrates
        25 * meal_fraction,       # fiber
    ])

    include_legumes = portion_legumes > 0.0
    nutr_cols = list(NUTR_COLS)
    if include_legumes:
        nutr_cols.append("portion_legumes")
        targets = np.concatenate((targets, [portion_legumes * meal_fraction]))

    for col in nutr_cols:
        if col not in products_df.columns:
            products_df = products_df.with_columns(pl.lit(0.0).alias(col))

    if "portion_maximale" not in products_df.columns:
        products_df = products_df.with_columns(pl.lit(200.0).alias("portion_maximale"))

    for col in nutr_cols:
        products_df = products_df.with_columns(pl.col(col).fill_nan(0.0).fill_null(0.0))

    arr = products_df.select(nutr_cols).to_numpy() / 100.0
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    M = arr.T
    upper_bounds = products_df["portion_maximale"].to_numpy()

    x, masque, sel_indices = _run_solver(M, targets, upper_bounds)

    selected_products = products_df.filter(
        pl.Series(range(len(products_df))).is_in(sel_indices.tolist())
    )

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


# ─── Data preparation ────────────────────────────────────────────────────────

def run_data_prep():
    from data_prep_nutriments import download_data, get_nutriments, clean_categories, add_tags

    with st.spinner("Téléchargement depuis Hugging Face…"):
        df = download_data(True)
    with st.spinner("Extraction des nutriments…"):
        df = get_nutriments(df)
    with st.spinner("Nettoyage des catégories…"):
        df = clean_categories(df)
    with st.spinner("Ajout des tags…"):
        df = add_tags(df)
    with st.spinner("Sauvegarde…"):
        df = df.collect()
        os.makedirs(DATA_DIR, exist_ok=True)
        df.write_csv(CSV_PATH)
    return df


# ─── UI ──────────────────────────────────────────────────────────────────────

st.title("Kōjin — コージン")
st.markdown('<p class="subtitle">donnons à votre corps les repas qu&#39;il mérite</p>', unsafe_allow_html=True)

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
regime = st.sidebar.selectbox("Type", list(REGIME_LABELS.keys()), format_func=lambda k: REGIME_LABELS[k])

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
        st.session_state[f"frac_{i}"] = max(FRAC_MIN, round(remaining * proportion / FRAC_STEP) * FRAC_STEP)

    # Correct rounding drift on the last other slider
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
        st.session_state[f"_prev_frac_{i}"] = st.session_state.get(f"frac_{i}", round(1.0 / num_bentos, 2))

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
    f'<p class="fraction-remaining {cls}">'
    f'Total : {total_fraction:.0%}</p>',
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
    else:
        used_codes: set[str] = set()

        for idx in range(num_bentos):
            frac = fractions[idx]
            name = BENTO_NAMES[idx]

            is_protein_bento = idx == protein_bento
            if is_protein_bento:
                bento_products = base_products
            else:
                bento_products = exclude_animal_protein(base_products)

            if used_codes:
                bento_products = bento_products.filter(~pl.col("code").is_in(list(used_codes)))

            bento_targets = [energy, protein_per_bento, fat, carbs]

            try:
                bento = optimize_bento(
                    bento_products, bento_targets, frac,
                    portion_legumes,
                    allow_one_animal=is_protein_bento,
                )
            except Exception as exc:
                st.error(f"Erreur bento {name} : {exc}")
                st.code(traceback.format_exc())
                bento = None

            if bento is not None and len(bento) > 0:
                selected_names = bento["Aliment"].to_list()
                matched = bento_products.filter(pl.col("product_name").is_in(selected_names))
                used_codes.update(matched["code"].to_list())

            prot_label = f" — {protein_per_bento}g prot"
            animal_label = " ◆ animal" if is_protein_bento else ""
            st.markdown(
                f'<div class="bento-header">弁当 {name} '
                f'<span style="font-size:0.8rem;color:#888">'
                f'{frac:.0%}{prot_label}{animal_label}</span></div>',
                unsafe_allow_html=True,
            )

            if bento is not None and len(bento) > 0:
                st.dataframe(
                    bento.to_pandas(),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("Aucun aliment trouvé pour ce bento.")
