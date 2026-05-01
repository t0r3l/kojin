import polars as pl
from huggingface_hub import hf_hub_download
import re
from typing import TypeVar

# ─── Filtres source Open Food Facts (France, non obsolète) ─────────────────────


def expr_france_not_obsolete() -> pl.Expr:
    """Produits listés pour la France et non marqués obsolètes dans le Parquet OFF."""
    return pl.col("countries_tags").list.contains("en:france") & (pl.col("obsolete") == False)


def expr_product_name_french() -> pl.Expr:
    """Après explode/unnest de ``product_name`` : ne garder que le libellé français."""
    return pl.col("lang") == "fr"


# ─── Filtres catalogue exporté (CSV / app Bento + Exploration) ───────────────
# Règles historiquement dans ``kojin_common.load_products`` : centralisées ici
# pour que ``run_data_prep`` et le script CLI produisent le même jeu que l’app.

EXCLUDED_CATEGORIES = (
    r"boissons-alcoolisees|bieres|biere|vins,|,vins$|vins-blancs|vins-rouges|"
    r"spiritueux|whisky|rhum|vodka|liqueurs?|cocktail|aperitifs?-alcoolise|"
    r"cidres?|champagnes?|cognac|gin,|,gin$|wine|wines|"
    r"alcools|alcohols|"
    r"sodas|soft-drinks|energy-drinks|"
    r"jus-de-fruits|jus-de-legumes|jus-d|fruit-juices|vegetable-juices|"
    r"nectars|smoothies|"
    r"proteines-en-poudre|protein-powder|whey|caseine|bcaa|"
    r"complements?-alimentaires|mass-gainer|creatine|isolat-de-proteine|"
    r"protein-shake|gainers|barres-proteinees|protein-bars|energy-bars|"
    r"complements-pour-le-bodybuilding|"
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
    r"sauces|ketchup|moutardes|mustards|mayonnaises|"
    r"vinaigrettes|salad-dressings|dressings|"
    r"condiments|"
    r"chips-et-frites|chips-and-fries|crisps|potato-crisps|"
    r"snacks-sales|salty-snacks|amuse-gueules|appetizers|"
    r"biscuits-aperitifs|tortillas|nachos|"
    r"barres|bars|cereal-bars|"
    r"plats-prepares|prepared-meals|plats-cuisines|ready-meals|"
    r"pizzas|quiches|tartes-salees|"
    r"sandwiches|sandwichs|wraps|burgers|"
    r"salades-composees|coleslaw|"
    r"plats-a-base-de-pates|plats-a-base-de-riz|"
    r"plats-traiteur|entrees-et-snacks|"
    r"soupes|soups|potages|velout|"
    r"surgeles|frozen-foods|"
    r"fast-food|restauration-rapide|menus-fast-food"
)

EXCLUDED_NAMES = (
    r"\bpowder\b|\bpoudre\b|protéines?|proteins?|\bsuper\b|"
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
    r"superfood|superaliment|moringa|baobab.?en.?poudre|açaï.?en.?poudre|"
    r"mix.*immunit|mix.*super|"
    r"dietary.?supplement|milkshake|"
    r"chips|crisps|pringles|doritos|nachos|lays|cheetos|"
    r"potato.?chip|tortilla.?chip|corn.?chip|kettle.?chip|"
    r"popcorn|crackers|bretzels?|pretzel|"
    r"snack.?mix|trail.?mix|"
    # Alcools : domaines, châteaux, crus, cuvées…
    r"ch[âa]teau |domaine |cuv[ée]e |cru |vignoble|vignerons?|"
    r"\bvin\b|\bvins\b|\bbière\b|\bbieres?\b|\bbeer\b|\bale\b|\blager\b|"
    r"\bwine\b|\bvin blanc\b|\bvin rouge\b|\bvin rosé\b|"
    r"champagne|prosecco|mousseux|crémant|bordeaux|bourgogne|"
    r"spiritueux|whisky|whiskey|rhum|vodka|cognac|armagnac|"
    r"pastis|absinthe|tequila|mezcal|"
    # Plats préparés / menus fast-food
    r"happy.?meal|maxi.?best|best.?of|big.?mac|mc.?nugget|"
    r"mc.?donald|mcdo|\bkfc\b|quick.?menu|menu.?enfant|"
    r"plat.?préparé|plat.?cuisiné|plat.?prepare|plat.?cuisine|"
    # Autres
    r"\biso\b|protein.?powder|"
    r"meal.?replacement.?powder|nutrition.?powder|"
    r"fruit.?shoot|pur.?jus|\bjus de\b|\bjuice\b|\bnectar\b|"
    r"\bsoda\b|\bcola\b|\bfanta\b|\bsprite\b|"
    r"sucette|lollipop|ice.?cream|crème glacée|glace |sorbet|"
    r"gâteau|gateau|cake|muffin|donut|beignet|"
    r"crêpe|pancake|waffle|gaufre|viennoiserie|"
    r"\bsauce\b|ketchup|moutarde|mustard|"
    r"mayonnaise|\bmayo\b|vinaigrette|dressing|"
    r"pizza|lasagne|quiche|gratin|"
    r"sandwich|burger|wrap |croque.?monsieur|croque.?madame"
)


def expr_catalog_export_filters() -> pl.Expr:
    """Filtres communs sur le DataFrame produits (post extract macros, post tags)."""
    return (
        (pl.col("nova_group").is_null() | (pl.col("nova_group") < 4))
        & ~pl.col("categories").str.to_lowercase().str.contains(EXCLUDED_CATEGORIES)
        & ~pl.col("product_name").str.to_lowercase().str.contains(EXCLUDED_NAMES)
        & ~((pl.col("carbohydrates") > 60) & (pl.col("proteins") < 5))
        & (pl.col("energy-kcal") > 0)
        & (pl.col("fiber") <= 40)
        & (pl.col("proteins") <= 85)
        & (pl.col("fat") <= 100)
        & (pl.col("carbohydrates") <= 100)
        # Exclure produits sans aucune valeur de macronutriment
        & ((pl.col("proteins") + pl.col("fat") + pl.col("carbohydrates") + pl.col("fiber")) > 0)
        # Exclure produits mal renseignés : kcal incohérents avec macros
        & (pl.col("energy-kcal") < (pl.col("proteins") * 4 + pl.col("fat") * 9 + pl.col("carbohydrates") * 4) * 2)
        & (pl.col("energy-kcal") > (pl.col("proteins") * 4 + pl.col("fat") * 9 + pl.col("carbohydrates") * 4) * 0.3)
    )


FrameT = TypeVar("FrameT", pl.DataFrame, pl.LazyFrame)


def filter_products_catalog(df: FrameT) -> FrameT:
    """Applique ``expr_catalog_export_filters()`` (NOVA, exclusions, plausibilité macros)."""
    return df.filter(expr_catalog_export_filters())


def download_data(force_download=False):
    # 1. Télécharger le Parquet
    print("Downloading data...")
    local_parquet = hf_hub_download(
        repo_id="openfoodfacts/product-database",
        repo_type="dataset",
        filename="food.parquet",
        local_dir="./data/",
        force_download=force_download,
    )
    print("Data downloaded.")
    useful_columns = [
        'additives_n',
        'additives_tags',
        'allergens_tags',
        'brands_tags',
        'brands',
        'categories',
        'categories_tags',
        'categories_properties',
        'ciqual_food_name_tags',
        'cities_tags',
        'code',
        'compared_to_category',
        'complete',
        'completeness',
        'data_quality_errors_tags',
        'data_quality_info_tags',
        'data_quality_warnings_tags',
        'environmental_score_data',
        'environmental_score_grade',
        'environmental_score_score',
        'environmental_score_tags',
        'emb_codes_tags',
        'emb_codes',
        'food_groups_tags',
        'generic_name',
        'ingredients_analysis_tags',
        'ingredients_from_palm_oil_n',
        'ingredients_n',
        'ingredients_original_tags',
        'ingredients_percent_analysis',
        'ingredients_tags',
        'ingredients_text',
        'ingredients_with_specified_percent_n',
        'ingredients_with_unspecified_percent_n',
        'ingredients_without_ciqual_codes_n',
        'ingredients_without_ciqual_codes',
        'ingredients',
        'known_ingredients_n',
        'labels_tags',
        'labels',
        'languages_tags',
        'last_updated_t',
        'manufacturing_places',
        'minerals_tags',
        'misc_tags',
        'new_additives_n',
        'no_nutrition_data',
        'nova_group',
        'nova_groups_tags',
        'nova_groups',
        'nucleotides_tags',
        'nutrient_levels_tags',
        'nutriments',
        'nutriscore_grade',
        'nutriscore_score',
        'nutrition_data_per',
        'origins_tags',
        'origins',
        'owner_fields',
        'owner',
        'packaging_tags',
        'packagings',
        'product_name',
        'product_quantity_unit',
        'product_quantity',
        'quantity',
        'rev',
        'serving_quantity',
        'serving_size',
        'vitamins_tags',
        'with_non_nutritive_sweeteners',
        'with_sweeteners',
    ]

    # 2. Créer le plan lazy
    cleaned_df = (
        pl.scan_parquet(local_parquet)
        .select(useful_columns + ["countries_tags", "obsolete"])
        # ne charger que la colonne nécessaire avant tout
        .filter(expr_france_not_obsolete())
        .drop("countries_tags", "obsolete")
        .explode("product_name").unnest("product_name").filter(expr_product_name_french()).select(
            # tous les autres champs sauf product_name
            *[c for c in useful_columns if c not in {"product_name"}],
            # on reprend "text" en l'appelant product_name
            pl.col("text").alias("product_name")
        )
    )

    return cleaned_df


def get_nutriments(data):
    nutriments = data.explode("nutriments").unnest("nutriments")

    macro_nutrients = [
        "energy-kcal",  # Énergie (kcal)
        "proteins",  # Protéines (g)
        "fat",  # Matières grasses totales (g)
        "carbohydrates",  # Glucides totaux (g)
        "fiber",  # Fibres alimentaires (g)
    ]

    # Pas utilisé car trop peu représenté dans le dataset
    # micro_nutriments = [
    #     "energy-kcal",
    #     "histidine",
    #     "isoleucine",
    #     "leucine",
    #     "methionine",
    #     "cystine",
    #     "phenylalanine",
    #     "tyrosine",
    #     "threonine",
    #     "tryptophan",
    #     "valine",
    #     "lysine",
    #     "fat",
    #     "fiber",
    #     "starch",
    #     "sugars",
    #     "glucose",
    #     "fructose",
    #     "lactose",
    #     "maltose",
    #     "galactose",
    #     "salt",
    #     "cholesterol",
    # ]

    main_information = [
        "code",
        "product_name",
        'categories',
        'nova_group',
        'labels'
    ]

    products_names_with_macro_nutriments = (
        nutriments
        .group_by(main_information)
        .agg([
            pl.col("100g")
            .filter(pl.col("name") == nutr)
            .first()
            .alias(nutr)
            for nutr in macro_nutrients
        ])
        .drop_nulls(macro_nutrients)
    )

    return products_names_with_macro_nutriments

# Regex native Polars : ne garder que les caractères latins, chiffres, ponctuation courante
_KEEP_LATIN_PATTERN = r"[^a-zA-ZÀ-ÿ0-9\s\-.,;:!?()\[\]{}]"

_ACCENT_MAP = [
    ("àáâãäå", "a"), ("èéêë", "e"), ("ìíîï", "i"), ("òóôõö", "o"),
    ("ùúûü", "u"), ("ýÿ", "y"), ("ñ", "n"), ("ç", "c"), ("æ", "ae"), ("œ", "oe"),
]


def _strip_accents_expr(col: str) -> pl.Expr:
    """Supprime les accents courants via chaîne de str.replace_all natifs (pas de map_elements).

    NOTE: Chains ~30 .str.replace_all() calls. Could be replaced with a single
    regex or mapping table if performance on larger datasets becomes an issue.
    """
    expr = pl.col(col)
    for chars, repl in _ACCENT_MAP:
        for c in chars:
            expr = expr.str.replace_all(c, repl, literal=True)
    return expr


def clean_categories(data: pl.LazyFrame) -> pl.LazyFrame:
    # Étape 1 : nettoyage + normalisation
    cleaned = data.with_columns(
        pl.col("categories")
        .fill_null("")
        .str.replace_all(_KEEP_LATIN_PATTERN, "")
        .str.to_lowercase()
        .str.replace_all(" ?, ?", ",")
        .str.replace_all(" ", "-")
        .alias("categories"),

        pl.col("labels")
        .fill_null("")
        .str.to_lowercase()
        .str.replace_all(" ?, ?", ",")
        .str.replace_all(" ", "-")
        .alias("labels"),
    )
    # Étape 2 : suppression des accents (expressions natives, sans map_elements)
    return cleaned.with_columns(
        _strip_accents_expr("categories").alias("categories"),
        _strip_accents_expr("labels").alias("labels"),
    )

def add_tags(data: pl.LazyFrame) -> pl.LazyFrame:
    return data.with_columns(
        # Tags régimes alimentaires :
        pl.col("labels").str.contains("halal").alias("halal"),
        pl.col("labels").str.contains("vegan").alias("vegan"),
        pl.col("labels").str.contains("bio").alias("bio"),
        pl.col("labels").str.contains("vegetarian").alias("vegetarian"),
        pl.col("labels").str.contains("gluten-free|sans-gluten|gluten-free|no-gluten").alias("gluten_free"),
        pl.col("labels").str.contains("koscher|kascher|casher").alias("kascher"),
        pl.col("labels").str.contains("sans-huile-de-palme|no-palm-oil").alias("no_palm_oil"),

        # Tags catégories nourritures
        pl.col("categories").str.contains("viandes|meat").alias("meat"),
        pl.col("categories").str.contains("pates|pasta").alias("pasta"),
        pl.col("categories").str.contains("boissons,|drinks").alias("drinks"),
        pl.col("categories").str.contains("produits-laitiers|lait|dairy").alias("lait"),
        pl.col("categories").str.contains("produits-de-la-mer|poisson|fish").alias("fish"),
        pl.col("categories").str.contains("snacks|chips,").alias("snacks"),
        pl.col("categories").str.contains("desserts|cakes").alias("desserts"),
        pl.col("categories").str.contains("condiments|sauce|epices").alias("condiments"),
        pl.col("categories").str.contains("plats-prepares").alias("plats_prepares"),
        pl.col("categories").str.contains("cereales-en-grains|cereales-et-pommes-de-terrre|feculents|pates").alias("feculents"),
        pl.col("categories").str.contains("pain|bread").alias("breads"),
    )


def process_batched(parquet_path, batch_size=50):
    """Process the parquet file in batches of row groups to stay within memory."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    n_groups = pf.metadata.num_row_groups
    needed_cols = [
        "code", "product_name", "categories", "nova_group",
        "labels", "nutriments", "countries_tags", "obsolete",
    ]
    macro_nutrients = ["energy-kcal", "proteins", "fat", "carbohydrates", "fiber"]
    main_info = ["code", "product_name", "categories", "nova_group", "labels"]

    chunks = []
    for start in range(0, n_groups, batch_size):
        end = min(start + batch_size, n_groups)
        rg_indices = list(range(start, end))
        table = pf.read_row_groups(rg_indices, columns=needed_cols)
        batch = pl.from_arrow(table)

        if len(batch) == 0:
            continue

        batch = (
            batch.lazy()
            .filter(expr_france_not_obsolete())
            .drop("countries_tags", "obsolete")
            .explode("product_name").unnest("product_name")
            .filter(expr_product_name_french())
            .select(
                *[c for c in main_info + ["nutriments"] if c not in {"product_name"}],
                pl.col("text").alias("product_name"),
            )
            .explode("nutriments").unnest("nutriments")
            .group_by(main_info)
            .agg([
                pl.col("100g")
                .filter(pl.col("name") == nutr)
                .first()
                .alias(nutr)
                for nutr in macro_nutrients
            ])
            .drop_nulls(macro_nutrients)
            .collect()
        )

        if len(batch) > 0:
            chunks.append(batch)

        print(f"  row groups {start}-{end-1}/{n_groups}  →  {len(batch)} products", flush=True)

    return pl.concat(chunks)


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    print("Ensuring parquet is downloaded…")
    local_parquet = hf_hub_download(
        repo_id="openfoodfacts/product-database",
        repo_type="dataset",
        filename="food.parquet",
        local_dir="./data/",
        force_download=False,
    )

    print("Processing in batches…")
    df = process_batched(local_parquet, batch_size=50)
    print(f"Products after nutriment extraction: {len(df)}")

    df = filter_products_catalog(add_tags(clean_categories(df.lazy())))
    df = df.collect()

    print(f"Nombre de lignes (après filtres catalogue) : {len(df)}")

    # Libellés type « Fraises » (cohérent avec kojin_common.load_products)
    from kojin_common import normalize_catalog_product_names

    df = normalize_catalog_product_names(df)
    df.write_csv("./data/products_names_with_macro_nutriments.csv")
    print("Done.")
