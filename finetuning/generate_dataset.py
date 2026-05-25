"""Generate training and evaluation datasets for NL → SQL fine-tuning.

Produces JSONL files compatible with Amazon Bedrock Custom Model fine-tuning
(Converse format: system / user / assistant messages).

Usage:
    python -m finetuning.generate_dataset [--output-dir ./finetuning/data] [--train-ratio 0.85]

The script generates diverse question/SQL pairs covering:
- Filtering by macro nutrients (kcal, proteins, fat, carbohydrates, fiber)
- Filtering by boolean tags (vegan, halal, meat, fish, lait, bio, gluten_free…)
- Sorting & ranking (top-N)
- Aggregations (AVG, COUNT, SUM, MIN, MAX)
- Combined conditions
- ILIKE text search on product_name / categories
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

# ─── Schema (matches the DuckDB table built from the processed CSV) ──────────

TABLE_NAME = "products"

SCHEMA = """\
  - code (VARCHAR)
  - product_name (VARCHAR)
  - categories (VARCHAR)
  - nova_group (BIGINT)
  - labels (VARCHAR)
  - energy-kcal (DOUBLE)
  - proteins (DOUBLE)
  - fat (DOUBLE)
  - carbohydrates (DOUBLE)
  - fiber (DOUBLE)
  - halal (BOOLEAN)
  - vegan (BOOLEAN)
  - bio (BOOLEAN)
  - vegetarian (BOOLEAN)
  - gluten_free (BOOLEAN)
  - kascher (BOOLEAN)
  - no_palm_oil (BOOLEAN)
  - meat (BOOLEAN)
  - pasta (BOOLEAN)
  - drinks (BOOLEAN)
  - lait (BOOLEAN)
  - fish (BOOLEAN)
  - snacks (BOOLEAN)
  - desserts (BOOLEAN)
  - condiments (BOOLEAN)
  - plats_prepares (BOOLEAN)
  - feculents (BOOLEAN)
  - breads (BOOLEAN)"""

SYSTEM_PROMPT = f"""Tu es un assistant qui traduit une question en langage naturel \
en UNE seule requête SQL DuckDB valide, exécutable contre la table `{TABLE_NAME}`.

Schéma de la table `{TABLE_NAME}` :
{SCHEMA}

Règles strictes :
- Réponds UNIQUEMENT avec la requête SQL — pas de markdown, pas de ```sql```, pas \
d'explication, pas de texte avant ou après.
- Utilise toujours `{TABLE_NAME}` comme nom de table.
- Limite à 200 lignes par défaut (`LIMIT 200`) sauf si la question demande \
explicitement un agrégat ou un autre nombre.
- Préfère un SELECT explicite des colonnes pertinentes plutôt que `SELECT *`.
- Pour les colonnes booléennes (vegan, halal, vegetarian, bio, gluten_free, \
kascher, meat, fish, lait, no_palm_oil) compare avec `TRUE` ou `FALSE`.
- Pour des recherches sur des chaînes (product_name, categories) utilise \
`ILIKE '%motif%'`.
- La colonne des calories s'appelle `"energy-kcal"` (avec guillemets doubles \
obligatoires car le tiret est un opérateur en SQL) : écris toujours \
`"energy-kcal"` et jamais `energy-kcal` ni `energy_kcal`.
- Si la question est ambiguë, choisis l'interprétation la plus utile pour un \
nutritionniste."""

# ─── Question/SQL templates ───────────────────────────────────────────────────


def _generate_pairs() -> list[dict]:
    """Generate diverse (question, sql) pairs programmatically."""
    pairs: list[dict] = []

    # --- 1. Simple macro filters ---
    macro_questions = [
        ("Quels produits ont plus de {v} g de protéines pour 100 g ?",
         "SELECT product_name, proteins, \"energy-kcal\" FROM products WHERE proteins > {v} ORDER BY proteins DESC LIMIT 200"),
        ("Liste les aliments avec moins de {v} kcal pour 100 g.",
         "SELECT product_name, \"energy-kcal\", proteins, fat, carbohydrates FROM products WHERE \"energy-kcal\" < {v} ORDER BY \"energy-kcal\" ASC LIMIT 200"),
        ("Quels aliments contiennent plus de {v} g de fibres ?",
         "SELECT product_name, fiber, \"energy-kcal\" FROM products WHERE fiber > {v} ORDER BY fiber DESC LIMIT 200"),
        ("Produits avec moins de {v} g de lipides pour 100 g.",
         "SELECT product_name, fat, \"energy-kcal\", proteins FROM products WHERE fat < {v} ORDER BY fat ASC LIMIT 200"),
        ("Aliments avec plus de {v} g de glucides.",
         "SELECT product_name, carbohydrates, \"energy-kcal\" FROM products WHERE carbohydrates > {v} ORDER BY carbohydrates DESC LIMIT 200"),
    ]
    for q_tpl, s_tpl in macro_questions:
        for v in [5, 10, 15, 20, 25, 30, 40, 50]:
            pairs.append({"question": q_tpl.format(v=v), "sql": s_tpl.format(v=v)})

    # --- 2. Boolean tag filters ---
    tag_questions = [
        ("Quels produits vegan ont plus de {v} g de protéines pour 100 g ?",
         "SELECT product_name, proteins, \"energy-kcal\" FROM products WHERE vegan = TRUE AND proteins > {v} ORDER BY proteins DESC LIMIT 200"),
        ("Liste les produits halal avec moins de {v} kcal.",
         "SELECT product_name, \"energy-kcal\", proteins FROM products WHERE halal = TRUE AND \"energy-kcal\" < {v} ORDER BY \"energy-kcal\" ASC LIMIT 200"),
        ("Produits bio avec plus de {v} g de fibres.",
         "SELECT product_name, fiber, \"energy-kcal\" FROM products WHERE bio = TRUE AND fiber > {v} ORDER BY fiber DESC LIMIT 200"),
        ("Aliments sans gluten avec plus de {v} g de protéines.",
         "SELECT product_name, proteins, \"energy-kcal\", fat FROM products WHERE gluten_free = TRUE AND proteins > {v} ORDER BY proteins DESC LIMIT 200"),
        ("Produits végétariens riches en protéines (plus de {v} g).",
         "SELECT product_name, proteins, \"energy-kcal\" FROM products WHERE vegetarian = TRUE AND proteins > {v} ORDER BY proteins DESC LIMIT 200"),
        ("Produits casher avec moins de {v} g de lipides.",
         "SELECT product_name, fat, proteins, \"energy-kcal\" FROM products WHERE kascher = TRUE AND fat < {v} ORDER BY fat ASC LIMIT 200"),
    ]
    for q_tpl, s_tpl in tag_questions:
        for v in [10, 15, 20, 25, 30]:
            pairs.append({"question": q_tpl.format(v=v), "sql": s_tpl.format(v=v)})

    # --- 3. Top-N rankings ---
    top_n_questions = [
        ("Top {n} des aliments les plus riches en protéines.",
         "SELECT product_name, proteins, \"energy-kcal\" FROM products ORDER BY proteins DESC LIMIT {n}"),
        ("Les {n} aliments avec le plus de fibres.",
         "SELECT product_name, fiber, \"energy-kcal\" FROM products ORDER BY fiber DESC LIMIT {n}"),
        ("Top {n} des produits les moins caloriques.",
         "SELECT product_name, \"energy-kcal\", proteins, fat, carbohydrates FROM products ORDER BY \"energy-kcal\" ASC LIMIT {n}"),
        ("Les {n} aliments les plus gras.",
         "SELECT product_name, fat, \"energy-kcal\" FROM products ORDER BY fat DESC LIMIT {n}"),
        ("Top {n} produits avec le meilleur ratio protéines/kcal.",
         "SELECT product_name, proteins, \"energy-kcal\", ROUND(proteins / NULLIF(\"energy-kcal\", 0) * 100, 2) AS ratio_prot_kcal FROM products WHERE \"energy-kcal\" > 0 ORDER BY ratio_prot_kcal DESC LIMIT {n}"),
    ]
    for q_tpl, s_tpl in top_n_questions:
        for n in [5, 10, 15, 20, 30, 50]:
            pairs.append({"question": q_tpl.format(n=n), "sql": s_tpl.format(n=n)})

    # --- 4. Aggregations ---
    agg_questions = [
        ("Combien de produits vegan y a-t-il dans la base ?",
         "SELECT COUNT(*) AS nb_vegan FROM products WHERE vegan = TRUE"),
        ("Quelle est la moyenne de protéines des produits à base de viande ?",
         "SELECT ROUND(AVG(proteins), 1) AS avg_proteins FROM products WHERE meat = TRUE"),
        ("Quel est le produit avec le plus de protéines ?",
         "SELECT product_name, proteins FROM products ORDER BY proteins DESC LIMIT 1"),
        ("Combien de produits contiennent plus de 20 g de protéines ?",
         "SELECT COUNT(*) AS nb_produits FROM products WHERE proteins > 20"),
        ("Quelle est la moyenne de kcal pour les produits de poisson ?",
         "SELECT ROUND(AVG(\"energy-kcal\"), 1) AS avg_kcal FROM products WHERE fish = TRUE"),
        ("Combien de produits halal et bio existent ?",
         "SELECT COUNT(*) AS nb FROM products WHERE halal = TRUE AND bio = TRUE"),
        ("Nombre de produits par catégorie lait vs viande.",
         "SELECT 'lait' AS categorie, COUNT(*) AS nb FROM products WHERE lait = TRUE UNION ALL SELECT 'meat', COUNT(*) FROM products WHERE meat = TRUE"),
        ("Moyenne de fibres pour les produits vegan.",
         "SELECT ROUND(AVG(fiber), 1) AS avg_fiber FROM products WHERE vegan = TRUE"),
        ("Produit le plus calorique.",
         "SELECT product_name, \"energy-kcal\" FROM products ORDER BY \"energy-kcal\" DESC LIMIT 1"),
        ("Combien de produits sans gluten ?",
         "SELECT COUNT(*) AS nb_gluten_free FROM products WHERE gluten_free = TRUE"),
    ]
    pairs.extend({"question": q, "sql": s} for q, s in agg_questions)

    # --- 5. Combined conditions ---
    combined = [
        ("Produits vegan avec plus de 15 g de protéines et moins de 200 kcal.",
         "SELECT product_name, proteins, \"energy-kcal\", fat FROM products WHERE vegan = TRUE AND proteins > 15 AND \"energy-kcal\" < 200 ORDER BY proteins DESC LIMIT 200"),
        ("Aliments halal riches en protéines (>20g) et pauvres en lipides (<10g).",
         "SELECT product_name, proteins, fat, \"energy-kcal\" FROM products WHERE halal = TRUE AND proteins > 20 AND fat < 10 ORDER BY proteins DESC LIMIT 200"),
        ("Produits bio, végétariens, avec plus de 5 g de fibres.",
         "SELECT product_name, fiber, proteins, \"energy-kcal\" FROM products WHERE bio = TRUE AND vegetarian = TRUE AND fiber > 5 ORDER BY fiber DESC LIMIT 200"),
        ("Aliments à base de poisson avec moins de 150 kcal et plus de 15g de protéines.",
         "SELECT product_name, \"energy-kcal\", proteins, fat FROM products WHERE fish = TRUE AND \"energy-kcal\" < 150 AND proteins > 15 ORDER BY proteins DESC LIMIT 200"),
        ("Féculents avec plus de 10 g de fibres.",
         "SELECT product_name, fiber, carbohydrates, \"energy-kcal\" FROM products WHERE feculents = TRUE AND fiber > 10 ORDER BY fiber DESC LIMIT 200"),
        ("Pains avec moins de 250 kcal.",
         "SELECT product_name, \"energy-kcal\", carbohydrates, fiber FROM products WHERE breads = TRUE AND \"energy-kcal\" < 250 ORDER BY \"energy-kcal\" ASC LIMIT 200"),
        ("Produits laitiers riches en protéines (>10g) et pauvres en sucre (<5g glucides).",
         "SELECT product_name, proteins, carbohydrates, \"energy-kcal\" FROM products WHERE lait = TRUE AND proteins > 10 AND carbohydrates < 5 ORDER BY proteins DESC LIMIT 200"),
        ("Viandes avec le meilleur ratio protéines/lipides.",
         "SELECT product_name, proteins, fat, ROUND(proteins / NULLIF(fat, 0), 2) AS ratio_prot_fat FROM products WHERE meat = TRUE AND fat > 0 ORDER BY ratio_prot_fat DESC LIMIT 50"),
    ]
    pairs.extend({"question": q, "sql": s} for q, s in combined)

    # --- 6. Text search (ILIKE) ---
    text_search = [
        ("Trouve les produits contenant le mot 'poulet' dans leur nom.",
         "SELECT product_name, proteins, \"energy-kcal\" FROM products WHERE product_name ILIKE '%poulet%' LIMIT 200"),
        ("Produits dont le nom contient 'tofu'.",
         "SELECT product_name, proteins, fat, \"energy-kcal\" FROM products WHERE product_name ILIKE '%tofu%' LIMIT 200"),
        ("Cherche les produits avec 'lentille' dans le nom.",
         "SELECT product_name, proteins, fiber, \"energy-kcal\" FROM products WHERE product_name ILIKE '%lentille%' LIMIT 200"),
        ("Aliments contenant 'quinoa'.",
         "SELECT product_name, proteins, fiber, carbohydrates, \"energy-kcal\" FROM products WHERE product_name ILIKE '%quinoa%' LIMIT 200"),
        ("Produits avec 'saumon' dans le nom.",
         "SELECT product_name, proteins, fat, \"energy-kcal\" FROM products WHERE product_name ILIKE '%saumon%' LIMIT 200"),
        ("Trouve les produits avec 'avoine' dans la catégorie.",
         "SELECT product_name, categories, fiber, carbohydrates FROM products WHERE categories ILIKE '%avoine%' LIMIT 200"),
        ("Aliments dont le nom contient 'riz'.",
         "SELECT product_name, carbohydrates, proteins, \"energy-kcal\" FROM products WHERE product_name ILIKE '%riz%' LIMIT 200"),
        ("Produits contenant 'oeuf' ou 'œuf' dans le nom.",
         "SELECT product_name, proteins, fat, \"energy-kcal\" FROM products WHERE product_name ILIKE '%oeuf%' OR product_name ILIKE '%œuf%' LIMIT 200"),
    ]
    pairs.extend({"question": q, "sql": s} for q, s in text_search)

    # --- 7. Multi-criteria with sorting ---
    multi_sort = [
        ("Liste les 20 produits vegan les plus protéinés avec leur ratio protéines/kcal.",
         "SELECT product_name, proteins, \"energy-kcal\", ROUND(proteins / NULLIF(\"energy-kcal\", 0) * 100, 2) AS ratio FROM products WHERE vegan = TRUE ORDER BY proteins DESC LIMIT 20"),
        ("Les 30 aliments les moins gras parmi ceux qui ont plus de 15g de protéines.",
         "SELECT product_name, fat, proteins, \"energy-kcal\" FROM products WHERE proteins > 15 ORDER BY fat ASC LIMIT 30"),
        ("Top 10 des produits halal avec le plus de fibres et moins de 300 kcal.",
         "SELECT product_name, fiber, \"energy-kcal\" FROM products WHERE halal = TRUE AND \"energy-kcal\" < 300 ORDER BY fiber DESC LIMIT 10"),
        ("Produits bio triés par ratio fibres/glucides décroissant, top 25.",
         "SELECT product_name, fiber, carbohydrates, ROUND(fiber / NULLIF(carbohydrates, 0), 3) AS ratio_fiber_carbs FROM products WHERE bio = TRUE AND carbohydrates > 0 ORDER BY ratio_fiber_carbs DESC LIMIT 25"),
    ]
    pairs.extend({"question": q, "sql": s} for q, s in multi_sort)

    # --- 8. NOVA group queries ---
    nova_queries = [
        ("Quels produits ont un nova_group de 1 ?",
         "SELECT product_name, nova_group, \"energy-kcal\", proteins FROM products WHERE nova_group = 1 LIMIT 200"),
        ("Nombre de produits par nova_group.",
         "SELECT nova_group, COUNT(*) AS nb FROM products GROUP BY nova_group ORDER BY nova_group"),
        ("Produits non transformés (nova 1) avec plus de 20g de protéines.",
         "SELECT product_name, proteins, \"energy-kcal\", nova_group FROM products WHERE nova_group = 1 AND proteins > 20 ORDER BY proteins DESC LIMIT 200"),
    ]
    pairs.extend({"question": q, "sql": s} for q, s in nova_queries)

    # --- 9. Natural language variations ---
    natural_variations = [
        ("Quels sont les aliments qui ont le plus de protéines et le moins de gras ?",
         "SELECT product_name, proteins, fat, \"energy-kcal\" FROM products WHERE proteins > 15 AND fat < 5 ORDER BY proteins DESC LIMIT 200"),
        ("Je cherche des aliments riches en fibres pour mon régime.",
         "SELECT product_name, fiber, \"energy-kcal\", carbohydrates FROM products WHERE fiber > 8 ORDER BY fiber DESC LIMIT 200"),
        ("Quels sont les meilleurs aliments pour une sèche musculaire ?",
         "SELECT product_name, proteins, \"energy-kcal\", fat FROM products WHERE proteins > 20 AND fat < 10 AND \"energy-kcal\" < 200 ORDER BY proteins DESC LIMIT 50"),
        ("Donne-moi des aliments adaptés à un régime keto (beaucoup de gras, peu de glucides).",
         "SELECT product_name, fat, carbohydrates, \"energy-kcal\", proteins FROM products WHERE fat > 30 AND carbohydrates < 10 ORDER BY fat DESC LIMIT 200"),
        ("Quels produits sont bons pour la prise de masse ?",
         "SELECT product_name, \"energy-kcal\", proteins, carbohydrates FROM products WHERE \"energy-kcal\" > 300 AND proteins > 15 ORDER BY \"energy-kcal\" DESC LIMIT 200"),
        ("Aliments idéaux pour le petit déjeuner d'un sportif.",
         "SELECT product_name, \"energy-kcal\", proteins, carbohydrates, fiber FROM products WHERE carbohydrates > 20 AND proteins > 10 AND fiber > 3 ORDER BY \"energy-kcal\" DESC LIMIT 50"),
        ("Quels produits vegan peuvent remplacer la viande en termes de protéines ?",
         "SELECT product_name, proteins, \"energy-kcal\", fat FROM products WHERE vegan = TRUE AND proteins > 18 ORDER BY proteins DESC LIMIT 50"),
        ("Liste moi des snacks sains (faible kcal, bonnes fibres).",
         "SELECT product_name, \"energy-kcal\", fiber, proteins FROM products WHERE \"energy-kcal\" < 150 AND fiber > 5 ORDER BY fiber DESC LIMIT 200"),
    ]
    pairs.extend({"question": q, "sql": s} for q, s in natural_variations)

    return pairs


def _generate_eval_pairs() -> list[dict]:
    """Generate 97 evaluation-only pairs, disjoint from _generate_pairs().

    Covers new thresholds, new SQL patterns (BETWEEN, HAVING, subqueries,
    CASE, COALESCE, multi-tag combos) and new food terms not used in training.
    """
    pairs: list[dict] = []

    # --- E1. Macro filters with thresholds not used in training ---
    macro_eval = [
        ("Produits avec exactement entre 12 et 18 g de protéines.",
         "SELECT product_name, proteins, \"energy-kcal\" FROM products WHERE proteins BETWEEN 12 AND 18 ORDER BY proteins DESC LIMIT 200"),
        ("Aliments entre 100 et 150 kcal pour 100 g.",
         "SELECT product_name, \"energy-kcal\", proteins, fat FROM products WHERE \"energy-kcal\" BETWEEN 100 AND 150 ORDER BY \"energy-kcal\" ASC LIMIT 200"),
        ("Produits avec entre 3 et 8 g de fibres.",
         "SELECT product_name, fiber, \"energy-kcal\" FROM products WHERE fiber BETWEEN 3 AND 8 ORDER BY fiber DESC LIMIT 200"),
        ("Aliments contenant entre 5 et 12 g de lipides.",
         "SELECT product_name, fat, proteins, \"energy-kcal\" FROM products WHERE fat BETWEEN 5 AND 12 ORDER BY fat ASC LIMIT 200"),
        ("Produits entre 200 et 400 kcal avec plus de 8 g de protéines.",
         "SELECT product_name, \"energy-kcal\", proteins, fat FROM products WHERE \"energy-kcal\" BETWEEN 200 AND 400 AND proteins > 8 ORDER BY proteins DESC LIMIT 200"),
        ("Aliments avec moins de 2 g de lipides pour 100 g.",
         "SELECT product_name, fat, proteins, \"energy-kcal\" FROM products WHERE fat < 2 ORDER BY fat ASC LIMIT 200"),
        ("Produits avec au moins 35 g de glucides.",
         "SELECT product_name, carbohydrates, \"energy-kcal\", fiber FROM products WHERE carbohydrates >= 35 ORDER BY carbohydrates DESC LIMIT 200"),
        ("Aliments avec plus de 500 kcal pour 100 g.",
         "SELECT product_name, \"energy-kcal\", fat, carbohydrates FROM products WHERE \"energy-kcal\" > 500 ORDER BY \"energy-kcal\" DESC LIMIT 200"),
        ("Produits avec moins de 1 g de fibres.",
         "SELECT product_name, fiber, carbohydrates, \"energy-kcal\" FROM products WHERE fiber < 1 ORDER BY fiber ASC LIMIT 200"),
        ("Aliments avec plus de 45 g de protéines pour 100 g.",
         "SELECT product_name, proteins, \"energy-kcal\", fat FROM products WHERE proteins > 45 ORDER BY proteins DESC LIMIT 200"),
    ]
    pairs.extend({"question": q, "sql": s} for q, s in macro_eval)

    # --- E2. New boolean tag combinations ---
    tag_eval = [
        ("Produits kascher et vegan avec plus de 10 g de protéines.",
         "SELECT product_name, proteins, \"energy-kcal\", fat FROM products WHERE kascher = TRUE AND vegan = TRUE AND proteins > 10 ORDER BY proteins DESC LIMIT 200"),
        ("Aliments sans gluten et sans huile de palme.",
         "SELECT product_name, fat, proteins, \"energy-kcal\" FROM products WHERE gluten_free = TRUE AND no_palm_oil = TRUE ORDER BY proteins DESC LIMIT 200"),
        ("Produits bio et kascher.",
         "SELECT product_name, proteins, \"energy-kcal\", fiber FROM products WHERE bio = TRUE AND kascher = TRUE LIMIT 200"),
        ("Snacks vegan avec moins de 200 kcal.",
         "SELECT product_name, \"energy-kcal\", proteins, fiber FROM products WHERE snacks = TRUE AND vegan = TRUE AND \"energy-kcal\" < 200 ORDER BY \"energy-kcal\" ASC LIMIT 200"),
        ("Desserts sans gluten avec moins de 300 kcal.",
         "SELECT product_name, \"energy-kcal\", fat, carbohydrates FROM products WHERE desserts = TRUE AND gluten_free = TRUE AND \"energy-kcal\" < 300 ORDER BY \"energy-kcal\" ASC LIMIT 200"),
        ("Condiments végétariens avec moins de 5 g de lipides.",
         "SELECT product_name, fat, proteins, \"energy-kcal\" FROM products WHERE condiments = TRUE AND vegetarian = TRUE AND fat < 5 ORDER BY fat ASC LIMIT 200"),
        ("Boissons vegan avec moins de 50 kcal.",
         "SELECT product_name, \"energy-kcal\", carbohydrates FROM products WHERE drinks = TRUE AND vegan = TRUE AND \"energy-kcal\" < 50 ORDER BY \"energy-kcal\" ASC LIMIT 200"),
        ("Pâtes bio avec plus de 12 g de protéines.",
         "SELECT product_name, proteins, carbohydrates, \"energy-kcal\" FROM products WHERE pasta = TRUE AND bio = TRUE AND proteins > 12 ORDER BY proteins DESC LIMIT 200"),
        ("Poisson halal avec moins de 10 g de lipides.",
         "SELECT product_name, fat, proteins, \"energy-kcal\" FROM products WHERE fish = TRUE AND halal = TRUE AND fat < 10 ORDER BY fat ASC LIMIT 200"),
        ("Produits végétariens sans huile de palme et avec plus de 5 g de fibres.",
         "SELECT product_name, fiber, proteins, \"energy-kcal\" FROM products WHERE vegetarian = TRUE AND no_palm_oil = TRUE AND fiber > 5 ORDER BY fiber DESC LIMIT 200"),
        ("Féculents végétariens avec plus de 15 g de protéines.",
         "SELECT product_name, proteins, carbohydrates, fiber FROM products WHERE feculents = TRUE AND vegetarian = TRUE AND proteins > 15 ORDER BY proteins DESC LIMIT 200"),
        ("Viandes bio avec moins de 15 g de lipides.",
         "SELECT product_name, fat, proteins, \"energy-kcal\" FROM products WHERE meat = TRUE AND bio = TRUE AND fat < 15 ORDER BY fat ASC LIMIT 200"),
    ]
    pairs.extend({"question": q, "sql": s} for q, s in tag_eval)

    # --- E3. Aggregations not in training ---
    agg_eval = [
        ("Quelle est la moyenne de kcal pour les produits vegan ?",
         "SELECT ROUND(AVG(\"energy-kcal\"), 1) AS avg_kcal_vegan FROM products WHERE vegan = TRUE"),
        ("Quel est le maximum de protéines parmi tous les produits ?",
         "SELECT MAX(proteins) AS max_proteins FROM products"),
        ("Quelle est la somme des kcal de tous les produits bio ?",
         "SELECT ROUND(SUM(\"energy-kcal\"), 0) AS total_kcal_bio FROM products WHERE bio = TRUE"),
        ("Combien de produits sont tagués vegetarian ?",
         "SELECT COUNT(*) AS nb_vegetarian FROM products WHERE vegetarian = TRUE"),
        ("Quelle est la médiane de protéines pour les produits de poisson ?",
         "SELECT ROUND(MEDIAN(proteins), 1) AS median_proteins FROM products WHERE fish = TRUE"),
        ("Quelle est la moyenne de fibres par nova_group ?",
         "SELECT nova_group, ROUND(AVG(fiber), 2) AS avg_fiber FROM products GROUP BY nova_group ORDER BY nova_group"),
        ("Quel produit a le minimum de lipides parmi les viandes ?",
         "SELECT product_name, fat FROM products WHERE meat = TRUE ORDER BY fat ASC LIMIT 1"),
        ("Combien de produits ont des données de protéines renseignées ?",
         "SELECT COUNT(*) AS nb_with_proteins FROM products WHERE proteins IS NOT NULL"),
        ("Quelle est la moyenne de glucides pour les pains ?",
         "SELECT ROUND(AVG(carbohydrates), 1) AS avg_carbs FROM products WHERE breads = TRUE"),
        ("Top 5 des catégories les plus représentées dans la base.",
         "SELECT categories, COUNT(*) AS nb FROM products WHERE categories IS NOT NULL GROUP BY categories ORDER BY nb DESC LIMIT 5"),
        ("Quel est le ratio moyen protéines/kcal pour les produits vegan ?",
         "SELECT ROUND(AVG(proteins / NULLIF(\"energy-kcal\", 0) * 100), 3) AS avg_ratio FROM products WHERE vegan = TRUE AND \"energy-kcal\" > 0"),
        ("Combien de produits no_palm_oil existent dans la base ?",
         "SELECT COUNT(*) AS nb_no_palm_oil FROM products WHERE no_palm_oil = TRUE"),
    ]
    pairs.extend({"question": q, "sql": s} for q, s in agg_eval)

    # --- E4. New text search terms ---
    text_eval = [
        ("Produits contenant 'sardine' dans le nom.",
         "SELECT product_name, proteins, fat, \"energy-kcal\" FROM products WHERE product_name ILIKE '%sardine%' LIMIT 200"),
        ("Aliments avec 'pois chiche' dans le nom.",
         "SELECT product_name, proteins, fiber, carbohydrates, \"energy-kcal\" FROM products WHERE product_name ILIKE '%pois chiche%' LIMIT 200"),
        ("Produits dont le nom contient 'thon'.",
         "SELECT product_name, proteins, fat, \"energy-kcal\" FROM products WHERE product_name ILIKE '%thon%' LIMIT 200"),
        ("Cherche les produits avec 'edamame' dans le nom ou les catégories.",
         "SELECT product_name, proteins, fiber, \"energy-kcal\" FROM products WHERE product_name ILIKE '%edamame%' OR categories ILIKE '%edamame%' LIMIT 200"),
        ("Aliments contenant 'potiron' ou 'courge'.",
         "SELECT product_name, fiber, carbohydrates, \"energy-kcal\" FROM products WHERE product_name ILIKE '%potiron%' OR product_name ILIKE '%courge%' LIMIT 200"),
        ("Produits avec 'brocoli' dans le nom.",
         "SELECT product_name, fiber, proteins, \"energy-kcal\" FROM products WHERE product_name ILIKE '%brocoli%' LIMIT 200"),
        ("Aliments dont le nom contient 'amande'.",
         "SELECT product_name, fat, proteins, \"energy-kcal\" FROM products WHERE product_name ILIKE '%amande%' LIMIT 200"),
        ("Trouve les produits avec 'épinard' dans le nom ou les catégories.",
         "SELECT product_name, proteins, fiber, \"energy-kcal\" FROM products WHERE product_name ILIKE '%épinard%' OR categories ILIKE '%épinard%' LIMIT 200"),
        ("Produits contenant 'haricot' dans le nom.",
         "SELECT product_name, proteins, fiber, carbohydrates FROM products WHERE product_name ILIKE '%haricot%' LIMIT 200"),
        ("Aliments avec 'noix de cajou' dans le nom.",
         "SELECT product_name, fat, proteins, \"energy-kcal\" FROM products WHERE product_name ILIKE '%noix de cajou%' LIMIT 200"),
    ]
    pairs.extend({"question": q, "sql": s} for q, s in text_eval)

    # --- E5. New top-N rankings with tag filter ---
    ranking_eval = [
        ("Top 7 des produits végétariens les plus riches en fibres.",
         "SELECT product_name, fiber, \"energy-kcal\" FROM products WHERE vegetarian = TRUE ORDER BY fiber DESC LIMIT 7"),
        ("Les 12 aliments bio les moins caloriques.",
         "SELECT product_name, \"energy-kcal\", proteins, fiber FROM products WHERE bio = TRUE ORDER BY \"energy-kcal\" ASC LIMIT 12"),
        ("Top 8 des poissons les plus riches en protéines.",
         "SELECT product_name, proteins, fat, \"energy-kcal\" FROM products WHERE fish = TRUE ORDER BY proteins DESC LIMIT 8"),
        ("Les 15 pâtes avec le plus de protéines.",
         "SELECT product_name, proteins, carbohydrates, \"energy-kcal\" FROM products WHERE pasta = TRUE ORDER BY proteins DESC LIMIT 15"),
        ("Top 5 des desserts les moins gras.",
         "SELECT product_name, fat, \"energy-kcal\", carbohydrates FROM products WHERE desserts = TRUE ORDER BY fat ASC LIMIT 5"),
        ("Les 10 produits sans gluten les plus riches en fibres.",
         "SELECT product_name, fiber, proteins, \"energy-kcal\" FROM products WHERE gluten_free = TRUE ORDER BY fiber DESC LIMIT 10"),
        ("Top 20 des féculents les moins caloriques.",
         "SELECT product_name, \"energy-kcal\", carbohydrates, fiber FROM products WHERE feculents = TRUE ORDER BY \"energy-kcal\" ASC LIMIT 20"),
        ("Les 6 meilleurs snacks en termes de ratio fibres/kcal.",
         "SELECT product_name, fiber, \"energy-kcal\", ROUND(fiber / NULLIF(\"energy-kcal\", 0) * 100, 3) AS ratio FROM products WHERE snacks = TRUE AND \"energy-kcal\" > 0 ORDER BY ratio DESC LIMIT 6"),
        ("Top 10 des viandes halal les moins grasses.",
         "SELECT product_name, fat, proteins, \"energy-kcal\" FROM products WHERE meat = TRUE AND halal = TRUE ORDER BY fat ASC LIMIT 10"),
        ("Les 3 produits kascher les plus riches en protéines.",
         "SELECT product_name, proteins, \"energy-kcal\" FROM products WHERE kascher = TRUE ORDER BY proteins DESC LIMIT 3"),
    ]
    pairs.extend({"question": q, "sql": s} for q, s in ranking_eval)

    # --- E6. CASE / COALESCE / computed columns ---
    computed_eval = [
        ("Classe les produits en 'riche en protéines' (>20g) ou 'pauvre en protéines' (<=20g).",
         "SELECT product_name, proteins, CASE WHEN proteins > 20 THEN 'riche' ELSE 'pauvre' END AS categorie_proteins FROM products ORDER BY proteins DESC LIMIT 200"),
        ("Remplace les valeurs nulles de fibres par 0 et trie par fibres décroissant.",
         "SELECT product_name, COALESCE(fiber, 0) AS fiber, \"energy-kcal\" FROM products ORDER BY COALESCE(fiber, 0) DESC LIMIT 200"),
        ("Produits avec leur densité nutritionnelle (protéines + fibres) / kcal.",
         "SELECT product_name, proteins, fiber, \"energy-kcal\", ROUND((proteins + COALESCE(fiber, 0)) / NULLIF(\"energy-kcal\", 0) * 100, 3) AS densite FROM products WHERE \"energy-kcal\" > 0 ORDER BY densite DESC LIMIT 50"),
        ("Classe chaque produit selon son nova_group : 1='non transformé', 2='peu transformé', 3='transformé', 4='ultra-transformé'.",
         "SELECT product_name, nova_group, CASE nova_group WHEN 1 THEN 'non transformé' WHEN 2 THEN 'peu transformé' WHEN 3 THEN 'transformé' WHEN 4 THEN 'ultra-transformé' ELSE 'inconnu' END AS label_nova FROM products ORDER BY nova_group LIMIT 200"),
        ("Nombre de kcal apportées par les protéines vs lipides pour chaque produit, top 20 par apport protéique.",
         "SELECT product_name, ROUND(proteins * 4, 1) AS kcal_proteins, ROUND(fat * 9, 1) AS kcal_fat, \"energy-kcal\" FROM products ORDER BY proteins DESC LIMIT 20"),
    ]
    pairs.extend({"question": q, "sql": s} for q, s in computed_eval)

    # --- E7. HAVING clauses ---
    having_eval = [
        ("Catégories ayant en moyenne plus de 15 g de protéines.",
         "SELECT categories, ROUND(AVG(proteins), 1) AS avg_proteins FROM products WHERE categories IS NOT NULL GROUP BY categories HAVING AVG(proteins) > 15 ORDER BY avg_proteins DESC LIMIT 50"),
        ("Catégories avec au moins 10 produits dans la base.",
         "SELECT categories, COUNT(*) AS nb FROM products WHERE categories IS NOT NULL GROUP BY categories HAVING COUNT(*) >= 10 ORDER BY nb DESC LIMIT 50"),
        ("Nova groups ayant une moyenne de kcal supérieure à 250.",
         "SELECT nova_group, ROUND(AVG(\"energy-kcal\"), 1) AS avg_kcal FROM products GROUP BY nova_group HAVING AVG(\"energy-kcal\") > 250 ORDER BY avg_kcal DESC"),
        ("Labels ayant en moyenne plus de 5 g de fibres.",
         "SELECT labels, ROUND(AVG(fiber), 2) AS avg_fiber FROM products WHERE labels IS NOT NULL GROUP BY labels HAVING AVG(fiber) > 5 ORDER BY avg_fiber DESC LIMIT 50"),
    ]
    pairs.extend({"question": q, "sql": s} for q, s in having_eval)

    # --- E8. NOVA group variations ---
    nova_eval = [
        ("Produits nova 2 avec plus de 10 g de fibres.",
         "SELECT product_name, fiber, nova_group, \"energy-kcal\" FROM products WHERE nova_group = 2 AND fiber > 10 ORDER BY fiber DESC LIMIT 200"),
        ("Nombre de produits vegan par nova_group.",
         "SELECT nova_group, COUNT(*) AS nb FROM products WHERE vegan = TRUE GROUP BY nova_group ORDER BY nova_group"),
        ("Moyenne de protéines par nova_group pour les produits halal.",
         "SELECT nova_group, ROUND(AVG(proteins), 1) AS avg_proteins FROM products WHERE halal = TRUE GROUP BY nova_group ORDER BY nova_group"),
        ("Produits nova 1 vegan avec moins de 100 kcal.",
         "SELECT product_name, \"energy-kcal\", proteins, fiber FROM products WHERE nova_group = 1 AND vegan = TRUE AND \"energy-kcal\" < 100 ORDER BY \"energy-kcal\" ASC LIMIT 200"),
    ]
    pairs.extend({"question": q, "sql": s} for q, s in nova_eval)

    # --- E9. Specific diet / sport context ---
    context_eval = [
        ("Aliments adaptés à un régime végétarien pour couvrir les besoins en protéines (>15g).",
         "SELECT product_name, proteins, \"energy-kcal\", fat FROM products WHERE vegetarian = TRUE AND proteins > 15 ORDER BY proteins DESC LIMIT 50"),
        ("Produits post-entraînement : riches en protéines (>25g) et en glucides (>30g).",
         "SELECT product_name, proteins, carbohydrates, \"energy-kcal\" FROM products WHERE proteins > 25 AND carbohydrates > 30 ORDER BY proteins DESC LIMIT 50"),
        ("Aliments faibles en FODMAP : faibles glucides (<5g) et fibres modérées (<4g).",
         "SELECT product_name, carbohydrates, fiber, \"energy-kcal\" FROM products WHERE carbohydrates < 5 AND fiber < 4 ORDER BY carbohydrates ASC LIMIT 200"),
        ("Produits riches en oméga-3 potentiels : poisson avec plus de 5 g de lipides.",
         "SELECT product_name, fat, proteins, \"energy-kcal\" FROM products WHERE fish = TRUE AND fat > 5 ORDER BY fat DESC LIMIT 50"),
        ("Aliments pour un régime hypocalorique strict : moins de 80 kcal et plus de 3 g de protéines.",
         "SELECT product_name, \"energy-kcal\", proteins, fiber FROM products WHERE \"energy-kcal\" < 80 AND proteins > 3 ORDER BY \"energy-kcal\" ASC LIMIT 200"),
        ("Produits riches en glucides complexes pour l'endurance (féculents > 50 g glucides).",
         "SELECT product_name, carbohydrates, fiber, \"energy-kcal\" FROM products WHERE feculents = TRUE AND carbohydrates > 50 ORDER BY carbohydrates DESC LIMIT 100"),
        ("Aliments haute densité calorique pour prise de masse : plus de 400 kcal et plus de 20g protéines.",
         "SELECT product_name, \"energy-kcal\", proteins, fat, carbohydrates FROM products WHERE \"energy-kcal\" > 400 AND proteins > 20 ORDER BY \"energy-kcal\" DESC LIMIT 100"),
        ("Sources végétales de calcium : produits laitiers végétaux (vegan, lait = FALSE) avec plus de 5 g protéines.",
         "SELECT product_name, proteins, fat, \"energy-kcal\" FROM products WHERE vegan = TRUE AND lait = FALSE AND proteins > 5 ORDER BY proteins DESC LIMIT 100"),
    ]
    pairs.extend({"question": q, "sql": s} for q, s in context_eval)

    # --- E10. UNION / multi-tag comparisons ---
    union_eval = [
        ("Compare les kcal moyennes entre produits bio et non bio.",
         "SELECT 'bio' AS type, ROUND(AVG(\"energy-kcal\"), 1) AS avg_kcal FROM products WHERE bio = TRUE UNION ALL SELECT 'non bio', ROUND(AVG(\"energy-kcal\"), 1) FROM products WHERE bio = FALSE"),
        ("Protéines moyennes : vegan vs non-vegan.",
         "SELECT 'vegan' AS type, ROUND(AVG(proteins), 1) AS avg_proteins FROM products WHERE vegan = TRUE UNION ALL SELECT 'non-vegan', ROUND(AVG(proteins), 1) FROM products WHERE vegan = FALSE"),
        ("Nombre de produits halal vs kascher vs sans gluten.",
         "SELECT 'halal' AS regime, COUNT(*) AS nb FROM products WHERE halal = TRUE UNION ALL SELECT 'kascher', COUNT(*) FROM products WHERE kascher = TRUE UNION ALL SELECT 'gluten_free', COUNT(*) FROM products WHERE gluten_free = TRUE"),
        ("Produits nova 1 ou nova 2 avec plus de 8 g de fibres.",
         "SELECT product_name, nova_group, fiber, \"energy-kcal\" FROM products WHERE nova_group IN (1, 2) AND fiber > 8 ORDER BY fiber DESC LIMIT 200"),
    ]
    pairs.extend({"question": q, "sql": s} for q, s in union_eval)

    # --- E11. Distinct / IS NULL / IS NOT NULL patterns ---
    null_eval = [
        ("Produits dont les labels ne sont pas renseignés.",
         "SELECT product_name, proteins, \"energy-kcal\" FROM products WHERE labels IS NULL LIMIT 200"),
        ("Produits dont les catégories sont renseignées.",
         "SELECT product_name, categories, proteins, \"energy-kcal\" FROM products WHERE categories IS NOT NULL LIMIT 200"),
        ("Combien de produits n'ont pas de nova_group ?",
         "SELECT COUNT(*) AS nb_sans_nova FROM products WHERE nova_group IS NULL"),
        ("Produits avec des fibres non renseignées.",
         "SELECT product_name, proteins, \"energy-kcal\" FROM products WHERE fiber IS NULL LIMIT 200"),
    ]
    pairs.extend({"question": q, "sql": s} for q, s in null_eval)

    # --- E12. Multi-ratio and advanced sorting ---
    ratio_eval = [
        ("Les 15 produits avec le meilleur ratio fibres/glucides parmi les féculents.",
         "SELECT product_name, fiber, carbohydrates, ROUND(fiber / NULLIF(carbohydrates, 0), 3) AS ratio FROM products WHERE feculents = TRUE AND carbohydrates > 0 ORDER BY ratio DESC LIMIT 15"),
        ("Top 10 des pains avec le meilleur ratio protéines/glucides.",
         "SELECT product_name, proteins, carbohydrates, ROUND(proteins / NULLIF(carbohydrates, 0), 3) AS ratio FROM products WHERE breads = TRUE AND carbohydrates > 0 ORDER BY ratio DESC LIMIT 10"),
        ("Produits poisson avec meilleur ratio protéines/lipides, top 20.",
         "SELECT product_name, proteins, fat, ROUND(proteins / NULLIF(fat, 0), 2) AS ratio_prot_fat FROM products WHERE fish = TRUE AND fat > 0 ORDER BY ratio_prot_fat DESC LIMIT 20"),
        ("Les 25 produits vegan avec le plus faible ratio lipides/kcal.",
         "SELECT product_name, fat, \"energy-kcal\", ROUND(fat / NULLIF(\"energy-kcal\", 0) * 100, 3) AS ratio_fat_kcal FROM products WHERE vegan = TRUE AND \"energy-kcal\" > 0 ORDER BY ratio_fat_kcal ASC LIMIT 25"),
        ("Top 10 des produits bio avec le meilleur score (protéines × fibres / kcal).",
         "SELECT product_name, proteins, fiber, \"energy-kcal\", ROUND(proteins * COALESCE(fiber, 0) / NULLIF(\"energy-kcal\", 0), 4) AS score FROM products WHERE bio = TRUE AND \"energy-kcal\" > 0 ORDER BY score DESC LIMIT 10"),
    ]
    pairs.extend({"question": q, "sql": s} for q, s in ratio_eval)

    # --- E13. Natural language variations (new phrasings) ---
    natural_eval = [
        ("Je veux manger sainement, quels produits sont à la fois vegan et sans gluten ?",
         "SELECT product_name, proteins, fiber, \"energy-kcal\" FROM products WHERE vegan = TRUE AND gluten_free = TRUE ORDER BY proteins DESC LIMIT 200"),
        ("Quels aliments me donneront le plus d'énergie avant le sport ?",
         "SELECT product_name, \"energy-kcal\", carbohydrates, proteins FROM products WHERE carbohydrates > 30 AND \"energy-kcal\" > 200 ORDER BY \"energy-kcal\" DESC LIMIT 50"),
        ("Montre-moi des alternatives végétales aux produits laitiers.",
         "SELECT product_name, proteins, fat, \"energy-kcal\" FROM products WHERE vegan = TRUE AND lait = FALSE AND proteins > 3 ORDER BY proteins DESC LIMIT 100"),
        ("Quels produits sont bons pour la santé intestinale (fibres élevées) ?",
         "SELECT product_name, fiber, carbohydrates, \"energy-kcal\" FROM products WHERE fiber > 10 ORDER BY fiber DESC LIMIT 100"),
        ("Donne-moi des idées d'aliments pour un régime anti-inflammatoire (poisson, légumes).",
         "SELECT product_name, proteins, fat, fiber, \"energy-kcal\" FROM products WHERE (fish = TRUE OR (vegan = TRUE AND fiber > 5)) ORDER BY proteins DESC LIMIT 100"),
        ("Y a-t-il des produits adaptés aux personnes intolérantes au gluten et au lactose ?",
         "SELECT product_name, proteins, fiber, \"energy-kcal\" FROM products WHERE gluten_free = TRUE AND lait = FALSE ORDER BY proteins DESC LIMIT 200"),
        ("Quels snacks bio peuvent convenir à un enfant sportif ?",
         "SELECT product_name, \"energy-kcal\", carbohydrates, proteins, fiber FROM products WHERE snacks = TRUE AND bio = TRUE AND \"energy-kcal\" BETWEEN 100 AND 350 ORDER BY proteins DESC LIMIT 50"),
        ("Aliments à préparer rapidement riches en protéines et peu transformés (nova <= 2).",
         "SELECT product_name, proteins, nova_group, \"energy-kcal\" FROM products WHERE proteins > 15 AND nova_group <= 2 ORDER BY proteins DESC LIMIT 100"),
        ("Quels sont les produits les mieux équilibrés en macros (protéines, lipides et glucides entre 10 et 30 g) ?",
         "SELECT product_name, proteins, fat, carbohydrates, \"energy-kcal\" FROM products WHERE proteins BETWEEN 10 AND 30 AND fat BETWEEN 10 AND 30 AND carbohydrates BETWEEN 10 AND 30 ORDER BY proteins DESC LIMIT 200"),
    ]
    pairs.extend({"question": q, "sql": s} for q, s in natural_eval)

    return pairs


# ─── Format for Bedrock Converse ──────────────────────────────────────────────


def _to_bedrock_converse(question: str, sql: str) -> dict:
    """Format a single example for Bedrock Nova fine-tuning.

    Nova Micro fine-tuning requires ``schemaVersion: bedrock-conversation-2023``
    with ``system`` as a top-level list and only ``user``/``assistant`` in
    ``messages`` (the system role must NOT appear inside ``messages``).
    Reference: https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-prepare.html
    """
    return {
        "schemaVersion": "bedrock-conversation-2023",
        "system": [{"text": SYSTEM_PROMPT}],
        "messages": [
            {"role": "user", "content": [{"text": question}]},
            {"role": "assistant", "content": [{"text": sql}]},
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Generate NL→SQL training data for Bedrock fine-tuning")
    parser.add_argument("--output-dir", default="./finetuning/data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling training pairs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_pairs = _generate_pairs()
    eval_pairs = _generate_eval_pairs()

    random.seed(args.seed)
    random.shuffle(train_pairs)
    random.shuffle(eval_pairs)

    train_path = os.path.join(args.output_dir, "train.jsonl")
    eval_path = os.path.join(args.output_dir, "eval.jsonl")

    for path, data in [(train_path, train_pairs), (eval_path, eval_pairs)]:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                record = _to_bedrock_converse(item["question"], item["sql"])
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✓ Jeu d'entraînement : {len(train_pairs)} exemples → {train_path}")
    print(f"✓ Jeu d'évaluation  : {len(eval_pairs)} exemples → {eval_path}")


if __name__ == "__main__":
    main()
