"""LLM ingredient resolver.

For each *abstract* ingredient string in a recipe (e.g. ``"chicken breast"``),
ask the fine-tuned NL→SQL agent to generate a DuckDB query that retrieves
matching real products from the Kōjin Open Food Facts catalogue. The query
is executed in read-only mode against ``products.duckdb`` and the resulting
sub-catalogue is returned as a Polars DataFrame, ready to be passed to
``optimize_bento`` (§3 of the integration plan).

Reuses the chain wiring from :mod:`app_pages.exploration`:
* ``get_chat_llm`` factory — selects the fine-tuned Bedrock model (or Groq).
* The strict SYSTEM prompt that constrains the model to a single SQL query.
"""

from __future__ import annotations

import os
import random
import re
import threading
import time
from dataclasses import dataclass

import duckdb
import polars as pl

from kojin_common import (
    CSV_PATH,
    DUCKDB_PATH,
    DUCKDB_TABLE,
    get_chat_llm,
)

from .recipes import Recipe, RecipeIngredient


# ── Prompt template ─────────────────────────────────────────────────────────
#
# Keeps the same wire format as ``app_pages/exploration.py`` so the fine-tuned
# model recognises it. We only replace the *question* with one that targets
# ingredient resolution for a single recipe slot.

_SYSTEM_PROMPT = """Tu es un assistant qui traduit une question en langage naturel \
en UNE seule requête SQL DuckDB valide, exécutable contre la table `{table}`.

Schéma de la table `{table}` :
{schema}

Règles strictes :
- Réponds UNIQUEMENT avec la requête SQL — pas de markdown, pas de ```sql```, pas \
d'explication, pas de texte avant ou après.
- Utilise toujours `{table}` comme nom de table.
- Limite par défaut à 25 lignes (`LIMIT 25`) sauf consigne contraire.
- Préfère un SELECT explicite des colonnes pertinentes plutôt que `SELECT *`.
- Pour les colonnes booléennes (vegan, halal, vegetarian, bio, gluten_free, \
kascher, meat, fish, lait, no_palm_oil) compare avec `TRUE` ou `FALSE`.
- Pour des recherches sur ``product_name`` ou ``categories`` utilise \
``ILIKE '%motif%'``.
- La colonne des calories s'appelle `"energy-kcal"` (avec guillemets doubles).
- Renvoie toujours product_name, "energy-kcal", proteins, fat, carbohydrates, fiber.
"""

_USER_TEMPLATE = """Trouve dans le catalogue les produits qui correspondent à \
l'ingrédient suivant pour composer une recette : « {ingredient} » \
(rôle nutritionnel : {role}).{regime_clause}{tags_clause} \
Trie par densité protéique si role=protein, par densité énergétique sinon. \
Limite à 25 lignes."""


def _ensure_duckdb() -> str:
    """Materialise products.duckdb on first call (mirrors exploration page)."""
    if not os.path.exists(DUCKDB_PATH):
        con = duckdb.connect(DUCKDB_PATH)
        con.execute(
            f"CREATE TABLE {DUCKDB_TABLE} AS "
            f"SELECT * FROM read_csv_auto('{CSV_PATH}', sample_size=-1)"
        )
        con.close()
    return DUCKDB_PATH


def _schema() -> str:
    path = _ensure_duckdb()
    con = duckdb.connect(path, read_only=True)
    try:
        cols = con.execute(f"DESCRIBE {DUCKDB_TABLE}").fetchall()
    finally:
        con.close()
    return "\n".join(f"  - {name} ({dtype})" for name, dtype, *_ in cols)


# Global semaphore: at most 4 concurrent LLM calls regardless of how many
# parallel threads are spawned across recipes. Keeps throughput under ~50 RPM
# assuming ~1-2 s per call (4 / 1.5 s ≈ 160 RPM theoretical ceiling, throttled
# down further by the retry backoff when 429s occur).
_LLM_SEMAPHORE = threading.Semaphore(4)

_CODE_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_WORD_BOUNDARY_RE_CACHE: dict[str, re.Pattern] = {}


def _clean_sql(raw: str) -> str:
    raw = raw.strip()
    m = _CODE_FENCE_RE.search(raw)
    if m:
        raw = m.group(1)
    raw = raw.strip().rstrip(";").strip()
    raw = re.sub(r'(?<!")\benergy-kcal\b(?!")', '"energy-kcal"', raw)
    return raw


def _relevance_score(product_name: str, ingredient: str) -> int:
    """0 = exact · 1 = starts-with · 2 = word-boundary · 3 = partial."""
    n = product_name.lower()
    ing = ingredient.lower().strip()
    if n == ing:
        return 0
    if n.startswith(ing + " ") or n.startswith(ing + ",") or n == ing:
        return 1
    pat = _WORD_BOUNDARY_RE_CACHE.get(ing)
    if pat is None:
        pat = re.compile(r"\b" + re.escape(ing) + r"\b")
        _WORD_BOUNDARY_RE_CACHE[ing] = pat
    if pat.search(n):
        return 2
    return 3


def _rerank_by_relevance(df: pl.DataFrame, ingredient: str) -> pl.DataFrame:
    """Re-sort results: exact match first, then starts-with, word-boundary, partial."""
    if df.height == 0 or "product_name" not in df.columns:
        return df
    scores = [_relevance_score(n, ingredient) for n in df["product_name"].to_list()]
    return df.with_columns(pl.Series("_rel", scores)).sort("_rel").drop("_rel")


@dataclass
class ResolvedIngredient:
    """Result of resolving one abstract recipe ingredient against the catalogue."""

    ingredient: RecipeIngredient
    sql: str
    products: pl.DataFrame  # candidate products with nutrient columns


class IngredientResolver:
    """Stateful resolver — reuses one LLM chain across all recipes of a meal plan."""

    def __init__(self, temperature: float = 0.0, max_tokens: int = 400) -> None:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        llm = get_chat_llm(temperature=temperature, max_tokens=max_tokens)
        prompt = ChatPromptTemplate.from_messages(
            [("system", _SYSTEM_PROMPT), ("human", "{question}")]
        )
        self._chain = prompt | llm | StrOutputParser()
        self._schema = _schema()

    # ── per-ingredient resolution ──────────────────────────────────────
    def _build_question(self, ing: RecipeIngredient, regime: str | None, recipe_tags: list[str]) -> str:
        regime_clause = f" Régime imposé : {regime}." if regime else ""
        tags_clause = f" Étiquettes recette : {', '.join(recipe_tags)}." if recipe_tags else ""
        return _USER_TEMPLATE.format(
            ingredient=ing.name,
            role=ing.role,
            regime_clause=regime_clause,
            tags_clause=tags_clause,
        )

    def resolve_one(
        self,
        ing: RecipeIngredient,
        *,
        regime: str | None = None,
        recipe_tags: list[str] | None = None,
        max_retries: int = 5,
    ) -> ResolvedIngredient:
        q = self._build_question(ing, regime, recipe_tags or [])
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                with _LLM_SEMAPHORE:
                    raw = self._chain.invoke(
                        {"table": DUCKDB_TABLE, "schema": self._schema, "question": q}
                    )
                break
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                if "429" in str(exc) or "rate_limit" in msg:
                    wait = (2 ** attempt) + random.uniform(0.5, 1.5)
                    time.sleep(wait)
                else:
                    raise
        else:
            raise RuntimeError(
                f"LLM rate-limit not resolved after {max_retries} retries for '{ing.name}'"
            ) from last_exc

        sql = _clean_sql(raw)
        con = duckdb.connect(_ensure_duckdb(), read_only=True)
        try:
            df = con.execute(sql).pl()
        finally:
            con.close()
        df = _rerank_by_relevance(df, ing.name)
        return ResolvedIngredient(ingredient=ing, sql=sql, products=df)

    # ── per-recipe resolution ──────────────────────────────────────────
    def resolve_recipe(
        self,
        recipe: Recipe,
        *,
        regime: str | None = None,
    ) -> list[ResolvedIngredient]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: list[ResolvedIngredient | None] = [None] * len(recipe.ingredients)

        def _resolve(idx: int, ing: RecipeIngredient) -> tuple[int, ResolvedIngredient]:
            return idx, self.resolve_one(ing, regime=regime, recipe_tags=recipe.tags)

        with ThreadPoolExecutor(max_workers=min(len(recipe.ingredients), 4)) as pool:
            futures = {pool.submit(_resolve, i, ing): i for i, ing in enumerate(recipe.ingredients)}
            for fut in as_completed(futures):
                idx, resolved = fut.result()
                results[idx] = resolved

        return results  # type: ignore[return-value]
