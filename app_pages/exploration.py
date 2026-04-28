"""Exploration des ingrédients — natural-language queries on the products
catalogue, powered by LangChain + Amazon Bedrock (Nova Micro) + DuckDB.

The user types a free-form question in French (or any language), an LLM
chain translates it into a DuckDB SQL query against the ``products`` table,
the query is executed in **read-only** mode, and the result is rendered as
a Streamlit table.
"""

from __future__ import annotations

import os
import re

import duckdb
import streamlit as st

from kojin_common import (
    BEDROCK_MODEL_ID,
    CSV_PATH,
    DUCKDB_PATH,
    DUCKDB_TABLE,
    apply_theme,
    ensure_csv_from_s3,
    get_chat_llm,
)

apply_theme()
ensure_csv_from_s3()

st.title("Exploration des ingrédients")
st.markdown(
    '<p class="subtitle">interrogez la base en langage naturel</p>',
    unsafe_allow_html=True,
)

if not os.path.exists(CSV_PATH):
    st.warning(
        "Le fichier de données n'a pas encore été préparé. "
        "Allez dans **Bento Maker** pour lancer la préparation."
    )
    st.stop()


# ─── DuckDB lazy build ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Construction de l'index DuckDB…")
def _build_duckdb_file() -> str:
    """Materialise the products CSV into a persistent DuckDB file once.

    Subsequent queries open this file in **read-only** mode so the LLM cannot
    DROP / UPDATE / DELETE anything even if it tried.
    """
    if not os.path.exists(DUCKDB_PATH):
        con = duckdb.connect(DUCKDB_PATH)
        con.execute(
            f"CREATE TABLE {DUCKDB_TABLE} AS "
            f"SELECT * FROM read_csv_auto('{CSV_PATH}', sample_size=-1)"
        )
        con.close()
    return DUCKDB_PATH


@st.cache_data(show_spinner=False)
def _table_schema() -> str:
    """Return a compact schema description fed to the LLM as context."""
    path = _build_duckdb_file()
    con = duckdb.connect(path, read_only=True)
    try:
        cols = con.execute(f"DESCRIBE {DUCKDB_TABLE}").fetchall()
    finally:
        con.close()
    return "\n".join(f"  - {name} ({dtype})" for name, dtype, *_ in cols)


def _open_readonly() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(_build_duckdb_file(), read_only=True)


# ─── LangChain SQL chain ─────────────────────────────────────────────────────

_SYSTEM_PROMPT = """Tu es un assistant qui traduit une question en langage naturel \
en UNE seule requête SQL DuckDB valide, exécutable contre la table `{table}`.

Schéma de la table `{table}` :
{schema}

Règles strictes :
- Réponds UNIQUEMENT avec la requête SQL — pas de markdown, pas de ```sql```, pas \
d'explication, pas de texte avant ou après.
- Utilise toujours `{table}` comme nom de table.
- Limite à 200 lignes par défaut (`LIMIT 200`) sauf si la question demande \
explicitement un agrégat ou un autre nombre.
- Préfère un SELECT explicite des colonnes pertinentes plutôt que `SELECT *`.
- Pour les colonnes booléennes (vegan, halal, vegetarian, bio, gluten_free, \
kascher, meat, fish, lait, no_palm_oil) compare avec `TRUE` ou `FALSE`.
- Pour des recherches sur des chaînes (product_name, categories) utilise \
`ILIKE '%motif%'`.
- Si la question est ambiguë, choisis l'interprétation la plus utile pour un \
nutritionniste.
"""


@st.cache_resource(show_spinner=False)
def _get_chain():
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    llm = get_chat_llm(temperature=0.0, max_tokens=600)
    prompt = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM_PROMPT), ("human", "{question}")]
    )
    return prompt | llm | StrOutputParser()


_CODE_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _clean_sql(raw: str) -> str:
    """Strip markdown fences and stray prose around the SQL."""
    raw = raw.strip()
    m = _CODE_FENCE_RE.search(raw)
    if m:
        raw = m.group(1)
    raw = raw.strip().rstrip(";").strip()
    return raw


# ─── UI ──────────────────────────────────────────────────────────────────────

st.markdown(
    f"Cette page utilise **Amazon Bedrock** (`{BEDROCK_MODEL_ID}`) via **LangChain** "
    "pour convertir votre question en SQL DuckDB, exécuté en lecture seule sur la "
    "base de produits. Des **credentials AWS valides** et l'accès au modèle dans la "
    "console Bedrock sont requis (en local : `aws configure` ou `aws sso login`). "
    "Quelques exemples :"
)

example_queries = [
    "Quels produits vegan ont plus de 25 g de protéines pour 100 g ?",
    "Top 20 des aliments avec le plus de fibres et au maximum 200 kcal pour 100 g.",
    "Liste les produits halal et sans gluten avec leur ratio protéines / kcal.",
    "Combien de produits bio dans la base par groupe NOVA ?",
]
with st.expander("Exemples de questions"):
    for ex in example_queries:
        st.markdown(f"- {ex}")

default_question = st.session_state.get(
    "exploration_question", example_queries[0]
)
question = st.text_area(
    "Votre question",
    value=default_question,
    key="exploration_question",
    height=80,
)

if st.button("Interroger la base", type="primary"):
    if not question.strip():
        st.warning("Posez une question avant de lancer.")
        st.stop()

    chain = _get_chain()
    schema = _table_schema()

    try:
        with st.spinner("Génération de la requête SQL…"):
            raw_sql = chain.invoke({
                "table": DUCKDB_TABLE,
                "schema": schema,
                "question": question,
            })
        sql = _clean_sql(raw_sql)
    except Exception as exc:
        st.error(
            "Erreur lors de l'appel à **Amazon Bedrock**. Vérifiez : credentials "
            "AWS valides (`aws sts get-caller-identity`), région `AWS_REGION` "
            "alignée avec celle de la console Bedrock, accès activé au modèle "
            f"`{BEDROCK_MODEL_ID}`, et permission IAM `bedrock:InvokeModel`."
        )
        st.exception(exc)
        st.stop()

    st.markdown("**Requête générée**")
    st.code(sql, language="sql")

    try:
        with st.spinner("Exécution de la requête…"):
            con = _open_readonly()
            try:
                result = con.execute(sql).pl()
            finally:
                con.close()
    except Exception as exc:
        st.error(f"Erreur lors de l'exécution SQL : {exc}")
        st.stop()

    if len(result) == 0:
        st.info("Aucun résultat.")
    else:
        st.caption(f"{len(result):,} ligne(s) — {len(result.columns)} colonne(s)")
        st.dataframe(result.to_pandas(), use_container_width=True, hide_index=True)
