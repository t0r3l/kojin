"""Exploration — langage naturel → SQL DuckDB.

Référence : **Bedrock** ou **Groq** (``LLM_PROVIDER`` / ``GROQ_API_KEY`` / ``auto``).
Comparaison : ``BEDROCK_COMPARE_MODEL_ID`` ou ``OPENAI_COMPAT_*``. Voir DEPLOYMENT.md §6C.
"""

from __future__ import annotations

import hashlib
import os
import re
import time

import duckdb
import streamlit as st

from kojin_common import (
    BEDROCK_MODEL_ID,
    CSV_PATH,
    DUCKDB_PATH,
    DUCKDB_TABLE,
    append_exploration_metrics_event,
    apply_theme,
    compare_agent_mode,
    compare_sql_llm_label,
    ensure_csv_from_s3,
    format_exploration_results_for_display,
    get_chat_llm,
    get_compare_sql_chat_llm,
    reference_llm_provider,
    reference_model_id_for_metrics,
)

apply_theme()
ensure_csv_from_s3()

st.title("Exploration des ingrédients")
st.markdown(
    '<p class="subtitle" style="font-style:italic;color:#AD9E7B">Interrogez la base en langage naturel</p>',
    unsafe_allow_html=True,
)

if reference_llm_provider() == "bedrock" and not BEDROCK_MODEL_ID:
    st.info(
        "🔧 **Aucun modèle LLM configuré.**\n\n"
        "La page Exploration nécessite un modèle Bedrock (Llama fine-tuné). "
        "Définissez `BEDROCK_MODEL_ID` dans la task definition une fois le fine-tuning terminé."
    )
    st.stop()

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
- La colonne des calories s'appelle `"energy-kcal"` (avec guillemets doubles \
obligatoires car le tiret est un opérateur en SQL) : écris toujours \
`"energy-kcal"` et jamais `energy-kcal` ni `energy_kcal`.
- Si la question est ambiguë, choisis l'interprétation la plus utile pour un \
nutritionniste.
"""


def _chain_from_llm(llm):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM_PROMPT), ("human", "{question}")]
    )
    return prompt | llm | StrOutputParser()


_CODE_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _clean_sql(raw: str) -> str:
    """Strip markdown fences and stray prose around the SQL.

    Also fixes unquoted ``energy-kcal`` which DuckDB parses as subtraction.
    """
    raw = raw.strip()
    m = _CODE_FENCE_RE.search(raw)
    if m:
        raw = m.group(1)
    raw = raw.strip().rstrip(";").strip()
    # Safety net: quote any bare energy-kcal not already surrounded by double quotes
    raw = re.sub(r'(?<!")\benergy-kcal\b(?!")', '"energy-kcal"', raw)
    return raw


def _invoke_nl_to_sql(chain, question: str, schema: str) -> str:
    raw = chain.invoke({
        "table": DUCKDB_TABLE,
        "schema": schema,
        "question": question,
    })
    return _clean_sql(raw)


def _execute_sql(sql: str):
    """Returns ``(polars DataFrame | None, error | None, exec_ms)``."""
    t0 = time.perf_counter()
    con = _open_readonly()
    try:
        df = con.execute(sql).pl()
        return df, None, (time.perf_counter() - t0) * 1000
    except Exception as e:
        return None, e, (time.perf_counter() - t0) * 1000
    finally:
        con.close()


def _render_result(df, err, duck_ms: float | None = None):
    if err is not None:
        st.error(f"SQL invalide ou refusé par DuckDB — {err}")
        return
    if len(df) == 0:
        st.info("Aucun résultat.")
        return
    df_disp = format_exploration_results_for_display(df)
    if df_disp is not None:
        st.dataframe(df_disp.to_pandas(), use_container_width=True, hide_index=True)


def _question_fingerprint(q: str) -> str:
    return hashlib.sha256(q.strip().encode("utf-8")).hexdigest()[:16]


# ─── UI ──────────────────────────────────────────────────────────────────────

_compare_model = compare_sql_llm_label()
_cmp_mode = compare_agent_mode()
_rlp = reference_llm_provider()

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
st.markdown(
    """<style>
    [data-testid="stTextArea"] textarea {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #444 !important;
    }
    </style>""",
    unsafe_allow_html=True,
)
question = st.text_area(
    "Votre question",
    value=default_question,
    key="exploration_question",
    height=80,
)

do_compare = False
if _compare_model:
    do_compare = st.checkbox(
        "Comparer les deux agents (référence vs second modèle)",
        value=False,
        key="exploration_compare_agents",
    )

if st.button("Interroger la base", type="primary"):
    if not question.strip():
        st.warning("Posez une question avant de lancer.")
        st.stop()

    schema = _table_schema()
    qfp = _question_fingerprint(question)
    ref_chain = _chain_from_llm(get_chat_llm(temperature=0.0, max_tokens=600))
    _mid = reference_model_id_for_metrics()

    if not do_compare:
        try:
            t0 = time.perf_counter()
            with st.spinner("Génération de la requête…"):
                sql_b = _invoke_nl_to_sql(ref_chain, question, schema)
            gen_ms = (time.perf_counter() - t0) * 1000
        except Exception as exc:
            if _rlp == "groq":
                st.error(
                    "Erreur lors de l'appel à **Groq**. Vérifiez `GROQ_API_KEY` et la connectivité."
                )
            else:
                st.error(
                    "Erreur lors de l'appel à **Amazon Bedrock**. Vérifiez vos credentials AWS et la région."
                )
            st.exception(exc)
            st.stop()

        with st.spinner("Exécution de la requête…"):
            df_b, err_b, duck_ms = _execute_sql(sql_b)
        _render_result(df_b, err_b, duck_ms)
        with st.expander("Requête SQL générée", expanded=False):
            st.code(sql_b, language="sql")

        append_exploration_metrics_event({
            "mode": "single",
            "question_fp": qfp,
            "llm_provider_ref": _rlp,
            "model_ref": _mid,
            "latency_sql_gen_ms": round(gen_ms, 2),
            "latency_duckdb_ms": round(duck_ms, 2),
            "duckdb_ok": err_b is None,
            "rows": len(df_b) if df_b is not None and err_b is None else None,
        })
        st.stop()

    # --- Mode comparaison ---
    col_ref, col_cmp = st.columns(2, gap="large")

    sql_b = None
    ms_b = None
    err_b = None
    df_b = None
    duck_b = None

    with col_ref:
        st.markdown("##### Référence")
        try:
            t0 = time.perf_counter()
            with st.spinner("Génération de la requête…"):
                sql_b = _invoke_nl_to_sql(ref_chain, question, schema)
            ms_b = (time.perf_counter() - t0) * 1000
        except Exception as exc:
            st.error("Échec de l'agent de référence.")
            st.exception(exc)
        if sql_b is not None:
            with st.spinner("Exécution…"):
                df_b, err_b, duck_b = _execute_sql(sql_b)
            _render_result(df_b, err_b, duck_b)
            with st.expander("Requête SQL générée", expanded=False):
                st.code(sql_b, language="sql")

    sql_c = None
    ms_c = None
    err_c = None
    df_c = None
    duck_c = None
    cmp_label = _compare_model or "?"

    with col_cmp:
        st.markdown("##### Second modèle")
        try:
            cmp_llm = get_compare_sql_chat_llm(temperature=0.0, max_tokens=600)
            cmp_chain = _chain_from_llm(cmp_llm)
        except Exception as exc:
            st.error("Impossible d'initialiser l'agent de comparaison.")
            st.exception(exc)
        else:
            try:
                t0 = time.perf_counter()
                with st.spinner("Génération de la requête…"):
                    sql_c = _invoke_nl_to_sql(cmp_chain, question, schema)
                ms_c = (time.perf_counter() - t0) * 1000
            except Exception as exc:
                st.error("Échec de l'agent de comparaison.")
                st.exception(exc)
            else:
                with st.spinner("Exécution…"):
                    df_c, err_c, duck_c = _execute_sql(sql_c)
                _render_result(df_c, err_c, duck_c)
                with st.expander("Requête SQL générée", expanded=False):
                    st.code(sql_c, language="sql")

    sql_identical = (
        sql_b is not None and sql_c is not None and sql_c.strip() == sql_b.strip()
    )
    append_exploration_metrics_event({
        "mode": "compare",
        "question_fp": qfp,
        "compare_backend": _cmp_mode,
        "llm_provider_ref": _rlp,
        "model_ref": _mid,
        "model_compare": cmp_label,
        "latency_ref_sql_ms": round(ms_b, 2) if ms_b is not None else None,
        "latency_compare_sql_ms": round(ms_c, 2) if ms_c is not None else None,
        "latency_ref_duckdb_ms": round(duck_b, 2) if duck_b is not None else None,
        "latency_compare_duckdb_ms": round(duck_c, 2) if duck_c is not None else None,
        "duckdb_ok_ref": err_b is None,
        "duckdb_ok_compare": err_c is None,
        "rows_ref": len(df_b) if df_b is not None and err_b is None else None,
        "rows_compare": len(df_c) if df_c is not None and err_c is None else None,
        "sql_text_equal": sql_identical,
    })
