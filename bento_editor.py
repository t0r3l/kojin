"""Édition interactive des bentos — recherche catalogue, quantités, locks, rééquilibrage."""

from __future__ import annotations

import html
import re
from collections import Counter
from typing import Callable

import polars as pl
import streamlit as st

from kojin_common import (
    BENTO_NAMES,
    NUTR_COLS,
    NUTR_DISPLAY,
    catalog_rows_matching_name,
    format_product_display_label,
    optimize_bento,
)

# ─── Constantes ───────────────────────────────────────────────────────────────

MAX_SEARCH_RESULTS = 30
MIN_QTY_G = 0.01  # plancher compat solveur / portions très petites (ex. huile < 1 g)
MAX_QTY_G = 500.0
# Libellé minimal pour boutons uniquement‑icône Material (couleur pilotée par CSS ``color``).
_ICON_ONLY_BTN_LABEL = "\u2060"


def _qty_in_widget_bounds(q: float) -> float:
    """Valeur utilisée par ``st.number_input`` (évite StreamlitValueBelowMinError / AboveMax)."""
    return max(float(MIN_QTY_G), min(float(MAX_QTY_G), float(q)))


def _scale_nutr_values(row: dict, factor: float) -> dict[str, float]:
    """Scale all NUTR_DISPLAY values in a row dict by *factor*."""
    return {NUTR_DISPLAY[nc]: round(float(row.get(NUTR_DISPLAY[nc], 0.0) or 0.0) * factor, 1) for nc in NUTR_COLS}


DAILY_MATCH_REL_TOL = 0.11  # tolérance (arrondis solver + prot / repas vs prot jour)

# Lors du rééquilibrage après édition, ``optimize_bento`` appelle scipy.nnls avec une colonne par produit :
# plusieurs dizaines de milliers de produits ⇒ plusieurs minutes CPU par appel × plusieurs repas.
_REOPT_POOL_MAX_ROWS = 3000


def _product_pool_cap_for_reoptimize(pool: pl.DataFrame, *, prefer_product_names: list[str] | None) -> pl.DataFrame:
    """Garde tous les candidats désirables puis complète par tirage pour rester sous le plafond."""
    n = len(pool)
    if n <= _REOPT_POOL_MAX_ROWS:
        return pool
    pname = pl.col("product_name")
    if prefer_product_names and "product_name" in pool.columns:
        uniq = sorted({str(x).strip() for x in prefer_product_names if str(x).strip()})
        if uniq:
            keep = pool.filter(pname.is_in(uniq))
            rest = pool.filter(~pname.is_in(uniq))
            nk = len(keep)
            if nk >= _REOPT_POOL_MAX_ROWS:
                return keep.sample(n=_REOPT_POOL_MAX_ROWS, shuffle=True, seed=43)
            need = _REOPT_POOL_MAX_ROWS - nk
            if len(rest) <= need:
                return pl.concat([keep, rest], how="vertical")
            return pl.concat([keep, rest.sample(n=need, shuffle=True, seed=42)], how="vertical")
    return pool.sample(n=_REOPT_POOL_MAX_ROWS, shuffle=True, seed=42)


def _kcal_per100_from_catalog_row(row: dict) -> str:
    """Énergie pour 100 g (données catalogue)."""
    raw = row.get("energy-kcal")
    try:
        v = float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        v = 0.0
    if v <= 0:
        return "—"
    if v >= 10:
        return f"{int(round(v))} kcal / 100 g"
    return f"{round(v, 1)} kcal / 100 g"


# ─── Targets journaliers & par bento ──────────────────────────────────────────


def full_daily_targets_dict(energy: float, proteins: float, fat: float, carbs: float) -> dict[str, float]:
    """Totaux journaliers alignés sur ``_adjust_bentos_after_edit`` / fibre fixe 25 g."""
    return {
        "energy-kcal": float(energy),
        "proteins": float(proteins),
        "fat": float(fat),
        "carbohydrates": float(carbs),
        "fiber": 25.0,
    }


def per_bento_macro_targets(
    energy_daily: float,
    fat_daily: float,
    carbs_daily: float,
    meal_fraction: float,
    protein_per_bento: float,
) -> dict[str, float]:
    """Cibles affichées pour un bento — même logique que ``optimize_bento`` (protéines déjà « par repas »)."""
    return {
        "energy-kcal": energy_daily * meal_fraction,
        "proteins": float(protein_per_bento),
        "fat": fat_daily * meal_fraction,
        "carbohydrates": carbs_daily * meal_fraction,
        "fiber": 25.0 * meal_fraction,
    }


# ─── Helpers nutrition ──────────────────────────────────────────────────────────


def _bento_totals(df: pl.DataFrame) -> dict[str, float]:
    totals: dict[str, float] = {}
    for nc, label in NUTR_DISPLAY.items():
        if label in df.columns:
            totals[nc] = float(df[label].sum())
        else:
            totals[nc] = 0.0
    return totals


def _recompute_row_nutrition(product_row: dict, qty_g: float) -> dict[str, float]:
    result = {}
    for nc, label in NUTR_DISPLAY.items():
        per100 = float(product_row.get(nc, 0.0) or 0.0)
        result[label] = round(per100 * qty_g / 100.0, 1)
    return result


def _rows_to_polars_preserve_order(edit_rows: list[dict]) -> pl.DataFrame:
    """Tableau bento sans réordonnancement (ordre stable pour l’UI ligne par ligne)."""
    if not edit_rows:
        return pl.DataFrame(
            schema={
                "Aliment": pl.String,
                "Quantité (g)": pl.Float64,
                **{label: pl.Float64 for label in NUTR_DISPLAY.values()},
            }
        )
    return pl.DataFrame(edit_rows)


def recalculate_bento_table_from_pandas(
    edited: object,
    catalog: pl.DataFrame,
    *,
    fallback_pl: pl.DataFrame | None,
) -> pl.DataFrame:
    """Recalcule les colonnes nutritionnelles après édition des quantités dans un tableau."""
    import pandas as pd

    edited = edited if isinstance(edited, pd.DataFrame) else pd.DataFrame(edited)
    if edited is None or len(edited) == 0:
        return _rows_to_polars_preserve_order([])

    rows_out: list[dict] = []
    for pos in range(len(edited)):
        rpd = edited.iloc[pos]
        name = str(rpd.get("Aliment", ""))
        qty = _qty_in_widget_bounds(float(rpd.get("Quantité (g)", MIN_QTY_G)))
        pm = catalog_rows_matching_name(catalog, name)
        if len(pm) > 0:
            prow = pm.row(0, named=True)
            nutr = _recompute_row_nutrition(prow, qty)
            rows_out.append({"Aliment": name, "Quantité (g)": round(qty, 1), **nutr})
        elif fallback_pl is not None and pos < len(fallback_pl):
            old = fallback_pl.row(pos, named=True)
            old_qty = max(float(old.get("Quantité (g)", MIN_QTY_G)), MIN_QTY_G)
            factor = qty / old_qty
            item: dict = {"Aliment": name, "Quantité (g)": round(qty, 1)}
            for _, lbl in NUTR_DISPLAY.items():
                v = float(old.get(lbl, 0.0) or 0.0)
                item[lbl] = round(v * factor, 1)
            rows_out.append(item)
        else:
            row_d = {"Aliment": name, "Quantité (g)": round(qty, 1)}
            for _, lbl in NUTR_DISPLAY.items():
                row_d[lbl] = round(float(rpd.get(lbl, 0) or 0), 1)
            rows_out.append(row_d)
    return _rows_to_polars_preserve_order(rows_out)


def pandas_qty_vectors_differ(orig_t: object, edit_t: object) -> bool:
    import pandas as pd

    orig = orig_t if isinstance(orig_t, pd.DataFrame) else pd.DataFrame(orig_t)
    edit = edit_t if isinstance(edit_t, pd.DataFrame) else pd.DataFrame(edit_t)
    if len(orig) != len(edit):
        return True
    if "Quantité (g)" not in orig.columns or "Quantité (g)" not in edit.columns:
        return True
    oq = pd.to_numeric(orig["Quantité (g)"], errors="coerce").fillna(0).to_numpy(dtype=float)
    eq = pd.to_numeric(edit["Quantité (g)"], errors="coerce").fillna(0).to_numpy(dtype=float)
    return bool((abs(eq - oq) > 5e-3).any())


def daily_totals_from_bentos(all_bentos: list[pl.DataFrame | None]) -> dict[str, float]:
    totals = {nc: 0.0 for nc in NUTR_COLS}
    for bento in all_bentos:
        if bento is not None and len(bento) > 0:
            t = _bento_totals(bento)
            for nc in NUTR_COLS:
                totals[nc] += t.get(nc, 0.0)
    return totals


def daily_targets_respected(actual: dict[str, float], target: dict[str, float], rel_tol: float) -> bool:
    for nc in NUTR_COLS:
        tgt = max(target[nc], 1e-6)
        if abs(actual[nc] - target[nc]) / tgt > rel_tol:
            return False
    return True


# ─── Recherche produits ─────────────────────────────────────────────────────────


def _multiset_overlap_no_spaces(pat_norm: str, text: str) -> int:
    """Intersection multiset des caractères (hors espaces), insensible à la casse."""
    pn = "".join(pat_norm.lower().split())
    nn = "".join(text.lower().split())
    cp = Counter(pn)
    cn = Counter(nn)
    return int(sum(min(cp[c], cn[c]) for c in cp))


def _longest_contiguous_pat_chunk_in_text(pat: str, text: str) -> int:
    """Longueur max d’un segment contigu du motif présent tel quel dans le nom.

    Recherche binaire sur la longueur + sliding window → O(n·log(n)) au lieu de O(n³).
    """
    p, n = pat.lower(), text.lower()
    if not p:
        return 0
    lp = len(p)

    def _has_chunk_of_len(cl: int) -> bool:
        for s in range(lp - cl + 1):
            if p[s: s + cl] in n:
                return True
        return False

    lo, hi, best = 1, lp, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if _has_chunk_of_len(mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _filter_products_by_substring(products_df: pl.DataFrame, pattern: str) -> pl.DataFrame:
    if not pattern or len(pattern.strip()) < 2:
        return pl.DataFrame(schema=products_df.schema)
    try:
        escaped = re.escape(pattern.strip())
        return products_df.filter(
            pl.col("product_name").str.to_lowercase().str.contains(escaped.lower())
        )
    except Exception:
        return pl.DataFrame(schema=products_df.schema)


def search_products_for_pattern(
    products_df: pl.DataFrame, pattern: str, top_n: int = MAX_SEARCH_RESULTS
) -> pl.DataFrame:
    """Filtre par sous-chaîne puis trie : plus de caractères en commun, puis plus long chevauchement contigu."""
    raw = pattern or ""
    hits = _filter_products_by_substring(products_df, raw)
    if len(hits) == 0:
        return hits
    pat_clean = raw.strip().lower()

    # Pré-calculer le Counter du pattern une seule fois
    pat_no_spaces = "".join(pat_clean.split())
    cp = Counter(pat_no_spaces)

    names = hits["product_name"].to_list()

    # Calculer les scores sur les noms uniquement (pas de .to_dicts())
    scores: list[tuple[int, int, int, int, str, int]] = []
    for i, nm_raw in enumerate(names):
        nm = str(nm_raw) if nm_raw is not None else ""
        nm_lower = nm.lower()
        # multiset overlap (réutilise cp pré-calculé)
        nn = "".join(nm_lower.split())
        cn = Counter(nn)
        ov = sum(min(cp[c], cn[c]) for c in cp)
        contiguous = _longest_contiguous_pat_chunk_in_text(pat_clean, nm_lower)
        inl = 0 if pat_clean in nm_lower else 1
        pre = 0 if nm_lower.startswith(pat_clean) else 1
        scores.append((-ov, -contiguous, inl, pre, nm_lower, i))

    scores.sort()
    indices = [s[5] for s in scores[:top_n]]
    return hits[indices]





def estimate_bento_table_height_px(
    n_rows: int,
    *,
    header_px: int = 42,
    row_px: int = 50,
    cap_px: int = 560,
) -> int:
    return min(cap_px, max(header_px + 36, header_px + max(1, n_rows) * row_px))


def estimate_replacement_list_height_px(
    n_matches: int,
    table_h_px: int,
    *,
    reserved_top_px: int = 120,
    row_px: int = 36,
    min_inner: int = 108,
    max_inner: int = 360,
) -> int:
    """Hauteur de la liste de résultats : suit le tableau (à droite) sans le dépasser."""
    avail = max(min_inner, table_h_px - reserved_top_px)
    want = row_px * max(3, min(12, max(3, min(n_matches, 12))))
    return max(min_inner, min(avail, want, max_inner))


def update_bento_row_quantity(
    df_pl: pl.DataFrame, row_idx: int, qty_new: float, catalog: pl.DataFrame
) -> pl.DataFrame:
    rows_dc = df_pl.to_dicts()
    if row_idx < 0 or row_idx >= len(rows_dc):
        return df_pl
    old = rows_dc[row_idx]
    name = str(old.get("Aliment", ""))
    qty = _qty_in_widget_bounds(qty_new)
    pm = catalog_rows_matching_name(catalog, name)
    if len(pm) > 0:
        prow = pm.row(0, named=True)
        nutr = _recompute_row_nutrition(prow, qty)
        rows_dc[row_idx] = {"Aliment": name, "Quantité (g)": round(qty, 1), **nutr}
    else:
        oq = max(float(old.get("Quantité (g)", MIN_QTY_G)), MIN_QTY_G)
        factor = qty / oq
        rows_dc[row_idx] = {"Aliment": name, "Quantité (g)": round(qty, 1), **_scale_nutr_values(old, factor)}
    return _rows_to_polars_preserve_order(rows_dc)


def replace_bento_row_ingredient(
    df_pl: pl.DataFrame, row_idx: int, new_product_name: str, catalog: pl.DataFrame
) -> pl.DataFrame:
    rows_dc = df_pl.to_dicts()
    if row_idx < 0 or row_idx >= len(rows_dc):
        return df_pl
    old = rows_dc[row_idx]
    qty = _qty_in_widget_bounds(float(old.get("Quantité (g)", MIN_QTY_G)))
    pm = catalog_rows_matching_name(catalog, new_product_name)
    if len(pm) == 0:
        return df_pl
    prow = pm.row(0, named=True)
    nutr = _recompute_row_nutrition(prow, qty)
    rows_dc[row_idx] = {"Aliment": new_product_name, "Quantité (g)": round(qty, 1), **nutr}
    return _rows_to_polars_preserve_order(rows_dc)


def clear_qty_widgets_for_bento(bento_idx: int) -> None:
    # Increment generation counter so new widget keys won't collide with stale
    # values that Streamlit re-inserts after widgets are rendered in the same run.
    gen_key = f"_qty_gen_{bento_idx}"
    st.session_state[gen_key] = st.session_state.get(gen_key, 0) + 1
    # Pop old keys (Streamlit may re-insert current-run widgets but old-gen keys are cleaned on next run)
    pref = f"qty_inline_{bento_idx}_"
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith(pref):
            st.session_state.pop(k, None)


def inject_kojin_bento_inline_css_once() -> None:
    if st.session_state.get("_kojin_inline_css_injected"):
        return
    st.session_state["_kojin_inline_css_injected"] = True
    st.markdown(
        """
<style>
/* Libellés aliments : même corps que la liste résultats, la ligne active et les boutons du tableau */
.kojin-aliment-active {
    background:#2a2a2a!important;
    color:#fafaf8!important;
    padding:10px 12px!important;
    border-radius:4px!important;
    border:1px solid #c4c4c0!important;
    font-family:Georgia,'Times New Roman',serif!important;
    font-size:0.9rem!important;
    font-weight:400!important;
    font-style:normal!important;
    line-height:1.38!important;
    text-transform:none!important;
    letter-spacing:normal!important;
    font-variant:normal!important;
}
.kojin-aliment-editor-line {
    font-family:Georgia,'Times New Roman',serif!important;
    font-size:0.9rem!important;
    font-weight:400!important;
    font-style:normal!important;
    line-height:1.38!important;
    text-transform:none!important;
    letter-spacing:normal!important;
    color:#fafaf8!important;
}
/* Résultats recherche : bouton nom (colonne 1) */
:is(section[data-testid="stMain"], :root) div[data-testid="stHorizontalBlock"]:has(div.kojin-hit-kcal)
  div[data-testid="column"]:first-child div[data-testid="stButton"] > button {
    font-family:Georgia,'Times New Roman',serif!important;
    font-size:0.9rem!important;
    font-weight:400!important;
    font-style:normal!important;
    line-height:1.38!important;
    text-transform:none!important;
    letter-spacing:normal!important;
    font-variant:normal!important;
}
/*
 * Tableau macros (pas la colonne « Produits correspondants » : elle contient .kojin-hit-kcal).
 * Cadenas = bouton tertiary Material ; monochrome via color + fill/stroke sur le svg.
 */
:is(section[data-testid="stMain"], :root) div[data-testid="stHorizontalBlock"]:has(div.kojin-nutr-cell):not(:has(div.kojin-hit-kcal))
  > div[data-testid="column"]:first-child
  div[data-testid="stHorizontalBlock"]
  > div[data-testid="column"]:first-child :is(.stButton, div[data-testid="stButton"]) > button {
    font-size:1.12rem!important;
    background:transparent!important;
    border:none!important;
    box-shadow:none!important;
    outline:none!important;
    padding:0.08rem 0.12rem!important;
    min-height:auto!important;
    line-height:1!important;
    gap:0!important;
    color:#AD9E7B!important;
    width:auto!important;
    min-width:unset!important;
}
:is(section[data-testid="stMain"], :root) div[data-testid="stHorizontalBlock"]:has(div.kojin-nutr-cell):not(:has(div.kojin-hit-kcal))
  > div[data-testid="column"]:first-child
  div[data-testid="stHorizontalBlock"]
  > div[data-testid="column"]:first-child :is(.stButton, div[data-testid="stButton"]) > button :is(svg, svg path) {
    fill:currentColor!important;
}
:is(section[data-testid="stMain"], :root) div[data-testid="stHorizontalBlock"]:has(div.kojin-nutr-cell):not(:has(div.kojin-hit-kcal))
  > div[data-testid="column"]:first-child
  div[data-testid="stHorizontalBlock"]:has(.kojin-aliment-active)
  > div[data-testid="column"]:first-child :is(.stButton, div[data-testid="stButton"]) > button {
    color:#fafaf8!important;
}
:is(section[data-testid="stMain"], :root) div[data-testid="stHorizontalBlock"]:has(div.kojin-nutr-cell):not(:has(div.kojin-hit-kcal))
  > div[data-testid="column"]:first-child
  div[data-testid="stHorizontalBlock"]
  > div[data-testid="column"]:nth-child(2) :is(.stButton, div[data-testid="stButton"]) > button {
    font-family:Georgia,'Times New Roman',serif!important;
    font-size:0.9rem!important;
    font-weight:400!important;
    font-style:normal!important;
    line-height:1.38!important;
    text-transform:none!important;
    letter-spacing:normal!important;
    font-variant:normal!important;
    color:#AD9E7B!important;
}
.kojin-nutr-cell {
    background:#2a2a2a!important;
    color:#fafaf8!important;
    padding:7px 8px!important;
    border-radius:4px!important;
    border:1px solid #e8e8e4!important;
    font-family:Arial, Helvetica, sans-serif!important;
    font-size:0.9rem!important;
    font-weight:400!important;
    text-transform:none!important;
    letter-spacing:normal!important;
    line-height:1.35!important;
    text-align:center!important;
}
/* Force number inputs (quantities) to same font/size as macro cells */
div[data-testid="stNumberInput"] input {
    font-family:Arial, Helvetica, sans-serif!important;
    font-size:0.9rem!important;
    font-weight:400!important;
    letter-spacing:normal!important;
    background-color:#1a1a1a!important;
    color:#fafaf8!important;
    border:1px solid #444!important;
}
.kojin-hit-btn button {
    text-transform:none!important;
    letter-spacing:normal!important;
    font-weight:400!important;
}
.stButton > button {
    text-transform:none!important;
    letter-spacing:normal!important;
}
.kojin-side-title {
    color:#333!important;
    font-family:Georgia,'Times New Roman',serif!important;
    font-size:0.95rem!important;
    font-weight:500!important;
    text-transform:none!important;
    letter-spacing:0.02em!important;
    margin-bottom:0.35rem!important;
}
.kojin-table-hdr-cell {
    background:#ffffff!important;
    color:#AD9E7B!important;
    padding:4px 6px 2px 6px!important;
    border-radius:4px!important;
    border:1px solid #e8e8e4!important;
    font-family:Georgia,'Times New Roman',serif!important;
    font-size:0.85rem!important;
    font-weight:600!important;
    text-transform:none!important;
    letter-spacing:normal!important;
    line-height:1.35!important;
    margin-bottom:2px!important;
    white-space:nowrap!important;
    overflow:hidden!important;
    text-overflow:ellipsis!important;
}
.kojin-aliment-hdr { text-align:left!important; }
/* Fix column alignment when sidebar is collapsed */
div[data-testid="stHorizontalBlock"] {
    gap:0.3rem!important;
}
.kojin-hit-kcal {
    font-family:Georgia,'Times New Roman',serif!important;
    font-size:0.9rem!important;
    font-weight:400!important;
    font-style:normal!important;
    color:#333!important;
    text-align:right!important;
    white-space:nowrap!important;
    text-transform:none!important;
    letter-spacing:normal!important;
    padding:0.5rem 0.15rem 0 0!important;
    line-height:1.35!important;
}
/* Ligne titre Bento + cadenas : align vertical au centre, sans marge haute du titre */
.bento-header-inline-gate {
  margin-top: 0 !important;
}
:is(section[data-testid="stMain"], :root) div[data-testid="stHorizontalBlock"]:has(.bento-header-inline-gate) {
  align-items: center !important;
  gap: 0 !important;
  column-gap: 0 !important;
}
:is(section[data-testid="stMain"], :root) div[data-testid="stHorizontalBlock"]:has(.bento-header-inline-gate) > div[data-testid="column"]:first-child {
  display: flex !important;
  align-items: center !important;
  padding-right: 0 !important;
  flex: 0 0 auto !important;
  width: auto !important;
}
:is(section[data-testid="stMain"], :root) div[data-testid="stHorizontalBlock"]:has(.bento-header-inline-gate) > div[data-testid="column"]:nth-child(2) {
  display: flex !important;
  align-items: center !important;
  padding-left: 0 !important;
  margin-left: -0.32rem !important;
}
/* Cadenas repas : Material noir, grand et proche du titre */
:is(section[data-testid="stMain"], :root) div[data-testid="stHorizontalBlock"]:has(.bento-header-inline-gate)
  > div[data-testid="column"]:first-child :is(.stButton, div[data-testid="stButton"]) > button {
    background:transparent!important;
    border:none!important;
    box-shadow:none!important;
    outline:none!important;
    display:inline-flex!important;
    align-items:center!important;
    justify-content:center!important;
    padding:0!important;
    margin-top:0.28rem!important;
    transform:translateY(0.12rem)!important;
    font-size:3.15rem!important;
    text-transform:none!important;
    font-weight:400!important;
    color:#AD9E7B!important;
    min-height:auto!important;
    line-height:1!important;
    gap:0!important;
    width:auto!important;
    min-width:unset!important;
}
:is(section[data-testid="stMain"], :root) div[data-testid="stHorizontalBlock"]:has(.bento-header-inline-gate)
  > div[data-testid="column"]:first-child :is(.stButton, div[data-testid="stButton"]) > button:is(:hover, :focus-visible) {
    background:transparent!important;
    color:#AD9E7B!important;
    border:none!important;
    box-shadow:none!important;
}
:is(section[data-testid="stMain"], :root) div[data-testid="stHorizontalBlock"]:has(.bento-header-inline-gate)
  > div[data-testid="column"]:first-child :is(.stButton, div[data-testid="stButton"]) > button svg {
    width:1em!important;
    height:1em!important;
    flex-shrink:0!important;
}
:is(section[data-testid="stMain"], :root) div[data-testid="stHorizontalBlock"]:has(.bento-header-inline-gate)
  > div[data-testid="column"]:first-child :is(.stButton, div[data-testid="stButton"]) > button svg path {
    fill:#AD9E7B!important;
}
/* Même chose sans ``section#main`` — certains thèmes Streamlit n’emploient pas la même arborescence. */
.stApp div[data-testid="stHorizontalBlock"]:has(.bento-header-inline-gate) > div[data-testid="column"]:first-child button {
    background:transparent!important;
    border:none!important;
    box-shadow:none!important;
    outline:none!important;
    display:inline-flex!important;
    align-items:center!important;
    justify-content:center!important;
    padding:0!important;
    margin-top:0.28rem!important;
    transform:translateY(0.12rem)!important;
    font-size:3.15rem!important;
    color:#AD9E7B!important;
    line-height:1!important;
    min-height:auto!important;
}
.stApp div[data-testid="stHorizontalBlock"]:has(.bento-header-inline-gate) > div[data-testid="column"]:first-child button svg {
    width:1em!important;
    height:1em!important;
    min-width:3.05rem!important;
    min-height:3.05rem!important;
}
/* Search input for ingredient replacement */
div[data-testid="stTextInput"] input {
    color:#fafaf8!important;
    background-color:#1a1a1a!important;
    border:1px solid #444!important;
}
div[data-testid="stTextInput"] input::placeholder {
    color:#888!important;
}
</style>
""",
        unsafe_allow_html=True,
    )



def ensure_ingredient_locks_for_bento_rows(bento_idx: int, df: pl.DataFrame | None) -> list[bool]:
    """Garantit une liste ``ingredient_locks[bento_idx]`` alignée aux lignes du frame."""
    il = st.session_state.setdefault("ingredient_locks", [])
    while len(il) <= bento_idx:
        il.append([])
    n = len(df) if df is not None else 0
    row = il[bento_idx]
    if len(row) < n:
        row.extend([False] * (n - len(row)))
    elif len(row) > n:
        il[bento_idx] = row[:n]
    sync_ingredient_locks_row_count(il, bento_idx, df)
    return il[bento_idx]


def sync_locked_bentos_from_ingredient_locks(n_comp: int) -> None:
    """locked_bentos[i] ≡ tous les ingrédients du repas sont verrouillés (repas entièrement figé)."""
    lb: list[bool] = []
    il = st.session_state.setdefault("ingredient_locks", [])
    for i in range(n_comp):
        row = il[i] if i < len(il) else []
        lb.append(len(row) > 0 and all(bool(x) for x in row))
    st.session_state.locked_bentos = lb


def render_bento_gate_header(
    *,
    bento_idx: int,
    title_inner_html: str,
    meal_n_rows: int,
) -> None:
    """Cadenas repas (Material, monochrome noir) : fermé uniquement si **tous** les aliments sont bloqués ;
    tant qu’au moins un aliment reste libre, le cadenas du repas reste ouvert. Clic : tout débloquer ou tout bloquer."""
    inject_kojin_bento_inline_css_once()
    il = st.session_state.setdefault("ingredient_locks", [])
    while len(il) <= bento_idx:
        il.append([])
    locks = il[bento_idx]
    if meal_n_rows > 0:
        if len(locks) < meal_n_rows:
            locks.extend([False] * (meal_n_rows - len(locks)))
            il[bento_idx] = locks
        elif len(locks) > meal_n_rows:
            il[bento_idx] = locks[:meal_n_rows]
    slice_lk = il[bento_idx][:meal_n_rows] if meal_n_rows > 0 else []
    closed = meal_n_rows > 0 and len(slice_lk) == meal_n_rows and all(bool(x) for x in slice_lk)
    gate_icon = ":material/lock:" if closed else ":material/lock_open:"
    lc, rc = st.columns([0.048, 0.952])
    with lc:
        if st.button(
            _ICON_ONLY_BTN_LABEL,
            key=f"bento_gate_{bento_idx}",
            type="tertiary",
            icon=gate_icon,
            width="content",
            help=(
                "Cadenas du repas : fermé seulement si tout le repas est bloqué. "
                "Tant qu’un aliment reste libre, il reste ouvert. Clic : tout débloquer ou tout bloquer."
            ),
        ):
            n = meal_n_rows if meal_n_rows > 0 else len(il[bento_idx])
            row = list(il[bento_idx][:n]) if n else []
            while len(row) < n:
                row.append(False)
            if n > 0:
                if all(bool(x) for x in row):
                    il[bento_idx] = [False] * n
                else:
                    il[bento_idx] = [True] * n
            st.rerun()
    with rc:
        st.markdown(
            f'<div class="bento-header bento-header-inline-gate">{title_inner_html}</div>',
            unsafe_allow_html=True,
        )


def render_bento_inline_table(
    *,
    bento_idx: int,
    df_view: pl.DataFrame,
    catalog: pl.DataFrame,
    chart_renderer=None,
) -> pl.DataFrame | None:
    """Remplacement d’ingrédient à gauche (saisie + liste de résultats cliquables, hauteur plafonnée) et lignes à droite.

    Le clic sur une ligne ouvre la recherche ; les résultats se mettent à jour à la frappe et un clic sur un nom
    applique tout de suite. La ligne active est en bloc blanc texte noir. Les macros sont affichées en noir sur blanc,
    sans capitalisation forcée."""
    inject_kojin_bento_inline_css_once()
    rows_dict = df_view.to_dicts()
    n_rows = len(rows_dict)
    locks = ensure_ingredient_locks_for_bento_rows(bento_idx, df_view)
    table_h_px = estimate_bento_table_height_px(n_rows)

    repl_key_row = f"bento_rep_row_{bento_idx}"
    pat_k = f"bento_rep_pat_{bento_idx}"

    active_pick = st.session_state.get(repl_key_row)
    editing_row = isinstance(active_pick, int) and 0 <= active_pick < n_rows

    left_w, right_w = st.columns([12, 22])

    with left_w:
        if not editing_row:
            if chart_renderer is not None:
                # Vertically center charts at mid-height of the table.
                chart_h = 210
                top_pad = max(0, (table_h_px - chart_h) // 2)
                if top_pad > 0:
                    st.markdown(f'<div style="height:{top_pad}px"></div>', unsafe_allow_html=True)
                chart_renderer(container=st)
        else:
            pat = st.text_input(
                "Pattern",
                key=pat_k,
                placeholder="ex. tofu, quinoa…",
                label_visibility="collapsed",
            )
            hits = search_products_for_pattern(catalog, pat or "")
            nh = len(hits)
            inner_h = int(estimate_replacement_list_height_px(max(5, nh + 3), table_h_px))

            st.caption(f"{nh} résultat(s)" if nh else "Aucun résultat")
            try:
                _cont = st.container(height=inner_h)
            except TypeError:
                _cont = st.container()
            with _cont:
                hit_rows = hits.to_dicts()
                for hi, hrow in enumerate(hit_rows):
                    pname = str(hrow.get("product_name", ""))
                    display_name = format_product_display_label(pname)
                    plabel = (display_name[:120] + "…") if len(display_name) > 120 else display_name
                    kcal_txt = html.escape(_kcal_per100_from_catalog_row(hrow))
                    c_n, c_k = st.columns([5, 2])
                    with c_n:
                        if st.button(
                            plabel,
                            key=f"bento_hit_{bento_idx}_{active_pick}_{hi}",
                            use_container_width=True,
                        ):
                            # Ne pas ``return`` ici : un retour prématuré empêchait le cycle de widgets
                            # complet et le rééquilibrage en fin de page. On diffère l’application.
                            clear_qty_widgets_for_bento(bento_idx)
                            st.session_state.pop(pat_k, None)
                            st.session_state.pop(repl_key_row, None)
                            st.session_state["_pending_bento_row_replace"] = {
                                "bento_idx": int(bento_idx),
                                "row_idx": int(active_pick),
                                "product_name": pname,
                            }
                            st.rerun()
                    with c_k:
                        st.markdown(
                            f'<div class="kojin-hit-kcal">{kcal_txt}</div>',
                            unsafe_allow_html=True,
                        )

            if st.button("Annuler", key=f"bento_rep_cancel_{bento_idx}", use_container_width=True):
                st.session_state.pop(pat_k, None)
                st.session_state.pop(repl_key_row, None)
                st.rerun()

    with right_w:
        hdr_rat = [5.2] + [1.06] + [1.0, 1.0, 1.0, 1.0, 1.2]
        hrow = right_w.columns(hdr_rat)
        title_cols = ["Ingrédients"] + ["g"] + [NUTR_DISPLAY[nc] for nc in NUTR_COLS]
        for j, ttl in enumerate(title_cols):
            with hrow[j]:
                ttl_esc = html.escape(str(ttl))
                align_cls = " kojin-aliment-hdr" if j == 0 else ""
                st.markdown(
                    f'<div class="kojin-table-hdr-cell{align_cls}">{ttl_esc}</div>',
                    unsafe_allow_html=True,
                )
        right_w.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

        _gen = st.session_state.get(f"_qty_gen_{bento_idx}", 0)
        for idx, rd in enumerate(rows_dict):
            row_c = right_w.columns(hdr_rat)
            raw_name = str(rd.get("Aliment", ""))
            name_plain = html.escape(format_product_display_label(raw_name))
            ov = float(_qty_in_widget_bounds(float(rd.get("Quantité (g)", MIN_QTY_G))))

            lk = locks[idx]

            with row_c[0]:
                c_pad, c_alim = st.columns([0.12, 0.88])
                with c_pad:
                    ing_icon = ":material/lock:" if lk else ":material/lock_open:"
                    if st.button(
                        _ICON_ONLY_BTN_LABEL,
                        key=f"ing_gate_{bento_idx}_{idx}",
                        type="tertiary",
                        icon=ing_icon,
                        width="content",
                        help="Aliment bloqué : épargné du réajustement auto des quantités. Cadenas ouvert ↔ libre.",
                    ):
                        locks[idx] = not bool(locks[idx])
                        st.session_state["_rebalance_anchor_bento"] = bento_idx
                        st.rerun()
                with c_alim:
                    if editing_row and active_pick == idx:
                        st.markdown(
                            f'<div class="kojin-aliment-active">{name_plain}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        dn = format_product_display_label(raw_name)
                        lbl_btn = dn[:200] + ("…" if len(dn) > 200 else "")
                        if st.button(
                            lbl_btn,
                            key=f"bento_pick_alim_{bento_idx}_{idx}",
                            type="secondary",
                            use_container_width=True,
                        ):
                            st.session_state[repl_key_row] = idx
                            if pat_k in st.session_state:
                                st.session_state.pop(pat_k, None)
                            st.rerun()

            qkey = f"qty_inline_{bento_idx}_{idx}_g{_gen}"
            with row_c[1]:
                st.number_input(
                    "g",
                    min_value=float(MIN_QTY_G),
                    max_value=float(MAX_QTY_G),
                    value=ov,
                    step=0.5,
                    key=qkey,
                    label_visibility="collapsed",
                )

            for k, nc in enumerate(NUTR_COLS):
                lbl = NUTR_DISPLAY[nc]
                with row_c[k + 2]:
                    v = rd.get(lbl, "")
                    st.markdown(
                        f'<div class="kojin-nutr-cell">{html.escape(str(v))}</div>',
                        unsafe_allow_html=True,
                    )

        dirty_qty = False
        nu_qty = df_view
        changed_indices: list[int] = []
        _gen = st.session_state.get(f"_qty_gen_{bento_idx}", 0)
        for idx, rd in enumerate(rows_dict):
            qkey = f"qty_inline_{bento_idx}_{idx}_g{_gen}"
            if qkey not in st.session_state:
                continue
            nv = float(st.session_state[qkey])
            ov = float(rd["Quantité (g)"])
            if abs(nv - ov) > 1e-3:
                nu_qty = update_bento_row_quantity(nu_qty, idx, nv, catalog)
                dirty_qty = True
                changed_indices.append(idx)
        if dirty_qty:
            # Persist the changed indices so run_daily_rebalance_after_anchor
            # temporarily locks them during rebalancing (not permanently in the UI).
            st.session_state["_rebalance_locked_rows"] = {bento_idx: list(changed_indices)}
            clear_qty_widgets_for_bento(bento_idx)
            st.session_state["_rebalance_anchor_bento"] = bento_idx
            return nu_qty

    return None


# ─── Ajustement de quantités ──────────────────────────────────────────────────


def _scale_bento_to_targets(
    bento_df: pl.DataFrame,
    target_energy: float,
    tolerance: float = 0.05,
) -> pl.DataFrame:
    if bento_df is None or len(bento_df) == 0:
        return bento_df

    cur_lbl = NUTR_DISPLAY["energy-kcal"]
    current_energy = float(bento_df[cur_lbl].sum())
    if current_energy <= 0 or target_energy <= 0:
        return bento_df

    ratio = target_energy / current_energy
    if abs(ratio - 1.0) < tolerance:
        return bento_df

    new_rows = []
    for row in bento_df.iter_rows(named=True):
        qty0 = float(row["Quantité (g)"])
        new_qty = qty0 * ratio
        new_qty = max(MIN_QTY_G, min(MAX_QTY_G, new_qty))
        per100 = {}
        for nc in NUTR_COLS:
            lbl = NUTR_DISPLAY[nc]
            v = float(row.get(lbl, 0.0) or 0.0)
            per100[nc] = v * 100.0 / max(qty0, 1e-6)
        nutr = _recompute_row_nutrition(per100, new_qty)
        new_rows.append({"Aliment": row["Aliment"], "Quantité (g)": round(new_qty, 1), **nutr})
    return pl.DataFrame(new_rows)


def sync_ingredient_locks_row_count(
    ingredient_locks: list[list[bool]], idx: int, df: pl.DataFrame | None
) -> None:
    """Aligne ``ingredient_locks[idx]`` sur le nombre de lignes du frame (nouvelles lignes = débloquées)."""
    while len(ingredient_locks) <= idx:
        ingredient_locks.append([])
    n = len(df) if df is not None else 0
    row = ingredient_locks[idx]
    if len(row) < n:
        row.extend([False] * (n - len(row)))
    elif len(row) > n:
        ingredient_locks[idx] = row[:n]


def _scale_bento_unlocked_to_energy(
    bento_df: pl.DataFrame,
    locks: list[bool],
    target_meal_energy: float,
    catalog: pl.DataFrame,
    tolerance: float = 0.04,
) -> pl.DataFrame:
    """Varie uniquement les quantités des lignes non verrouillées pour viser une énergie de repas."""
    if len(bento_df) == 0:
        return bento_df
    nk = len(bento_df)
    lk = (locks[:nk] if len(locks) >= nk else locks + [False] * (nk - len(locks)))[:nk]

    lbl_e = NUTR_DISPLAY["energy-kcal"]
    unlocked_e = 0.0
    locked_e = 0.0
    rows_named = list(bento_df.iter_rows(named=True))
    for j, row in enumerate(rows_named):
        e = float(row.get(lbl_e, 0.0) or 0.0)
        if lk[j]:
            locked_e += e
        else:
            unlocked_e += e
    if unlocked_e <= 1e-3:
        return bento_df
    desired_unlock = float(target_meal_energy) - locked_e
    if desired_unlock <= 1e-3:
        return bento_df
    ratio = desired_unlock / unlocked_e
    if abs(ratio - 1.0) <= tolerance:
        return bento_df

    def _frozen_row(rd: dict) -> dict[str, float | str]:
        al = str(rd.get("Aliment", ""))
        q = float(rd.get("Quantité (g)", MIN_QTY_G))
        return {"Aliment": al, "Quantité (g)": round(q, 1), **_scale_nutr_values(rd, 1.0)}

    new_rows: list[dict] = []
    for j, row in enumerate(rows_named):
        rd = dict(row)
        an = str(rd.get("Aliment", ""))
        qty0 = _qty_in_widget_bounds(float(rd.get("Quantité (g)", MIN_QTY_G)))
        if lk[j]:
            new_rows.append(_frozen_row(rd))
            continue
        new_qty = _qty_in_widget_bounds(qty0 * ratio)
        pm = catalog_rows_matching_name(catalog, an)
        if len(pm) > 0:
            prow = pm.row(0, named=True)
            nutr = _recompute_row_nutrition(prow, new_qty)
            new_rows.append({"Aliment": an, "Quantité (g)": round(new_qty, 1), **nutr})
        else:
            factor = new_qty / max(qty0, 1e-6)
            new_rows.append({"Aliment": an, "Quantité (g)": round(new_qty, 1), **_scale_nutr_values(rd, factor)})
    return pl.DataFrame(new_rows)


def _dominant_macro_of_row(row: dict, lbl_e: str) -> str | None:
    """Return the dominant macro nutrient key for a row, or None if indeterminate.
    Uses the macro contributing the most kcal to the row."""
    e = float(row.get(lbl_e, 0.0) or 0.0)
    if e <= 0:
        return None
    prot_g = float(row.get(NUTR_DISPLAY["proteins"], 0.0) or 0.0)
    fat_g = float(row.get(NUTR_DISPLAY["fat"], 0.0) or 0.0)
    carb_g = float(row.get(NUTR_DISPLAY["carbohydrates"], 0.0) or 0.0)
    prot_e = prot_g * 4.0
    fat_e = fat_g * 9.0
    carb_e = carb_g * 4.0
    best = max(prot_e, fat_e, carb_e)
    if best <= 0:
        return None
    if prot_e == best:
        return "proteins"
    if fat_e == best:
        return "fat"
    return "carbohydrates"


def _scale_bento_unlocked_macro_aware(
    bento_df: pl.DataFrame,
    locks: list[bool],
    target_meal_energy: float,
    target_meal_protein: float,
    catalog: pl.DataFrame,
    tolerance: float = 0.005,
    edited_row_indices: list[int] | None = None,
    target_meal_fat: float | None = None,
    target_meal_carbs: float | None = None,
) -> pl.DataFrame:
    """Scale unlocked rows per macro-category to respect ALL macro targets.

    Rows are grouped by dominant macro (proteins/fat/carbs). A 3×3 linear system
    is solved to find per-group scaling ratios that simultaneously hit all three
    macro targets, accounting for cross-contributions (e.g. rice has protein too).
    """
    import numpy as np

    if len(bento_df) == 0:
        return bento_df
    nk = len(bento_df)
    lk = (locks[:nk] if len(locks) >= nk else locks + [False] * (nk - len(locks)))[:nk]

    lbl_e = NUTR_DISPLAY["energy-kcal"]
    rows_named = list(bento_df.iter_rows(named=True))

    macro_keys = ["proteins", "fat", "carbohydrates"]
    macro_lbls = [NUTR_DISPLAY[mk] for mk in macro_keys]

    # Macro targets for this bento (grams)
    targets = [
        target_meal_protein,
        target_meal_fat if target_meal_fat is not None else target_meal_energy * 0.30 / 9.0,
        target_meal_carbs if target_meal_carbs is not None else target_meal_energy * 0.45 / 4.0,
    ]

    # Classify unlocked rows by dominant macro; accumulate locked contributions
    groups: dict[str, list[int]] = {"proteins": [], "fat": [], "carbohydrates": []}
    locked_macros = [0.0, 0.0, 0.0]  # [P, F, C] from locked rows

    for j, row in enumerate(rows_named):
        if lk[j]:
            for mi, ml in enumerate(macro_lbls):
                locked_macros[mi] += float(row.get(ml, 0.0) or 0.0)
            continue
        dm = _dominant_macro_of_row(row, lbl_e)
        if dm is None:
            dm = "carbohydrates"
        groups[dm].append(j)

    # If all rows are locked, nothing to scale
    total_unlocked = sum(len(v) for v in groups.values())
    if total_unlocked == 0:
        return bento_df

    # Build 3×3 coefficient matrix: A[macro_i][group_j] = sum of macro_i grams in group_j
    # and RHS: b[macro_i] = target_i - locked_i
    group_order = ["proteins", "fat", "carbohydrates"]
    A = np.zeros((3, 3))
    b = np.array([targets[i] - locked_macros[i] for i in range(3)])

    for gi, gk in enumerate(group_order):
        for j in groups[gk]:
            row = rows_named[j]
            for mi, ml in enumerate(macro_lbls):
                A[mi, gi] += float(row.get(ml, 0.0) or 0.0)

    # Solve for ratios; fall back to uniform energy scaling if singular
    ratios = np.ones(3)
    try:
        if np.linalg.matrix_rank(A) >= 3:
            ratios = np.linalg.solve(A, b)
        else:
            # Under-determined: use least-squares
            result_ls, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            ratios = result_ls
    except np.linalg.LinAlgError:
        return _scale_bento_unlocked_to_energy(bento_df, locks, target_meal_energy, catalog, tolerance)

    # Clamp ratios to reasonable range [0.1, 5.0] to avoid extreme scaling
    ratios = np.clip(ratios, 0.1, 5.0)

    # Check if any change is needed
    if all(abs(r - 1.0) <= tolerance for r in ratios):
        return bento_df

    # Map each row index to its group's ratio
    row_ratio: dict[int, float] = {}
    for gi, gk in enumerate(group_order):
        for idx in groups[gk]:
            row_ratio[idx] = float(ratios[gi])

    # Build new rows
    new_rows: list[dict] = []
    for j, row in enumerate(rows_named):
        rd = dict(row)
        an = str(rd.get("Aliment", ""))
        qty0 = _qty_in_widget_bounds(float(rd.get("Quantité (g)", MIN_QTY_G)))
        if lk[j] or j not in row_ratio:
            new_rows.append({"Aliment": an, "Quantité (g)": round(qty0, 1), **_scale_nutr_values(rd, 1.0)})
            continue
        ratio = row_ratio[j]
        new_qty = _qty_in_widget_bounds(qty0 * ratio)
        pm = catalog_rows_matching_name(catalog, an)
        if len(pm) > 0:
            prow = pm.row(0, named=True)
            nutr = _recompute_row_nutrition(prow, new_qty)
            new_rows.append({"Aliment": an, "Quantité (g)": round(new_qty, 1), **nutr})
        else:
            factor = new_qty / max(qty0, 1e-6)
            new_rows.append({"Aliment": an, "Quantité (g)": round(new_qty, 1), **_scale_nutr_values(rd, factor)})

    return pl.DataFrame(new_rows)


def _adjust_bentos_after_edit(
    modified_bento: pl.DataFrame,
    modified_index: int,
    all_bentos: list[pl.DataFrame | None],
    locked_bentos: list[bool],
    energy: float,
    proteins_daily: float,
    protein_per_bento: float,
    fat: float,
    carbs: float,
    meal_fractions: list[float],
    product_pools: list[pl.DataFrame],
    allow_one_animal_list: list[bool],
    portion_legumes: float,
    *,
    ingredient_locks: list[list[bool]],
    catalog_by_bento: list[pl.DataFrame],
    extra_locks: dict[int, list[int]] | None = None,
) -> list[pl.DataFrame | None]:
    """Rééquilibre après édition : quantités proportionnelles sur lignes déverrouillées ; réoptimisation
    seulement si aucun ingrédient du bento n’est verrouillé (sans retirer les lignes bloquées ailleurs)."""
    n = len(all_bentos)
    result: list[pl.DataFrame | None] = []
    for b in all_bentos:
        if b is not None and len(b) > 0:
            result.append(b.clone())
        else:
            result.append(None)
    result[modified_index] = modified_bento.clone() if len(modified_bento) > 0 else modified_bento

    for bi in range(n):
        sync_ingredient_locks_row_count(ingredient_locks, bi, result[bi])

    # Build effective_locks: ingredient_locks + extra_locks (never written back to session_state)
    effective_locks: list[list[bool]] = []
    for _bi in range(n):
        _base = list(ingredient_locks[_bi]) if _bi < len(ingredient_locks) else []
        if extra_locks and _bi in extra_locks:
            for _r in extra_locks[_bi]:
                while len(_base) <= _r:
                    _base.append(False)
                _base[_r] = True
        effective_locks.append(_base)

    # ── Étape 0 : compenser d'abord dans le bento modifié lui-même ────────────────────────────
    # Les lignes déverrouillées du bento édité sont rescalées en priorité sur les
    # sources de protéines (macro-aware) puis les autres lignes pour l'énergie.
    _mod_frac = meal_fractions[modified_index] if modified_index < len(meal_fractions) else 1.0 / max(n, 1)
    _tgt_mod_energy = energy * _mod_frac
    _tgt_mod_protein = protein_per_bento
    _tgt_mod_fat = fat * _mod_frac
    _tgt_mod_carbs = carbs * _mod_frac
    _mod_lk = effective_locks[modified_index] if modified_index < len(effective_locks) else []
    _mod_df = result[modified_index]
    _mod_n = len(_mod_df) if _mod_df is not None else 0
    _mod_has_unlocked = any(not ln for ln in (_mod_lk[:_mod_n] if _mod_n else []))

    if _mod_has_unlocked and _mod_df is not None and _mod_n > 0:
        _cat_mod = catalog_by_bento[modified_index] if modified_index < len(catalog_by_bento) else catalog_by_bento[0]
        _before_q = _mod_df["Quantité (g)"].to_list()
        _edited_rows = extra_locks.get(modified_index, []) if extra_locks else []
        result[modified_index] = _scale_bento_unlocked_macro_aware(
            _mod_df, _mod_lk, _tgt_mod_energy, _tgt_mod_protein, _cat_mod, tolerance=0.005,
            edited_row_indices=_edited_rows,
            target_meal_fat=_tgt_mod_fat,
            target_meal_carbs=_tgt_mod_carbs,
        )
        _after_q = result[modified_index]["Quantité (g)"].to_list()
        _lbl_e2 = NUTR_DISPLAY["energy-kcal"]
        _e_after = float(result[modified_index][_lbl_e2].sum())

        sync_ingredient_locks_row_count(ingredient_locks, modified_index, result[modified_index])
    full_daily = full_daily_targets_dict(energy, proteins_daily, fat, carbs)

    def totals_sum(bentos: list[pl.DataFrame | None]) -> dict[str, float]:
        return daily_totals_from_bentos(bentos)

    current = totals_sum(result)
    deficit = {nc: full_daily[nc] - current[nc] for nc in NUTR_COLS}

    def has_unlocked_row_bi(i: int) -> bool:
        b = result[i]
        if b is None or len(b) == 0:
            return False
        lk = effective_locks[i][: len(b)] if i < len(effective_locks) else []
        return any(not ln for ln in lk)

    unlocked_indices = [
        i
        for i in range(n)
        if i != modified_index and has_unlocked_row_bi(i) and result[i] is not None and len(result[i]) > 0
    ]

    if not unlocked_indices:
        locked_indices = [
            i for i in range(n) if i != modified_index and result[i] is not None and len(result[i]) > 0
        ]
        targets_indices = locked_indices
    else:
        targets_indices = unlocked_indices

    if not targets_indices:
        return result

    per_bento_adjust = {nc: deficit[nc] / len(targets_indices) for nc in NUTR_COLS}

    for i in targets_indices:
        b = result[i]
        if b is None or len(b) == 0:
            continue
        lk = effective_locks[i] if i < len(effective_locks) else []

        tgt_e = float(_bento_totals(b).get("energy-kcal", 0.0)) + per_bento_adjust["energy-kcal"]
        tgt_e = max(50.0, tgt_e)

        catalog = catalog_by_bento[i] if i < len(catalog_by_bento) else catalog_by_bento[0]
        scaled = _scale_bento_unlocked_to_energy(b, lk, tgt_e, catalog)
        result[i] = scaled
        sync_ingredient_locks_row_count(ingredient_locks, i, result[i])

    current2 = totals_sum(result)
    still_off = not daily_targets_respected(current2, full_daily, rel_tol=0.12)

    def all_rows_unlocked(i: int) -> bool:
        b = result[i]
        if b is None or len(b) == 0:
            return True
        lk = (effective_locks[i] if i < len(effective_locks) else [])[: len(b)]
        if len(lk) < len(b):
            lk = lk + [False] * (len(b) - len(lk))
        return len(lk) == len(b) and not any(lk)

    if still_off:
        for i in targets_indices:
            if result[i] is None or len(result[i]) == 0:
                continue
            if not all_rows_unlocked(i):
                continue
            pool_raw = product_pools[i] if i < len(product_pools) else product_pools[0]
            if len(pool_raw) == 0:
                continue
            names_pref = (
                [str(x) for x in result[i]["Aliment"].to_list()] if "Aliment" in result[i].columns else None
            )
            pool = _product_pool_cap_for_reoptimize(pool_raw, prefer_product_names=names_pref)
            reopt = optimize_bento(
                pool,
                [energy, protein_per_bento, fat, carbs],
                meal_fractions[i],
                portion_legumes,
                allow_one_animal=allow_one_animal_list[i],
            )
            if reopt is not None and len(reopt) > 0:
                result[i] = reopt
                ingredient_locks[i] = [False] * len(reopt)
                sync_ingredient_locks_row_count(ingredient_locks, i, result[i])

    current3 = totals_sum(result)
    if daily_targets_respected(current3, full_daily, rel_tol=0.12):
        return result

    fallback = [
        i
        for i in range(n)
        if locked_bentos[i] and i != modified_index and result[i] is not None and len(result[i]) > 0
    ]
    for i in fallback:
        if not all_rows_unlocked(i):
            continue
        pool_raw = product_pools[i] if i < len(product_pools) else product_pools[0]
        if len(pool_raw) == 0:
            continue
        rp = result[i]
        names_pref = [str(x) for x in rp["Aliment"].to_list()] if rp is not None and "Aliment" in rp.columns else None
        pool = _product_pool_cap_for_reoptimize(pool_raw, prefer_product_names=names_pref)
        reopt = optimize_bento(
            pool,
            [energy, protein_per_bento, fat, carbs],
            meal_fractions[i],
            portion_legumes,
            allow_one_animal=allow_one_animal_list[i],
        )
        if reopt is not None and len(reopt) > 0:
            result[i] = reopt
            ingredient_locks[i] = [False] * len(reopt)
            sync_ingredient_locks_row_count(ingredient_locks, i, result[i])
            if daily_targets_respected(totals_sum(result), full_daily, rel_tol=0.15):
                break

    return result


def run_daily_rebalance_after_anchor(
    *,
    anchor_bento_idx: int,
    all_frames: list[pl.DataFrame | None],
    ingredient_locks: list[list[bool]],
    catalog_by_bento: list[pl.DataFrame],
    energy: float,
    proteins_daily: float,
    protein_per_bento: float,
    fat: float,
    carbs: float,
    meal_fractions: list[float],
    allow_one_animal_list: list[bool],
    portion_legumes: float,
) -> list[pl.DataFrame | None]:
    """Recalcule les autres repas pour coller aux objectifs journaliers (lignes verrouillées respectées).

    Les réoptimisations complètes passent par ``optimize_bento`` sur un catalogue **plafonné** (voir
    ``_REOPT_POOL_MAX_ROWS``) pour éviter des calculs de plusieurs minutes sur le catalogue entier.
    """
    n = len(all_frames)
    if anchor_bento_idx < 0 or anchor_bento_idx >= n:
        return all_frames
    mb = all_frames[anchor_bento_idx]
    if mb is None or len(mb) == 0:
        return all_frames
    for bi in range(n):
        sync_ingredient_locks_row_count(ingredient_locks, bi, all_frames[bi])

    # Load manually-changed row indices (stored by render_bento_inline_table).
    # Passed as extra_locks so _adjust_bentos_after_edit treats them as locked
    # during scaling WITHOUT ever writing to ingredient_locks in session_state.
    extra_locks: dict[int, list[int]] = st.session_state.pop("_rebalance_locked_rows", {})

    sync_locked_bentos_from_ingredient_locks(n)
    lb = list(st.session_state.locked_bentos)
    return _adjust_bentos_after_edit(
        modified_bento=mb.clone(),
        modified_index=anchor_bento_idx,
        all_bentos=list(all_frames),
        locked_bentos=lb,
        energy=energy,
        proteins_daily=proteins_daily,
        protein_per_bento=protein_per_bento,
        fat=fat,
        carbs=carbs,
        meal_fractions=meal_fractions,
        product_pools=catalog_by_bento,
        allow_one_animal_list=allow_one_animal_list,
        portion_legumes=portion_legumes,
        ingredient_locks=ingredient_locks,
        catalog_by_bento=catalog_by_bento,
        extra_locks=extra_locks,
    )


def _clear_editor_widget_state(bento_index: int) -> None:
    for key in (
        f"_be_search_{bento_index}",
        f"_be_edit_{bento_index}",
        f"_be_pending_{bento_index}",
    ):
        st.session_state.pop(key, None)
    for nk in list(st.session_state.keys()):
        if isinstance(nk, str) and nk.startswith("_be_qty_"):
            suf = nk.split("_be_qty_", 1)[-1]
            parts = suf.split("_")
            if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == bento_index:
                st.session_state.pop(nk, None)
        elif isinstance(nk, str) and nk.startswith(f"_be_del_{bento_index}_"):
            st.session_state.pop(nk, None)


# ─── Composants Streamlit ─────────────────────────────────────────────────────


def render_bento_editor(
    bento_index: int,
    bento_df: pl.DataFrame | None,
    all_bentos: list[pl.DataFrame | None],
    catalog_df: pl.DataFrame,
    bento_macro_targets: dict[str, float],
    full_daily: dict[str, float],
    portion_legumes: float,
    locked_bentos: list[bool],
    product_pools: list[pl.DataFrame],
    meal_fractions: list[float],
    allow_one_animal_list: list[bool],
    energy: float,
    proteins_daily: float,
    protein_per_bento: float,
    fat: float,
    carbs: float,
    on_update: Callable[[list[pl.DataFrame | None]], None] | None = None,
) -> None:
    inject_kojin_bento_inline_css_once()
    bento_name = BENTO_NAMES[bento_index] if bento_index < len(BENTO_NAMES) else f"Bento {bento_index + 1}"
    search_key = f"_be_search_{bento_index}"
    edit_key = f"_be_edit_{bento_index}"
    pending_key = f"_be_pending_{bento_index}"

    if search_key not in st.session_state:
        st.session_state[search_key] = ""
    if edit_key not in st.session_state:
        if bento_df is not None and len(bento_df) > 0:
            st.session_state[edit_key] = bento_df.to_dicts()
        else:
            st.session_state[edit_key] = []
    if pending_key not in st.session_state:
        st.session_state[pending_key] = None

    edit_rows: list[dict] = st.session_state[edit_key]

    def current_totals() -> dict[str, float]:
        t: dict[str, float] = {nc: 0.0 for nc in NUTR_COLS}
        for row in edit_rows:
            for nc, label in NUTR_DISPLAY.items():
                t[nc] += float(row.get(label, 0.0) or 0.0)
        return t

    totals = current_totals()

    st.markdown(f'<div class="bento-header">✎ Édition — {bento_name}</div>', unsafe_allow_html=True)
    tgt_lines = " · ".join(
        f"<strong>{int(bento_macro_targets[nc])}</strong> {NUTR_DISPLAY[nc]}" for nc in NUTR_COLS
    )
    st.markdown(
        f'<div class="targets-box">Objectif bento : {tgt_lines}</div>',
        unsafe_allow_html=True,
    )

    if edit_rows:
        st.markdown("##### Ingrédients actuels")
        for idx, row in enumerate(edit_rows):
            cols = st.columns([3, 1.5, 1])
            with cols[0]:
                al_disp = html.escape(format_product_display_label(str(row["Aliment"])))
                st.markdown(
                    f'<span class="kojin-aliment-editor-line">{al_disp}</span>',
                    unsafe_allow_html=True,
                )
            with cols[1]:
                new_qty = st.number_input(
                    "g",
                    min_value=float(MIN_QTY_G),
                    max_value=float(MAX_QTY_G),
                    value=_qty_in_widget_bounds(row["Quantité (g)"]),
                    step=1.0,
                    key=f"_be_qty_{bento_index}_{idx}",
                    label_visibility="collapsed",
                )
                if new_qty != float(row["Quantité (g)"]):
                    product_match = catalog_rows_matching_name(catalog_df, str(row["Aliment"]))
                    if len(product_match) > 0:
                        prow = product_match.row(0, named=True)
                        new_nutr = _recompute_row_nutrition(prow, new_qty)
                        edit_rows[idx] = {
                            "Aliment": row["Aliment"],
                            "Quantité (g)": round(new_qty, 1),
                            **new_nutr,
                        }
                    else:
                        factor = new_qty / max(float(row["Quantité (g)"]), 1e-6)
                        edit_rows[idx] = {
                            k: round(v * factor, 1) if k not in ("Aliment",) else v
                            for k, v in row.items()
                        }
                        edit_rows[idx]["Quantité (g)"] = round(new_qty, 1)
                    st.session_state[edit_key] = edit_rows
                    st.rerun()
            with cols[2]:
                if st.button("✕", key=f"_be_del_{bento_index}_{idx}", help="Retirer cet aliment"):
                    edit_rows.pop(idx)
                    st.session_state[edit_key] = edit_rows
                    st.rerun()

        totals = current_totals()
        _render_nutrition_bar(totals, bento_macro_targets)
    else:
        st.info("Aucun ingrédient dans ce bento. Recherchez un produit ci-dessous pour en ajouter.")

    st.divider()
    st.markdown("##### Ajouter un ingrédient")

    search_val = st.text_input(
        "Rechercher un produit…",
        value=st.session_state[search_key],
        key=f"_be_text_{bento_index}",
        placeholder="ex : poulet, quinoa, tofu…",
    )
    st.session_state[search_key] = search_val

    if search_val and len(search_val.strip()) >= 2:
        results = search_products_for_pattern(catalog_df, search_val)
        if len(results) > 0:
            st.caption(f"{len(results)} produit(s) — cliquez pour sélectionner")
            try:
                add_h = min(320, 28 + min(len(results), 12) * 34)
                add_cont = st.container(height=int(add_h))
            except TypeError:
                add_cont = st.container()
            with add_cont:
                for ai, hrow in enumerate(results.to_dicts()):
                    pname = str(hrow.get("product_name", ""))
                    display_name = format_product_display_label(pname)
                    lbl = (display_name[:110] + "…") if len(display_name) > 110 else display_name
                    kcal_txt = html.escape(_kcal_per100_from_catalog_row(hrow))
                    c_n, c_k = st.columns([5, 2])
                    with c_n:
                        if st.button(lbl, key=f"_be_hit_{bento_index}_{ai}", use_container_width=True):
                            st.session_state[pending_key] = pname
                            st.rerun()
                    with c_k:
                        st.markdown(
                            f'<div class="kojin-hit-kcal">{kcal_txt}</div>',
                            unsafe_allow_html=True,
                        )
        else:
            st.caption("Aucun résultat pour ce pattern.")
    else:
        st.caption("Tapez au moins 2 caractères pour lancer la recherche.")

    if st.session_state[pending_key]:
        pending_name = st.session_state[pending_key]
        prow_df = catalog_rows_matching_name(catalog_df, pending_name)
        if len(prow_df) > 0:
            prow = prow_df.row(0, named=True)
            add_col1, add_col2 = st.columns([2, 1])
            with add_col1:
                qty_add = st.number_input(
                    f"Quantité pour « {format_product_display_label(pending_name)[:40]}{'…' if len(pending_name) > 40 else ''} » (g)",
                    min_value=float(MIN_QTY_G),
                    max_value=float(MAX_QTY_G),
                    value=100.0,
                    step=10.0,
                    key=f"_be_qty_add_{bento_index}",
                )
            with add_col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("＋ Ajouter", key=f"_be_add_{bento_index}", type="primary"):
                    already = [r["Aliment"] for r in edit_rows]
                    if pending_name in already:
                        for r in edit_rows:
                            if r["Aliment"] == pending_name:
                                new_qty = float(r["Quantité (g)"]) + qty_add
                                new_qty = min(new_qty, MAX_QTY_G)
                                new_nutr = _recompute_row_nutrition(prow, new_qty)
                                r.clear()
                                r.update({"Aliment": pending_name, "Quantité (g)": round(new_qty, 1), **new_nutr})
                    else:
                        new_nutr = _recompute_row_nutrition(prow, qty_add)
                        edit_rows.append(
                            {"Aliment": pending_name, "Quantité (g)": round(qty_add, 1), **new_nutr}
                        )
                    st.session_state[edit_key] = edit_rows
                    st.session_state[pending_key] = None
                    st.session_state[search_key] = ""
                    st.rerun()

    st.divider()
    totals = current_totals()
    _render_target_validation(totals, bento_macro_targets)

    btn_col1, btn_col2 = st.columns(2)
    edit_key_snapshot = edit_key

    with btn_col1:
        if st.button(
            "Ajuster ce bento",
            key=f"_be_apply_{bento_index}",
            type="primary",
            help="Enregistre ce bento seulement si les totaux journaliers restent dans les objectifs.",
        ):
            new_bento_df = _edit_rows_to_df(edit_rows)
            new_all = list(all_bentos)
            new_all[bento_index] = new_bento_df if len(new_bento_df) > 0 else None
            merged = daily_totals_from_bentos(new_all)
            if not daily_targets_respected(merged, full_daily, DAILY_MATCH_REL_TOL):
                st.error(
                    "Les totaux journaliers (kcal, macros, fibres) sortent des objectifs. "
                    "Ajustez les quantités ou utilisez « Ajuster les autres bentos »."
                )
            else:
                _clear_editor_widget_state(bento_index)
                st.session_state.pop(edit_key_snapshot, None)
                if on_update:
                    on_update(new_all)
                st.success(f"✓ {bento_name} mis à jour.")
                st.rerun()

    with btn_col2:
        if st.button(
            "Ajuster les autres bentos",
            key=f"_be_apply_others_{bento_index}",
            type="secondary",
            help="Enregistre ce bento et rééquilibre les autres (quantités puis optimiseur si besoin).",
        ):
            new_bento_df = _edit_rows_to_df(edit_rows)
            il_ad = st.session_state.setdefault("ingredient_locks", [])
            while len(il_ad) <= bento_index:
                il_ad.append([])
            for bi in range(len(all_bentos)):
                sync_ingredient_locks_row_count(il_ad, bi, all_bentos[bi])
            sync_locked_bentos_from_ingredient_locks(len(all_bentos))
            new_all = _adjust_bentos_after_edit(
                modified_bento=new_bento_df,
                modified_index=bento_index,
                all_bentos=all_bentos,
                locked_bentos=list(st.session_state.locked_bentos),
                energy=energy,
                proteins_daily=proteins_daily,
                protein_per_bento=protein_per_bento,
                fat=fat,
                carbs=carbs,
                meal_fractions=meal_fractions,
                product_pools=product_pools,
                allow_one_animal_list=allow_one_animal_list,
                portion_legumes=portion_legumes,
                ingredient_locks=il_ad,
                catalog_by_bento=product_pools,
            )
            _clear_editor_widget_state(bento_index)
            st.session_state.pop(edit_key_snapshot, None)
            if on_update:
                on_update(new_all)
            st.success("✓ Bentos rééquilibrés.")
            st.rerun()

    locked = locked_bentos[bento_index] if bento_index < len(locked_bentos) else False
    st.caption(
        "🔒 Ce bento est **verrouillé** — ajustement automatique en dernier recours seulement."
        if locked
        else "🔓 Ce bento est **libre** — peut être rééquilibré automatiquement."
    )


def render_lock_panel(
    locked_bentos: list[bool],
    on_change: Callable[[list[bool]], None],
) -> None:
    st.markdown('<div class="bento-header">Verrouillage des bentos</div>', unsafe_allow_html=True)
    st.caption(
        "Les bentos verrouillés ne sont modifiés qu’en dernier recours pour respecter les objectifs journaliers."
    )
    cols = st.columns(len(locked_bentos))
    new_locked = list(locked_bentos)
    for i, col in enumerate(cols):
        bnm = BENTO_NAMES[i] if i < len(BENTO_NAMES) else f"Bento {i + 1}"
        with col:
            val = st.checkbox(
                f"🔒 {bnm}",
                value=locked_bentos[i],
                key=f"_lock_bento_{i}",
            )
            new_locked[i] = val
    if all(new_locked):
        st.warning("Tous les bentos sont verrouillés — le dernier sera traité comme déverrouillé.")
        new_locked[-1] = False
    if new_locked != locked_bentos:
        on_change(new_locked)


def render_all_bento_editors(
    all_bentos: list[pl.DataFrame | None],
    catalog_by_bento: list[pl.DataFrame],
    energy: float,
    proteins_daily: float,
    protein_per_bento: float,
    fat: float,
    carbs: float,
    portion_legumes: float,
    meal_fractions: list[float],
    allow_one_animal_per_bento: list[bool],
    locked_bentos: list[bool],
    on_update: Callable[[list[pl.DataFrame | None]], None],
    on_lock_change: Callable[[list[bool]], None],
    *,
    show_lock_panel: bool = True,
) -> None:
    full_daily = full_daily_targets_dict(energy, proteins_daily, fat, carbs)
    if show_lock_panel:
        render_lock_panel(locked_bentos, on_lock_change)
        st.divider()
    n = len(all_bentos)
    tab_labels = [
        (BENTO_NAMES[i] if i < len(BENTO_NAMES) else f"Bento {i + 1}") + (" 🔒" if locked_bentos[i] else "")
        for i in range(n)
    ]
    tabs = st.tabs(tab_labels)
    product_pools = catalog_by_bento
    for i, tab in enumerate(tabs):
        with tab:
            bento_tgt = per_bento_macro_targets(energy, fat, carbs, meal_fractions[i], protein_per_bento)
            render_bento_editor(
                bento_index=i,
                bento_df=all_bentos[i],
                all_bentos=all_bentos,
                catalog_df=catalog_by_bento[i],
                bento_macro_targets=bento_tgt,
                full_daily=full_daily,
                portion_legumes=portion_legumes,
                locked_bentos=locked_bentos,
                product_pools=product_pools,
                meal_fractions=meal_fractions,
                allow_one_animal_list=allow_one_animal_per_bento,
                energy=energy,
                proteins_daily=proteins_daily,
                protein_per_bento=protein_per_bento,
                fat=fat,
                carbs=carbs,
                on_update=on_update,
            )


def _edit_rows_to_df(edit_rows: list[dict]) -> pl.DataFrame:
    if not edit_rows:
        return pl.DataFrame(
            schema={
                "Aliment": pl.String,
                "Quantité (g)": pl.Float64,
                **{label: pl.Float64 for label in NUTR_DISPLAY.values()},
            }
        )
    return pl.DataFrame(edit_rows).sort("Quantité (g)", descending=True)


def _render_nutrition_bar(totals: dict[str, float], targets: dict[str, float]) -> None:
    cols = st.columns(len(NUTR_COLS))
    for col, nc in zip(cols, NUTR_COLS):
        label = NUTR_DISPLAY[nc]
        val = totals.get(nc, 0.0)
        tgt = targets.get(nc, 0.0)
        ratio = val / tgt if tgt > 0 else 0.0
        color = "#c44" if ratio > 1.12 else ("#888" if ratio >= 0.88 else "#3a3")
        with col:
            st.markdown(
                f"""
                <div style="text-align:center">
                    <div style="font-size:1.1rem;font-weight:500;color:{color}">{int(val)}</div>
                    <div style="font-size:0.7rem;color:#888">{label}</div>
                    <div style="font-size:0.7rem;color:#aaa">/ {int(tgt)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_target_validation(totals: dict[str, float], targets: dict[str, float]) -> None:
    energy_val = totals.get("energy-kcal", 0.0)
    energy_tgt = targets.get("energy-kcal", 1.0)
    ratio = energy_val / energy_tgt if energy_tgt > 0 else 0.0
    if 0.88 <= ratio <= 1.12:
        st.markdown(
            '<div style="color:#363;font-size:0.82rem">✓ Apport énergétique du bento proche de la cible (±12 %).</div>',
            unsafe_allow_html=True,
        )
    elif ratio > 1.12:
        st.markdown(
            f'<div style="color:#c44;font-size:0.82rem">⚠ Excès d’environ <strong>+{int(energy_val - energy_tgt)} kcal</strong> vs objectif du bento.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="color:#a84;font-size:0.82rem">⚠ Déficit d’environ <strong>{int(energy_tgt - energy_val)} kcal</strong> vs objectif du bento.</div>',
            unsafe_allow_html=True,
        )


def sync_locked_bentos_from_inline_widgets(n_comp: int) -> None:
    """Alias : verrouillage bento dérivé des cadenas ingrédients."""
    sync_locked_bentos_from_ingredient_locks(n_comp)


def clear_streamlit_editor_keys(num_bentos: int | None = None) -> None:
    """À appeler après une nouvelle composition pour repartir d’un éditeur propre."""
    # Increment generation counters BEFORE clearing widget keys, so widgets
    # rendered later in the same script run use fresh keys without stale values.
    _max_b = num_bentos if num_bentos is not None else 10
    for _bi in range(_max_b):
        gk = f"_qty_gen_{_bi}"
        st.session_state[gk] = st.session_state.get(gk, 0) + 1

    for k in list(st.session_state.keys()):
        if not isinstance(k, str):
            continue
        if (
            k.startswith("_be_")
            or k.startswith("_lock_bento_")
            or k.startswith("bento_inline_lock_")
            or k.startswith("bento_qty_de_")
            or k.startswith("bento_rep_row_")
            or k.startswith("bento_rep_pat_")
            or k.startswith("bento_hit_")
            or k.startswith("bento_rep_apply_")
            or k.startswith("bento_rep_cancel_")
            or k.startswith("bento_gate_")
            or k.startswith("ing_gate_")
            or k == "_rebalance_anchor_bento"
            or k == "_replace_lock_mark"
            or k == "_pending_bento_row_replace"
            or k.startswith("bento_pick_alim_")
            or k.startswith("qty_inline_")
        ):
            st.session_state.pop(k, None)
