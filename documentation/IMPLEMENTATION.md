# Kōjin — Implementation Walkthrough

This document describes how each subsystem of the Kōjin project is implemented,
ordered roughly along the data‑flow: from raw Open Food Facts dumps all the way
to the reinforcement‑learning meal planner.

---

## 1. Data preparation — [data_prep_nutriments.py](../data_prep_nutriments.py)

Goal: turn the raw Open Food Facts (OFF) Parquet dump into a curated CSV
(`data/products_names_with_macro_nutriments.csv`) that the app, the SQL agent
and the RL pipeline all consume.

### 1.1 Source ingestion
- `download_data(force_download: bool)` pulls the OFF Parquet from
  Hugging Face Hub (`hf_hub_download`).
- `expr_france_not_obsolete()` keeps only items whose `countries_tags`
  contains `en:france` and where `obsolete == False`.
- `expr_product_name_french()` selects the French label after exploding the
  multilingual `product_name` struct.

### 1.2 Nutrient extraction — `get_nutriments`
- Flattens the `nutriments` map into typed columns: `energy-kcal`, `proteins`,
  `fat`, `carbohydrates`, `fiber`, sugars, salt, sat. fat (all per 100 g).
- Drops rows with no usable macro information (kcal + at least one macro).

### 1.3 Category cleanup — `clean_categories`
- Lowercases and strips diacritics on `categories_tags`.
- Removes regional/quality prefixes (`en:`, `fr:`) so the regex filter is
  language‑agnostic.

### 1.4 Tag derivation — `add_tags`
- Computes boolean columns used everywhere downstream:
  `vegan`, `vegetarian`, `halal`, `kascher`, `bio`, `meat`, `fish`,
  `lait`, `no_palm_oil`, `gluten_free`.
- Each flag is a regex/keyword test against `categories`, `labels` and
  `ingredients_text`.

### 1.5 Catalogue filtering — `filter_products_catalog`
The single source of truth for *what counts as a usable food product*. The
`EXCLUDED_CATEGORIES` regex (see top of file) removes alcohols, sugary drinks,
candy, supplements, ready meals, pizzas, snacks, ice cream, etc. — every
category that would derail the optimiser.

### 1.6 Output
- A single CSV with one row per French, non‑obsolete, non‑excluded product
  carrying macros + tag flags.
- Written to `CSV_PATH` (defined in [kojin_common.py](../kojin_common.py)).

### 1.7 DuckDB materialisation
The CSV is materialised into `data/products.duckdb` on demand (see
`_ensure_duckdb` in [app_pages/exploration.py](../app_pages/exploration.py)
and [reciperl/recipe_prompts.py](../reciperl/recipe_prompts.py)) so the SQL
agent and the RL ingredient resolver query a fast columnar store.

---

## 2. Shared core — [kojin_common.py](../kojin_common.py)

Single import surface for every page.

### 2.1 Paths & S3 hydration
- `DATA_DIR`, `CSV_PATH`, `DUCKDB_PATH`, `DUCKDB_TABLE` are exported once and
  reused everywhere.
- `ensure_csv_from_s3()` downloads the catalogue from S3 (`DATA_S3_URI`) when
  the file is missing — used in ECS / Streamlit Cloud deployments.

### 2.2 Domain constants
- `BENTO_NAMES`, `DAILY_ACTIVITY`, `GOALS`, `NUTR_COLS`, `NUTR_DISPLAY`,
  regime list — frozen vocabularies used by the UI, the LLM prompts and the
  RL recipe catalogue.

### 2.3 Theme
- `apply_theme()` injects a Japanese‑minimalist CSS bundle (custom variables,
  sidebar, typography, sliders, tables, buttons, expanders, scrollbar).

### 2.4 Bento optimiser — `optimize_bento`
The numerical heart of the app.

- **Inputs**: a Polars `DataFrame` of candidate products, the user's daily
  macro targets `[kcal, prot, fat, carb]`, the meal's `meal_fraction`
  (share of daily kcal/fat/carb — protein is treated as absolute), the
  legume quota, and an `allow_one_animal` flag.
- **Targets vector**: `targets = [kcal·f, prot, fat·f, carb·f, 25·f]`, with
  `25` being a target fibre level; the legume quota is appended as a 6th
  row when active.
- **Constraint matrix**: per‑100 g nutrient values of every candidate row,
  divided by 100 to express grams.
- **Solver**: `_run_solver` first tries `scipy.optimize.nnls`, falls back to
  `lsq_linear` (BVLS) with per‑product upper bounds (`portion_maximale`).
  Solutions below `1e‑6` g are dropped.
- **Post‑processing**: products are bucketed into animal / oil / others;
  one animal item (when allowed) and one oil are kept, plus everything in
  the "other" bucket. Output is a Polars `DataFrame` sorted by grams.

### 2.5 LLM factory — `get_chat_llm`
- Provider‑agnostic LangChain factory.
- `LLM_PROVIDER=bedrock` (default in prod) returns a `ChatBedrock` wrapping
  the fine‑tuned Llama model (`BEDROCK_MODEL_ID`).
- `LLM_PROVIDER=groq` or auto‑fallback returns an OpenAI‑compatible
  `ChatOpenAI` pointing at Groq's HTTP endpoint (`GROQ_API_KEY`).
- `get_compare_sql_chat_llm()` exposes a second model for side‑by‑side
  comparison on the Exploration page.

---

## 3. NL → SQL fine‑tuning — [finetuning/](../finetuning/)

### 3.1 Dataset generation — [finetuning/generate_dataset.py](../finetuning/generate_dataset.py)
- Emits JSONL files in Bedrock Converse format
  (`system` / `user` / `assistant` messages).
- The system message embeds the strict instruction set ("answer with ONE
  SQL query, no markdown, no prose, table is `products`, schema is …").
- Generates diverse pairs covering: macro filters, boolean tags, sort/rank,
  aggregations (`AVG`, `COUNT`, `SUM`, `MIN`, `MAX`), combined predicates
  and `ILIKE` text search on `product_name` / `categories`.
- Writes `finetuning/data/train.jsonl` and `eval.jsonl`.

### 3.2 Training — [finetuning/launch_finetune.py](../finetuning/launch_finetune.py)
- Uploads the JSONL files to S3.
- Calls `bedrock:CreateModelCustomizationJob` on
  `meta.llama3-1-8b-instruct-v1:0` (configurable).
- Polls the job until completion and writes the resulting custom‑model ARN
  to `finetuning/data/last_job.json`.
- IAM is enforced through the provided trust policies
  (`trust-bedrock-finetune.json`, `bedrock-finetune-policy.json`,
  `s3-read-policy.json`).

### 3.3 Evaluation — [finetuning/evaluate.py](../finetuning/evaluate.py)
- Runs every question of `eval.jsonl` through both the fine‑tuned model and
  a reference model.
- Executes the produced SQL on DuckDB and compares: parse success, execution
  success, row‑set equality.
- Aggregated metrics + per‑question diffs are written to
  `finetuning/data/eval_results.json`.

---

## 4. Streamlit application — [streamlit_app.py](../streamlit_app.py) + [app_pages/](../app_pages/)

### 4.1 Entry point
- `streamlit_app.py` declares a multipage layout via
  `st.navigation([...])` and dispatches to:
  - **Bento Maker** (default)
  - **Exploration** (NL→SQL playground)
  - **RL Planner** (Phase C, optional)

### 4.2 Bento Maker — [app_pages/bento_maker.py](../app_pages/bento_maker.py) + [bento_editor.py](../bento_editor.py)
- Sidebar collects: weight, activity, goal, daily macro targets, regime,
  number of bentos, anchor bento (ratios per meal).
- Loads the catalogue once via `pl.read_csv(CSV_PATH, ...)`.
- For each bento slot, calls `optimize_bento(...)` with the slot's
  `meal_fraction`, then renders an editable table (`render_bento_inline_table`).
- Edits propagate through `sync_locked_bentos_from_ingredient_locks` →
  `run_daily_rebalance_after_anchor` so locking an ingredient in one
  bento re‑optimises the others.

### 4.3 Exploration — [app_pages/exploration.py](../app_pages/exploration.py)
- Builds a `ChatPromptTemplate` (`system` + `human`) wired to
  `get_chat_llm()`.
- `_clean_sql` strips code fences, normalises the `"energy-kcal"` quoting
  and trims trailing semicolons.
- `_execute_sql` runs the query against `DUCKDB_PATH` in read‑only mode and
  returns a Polars `DataFrame` for `st.dataframe`.
- A "compare" toggle calls `get_compare_sql_chat_llm()` to display
  side‑by‑side outputs and writes telemetry through
  `append_exploration_metrics_event`.

---

## 5. RecipeRL — [reciperl/](../reciperl/)

Implementation of *Liu et al., "An interactive food recommendation system
using reinforcement learning", ESWA 254 (2024)*, adapted to operate at recipe
granularity so it can drive the Kōjin app.

### 5.1 Configuration — [reciperl/config.py](../reciperl/config.py)
A single `@dataclass RecipeRLConfig` holds every hyperparameter:
- catalogue mode (`use_recipes`, `checkpoint_path`),
- NCF (`cf_embed_dim`, `cf_hidden`, `cf_epochs`),
- state (`window_k`, `attn_heads`, `gaussian_noise_std`, `alpha_sequential`),
- PPO (`horizon_T`, `gamma`, `clip_eps`, `ppo_epochs`, `total_updates`,
  `rollout_episodes`),
- data synthesis (`num_synthetic_users`, `ratings_per_user`),
- runtime (`seed`, `device`).

### 5.2 Recipe catalogue — [data/recipes.json](../data/recipes.json) + [reciperl/recipes.py](../reciperl/recipes.py)
- 30 hand‑seeded recipes covering 4 categories
  (`petit_dej`, `plat`, `snack`, `dessert`) and 6 cuisines.
- Each recipe carries: id, name, category, cuisine, regime tags,
  abstract ingredient list `[{name, approx_g, role}]`, per‑serving macros.
- `load_recipes()` validates contiguous ids and returns a
  `RecipeCatalogue` with helpers (`names`, `category_ids`,
  `filter_by_tag`).

### 5.3 Dataset adapter — [reciperl/data.py](../reciperl/data.py)
- `load_dataset(cfg)` dispatches on `cfg.use_foodcom`:
  - **Food.com** (primary): delegates to `foodcom_data.load_foodcom_dataset()`
    which reads real recipes and user ratings from the Kaggle zip.
  - **hand-seeded fallback**: items = 30 recipes from `data/recipes.json`,
    categories = 4; ratings synthesised via `synthesize_ratings`.
- `synthesize_ratings` draws a per‑user Dirichlet preference over
  categories then samples `min(ratings_per_user, n_items)` items per user
  with Gaussian noise.
- A train/test split is performed at the *user* level.

### 5.3.1 Food.com dataset loader — [reciperl/foodcom_data.py](../reciperl/foodcom_data.py)
Parses the Kaggle Food.com zip (`data/foodcom-recipes-and-reviews.zip`) into
the structures the rest of the pipeline expects.
- `_parse_r_vector`: handles R‑style `c("a","b","c")` strings used by the CSV.
- `_parse_quantities` / `_quantities_to_proportions`: converts quantity strings
  like `"1"`, `"1/2"`, `"2 1/2"` to a float array then normalises to sum = 1.
  These proportions are stored as `RecipeIngredient.proportion` and later used
  as soft constraints in `optimize_bento`.
- `load_foodcom_recipes`: filters recipes with ≥ 1 ingredient and maps
  `RecipeCategory` strings to Kōjin categories via `_CATEGORY_RULES`.
- `load_foodcom_dataset`: additionally loads `reviews.csv`, filters to users
  with ≥ 20 ratings of 4–5, remaps ids to contiguous integers, and returns a
  `(RecipeRLDataset, RecipeCatalogue)` tuple.
- `catalogue_to_blob` / `catalogue_from_blob`: serialise the Food.com catalogue
  into the checkpoint so inference-time needs no access to the original CSVs.

### 5.4 NCF simulator — [reciperl/ncf.py](../reciperl/ncf.py)
Implements §3.2 of the paper.
- `NCF(nn.Module)`: user/item embeddings concatenated, fed through a
  3‑layer MLP regressing the rating (MSE loss).
- `train_ncf(dataset, cfg, device)` trains on the train split.
- `precompute_rating_matrix(...)` materialises ŷ for *every*
  (user, item) pair into a NumPy array — this acts as the offline
  simulator used by the RL environment (no need to query the network
  during rollouts).

### 5.5 Fused state — [reciperl/state.py](../reciperl/state.py)
Implements §3.3 (Eqs. 8–15) of the paper.
- `HistoricalMemory`: sliding window over the last `k` interactions.
  - Adds Gaussian noise during training only (`gaussian_noise_std`).
  - Forgetting score `λ_r = (1 − rec_counts.clamp_max(k)/k).clamp_min(0)`.
  - Probability mask `p_mask = (f_norm · λ_r).unsqueeze(-1)`.
  - Cross‑attention through `nn.MultiheadAttention(batch_first=True)`.
- `FusedState` concatenates three blocks:
  - `s_UI = pᵤ · wᵢ · qᵢ` — user / last‑item interaction (Eq. 12).
  - `s_ACH` — attention over the sliding window (Eq. 13).
  - `s_UC = pᵤ · wᵢ · c_{gᵢ}` — user / category coupling (Eq. 14).
- `out_dim = 3 · embed_dim` (e.g. 96 with `cf_embed_dim=32`).
- `warmstart_from_ncf` copies the trained NCF embeddings into the state
  module before PPO begins.

### 5.6 Policy / value network — [reciperl/policy.py](../reciperl/policy.py)
Implements the Residual PPO actor‑critic of §3.4 / Fig. 4.
- `_ResBlock` + `ResidualBackbone`: `Conv1d → ResBlock → Conv1d → Flatten → Linear`.
- `ActorCritic` exposes `forward` (returns a `Categorical` distribution and
  a value scalar) and `act(state, mask=None)` which samples an action and
  optionally masks invalid items with `logits.masked_fill(~mask, -inf)`.
  Masking is what lets `compose_day` constrain picks per meal category.

### 5.7 Environment — [reciperl/env.py](../reciperl/env.py)
- `reset(user, rng)`: cold‑starts by populating the sliding window with the
  user's top‑rated items from the simulator.
- `step(action)` looks up the simulator's rating for `(user, action)` and
  computes the paper's reward:
  - `empirical = (rating − 2.5) / 2.5`,
  - `c_p`/`c_n` track streaks of positive (≥ 3.0) / non‑positive ratings,
  - returns `empirical + α · (c_p − c_n)` (Eq. 6).
- `_observe()` builds the state via `FusedState(...)` with the current
  history, ratings, recommendation counts, last item and last category.

### 5.8 PPO trainer — [reciperl/ppo.py](../reciperl/ppo.py)
- `collect(rng)`: runs `cfg.rollout_episodes` episodes of horizon
  `cfg.horizon_T`, gathering states, actions, log‑probs, rewards, values
  and dones.
- `_gae`: standard GAE‑λ over rewards/values/dones.
- `update(roll)`: PPO‑clip objective —
  `ratio = exp(log_prob − old_log_prob)`,
  `surr1 = ratio · adv`,
  `surr2 = clamp(ratio, 1−ε, 1+ε) · adv`,
  `policy_loss = −min(surr1, surr2).mean()`,
  plus value MSE + entropy bonus. A single Adam optimises both the policy
  and the state module.

### 5.9 Evaluation — [reciperl/evaluate.py](../reciperl/evaluate.py)
- `eval_topk` runs deterministic greedy rollouts on held‑out users and
  reports Reward@T, Precision@T (hits where rating ≥
  `positive_threshold`) and NDCG@T (§4.2 of the paper).

### 5.10 Training CLI — [reciperl/train.py](../reciperl/train.py)
End‑to‑end driver:
1. Load dataset (Food.com or hand-seeded fallback).
2. Train NCF, materialise the rating matrix.
3. Build the fused state, warmstart from NCF.
4. Train the residual PPO policy for `cfg.total_updates` updates.
5. Evaluate on held‑out users.
6. `save_checkpoint(...)` writes
   `{config, state_module, policy, rating_matrix, item_names, recipe_catalogue}` to
   `data/reciperl.pt` so the app can reload without retraining.

Run:
```bash
python -m reciperl.train --foodcom --steps 200      # Food.com mode (primary)
python -m reciperl.train --steps 50                 # hand-seeded dev fallback
python -m reciperl.train --foodcom --max-recipes 500 --steps 200
```

---

## 6. App integration — recipe → ingredients → quantities

### 6.1 LLM ingredient resolver — [reciperl/recipe_prompts.py](../reciperl/recipe_prompts.py)
For each abstract ingredient slot of a chosen recipe (e.g.
`"chicken breast"` with `role=protein`):

- A strict French system prompt (mirrors `app_pages/exploration.py`) keeps
  the fine‑tuned model on rails: one SQL query, table = `products`, schema
  injected verbatim, default `LIMIT 25`, `"energy-kcal"` quoting enforced.
- `IngredientResolver`:
  - builds the prompt with `build_question(ingredient, regime, recipe_tags)`,
  - invokes the chain via `get_chat_llm(...)`,
  - cleans the answer (`_clean_sql`) and executes it against DuckDB in
    read‑only mode.
- `resolve_recipe(recipe, regime)` returns one `ResolvedIngredient` per
  ingredient slot, each holding the SQL, the regime context and the
  Polars `DataFrame` of candidate products.

### 6.2 Day composer — [reciperl/compose.py](../reciperl/compose.py)
The final glue between RL, LLM and optimiser.

1. `load_policy(path)` rebuilds `FusedState` + `ActorCritic` from the
   checkpoint and reloads the rating matrix used to cold‑start the
   user history.
2. `compose_day(loaded, user_id, daily_targets, meal_slots, resolver, regime)`
   iterates over the requested `MealSlot`s. For each slot it:
   - builds a boolean mask: recipes whose category matches the slot AND
     that haven't been picked today,
   - constructs the state vector from the rolling history
     (`history`, `history_ratings`, per‑window `rec_counts`, `last_item`,
     `last_cat`),
   - calls `policy.act(state, mask=...)` to sample a recipe id,
   - asks the resolver for candidate products per ingredient,
   - concatenates them with `pl.concat(..., how="diagonal_relaxed").unique()`,
   - calls `optimize_bento(products_df, daily_targets, slot.fraction,
     slot.portion_legumes)` to convert the recipe into concrete gram
     quantities.
3. The result is a `ComposedDay(meals=[ComposedMeal(...)])` where each
   meal carries the chosen `Recipe`, the SQL log, the candidate products
   and the solved `DataFrame`.

### 6.3 Sanity check
```bash
python -m reciperl.train --steps 3            # writes data/reciperl.pt
python - <<'PY'
from reciperl.compose import load_policy, compose_day, MealSlot
import numpy as np
loaded = load_policy('data/reciperl.pt')
day = compose_day(
    loaded, user_id=0,
    daily_targets=[2000, 120, 70, 250],
    meal_slots=[
        MealSlot('Petit-déjeuner', 'petit_dej', 0.25),
        MealSlot('Déjeuner',       'plat',      0.40),
        MealSlot('Dîner',          'plat',      0.30),
        MealSlot('Collation',      'snack',     0.05),
    ],
    resolver=None,
    rng=np.random.default_rng(0),
)
for m in day.meals:
    print(m.slot.label, '->', m.recipe.name)
PY
```

Expected output: one recipe per slot, no duplicates, categories matching
the slot — confirms the masked policy + history rolling + checkpoint
loading all behave.

---

## 7. Deployment

See [DEPLOYMENT.md](../DEPLOYMENT.md) for the full ECS / Bedrock setup.
Highlights:
- Docker image: `Dockerfile` (Streamlit + Python 3.10 + Polars + DuckDB +
  PyTorch CPU).
- ECS Task definition: `task-def.json` — Bedrock invoke + S3 read +
  CloudWatch logs.
- Fine‑tuning IAM: trust policies and inline policies in repo root.
- Catalogue hydration: `DATA_S3_URI` triggers `ensure_csv_from_s3()` at
  app boot.

---

## 9. Final-app component map: recommendation pipeline → linear solver

This section focuses exclusively on the components that run **at inference time**
in the deployed Streamlit app and explains how they connect end-to-end.
Training-time code (`reciperl/data.py`, `reciperl/ncf.py`, `reciperl/env.py`,
`reciperl/ppo.py`, `reciperl/train.py`, `reciperl/evaluate.py`,
`finetuning/`) is **not** loaded by the app; its only artifact consumed at
runtime is `data/reciperl.pt` (the checkpoint).

---

### 9.1 Two runtime paths

The app exposes two independent pipelines that both terminate at the same
linear solver (`optimize_bento`).

```
┌────────────────────────────────────────────────────────────────────────────┐
│  PATH A — Bento Maker (direct, no RL)                                      │
│                                                                            │
│  CSV ──► load_products ──► apply_regime ──► optimize_bento ──► UI table   │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│  PATH B — RL-driven day composition (recommendation + linear solver)       │
│                                                                            │
│  reciperl.pt ──► load_policy                                               │
│       │                                                                    │
│       ▼                                                                    │
│  FusedState + ActorCritic                                                  │
│       │  (masked policy.act → recipe_id)                                   │
│       ▼                                                                    │
│  recipes.json ──► Recipe (name, ingredients, category, tags)               │
│       │                                                                    │
│       ▼                                                                    │
│  IngredientResolver                                                        │
│    get_chat_llm → LangChain prompt → LLM (Bedrock / Groq)                 │
│       │  (NL → SQL)                                                        │
│       ▼                                                                    │
│  DuckDB (products.duckdb) ──► candidate products DataFrame                │
│       │                                                                    │
│       ▼                                                                    │
│  optimize_bento (NNLS → BVLS) ──► gram quantities ──► ComposedDay        │
└────────────────────────────────────────────────────────────────────────────┘
```

---

### 9.2 Data preparation (offline prerequisite)

**File:** [data_prep_nutriments.py](../data_prep_nutriments.py)

Runs once (locally or via `run_data_prep()` in the app sidebar) and produces
two artefacts consumed at runtime:

| Artefact | Consumer |
|---|---|
| `data/products_names_with_macro_nutriments.csv` | `load_products` (Path A), `_ensure_duckdb` |
| `data/products.duckdb` | `IngredientResolver` (Path B), Exploration page |

The pipeline is a sequential chain:

```
download_data()         ← Hugging Face OFF Parquet
      │
expr_france_not_obsolete()   ← keep FR, non-obsolete rows
      │
expr_product_name_french()   ← extract French product label
      │
get_nutriments()        ← flatten nutriments map → typed macro columns
      │
clean_categories()      ← strip en:/fr: prefixes, normalise accents
      │
add_tags()              ← derive boolean flags (vegan, halal, meat, …)
      │
filter_products_catalog()    ← drop excluded categories (alcohol, snacks…)
      │
normalize_catalog_product_names()   ← uniform capitalisation
      │
      └─► CSV_PATH  (+ DuckDB materialised on first resolver call)
```

Each function is **pure and composable**; `filter_products_catalog` acts as
the gatekeeper that guarantees only nutritionally relevant products reach the
solver.

---

### 9.3 Shared core — `kojin_common.py`

Single import surface for both paths.  Three elements matter at runtime:

**`load_products(CSV_PATH)`**
- Calls `filter_products_catalog` again on the CSV (second guard) and
  `normalize_catalog_product_names`.
- Cached by `@st.cache_data` so the CSV is only parsed once per session.
- Output is the Polars DataFrame passed to `optimize_bento` in Path A.

**`optimize_bento(products_df, user_targets, meal_fraction, portion_legumes)`**
- Receives a Polars DataFrame of *candidate products* (from either Path A or
  Path B) and a 4-element vector of daily macro targets.
- Builds the constraint matrix `M` (nutrient values per 100 g, divided by
  100).
- Builds the target vector as `[kcal·f, prot, fat·f, carb·f, 25·f]` where
  `f = meal_fraction`.
- **Stage 1 — NNLS (`scipy.optimize.nnls`)**: initial unconstrained
  non-negative least-squares fit; selects the support set of non-zero products.
- **Stage 2 — BVLS (`scipy.optimize.lsq_linear`)**: re-solves over the
  support set with per-product upper bounds (`portion_maximale`), method
  `"bvls"`.
- Post-processing bucketing: keeps at most one animal-protein item and one oil
  item, all non-animal/non-oil items are kept freely.
- Returns a sorted Polars DataFrame with columns `Aliment`, `Quantité (g)`,
  and per-meal nutrient totals.

**`get_chat_llm(temperature, max_tokens)`**
- Provider-agnostic LangChain factory: resolves to Bedrock, Groq, or OpenAI
  depending on `LLM_PROVIDER` env var and available API keys.
- Cached by `@st.cache_resource`; the same model instance is shared between
  the Exploration page and `IngredientResolver`.

---

### 9.4 Path A — Bento Maker (direct optimization)

**Files:** [streamlit_app.py](../streamlit_app.py), [app_pages/bento_maker.py](../app_pages/bento_maker.py), [bento_editor.py](../bento_editor.py)

```
User sidebar input (weight, activity, goal, regime, n_bentos, fractions)
       │
compute_targets()     ← Harris-Benedict + TDEE → [kcal, prot, fat, carb]
       │
load_products(CSV_PATH)
       │
apply_regime(products, regime)   ← filter boolean tag columns
       │
for each bento slot:
    optimize_bento(products, targets, meal_fraction, portion_legumes)
       │                │
       │         _run_solver (NNLS → BVLS)
       │
render_bento_inline_table()   ← editable Streamlit table (bento_editor.py)
       │
user locks an ingredient
       │
sync_locked_bentos_from_ingredient_locks()
       │
run_daily_rebalance_after_anchor()   ← re-runs optimize_bento on unlocked slots
```

`bento_editor.py` is purely a presentation layer; it calls `optimize_bento`
via `run_daily_rebalance_after_anchor` to propagate any locked-ingredient
constraint back through the solver.

---

### 9.5 Path B — RL recommendation pipeline

This path is the **recommendation system associated with the linear solver**.
It chains five components sequentially for each meal slot of the day.

#### 9.5.1 Checkpoint loader — `reciperl/compose.py : load_policy`

Reads `data/reciperl.pt` (written by `reciperl/train.py`) and
reconstructs:
- `RecipeRLConfig` — all hyperparameters needed to size the networks.
- `FusedState` (from [reciperl/state.py](../reciperl/state.py)) — the
  state-encoding module; weights loaded from checkpoint.
- `ActorCritic` (from [reciperl/policy.py](../reciperl/policy.py)) — the PPO
  actor-critic; weights loaded from checkpoint.
- `rating_matrix` (NumPy array, shape `[n_users, n_items]`) — the precomputed
  NCF simulator outputs used for cold-start and reward lookup.
- `item_names` — recipe names aligned with item indices.

Neither `ncf.py` nor `env.py` nor `ppo.py` is imported at this stage; only
their serialised outputs matter.

#### 9.5.2 Recipe catalogue — `reciperl/recipes.py` + `data/recipes.json`

`load_recipes()` returns a `RecipeCatalogue` (30 recipes) with helpers
`category_ids` and `filter_by_tag` used by `compose_day` to build the boolean
action mask and to extract ingredient slots.

Each `Recipe` carries: `id`, `name`, `category`, `cuisine`, `tags`,
`ingredients: list[RecipeIngredient]`, `macros_per_serving`.  The ingredient
list is what gets passed to `IngredientResolver`.

#### 9.5.3 FusedState — `reciperl/state.py`

Encodes the user's interaction history into a fixed-size state vector fed to
the policy.  Inputs at each step:

| Tensor | Shape | Source |
|---|---|---|
| `u` | `[1]` | user id |
| `h` | `[1, k]` | sliding window of last *k* recipe ids |
| `hr` | `[1, k]` | corresponding NCF-predicted ratings |
| `rc` | `[1, k]` | recommendation counts inside the window |
| `last_item` | `[1]` | most recently picked recipe id |
| `last_cat` | `[1]` | category id of `last_item` |

`FusedState.forward` produces a vector of dimension `3 × embed_dim` (e.g. 96)
by concatenating three attention-weighted blocks (user/item, sequential
history, user/category).

#### 9.5.4 ActorCritic — `reciperl/policy.py`

Receives the fused-state vector and a boolean action mask (recipes of the
right category that haven't been picked yet today).  `policy.act(state, mask)`
applies `logits.masked_fill(~mask, -inf)` before sampling, guaranteeing
category-correct, non-duplicate picks.  Returns `(recipe_id, log_prob, value)`.

#### 9.5.5 IngredientResolver — `reciperl/recipe_prompts.py`

For each abstract `RecipeIngredient` (e.g. `{name: "chicken breast", role: "protein"}`):

1. `_build_question(ing, regime, recipe_tags)` fills `_USER_TEMPLATE` with
   the ingredient name, nutritional role, dietary regime and recipe tags.
2. The LangChain chain `prompt | get_chat_llm() | StrOutputParser()` sends
   the question to the LLM under a strict system prompt that constrains the
   output to a single DuckDB SQL query.
3. `_clean_sql(raw)` strips markdown fences, normalises the `"energy-kcal"`
   quoting and removes trailing semicolons.
4. The cleaned SQL is executed against `products.duckdb` (read-only) and the
   result is returned as a Polars DataFrame (`ResolvedIngredient.products`).

The resolver reuses one LangChain chain across all ingredients of a day
(`@__init__` caches the chain), so the LLM is loaded only once.

#### 9.5.6 compose_day — `reciperl/compose.py`

The orchestrator that calls 9.5.1–9.5.5 in sequence for each `MealSlot`:

```
load_policy(path)
       │
_build_initial_history(user_id, rating_matrix, cfg)   ← cold-start from top-k ratings
       │
for slot in meal_slots:
    _category_mask(catalogue, slot.category, chosen)   ← boolean mask over 30 recipes
       │
    _observe_state(state_module, user_id, history, …)  ← FusedState forward pass
       │
    policy.act(state, mask)                             ← ActorCritic → recipe_id
       │
    resolver.resolve_recipe(recipe, regime)             ← IngredientResolver × n_ingredients
       │
    pl.concat(frames).unique("product_name")            ← merge candidate DataFrames
       │
    optimize_bento(products_df, daily_targets,          ← linear solver
                   meal_fraction, portion_legumes)
       │
    ComposedMeal(slot, recipe, sql_log, products_df, solution)
       │
history window slides → next slot
       │
ComposedDay(meals=[…])
```

The rolling history update (lines 212–217 of [compose.py](../reciperl/compose.py))
ensures that each subsequent meal's state reflects the recipes already chosen
today, so the policy naturally diversifies across meal slots.

---

### 9.6 Component interaction summary

```
data_prep_nutriments.py
    ↓ CSV / DuckDB
kojin_common.py (load_products, get_chat_llm, optimize_bento)
    ├── Path A: bento_maker.py → bento_editor.py
    │              apply_regime → optimize_bento → UI
    │
    └── Path B: reciperl/compose.py (compose_day)
                   ├── reciperl/state.py  (FusedState)     ┐
                   ├── reciperl/policy.py (ActorCritic)    ├── loaded from reciperl.pt
                   ├── rating_matrix (NumPy)               ┘
                   ├── reciperl/recipes.py (RecipeCatalogue)
                   ├── reciperl/recipe_prompts.py (IngredientResolver)
                   │       └── get_chat_llm → LLM → SQL → DuckDB → DataFrame
                   └── optimize_bento → ComposedDay
```

**Components not loaded at runtime** (training artefacts only):
`reciperl/data.py`, `reciperl/ncf.py`, `reciperl/env.py`,
`reciperl/ppo.py`, `reciperl/train.py`, `reciperl/evaluate.py`,
`finetuning/generate_dataset.py`, `finetuning/launch_finetune.py`,
`finetuning/evaluate.py`.

---

## 8. File map

| Concern | Path |
|---|---|
| Raw data → curated CSV | [data_prep_nutriments.py](../data_prep_nutriments.py) |
| Shared helpers + optimiser + LLM factory | [kojin_common.py](../kojin_common.py) |
| Streamlit entry point | [streamlit_app.py](../streamlit_app.py) |
| Bento page | [app_pages/bento_maker.py](../app_pages/bento_maker.py), [bento_editor.py](../bento_editor.py) |
| Exploration page | [app_pages/exploration.py](../app_pages/exploration.py) |
| NL→SQL fine‑tuning | [finetuning/generate_dataset.py](../finetuning/generate_dataset.py), [finetuning/launch_finetune.py](../finetuning/launch_finetune.py), [finetuning/evaluate.py](../finetuning/evaluate.py) |
| Recipe catalogue | [data/recipes.json](../data/recipes.json), [reciperl/recipes.py](../reciperl/recipes.py) |
| Food.com dataset loader | [reciperl/foodcom_data.py](../reciperl/foodcom_data.py) |
| RL config / data | [reciperl/config.py](../reciperl/config.py), [reciperl/data.py](../reciperl/data.py) |
| Simulator | [reciperl/ncf.py](../reciperl/ncf.py) |
| State / policy / env / PPO | [reciperl/state.py](../reciperl/state.py), [reciperl/policy.py](../reciperl/policy.py), [reciperl/env.py](../reciperl/env.py), [reciperl/ppo.py](../reciperl/ppo.py) |
| Training & evaluation | [reciperl/train.py](../reciperl/train.py), [reciperl/evaluate.py](../reciperl/evaluate.py) |
| LLM ingredient resolver | [reciperl/recipe_prompts.py](../reciperl/recipe_prompts.py) |
| End‑to‑end day composition | [reciperl/compose.py](../reciperl/compose.py) |
