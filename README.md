# Kōjin — コージン

> *« Donnons à votre corps les repas qu'il mérite. »*

**Kōjin** est une application web de planification nutritionnelle. Elle compose des bentos optimisés à partir du catalogue Open Food Facts et, en mode avancé, planifie une journée complète pilotée par une politique d'apprentissage par renforcement entraînée sur les préférences réelles d'utilisateurs Food.com.

---

## Architecture

L'application repose sur **deux chemins de composition** et **quatre pages Streamlit**.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              streamlit_app.py                                │
│                                                                              │
│  ┌────────────┐  ┌──────────────────┐  ┌────────────────────┐  ┌──────────┐ │
│  │   Profil   │  │  Bento Planner   │  │  Exploration des   │  │ RL       │ │
│  │            │  │   (Path A)       │  │  ingrédients       │  │ Planner  │ │
│  └─────┬──────┘  └────────┬─────────┘  └────────┬───────────┘  └────┬─────┘ │
└────────┼─────────────────-┼───────────────────--┼───────────────────┼───────┘
         │                  │                      │                   │
         ▼                  ▼                      ▼                   ▼
    kojin.db           Mifflin–St Jeor       LLM (Groq /        reciperl.pt
    SQLite             compute_targets()     Bedrock /           load_policy()
    users              │                    OpenAI-compat /            │
    meal_slots         ▼                    Anthropic)                 ▼
    user_rl       optimize_bento()                │               start_session()
                  (NNLS + BVLS)            NL → SQL DuckDB       PPO policy
                  Open Food Facts                 │               all slots at once
                  catalogue               products.duckdb               │
                                          st.dataframe          ┌───────┴──────────┐
                                                                │ resolver=None    │
                                                                │ recipe names only│
                                                                └──────────────────┘
                                                                ┌───────────────────┐
                                                                │ resolver=LLM      │
                                                                │ IngredientResolver│
                                                                │ parallel threads  │
                                                                │ → DuckDB          │
                                                                │ → optimize_bento()│
                                                                └───────────────────┘
```

### Path A — Bento Planner

Optimisation directe : les **objectifs nutritionnels journaliers** (kcal, protéines, lipides, glucides, légumes) sont calculés via Mifflin–St Jeor, puis répartis entre les bentos. Pour chaque bento, un solveur hybride NNLS + BVLS (`scipy`) trouve les quantités (en grammes) d'aliments Open Food Facts qui minimisent l'écart aux cibles macros, sous contraintes de portion et de diversité.

### Page Profil

Formulaire de configuration du compte utilisateur, en deux onglets :

- **Informations personnelles** — genre, âge, poids, taille, activité physique, objectif nutritionnel (sèche / recomposition / prise de masse), régime alimentaire. Les cibles caloriques sont recalculées en direct (Mifflin–St Jeor).
- **Plan de repas** — nombre de repas par jour (1 à 5), avec pour chacun un nom libre, une catégorie (`plat`, `petit_dej`, `snack`) et un slider de proportion kcal. Les proportions doivent totaliser 100 % pour être sauvegardées.

Le profil est persisté dans **`data/kojin.db`** (SQLite), tables `users`, `meal_slots`, `user_rl`. Un seul compte suffit ; la connexion se fait par pseudo depuis n'importe quelle page.

### Path B — RL Planner

Planification par politique PPO : la politique **génère simultanément une recette pour chaque slot** (configurés dans le Profil) en tenant compte du profil utilisateur (embedding personnalisé + historique des recettes acceptées). Si un LLM est configuré, chaque ingrédient abstrait est résolu en produits réels du catalogue Open Food Facts via `IngredientResolver` (NL→SQL→DuckDB), puis `optimize_bento` calcule les quantités.

### Page Exploration

Interface de **requêtes en langage naturel** sur le catalogue produits : un LLM génère une requête SQL DuckDB, qui s'exécute en lecture seule et retourne un tableau.

---

## Stack technique

| Couche | Bibliothèques |
|---|---|
| UI | Streamlit (multipage via `st.navigation`) |
| Données | Polars, DuckDB, Hugging Face Hub |
| Optimisation nutritionnelle | SciPy (`nnls`, `lsq_linear` BVLS) |
| Recommandation RL | PyTorch — NCF + PPO résiduel (Liu et al., 2024) |
| LLM / NL→SQL | LangChain Core, LangChain AWS, LangChain Anthropic, Groq, Bedrock |
| Infrastructure AWS | boto3 (S3, Bedrock), ECS Fargate (voir `DEPLOYMENT.md`) |

---

## Démarrage rapide (local)

### 1. Cloner et installer les dépendances

```bash
git clone https://github.com/t0r3l/kojin.git
cd kojin
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### 2. Préparer le catalogue produits

La première exécution télécharge `food.parquet` (~6,7 Go) depuis Hugging Face et produit le CSV filtré (~95 Mo) :

```bash
python -m data_prep_nutriments
```

Ce script :
- filtre les produits français non obsolètes,
- extrait les cinq macronutriments (kcal, protéines, lipides, glucides, fibres),
- ajoute des tags booléens (`vegan`, `halal`, `vegetarian`, `gluten_free`, `kascher`, `meat`, `fish`, `lait`, `no_palm_oil`, `bio`),
- exclut les produits NOVA 4 et les valeurs aberrantes,
- écrit `data/products_names_with_macro_nutriments.csv`.

> Durée : 5 à 15 minutes. L'étape peut aussi se déclencher depuis l'UI via le bouton **« Lancer la préparation des données »** au premier lancement.

### 3. Configurer le LLM (facultatif)

Requis pour la page **Exploration** et la résolution complète d'ingrédients dans le **RL Planner**. Le **Bento Planner** fonctionne sans LLM.

**Option A — Anthropic / Claude (recommandé en local)**

Placer la clé dans un fichier `.claude_key` à la racine du projet (chargé automatiquement au démarrage, aucune variable d'environnement à exporter) :

```
CLAUDE_API_KEY=sk-ant-api03-...
```

Ou via variable d'environnement :

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-...
```

Modèle utilisé par défaut : `claude-haiku-4-5-20251001` (configurable via `ANTHROPIC_MODEL_ID`).

**Option B — Groq (gratuit)**

```bash
export GROQ_API_KEY=gsk_...
export LLM_PROVIDER=groq
```

**Option C — Amazon Bedrock**

```bash
aws sso login --profile mon-profil
export AWS_PROFILE=mon-profil
export AWS_REGION=eu-west-1
# Activer `amazon.nova-pro-v1:0` dans Bedrock → Model access
```

**Option D — OpenAI / serveur compatible (Ollama, etc.)**

```bash
export OPENAI_API_KEY=sk-...
export LLM_PROVIDER=openai
# ou un serveur local :
export OPENAI_COMPAT_BASE_URL=http://127.0.0.1:11434/v1
export OPENAI_COMPAT_MODEL=qwen2.5:7b
export OPENAI_COMPAT_API_KEY=ollama
```

En mode `auto` (défaut), la priorité de détection est : **Anthropic** (`ANTHROPIC_API_KEY`) → OpenAI (`OPENAI_API_KEY`) → Groq (`GROQ_API_KEY`) → Bedrock (IAM).

### 4. Entraîner la politique RL (facultatif, pour le RL Planner)

Le checkpoint `data/reciperl.pt` doit exister pour que la page **RL Planner** soit utilisable. Si vous n'avez pas de checkpoint pré-entraîné, téléchargez d'abord le dataset Food.com :

```python
# dans un shell Python avec kagglehub installé
# Définir la variable KAGGLE_TOKEN ou utiliser ~/.kaggle/access_token
import kagglehub
kagglehub.dataset_download("irkaal/foodcom-recipes-and-reviews", path="data/")
```

Puis entraînez :

```bash
python -m reciperl.train --foodcom --max-recipes 50000 --steps 50
```

| Paramètre | Description |
|---|---|
| `--foodcom` | Mode Food.com (vraies évaluations d'utilisateurs) |
| `--max-recipes N` | Nombre de recettes chargées (défaut 2000 ; 50 000 pour un entraînement complet) |
| `--steps N` | Nombre de mises à jour PPO (`total_updates`) |
| `--device cpu\|cuda` | Dispositif d'entraînement |
| `--checkpoint path` | Chemin de sortie du checkpoint (défaut `data/reciperl.pt`) |

L'entraînement avec `--max-recipes 50000 --steps 50` prend environ **3 minutes sur CPU** et produit un checkpoint de ~529 Mo (39 280 recettes filtrées, matrice de préférence 2 855 × 39 280, modules NCF + FusedState + ActorCritic). Les desserts et produits non-alimentaires sont exclus lors du chargement des données.

### 5. Lancer l'application

```bash
streamlit run streamlit_app.py
```

L'application s'ouvre sur [http://localhost:8501](http://localhost:8501).

---

## Flux utilisateur

### Bento Planner (page d'accueil)

1. **Profil** — renseigner genre, âge, poids, taille dans la barre latérale.
2. **Activité & objectif** — choisir le niveau d'activité quotidienne, la fréquence sportive, et l'objectif (sèche, recomposition, prise de masse).
3. **Régime** — filtrer le catalogue : aucun, vegan, végétarien, halal, casher, sans gluten, bio.
4. **Bentos** — choisir 1 à 5 bentos, ajuster la fraction calorique de chacun via des sliders couplés, désigner le bento qui reçoit la protéine animale.
5. **Composer** — cliquer sur **« Composer les bentos »** : chaque bento s'affiche avec ses aliments, quantités en grammes, et contribution macro par aliment.
6. **Édition interactive** — modifier les quantités, remplacer un aliment, verrouiller des lignes ; le solveur rééquilibre automatiquement les autres bentos pour maintenir les cibles journalières.

### Exploration des ingrédients

1. Taper une question en français (ou autre langue), par exemple :
   - *« Quels produits vegan ont plus de 25 g de protéines pour 100 g ? »*
   - *« Top 20 des aliments riches en fibres avec moins de 200 kcal. »*
2. Le LLM génère une requête SQL DuckDB affichée à l'écran.
3. La requête s'exécute en lecture seule sur `products.duckdb` ; le résultat s'affiche sous forme de tableau.
4. Optionnel : activer **« Comparer les deux agents »** pour exécuter la même question sur un second modèle (Bedrock fine-tuné ou serveur OpenAI-compatible).

### RL Planner

1. **Connexion** — entrer un pseudo dans la barre latérale (créé automatiquement à la première visite). Le profil et l'embedding personnalisé sont chargés depuis `kojin.db`.
2. **Profil nutritionnel** — configurable dans la page **Profil** (genre, âge, poids, taille, activité, objectif, régime) ainsi que le nombre de repas et leurs proportions caloriques. La page RL Planner lit ces données directement depuis la base.
3. **Nouvelle journée** — cliquer sur **« Nouvelle journée »** ; la politique PPO génère simultanément une recette pour chaque slot défini dans le profil et les affiche en grille.
4. **Affiner** — cliquer **« Changer »** sur n'importe quel repas pour obtenir une nouvelle proposition pour ce slot uniquement ; les autres restent inchangés.
5. **Valider** — cliquer **« Valider la journée »** : toutes les recettes sont acceptées, l'historique est mis à jour, l'embedding utilisateur est affiné, et si un LLM est configuré, les ingrédients de tous les repas sont résolus **en parallèle** en produits Open Food Facts avec quantités optimisées.

---

## Variables d'environnement

| Variable | Défaut | Rôle |
|---|---|---|
| `LLM_PROVIDER` | `auto` | `auto` \| `anthropic` \| `groq` \| `bedrock` \| `openai` — choix du LLM |
| `ANTHROPIC_API_KEY` | — | Clé Anthropic — prioritaire en mode `auto` ; peut aussi être placée dans `.claude_key` sous `CLAUDE_API_KEY=` |
| `ANTHROPIC_MODEL_ID` | `claude-haiku-4-5-20251001` | Modèle Claude utilisé |
| `GROQ_API_KEY` | — | Clé Groq |
| `GROQ_MODEL_ID` | `llama-3.1-8b-instant` | Modèle Groq |
| `OPENAI_API_KEY` | — | Clé OpenAI |
| `OPENAI_MODEL_ID` | `gpt-4o` | Modèle OpenAI |
| `BEDROCK_MODEL_ID` | `us.amazon.nova-pro-v1:0` | Modèle Bedrock référence |
| `BEDROCK_COMPARE_MODEL_ID` | — | Second modèle Bedrock pour la comparaison (optionnel) |
| `AWS_REGION` | `eu-west-1` | Région Bedrock |
| `OPENAI_COMPAT_BASE_URL` | — | URL base serveur OpenAI-compatible (ex. Ollama) |
| `OPENAI_COMPAT_MODEL` | — | Nom du modèle sur le serveur compatible |
| `OPENAI_COMPAT_API_KEY` | — | Clé API du serveur compatible |
| `DATA_S3_URI` | — | URI S3 du CSV produits (téléchargé au démarrage si absent) |
| `KAGGLE_TOKEN` | — | Token Kaggle (`KGAT_...`) pour télécharger Food.com |
| `KOJIN_EXPLORATION_LOG_JSONL` | `0` | `1` pour journaliser les métriques de comparaison en NDJSON |
| `KOJIN_EXPLORATION_LOG_PATH` | `/tmp/...` | Chemin du fichier de métriques |

---

## Architecture RecipeRL (Path B)

L'implémentation suit Liu et al., *"An Interactive Food Recommendation System Using Reinforcement Learning"*, 2024.

Le système repose sur **deux phases distinctes** : un entraînement hors-ligne unique et une inférence interactive par session.

---

### Phase 1 — Entraînement hors-ligne (une seule fois)

```
Food.com (522 K recettes, 1,4 M avis)
  │  filtrage : avis ≥ 4, utilisateurs ≥ 20 avis
  ▼
RecipeRLDataset — 2 855 utilisateurs × 50 000 recettes
  │
  ├──▶ NCF (§3.2) — Neural Collaborative Filtering
  │      Embedding user Pᵤ + item Qᵢ → MLP → rating prédit r̂ᵤᵢ
  │      Résultat : matrice R de taille 2 855 × 50 000
  │      "Pour l'utilisateur u, la recette i vaut environ r̂ᵤᵢ / 5"
  │
  └──▶ PPO résiduel (§3.4) — entraîné avec NCF comme simulateur
         À chaque étape d'entraînement :
           1. La politique propose une recette i pour l'utilisateur u
           2. reward = R[u, i]  ← note NCF, pas un vrai utilisateur
           3. PPO met à jour les poids via backprop

         Les poids sont figés après entraînement.
         Le checkpoint data/reciperl.pt embarque R, FusedState, ActorCritic.
```

> **Point clé :** pendant l'entraînement, les "utilisateurs" sont les 2 855 profils Food.com connus. Le NCF a appris leurs préférences à partir de leurs vrais historiques. La politique PPO apprend à maximiser les notes NCF de **ces utilisateurs précis**.

---

### Phase 2 — Inférence interactive (par session utilisateur)

```
                    ┌─────────────────────────────────────────────┐
                    │           Nouvel utilisateur Kōjin           │
                    │    (non présent dans les 2 855 du training)  │
                    └────────────────────┬────────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────────┐
                    │         Construction de l'état sₜ           │
                    │                                             │
                    │  s_UI  = mean_user_vec ⊗ wᵢ qᵢ  (Eq. 7)   │
                    │          ↑ centroïde de TOUS les            │
                    │            embeddings utilisateurs           │
                    │            (pas d'embedding propre)         │
                    │                                             │
                    │  s_ACH = cross-attention sur les            │
                    │          k dernières recettes acceptées      │
                    │          pondérées par p_mask (Eq. 10–13)   │
                    │          ↑ SEUL composant vraiment           │
                    │            personnalisé par l'interaction    │
                    │                                             │
                    │  s_UC  = mean_user_vec ⊗ wᵢ c_gᵢ (Eq. 14) │
                    │          ↑ même centroïde, catégorie        │
                    │            de la dernière recette           │
                    └────────────────────┬────────────────────────┘
                                         │ sₜ = [s_UI | s_ACH | s_UC]
                                         ▼
                    ┌────────────────────────────────────────────┐
                    │   Politique PPO — π(a | sₜ)                │
                    │   masque catégorie + recettes déjà vues     │
                    └────────────────────┬───────────────────────┘
                                         │ recette proposée
                                         ▼
                              ┌──────────────────┐
                              │  Affichée à l'UI │
                              └────────┬─────────┘
                                       │
               ┌───────────────────────┼──────────────────────────┐
               │ "Changer"             │                           │ "Valider la journée"
               ▼                       │                           ▼
    recette masquée pour ce slot       │          recette ajoutée à history (fenêtre k=10)
    (non stockée, non apprise)         │          rating stocké = R_mean[i]  ← moyenne NCF
    politique re-propose               │          (note globale, pas personnalisée)
    immédiatement                      │          poids mis à jour dans p_mask
                                       │          profil JSON sauvegardé sur disque
```

---

### Impact réel du feedback utilisateur

| Action | Effet immédiat | Effet long terme |
|---|---|---|
| **Changer** | Masque la recette pour ce slot dans la session courante | Contribue au signal négatif lors de l'adaptation d'embedding (target 1.0 dans le MSE) |
| **Valider** | Recette ajoutée à `history`, `rating = R_mean[i]` | `s_ACH` s'enrichit ; embedding `user_vec` affiné avec les recettes acceptées (target 5.0) |
| **Journées accumulées** | — | `user_vec` dérive du centroïde vers les préférences personnelles ; `history` window guide `s_ACH` |

**Adaptation de l'embedding utilisateur (`_adapt_user_embedding`)** — après chaque validation, le vecteur `user_vec` (64 floats) est affiné par 30 pas Adam (lr=1e-3) sur la perte MSE entre les scores NCF prédits et les cibles :

```
Recettes acceptées → target 5.0
Recettes rejetées (vues via "Changer") → target 1.0
```

Seul `user_vec` est mis à jour ; tous les autres poids du modèle restent gelés. Le vecteur est persisté dans `kojin.db` (`user_rl.user_embedding`) et rechargé à la connexion suivante, permettant une personnalisation croissante session après session.

---

### Limite : le problème du cold-start pour les nouveaux utilisateurs

Le papier **évalue uniquement sur des utilisateurs Food.com déjà connus** (split test 80/20). Pour eux, `R[u, i]` est une prédiction NCF personnalisée basée sur leur historique réel.

Dans Kōjin, tous les utilisateurs sont nouveaux. On substitue :

```
R[u, i]       →     R_mean[i] = moyenne de R[:, i]
Embedding Pᵤ  →     mean(P)   = centroïde de tous les embeddings
```

Conséquence : la politique démarre au "goût moyen Food.com" et se personnalise uniquement par accumulation d'items dans `history` au fil des journées — non par un vrai profil de préférences appris.

**Pour aller plus loin** (hors scope actuel) : réentraîner le NCF sur les interactions réelles des utilisateurs Kōjin, ou remplacer `R_mean[i]` par un score explicite basé sur le comportement (ex. 5.0 pour les recettes acceptées sans "Changer", moins pour celles acceptées après plusieurs rejets).

---

### Résolution d'ingrédients parallèle

Lors de **« Valider la journée »**, chaque ingrédient de chaque recette nécessite un appel LLM pour générer une requête SQL DuckDB. Avec 4 repas × ~10 ingrédients, cela représente ~40 appels séquentiels (~60–90 s). Deux niveaux de parallélisme ont été implémentés pour réduire ce temps à ~5–10 s :

```
validate_day()
│
├── Repas 1 ──┐
├── Repas 2 ──┤  ThreadPoolExecutor (max 4 repas en parallèle)
├── Repas 3 ──┤
└── Repas 4 ──┘
               │
               └─ resolve_recipe()
                  │
                  ├── Ingrédient A ──┐
                  ├── Ingrédient B ──┤  ThreadPoolExecutor (max 4 ingrédients)
                  ├── ...            │
                  └── Ingrédient N ──┘
                                     │
                                     └─ _LLM_SEMAPHORE(4)
                                        sémaphore global — au plus 4 appels
                                        LLM simultanés tous threads confondus
```

**Gestion du rate limit (50 req/min sur le tier Anthropic gratuit)** : un sémaphore global (`threading.Semaphore(4)`) limite le nombre d'appels LLM actifs simultanément. Si l'API répond `429 Too Many Requests`, chaque thread attend `2^attempt + jitter` secondes avant de réessayer (jusqu'à 5 tentatives, soit ~30 s d'attente maximale). Ce backoff exponentiel avec jitter évite que tous les threads retentent en même temps.

---

Le checkpoint `data/reciperl.pt` (~529 Mo) embarque :
- la configuration d'entraînement,
- les poids FusedState + ActorCritic,
- la matrice de préférence NCF R (2 855 × 39 280),
- le catalogue de 39 280 recettes Food.com filtrées (noms, ingrédients, macros, tags — desserts et produits non-alimentaires exclus).

---

## Structure du projet

```
kojin/
├── streamlit_app.py            # Entry point — st.navigation (4 pages)
├── kojin_common.py             # CSS, constantes, solveur, factory LLM
├── bento_editor.py             # Éditeur bento interactif (quantités, verrous, rééquilibrage)
├── app_pages/
│   ├── profile.py              # Page Profil — infos personnelles + plan de repas
│   ├── bento_maker.py          # Page Bento Planner (Path A)
│   ├── exploration.py          # Page Exploration NL→SQL
│   └── rl_planner.py           # Page RL Planner (Path B)
├── reciperl/
│   ├── config.py               # RecipeRLConfig (hyperparamètres)
│   ├── train.py                # CLI d'entraînement (NCF + PPO)
│   ├── compose.py              # load_policy(), start_session(), validate_day()
│   ├── recipe_prompts.py       # IngredientResolver (LLM→SQL→DuckDB, parallèle)
│   ├── db.py                   # Couche SQLite — users, meal_slots, user_rl
│   ├── profiles.py             # Ancien JSON (conservé, migration auto vers SQLite)
│   ├── policy.py               # ActorCritic PPO résiduel
│   ├── ppo.py                  # Entraîneur PPO-clip (§3.4, Eqs. 16–17)
│   ├── state.py                # FusedState (§3.3)
│   ├── ncf.py                  # Neural Collaborative Filtering (§3.2)
│   ├── env.py                  # RecipeEnv (environnement RL)
│   ├── data.py                 # RecipeRLDataset
│   ├── foodcom_data.py         # Chargeur dataset Food.com / Kaggle + catégorisation
│   ├── recipes.py              # Recipe, RecipeCatalogue, RecipeIngredient
│   └── evaluate.py             # Precision@T, NDCG@T
├── data_prep_nutriments.py     # Pipeline Open Food Facts → CSV
├── finetuning/                 # Fine-tuning NL→SQL (Llama 3.1 8B, Bedrock)
│   ├── generate_dataset.py
│   ├── launch_finetune.py
│   └── evaluate.py
├── requirements.txt
├── .env.example                # Variables d'environnement à copier
├── DEPLOYMENT.md               # Déploiement ECS Fargate + S3 + ALB + Bedrock
└── data/                       # Non versionné
    ├── products_names_with_macro_nutriments.csv
    ├── products.duckdb
    ├── kojin.db                # Profils utilisateurs SQLite (users, meal_slots, user_rl)
    ├── recipes.csv             # Food.com (optionnel, pour ré-entraînement)
    ├── reviews.csv             # Food.com (optionnel, pour ré-entraînement)
    └── reciperl.pt             # Checkpoint PPO (~529 Mo, 39 280 recettes filtrées)
```

---

## Algorithme de composition (Path A)

**Mifflin–St Jeor** pour les cibles journalières :

```
BMR       = 10·poids + 6.25·taille − 5·âge ± constante genre
TDEE      = BMR × min(activité + sport, 1.95)
energy    = TDEE × facteur_objectif    # 0.90 sèche / 1.00 recompo / 1.15 masse
proteins  = kg × facteur_prot          # 2.2 / 1.8 / 1.6 g/kg
fat       = energy × fat_pct / 9
carbs     = (energy − 4·prot − 9·fat) / 4   (plancher 50 g)
```

**Solveur hybride** par bento :
1. **NNLS** (`scipy.optimize.nnls`) pour présélectionner les colonnes pertinentes.
2. **BVLS** (`lsq_linear(method="bvls")`) sur le sous-ensemble, avec bornes `0 ≤ x ≤ 200 g`.

Contraintes métier : au plus 1 protéine animale (sur le bento désigné), au plus 1 huile, pas de doublon de produit entre bentos.

---

## Fine-tuning NL→SQL (optionnel)

Pour spécialiser un modèle Llama 3.1 8B sur la génération SQL du schéma Kōjin :

```bash
# 1. Générer le jeu de données (~141 exemples train, 97 eval)
python -m finetuning.generate_dataset

# 2. Lancer le fine-tuning sur Bedrock
python -m finetuning.launch_finetune \
  --bucket $DATA_BUCKET \
  --role-arn arn:aws:iam::$AWS_ACCOUNT_ID:role/BedrockFineTuneRole \
  --region us-east-1

# 3. Évaluer fine-tuné vs référence
python -m finetuning.evaluate \
  --finetuned-model-id <custom-model-arn> \
  --reference-model-id amazon.nova-micro-v1:0
```

Voir **[DEPLOYMENT.md §6A](DEPLOYMENT.md)** pour les détails IAM et le branchement dans l'app via `BEDROCK_COMPARE_MODEL_ID`.

---

## Déploiement

Le déploiement AWS (ECS Fargate + ALB + S3 + Bedrock) est documenté dans **[DEPLOYMENT.md](DEPLOYMENT.md)**.

En production, le CSV produits est téléchargé depuis S3 au démarrage via `DATA_S3_URI`.

---

## Crédits

- Catalogue produits : [Open Food Facts](https://world.openfoodfacts.org/) — licence [ODbL](https://opendatacommons.org/licenses/odbl/1-0/).
- Recettes et avis : [Food.com Recipes and Reviews](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews) (Kaggle).
- Architecture RL : Liu et al., *"An Interactive Food Recommendation System Using Reinforcement Learning"*, 2024.
