# Kōjin — コージン

> *« Donnons à votre corps les repas qu'il mérite. »*

**Kōjin** est une application web qui compose des **bentos nutritionnellement optimisés** à partir de la base Open Food Facts, en fonction du profil et des objectifs de l'utilisateur.

L'application Streamlit est organisée en **deux pages** :

| Page | Rôle |
|---|---|
| **Bento Maker** *(page d'accueil)* | Compose les bentos optimisés selon le profil et les objectifs. |
| **Exploration des ingrédients** | Interroge la base de produits en **langage naturel** ; un LLM (Bedrock + LangChain) génère le SQL DuckDB, le résultat est rendu sous forme de tableau. |

Pour chaque journée, le **Bento Maker** :
1. calcule les **besoins nutritionnels** (kcal, protéines, lipides, glucides, légumes) via la formule Mifflin–St Jeor ajustée par l'activité et l'objectif ;
2. filtre la base de produits selon le **régime alimentaire** (vegan, végétarien, halal, casher, sans gluten, bio) ;
3. résout un **système linéaire sous contraintes** (NNLS + BVLS, `scipy`) pour proposer des quantités d'aliments qui collent aux cibles macro par bento ;
4. applique des règles métier (max 1 protéine animale, max 1 huile, pas de doublons entre bentos) ;
5. affiche les bentos dans une UI Streamlit épurée inspirée de la typographie japonaise.

## Stack

- **Streamlit** (multi-pages via `st.navigation`) — UI web.
- **Polars** — chargement et filtrage rapide du catalogue produits (~dizaines de milliers de lignes).
- **SciPy** (`nnls`, `lsq_linear` avec méthode BVLS) — solveur d'optimisation linéaire.
- **Hugging Face Hub** — téléchargement du dataset Open Food Facts (`food.parquet`).
- **boto3** — téléchargement du CSV préparé depuis S3 en production, et appel d'**Amazon Bedrock** (Nova Micro).
- **LangChain** (`langchain-core`, `langchain-aws`) — chaîne de génération de SQL pour la page d'exploration.
- **DuckDB** — moteur SQL embarqué qui exécute les requêtes générées par le LLM, en lecture seule, sur la base produits.

## Démarrage rapide (local)

### 1. Cloner et installer

```bash
git clone https://github.com/t0r3l/kojin.git
cd kojin

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Préparer les données

La première exécution nécessite de générer le CSV produits (~95 Mo) depuis le parquet Open Food Facts (~6.7 Go téléchargés, traités en streaming avec Polars) :

```bash
python data_prep_nutriments.py
```

Cette commande :
- télécharge `food.parquet` depuis Hugging Face (`openfoodfacts/product-database`) dans `data/`,
- filtre les produits France non obsolètes,
- extrait les 5 macro-nutriments (kcal, protéines, lipides, glucides, fibres),
- nettoie les catégories et labels (accents, séparateurs),
- ajoute des **tags booléens** (`vegan`, `halal`, `meat`, `fish`, `lait`, `gluten_free`…),
- exclut les produits NOVA 4 (ultra-transformés),
- écrit `data/products_names_with_macro_nutriments.csv`.

> Durée : ~5 à 15 minutes selon la connexion et la machine. L'étape se fait en batch par row-groups Parquet pour rester en mémoire raisonnable.

### 3. Lancer l'application

```bash
streamlit run streamlit_app.py
```

L'app s'ouvre sur [http://localhost:8501](http://localhost:8501).

Si le CSV n'est pas présent localement au lancement, un bouton **« Lancer la préparation des données »** permet de déclencher la prépa directement depuis l'UI.

## Comment l'utiliser

L'application a deux pages, accessibles via la navigation Streamlit dans la barre latérale.

### Page « Bento Maker » *(page d'accueil)*

Dans la barre latérale :

| Section | Choix |
|---|---|
| **Profil** | Genre, âge, poids, taille |
| **Activité** | Activité quotidienne (sédentaire → actif) + fréquence sportive |
| **Objectif** | Sèche musculaire / Recomposition / Prise de masse |
| **Régime** | Aucun, Vegan, Végétarien, Halal, Casher, Sans Gluten, Bio |
| **Bentos** | 1 à 5 bentos par jour, fraction calorique par bento (sliders couplés), bento qui reçoit la protéine animale |

L'encadré en haut affiche les **objectifs journaliers calculés** (kcal, macros, portion légumes). Cliquer sur **« Composer les bentos »** déclenche l'optimisation : chaque bento s'affiche avec sa liste d'aliments, les quantités en grammes, et la contribution macro par aliment.

### Page « Exploration des ingrédients »

Vous tapez une question en français (ou n'importe quelle langue), par exemple :

- *« Quels produits vegan ont plus de 25 g de protéines pour 100 g ? »*
- *« Top 20 des aliments avec le plus de fibres et au maximum 200 kcal pour 100 g. »*
- *« Liste les produits halal et sans gluten avec leur ratio protéines / kcal. »*

Une chaîne **LangChain** envoie la question + le schéma de la table à **Amazon Bedrock** (modèle configurable, défaut Nova Micro), récupère une requête SQL DuckDB, l'exécute en **lecture seule** contre la base produits, puis affiche le résultat sous forme de tableau Streamlit. Le SQL généré est également affiché pour transparence et debug. **Aucun autre fournisseur LLM n'est utilisé** — en local comme en prod, il faut des credentials AWS valides et l'accès au modèle dans la console Bedrock.

## Calcul des objectifs nutritionnels

Formule **Mifflin–St Jeor** pour le métabolisme de base (BMR), ajustée en deux temps :

```
BMR       = 10·poids + 6.25·taille − 5·âge + (5 si homme, −161 si femme)
multiplier = activité_quotidienne + fréquence_sport   (plafonné à 1.95)
TDEE      = BMR × multiplier

energy    = TDEE × cal_factor        # 0.90 lean / 1.00 balanced / 1.15 bulk
proteins  = prot_per_kg × poids      # 2.2 lean / 1.8 balanced / 1.6 bulk
fat       = energy × fat_pct / 9     # Atwater
carbs     = (energy − 4·proteins − 9·fat) / 4   (plancher 50 g)
```

Les 5 valeurs (`energy, proteins, fat, carbs, portion_légumes`) deviennent les cibles globales passées au solveur, réparties par bento selon les fractions configurées.

## Algorithme de composition

Pour chaque bento, on construit :
- un **vecteur cible** `[kcal·frac, protéines/bento, lip·frac, gluc·frac, fibres·frac, (légumes·frac)]`,
- une **matrice `M`** des nutriments pour 100 g des produits filtrés,
- des **bornes par produit** (`portion_maximale`, défaut 200 g).

Solveur **hybride** :
1. **NNLS** (`scipy.optimize.nnls`) pour présélectionner les produits pertinents.
2. **BVLS** (`lsq_linear` avec `method="bvls"`) sur le sous-ensemble, avec contraintes `0 ≤ x ≤ portion_max`.

Post-traitement métier :
- séparation animal / huile / autres,
- au plus **1 protéine animale** (et seulement sur le bento désigné),
- au plus **1 huile**,
- pas de doublons d'aliments d'un bento à l'autre (via `code` produit).

## Exploration en langage naturel (LangChain + Bedrock + DuckDB)

La page **« Exploration des ingrédients »** repose sur une chaîne LangChain minimale :

```
Question NL ──► ChatPromptTemplate (système + schéma DuckDB)
              │
              ▼
       ChatBedrockConverse  ◄── Amazon Bedrock (Nova Micro, region AWS)
              │
              ▼
        StrOutputParser ──► requête SQL DuckDB
              │
              ▼
        DuckDB (read-only) ──► Polars DataFrame ──► st.dataframe
```

**Pourquoi ce design**
- **`amazon.nova-micro-v1:0`** est le modèle le moins cher du catalogue Bedrock (~$0,000035 / 1K tokens entrée, ~$0,00014 / 1K tokens sortie). Pour générer une requête SQL de quelques centaines de tokens, le coût par question est négligeable.
- **Auth via le rôle IAM de la tâche** — pas de clé d'API à gérer, pas de trafic sortant hors AWS, audit CloudTrail, facturation Cost Explorer.
- **DuckDB en read-only** sur un fichier `data/products.duckdb` matérialisé une fois depuis le CSV : aucun risque qu'une requête générée par le LLM modifie ou détruise des données.
- La table indexée s'appelle **`products`**, et le **schéma exact** (avec types DuckDB) est injecté dans le prompt système à chaque requête, ce qui rend les colonnes booléennes (`vegan`, `halal`, `vegetarian`, `bio`, `gluten_free`, `kascher`, `meat`, `fish`, `lait`, `no_palm_oil`) directement utilisables.

**Activer Bedrock**

1. Dans la console **Bedrock → Model access** de la région choisie (ex. `eu-west-1`), activer l'accès à `amazon.nova-micro-v1:0`.
2. Ajouter au rôle de tâche ECS (voir `DEPLOYMENT.md`, §6.3) :

   ```json
   {
     "Effect": "Allow",
     "Action": ["bedrock:InvokeModel"],
     "Resource": "arn:aws:bedrock:*::foundation-model/amazon.nova-micro-v1:0"
   }
   ```

**Variables d'environnement**

| Variable | Défaut | Rôle |
|---|---|---|
| `BEDROCK_MODEL_ID` | `amazon.nova-micro-v1:0` | ID du modèle Bedrock invoqué par `ChatBedrockConverse` |
| `AWS_REGION` | `eu-west-1` | Région du client Bedrock (doit correspondre à celle où l'accès au modèle est activé) |

**Développement local** — la page Exploration appelle Bedrock comme en production. Configurez par exemple `aws configure`, `aws sso login --profile …` + `export AWS_PROFILE=…`, ou des variables `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` (+ `AWS_SESSION_TOKEN` si besoin). Vérifiez avec `aws sts get-caller-identity` et testez une invocation Bedrock dans la même région que `AWS_REGION`.

Le **Bento Maker** ne fait jamais d'appel LLM — il fonctionne 100 % en local sans Bedrock (au-delà du téléchargement S3 optionnel pour le CSV produits).

## Structure du projet

```
kojin/
├── README.md                          # Ce fichier
├── DEPLOYMENT.md                      # Déploiement AWS (ECS Fargate + S3 + ALB + Bedrock)
├── requirements.txt                   # Dépendances Python
├── streamlit_app.py                   # Entry point st.navigation (Bento Maker / Exploration)
├── kojin_common.py                    # CSS, constantes, chargement, optimiseur, factory Bedrock
├── app_pages/                         # Renommé pour éviter l'auto-discovery legacy de Streamlit
│   ├── bento_maker.py                 # Page Bento Maker (page d'accueil)
│   └── exploration.py                 # Page Exploration des ingrédients (LangChain + DuckDB)
├── data_prep_nutriments.py            # Pipeline Open Food Facts → CSV
├── .gitignore
└── data/                              # Non versionné
    ├── food.parquet                   # Téléchargé depuis Hugging Face
    ├── products_names_with_macro_nutriments.csv   # Produit par data_prep
    └── products.duckdb                # Construit au premier lancement de la page Exploration
```

## Déploiement

Le déploiement sur AWS (ECS Fargate + ALB + S3 pour le CSV) est documenté en détail dans **[DEPLOYMENT.md](DEPLOYMENT.md)**.

En production, l'application lit le CSV depuis un bucket S3 au démarrage via la variable d'environnement `DATA_S3_URI` (ex : `s3://kojin-data-123456789012/products_names_with_macro_nutriments.csv`).

## Crédits & licence

- Données : [Open Food Facts](https://world.openfoodfacts.org/), distribuées sous licence [ODbL](https://opendatacommons.org/licenses/odbl/1-0/).
- Typographie : Noto Serif JP, Inter (Google Fonts).
