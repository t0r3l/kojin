# Kōjin — コージン

> *« Donnons à votre corps les repas qu'il mérite. »*

**Kōjin** est une application web qui compose des **bentos nutritionnellement optimisés** à partir de la base Open Food Facts, en fonction du profil et des objectifs de l'utilisateur.

Pour chaque journée, l'application :
1. calcule les **besoins nutritionnels** (kcal, protéines, lipides, glucides, légumes) via la formule Mifflin–St Jeor ajustée par l'activité et l'objectif ;
2. filtre la base de produits selon le **régime alimentaire** (vegan, végétarien, halal, casher, sans gluten, bio) ;
3. résout un **système linéaire sous contraintes** (NNLS + BVLS, `scipy`) pour proposer des quantités d'aliments qui collent aux cibles macro par bento ;
4. applique des règles métier (max 1 protéine animale, max 1 huile, pas de doublons entre bentos) ;
5. affiche les bentos dans une UI Streamlit épurée inspirée de la typographie japonaise.

## Stack

- **Streamlit** — UI web.
- **Polars** — chargement et filtrage rapide du catalogue produits (~dizaines de milliers de lignes).
- **SciPy** (`nnls`, `lsq_linear` avec méthode BVLS) — solveur d'optimisation linéaire.
- **Hugging Face Hub** — téléchargement du dataset Open Food Facts (`food.parquet`).
- **boto3** — téléchargement du CSV préparé depuis S3 en production, et appel d'**Amazon Bedrock** (Nova Micro) pour la génération des consignes de préparation.

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

Dans la barre latérale :

| Section | Choix |
|---|---|
| **Profil** | Genre, âge, poids, taille |
| **Activité** | Activité quotidienne (sédentaire → actif) + fréquence sportive |
| **Objectif** | Sèche musculaire / Recomposition / Prise de masse |
| **Régime** | Aucun, Vegan, Végétarien, Halal, Casher, Sans Gluten, Bio |
| **Bentos** | 1 à 5 bentos par jour, fraction calorique par bento (sliders couplés), bento qui reçoit la protéine animale |

L'encadré en haut affiche les **objectifs journaliers calculés** (kcal, macros, portion légumes). Cliquer sur **« Composer les bentos »** déclenche l'optimisation : chaque bento s'affiche avec sa liste d'aliments, les quantités en grammes, et la contribution macro par aliment.

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

## Consignes de préparation (LLM)

Chaque bento généré affiche un bouton **« Consignes de préparation »** qui appelle un LLM pour proposer une recette (cuissons, assaisonnements, ordre de montage) à partir des ingrédients pesés du bento.

**Backend par défaut : [Amazon Bedrock](https://aws.amazon.com/bedrock/) avec `amazon.nova-micro-v1:0`** — le modèle le moins cher du catalogue Bedrock ($0,000035 / 1K tokens entrée, $0,00014 / 1K tokens sortie au moment de la rédaction), largement suffisant pour 4–6 étapes de préparation.

Intégration native AWS :
- pas de clé d'API à gérer — l'auth passe par le **rôle IAM de la tâche ECS** (`bedrock:InvokeModel`) ;
- pas de sortie de trafic hors AWS ;
- facturation unifiée via Cost Explorer, logs via CloudTrail.

### Activer Bedrock

1. Dans la console AWS **Bedrock → Model access** de la région choisie (ex. `eu-west-1` ou `eu-west-3`), activer l'accès à `amazon.nova-micro-v1:0`.
2. Ajouter au rôle de tâche ECS (voir `DEPLOYMENT.md`, §6.3) l'autorisation d'invoquer le modèle :

   ```json
   {
     "Effect": "Allow",
     "Action": ["bedrock:InvokeModel"],
     "Resource": "arn:aws:bedrock:*::foundation-model/amazon.nova-micro-v1:0"
   }
   ```

Variables d'environnement disponibles :

| Variable | Défaut | Rôle |
|---|---|---|
| `LLM_BACKEND` | `auto` | `bedrock`, `pollinations`, ou `auto` (Bedrock puis repli Pollinations) |
| `BEDROCK_MODEL_ID` | `amazon.nova-micro-v1:0` | ID du modèle Bedrock |
| `AWS_REGION` | `eu-west-1` | Région Bedrock |

### Développement local

Sans credentials AWS, le backend `auto` retombe automatiquement sur [Pollinations.ai](https://pollinations.ai), un service gratuit sans clé, ce qui permet de tester la fonctionnalité en local sans configurer Bedrock. Pour forcer ce mode : `export LLM_BACKEND=pollinations`.

## Structure du projet

```
kojin/
├── README.md                          # Ce fichier
├── DEPLOYMENT.md                      # Déploiement AWS (ECS Fargate + S3 + ALB)
├── requirements.txt                   # Dépendances Python
├── streamlit_app.py                   # UI + solveur + logique métier
├── data_prep_nutriments.py            # Pipeline Open Food Facts → CSV
├── .gitignore
└── data/                              # Non versionné
    ├── food.parquet                   # Téléchargé depuis Hugging Face
    └── products_names_with_macro_nutriments.csv   # Produit par data_prep
```

## Déploiement

Le déploiement sur AWS (ECS Fargate + ALB + S3 pour le CSV) est documenté en détail dans **[DEPLOYMENT.md](DEPLOYMENT.md)**.

En production, l'application lit le CSV depuis un bucket S3 au démarrage via la variable d'environnement `DATA_S3_URI` (ex : `s3://kojin-data-123456789012/products_names_with_macro_nutriments.csv`).

## Crédits & licence

- Données : [Open Food Facts](https://world.openfoodfacts.org/), distribuées sous licence [ODbL](https://opendatacommons.org/licenses/odbl/1-0/).
- Typographie : Noto Serif JP, Inter (Google Fonts).
