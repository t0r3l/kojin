# Déploiement AWS — Kōjin (Streamlit)

Guide de déploiement de l'application Streamlit **Kōjin — コージン** sur AWS.

L'application est une UI web mono-conteneur Streamlit (multi-pages) qui :
- sert deux pages — **Bento Maker** (page d'accueil) et **Exploration des ingrédients** ;
- embarque l'optimiseur de bentos (NNLS + BVLS, `scipy`) ;
- **télécharge le CSV de produits depuis un bucket S3 au démarrage** (variable d'environnement `DATA_S3_URI`) ;
- matérialise une base **DuckDB** locale en lecture seule pour la page Exploration ;
- appelle **Amazon Bedrock** via **LangChain** pour les requêtes langage naturel → SQL (**référence** + comparateur Llama/OpenAI-compat optionnel) ;

## 1. Architecture cible

```
                       ┌───────────────────────────┐
Utilisateur  ── HTTPS ─▶│  Application Load Balancer │──┐
                       └───────────────────────────┘  │
                                                      │  HTTP/WebSocket :8501
                                                      ▼
                    ┌────────────────────────────────────────────┐
                    │              ECS Fargate                   │
                    │  ┌──────────────────────────────────────┐  │
                    │  │ Container Streamlit (Kōjin)          │  │
                    │  │  • streamlit_app.py (st.navigation)  │  │
                    │  │  • app_pages/bento_maker.py          │  │
                    │  │  • app_pages/exploration.py          │  │
                    │  │      (LangChain → Bedrock / DuckDB)  │  │
                    │  │  • CSV téléchargé depuis S3 au boot  │  │
                    │  └──────────────────────────────────────┘  │
                    └────────────────────────────────────────────┘
                         │ s3:GetObject            │ bedrock:InvokeModel
                         ▼                         ▼
                 ┌───────────────┐       ┌──────────────────────┐
                 │   S3 Bucket   │       │  Amazon Bedrock      │
                 │ products_*.csv│       │  (Nova / Llama, etc.)│
                 └───────────────┘       └──────────────────────┘

Logs : CloudWatch Logs    |   Images : ECR    |   IAM : rôle exécution + rôle tâche
```

## 1.1 Justification architecturale : Fargate vs alternatives

### Qu'est-ce que Fargate ?

**Fargate** est un modèle de calcul AWS serverless pour ECS/Fargate : vous décrivez le conteneur (vCPU, RAM), AWS gère l'infrastructure sous-jacente. Pas d'EC2 à provisionner, patcher, monitorer.

### Comparaison avec les alternatives

| **Option** | **Coût mensuel (estimé)** | **Avantages** | **Inconvénients** |
|---|---|---|---|
| **ECS Fargate** (1 vCPU, 2 GB RAM) | ~$35 | Pas de gestion EC2 • Scaling automatique • Pay-per-use • CDN intégré | Plus cher à charge stable |
| **ECS EC2** (t3.small, 2 vCPU, 2 GB) | ~$10–15 | Très bon marché • Réservé ~50% cheaper | Gestion manuelle • Patchs • Scaling complexe • VPC obligatoire |
| **Lambda** (via ALB) | ~$20–40 | Serverless • Très scalable | WebSocket non natif • Timeout 15 min • Streamlit problématique |
| **AppRunner** (1 vCPU) | ~$50–60 | Simple (pas VPC) • Déploiement git | Nouveau service • Plus cher que Fargate • Moins flexible |
| **Lightsail** (micro, 0.5 GB) | ~$5 | Ultra bon marché • Simple | Limite ressources • Pas cloud-native • Scalabilité manuel |

### Pourquoi Fargate ici ?

✅ **Choix Fargate** :
- **Streamlit = WebSocket** : besoin de connexions persistantes (Lambda timeout 15 min inadapté)
- **Charge imprévisible** : le fine-tuning Bedrock crée des pics de requêtes SQL; Fargate scale automatiquement
- **DevOps simple** : pas d'EC2 à gérer, patcher, monitorer — focus sur l'app
- **Coût acceptable** : ~$35/mois pour un prototype / PoC est très raisonnable pour AWS
- **Production-ready** : ALB + ECS Fargate est le standard AWS pour apps web modernes

### Alternatives moins chères

Si le budget est critique :

1. **ECS EC2 + t3.small** (~$10–15/mois) : économie ~50%, mais vous gérez les patchs OS, scaling, failover manuel
   ```bash
   # Nécessiterait de modifier task-def.json :
   # "requiresCompatibilities": ["EC2"]  (au lieu de ["FARGATE"])
   # + créer EC2 instance manuellement + attacher au cluster ECS
   ```

2. **Lightsail + Docker** (~$5/mois) : ultra bon marché, mais perte de cloud-native (no auto-scaling, no ALB)
   ```bash
   # Déployer directement avec docker run sur instance Lightsail
   # Pro: très économique | Contre: pas AWS-native, backup/monitoring manuels
   ```

3. **Hybrid**: **Fargate dev** + **Lightsail prod** : Fargate pour prototyper rapidement, Lightsail pour coûts stables

### VPC et subnets — pourquoi obligatoires ?

**Fargate** requiert le **mode réseau `awsvpc`** (vs `bridge` / `host`) :
- Chaque conteneur obtient une **ENI (Elastic Network Interface)** privée
- L'ENI doit être dans un **subnet** du VPC
- L'ALB redirige le trafic public vers l'ENI privée via le subnet

**Subnets publics recommandés** : 2 minimum (redondance multi-AZ) :
- AZ 1 : subnet-123abc (ex. `eu-west-1a`)
- AZ 2 : subnet-456def (ex. `eu-west-1b`)

Si une AZ tombe, l'ALB redirige le trafic vers la seconde, l'application reste online.

```bash
# Les subnets par défaut AWS suffisent :
aws ec2 describe-subnets --query "Subnets[].{Id:SubnetId,AZ:AvailabilityZone,CIDR:CidrBlock}"
```

---

## 2. Prérequis

- **Compte AWS** avec droits ECS, ECR, IAM, EC2 (VPC/ALB), CloudWatch, S3.
- **AWS CLI v2** configuré (`aws configure`, région par défaut `eu-west-1`).
- **Docker** installé et en cours d'exécution.
- Un **VPC** avec au moins 2 sous-réseaux publics (le VPC par défaut suffit pour démarrer).
- `jq` (optionnel, pratique pour parser la sortie AWS CLI).

Variables d'environnement utilisées dans ce guide :

```bash
export AWS_REGION=eu-west-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export APP_NAME=kojin
export ECR_REPO=${APP_NAME}-streamlit
export CLUSTER=${APP_NAME}-cluster
export SERVICE=${APP_NAME}-service
export TASK_FAMILY=${APP_NAME}-task
export CONTAINER_NAME=${APP_NAME}-web
export IMAGE_TAG=latest
export DATA_BUCKET=${APP_NAME}-data-${AWS_ACCOUNT_ID}
export DATA_KEY=products_names_with_macro_nutriments.csv
```

## 3. Bucket S3 et upload du CSV

### 3.1 Générer le CSV localement

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data_prep_nutriments.py
# → écrit data/products_names_with_macro_nutriments.csv
```

### 3.2 Créer le bucket

```bash
aws s3api create-bucket \
  --bucket ${DATA_BUCKET} \
  --region ${AWS_REGION} \
  --create-bucket-configuration LocationConstraint=${AWS_REGION}

aws s3api put-bucket-versioning \
  --bucket ${DATA_BUCKET} \
  --versioning-configuration Status=Enabled

aws s3api put-public-access-block \
  --bucket ${DATA_BUCKET} \
  --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
```

### 3.3 Uploader le CSV

```bash
aws s3 cp data/${DATA_KEY} s3://${DATA_BUCKET}/${DATA_KEY}
```

## 4. Préparer l'image Docker

### 4.1 Dockerfile

Le `Dockerfile` est à la racine du projet. Il copie le code et installe les dépendances Python :

```dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY streamlit_app.py kojin_common.py bento_editor.py data_prep_nutriments.py ./
COPY app_pages/ ./app_pages/
COPY .streamlit/ ./.streamlit/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
```

> Le CSV **n'est pas copié dans l'image** — il sera téléchargé depuis S3 au démarrage (voir §4.3).

### 4.2 Build

```bash
docker build -t ${APP_NAME}:${IMAGE_TAG} .
```

### 4.3 Build et test local

Le CSV **n'est pas copié dans l'image** : il sera téléchargé depuis S3 au démarrage par `streamlit_app.py` via `boto3` (voir `_ensure_csv_from_s3`). Le bucket S3 doit donc exister et contenir le CSV **avant** de lancer le conteneur — sinon l'app retourne une erreur 404 `HeadObject`.

**Vérifier que le CSV est bien uploadé :**

```bash
aws s3 ls s3://${DATA_BUCKET}/
```

Si le bucket est vide, uploader d'abord (voir §3.3).

**Builder et tester :**

```bash
docker build -t ${APP_NAME}:${IMAGE_TAG} .

docker run --rm -p 8501:8501 \
  -e DATA_S3_URI=s3://${DATA_BUCKET}/${DATA_KEY} \
  -e AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id) \
  -e AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key) \
  -e AWS_REGION=${AWS_REGION} \
  -e LLM_PROVIDER=groq \
  -e GROQ_API_KEY=<votre_clé_groq> \
  ${APP_NAME}:${IMAGE_TAG}
# → http://localhost:8501
```

En dev pur (sans S3), il suffit d'omettre `DATA_S3_URI` : l'app utilisera le CSV local dans `data/` s'il existe.

## 5. Créer et peupler le dépôt ECR

> **Astuce** : désactivez le pager CLI AWS pour éviter le mode `less` : `aws configure set cli_pager ""`

```bash-
ECR_TOKEN=$(aws ecr get-login-password --region ${AWS_REGION})
echo "$ECR_TOKEN" | docker login --username AWS --password-stdin \
  ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

export IMAGE_URI=${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}

docker tag ${APP_NAME}:${IMAGE_TAG} ${IMAGE_URI}
docker push ${IMAGE_URI}
```

## 6. Rôles IAM

### 6.1 Trust policy commune (réutilisée par les deux rôles)

```bash
cat > trust-ecs-tasks.json <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Service": "ecs-tasks.amazonaws.com" },
    "Action": "sts:AssumeRole"
  }]
}
EOF
```

### 6.2 Rôle d'exécution ECS (pull ECR, logs CloudWatch)

```bash
aws iam create-role \
  --role-name ${APP_NAME}-ecs-execution-role \
  --assume-role-policy-document file://trust-ecs-tasks.json

aws iam attach-role-policy \
  --role-name ${APP_NAME}-ecs-execution-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
```

### 6.3 Rôle de tâche (lecture S3 + invocation Bedrock depuis le conteneur)

Le conteneur a besoin de deux permissions :
- lire le CSV produits depuis S3 (`DATA_S3_URI`) ;
- invoquer **Amazon Bedrock Nova Micro** pour la page « Exploration des ingrédients » qui traduit les questions en langage naturel en SQL DuckDB via LangChain (voir section « Exploration en langage naturel » du README).

```bash
aws iam create-role \
  --role-name ${APP_NAME}-ecs-task-role \
  --assume-role-policy-document file://trust-ecs-tasks.json

cat > s3-read-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["s3:GetObject", "s3:ListBucket"],
    "Resource": [
      "arn:aws:s3:::${DATA_BUCKET}",
      "arn:aws:s3:::${DATA_BUCKET}/*"
    ]
  }]
}
EOF

aws iam put-role-policy \
  --role-name ${APP_NAME}-ecs-task-role \
  --policy-name ${APP_NAME}-s3-read \
  --policy-document file://s3-read-policy.json

cat > bedrock-invoke-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["bedrock:InvokeModel"],
    "Resource": "arn:aws:bedrock:*:${AWS_ACCOUNT_ID}:inference-profile/us.anthropic.claude-sonnet-4-6"
  }]
}
EOF

aws iam put-role-policy \
  --role-name ${APP_NAME}-ecs-task-role \
  --policy-name ${APP_NAME}-bedrock-invoke \
  --policy-document file://bedrock-invoke-policy.json
```

> **Modèles récents Bedrock** : depuis fin 2024, les modèles Anthropic/Amazon ne s'invoquent plus directement par leur `modelId` — il faut utiliser un **inference profile** (préfixe `eu.` pour l'Europe, `us.` pour les US). La page *Model Access* a été retirée ; les modèles s'activent automatiquement au premier appel.

### 6.3.2 Rôle de service Bedrock Fine-tuning (optionnel — si fine-tuning activé)

Si vous envisagez un **fine-tuning de modèles Bedrock** (ex. Nova Micro sur vos données), créez un rôle de service dédié que Bedrock utilisera pour lire les données d'entraînement en S3 :

```bash
aws iam create-role \
  --role-name BedrockFineTuneRole \
  --assume-role-policy-document file://trust-bedrock-finetune.json

cat > bedrock-finetune-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["s3:GetObject", "s3:ListBucket"],
    "Resource": [
      "arn:aws:s3:::kojin-finetune-\${AWS_ACCOUNT_ID}",
      "arn:aws:s3:::kojin-finetune-\${AWS_ACCOUNT_ID}/*"
    ]
  }]
}
EOF

aws iam put-role-policy \
  --role-name BedrockFineTuneRole \
  --policy-name bedrock-finetune-s3-read \
  --policy-document file://bedrock-finetune-policy.json
```

Ce rôle est passé en paramètre `--role-arn` du script `finetuning/launch_finetune.py` (voir **§6A** ci-dessous).

> **Région** : Le fine-tuning Amazon Nova Micro est disponible **uniquement en `us-east-1`**. Le bucket S3 doit être créé dans la même région (Bedrock ne peut pas lire les buckets cross-région) :
> ```bash
> aws s3api create-bucket --bucket kojin-finetune-${AWS_ACCOUNT_ID} --region us-east-1 \
>   --create-bucket-configuration LocationConstraint=us-east-1
> ```

### 6.3.1 Choisir et vérifier le modèle de référence

**Lister les inference profiles disponibles dans votre région :**

```bash
aws bedrock list-inference-profiles --region ${AWS_REGION} \
  --query "inferenceProfileSummaries[].{id:inferenceProfileId,arn:inferenceProfileArn}" \
  --output table
```

Les profils pertinents pour Kōjin (texte → SQL) :

| Inference Profile ID | Modèle | Qualité SQL | Coût (input/output) |
|---|---|---|---|
| `us.anthropic.claude-sonnet-4-6` | Claude Sonnet 4.6 | ⭐⭐⭐⭐⭐ | ~$3 / $15 per MTok |
| `eu.anthropic.claude-sonnet-4-5-20250929-v1:0` | Claude Sonnet 4.5 | ⭐⭐⭐⭐ | ~$3 / $15 per MTok |
| `eu.anthropic.claude-haiku-4-5-20251001-v1:0` | Claude Haiku 4.5 | ⭐⭐⭐ | ~$0.8 / $4 per MTok |
| `amazon.nova-pro-v1:0` | Nova Pro | ⭐⭐⭐ | ~$0.8 / $3.2 per MTok |
| `amazon.nova-micro-v1:0` | Nova Micro | ⭐⭐ | ~$0.035 / $0.14 per MTok |

Le modèle par défaut est **`us.anthropic.claude-sonnet-4-6`** (défini dans `kojin_common.py`). Pour le changer sans reconstruire l'image, modifiez `BEDROCK_MODEL_ID` dans la task definition.

**Tester l'accès au modèle avant déploiement :**

```bash
aws bedrock-runtime invoke-model \
  --model-id us.anthropic.claude-sonnet-4-6 \
  --region ${AWS_REGION} \
  --body $(echo -n '{"anthropic_version":"bedrock-2023-05-31","max_tokens":10,"messages":[{"role":"user","content":"Hi"}]}' | base64 -w0) \
  --content-type application/json \
  /tmp/test_bedrock.json && echo "✅ Accès OK" || echo "❌ Erreur — vérifier IAM"
```

**Changer le modèle de référence sans reconstruire l'image :**

```bash
# Mettre à jour la task definition avec le nouveau modèle
# 1. Modifier BEDROCK_MODEL_ID dans task-def.json
# 2. Ré-enregistrer
aws ecs register-task-definition --cli-input-json file://task-def.json

# 3. Forcer un redéploiement
aws ecs update-service \
  --cluster ${CLUSTER} --service ${SERVICE} \
  --force-new-deployment
```

### 6.4 Policy IAM élargie — référence + Meta Llama + modèles custom

Si vous utilisez un **deuxième agent Bedrock** (Llama fondation ou **Nova Micro fine-tuné** / importé), l’action `bedrock:InvokeModel` doit cibler les **ARN** réels visibles dans la console Bedrock (fondation, profil d’inférence, modèle personnalisé). Exemple **à resserrer en production** :

```json
cat > bedrock-invoke-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream"
    ],
    "Resource": [
      "arn:aws:bedrock:*::foundation-model/*",
      "arn:aws:bedrock:eu-west-1:985096928360:inference-profile/*",
      "arn:aws:bedrock:eu-west-1:985096928360:provisioned-model/*",
      "arn:aws:bedrock:eu-west-1:985096928360:custom-model/*"
    ]
  }]
}
EOF
```

Remplacez les placeholders par votre compte et région. Les préfixes exacts (`inference-profile`, `custom-model`, etc.) dépendent du type de déploiement Bedrock — **copiez les ARN** depuis la console après le fine-tuning ou l’import.

---
## 6.5 Récapitulatif complet : rôles IAM et permissions

Synthèse des 5 identités IAM du projet Kōjin — à créer et configurer avant déploiement.

### Vue d'ensemble

| Rôle | Principal Trust | Contexte |
|---|---|---|
| `kojin-ecs-execution-role` | `ecs-tasks.amazonaws.com` | Pull ECR + logs CloudWatch (infrastructure) |
| `kojin-ecs-task-role` | `ecs-tasks.amazonaws.com` | S3 read + Bedrock invoke (application) |
| `BedrockFineTuneRole` | `bedrock.amazonaws.com` | S3 read pour données fine-tuning (optionnel) |
| Identité CI/Humain | N/A (human/role IAM) | S3 upload + création job Bedrock (opérations) |

### **1. kojin-ecs-execution-role** — Infrastructure ECS/Fargate

**Trust policy** : `ecs-tasks.amazonaws.com`

**Permissions attachées** (policy managée AWS) :
- `arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy`

**Actions et ressources** :
```
Actions           : ecr:GetAuthorizationToken, ecr:BatchGetImage, ecr:GetDownloadUrlForLayer
                    logs:CreateLogStream, logs:PutLogEvents
Ressources        : ECR: arn:aws:ecr:${REGION}:${ACCOUNT}:repository/kojin-streamlit
                    CloudWatch: arn:aws:logs:${REGION}:${ACCOUNT}:log-group:/ecs/kojin:*
```

**Rôle** : Utilisé par le **runtime ECS** (pas le conteneur). Permet à ECS de tirer l'image Docker et d'envoyer les logs. Aucun accès aux ressources applicatives.

---

### **2. kojin-ecs-task-role** — Application Streamlit (conteneur)

**Trust policy** : `ecs-tasks.amazonaws.com`

**Permissions (2 policies inline)**

#### a) S3 read — CSV produits

```json
{
  "Effect": "Allow",
  "Action": ["s3:GetObject", "s3:ListBucket"],
  "Resource": [
    "arn:aws:s3:::kojin-data-${ACCOUNT}",
    "arn:aws:s3:::kojin-data-${ACCOUNT}/*"
  ]
}
```

**Objectif** : télécharger `products_names_with_macro_nutriments.csv` depuis S3 au démarrage (variable `DATA_S3_URI`).

#### b) Bedrock invoke — génération SQL

```json
{
  "Effect": "Allow",
  "Action": [
    "bedrock:InvokeModel",
    "bedrock:InvokeModelWithResponseStream"
  ],
  "Resource": [
    "arn:aws:bedrock:*::foundation-model/*",
    "arn:aws:bedrock:${REGION}:${ACCOUNT}:inference-profile/*",
    "arn:aws:bedrock:${REGION}:${ACCOUNT}:provisioned-model/*",
    "arn:aws:bedrock:${REGION}:${ACCOUNT}:custom-model/*"
  ]
}
```

**Objectif** : invoquer modèles Bedrock pour la page Exploration (question NL → SQL via LangChain).

**⚠️ Resserrement production** : remplacer les wildcards par les ARN exacts utilisés (ex. `inference-profile/us.anthropic.claude-sonnet-4-6`).

---

### **3. BedrockFineTuneRole** — Service Bedrock Fine-tuning (optionnel)

**Trust policy** : `bedrock.amazonaws.com` (⚠️ pas `ecs-tasks.amazonaws.com`)

**Permissions (policy inline)** :

```json
{
  "Effect": "Allow",
  "Action": ["s3:GetObject", "s3:ListBucket"],
  "Resource": [
    "arn:aws:s3:::kojin-finetune-${ACCOUNT}",
    "arn:aws:s3:::kojin-finetune-${ACCOUNT}/*"
  ]
}
```

**Objectif** : Passé en paramètre `--role-arn` au job `CreateModelCustomizationJob` de Bedrock. Permet au service Bedrock de lire train.jsonl et eval.jsonl depuis S3.

**Région** : Fine-tuning Nova Micro disponible **uniquement en `us-east-1`**. Bucket S3 doit être dans la même région.

---

### **4. Identité CI/Humain** — Script fine-tuning

**Type** : N/A (utilisateur humain via CLI ou rôle IAM CI/CD)

**Permissions nécessaires** :

| Action | Ressource | Contexte |
|---|---|---|
| `s3:PutObject` | `arn:aws:s3:::kojin-finetune-${ACCOUNT}/*` | Upload train.jsonl, eval.jsonl |
| `bedrock:CreateModelCustomizationJob` | `*` (ou ARN job Bedrock) | Créer job fine-tuning |
| `iam:PassRole` | `arn:aws:iam::${ACCOUNT}:role/BedrockFineTuneRole` | Passer le rôle à Bedrock |

**Objectif** : Exécuter `python3 -m finetuning.launch_finetune`. Utilisée **uniquement en opérations**, jamais en production.

---

### Bonnes pratiques IAM

✅ **Séparation des rôles** : exécution ECS (tier infra) ≠ tâche ECS (tier applicatif)

✅ **Principle of least privilege** : chaque rôle n'a que les actions minimales nécessaires

✅ **Resserrement Bedrock** : remplacer wildcards par ARN exacts en production

✅ **Audit trail** : activer CloudTrail sur les actions `bedrock:InvokeModel` et `bedrock:CreateModelCustomizationJob`

---

---
## 6A. Fine-tuner Amazon Nova Micro sur Bedrock (texte → SQL)

Objectif : produire un **modèle personnalisé** invoqué comme le modèle de base, mais spécialisé sur vos paires (question + schéma `products` → SQL DuckDB).

Le répertoire `finetuning/` automatise l'intégralité du pipeline.

### A.1 Données d'entraînement

```bash
python3 -m finetuning.generate_dataset
# → finetuning/data/train.jsonl  (~141 exemples)
# → finetuning/data/eval.jsonl   (97 exemples)
```

Le script génère des paires supervisées au format **`bedrock-conversation-2023`** requis par Nova : `schemaVersion` + `system` (top-level) + `messages` avec `user`/`assistant` (le rôle `system` **ne doit pas** être dans `messages`). Le prompt système inclut le schéma complet de la table `products` et les règles SQL.

Références AWS : [Prepare data for fine-tuning](https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-prepare.html), [Customize a model with fine-tuning](https://docs.aws.amazon.com/bedrock/latest/userguide/custom-model-fine-tuning.html).

### A.2 Lancer la personnalisation

**Prérequis** : Créer le rôle `BedrockFineTuneRole` (voir **§6.3.2** ci-dessus) et le bucket S3 de fine-tuning en `us-east-1`.

```bash
python3 -m finetuning.launch_finetune \
  --s3-bucket kojin-finetune-${AWS_ACCOUNT_ID} \
  --role-arn arn:aws:iam::${AWS_ACCOUNT_ID}:role/BedrockFineTuneRole \
  --region us-east-1
```

Le script :
1. Upload `train.jsonl` et `eval.jsonl` vers `s3://kojin-finetune-${AWS_ACCOUNT_ID}/bedrock-finetune/`.
2. Appelle `CreateModelCustomizationJob` avec le modèle de base **Amazon Nova Micro** (`amazon.nova-micro-v1:0:128k`).
3. Affiche le `jobArn` — suivez la progression via la console **Bedrock → Custom models** ou `GetModelCustomizationJob`.

> **Région** : Le fine-tuning Nova Micro est **uniquement disponible en `us-east-1`**. La région du bucket S3 doit être alignée (voir **§6.3.2**).

### A.3 Évaluer le modèle fine-tuné en le comparant avec un modèle de référence

> **Prérequis** : pour invoquer un custom model Bedrock, un **Provisioned Throughput** est nécessaire. Les nouveaux comptes AWS doivent ouvrir un ticket support : https://console.aws.amazon.com/support/home — *"Request to enable Provisioned Throughput for Bedrock custom models"*.
> En attendant, évaluez **Nova Micro de base vs Claude** pour établir une base de comparaison.

Récupérer l'ARN du modèle fine-tuné :

```bash
aws bedrock get-model-customization-job \
  --job-identifier $(python3 -c "import json; print(json.load(open('finetuning/data/last_job.json'))['job_arn'])") \
  --region us-east-1 \
  --query '{status:status,modelArn:outputModelArn}' --output table
```

Une fois le Provisioned Throughput accordé et l'ARN disponible :

```bash
source myenv/bin/activate
python3 -m finetuning.evaluate \
  --finetuned-model-id us.amazon.nova-micro-v1:0 \
  --reference-model-id us.anthropic.claude-sonnet-4-6 \
  --region us-east-1 \
  --max-samples 3 2>&1
```

**Sans Provisioned Throughput** (Nova Micro de base vs Claude, pour valider le pipeline) :

```bash
source myenv/bin/activate
python3 -m finetuning.evaluate \
  --finetuned-model-id us.amazon.nova-micro-v1:0 \
  --reference-model-id us.anthropic.claude-sonnet-4-6 \
  --region us-east-1
```

Le script :
- Exécute chaque question de `eval.jsonl` sur les deux modèles ;
- Valide le SQL généré via DuckDB (exécution réelle) ;
- Compare : **taux de réussite SQL**, **exact match**, **latence moyenne** ;
- Sauvegarde les résultats dans `finetuning/data/eval_results.json`.


### A.4 Brancher Kōjin sur le Nova Micro fine-tuné

Une fois le fine-tuning terminé et le modèle évalué (§A.3), configurez-le comme **modèle de référence** (seul modèle utilisé en production) :

```bash
# Récupérer l'ARN / inference-profile-id retourné par Bedrock après le fine-tuning
# Exemple : arn:aws:bedrock:us-east-1:985096928360:inference-profile/us.meta.llama3-1-8b-...
```

Mettre à jour `BEDROCK_MODEL_ID` dans `task-def.json` :

```json
{ "name": "BEDROCK_MODEL_ID", "value": "<votre-llama-finetuné-inference-profile-id>" }
```

Puis redéployer :

```bash
aws ecs register-task-definition --cli-input-json file://task-def.json
aws ecs update-service --cluster ${CLUSTER} --service ${SERVICE} --force-new-deployment
```

> **Architecture** : le Nova Micro fine-tuné est le **seul modèle utilisé en production** pour la page Exploration. Claude Sonnet 4.6 (`us.anthropic.claude-sonnet-4-6`) sert uniquement de **référence d’évaluation** dans le script `finetuning/evaluate` (jamais appelé en prod). Tant que `BEDROCK_MODEL_ID` est vide, la page Exploration affiche un message indiquant qu’aucun modèle n’est configuré.

**Comparaison optionnelle (mode dev)** : définir `BEDROCK_COMPARE_MODEL_ID` pour afficher côte-à-côte le Nova Micro fine-tuné et un autre modèle dans l’UI Exploration via la checkbox **Comparer les deux agents**.

---

## 6B. Monitoring : écart de performance entre les deux agents

### B.1 Dans l’interface Streamlit

La page **Exploration** affiche pour chaque requête (mode comparaison) :

- **Temps de génération SQL** (ms) par agent ;
- **Temps d’exécution DuckDB** (ms) ;
- **Égalité textuelle** des deux SQL (indicateur rapide, pas une preuve sémantique) ;
- Un **historique de session** (tableau des dernières exécutions).

### B.2 Journal NDJSON (ECS / analyse offline)

Activez les variables suivantes sur la **task definition** (ou en local) :

| Variable | Exemple | Rôle |
|--------|---------|------|
| `KOJIN_EXPLORATION_LOG_JSONL` | `1` | Si truthy, chaque clic « Interroger » append une ligne JSON. |
| `KOJIN_EXPLORATION_LOG_PATH` | `/tmp/kojin_exploration_metrics.ndjson` | Fichier NDJSON (défaut : `/tmp/...`). |

Chaque ligne contient notamment : `mode` (`single` / `compare`), `llm_provider_ref` (`groq` / `bedrock`), `model_ref`, `model_compare`, latences SQL et DuckDB, `duckdb_ok_*`, `rows_*`, `sql_text_equal`. Vous pouvez :

- **Copier** le fichier vers S3 via un sidecar ou un cron ;
- **Parser** dans QuickSight / Athena / notebook ;
- **Émettre des métriques** CloudWatch avec un Lambda ou le **CloudWatch agent** sur un pattern de log.

### B.3 CloudWatch et coûts

- **Logs** : le group `/ecs/${APP_NAME}` contient déjà stdout/stderr Streamlit — si vous `print` les événements (optionnel), filtrez par préfixe.
- **Bedrock** : [Amazon Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/) — comparer coût par token **référence** vs **Llama custom** dans **Cost Explorer** en filtrant par modèle / tag (si vous taguez les workloads).

---

## 6C. Variables d’environnement LLM (récapitulatif conteneur)

| Variable | Obligatoire | Description |
|----------|-------------|-------------|
| `AWS_REGION` | oui | Région Bedrock si la **référence** utilise Bedrock (alignée model access IAM). |
| `DATA_S3_URI` | prod | URI S3 du CSV produits. |
| `LLM_PROVIDER` | non | `auto` \| `openai` \| `groq` \| `bedrock`. Défaut `auto` : OpenAI si `OPENAI_API_KEY`, sinon Groq si `GROQ_API_KEY`, sinon Bedrock. En pile **ECS pure AWS**, fixer **`bedrock`**. |
| `OPENAI_API_KEY` | si OpenAI | Clé depuis **platform.openai.com** ; en ECS préférez **Secrets Manager**. |
| `OPENAI_MODEL_ID` | non | Modèle OpenAI (défaut : `gpt-4o`). |
| `GROQ_API_KEY` | si Groq | Clé depuis **Groq Console** ; en ECS préférez **Secrets Manager**. |
| `GROQ_MODEL_ID` | non | Modèle Groq (`llama-3.1-8b-instant`, etc.). Requiert egress HTTPS vers `api.groq.com`. |
| `BEDROCK_MODEL_ID` | oui si réf. Bedrock | Modèle **référence** Exploration (Nova Micro fine-tuné en prod). **Laisser vide** tant que le fine-tuning n’est pas terminé — la page Exploration sera désactivée. |
| `BEDROCK_COMPARE_MODEL_ID` | non | Second modèle Bedrock (ex. Nova Micro fine-tuné) pour la comparaison. |
| `OPENAI_COMPAT_BASE_URL` / `OPENAI_COMPAT_MODEL` | non | Alternative au second Bedrock (serveur OpenAI-compatible). |
| `KOJIN_EXPLORATION_LOG_JSONL` | non | `1` pour activer le fichier NDJSON. |
| `KOJIN_EXPLORATION_LOG_PATH` | non | Chemin du fichier de métriques. |

## 7. Cluster ECS, Task Definition, Service

### 7.1 Cluster + log group

```bash
# --capacity-providers est omis volontairement : Fargate est disponible par défaut
# et le flag déclenche une erreur de service-linked role sur les comptes récents
aws ecs create-cluster \
  --cluster-name ${CLUSTER} \
  --region ${AWS_REGION}

aws logs create-log-group \
  --log-group-name /ecs/${APP_NAME} \
  --region ${AWS_REGION}
```

### 7.2 Task definition (`task-def.json`)

Générer et enregistrer directement (les variables d'env du §2 doivent être définies) :

```bash
cat > task-def.json << EOF
{
  "family": "${TASK_FAMILY}",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/${APP_NAME}-ecs-execution-role",
  "taskRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/${APP_NAME}-ecs-task-role",
  "containerDefinitions": [
    {
      "name": "${CONTAINER_NAME}",
      "image": "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}",
      "essential": true,
      "portMappings": [{ "containerPort": 8501, "protocol": "tcp" }],
      "environment": [
        { "name": "STREAMLIT_SERVER_HEADLESS", "value": "true" },
        { "name": "AWS_REGION",               "value": "${AWS_REGION}" },
        { "name": "DATA_S3_URI",              "value": "s3://${DATA_BUCKET}/${DATA_KEY}" },
        { "name": "LLM_PROVIDER",             "value": "bedrock" },
        { "name": "BEDROCK_MODEL_ID",         "value": "us.anthropic.claude-sonnet-4-6" },
        { "name": "BEDROCK_COMPARE_MODEL_ID", "value": "" },
        { "name": "KOJIN_EXPLORATION_LOG_JSONL", "value": "0" },
        { "name": "KOJIN_EXPLORATION_LOG_PATH",  "value": "/tmp/kojin_exploration_metrics.ndjson" }
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -fsS http://localhost:8501/_stcore/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      },
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/${APP_NAME}",
          "awslogs-region": "${AWS_REGION}",
          "awslogs-stream-prefix": "web"
        }
      }
    }
  ]
}
EOF

aws ecs register-task-definition --cli-input-json file://task-def.json \
  --query 'taskDefinition.{family:family,revision:revision,status:status}' \
  --output table
```

> **Sizing** : 1 vCPU / 4 Go. Polars + scipy + le CSV en cache Streamlit tournent confortablement à ce niveau. Pour > 20 utilisateurs simultanés, passer à 2 vCPU / 8 Go ou scaler horizontalement (§7.5).

### 7.3 Application Load Balancer

Streamlit utilise des **WebSockets** — l'ALB les supporte nativement, mais il faut activer les **sticky sessions** pour qu'une session utilisateur reste collée à la même task.

```bash
# Sous-réseaux publics du VPC par défaut
SUBNETS=$(aws ec2 describe-subnets \
  --filters "Name=default-for-az,Values=true" \
  --query "Subnets[].SubnetId" --output text | tr '\t' ',')

VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" \
  --query "Vpcs[0].VpcId" --output text)

# Security group ALB (80/443 ouverts au monde)
ALB_SG=$(aws ec2 create-security-group \
  --group-name ${APP_NAME}-alb-sg \
  --description "ALB SG" --vpc-id ${VPC_ID} \
  --query GroupId --output text)
aws ec2 authorize-security-group-ingress --group-id ${ALB_SG} \
  --protocol tcp --port 80 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id ${ALB_SG} \
  --protocol tcp --port 443 --cidr 0.0.0.0/0

# Security group service (8501 uniquement depuis l'ALB)
SVC_SG=$(aws ec2 create-security-group \
  --group-name ${APP_NAME}-svc-sg \
  --description "Service SG" --vpc-id ${VPC_ID} \
  --query GroupId --output text)
aws ec2 authorize-security-group-ingress --group-id ${SVC_SG} \
  --protocol tcp --port 8501 --source-group ${ALB_SG}

# ALB
ALB_ARN=$(aws elbv2 create-load-balancer \
  --name ${APP_NAME}-alb \
  --subnets $(echo ${SUBNETS} | tr ',' ' ') \
  --security-groups ${ALB_SG} \
  --scheme internet-facing --type application \
  --query 'LoadBalancers[0].LoadBalancerArn' --output text)

# Target group (IP targets, sticky sessions + healthcheck Streamlit)
TG_ARN=$(aws elbv2 create-target-group \
  --name ${APP_NAME}-tg \
  --protocol HTTP --port 8501 --target-type ip \
  --vpc-id ${VPC_ID} \
  --health-check-path /_stcore/health \
  --health-check-interval-seconds 30 \
  --healthy-threshold-count 2 \
  --query 'TargetGroups[0].TargetGroupArn' --output text)

aws elbv2 modify-target-group-attributes \
  --target-group-arn ${TG_ARN} \
  --attributes \
     Key=stickiness.enabled,Value=true \
     Key=stickiness.type,Value=lb_cookie \
     Key=stickiness.lb_cookie.duration_seconds,Value=86400 \
     Key=deregistration_delay.timeout_seconds,Value=30

# Listener HTTP (en prod, ajouter un listener 443 avec certificat ACM — §11)
aws elbv2 create-listener \
  --load-balancer-arn ${ALB_ARN} \
  --protocol HTTP --port 80 \
  --default-actions Type=forward,TargetGroupArn=${TG_ARN}
```

### 7.4 Service ECS

```bash
aws ecs create-service \
  --cluster ${CLUSTER} \
  --service-name ${SERVICE} \
  --task-definition ${TASK_FAMILY} \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[${SUBNETS}],securityGroups=[${SVC_SG}],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=${TG_ARN},containerName=${CONTAINER_NAME},containerPort=8501" \
  --health-check-grace-period-seconds 60
```

Récupérer l'URL publique :

```bash
aws elbv2 describe-load-balancers \
  --load-balancer-arns ${ALB_ARN} \
  --query 'LoadBalancers[0].DNSName' --output text
# → http://kojin-alb-XXXXX.eu-west-1.elb.amazonaws.com
```

### 7.5 Auto-scaling (optionnel)

```bash
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --resource-id service/${CLUSTER}/${SERVICE} \
  --scalable-dimension ecs:service:DesiredCount \
  --min-capacity 1 --max-capacity 4

aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --resource-id service/${CLUSTER}/${SERVICE} \
  --scalable-dimension ecs:service:DesiredCount \
  --policy-name ${APP_NAME}-cpu-scale \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 60.0,
    "PredefinedMetricSpecification": {"PredefinedMetricType": "ECSServiceAverageCPUUtilization"},
    "ScaleInCooldown": 300,
    "ScaleOutCooldown": 120
  }'
```

> Avec sticky sessions, un utilisateur reste collé à une task ; lors d'un scale-out, seules les **nouvelles sessions** bénéficient des nouvelles tasks.

## 8. Accéder à l'application

Ouvrir dans le navigateur l'URL publique retournée au §7.4. Pour une mise en prod, passer à HTTPS (§11).

## 9. Mises à jour

### 9.1 Mise à jour de l'application (code)

```bash
docker build -t ${APP_NAME}:${IMAGE_TAG} .
docker tag ${APP_NAME}:${IMAGE_TAG} ${IMAGE_URI}
docker push ${IMAGE_URI}

aws ecs update-service \
  --cluster ${CLUSTER} --service ${SERVICE} \
  --force-new-deployment
```

En CI (GitHub Actions par ex.), préférer des tags immuables (`sha-$(git rev-parse --short HEAD)`) et mettre à jour la task definition à chaque push.

### 9.2 Rafraîchissement des données (CSV)

Le CSV est téléchargé **au démarrage de chaque task** : il suffit de le remplacer dans S3 puis de redémarrer le service.

```bash
# 1. Regénérer le CSV localement
python data_prep_nutriments.py

# 2. Uploader dans S3 (versioning activé → historique conservé)
aws s3 cp data/${DATA_KEY} s3://${DATA_BUCKET}/${DATA_KEY}

# 3. Forcer le redéploiement (les nouvelles tasks retéléchargent le CSV)
aws ecs update-service \
  --cluster ${CLUSTER} --service ${SERVICE} \
  --force-new-deployment
```

Aucune reconstruction d'image n'est nécessaire. Le versioning S3 permet un retour arrière rapide (`aws s3api copy-object` avec l'ancien `VersionId`).

## 10. Observabilité

- **Logs** : CloudWatch Logs `/ecs/kojin`, stream `web/*`.
- **Exploration (deux agents)** : activer `KOJIN_EXPLORATION_LOG_JSONL=1` et un `KOJIN_EXPLORATION_LOG_PATH` persistent ou shipper vers S3 (voir §6B). Chaque requête enrichit un NDJSON exploitable pour comparer latences et succès DuckDB.
- **Métriques ECS** : `CPUUtilization`, `MemoryUtilization` (namespace `AWS/ECS`).
- **Métriques ALB** : `TargetResponseTime`, `HTTPCode_Target_5XX_Count`, `UnHealthyHostCount`.
- **Coûts LLM** : filtrer **Bedrock** dans Cost Explorer par modèle / région pour comparer **référence** vs **Nova Micro fine-tuné** (`BEDROCK_COMPARE_MODEL_ID`).
- **Alarmes recommandées** :
  - CPU > 80 % pendant 10 min
  - Mémoire > 85 % pendant 5 min
  - 5xx > 1 % des requêtes
  - `UnHealthyHostCount >= 1` pendant 3 min

## 11. HTTPS + domaine (production)

1. Enregistrer un domaine (Route 53 ou externe).
2. Demander un certificat ACM dans la région de l'ALB.
3. Ajouter un listener HTTPS 443 à l'ALB, rediriger 80 → 443.
4. Créer un enregistrement ALIAS vers le DNS de l'ALB.

## 12. Coûts estimés (eu-west-1, ordre de grandeur)

| Ressource | Conf. | Coût mensuel |
|---|---|---|
| Fargate 1 vCPU / 4 Go, 24/7 | 1 task | ~30 € |
| ALB | 1 ALB, trafic faible | ~18 € |
| ECR | < 1 Go stocké | < 1 € |
| CloudWatch Logs | ~1 Go | ~0.6 € |
| S3 | < 1 Go (CSV + versions) | < 0.1 € |
| **Total** | | **~50 €/mois** |

Pour une démo **coupée la nuit**, un `scheduled scaling` ramenant `desiredCount` à 0 la nuit et le week-end divise la facture compute par ~3.

## 13. Nettoyage

```bash
aws ecs update-service --cluster ${CLUSTER} --service ${SERVICE} --desired-count 0
aws ecs delete-service --cluster ${CLUSTER} --service ${SERVICE} --force
aws elbv2 delete-listener     --listener-arn <LISTENER_ARN>
aws elbv2 delete-target-group --target-group-arn ${TG_ARN}
aws elbv2 delete-load-balancer --load-balancer-arn ${ALB_ARN}
aws ecs delete-cluster --cluster ${CLUSTER}
aws logs delete-log-group --log-group-name /ecs/${APP_NAME}
aws ecr delete-repository --repository-name ${ECR_REPO} --force
aws s3 rm s3://${DATA_BUCKET} --recursive
aws s3api delete-bucket --bucket ${DATA_BUCKET}
```

## 14. Checklist de mise en prod

- [ ] Bucket S3 créé, versioning activé, accès public bloqué.
- [ ] CSV généré et uploadé dans le bucket.
- [ ] Image buildée (sans CSV), taggée `sha-*` et pushée sur ECR.
- [ ] Rôle de tâche : S3 read **et** `bedrock:InvokeModel` sur les ARN référence + comparaison (§6.4).
- [ ] Bedrock : tester l'accès au modèle `BEDROCK_MODEL_ID` avec `invoke-model` (§6.3.1) — les modèles s'activent automatiquement au premier appel depuis fin 2024.
- [ ] Task definition : `DATA_S3_URI`, `BEDROCK_MODEL_ID`, fac. `BEDROCK_COMPARE_MODEL_ID`, fac. logs `KOJIN_EXPLORATION_*`.
- [ ] ALB avec target group `/_stcore/health`, stickiness activée.
- [ ] Security groups restreints (8501 uniquement depuis l'ALB).
- [ ] Listener HTTPS avec certificat ACM, redirection 80 → 443.
- [ ] Alarmes CloudWatch configurées.
- [ ] Auto-scaling si > 10 utilisateurs simultanés attendus.

## 15. Commandes de diagnostic et d'inspection

Ces commandes permettent d'observer le routage ALB -> target group -> tasks ECS, puis d'inspecter les données S3 et DuckDB utilisées par l'application.

### 15.1 Voir comment ECS et l'ALB gèrent les requêtes

Récupérer les ARN utiles :

```bash
ALB_ARN=$(aws elbv2 describe-load-balancers \
  --names ${APP_NAME}-alb \
  --query 'LoadBalancers[0].LoadBalancerArn' --output text)

TG_ARN=$(aws elbv2 describe-target-groups \
  --names ${APP_NAME}-tg \
  --query 'TargetGroups[0].TargetGroupArn' --output text)

echo "ALB_ARN=${ALB_ARN}"
echo "TG_ARN=${TG_ARN}"
```

Voir l'état du service ECS :

```bash
aws ecs describe-services \
  --cluster ${CLUSTER} \
  --services ${SERVICE} \
  --query 'services[0].{desired:desiredCount,running:runningCount,pending:pendingCount,status:status,events:events[0:5]}' \
  --output yaml
```

Lister les tasks du service puis inspecter leurs IP privées :

```bash
TASKS=$(aws ecs list-tasks \
  --cluster ${CLUSTER} \
  --service-name ${SERVICE} \
  --query 'taskArns' --output text)

aws ecs describe-tasks \
  --cluster ${CLUSTER} \
  --tasks ${TASKS} \
  --query 'tasks[].{task:taskArn,last:lastStatus,health:healthStatus,ips:attachments[0].details[?name==`privateIPv4Address`].value | [0]}' \
  --output table
```

Voir vers quelles targets l'ALB envoie le trafic :

```bash
aws elbv2 describe-target-health \
  --target-group-arn ${TG_ARN} \
  --query 'TargetHealthDescriptions[].{ip:Target.Id,port:Target.Port,state:TargetHealth.State,reason:TargetHealth.Reason,description:TargetHealth.Description}' \
  --output table
```

Suivre les logs applicatifs pour observer les requêtes côté conteneur :

```bash
aws logs tail /ecs/${APP_NAME} --since 30m --follow
```

Lire quelques métriques ALB utiles pour comprendre la charge et les erreurs :

```bash
aws cloudwatch get-metric-statistics \
  --namespace AWS/ApplicationELB \
  --metric-name TargetResponseTime \
  --dimensions Name=LoadBalancer,Value=$(echo ${ALB_ARN} | cut -d: -f6-) \
  --statistics Average p95 Maximum \
  --period 300 \
  --start-time $(date -u -d '1 hour ago' +%FT%TZ) \
  --end-time $(date -u +%FT%TZ)
```

```bash
aws cloudwatch get-metric-statistics \
  --namespace AWS/ApplicationELB \
  --metric-name UnHealthyHostCount \
  --dimensions Name=TargetGroup,Value=$(echo ${TG_ARN} | cut -d: -f6-) Name=LoadBalancer,Value=$(echo ${ALB_ARN} | cut -d: -f6-) \
  --statistics Maximum \
  --period 300 \
  --start-time $(date -u -d '1 hour ago' +%FT%TZ) \
  --end-time $(date -u +%FT%TZ)
```

### 15.2 Requêter et inspecter S3

Lister les objets du bucket :

```bash
aws s3 ls s3://${DATA_BUCKET}/
```

Voir les métadonnées du CSV :

```bash
aws s3api head-object \
  --bucket ${DATA_BUCKET} \
  --key ${DATA_KEY}
```

Télécharger un aperçu local puis afficher les premières lignes :

```bash
aws s3 cp s3://${DATA_BUCKET}/${DATA_KEY} /tmp/${DATA_KEY}
head -20 /tmp/${DATA_KEY}
```

Si le versioning est activé, voir l'historique des versions du CSV :

```bash
aws s3api list-object-versions \
  --bucket ${DATA_BUCKET} \
  --prefix ${DATA_KEY} \
  --query '{versions:Versions[].{version:VersionId,last_modified:LastModified,size:Size,is_latest:IsLatest}}' \
  --output table
```

### 15.3 Requêter DuckDB localement (dev)

Depuis l'environnement virtuel du projet :

```bash
source myenv/bin/activate
```

Construire la base DuckDB locale si elle n'existe pas encore :

```bash
python - <<'PY'
import duckdb
from kojin_common import CSV_PATH, DUCKDB_PATH, DUCKDB_TABLE

con = duckdb.connect(DUCKDB_PATH)
con.execute(f"CREATE OR REPLACE TABLE {DUCKDB_TABLE} AS SELECT * FROM read_csv_auto('{CSV_PATH}', sample_size=-1)")
con.close()
print(DUCKDB_PATH)
PY
```

Afficher le schéma de la table `products` :

```bash
python - <<'PY'
import duckdb
from kojin_common import DUCKDB_PATH

con = duckdb.connect(DUCKDB_PATH, read_only=True)
print(con.execute("DESCRIBE products").fetchdf().to_string(index=False))
con.close()
PY
```

Exécuter une requête simple :

```bash
python - <<'PY'
import duckdb
from kojin_common import DUCKDB_PATH

sql = '''
SELECT product_name, proteins, "energy-kcal"
FROM products
WHERE proteins > 20
ORDER BY proteins DESC
LIMIT 10
'''

con = duckdb.connect(DUCKDB_PATH, read_only=True)
print(con.execute(sql).fetchdf().to_string(index=False))
con.close()
PY
```

Compter les lignes et vérifier quelques agrégats :

```bash
python - <<'PY'
import duckdb
from kojin_common import DUCKDB_PATH

con = duckdb.connect(DUCKDB_PATH, read_only=True)
print(con.execute('SELECT COUNT(*) AS n_rows, ROUND(AVG(proteins), 2) AS avg_proteins, ROUND(AVG("energy-kcal"), 2) AS avg_kcal FROM products').fetchdf().to_string(index=False))
con.close()
PY
```

### 15.4 Requêter DuckDB et S3 **depuis une task ECS en production**

Tu peux inspecter directement le CSV en S3 et les requêtes DuckDB en cours d'exécution dans la task ECS via **AWS Systems Manager Session Manager** (intégré à `aws ecs execute-command`).

#### Prérequis

La task ECS doit avoir les permissions IAM pour Session Manager. Ajouter cette policy au rôle de tâche (`kojin-ecs-task-role`) :

```bash
cat > ecs-ssm-exec-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": [
      "ssmmessages:CreateControlChannel",
      "ssmmessages:CreateDataChannel",
      "ssmmessages:OpenControlChannel",
      "ssmmessages:OpenDataChannel"
    ],
    "Resource": "*"
  }]
}
EOF

aws iam put-role-policy \
  --role-name ${APP_NAME}-ecs-task-role \
  --policy-name ${APP_NAME}-ecs-ssm-exec \
  --policy-document file://ecs-ssm-exec-policy.json
```

Puis redéployer la task :

```bash
aws ecs update-service \
  --cluster ${CLUSTER} --service ${SERVICE} \
  --force-new-deployment
```

#### Ouvrir un shell interactif dans la task

Récupérer l'ARN de la task en cours d'exécution :

```bash
TASK_ARN=$(aws ecs list-tasks \
  --cluster ${CLUSTER} \
  --service-name ${SERVICE} \
  --query 'taskArns[0]' --output text)

echo "Task: ${TASK_ARN}"
```

Lancer un shell bash interactif dans le conteneur :

```bash
aws ecs execute-command \
  --cluster ${CLUSTER} \
  --task ${TASK_ARN} \
  --container ${CONTAINER_NAME} \
  --command "/bin/bash" \
  --interactive
```

Tu es maintenant **à l'intérieur du conteneur** en production. Tu peux y exécuter n'importe quelle commande disponible.

#### Vérifier le CSV depuis S3 (dans le shell ECS)

```bash
# Voir la taille et la date du CSV dans S3
aws s3api head-object \
  --bucket ${DATA_BUCKET} \
  --key ${DATA_KEY}

# Télécharger un aperçu et compter les lignes
aws s3 cp s3://${DATA_BUCKET}/${DATA_KEY} /tmp/${DATA_KEY}
wc -l /tmp/${DATA_KEY}
head -5 /tmp/${DATA_KEY}
```

#### Requêter DuckDB depuis la task (en production)

```bash
# Exécuter les mêmes commandes Python que la section 15.3
python3 - <<'PY'
import duckdb
from kojin_common import DUCKDB_PATH

con = duckdb.connect(DUCKDB_PATH, read_only=True)
rows = con.execute("SELECT COUNT(*) AS n_rows FROM products").fetchone()
print(f"Nombre de lignes : {rows[0]}")

# Requête plus complète
result = con.execute('''
  SELECT 
    COUNT(*) AS n_rows,
    ROUND(AVG(proteins), 2) AS avg_proteins,
    ROUND(AVG("energy-kcal"), 2) AS avg_kcal,
    MIN(proteins) AS min_proteins,
    MAX(proteins) AS max_proteins
  FROM products
''').fetchdf()
print(result.to_string(index=False))
con.close()
PY
```

Vérifier le schéma de la table `products` :

```bash
python3 - <<'PY'
import duckdb
from kojin_common import DUCKDB_PATH

con = duckdb.connect(DUCKDB_PATH, read_only=True)
schema = con.execute("DESCRIBE products").fetchdf()
print(schema.to_string(index=False))
con.close()
PY
```

Tester une requête NL → SQL générée récemment (si `KOJIN_EXPLORATION_LOG_JSONL=1`, voir §6C) :

```bash
# Regarder les requêtes loggées en NDJSON
tail -20 /tmp/kojin_exploration_metrics.ndjson | python3 -m json.tool | head -50

# Ou exécuter une requête manuelle pour tester
python3 - <<'PY'
import duckdb
from kojin_common import DUCKDB_PATH

sql = "SELECT product_name, proteins FROM products WHERE proteins > 25 LIMIT 5"
con = duckdb.connect(DUCKDB_PATH, read_only=True)
print(con.execute(sql).fetchdf().to_string(index=False))
con.close()
PY
```

#### Voir les logs applicatifs Streamlit (dans le shell ECS)

```bash
# Les logs Streamlit sont envoyés à stdout
# Voir les 100 dernières lignes capturées
tail -100 ~/.streamlit/logs/streamlit.log
```

#### Quitter le shell et revenir au terminal local

```bash
exit
```

#### Automatiser une commande unique (sans shell interactif)

Si tu ne veux pas d'interactivité, exécuter une seule commande :

```bash
aws ecs execute-command \
  --cluster ${CLUSTER} \
  --task ${TASK_ARN} \
  --container ${CONTAINER_NAME} \
  --command "python3 -c \"import duckdb; con = duckdb.connect('/tmp/products.duckdb', read_only=True); print(con.execute('SELECT COUNT(*) FROM products').fetchone())\"" \
  --output text
```

#### Troubleshooting : si execute-command échoue

1. **Vérifier que la task est en cours d'exécution** :
   ```bash
   aws ecs describe-tasks --cluster ${CLUSTER} --tasks ${TASK_ARN} \
     --query 'tasks[0].{lastStatus:lastStatus,desiredStatus:desiredStatus}' --output table
   ```

2. **Vérifier l'agent ECS Exec dans l'image Docker** : le Dockerfile doit avoir `/bin/bash` ou `/bin/sh` (ils y sont par défaut en Python slim).

3. **Vérifier la policy IAM** : la tâche doit avoir les permissions `ssmmessages:*` (cf. prérequis ci-dessus).

4. **Vérifier la connectivité réseau** : la task ECS doit accéder à `ssmmessages.*.amazonaws.com` (port 443). Si derrière un NAT/proxy, ajouter une route.

---
