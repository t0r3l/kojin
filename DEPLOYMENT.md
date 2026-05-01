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

### 4.1 `Dockerfile`

Créer un `Dockerfile` à la racine. Le CSV **n'est pas copié dans l'image** : il sera téléchargé depuis S3 au démarrage par `streamlit_app.py` via `boto3` (voir `_ensure_csv_from_s3`).

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

COPY streamlit_app.py data_prep_nutriments.py ./

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
```

### 4.2 `.dockerignore`

```
.git
.venv
venv
__pycache__
*.pyc
README*.md
DEPLOYMENT.md
data/
```

> `data/` est entièrement ignoré : pas de CSV ni de parquet dans l'image.

### 4.3 Build et test local

Pour tester en local avec S3, fournir des credentials AWS (profil ou variables) et la variable `DATA_S3_URI` :

```bash
docker build -t ${APP_NAME}:${IMAGE_TAG} .
docker run --rm -p 8501:8501 \
  -e DATA_S3_URI=s3://${DATA_BUCKET}/${DATA_KEY} \
  -e AWS_ACCESS_KEY_ID=... \
  -e AWS_SECRET_ACCESS_KEY=... \
  -e AWS_REGION=${AWS_REGION} \
  ${APP_NAME}:${IMAGE_TAG}
# → http://localhost:8501
```

En dev pur (sans S3), il suffit d'omettre `DATA_S3_URI` : l'app utilisera le CSV local dans `data/` s'il existe.

## 5. Créer et peupler le dépôt ECR

```bash
aws ecr create-repository \
  --repository-name ${ECR_REPO} \
  --region ${AWS_REGION} \
  --image-scanning-configuration scanOnPush=true

aws ecr get-login-password --region ${AWS_REGION} \
  | docker login --username AWS --password-stdin \
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
    "Resource": "arn:aws:bedrock:*::foundation-model/amazon.nova-micro-v1:0"
  }]
}
EOF

aws iam put-role-policy \
  --role-name ${APP_NAME}-ecs-task-role \
  --policy-name ${APP_NAME}-bedrock-invoke \
  --policy-document file://bedrock-invoke-policy.json
```

> **Note :** dans la console AWS **Bedrock → Model access** de la région `${AWS_REGION}`, activez d'abord l'accès au modèle `amazon.nova-micro-v1:0` (étape manuelle et unique).

### 6.4 Policy IAM élargie — référence + Meta Llama + modèles custom

Si vous utilisez un **deuxième agent Bedrock** (Llama fondation ou **Llama fine-tuné** / importé), l’action `bedrock:InvokeModel` doit cibler les **ARN** réels visibles dans la console Bedrock (fondation, profil d’inférence, modèle personnalisé). Exemple **à resserrer en production** :

```json
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
      "arn:aws:bedrock:${AWS_REGION}:${AWS_ACCOUNT_ID}:inference-profile/*",
      "arn:aws:bedrock:${AWS_REGION}:${AWS_ACCOUNT_ID}:provisioned-model/*",
      "arn:aws:bedrock:${AWS_REGION}:${AWS_ACCOUNT_ID}:custom-model/*"
    ]
  }]
}
```

Remplacez les placeholders par votre compte et région. Les préfixes exacts (`inference-profile`, `custom-model`, etc.) dépendent du type de déploiement Bedrock — **copiez les ARN** depuis la console après le fine-tuning ou l’import.

---

## 6A. Fine-tuner un Meta Llama sur Amazon Bedrock (texte → SQL)

Objectif : produire un **modèle personnalisé** invoqué comme le modèle de base, mais spécialisé sur vos paires (question + schéma `products` → SQL DuckDB).

Le répertoire `finetuning/` automatise l'intégralité du pipeline.

### A.1 Données d'entraînement

```bash
python3 -m finetuning.generate_dataset
# → finetuning/data/train.jsonl  (~120 exemples)
# → finetuning/data/eval.jsonl   (~20 exemples)
```

Le script génère des paires supervisées au format **Converse** (messages `system` / `user` / `assistant`). Le prompt système inclut le schéma complet de la table `products`, les règles de génération SQL et le rôle d'assistant. Le `user` contient la question en langage naturel, l'`assistant` la requête SQL seule.

Références AWS : [Prepare data for fine-tuning](https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-prepare.html), [Customize a model with fine-tuning](https://docs.aws.amazon.com/bedrock/latest/userguide/custom-model-fine-tuning.html).

### A.2 Lancer la personnalisation

```bash
python3 -m finetuning.launch_finetune \
  --bucket ${DATA_BUCKET} \
  --role-arn arn:aws:iam::${AWS_ACCOUNT_ID}:role/BedrockFineTuneRole \
  --region us-east-1
```

Le script :
1. Upload `train.jsonl` et `eval.jsonl` vers `s3://${DATA_BUCKET}/bedrock-finetune/`.
2. Appelle `CreateModelCustomizationJob` avec le modèle de base **Meta Llama 3.1 8B Instruct** (`meta.llama3-1-8b-instruct-v1:0`).
3. Affiche le `jobArn` — suivez la progression via la console **Bedrock → Custom models** ou `GetModelCustomizationJob`.

> **Région** : Le fine-tuning Llama est souvent disponible uniquement dans `us-east-1` ou `us-west-2` — vérifiez la console Bedrock.

### A.3 Évaluer le modèle fine-tuné en le comparant avec un modèle de référence

```bash
python3 -m finetuning.evaluate \
  --finetuned-model-id <custom-model-arn-ou-inference-profile> \
  --reference-model-id amazon.nova-micro-v1:0 \
  --region us-east-1
```

Le script :
- Exécute chaque question de `eval.jsonl` sur les deux modèles ;
- Valide le SQL généré via DuckDB (exécution réelle) ;
- Compare : **taux de réussite SQL**, **exact match**, **latence moyenne** ;
- Affiche un rapport console structuré.


### A.4 Brancher Kōjin sur le Llama fine-tuné

Dans le conteneur ECS (ou en local) :

- **`BEDROCK_MODEL_ID`** — gardez un modèle **référence** peu coûteux pour la prod simple (ex. `amazon.nova-micro-v1:0`) ou un Llama instruct non fine-tuné si vous ne comparez pas.
- **`BEDROCK_COMPARE_MODEL_ID`** — collez l’ID / ARN / identifiant d’inférence **retourné par Bedrock** pour votre **Llama fine-tuné** (même région `AWS_REGION` que l’accès IAM).

Dans l’UI **Exploration des ingrédients**, cochez **Comparer les deux agents** : colonne gauche = référence, colonne droite = Llama custom. **Priorité** : si `BEDROCK_COMPARE_MODEL_ID` est défini, le second agent est Bedrock ; sinon le second agent utilise `OPENAI_COMPAT_*` (Ollama, vLLM, etc.).

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
| `LLM_PROVIDER` | non | `auto` \| `groq` \| `bedrock`. Défaut `auto` : Groq si `GROQ_API_KEY`, sinon Bedrock. En pile **ECS pure AWS**, fixer **`bedrock`** si vous ne voulez jamais tenter Groq. |
| `GROQ_API_KEY` | si Groq | Clé depuis **Groq Console** ; en ECS préférez **Secrets Manager** injecté comme env (ne pas committer dans l’image). |
| `GROQ_MODEL_ID` | non | Modèle Groq (`llama-3.1-8b-instant`, etc.). Requiert egress HTTPS vers `api.groq.com`. |
| `BEDROCK_MODEL_ID` | oui si réf. Bedrock | Modèle **référence** Exploration (ex. Nova Micro). Ignoré si la référence est Groq. |
| `BEDROCK_COMPARE_MODEL_ID` | non | Second modèle Bedrock (ex. Llama fine-tuné) pour la comparaison. |
| `OPENAI_COMPAT_BASE_URL` / `OPENAI_COMPAT_MODEL` | non | Alternative au second Bedrock (serveur OpenAI-compatible). |
| `KOJIN_EXPLORATION_LOG_JSONL` | non | `1` pour activer le fichier NDJSON. |
| `KOJIN_EXPLORATION_LOG_PATH` | non | Chemin du fichier de métriques. |

## 7. Cluster ECS, Task Definition, Service

### 7.1 Cluster + log group

```bash
aws ecs create-cluster \
  --cluster-name ${CLUSTER} \
  --capacity-providers FARGATE \
  --region ${AWS_REGION}

aws logs create-log-group \
  --log-group-name /ecs/${APP_NAME} \
  --region ${AWS_REGION}
```

### 7.2 Task definition (`task-def.json`)

```json
{
  "family": "kojin-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::ACCOUNT_ID:role/kojin-ecs-execution-role",
  "taskRoleArn": "arn:aws:iam::ACCOUNT_ID:role/kojin-ecs-task-role",
  "containerDefinitions": [
    {
      "name": "kojin-web",
      "image": "ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/kojin-streamlit:latest",
      "essential": true,
      "portMappings": [{ "containerPort": 8501, "protocol": "tcp" }],
      "environment": [
        { "name": "STREAMLIT_SERVER_HEADLESS", "value": "true" },
        { "name": "AWS_REGION", "value": "REGION" },
        { "name": "DATA_S3_URI", "value": "s3://DATA_BUCKET/DATA_KEY" },
        { "name": "LLM_PROVIDER", "value": "bedrock" },
        { "name": "BEDROCK_MODEL_ID", "value": "amazon.nova-micro-v1:0" },
        { "name": "BEDROCK_COMPARE_MODEL_ID", "value": "" },
        { "name": "KOJIN_EXPLORATION_LOG_JSONL", "value": "0" },
        { "name": "KOJIN_EXPLORATION_LOG_PATH", "value": "/tmp/kojin_exploration_metrics.ndjson" }
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
          "awslogs-group": "/ecs/kojin",
          "awslogs-region": "REGION",
          "awslogs-stream-prefix": "web"
        }
      }
    }
  ]
}
```

Substituer les placeholders puis enregistrer :

```bash
sed -i \
  -e "s/ACCOUNT_ID/${AWS_ACCOUNT_ID}/g" \
  -e "s/REGION/${AWS_REGION}/g" \
  -e "s/DATA_BUCKET/${DATA_BUCKET}/g" \
  -e "s/DATA_KEY/${DATA_KEY}/g" \
  task-def.json

aws ecs register-task-definition --cli-input-json file://task-def.json
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
- **Coûts LLM** : filtrer **Bedrock** dans Cost Explorer par modèle / région pour comparer **référence** vs **Llama fine-tuné** (`BEDROCK_COMPARE_MODEL_ID`).
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
- [ ] Bedrock : **Model access** pour `BEDROCK_MODEL_ID` et pour `BEDROCK_COMPARE_MODEL_ID` si Llama custom.
- [ ] Task definition : `DATA_S3_URI`, `BEDROCK_MODEL_ID`, fac. `BEDROCK_COMPARE_MODEL_ID`, fac. logs `KOJIN_EXPLORATION_*`.
- [ ] ALB avec target group `/_stcore/health`, stickiness activée.
- [ ] Security groups restreints (8501 uniquement depuis l'ALB).
- [ ] Listener HTTPS avec certificat ACM, redirection 80 → 443.
- [ ] Alarmes CloudWatch configurées.
- [ ] Auto-scaling si > 10 utilisateurs simultanés attendus.
