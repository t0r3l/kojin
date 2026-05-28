# RecipeRL — implementation

Self-contained PyTorch implementation of *"An interactive food recommendation
system using reinforcement learning"* (Liu et al., **Expert Systems with
Applications 254**, 2024) adapted to the Kōjin food catalogue.

## Architecture overview

```
                       ┌──────────────────────────┐
                       │   Environment simulator  │   §3.2
   user, food ─────►   │   (Neural CF, Eqs. 3-5)  │ ─► predicted rating ŷᵤᵢ
                       └──────────────────────────┘
                                    │
                                    ▼
                       ┌──────────────────────────┐
   interaction (aₜ,fₜ) │      Reward function     │   §3.2, Eq. 6
   ───────────────────►│   r = fᵢⱼ + α (c_p − c_n) │
                       └──────────────────────────┘
                                    │
                                    ▼
        ┌─────────────────────────────────────────────────────────┐
        │  Fused state representation (Eq. 15)                    │   §3.3
        │    s_UI  : DDR-u user×food         (Eq. 7)              │
        │    s_ACH : sliding window + P_Mask + Cross-Attention    │
        │            (Eqs. 8-13, forgetting λ_r)                  │
        │    s_UC  : user × food-category    (Eq. 14)             │
        └─────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                       ┌──────────────────────────┐
                       │   Residual PPO policy    │   §3.4, Fig. 4
                       │   (actor + critic)       │
                       └──────────────────────────┘
                                    │
                                    ▼
                            recommended food aₜ
```

## Files

| File | Purpose |
|---|---|
| `config.py` | Default hyperparameters (§4.4 of the paper). |
| `data.py` | Load Kōjin catalogue, build user/category vocabularies, synthesise ratings when no review dataset is available. |
| `ncf.py` | Neural Collaborative Filtering simulator (Eqs. 3-5). |
| `state.py` | Fused state representation, including Cross Attention (§3.3). |
| `policy.py` | Residual actor & critic networks (Fig. 4). |
| `env.py` | Gym-like multi-step environment, sequential reward (Eq. 6). |
| `ppo.py` | PPO-clip update with GAE (Eqs. 16-17). |
| `evaluate.py` | Precision@T, NDCG@T, Reward@T (Eqs. 18-21). |
| `train.py` | CLI entrypoint: pre-train CF, train PPO, evaluate. |

## Quickstart

```bash
pip install torch  # not in core requirements.txt — only needed for RecipeRL
python -m reciperl.train --steps 2000 --top-k 10
```

The default run uses a small synthetic-rating subset of the Kōjin catalogue
so the pipeline can be exercised end-to-end without external data. To plug
in real ratings (e.g. the Food.com / Kaggle dataset used in the paper),
override `data.load_ratings()` with your own loader returning a
`(user_id, item_id, rating)` table.

## Mapping to the paper

| Paper symbol      | Code location |
|-------------------|---------------|
| Eq. 3 ŷᵤᵢ          | `ncf.NCF.forward` |
| Eq. 5 Lₛqᵣ          | `ncf.train_ncf` |
| Eq. 6 r(sₜ,aₜ)     | `env.RecipeEnv._reward` |
| Eq. 7 s_UI         | `state.FusedState._ddr_u` |
| Eqs. 8-13 s_ACH    | `state.HistoricalMemory.forward` |
| Eq. 14 s_UC        | `state.FusedState._user_category` |
| Eq. 15 sₜ          | `state.FusedState.forward` |
| Eqs. 16-17 PPO     | `ppo.PPOTrainer.update` |
| Eqs. 18-21 metrics | `evaluate.eval_topk` |
| Fig. 4 ResNet head | `policy.ResidualBackbone` |
```
