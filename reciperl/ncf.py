"""Neural Collaborative Filtering simulator — §3.2, Eqs. (3)-(5).

The simulator predicts a rating ŷᵤᵢ for any (user, item) pair. During RL
training it acts as the "user" of the interactive environment, providing
feedback for arbitrary recommendations even for (u, i) pairs that are
absent from the observed interaction matrix (data augmentation).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import RecipeRLConfig
from .data import RecipeRLDataset


class NCF(nn.Module):
    """Neural Collaborative Filtering (He et al., 2017).

    Implements Eqs. (3)-(4):
        ŷᵤᵢ = g(Pᵀvᵤ, Qᵀvᵢ | Θ_g) + bᵢ
    where g is an MLP and bᵢ a per-item bias.
    """

    def __init__(self, n_users: int, n_items: int, cfg: RecipeRLConfig):
        super().__init__()
        d = cfg.cf_embed_dim
        self.user_emb = nn.Embedding(n_users, d)
        self.item_emb = nn.Embedding(n_items, d)
        self.item_bias = nn.Embedding(n_items, 1)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.item_bias.weight)

        layers: list[nn.Module] = []
        in_dim = 2 * d
        for h in cfg.cf_hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))  # φ_out
        self.mlp = nn.Sequential(*layers)

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.user_emb(u), self.item_emb(i)], dim=-1)
        return self.mlp(z).squeeze(-1) + self.item_bias(i).squeeze(-1)


def train_ncf(
    dataset: RecipeRLDataset,
    cfg: RecipeRLConfig,
    device: torch.device,
) -> NCF:
    """Train NCF by weighted MSE between ŷᵤᵢ and observed ratings (Eq. 5)."""
    model = NCF(dataset.n_users, dataset.n_items, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.cf_lr)

    train = dataset.ratings[dataset.train_mask]
    ds = TensorDataset(
        torch.as_tensor(train[:, 0], dtype=torch.long),
        torch.as_tensor(train[:, 1], dtype=torch.long),
        torch.as_tensor(train[:, 2], dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=cfg.cf_batch_size, shuffle=True)

    for epoch in range(cfg.cf_epochs):
        total = 0.0
        n = 0
        for u, i, r in loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            pred = model(u, i)
            loss = ((pred - r) ** 2).mean()  # uniform w_ui = 1, paper Eq. 5
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * u.size(0)
            n += u.size(0)
        print(f"[NCF] epoch {epoch + 1}/{cfg.cf_epochs}  mse={total / max(n, 1):.4f}")
    return model


@torch.no_grad()
def precompute_rating_matrix(model: NCF, n_users: int, n_items: int, device: torch.device) -> np.ndarray:
    """Materialise ŷ for every (u, i) — the augmented interaction matrix."""
    model.eval()
    users = torch.arange(n_users, device=device)
    items = torch.arange(n_items, device=device)
    out = torch.empty(n_users, n_items, device=device)
    for u in users:
        out[u] = model(u.expand_as(items), items)
    return out.clamp_(0.0, 5.0).cpu().numpy()
