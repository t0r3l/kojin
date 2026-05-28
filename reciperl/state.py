"""Fused state representation — §3.3 of Liu et al. (2024).

Three sub-states are concatenated (Eq. 15):
    sₜ = [ s_UI | s_ACH | s_UC ]

* s_UI  : DDR-u user×food interaction (Eq. 7).
* s_ACH : sliding-window historical memory weighted by a probabilistic mask
          P_Mask = (F'_H + n_u) · λ_r  (Eqs. 8-13), fused with cross-attention.
* s_UC  : user dynamic preference over the food category of the last
          recommendation (Eq. 14).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .config import RecipeRLConfig


class HistoricalMemory(nn.Module):
    """Eqs. (8)-(13) — sliding window + probability mask + Cross Attention."""

    def __init__(self, embed_dim: int, cfg: RecipeRLConfig):
        super().__init__()
        self.k = cfg.window_k
        self.noise_std = cfg.gaussian_noise_std
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=cfg.attn_heads,
            batch_first=True,
        )

    def forward(
        self,
        item_embeds: torch.Tensor,   # (B, k, d) — embeddings of last-k items
        ratings: torch.Tensor,        # (B, k)   — feedback fᵢ in [0, 5]
        rec_counts: torch.Tensor,     # (B, k)   — Cᵢ for the forgetting score
    ) -> torch.Tensor:
        # s_W = windowed embeddings (Eq. 8).
        s_W = item_embeds

        # F'_H : per-step probability weighting (Eq. 11).
        f_norm = ratings.clamp(0.0, 5.0) / 5.0
        # Gaussian noise n_u (added only during training).
        if self.training and self.noise_std > 0.0:
            f_norm = f_norm + torch.randn_like(f_norm) * self.noise_std
        # Forgetting score λ_r (Eq. 9).
        lam_r = (1.0 - rec_counts.clamp_max(self.k) / self.k).clamp_min(0.0)
        # Probability mask (Eq. 10).
        p_mask = (f_norm * lam_r).unsqueeze(-1)                # (B, k, 1)

        # User-preferred state (Eq. 12).
        s_CH = s_W * p_mask

        # Cross Attention (Eq. 13): queries = s_W, keys/values = s_CH.
        s_ACH, _ = self.cross_attn(s_W, s_CH, s_CH, need_weights=False)
        # Aggregate the sequence to a single vector.
        return s_ACH.mean(dim=1)


class FusedState(nn.Module):
    """Concatenates the three sub-states (Eq. 15) into the agent's input."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_categories: int,
        cfg: RecipeRLConfig,
    ):
        super().__init__()
        d = cfg.cf_embed_dim
        self.user_emb = nn.Embedding(n_users, d)
        self.item_emb = nn.Embedding(n_items, d)
        self.cat_emb = nn.Embedding(n_categories, d)
        self.item_w = nn.Embedding(n_items, 1)         # scalar wᵢ in Eq. 7
        for emb in (self.user_emb, self.item_emb, self.cat_emb):
            nn.init.normal_(emb.weight, std=0.01)
        nn.init.constant_(self.item_w.weight, 1.0)

        self.history = HistoricalMemory(d, cfg)
        self.out_dim = 3 * d

    def warmstart_from_ncf(self, ncf_user: nn.Embedding, ncf_item: nn.Embedding) -> None:
        """Initialise user/item embeddings from the trained NCF (P, Q in Eq. 7)."""
        with torch.no_grad():
            self.user_emb.weight.copy_(ncf_user.weight)
            self.item_emb.weight.copy_(ncf_item.weight)

    # ── Sub-state builders ─────────────────────────────────────────────
    def _ddr_u(
        self,
        user: torch.Tensor,
        last_item: torch.Tensor,
        user_vec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """s_UI = pᵤ ⊗ wᵢ qᵢ (Eq. 7).

        ``user_vec`` overrides the embedding lookup — used for anonymous users
        (e.g. the centroid of all learned embeddings).
        """
        p_u = user_vec if user_vec is not None else self.user_emb(user)
        return p_u * self.item_w(last_item) * self.item_emb(last_item)

    def _user_category(
        self,
        user: torch.Tensor,
        last_item: torch.Tensor,
        cat: torch.Tensor,
        user_vec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """s_UC = pᵤ ⊗ wᵢ c_{gᵢ} (Eq. 14)."""
        p_u = user_vec if user_vec is not None else self.user_emb(user)
        return p_u * self.item_w(last_item) * self.cat_emb(cat)

    # ── Forward ────────────────────────────────────────────────────────
    def forward(
        self,
        user: torch.Tensor,                    # (B,)
        history_items: torch.Tensor,           # (B, k)
        history_ratings: torch.Tensor,
        rec_counts: torch.Tensor,
        last_item: torch.Tensor,               # (B,)
        last_cat: torch.Tensor,                # (B,)
        user_vec: torch.Tensor | None = None,  # (B, d) override — anonymous user
    ) -> torch.Tensor:
        s_UI = self._ddr_u(user, last_item, user_vec)
        s_ACH = self.history(self.item_emb(history_items), history_ratings, rec_counts)
        s_UC = self._user_category(user, last_item, cat=last_cat, user_vec=user_vec)
        return torch.cat([s_UI, s_ACH, s_UC], dim=-1)
