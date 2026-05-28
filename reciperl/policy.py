"""Residual PPO actor & critic — §3.4 and Fig. 4 of Liu et al. (2024).

Replaces the original PPO-clip linear head by a small 1-D ConvNet with a
residual block, followed by an FC head. Shared backbone, two output heads:
    * actor  : logits over the catalogue → π(aₜ | sₜ)
    * critic : scalar value V(sₜ)
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


class _ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return self.act(x + h)


class ResidualBackbone(nn.Module):
    """Conv → ResBlock → Conv → FC (matches Fig. 4)."""

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.in_dim = in_dim
        self.proj = nn.Conv1d(1, hidden, kernel_size=3, padding=1)
        self.res = _ResBlock(hidden)
        self.conv_out = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden * in_dim, hidden),
            nn.ReLU(),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = s.unsqueeze(1)                  # (B, 1, D)
        x = torch.relu(self.proj(x))
        x = self.res(x)
        x = torch.relu(self.conv_out(x))
        return self.fc(x)                    # (B, hidden)


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.backbone = ResidualBackbone(state_dim, hidden)
        self.actor_head = nn.Linear(hidden, n_actions)
        self.critic_head = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        h = self.backbone(state)
        logits = self.actor_head(h)
        value = self.critic_head(h).squeeze(-1)
        return Categorical(logits=logits), value

    def act(self, state: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action and return (action, log_prob, value)."""
        h = self.backbone(state)
        logits = self.actor_head(h)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), self.critic_head(h).squeeze(-1)
