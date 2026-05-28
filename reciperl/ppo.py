"""PPO-clip trainer — §3.4, Eqs. (16)-(17) of Liu et al. (2024)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch
from torch import nn

from .config import RecipeRLConfig
from .env import RecipeEnv
from .policy import ActorCritic


@dataclass
class Rollout:
    states: list[torch.Tensor] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[torch.Tensor] = field(default_factory=list)
    values: list[torch.Tensor] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)


class PPOTrainer:
    def __init__(
        self,
        env: RecipeEnv,
        state_module: nn.Module,
        policy: ActorCritic,
        cfg: RecipeRLConfig,
        device: torch.device,
    ):
        self.env = env
        self.state_module = state_module
        self.policy = policy
        self.cfg = cfg
        self.device = device
        # Share the optimiser between actor & critic and the state module
        # so embeddings keep adapting during RL (§3.3).
        self.opt = torch.optim.Adam(
            list(policy.parameters()) + list(state_module.parameters()),
            lr=cfg.actor_lr,
        )

    # ── Rollout collection ─────────────────────────────────────────────
    def collect(self, rng: np.random.Generator) -> Rollout:
        roll = Rollout()
        for _ in range(self.cfg.rollout_episodes):
            state = self.env.reset(rng=rng)
            done = False
            while not done:
                with torch.no_grad():
                    action, log_prob, value = self.policy.act(state.unsqueeze(0))
                a = int(action.item())
                next_state, reward, done, _ = self.env.step(a)
                roll.states.append(state)
                roll.actions.append(a)
                roll.log_probs.append(log_prob.squeeze(0))
                roll.values.append(value.squeeze(0))
                roll.rewards.append(reward)
                roll.dones.append(done)
                state = next_state
        return roll

    # ── GAE advantage ──────────────────────────────────────────────────
    def _gae(self, rewards: list[float], values: list[torch.Tensor], dones: list[bool]) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros(len(rewards), device=self.device)
        last_adv = 0.0
        values_arr = torch.stack(values + [torch.zeros(1, device=self.device).squeeze()])
        for t in reversed(range(len(rewards))):
            non_terminal = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.cfg.gamma * values_arr[t + 1] * non_terminal - values_arr[t]
            last_adv = delta + self.cfg.gamma * self.cfg.gae_lambda * non_terminal * last_adv
            advantages[t] = last_adv
        returns = advantages + torch.stack(values)
        return advantages.detach(), returns.detach()

    # ── PPO update (Eqs. 16-17) ────────────────────────────────────────
    def update(self, roll: Rollout) -> dict[str, float]:
        states = torch.stack(roll.states)
        actions = torch.as_tensor(roll.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.stack(roll.log_probs).detach()
        advantages, returns = self._gae(roll.rewards, roll.values, roll.dones)
        # Advantage normalisation — common stabilisation trick.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        idx = np.arange(states.size(0))
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0, "n": 0}
        for _ in range(self.cfg.ppo_epochs):
            np.random.shuffle(idx)
            for start in range(0, len(idx), self.cfg.ppo_batch_size):
                b = idx[start : start + self.cfg.ppo_batch_size]
                b_states = states[b]
                b_actions = actions[b]
                b_old_lp = old_log_probs[b]
                b_adv = advantages[b]
                b_ret = returns[b]

                dist, value = self.policy(b_states)
                log_prob = dist.log_prob(b_actions)
                ratio = torch.exp(log_prob - b_old_lp)  # ρₜ(θ) in Eq. 17

                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (value - b_ret).pow(2).mean()
                entropy = dist.entropy().mean()

                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.opt.step()

                stats["policy_loss"] += policy_loss.item() * len(b)
                stats["value_loss"] += value_loss.item() * len(b)
                stats["entropy"] += entropy.item() * len(b)
                stats["kl"] += (b_old_lp - log_prob).mean().item() * len(b)
                stats["n"] += len(b)

        n = max(stats.pop("n"), 1)
        return {k: v / n for k, v in stats.items()}
