"""
3D-CNN actor-critic policy for viewpoint selection PPO.

Architecture
------------
  Encoder  : Conv3d (1→16→32→64) with BatchNorm + ReLU, max-pool ×3
             → flatten → concat(cam_pos, cam_dir, explored)
             → Linear → 128-dim feature
  Actor    : Linear(128→64→4) outputting μ (mean) of tanh-Gaussian
             + a learned log_std parameter
  Critic   : Linear(128→64→1) scalar state value

The observation dict is:
  "occupancy" : (B, 1, R, R, R)  float32 in {0, 1, 2}
  "cam_pos"   : (B, 3)
  "cam_dir"   : (B, 3)
  "explored"  : (B, 1)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class OccupancyEncoder(nn.Module):
    """3D-CNN that compresses a (1, R, R, R) occupancy grid to a vector."""

    def __init__(self, grid_res: int = 16):
        super().__init__()
        # Input: (B, 1, R, R, R) — we normalise 0-2 → 0-1 inside forward
        self.conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),                       # R/2

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),                       # R/4

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),               # → (B, 64, 1, 1, 1)
        )
        self.out_dim = 64

    def forward(self, occ: torch.Tensor) -> torch.Tensor:
        """occ: (B, 1, R, R, R) float32 with values in {0, 1, 2}."""
        x = occ / 2.0  # normalise to [0, 1]
        x = self.conv(x)
        return x.view(x.size(0), -1)  # (B, 64)


class ViewpointActorCritic(nn.Module):
    """Combined actor-critic with a shared 3D-CNN encoder trunk.

    Actor  : outputs μ ∈ ℝ⁴ (tanh-squashed Gaussian)
    Critic : outputs V(s) ∈ ℝ
    """

    def __init__(
        self,
        grid_res: int = 16,
        hidden_dim: int = 128,
        action_dim: int = 4,
        init_log_std: float = -0.5,
    ):
        super().__init__()
        self.encoder = OccupancyEncoder(grid_res)
        # Extra features: cam_pos(3) + cam_dir(3) + explored(1) = 7
        extra_dim = 7
        feat_dim = self.encoder.out_dim + extra_dim  # 71

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Actor head — outputs mean action (4-D)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_dim),   # raw mean, will be tanh-squashed
        )
        # Learnable log standard deviation (shared across states)
        self.log_std = nn.Parameter(
            torch.full((action_dim,), init_log_std))

        # Critic head — scalar value
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        self._action_dim = action_dim
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv3d)):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Actor final layer — small init for exploration
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        # Critic final layer — normal init
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def _encode(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Produce the shared feature vector from an observation dict."""
        occ_feat = self.encoder(obs["occupancy"])      # (B, 64)
        extra = torch.cat([
            obs["cam_pos"],   # (B, 3)
            obs["cam_dir"],   # (B, 3)
            obs["explored"],  # (B, 1)
        ], dim=-1)                                      # (B, 7)
        x = torch.cat([occ_feat, extra], dim=-1)       # (B, 71)
        return self.trunk(x)                            # (B, hidden)

    def forward(
        self, obs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (action_mean, state_value)."""
        h = self._encode(obs)
        mu = self.actor(h)           # (B, 4)
        value = self.critic(h)       # (B, 1)
        return mu, value.squeeze(-1)

    def get_distribution(
        self, obs: dict[str, torch.Tensor],
    ) -> tuple[Normal, torch.Tensor]:
        """Return (action_distribution, state_value)."""
        mu, value = self.forward(obs)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value

    def act(
        self,
        obs: dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action (or take the mean) and return (action, log_prob, value).

        The action is **tanh-squashed** to [-1, 1].
        """
        dist, value = self.get_distribution(obs)
        if deterministic:
            raw = dist.mean
        else:
            raw = dist.rsample()

        action = torch.tanh(raw)

        # Log-prob with tanh correction:
        #   log π(a|s) = log N(u; μ, σ) − Σ log(1 − tanh²(u))
        log_prob = dist.log_prob(raw) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)  # (B,)

        return action, log_prob, value

    def evaluate_actions(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate stored actions for PPO update.

        Parameters
        ----------
        obs : dict    — batched observations
        actions : (B, 4) — tanh-squashed actions from rollout

        Returns
        -------
        log_probs : (B,)
        values    : (B,)
        entropy   : (B,)
        """
        dist, value = self.get_distribution(obs)

        # Invert tanh to recover raw (pre-squash) actions
        raw = torch.atanh(actions.clamp(-0.999, 0.999))

        log_prob = dist.log_prob(raw) - torch.log(1 - actions.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)

        return log_prob, value, entropy
