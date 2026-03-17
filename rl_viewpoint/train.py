#!/usr/bin/env python3
"""
PPO training loop for the viewpoint selection policy.

Supports two modes:
  - ``cached`` — fast offline training from pre-recorded transitions
  - ``sim``    — fully simulated scenes with random primitives (no arm)

Usage
-----
    # Cached mode (requires pre-recorded data)
    python -m rl_viewpoint.train --mode cached \
        --data transitions.npz --epochs 200

    # Simulated mode (no data needed)
    python -m rl_viewpoint.train --mode sim \
        --epochs 500 --steps-per-epoch 1024

The training loop follows a standard PPO recipe:
  1. Collect a rollout of ``steps_per_epoch`` transitions.
  2. Compute GAE advantages + discounted returns.
  3. Run ``ppo_epochs`` mini-batch updates with clipped surrogate loss.
  4. Log metrics, save checkpoints periodically.
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .env import CachedViewpointEnv, SimulatedViewpointEnv
from .policy import ViewpointActorCritic

logger = logging.getLogger(__name__)


# ======================================================================
# Rollout buffer
# ======================================================================

class RolloutBuffer:
    """Fixed-size buffer that stores one epoch of transitions."""

    def __init__(self, capacity: int, obs_keys: list[str], action_dim: int):
        self.capacity = capacity
        self.obs_keys = obs_keys
        self.action_dim = action_dim
        self.reset()

    def reset(self) -> None:
        self.obs: dict[str, list[np.ndarray]] = {k: [] for k in self.obs_keys}
        self.actions: list[np.ndarray] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.ptr = 0

    def add(
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        for k in self.obs_keys:
            self.obs[k].append(obs[k])
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.ptr += 1

    def to_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        """Convert all stored data to tensors on *device*."""
        out: dict[str, torch.Tensor] = {}
        for k in self.obs_keys:
            out[f"obs_{k}"] = torch.tensor(
                np.array(self.obs[k]), dtype=torch.float32, device=device)
        out["actions"] = torch.tensor(
            np.array(self.actions), dtype=torch.float32, device=device)
        out["log_probs"] = torch.tensor(
            self.log_probs, dtype=torch.float32, device=device)
        out["values"] = torch.tensor(
            self.values, dtype=torch.float32, device=device)
        out["rewards"] = torch.tensor(
            self.rewards, dtype=torch.float32, device=device)
        out["dones"] = torch.tensor(
            self.dones, dtype=torch.float32, device=device)
        return out

    def __len__(self) -> int:
        return self.ptr


# ======================================================================
# GAE computation
# ======================================================================

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
    last_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalised Advantage Estimation.

    Returns (advantages, returns) both shape (T,).
    """
    T = len(rewards)
    advantages = torch.zeros(T, device=rewards.device)
    last_gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_value
        else:
            next_value = values[t + 1].item()
        next_non_done = 1.0 - dones[t].item()
        delta = rewards[t] + gamma * next_value * next_non_done - values[t]
        advantages[t] = last_gae = delta + gamma * lam * next_non_done * last_gae
    returns = advantages + values
    return advantages, returns


# ======================================================================
# PPO update
# ======================================================================

def ppo_update(
    model: ViewpointActorCritic,
    optimizer: optim.Optimizer,
    data: dict[str, torch.Tensor],
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    n_epochs: int = 4,
    batch_size: int = 64,
) -> dict[str, float]:
    """Run multiple PPO mini-batch gradient steps on collected data."""
    T = len(advantages)
    obs_keys = [k.replace("obs_", "") for k in data if k.startswith("obs_")]

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    n_updates = 0

    for _ in range(n_epochs):
        indices = np.random.permutation(T)
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            mb_idx = indices[start:end]
            mb_idx_t = torch.tensor(mb_idx, dtype=torch.long,
                                     device=advantages.device)

            # Build observation dict for mini-batch
            mb_obs = {}
            for k in obs_keys:
                mb_obs[k] = data[f"obs_{k}"][mb_idx]

            mb_actions = data["actions"][mb_idx]
            mb_old_log_probs = data["log_probs"][mb_idx]
            mb_advantages = advantages[mb_idx]
            mb_returns = returns[mb_idx]

            # Normalise advantages per mini-batch
            if len(mb_advantages) > 1:
                mb_advantages = (
                    (mb_advantages - mb_advantages.mean())
                    / (mb_advantages.std() + 1e-8)
                )

            # Forward
            new_log_probs, new_values, entropy = model.evaluate_actions(
                mb_obs, mb_actions)

            # Policy loss (clipped surrogate)
            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio,
                                1.0 + clip_ratio) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (clipped)
            value_loss = F.mse_loss(new_values, mb_returns)

            # Entropy bonus
            entropy_loss = -entropy.mean()

            loss = (policy_loss
                    + value_coef * value_loss
                    + entropy_coef * entropy_loss)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            n_updates += 1

    return {
        "policy_loss": total_policy_loss / max(n_updates, 1),
        "value_loss": total_value_loss / max(n_updates, 1),
        "entropy": total_entropy / max(n_updates, 1),
        "n_updates": n_updates,
    }


# ======================================================================
# Main training function
# ======================================================================

def train(args: argparse.Namespace) -> None:
    """Full PPO training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Environment ────────────────────────────────────────────────
    if args.mode == "sim":
        # Build a simple uniform candidate bank for simulation
        from .simulator import simulated_fk_bank
        n_az, n_el = 12, 5
        candidates = []
        for i_az in range(n_az):
            for i_el in range(n_el):
                # Dummy joint angles — simulated_fk_bank ignores them
                candidates.append([i_az * 30.0, i_el * 18.0, 0, 0, 0, 0])
        env = SimulatedViewpointEnv(
            candidates=candidates,
            obs_res=args.obs_res,
            grid_res=args.grid_res,
            max_steps=args.max_ep_steps,
            half_extent=args.half_extent,
            n_objects=(args.min_objects, args.max_objects),
            object_size_range=(args.min_obj_size, args.max_obj_size),
        )
        print(f"Simulated env: {args.grid_res}³ grid, "
              f"{args.min_objects}-{args.max_objects} objects, "
              f"bank size {len(candidates)}")
    else:
        assert args.data is not None, "--data required for cached mode"
        env = CachedViewpointEnv(
            data_path=args.data,
            obs_res=args.obs_res,
            max_steps=args.max_ep_steps,
        )

    # ── Model + Optimiser ──────────────────────────────────────────
    model = ViewpointActorCritic(
        grid_res=args.obs_res,
        hidden_dim=args.hidden_dim,
        action_dim=4,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Rollout buffer ─────────────────────────────────────────────
    obs_keys = list(env.observation_space.spaces.keys())
    buffer = RolloutBuffer(
        capacity=args.steps_per_epoch,
        obs_keys=obs_keys,
        action_dim=4,
    )

    # ── Logging ────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    episode_rewards: deque[float] = deque(maxlen=100)
    best_avg_reward = -float("inf")
    log_file = os.path.join(args.save_dir, "training_log.txt")

    with open(os.path.join(args.save_dir, "config.pkl"), "wb") as f:
        pickle.dump(vars(args), f)

    print(f"\n{'='*60}")
    print("  PPO TRAINING — Viewpoint Selection")
    print(f"{'='*60}")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print(f"{'='*60}\n")

    # ── Training loop ──────────────────────────────────────────────
    global_step = 0
    obs, _ = env.reset()
    ep_reward = 0.0
    ep_len = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        buffer.reset()
        model.eval()

        # ── Collect rollout ────────────────────────────────────────
        for _ in range(args.steps_per_epoch):
            with torch.no_grad():
                obs_t = {
                    k: torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
                    for k, v in obs.items()
                }
                action, log_prob, value = model.act(obs_t)
                action_np = action.cpu().numpy().squeeze(0)
                log_prob_np = log_prob.cpu().item()
                value_np = value.cpu().item()

            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            buffer.add(obs, action_np, log_prob_np, value_np, reward, done)
            global_step += 1
            ep_reward += reward
            ep_len += 1

            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                ep_len = 0
                obs, _ = env.reset()
            else:
                obs = next_obs

        # ── Compute last value for GAE ─────────────────────────────
        with torch.no_grad():
            obs_t = {
                k: torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
                for k, v in obs.items()
            }
            _, _, last_val = model.act(obs_t, deterministic=True)
            last_value = last_val.cpu().item()

        # ── GAE + PPO update ───────────────────────────────────────
        data = buffer.to_tensors(device)
        advantages, returns = compute_gae(
            data["rewards"], data["values"], data["dones"],
            gamma=args.gamma, lam=args.gae_lambda,
            last_value=last_value,
        )

        model.train()
        loss_info = ppo_update(
            model, optimizer, data, advantages, returns,
            clip_ratio=args.clip_ratio,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            n_epochs=args.ppo_epochs,
            batch_size=args.batch_size,
        )

        elapsed = time.time() - t0
        avg_rwd = np.mean(episode_rewards) if episode_rewards else 0.0

        # ── Logging ────────────────────────────────────────────────
        if epoch % args.log_every == 0 or epoch == 1:
            msg = (
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"steps={global_step:6d} | "
                f"avg_reward={avg_rwd:.5f} | "
                f"pi_loss={loss_info['policy_loss']:.4f} | "
                f"v_loss={loss_info['value_loss']:.4f} | "
                f"entropy={loss_info['entropy']:.3f} | "
                f"{elapsed:.1f}s"
            )
            print(msg)
            with open(log_file, "a") as f:
                f.write(msg + "\n")

        # ── Checkpointing ─────────────────────────────────────────
        if avg_rwd > best_avg_reward and len(episode_rewards) >= 10:
            best_avg_reward = avg_rwd
            path = os.path.join(args.save_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_reward": avg_rwd,
            }, path)
            print(f"  ★ New best avg reward: {avg_rwd:.5f} → {path}")

        if epoch % args.save_every == 0:
            path = os.path.join(args.save_dir, f"checkpoint_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_reward": avg_rwd,
            }, path)

    # Final save
    path = os.path.join(args.save_dir, "final_model.pt")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "avg_reward": avg_rwd,
    }, path)
    print(f"\nTraining complete. Final model saved to {path}")
    print(f"Best avg reward: {best_avg_reward:.5f}")


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PPO training for viewpoint selection policy")

    # Mode
    parser.add_argument("--mode", choices=["cached", "sim"], default="sim",
                        help="Environment mode: 'cached' (replay data) or 'sim' (random scenes)")

    # Data (cached mode only)
    parser.add_argument("--data", default=None,
                        help="Path to transitions.npz (required for cached mode)")
    parser.add_argument("--obs-res", type=int, default=16,
                        help="Down-sampled occupancy grid resolution (default: 16)")

    # Simulation params (sim mode only)
    parser.add_argument("--grid-res", type=int, default=64,
                        help="Sim: full occupancy grid resolution (default: 64)")
    parser.add_argument("--half-extent", type=float, default=0.3,
                        help="Sim: half side-length of voxel volume in metres")
    parser.add_argument("--min-objects", type=int, default=3,
                        help="Sim: min random objects per scene")
    parser.add_argument("--max-objects", type=int, default=8,
                        help="Sim: max random objects per scene")
    parser.add_argument("--min-obj-size", type=float, default=0.02,
                        help="Sim: min object half-extent (metres)")
    parser.add_argument("--max-obj-size", type=float, default=0.08,
                        help="Sim: max object half-extent (metres)")

    # Environment
    parser.add_argument("--max-ep-steps", type=int, default=20,
                        help="Max steps per episode (default: 20)")

    # Model
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden dimension in trunk (default: 128)")

    # PPO hyperparams
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--ppo-epochs", type=int, default=4,
                        help="Mini-batch passes per PPO update (default: 4)")
    parser.add_argument("--batch-size", type=int, default=64)

    # Training schedule
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of collection+update epochs (default: 200)")
    parser.add_argument("--steps-per-epoch", type=int, default=512,
                        help="Env steps per epoch (default: 512)")

    # Logging / saving
    parser.add_argument("--save-dir", default="checkpoints/viewpoint_ppo",
                        help="Checkpoint directory")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=50)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    train(args)


if __name__ == "__main__":
    main()
