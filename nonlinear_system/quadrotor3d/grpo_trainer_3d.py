#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO-style RL trainer for Quadrotor3DEnv.

This implements grouped relative policy optimization for continuous control:
1) Sample groups of rollouts from the same initial state.
2) Compute group-relative normalized returns as advantages.
3) Update stochastic policy with PPO-style clipped objective using those advantages.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quadrotor3d_env import Quadrotor3DEnv


def parse_args():
    p = argparse.ArgumentParser(description="GRPO trainer for quadrotor3d")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--max_time", type=float, default=10.0)
    p.add_argument("--pos_bound", type=float, default=0.3)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--updates", type=int, default=10000)
    p.add_argument("--groups_per_update", type=int, default=48)
    p.add_argument("--group_size", type=int, default=8)
    p.add_argument("--ppo_epochs", type=int, default=10)
    p.add_argument("--minibatch_size", type=int, default=2048)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--ent_coef", type=float, default=0.005)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument(
        "--save_dir",
        type=str,
        default="saved_models/grpo",
    )
    return p.parse_args()


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(obs_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, act_dim)]
        self.net = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def forward(self, obs: torch.Tensor):
        mean = self.net(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    @staticmethod
    def _atanh(x: torch.Tensor):
        x = torch.clamp(x, -0.999999, 0.999999)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def sample_action_and_logprob(self, obs: torch.Tensor):
        mean, std = self(obs)
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()
        action_norm = torch.tanh(z)
        # tanh correction
        log_prob = dist.log_prob(z) - torch.log(1 - action_norm.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action_norm, log_prob, entropy

    def evaluate_logprob_entropy(self, obs: torch.Tensor, action_norm: torch.Tensor):
        mean, std = self(obs)
        dist = torch.distributions.Normal(mean, std)
        z = self._atanh(action_norm)
        log_prob = dist.log_prob(z) - torch.log(1 - action_norm.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


@dataclass
class EpisodeBatch:
    obs: torch.Tensor
    action_norm: torch.Tensor
    old_logprob: torch.Tensor
    advantage: torch.Tensor
    ep_return: float


class GRPOTrainer3D:
    def __init__(self, args):
        self.args = args
        if args.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(args.device)

        self.env = Quadrotor3DEnv(dt=args.dt, max_time=args.max_time, pos_bound=args.pos_bound, seed=args.seed)
        self.obs_dim = int(self.env.observation_space.shape[0])
        self.act_dim = int(self.env.action_space.shape[0])
        self.action_low = torch.tensor(self.env.action_space.low, dtype=torch.float32, device=self.device)
        self.action_high = torch.tensor(self.env.action_space.high, dtype=torch.float32, device=self.device)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

        self.policy = GaussianPolicy(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.lr)

        os.makedirs(args.save_dir, exist_ok=True)
        self.avg_return_history = []
        self.loss_history = []
        self.kl_history = []

    def _norm_to_physical(self, action_norm: torch.Tensor):
        return self.action_bias + self.action_scale * action_norm

    def _rollout_episode(self, init_state: np.ndarray) -> Dict:
        _ = self.env.reset()
        self.env.x_current = torch.tensor(init_state, dtype=torch.float32)
        obs = init_state.astype(np.float32)

        obs_list = []
        action_norm_list = []
        logprob_list = []
        rewards = []

        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action_norm, logprob, _ = self.policy.sample_action_and_logprob(obs_t)
            action_norm = action_norm.squeeze(0)
            logprob = logprob.squeeze(0)
            action = self._norm_to_physical(action_norm).detach().cpu().numpy().astype(np.float32)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)

            obs_list.append(obs.copy())
            action_norm_list.append(action_norm.detach().cpu().numpy().copy())
            logprob_list.append(float(logprob.detach().cpu().item()))
            rewards.append(float(reward))

            obs = next_obs.astype(np.float32)
            done = bool(terminated or truncated)

        return {
            "obs": np.asarray(obs_list, dtype=np.float32),
            "action_norm": np.asarray(action_norm_list, dtype=np.float32),
            "old_logprob": np.asarray(logprob_list, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "return": float(np.sum(rewards)),
        }

    def _sample_group_batches(self) -> List[EpisodeBatch]:
        batches = []
        for _ in range(self.args.groups_per_update):
            obs0, _ = self.env.reset()
            init_state = self.env.x_current.detach().cpu().numpy().astype(np.float32)

            group_rollouts = [self._rollout_episode(init_state) for _ in range(self.args.group_size)]
            group_returns = np.asarray([ep["return"] for ep in group_rollouts], dtype=np.float32)
            r_mean = float(np.mean(group_returns))
            r_std = float(np.std(group_returns))
            adv_group = (group_returns - r_mean) / (r_std + 1e-6)

            for ep, adv in zip(group_rollouts, adv_group):
                T = ep["obs"].shape[0]
                if T == 0:
                    continue
                batches.append(
                    EpisodeBatch(
                        obs=torch.tensor(ep["obs"], dtype=torch.float32, device=self.device),
                        action_norm=torch.tensor(ep["action_norm"], dtype=torch.float32, device=self.device),
                        old_logprob=torch.tensor(ep["old_logprob"], dtype=torch.float32, device=self.device),
                        advantage=torch.full((T,), float(adv), dtype=torch.float32, device=self.device),
                        ep_return=float(ep["return"]),
                    )
                )
        return batches

    def _optimize_policy(self, batches: List[EpisodeBatch]):
        obs = torch.cat([b.obs for b in batches], dim=0)
        action_norm = torch.cat([b.action_norm for b in batches], dim=0)
        old_logprob = torch.cat([b.old_logprob for b in batches], dim=0)
        adv = torch.cat([b.advantage for b in batches], dim=0)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        N = obs.shape[0]
        all_idx = np.arange(N)
        loss_meter = []
        kl_meter = []

        for _ in range(self.args.ppo_epochs):
            np.random.shuffle(all_idx)
            for start in range(0, N, self.args.minibatch_size):
                idx = all_idx[start : start + self.args.minibatch_size]
                idx_t = torch.tensor(idx, device=self.device, dtype=torch.long)

                obs_b = obs[idx_t]
                action_b = action_norm[idx_t]
                old_logprob_b = old_logprob[idx_t]
                adv_b = adv[idx_t]

                new_logprob, entropy = self.policy.evaluate_logprob_entropy(obs_b, action_b)
                ratio = torch.exp(new_logprob - old_logprob_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.args.clip_eps, 1.0 + self.args.clip_eps) * adv_b
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                entropy_loss = -self.args.ent_coef * torch.mean(entropy)
                loss = policy_loss + entropy_loss

                approx_kl = torch.mean(old_logprob_b - new_logprob).detach().cpu().item()
                kl_meter.append(float(approx_kl))
                loss_meter.append(float(loss.detach().cpu().item()))

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

        return float(np.mean(loss_meter)), float(np.mean(kl_meter))

    def _save(self, step: int):
        ckpt = os.path.join(self.args.save_dir, f"grpo_quadrotor3d_step{step:04d}.pth")
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "args": vars(self.args),
                "step": step,
            },
            ckpt,
        )
        print(f"Saved checkpoint: {ckpt}")

    def _save_plots(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.avg_return_history, label="avg group return")
        plt.xlabel("update")
        plt.ylabel("return")
        plt.title("GRPO Quadrotor3D: Return")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out1 = os.path.join(self.args.save_dir, "grpo_return_curve.png")
        plt.savefig(out1, dpi=200)
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history, label="policy loss")
        plt.plot(self.kl_history, label="approx KL")
        plt.xlabel("update")
        plt.ylabel("value")
        plt.title("GRPO Quadrotor3D: Optimization Metrics")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out2 = os.path.join(self.args.save_dir, "grpo_loss_kl_curve.png")
        plt.savefig(out2, dpi=200)
        plt.close()
        print(f"Saved plots: {out1}, {out2}")

    def train(self):
        for update in range(1, self.args.updates + 1):
            batches = self._sample_group_batches()
            if len(batches) == 0:
                print(f"[update {update:04d}] no valid episodes sampled, skipping")
                continue

            avg_return = float(np.mean([b.ep_return for b in batches]))
            loss, approx_kl = self._optimize_policy(batches)

            self.avg_return_history.append(avg_return)
            self.loss_history.append(loss)
            self.kl_history.append(approx_kl)

            print(
                f"[update {update:04d}] episodes={len(batches)} "
                f"avg_return={avg_return:.6f} loss={loss:.6f} approx_kl={approx_kl:.6f}"
            )

            if update % self.args.save_every == 0:
                self._save(update)

        self._save(self.args.updates)
        self._save_plots()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    trainer = GRPOTrainer3D(args)
    trainer.train()


if __name__ == "__main__":
    main()
