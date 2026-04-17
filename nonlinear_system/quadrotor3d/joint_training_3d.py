#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jointly train quadrotor3d controller and stability certificate with multi-step Lyapunov loss.

Alternating loop per iteration:
1) RL policy update (SAC) on environment reward.
2) Certificate update (StepNet + LyapunovNet) on multi-step Lyapunov violation.
3) Controller fine-tune (actor only) to reduce the same multi-step Lyapunov violation.
"""

import argparse
import os
from typing import Tuple

import numpy as np
import torch
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from quadrotor3d_env import Quadrotor3DEnv
from network.PolicyNet import StepNet
from network.lyapunov_net import LyapunovNet


def parse_args():
    parser = argparse.ArgumentParser(description="Joint controller + certificate training for Quadrotor3D")
    parser.add_argument("--device", type=str, default="auto", help="cuda/cpu/auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--algo", type=str, default="sac", choices=["sac"])
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--max_time", type=float, default=2.0)
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--n_steps", type=int, default=15, help="Lyapunov multi-step horizon")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--policy_steps_per_iter", type=int, default=5000)
    parser.add_argument("--cert_updates_per_iter", type=int, default=50)
    parser.add_argument("--ctrl_updates_per_iter", type=int, default=20)
    parser.add_argument("--cert_batch_size", type=int, default=128)
    parser.add_argument("--ctrl_batch_size", type=int, default=64)
    parser.add_argument("--cert_lr", type=float, default=2e-4)
    parser.add_argument("--ctrl_lr", type=float, default=1e-4)
    parser.add_argument("--lyap_penalty", type=float, default=1.0)
    parser.add_argument("--action_reg", type=float, default=1e-4)
    parser.add_argument(
        "--init_model_path",
        type=str,
        default="nonlinear_system/quadrotor3d/saved_models/ppo/quadrotor3d_model.zip",
        help="Optional initial checkpoint (SAC or PPO). If PPO, SAC actor is warm-started by imitation.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="nonlinear_system/quadrotor3d/saved_models/joint",
        help="Directory for joint training checkpoints",
    )
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument(
        "--warmstart_steps",
        type=int,
        default=1000,
        help="Actor warm-start gradient steps if PPO checkpoint is used as base",
    )
    parser.add_argument(
        "--warmstart_batch_size",
        type=int,
        default=512,
        help="Batch size for PPO->SAC actor warm-start",
    )
    return parser.parse_args()


class JointTrainerQuadrotor3D:
    def __init__(self, args):
        self.args = args
        if args.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(args.device)

        self.dtype = torch.float32
        self.n_steps = int(args.n_steps)
        self.alpha = float(args.alpha)
        self.beta = float(args.beta)
        self.lyap_penalty = float(args.lyap_penalty)
        self.action_reg = float(args.action_reg)
        self.dt = float(args.dt)

        self.rollout_env = Quadrotor3DEnv(dt=args.dt, max_time=args.max_time)

        def _make_env():
            return Monitor(Quadrotor3DEnv(dt=args.dt, max_time=args.max_time))

        self.train_env = make_vec_env(_make_env, n_envs=args.n_envs, seed=args.seed)

        self.state_dim = int(self.rollout_env.observation_space.shape[0])
        self.action_dim = int(self.rollout_env.action_space.shape[0])
        self.action_low = torch.tensor(
            self.rollout_env.action_space.low, dtype=self.dtype, device=self.device
        )
        self.action_high = torch.tensor(
            self.rollout_env.action_space.high, dtype=self.dtype, device=self.device
        )

        self.model = self._load_or_create_sac(args.init_model_path)

        self.stepnet = StepNet(
            n_input=self.state_dim,
            n_hidden=args.hidden_dim,
            n_steps=self.n_steps,
            n_layers=args.n_layers,
        ).to(self.device)
        self.residual_net = LyapunovNet(
            n_input=self.state_dim,
            n_hidden=args.hidden_dim,
            n_layers=args.n_layers,
        ).to(self.device)

        self.stepnet_opt = torch.optim.Adam(self.stepnet.parameters(), lr=args.cert_lr)
        self.residual_opt = torch.optim.Adam(self.residual_net.parameters(), lr=args.cert_lr)
        self.actor_opt = torch.optim.Adam(self.model.actor.parameters(), lr=args.ctrl_lr)

        self.eq_obs = self._get_equilibrium_obs()
        self.eq_obs_t = torch.tensor(self.eq_obs, dtype=self.dtype, device=self.device).unsqueeze(0)
        with torch.no_grad():
            self.v_eq = float(self._vrl(self.eq_obs_t, detach=True).item())

        os.makedirs(args.save_dir, exist_ok=True)
        print(f"Using device: {self.device}")
        print(f"Initial model path: {args.init_model_path}")
        print(f"Checkpoint directory: {os.path.abspath(args.save_dir)}")

    def _sample_obs_uniform_np(self, batch_size: int) -> np.ndarray:
        low = np.asarray(self.rollout_env.observation_space.low, dtype=np.float32)
        high = np.asarray(self.rollout_env.observation_space.high, dtype=np.float32)
        low = np.where(np.isfinite(low), low, -1.0)
        high = np.where(np.isfinite(high), high, 1.0)
        return np.random.uniform(low, high, size=(batch_size, self.state_dim)).astype(np.float32)

    def _to_normalized_action(self, action_physical: np.ndarray) -> np.ndarray:
        low = self.rollout_env.action_space.low.astype(np.float32)
        high = self.rollout_env.action_space.high.astype(np.float32)
        denom = np.maximum(high - low, 1e-6)
        a_norm = 2.0 * (action_physical - low) / denom - 1.0
        return np.clip(a_norm, -1.0, 1.0).astype(np.float32)

    def _warmstart_actor_from_ppo(self, sac_model: SAC, ppo_model: PPO):
        print(
            f"Warm-starting SAC actor from PPO policy "
            f"(steps={self.args.warmstart_steps}, batch={self.args.warmstart_batch_size})"
        )
        opt = torch.optim.Adam(sac_model.actor.parameters(), lr=self.args.ctrl_lr)
        for step in range(self.args.warmstart_steps):
            obs_np = self._sample_obs_uniform_np(self.args.warmstart_batch_size)
            teacher_action, _ = ppo_model.predict(obs_np, deterministic=True)
            teacher_action = self._to_normalized_action(np.asarray(teacher_action, dtype=np.float32))

            obs_t = torch.tensor(obs_np, dtype=self.dtype, device=self.device)
            target_t = torch.tensor(teacher_action, dtype=self.dtype, device=self.device)
            pred_t = sac_model.actor(obs_t)
            loss = torch.mean((pred_t - target_t) ** 2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sac_model.actor.parameters(), 5.0)
            opt.step()

            if (step + 1) % max(1, self.args.warmstart_steps // 5) == 0:
                print(f"  warmstart step {step + 1}/{self.args.warmstart_steps} | mse={loss.item():.6f}")

    def _load_or_create_sac(self, init_path: str):
        if os.path.exists(init_path):
            try:
                print(f"Loading SAC from {init_path}")
                return SAC.load(init_path, env=self.train_env, device=self.device)
            except Exception:
                print(f"SAC load failed for {init_path}, trying PPO warm-start")
                ppo_teacher = PPO.load(init_path, env=self.train_env, device=self.device)
                sac_model = SAC(
                    "MlpPolicy",
                    self.train_env,
                    policy_kwargs=dict(net_arch=dict(pi=[512, 512], qf=[512, 512])),
                    verbose=0,
                    device=self.device,
                    seed=self.args.seed,
                    buffer_size=300_000,
                    batch_size=256,
                    gamma=0.99,
                    learning_rate=3e-4,
                    train_freq=1,
                    gradient_steps=1,
                )
                self._warmstart_actor_from_ppo(sac_model, ppo_teacher)
                return sac_model
        print("Initial model not found, creating SAC from scratch")
        return SAC(
            "MlpPolicy",
            self.train_env,
            policy_kwargs=dict(net_arch=dict(pi=[512, 512], qf=[512, 512])),
            verbose=0,
            device=self.device,
            seed=self.args.seed,
            buffer_size=300_000,
            batch_size=256,
            gamma=0.99,
            learning_rate=3e-4,
            train_freq=1,
            gradient_steps=1,
        )

    def _get_equilibrium_obs(self) -> np.ndarray:
        eq = self.rollout_env.obs_equ
        if torch.is_tensor(eq):
            return eq.detach().cpu().numpy().astype(np.float32)
        return np.asarray(eq, dtype=np.float32)

    def _sample_initial_obs(self, batch_size: int) -> np.ndarray:
        low = np.asarray(self.rollout_env.observation_space.low, dtype=np.float32)
        high = np.asarray(self.rollout_env.observation_space.high, dtype=np.float32)
        low = np.where(np.isfinite(low), low, -1.0)
        high = np.where(np.isfinite(high), high, 1.0)
        return np.random.uniform(low, high, size=(batch_size, self.state_dim)).astype(np.float32)

    def _env_reset_to_obs(self, obs0: np.ndarray) -> np.ndarray:
        self.rollout_env.reset()
        self.rollout_env.x_current = torch.tensor(obs0, dtype=torch.float32)
        return obs0.astype(np.float32)

    def _collect_trajectory(self, obs0: np.ndarray, horizon: int) -> np.ndarray:
        obs = self._env_reset_to_obs(obs0)
        states = []
        for _ in range(horizon):
            states.append(obs.copy())
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = self.rollout_env.step(action)
            if terminated or truncated:
                break
        if len(states) == 0:
            return np.zeros((0, self.state_dim), dtype=np.float32)
        return np.asarray(states, dtype=np.float32)

    def _normalize_sigma(self, sigma_raw: torch.Tensor) -> torch.Tensor:
        sigma = torch.relu(sigma_raw).view(-1)[: self.n_steps]
        return sigma / (sigma.sum() + 1e-8)

    def _vrl(self, obs_batched: torch.Tensor, detach: bool) -> torch.Tensor:
        if detach:
            with torch.no_grad():
                action = self.model.actor(obs_batched)
                q1, q2 = self.model.critic(obs_batched, action)
                return torch.min(q1, q2).view(-1)
        action = self.model.actor(obs_batched)
        q1, q2 = self.model.critic(obs_batched, action)
        return torch.min(q1, q2).view(-1)

    def _lyapunov_value(self, obs_batched: torch.Tensor, detach_vrl: bool = True) -> torch.Tensor:
        v_rl = self._vrl(obs_batched, detach=detach_vrl)
        term1 = torch.abs(v_rl - self.v_eq)

        phi_x = self.residual_net(obs_batched).view(-1)
        phi_eq = self.residual_net(self.eq_obs_t).view(-1)[0]
        term2 = (phi_x - phi_eq) ** 2

        diff = obs_batched - self.eq_obs_t
        term3 = self.beta * torch.sum(diff**2, dim=-1)
        return term1 + term2 + term3

    def _certificate_loss(self, batch_obs0: np.ndarray) -> torch.Tensor:
        total = torch.zeros((), device=self.device)
        valid = 0
        for obs0 in batch_obs0:
            traj_np = self._collect_trajectory(obs0, horizon=self.n_steps + 1)
            if traj_np.shape[0] < self.n_steps + 1:
                continue
            valid += 1

            traj = torch.tensor(traj_np[: self.n_steps + 1], dtype=self.dtype, device=self.device)
            x0 = traj[0:1]
            sigma = self._normalize_sigma(self.stepnet(x0))
            V = self._lyapunov_value(traj, detach_vrl=True)
            V0 = V[0]
            future = V[1:]
            weighted_future = torch.sum(sigma * future)
            violation = torch.relu(weighted_future - (1.0 - self.alpha) * V0)
            total = total + violation

        if valid == 0:
            return total
        return total / float(valid)

    def _unscale_action(self, a_norm: torch.Tensor) -> torch.Tensor:
        return self.action_low + (a_norm + 1.0) * 0.5 * (self.action_high - self.action_low)

    def _dynamics_step(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # x: [B,12], u:[B,4]
        m = 0.468
        g = 9.81
        arm = 0.225
        zfac = 1.1 / 29.0
        Ixx, Iyy, Izz = 4.9e-3, 4.9e-3, 8.8e-3

        roll = x[:, 3]
        pitch = x[:, 4]
        yaw = x[:, 5]
        vel = x[:, 6:9]
        wx = x[:, 9]
        wy = x[:, 10]
        wz = x[:, 11]
        omega = x[:, 9:12]

        total_thrust = torch.sum(u, dim=1)
        tau_x = arm * (u[:, 1] - u[:, 3])
        tau_y = arm * (-u[:, 0] + u[:, 2])
        tau_z = zfac * (u[:, 0] - u[:, 1] + u[:, 2] - u[:, 3])

        sr, cr = torch.sin(roll), torch.cos(roll)
        sp, cp = torch.sin(pitch), torch.cos(pitch)
        sy, cy = torch.sin(yaw), torch.cos(yaw)
        tp = sp / (cp + 1e-6)

        # Third column of Rz(yaw)Ry(pitch)Rx(roll)
        r13 = cy * sp * cr + sy * sr
        r23 = sy * sp * cr - cy * sr
        r33 = cp * cr
        pos_ddot = torch.stack(
            [r13 * total_thrust / m, r23 * total_thrust / m, -g + r33 * total_thrust / m],
            dim=1,
        )

        rpy_dot = torch.stack(
            [
                wx + sr * tp * wy + cr * tp * wz,
                cr * wy - sr * wz,
                sr / (cp + 1e-6) * wy + cr / (cp + 1e-6) * wz,
            ],
            dim=1,
        )

        omega_dot = torch.stack(
            [
                ((Iyy - Izz) / Ixx) * wy * wz + tau_x / Ixx,
                ((Izz - Ixx) / Iyy) * wx * wz + tau_y / Iyy,
                ((Ixx - Iyy) / Izz) * wx * wy + tau_z / Izz,
            ],
            dim=1,
        )

        xdot = torch.cat([vel, rpy_dot, pos_ddot, omega_dot], dim=1)
        x_next = x + self.dt * xdot

        # Wrap yaw to [-pi, pi]
        yaw_next = torch.atan2(torch.sin(x_next[:, 5]), torch.cos(x_next[:, 5]))
        x_next = x_next.clone()
        x_next[:, 5] = yaw_next
        return x_next

    def _controller_lyapunov_loss(self, batch_size: int) -> torch.Tensor:
        obs0_np = self._sample_initial_obs(batch_size)
        x = torch.tensor(obs0_np, dtype=self.dtype, device=self.device)
        traj = [x]
        action_norm_hist = []

        for _ in range(self.n_steps):
            a_norm = self.model.actor(x)
            action_norm_hist.append(a_norm)
            a = self._unscale_action(a_norm)
            x = self._dynamics_step(x, a)
            traj.append(x)

        traj_t = torch.stack(traj, dim=1)  # [B, n_steps+1, state_dim]
        x0 = traj_t[:, 0, :]
        sigma = self._normalize_sigma(self.stepnet(x0)).unsqueeze(0).repeat(batch_size, 1)

        flat = traj_t.reshape(-1, self.state_dim)
        V = self._lyapunov_value(flat, detach_vrl=True).reshape(batch_size, self.n_steps + 1)
        V0 = V[:, 0]
        future = V[:, 1:]
        weighted_future = torch.sum(sigma * future, dim=1)
        violation = torch.relu(weighted_future - (1.0 - self.alpha) * V0)

        action_reg = torch.mean(torch.stack([a.pow(2).mean() for a in action_norm_hist]))
        return self.lyap_penalty * violation.mean() + self.action_reg * action_reg

    def _set_requires_grad(self, module: torch.nn.Module, requires_grad: bool):
        for p in module.parameters():
            p.requires_grad = requires_grad

    def _update_certificates(self):
        cert_losses = []
        for _ in range(self.args.cert_updates_per_iter):
            batch = self._sample_initial_obs(self.args.cert_batch_size)
            self.stepnet_opt.zero_grad(set_to_none=True)
            self.residual_opt.zero_grad(set_to_none=True)
            loss = self._certificate_loss(batch)
            if loss.requires_grad and loss.grad_fn is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.stepnet.parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(self.residual_net.parameters(), 5.0)
                self.stepnet_opt.step()
                self.residual_opt.step()
            cert_losses.append(float(loss.detach().cpu().item()))
        return float(np.mean(cert_losses)) if cert_losses else 0.0

    def _update_controller(self):
        ctrl_losses = []
        self._set_requires_grad(self.stepnet, False)
        self._set_requires_grad(self.residual_net, False)
        self._set_requires_grad(self.model.critic, False)
        self._set_requires_grad(self.model.actor, True)

        for _ in range(self.args.ctrl_updates_per_iter):
            self.actor_opt.zero_grad(set_to_none=True)
            loss = self._controller_lyapunov_loss(self.args.ctrl_batch_size)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), 5.0)
            self.actor_opt.step()
            ctrl_losses.append(float(loss.detach().cpu().item()))

        self._set_requires_grad(self.model.critic, True)
        self._set_requires_grad(self.stepnet, True)
        self._set_requires_grad(self.residual_net, True)
        return float(np.mean(ctrl_losses)) if ctrl_losses else 0.0

    def save(self, tag: str):
        model_path = os.path.join(self.args.save_dir, f"joint_sac_quadrotor3d_{tag}")
        self.model.save(model_path)
        torch.save(self.stepnet.state_dict(), os.path.join(self.args.save_dir, f"stepnet_{tag}.pth"))
        torch.save(
            self.residual_net.state_dict(),
            os.path.join(self.args.save_dir, f"residual_net_{tag}.pth"),
        )

    def train(self):
        for it in range(1, self.args.iterations + 1):
            self.model.learn(
                total_timesteps=self.args.policy_steps_per_iter,
                reset_num_timesteps=False,
                progress_bar=False,
            )
            cert_loss = self._update_certificates()
            ctrl_loss = self._update_controller()

            eval_batch = self._sample_initial_obs(32)
            with torch.no_grad():
                eval_violation = float(self._certificate_loss(eval_batch).detach().cpu().item())

            print(
                f"[iter {it:04d}] cert_loss={cert_loss:.6f} ctrl_lyap_loss={ctrl_loss:.6f} "
                f"eval_violation={eval_violation:.6f}"
            )

            if it % self.args.save_every == 0 or it == self.args.iterations:
                self.save(tag=f"iter{it:04d}")


def main():
    args = parse_args()
    trainer = JointTrainerQuadrotor3D(args)
    trainer.train()


if __name__ == "__main__":
    main()
