#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train two specialised quadrotor controllers using Quadrotor3DEnv.

Both controllers are plain Quadrotor3DEnv instances — differentiated only
by their LQR cost weights (q_pos, q_angle, q_vel, q_omega).

  Controller A — Attitude stabiliser
      q_angle / q_omega large  → strong attitude + rate penalty
      q_pos / q_vel small      → position is secondary

  Controller B — Position stabiliser
      q_pos / q_vel large      → strong position + velocity penalty
      q_angle moderate         → keep level but secondary goal

Curriculum
----------
CurriculumQuadrotor3DEnv is a one-method subclass of Quadrotor3DEnv.
It stores the full-scale (x_lo, x_up) at construction and exposes
set_scale(s) so CurriculumCallback can linearly expand the reset
distribution from init_scale → 1.0 over the first anneal_fraction of
training, then hold at 1.0.
"""

import os
import sys
import argparse
import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quadrotor3d_env import Quadrotor3DEnv, wrap_to_pi


# ============================================================================
# Curriculum wrapper — the ONLY addition on top of Quadrotor3DEnv
# ============================================================================

class CurriculumQuadrotor3DEnv(Quadrotor3DEnv):
    """
    Extends Quadrotor3DEnv with a single method: set_scale(s).

    At construction the full x_lo / x_up are saved. set_scale(s) replaces
    self.x_lo / self.x_up with s * full_bounds, keeping:
      - pz lower bound = 0  (never below ground)
      - yaw range = [-π, π] always
    The observation_space is never changed — it always spans the full range.
    """

    def __init__(self, init_scale: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        # Save the full-scale bounds set by Quadrotor3DEnv.__init__
        self._x_lo_full = self.x_lo.clone()
        self._x_up_full = self.x_up.clone()
        # Apply the starting curriculum scale
        self.set_scale(init_scale)

    def set_scale(self, scale: float):
        """Shrink/expand the reset sampling range. Called by CurriculumCallback."""
        s = float(np.clip(scale, 0.0, 1.0))
        # Scale deviation from equilibrium, not raw bounds
        equ_z = self.obs_equ[2].item()
        lo_dev = self._x_lo_full.clone(); lo_dev[2] -= equ_z
        up_dev = self._x_up_full.clone(); up_dev[2] -= equ_z
        self.x_lo = self._x_lo_full.clone(); self.x_lo[2] = torch.tensor(equ_z + lo_dev[2].item() * s, dtype=self.dtype)
        self.x_up = self._x_up_full.clone(); self.x_up[2] = torch.tensor(equ_z + up_dev[2].item() * s, dtype=self.dtype)
        # Scale x/y and velocity/rate dimensions
        for i in [0, 1, 6, 7, 8, 9, 10, 11]:
            self.x_lo[i] = self._x_lo_full[i] * s
            self.x_up[i] = self._x_up_full[i] * s
        # Invariants: yaw always full circle, z never below ground
        self.x_lo[2] = torch.clamp(self.x_lo[2], min=torch.tensor(0.5, dtype=self.dtype))
        self.x_lo[5] = torch.tensor(-np.pi, dtype=self.dtype)
        self.x_up[5] = torch.tensor( np.pi, dtype=self.dtype)


# ============================================================================
# Curriculum callback
# ============================================================================

class CurriculumCallback(BaseCallback):
    """
    Increases reset_scale only when the success rate over the past
    `window` episodes exceeds `success_threshold`.  Scale advances by
    `scale_step` each time the threshold is met, up to 1.0.
    """

    def __init__(
        self,
        init_scale: float         = 0.2,
        scale_step: float         = 0.05,
        success_threshold: float  = 0.5,
        window: int               = 100,
        verbose: int              = 1,
    ):
        super().__init__(verbose)
        self._scale             = float(init_scale)
        self.scale_step         = float(scale_step)
        self.success_threshold  = float(success_threshold)
        self.window             = int(window)
        self._recent: list      = []   # ring buffer of bool (success per episode)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            self._recent.append(bool(info.get("hover_region", False)))
            if len(self._recent) > self.window:
                self._recent.pop(0)

            if len(self._recent) == self.window and self._scale < 1.0:
                rate = sum(self._recent) / self.window
                if rate >= self.success_threshold:
                    old = self._scale
                    self._scale = min(1.0, self._scale + self.scale_step)
                    self._recent.clear()
                    self.training_env.env_method("set_scale", self._scale)
                    if self.verbose >= 1:
                        print(
                            f"  [curriculum] step={self.num_timesteps:,}  "
                            f"success={rate:.2f} >= {self.success_threshold:.2f}  "
                            f"scale {old:.3f} -> {self._scale:.3f}"
                        )
        return True


class EpisodeLogger(BaseCallback):
    def __init__(self, print_freq=1000, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self._ep = 0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._ep += 1
                if self._ep % self.print_freq == 0:
                    r = info["episode"]["r"]
                    l = info["episode"]["l"]
                    print(f"  ep {self._ep:6d} | return {r:9.3f} | length {l}")
        return True


# ============================================================================
# SAC with larger network
# ============================================================================

def build_sac(env, seed, device):
    return SAC(
        "MlpPolicy",
        env,
        policy_kwargs=dict(
            net_arch=dict(pi=[512, 512, 256], qf=[512, 512, 256]),
        ),
        verbose=0,
        device=device,
        seed=seed,
        buffer_size=500_000,
        batch_size=512,
        gamma=0.99,
        learning_rate=3e-4,
        train_freq=1,
        gradient_steps=2,
        ent_coef="auto",
        learning_starts=5_000,
    )


# ============================================================================
# Training
# ============================================================================

def train(
    env_kwargs,
    name,
    save_dir,
    timesteps,
    n_envs=4,
    seed=0,
    init_scale=0.2,
    scale_step=0.05,
    success_threshold=0.5,
    window=100,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"Training : {name}")
    print(f"Device   : {device}  |  envs: {n_envs}  |  steps: {timesteps:,}")
    print(f"Curriculum: init_scale={init_scale}  scale_step={scale_step}  "
          f"threshold={success_threshold:.0%}  window={window}")
    print(f"{'='*60}")

    def _make():
        return Monitor(CurriculumQuadrotor3DEnv(**env_kwargs))

    train_env = make_vec_env(_make, n_envs=n_envs, seed=seed)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                             clip_obs=10.0, clip_reward=20.0)
    model = build_sac(train_env, seed, device)

    ckpt_dir = os.path.join(save_dir, f"{name}_ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    callbacks = [
        CurriculumCallback(
            init_scale=init_scale,
            scale_step=scale_step,
            success_threshold=success_threshold,
            window=window,
            verbose=1,
        ),
        EpisodeLogger(print_freq=500),
        CheckpointCallback(
            save_freq=max(1, 100_000 // n_envs),
            save_path=ckpt_dir,
            name_prefix=name,
        ),
    ]

    model.learn(total_timesteps=timesteps, callback=callbacks)

    out = os.path.join(save_dir, name)
    model.save(out)
    vecnorm_out = out + "_vecnorm.pkl"
    train_env.save(vecnorm_out)
    print(f"Saved -> {out}.zip  |  VecNormalize -> {vecnorm_out}")
    train_env.close()
    return model


# ============================================================================
# Evaluation
# ============================================================================

def _make_eval_env(env_kwargs, vecnorm_path=None):
    """Create a (optionally VecNormalize-wrapped) single eval env."""
    raw = CurriculumQuadrotor3DEnv(**env_kwargs)
    vec = DummyVecEnv([lambda: raw])
    if vecnorm_path is not None and os.path.exists(vecnorm_path):
        vec = VecNormalize.load(vecnorm_path, vec)
        vec.training = False
        vec.norm_reward = False
    return vec, raw


def evaluate(model, env_kwargs, n_episodes=20, seed=0, vecnorm_path=None):
    """Evaluate at full scale (init_scale=1.0)."""
    full_kwargs = {**env_kwargs, "init_scale": 1.0}
    results = []
    for ep in range(n_episodes):
        vec_env, raw_env = _make_eval_env(full_kwargs, vecnorm_path)
        obs = vec_env.reset()
        ep_ret, ep_len = 0.0, 0
        success = False

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, done, info_list = vec_env.step(action)
            info = info_list[0]
            ep_ret += float(rew[0])
            ep_len += 1
            if info.get("hover_region"):
                success = True
            if done[0]:
                break

        vec_env.close()
        results.append({"return": ep_ret, "length": ep_len, "success": success})
    return results


# ============================================================================
# Rollout plot
# ============================================================================

def rollout_and_plot(model, env_kwargs, save_path, seed=0, n_steps=500, vecnorm_path=None):
    full_kwargs = {**env_kwargs, "init_scale": 1.0}
    vec_env, raw_env = _make_eval_env(full_kwargs, vecnorm_path)
    obs = vec_env.reset()

    states, actions, rewards = [], [], []
    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, done, _ = vec_env.step(action)
        # record raw (unnormalized) state for plotting
        raw_state = raw_env.x_current.detach().cpu().numpy().astype(np.float32).copy()
        raw_state[0:3] -= raw_env.obs_equ.numpy()[0:3]
        states.append(raw_state)
        actions.append(action[0].copy())
        rewards.append(float(rew[0]))
        if done[0]:
            break

    vec_env.close()
    states  = np.array(states,  dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)
    T = states.shape[0]
    t = np.arange(T) * env_kwargs.get("dt", 0.01)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    ax = axes[0]
    for i, lbl in enumerate(["x","y","z","roll","pitch","yaw","vx","vy","vz","wx","wy","wz"]):
        ax.plot(t, states[:, i], label=lbl)
    ax.set_title("State trajectory (12D)")
    ax.set_xlabel("time (s)")
    ax.legend(ncol=6, fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i, lbl in enumerate(["u1","u2","u3","u4"]):
        ax.plot(t, actions[:, i], label=lbl)
    ax.set_title("Control inputs")
    ax.set_xlabel("time (s)")
    ax.legend(ncol=4, fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(t, rewards)
    ax.set_title("Reward")
    ax.set_xlabel("time (s)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()
    print(f"Rollout plot -> {save_path}")


def plot_comparison(results_A, results_B, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    def _box(ax, dataA, dataB, title, ylabel):
        ax.boxplot([dataA, dataB], labels=["Attitude\nStabiliser", "Position\nStabiliser"])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    _box(axes[0],
         [r["return"] for r in results_A],
         [r["return"] for r in results_B],
         "Episode Return", "Return")
    _box(axes[1],
         [r["length"] for r in results_A],
         [r["length"] for r in results_B],
         "Episode Length", "Steps")

    succ_A = np.mean([r["success"] for r in results_A])
    succ_B = np.mean([r["success"] for r in results_B])
    fig.suptitle(
        f"Controller comparison  |  "
        f"Attitude hover={succ_A:.0%}   Position hover={succ_B:.0%}",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()
    print(f"Comparison plot -> {save_path}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps_A",     type=int,   default=2_000_000)
    p.add_argument("--timesteps_B",     type=int,   default=3_000_000)
    p.add_argument("--n_envs",          type=int,   default=4)
    p.add_argument("--dt",              type=float, default=0.01)
    p.add_argument("--max_time",        type=float, default=5.0)
    # shared reset bounds (quadrotor3d_env defaults)
    p.add_argument("--pos_bound",       type=float, default=10.0)
    p.add_argument("--angle_bound",     type=float, default=0.7)
    p.add_argument("--vel_bound",       type=float, default=3.0)
    p.add_argument("--omega_bound",     type=float, default=0.2)
    # curriculum
    p.add_argument("--init_scale",          type=float, default=0.2)
    p.add_argument("--scale_step",          type=float, default=0.05)
    p.add_argument("--success_threshold",   type=float, default=0.5)
    p.add_argument("--window",              type=int,   default=100)
    p.add_argument("--seed",                type=int,   default=0)
    p.add_argument("--eval_only",       action="store_true")
    return p.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    save_dir    = os.path.join(script_dir, "saved_models", "two_controllers")
    results_dir = os.path.join(script_dir, "results",      "two_controllers")
    os.makedirs(save_dir,    exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Shared env construction kwargs
    shared = dict(
        dt          = args.dt,
        max_time    = args.max_time,
        pos_bound   = args.pos_bound,
        angle_bound = args.angle_bound,
        vel_bound   = args.vel_bound,
        omega_bound = args.omega_bound,
        init_scale  = args.init_scale,
    )

    # ── Controller A: attitude stabiliser ────────────────────────────────
    # High angle + omega weights; low position weight.
    kwargs_A = {
        **shared,
        "q_pos":   1.0,    # position drift is tolerated
        "q_angle": 50.0,   # strongly penalise roll / pitch / yaw
        "q_vel":   0.5,    # light velocity penalty
        "q_omega": 20.0,   # strongly penalise angular rates
        "r_thrust": 0.1,
        "reward_scale":     1e-3,
        "smoothness_weight": 0.05,
        "hover_bonus":      0.2,
        "terminate_penalty": 10.0,
        "angle_tol":  0.08,
        "omega_tol":  0.15,
        "pos_tol":    0.5,   # loose — not the main goal
        "vel_tol":    0.5,
    }

    # ── Controller B: full-equilibrium stabiliser ─────────────────────────
    # Co-primary goals: drive position/velocity AND roll/pitch/yaw/rates to
    # zero simultaneously.  Receives the drone after Controller A has already
    # brought attitude close to level, so initial angle/omega bounds are tight
    # (matching the switch condition of Controller A).
    kwargs_B = {
        **shared,
        # Reset distribution: attitude already near equilibrium at handoff
        "angle_bound": 0.08,   # matches Controller A's angle_tol
        "omega_bound": 0.15,   # matches Controller A's omega_tol
        "q_pos":   50.0,   # strongly penalise position error
        "q_angle": 20.0,   # co-primary: must keep/drive attitude to zero
        "q_vel":   20.0,   # strongly penalise velocity
        "q_omega": 10.0,   # angular rates must converge too
        "r_thrust": 0.1,
        "reward_scale":     1e-3,
        "smoothness_weight": 0.05,
        "hover_bonus":      0.2,
        "terminate_penalty": 10.0,
        "pos_tol":    0.08,
        "vel_tol":    0.20,
        "angle_tol":  0.08,
        "omega_tol":  0.15,
    }

    path_A = os.path.join(save_dir, "attitude_stabiliser")
    path_B = os.path.join(save_dir, "position_stabiliser")

    # ── Train or load ─────────────────────────────────────────────────────
    if args.eval_only:
        print("Loading pre-trained models...")
        model_A = SAC.load(path_A)
        model_B = SAC.load(path_B)
    else:
        model_A = train(
            kwargs_A,
            name="attitude_stabiliser",
            save_dir=save_dir,
            timesteps=args.timesteps_A,
            n_envs=args.n_envs,
            seed=args.seed,
            init_scale=args.init_scale,
            scale_step=args.scale_step,
            success_threshold=args.success_threshold,
            window=args.window,
        )
        model_B = train(
            kwargs_B,
            name="position_stabiliser",
            save_dir=save_dir,
            timesteps=args.timesteps_B,
            n_envs=args.n_envs,
            seed=args.seed,
            init_scale=args.init_scale,
            scale_step=args.scale_step,
            success_threshold=args.success_threshold,
            window=args.window,
        )

    vecnorm_A = os.path.join(save_dir, "attitude_stabiliser_vecnorm.pkl")
    vecnorm_B = os.path.join(save_dir, "position_stabiliser_vecnorm.pkl")

    # ── Evaluate ──────────────────────────────────────────────────────────
    print("\n--- Evaluating Controller A (Attitude Stabiliser) ---")
    results_A = evaluate(model_A, kwargs_A, n_episodes=20, seed=args.seed,
                         vecnorm_path=vecnorm_A)

    print("\n--- Evaluating Controller B (Position Stabiliser) ---")
    results_B = evaluate(model_B, kwargs_B, n_episodes=20, seed=args.seed,
                         vecnorm_path=vecnorm_B)

    for label, results in [("A (Attitude)", results_A), ("B (Position)", results_B)]:
        returns = [r["return"] for r in results]
        succs   = [r["success"] for r in results]
        print(f"\nController {label}:")
        print(f"  Mean return  : {np.mean(returns):.3f} ± {np.std(returns):.3f}")
        print(f"  Hover rate   : {np.mean(succs):.0%}")

    # ── Rollout plots ──────────────────────────────────────────────────────
    rollout_and_plot(model_A, kwargs_A,
                     save_path=os.path.join(results_dir, "rollout_attitude_stabiliser.png"),
                     vecnorm_path=vecnorm_A)
    rollout_and_plot(model_B, kwargs_B,
                     save_path=os.path.join(results_dir, "rollout_position_stabiliser.png"),
                     vecnorm_path=vecnorm_B)
    plot_comparison(results_A, results_B,
                    save_path=os.path.join(results_dir, "controller_comparison.png"))


if __name__ == "__main__":
    main()
