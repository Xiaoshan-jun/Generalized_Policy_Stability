#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train PPO or SAC on Quadrotor3DEnv (Gymnasium API) and save the model.

Includes a quadrotor-friendly rollout plot:
  - states x(t) (12 dims)
  - actions u(t) (4 dims)
  - reward(t)
"""

import os
import argparse
import numpy as np
import torch

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gymnasium as gym  # noqa: F401

# ---- CHANGE THIS IMPORT to match your file name ----
# from quadrotor3d_env import Quadrotor3DEnv
from quadrotor3d_env import Quadrotor3DEnv  # <-- edit module name if needed


class CurriculumCallback(BaseCallback):
    """Increases reset distribution scale only when episodes reach equilibrium.

    After `success_window` consecutive episodes where at least `success_threshold`
    fraction reached equilibrium, the scale increases by `scale_step`, up to 1.0.
    """

    def __init__(
        self,
        init_scale: float = 0.1,
        scale_step: float = 0.05,
        success_threshold: float = 0.5,
        success_window: int = 20,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self._scale            = float(init_scale)
        self.scale_step        = float(scale_step)
        self.success_threshold = float(success_threshold)
        self.success_window    = int(success_window)
        # ring buffer: True if that episode reached equilibrium
        self._recent: list     = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" not in info:
                continue
            # SB3 Monitor wraps episode info; hover_region is set per-step not per-episode.
            # We detect success from the episode's final info dict (last step before done).
            reached = bool(info.get("hover_region", False))
            self._recent.append(reached)
            if len(self._recent) > self.success_window:
                self._recent.pop(0)

            if len(self._recent) == self.success_window and self._scale < 1.0:
                rate = sum(self._recent) / self.success_window
                if rate >= self.success_threshold:
                    old = self._scale
                    self._scale = min(1.0, self._scale + self.scale_step)
                    self._recent.clear()
                    self.training_env.env_method("set_scale", self._scale)
                    if self.verbose >= 1:
                        print(
                            f"  [curriculum] step={self.num_timesteps:,}  "
                            f"success_rate={rate:.2f} >= {self.success_threshold:.2f}  "
                            f"scale {old:.3f} → {self._scale:.3f}"
                        )
        return True


class TrainingMetricsCallback(BaseCallback):
    """Collect episode-level metrics and print a summary every print_freq episodes."""

    def __init__(self, algo: str, print_freq: int = 500):
        super().__init__()
        self.algo = algo.lower()
        self.print_freq = int(print_freq)
        if self.algo == "ppo":
            self.loss_keys = ["train/loss", "train/value_loss", "train/policy_gradient_loss"]
        else:
            self.loss_keys = ["train/critic_loss", "train/actor_loss", "train/ent_coef_loss"]
        self.loss_points = {k: [] for k in self.loss_keys}
        self.num_episodes = 0
        self._ep_returns: list = []
        self._ep_lengths: list = []
        self._ep_successes: list = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" not in info:
                continue
            self.num_episodes += 1
            self._ep_returns.append(float(info["episode"]["r"]))
            self._ep_lengths.append(int(info["episode"]["l"]))
            self._ep_successes.append(bool(info.get("hover_region", False)))

            if self.num_episodes % self.print_freq == 0:
                mean_ret = np.mean(self._ep_returns[-self.print_freq:])
                mean_len = np.mean(self._ep_lengths[-self.print_freq:])
                suc_rate = np.mean(self._ep_successes[-self.print_freq:])
                print(
                    f"  ep={self.num_episodes:>7,}  steps={self.num_timesteps:>10,}"
                    f"  mean_ret={mean_ret:+.3f}  mean_len={mean_len:.0f}"
                    f"  success={suc_rate:.2f}"
                )

        values = getattr(self.model.logger, "name_to_value", {})
        episode_anchor = max(1, self.num_episodes)
        for k in self.loss_keys:
            v = values.get(k, None)
            if v is not None and np.isfinite(v):
                self.loss_points[k].append((episode_anchor, float(v)))
        return True


def save_loss_plot(metrics_cb: TrainingMetricsCallback, save_dir: str, algo: str):
    """Save train loss vs episode plot next to model checkpoints."""
    fig = plt.figure(figsize=(10, 6))
    plotted = False
    for key, points in metrics_cb.loss_points.items():
        if len(points) == 0:
            continue
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        plt.plot(x, y, label=key.replace("train/", ""), linewidth=1.3)
        plotted = True

    if not plotted:
        plt.close(fig)
        print(f"No loss metrics captured for {algo.upper()}, skipping loss plot.")
        return

    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title(f"{algo.upper()} Loss vs Episode")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path = os.path.join(save_dir, f"{algo}_loss_vs_episode.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print("Saved loss plot to:", out_path)


def train_quadrotor3d(
    algo="ppo",
    time_steps=1_000_000,
    n_envs=1,
    seed=0,
    exploration_coef=0.02,
    save_subdir="saved_models",
    do_initial_rollout=True,
    initial_rollout_steps=500,
    curriculum_init_scale=0.1,
    curriculum_scale_step=0.05,
    curriculum_success_threshold=0.5,
    curriculum_success_window=20,
):
    """
    Train PPO/SAC on Quadrotor3DEnv.

    Saves to: ./saved_models/<algo>/quadrotor3d_model.zip
    """
    algo = algo.lower().strip()
    if algo not in ["ppo", "sac"]:
        raise ValueError("algo must be 'ppo' or 'sac'")

    def _make_env():
        env = Quadrotor3DEnv(init_scale=curriculum_init_scale)
        env = Monitor(env)
        return env

    env = make_vec_env(_make_env, n_envs=n_envs, seed=seed)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=20.0)
    metrics_cb = TrainingMetricsCallback(algo)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if algo == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), squash_output=True),
            use_sde=True,
            verbose=0,
            device=device,
            seed=seed,
            batch_size=256,
            gamma=0.99,
            learning_rate=3e-4,
            n_steps=2048 // max(1, n_envs),  # keeps total batch roughly stable
            ent_coef=exploration_coef,
        )
    else:
        model = SAC(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=dict(pi=[512, 512], qf=[512, 512])),
            verbose=0,
            device=device,
            seed=seed,
            buffer_size=300_000,
            batch_size=256,
            gamma=0.99,
            learning_rate=3e-4,
            train_freq=1,
            gradient_steps=max(1, n_envs),   # keep update:sample ratio ≈ 1
            ent_coef=exploration_coef,
        )

    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), save_subdir, algo))
    os.makedirs(save_dir, exist_ok=True)

    if do_initial_rollout:
        init_plot_path = os.path.join(save_dir, f"{algo.upper()}_initial_controller_rollout.png")
        init_stats = rollout_and_plot_3d(
            model,
            num_steps=int(initial_rollout_steps),
            seed=seed,
            filename=init_plot_path,
        )
        init_summary_path = os.path.join(save_dir, "initial_controller_summary.txt")
        with open(init_summary_path, "w", encoding="utf-8") as f:
            f.write(f"algo: {algo}\n")
            f.write(f"seed: {seed}\n")
            f.write(f"reached_equilibrium: {init_stats['reached_equilibrium']}\n")
            f.write(f"first_reach_step: {init_stats['first_reach_step']}\n")
            f.write(f"episode_return: {init_stats['episode_return']:.6f}\n")
            f.write(f"steps: {init_stats['steps']}\n")
            f.write(f"best_pos_err: {init_stats['best_pos_err']:.6f}\n")
            f.write(f"best_angle_err: {init_stats['best_angle_err']:.6f}\n")
            f.write(f"best_vel_err: {init_stats['best_vel_err']:.6f}\n")
            f.write(f"best_omega_err: {init_stats['best_omega_err']:.6f}\n")
        print(f"Saved initial controller summary to: {init_summary_path}")

    curriculum_cb = CurriculumCallback(
        init_scale=curriculum_init_scale,
        scale_step=curriculum_scale_step,
        success_threshold=curriculum_success_threshold,
        success_window=curriculum_success_window,
        verbose=0,
    )
    model.learn(total_timesteps=int(time_steps), callback=CallbackList([curriculum_cb, metrics_cb]))

    save_path = os.path.join(save_dir, "quadrotor3d_model")
    model.save(save_path)
    vecnorm_path = os.path.join(save_dir, "vecnormalize.pkl")
    env.save(vecnorm_path)
    print("Saved {} model to: {}.zip".format(algo.upper(), save_path))
    print("Saved VecNormalize stats to:", vecnorm_path)
    save_loss_plot(metrics_cb, save_dir, algo)

    return model


def rollout_and_plot_3d(
    model,
    num_steps=400,
    seed=0,
    filename="quadrotor3d_rollout.png",
    vecnorm_path=None,
):
    """
    Roll out the trained controller in a single env and plot:
      - states x(t) (12 dims)
      - actions u(t) (4 dims)
      - reward(t)

    If vecnorm_path is provided, loads saved VecNormalize stats so observations
    are normalised the same way as during training.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv
    raw_env = Quadrotor3DEnv()
    vec_env = DummyVecEnv([lambda: raw_env])
    if vecnorm_path is not None and os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False   # freeze running stats during evaluation
        vec_env.norm_reward = False
    env = vec_env
    obs = env.reset()

    states = []
    actions = []
    rewards = []
    reached_equilibrium = False
    first_reach_step = None
    min_pos_err = np.inf
    min_angle_err = np.inf
    min_vel_err = np.inf
    min_omega_err = np.inf

    # env is a VecEnv; get dt from the underlying raw env
    raw_dt = raw_env.dt

    for _t in range(int(num_steps)):
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, done, info_list = env.step(action)

        # VecEnv returns shape (n_envs, dim); extract env 0
        raw_obs = raw_env.x_current.detach().cpu().numpy().astype(np.float32)
        raw_obs_err = raw_obs.copy()
        raw_obs_err[0:3] = raw_obs[0:3] - raw_env.obs_equ.numpy()[0:3]

        states.append(raw_obs_err.copy())
        actions.append(np.asarray(action[0], dtype=np.float32).copy())
        rewards.append(float(reward[0]))

        # Error is already in relative coords (obs = error from equilibrium)
        err = np.abs(raw_obs_err)
        min_pos_err   = min(min_pos_err,   float(np.max(err[0:3])))
        min_angle_err = min(min_angle_err, float(np.max(err[3:6])))
        min_vel_err   = min(min_vel_err,   float(np.max(err[6:9])))
        min_omega_err = min(min_omega_err, float(np.max(err[9:12])))

        info = info_list[0] if isinstance(info_list, list) else info_list
        if (not reached_equilibrium) and bool(info.get("hover_region", False)):
            reached_equilibrium = True
            first_reach_step = _t + 1

        obs = next_obs
        if done[0]:
            break

    states = np.asarray(states, dtype=np.float32)    # [T,12]
    actions = np.asarray(actions, dtype=np.float32)  # [T,4]
    rewards = np.asarray(rewards, dtype=np.float32)  # [T]

    T = int(states.shape[0])
    t_axis = np.arange(T, dtype=np.float32) * float(raw_dt)

    # Helpful labels for 12D state
    state_labels = [
        "x", "y", "z",
        "roll", "pitch", "yaw",
        "vx", "vy", "vz",
        "wx", "wy", "wz",
    ]
    action_labels = ["u1", "u2", "u3", "u4"]

    fig = plt.figure(figsize=(14, 12))

    # ---- states ----
    ax1 = plt.subplot(3, 1, 1)
    for i in range(states.shape[1]):
        lbl = state_labels[i] if i < len(state_labels) else "x[{}]".format(i)
        ax1.plot(t_axis, states[:, i], label=lbl)
    ax1.set_title("Quadrotor3D State Trajectory (12D)")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("state")
    ax1.legend(ncol=4, fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ---- actions ----
    ax2 = plt.subplot(3, 1, 2)
    for i in range(actions.shape[1]):
        lbl = action_labels[i] if i < len(action_labels) else "u[{}]".format(i)
        ax2.plot(t_axis, actions[:, i], label=lbl)
    ax2.set_title("Control Inputs (4 thrusts)")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("thrust")
    ax2.legend(ncol=4, fontsize=10)
    ax2.grid(True, alpha=0.3)

    # ---- reward ----
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(t_axis, rewards, label="reward")
    ax3.set_title("Reward")
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("reward")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print("Saved rollout plot to:", filename)
    if reached_equilibrium:
        print(f"Episode reached equilibrium neighborhood at step {first_reach_step}")
    else:
        print("Episode did NOT reach equilibrium neighborhood.")
    print(
        "Best max-abs errors over rollout: "
        f"pos={min_pos_err:.4f}, angle={min_angle_err:.4f}, "
        f"vel={min_vel_err:.4f}, omega={min_omega_err:.4f}"
    )
    return {
        "reached_equilibrium": bool(reached_equilibrium),
        "first_reach_step": first_reach_step,
        "episode_return": float(np.sum(rewards)),
        "steps": int(T),
        "best_pos_err": float(min_pos_err),
        "best_angle_err": float(min_angle_err),
        "best_vel_err": float(min_vel_err),
        "best_omega_err": float(min_omega_err),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO/SAC on Quadrotor3D from random initialization")
    p.add_argument("--algos", type=str, default="sac", help="Comma-separated list, e.g. 'ppo,sac' or 'sac'")
    p.add_argument("--time_steps_ppo", type=int, default=5000000)
    p.add_argument("--time_steps_sac", type=int, default=300000000)
    p.add_argument("--n_envs", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exploration_coef", type=float, default=0.02)
    p.add_argument("--rollout_count", type=int, default=5)
    p.add_argument("--rollout_steps", type=int, default=500)
    p.add_argument(
        "--save_subdir",
        type=str,
        default="saved_models",
        help="Relative folder under quadrotor3d/ where trained RL controllers are saved",
    )
    p.add_argument(
        "--do_initial_rollout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run and save a rollout before training",
    )
    p.add_argument("--initial_rollout_steps", type=int, default=500)
    p.add_argument("--curriculum_init_scale",         type=float, default=0.1)
    p.add_argument("--curriculum_scale_step",         type=float, default=0.05)
    p.add_argument("--curriculum_success_threshold",  type=float, default=0.5)
    p.add_argument("--curriculum_success_window",     type=int,   default=20)
    return p.parse_args()


def main():
    args = parse_args()
    algos = [a.strip().lower() for a in args.algos.split(",") if a.strip()]
    for a in algos:
        if a not in {"ppo", "sac"}:
            raise ValueError(f"Unsupported algo in --algos: {a}")
    for algo in algos:
        time_steps = args.time_steps_ppo if algo == "ppo" else args.time_steps_sac
        print(f"\n===== Training {algo.upper()} =====")

        # ---- TRAIN ----
        model = train_quadrotor3d(
            algo=algo,
            time_steps=time_steps,
            n_envs=args.n_envs,
            seed=args.seed,
            exploration_coef=args.exploration_coef,
            save_subdir=args.save_subdir,
            do_initial_rollout=args.do_initial_rollout,
            initial_rollout_steps=args.initial_rollout_steps,
            curriculum_init_scale=args.curriculum_init_scale,
            curriculum_scale_step=args.curriculum_scale_step,
            curriculum_success_threshold=args.curriculum_success_threshold,
            curriculum_success_window=args.curriculum_success_window,
        )

        # ---- MULTI-ROLLOUT EVALUATION ----
        rollout_count = args.rollout_count
        rollout_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), args.save_subdir, algo))
        vecnorm_path = os.path.join(rollout_dir, "vecnormalize.pkl")
        os.makedirs(rollout_dir, exist_ok=True)
        rollout_stats = []

        for ep in range(rollout_count):
            stats = rollout_and_plot_3d(
                model,
                num_steps=args.rollout_steps,
                seed=args.seed + ep,
                filename=os.path.join(
                    rollout_dir, "{}_quadrotor3d_rollout_ep{:02d}.png".format(algo.upper(), ep + 1)
                ),
                vecnorm_path=vecnorm_path,
            )
            rollout_stats.append(stats)

        success_rate = np.mean([s["reached_equilibrium"] for s in rollout_stats])
        mean_return = np.mean([s["episode_return"] for s in rollout_stats])
        summary_path = os.path.join(rollout_dir, "{}_rollout_summary.txt".format(algo))
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("rollout_count: {}\n".format(rollout_count))
            f.write("success_rate: {:.4f}\n".format(success_rate))
            f.write("mean_return: {:.6f}\n".format(mean_return))
            for i, s in enumerate(rollout_stats, start=1):
                f.write(
                    "episode {:02d}: reached_equilibrium={} first_reach_step={} return={:.6f} steps={}\n".format(
                        i,
                        s["reached_equilibrium"],
                        s["first_reach_step"],
                        s["episode_return"],
                        s["steps"],
                    )
                )
        print("Saved rollout summary to:", summary_path)


if __name__ == "__main__":
    main()
