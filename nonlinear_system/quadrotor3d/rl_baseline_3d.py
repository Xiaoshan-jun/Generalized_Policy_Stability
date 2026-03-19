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
import numpy as np
import torch

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gymnasium as gym  # noqa: F401

# ---- CHANGE THIS IMPORT to match your file name ----
# from quadrotor3d_env import Quadrotor3DEnv
from quadrotor3d_env import Quadrotor3DEnv  # <-- edit module name if needed


def train_quadrotor3d(
    algo="ppo",
    time_steps=1_000_000,
    n_envs=1,
    seed=0,
    max_time=2.0,
    dt=0.01,
):
    """
    Train PPO/SAC on Quadrotor3DEnv.

    Saves to: ./saved_models/<algo>/quadrotor3d_model.zip
    """
    algo = algo.lower().strip()
    if algo not in ["ppo", "sac"]:
        raise ValueError("algo must be 'ppo' or 'sac'")

    def _make_env():
        env = Quadrotor3DEnv(dt=dt, max_time=max_time)  # truncated internally
        env = Monitor(env)
        return env

    env = make_vec_env(_make_env, n_envs=n_envs, seed=seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if algo == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device=device,
            seed=seed,
            batch_size=256,
            gamma=0.99,
            learning_rate=3e-4,
            n_steps=2048 // max(1, n_envs),  # keeps total batch roughly stable
        )
        model.learn(total_timesteps=int(time_steps))
    else:
        model = SAC(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[512, 512])),
            verbose=1,
            device=device,
            seed=seed,
            buffer_size=300_000,
            batch_size=256,
            gamma=0.99,
            learning_rate=3e-4,
            train_freq=1,
            gradient_steps=1,
        )
        model.learn(total_timesteps=int(time_steps))

    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "saved_models", algo))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "quadrotor3d_model")
    model.save(save_path)
    print("Saved {} model to: {}.zip".format(algo.upper(), save_path))

    return model


def rollout_and_plot_3d(
    model,
    num_steps=400,
    seed=0,
    max_time=2.0,
    dt=0.01,
    filename="quadrotor3d_rollout.png",
):
    """
    Roll out the trained controller in a single env and plot:
      - states x(t) (12 dims)
      - actions u(t) (4 dims)
      - reward(t)
    """
    env = Quadrotor3DEnv(dt=dt, max_time=max_time)
    obs, info = env.reset(seed=seed)

    states = []
    actions = []
    rewards = []

    for _t in range(int(num_steps)):
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, terminated, truncated, info = env.step(action)

        states.append(np.asarray(obs, dtype=np.float32).copy())
        actions.append(np.asarray(action, dtype=np.float32).copy())
        rewards.append(float(reward))

        obs = next_obs
        if terminated or truncated:
            break

    states = np.asarray(states, dtype=np.float32)    # [T,12]
    actions = np.asarray(actions, dtype=np.float32)  # [T,4]
    rewards = np.asarray(rewards, dtype=np.float32)  # [T]

    T = int(states.shape[0])
    t_axis = np.arange(T, dtype=np.float32) * float(dt)

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


def main():
    # ---- CONFIG ----
    algo = "sac"  # "ppo" or "sac"
    time_steps = 1_000_000 if algo == "ppo" else 600_000
    n_envs = 1
    seed = 0

    # env horizon
    dt = 0.01
    max_time = 2.0

    # ---- TRAIN ----
    model = train_quadrotor3d(
        algo=algo,
        time_steps=time_steps,
        n_envs=n_envs,
        seed=seed,
        max_time=max_time,
        dt=dt,
    )

    # ---- QUICK ROLLOUT PLOT ----
    rollout_and_plot_3d(
        model,
        num_steps=400,
        seed=seed,
        max_time=max_time,
        dt=dt,
        filename="{}_quadrotor3d_rollout.png".format(algo.upper()),
    )


if __name__ == "__main__":
    main()