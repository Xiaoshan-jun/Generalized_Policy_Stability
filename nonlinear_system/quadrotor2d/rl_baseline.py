#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train PPO or SAC on Quadrotor2DEnv (Gymnasium API) and save the model.

This replaces Pendulum-specific evaluation/plotting with a quadrotor-friendly rollout plot.
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

import gymnasium as gym

# ---- CHANGE THIS IMPORT to match your file name ----
# from quadrotor2d_env import Quadrotor2DEnv
from quadrotor2d_env import Quadrotor2DEnv  # <-- edit module name if needed


def train_quadrotor(algo="ppo", time_steps=1_000_000, n_envs=1, seed=0, max_time=2.0, dt=0.01):
    """
    Train PPO/SAC on Quadrotor2DEnv.

    Saves to: ../saved_models/<algo>/quadrotor_model.zip
    """
    algo = algo.lower().strip()
    if algo not in ["ppo", "sac"]:
        raise ValueError("algo must be 'ppo' or 'sac'")

    def _make_env():
        env = Quadrotor2DEnv(dt=dt, max_time=max_time)  # already truncated internally
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
        )
        model.learn(total_timesteps=int(time_steps))
    else:
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            device=device,
            seed=seed,
            buffer_size=200_000,
            batch_size=256,
            gamma=0.99,
            learning_rate=3e-4,
            train_freq=1,
            gradient_steps=1,
        )
        model.learn(total_timesteps=int(time_steps))

    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "saved_models", algo))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "quadrotor_model")
    model.save(save_path)
    print(f"Saved {algo.upper()} model to: {save_path}.zip")

    return model


def rollout_and_plot(model, num_steps=300, seed=0, max_time=2.0, dt=0.01, filename="quadrotor_rollout.png"):
    """
    Roll out the trained controller in a single env and plot:
      - states x(t) (6 dims)
      - actions u(t) (2 dims)
      - reward(t)

    This replaces the pendulum (theta/omega) plots.
    """
    env = Quadrotor2DEnv(dt=dt, max_time=max_time)
    obs, info = env.reset(seed=seed)

    states = []
    actions = []
    rewards = []

    for t in range(num_steps):
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, terminated, truncated, info = env.step(action)

        states.append(obs.copy())
        actions.append(np.array(action).copy())
        rewards.append(reward)

        obs = next_obs
        if terminated or truncated:
            break

    states = np.asarray(states, dtype=np.float32)   # [T,6]
    actions = np.asarray(actions, dtype=np.float32) # [T,2]
    rewards = np.asarray(rewards, dtype=np.float32) # [T]

    T = states.shape[0]
    t_axis = np.arange(T) * dt

    fig = plt.figure(figsize=(14, 10))

    ax1 = plt.subplot(3, 1, 1)
    for i in range(states.shape[1]):
        ax1.plot(t_axis, states[:, i], label=f"x[{i}]")
    ax1.set_title("Quadrotor2D State Trajectory")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("state")
    ax1.legend(ncol=3, fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(3, 1, 2)
    for i in range(actions.shape[1]):
        ax2.plot(t_axis, actions[:, i], label=f"u[{i}]")
    ax2.set_title("Control Inputs")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("action")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(t_axis, rewards, label="reward")
    ax3.set_title("Reward")
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("reward")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved rollout plot to: {filename}")


def main():
    # ---- CONFIG ----
    algo = "sac"          # "ppo" or "sac"
    time_steps = 1_000_000 if algo == "ppo" else 500_000
    n_envs = 1
    seed = 0

    # env horizon
    dt = 0.01
    max_time = 2.0        # truncated after max_time seconds (built into env)

    # ---- TRAIN ----
    model = train_quadrotor(
        algo=algo,
        time_steps=time_steps,
        n_envs=n_envs,
        seed=seed,
        max_time=max_time,
        dt=dt,
    )

    # ---- QUICK ROLLOUT PLOT ----
    rollout_and_plot(
        model,
        num_steps=400,
        seed=seed,
        max_time=max_time,
        dt=dt,
        filename=f"{algo.upper()}_quadrotor_rollout.png",
    )


if __name__ == "__main__":
    main()
