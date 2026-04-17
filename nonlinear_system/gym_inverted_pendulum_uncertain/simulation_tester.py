#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simulation tester for trained RL controllers in uncertain pendulum dynamics."""

import argparse
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC

from uncertain_env import UncertainDisturbancePendulumEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained controller in uncertain dynamics")
    parser.add_argument(
        "--algo", type=str, default="sac", choices=["ppo", "sac"], help="Controller type"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=(
            "nonlinear_system/gym_inverted_pendulum_uncertain/saved_models/sac/"
            "sac_pendulum_uncertain_uncertain_d0p5.zip"
        ),
        help="Path to trained controller (.zip for SB3 models)",
    )
    parser.add_argument(
        "--episodes", type=int, default=20, help="Number of episodes to simulate"
    )
    parser.add_argument(
        "--steps_per_episode", type=int, default=300, help="Max steps per episode"
    )
    parser.add_argument(
        "--disturbance_max", type=float, default=0.5, help="Disturbance bound for test env"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="nonlinear_system/gym_inverted_pendulum_uncertain/results/simulation_tester",
        help="Directory to save simulation plots/results",
    )
    return parser.parse_args()


def load_model(algo: str, model_path: str):
    if algo == "ppo":
        return PPO.load(model_path)
    return SAC.load(model_path)


def angle_from_obs(obs):
    return float(np.arctan2(obs[1], obs[0]))


def run_episode(model, env, steps_per_episode):
    obs, _ = env.reset()
    rewards = []
    thetas = []
    omegas = []
    actions = []
    disturbances = []

    for _ in range(steps_per_episode):
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(float(reward))
        thetas.append(angle_from_obs(obs))
        omegas.append(float(obs[2]))
        actions.append(float(action[0] if isinstance(action, np.ndarray) else action))
        disturbances.append(float(info.get("disturbance", 0.0)))

        obs = next_obs
        if terminated or truncated:
            break

    final_theta = angle_from_obs(obs)
    final_omega = float(obs[2])
    episode_return = float(np.sum(rewards))
    return {
        "return": episode_return,
        "final_theta": final_theta,
        "final_omega": final_omega,
        "mean_abs_theta": float(np.mean(np.abs(thetas))) if thetas else np.nan,
        "max_abs_theta": float(np.max(np.abs(thetas))) if thetas else np.nan,
        "thetas": thetas,
        "omegas": omegas,
        "actions": actions,
        "disturbances": disturbances,
        "rewards": rewards,
    }


def save_plots(first_episode, returns, output_dir):
    t = np.arange(len(first_episode["thetas"]))

    plt.figure(figsize=(12, 9))
    plt.subplot(4, 1, 1)
    plt.plot(t, first_episode["thetas"], linewidth=1.5)
    plt.ylabel("theta (rad)")
    plt.grid(alpha=0.3)

    plt.subplot(4, 1, 2)
    plt.plot(t, first_episode["omegas"], linewidth=1.5)
    plt.ylabel("omega (rad/s)")
    plt.grid(alpha=0.3)

    plt.subplot(4, 1, 3)
    plt.plot(t, first_episode["actions"], linewidth=1.5)
    plt.ylabel("action")
    plt.grid(alpha=0.3)

    plt.subplot(4, 1, 4)
    plt.plot(t, first_episode["disturbances"], linewidth=1.5)
    plt.ylabel("disturbance")
    plt.xlabel("step")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "episode1_rollout.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(returns, bins=min(20, len(returns)), alpha=0.8)
    plt.xlabel("Episode return")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "returns_hist.png"), dpi=200)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    model = load_model(args.algo, args.model_path)
    env = UncertainDisturbancePendulumEnv(
        disturbance_max=args.disturbance_max, g=9.81, m=1.0, l=1.0, b=0.13
    )

    episode_stats = []
    for ep in range(args.episodes):
        stats = run_episode(model, env, args.steps_per_episode)
        episode_stats.append(stats)
        print(
            f"[episode {ep + 1:03d}] return={stats['return']:.3f} "
            f"final_theta={stats['final_theta']:.3f} final_omega={stats['final_omega']:.3f}"
        )

    returns = np.array([s["return"] for s in episode_stats], dtype=np.float64)
    final_thetas = np.array([s["final_theta"] for s in episode_stats], dtype=np.float64)
    final_omegas = np.array([s["final_omega"] for s in episode_stats], dtype=np.float64)
    mean_abs_thetas = np.array([s["mean_abs_theta"] for s in episode_stats], dtype=np.float64)

    success = (np.abs(final_thetas) < 0.2) & (np.abs(final_omegas) < 0.5)
    success_rate = float(np.mean(success))

    summary = (
        f"model_path: {args.model_path}\n"
        f"algo: {args.algo}\n"
        f"episodes: {args.episodes}\n"
        f"steps_per_episode: {args.steps_per_episode}\n"
        f"disturbance_max: {args.disturbance_max}\n"
        f"mean_return: {returns.mean():.6f}\n"
        f"std_return: {returns.std():.6f}\n"
        f"best_return: {returns.max():.6f}\n"
        f"worst_return: {returns.min():.6f}\n"
        f"mean_final_abs_theta: {np.mean(np.abs(final_thetas)):.6f}\n"
        f"mean_final_abs_omega: {np.mean(np.abs(final_omegas)):.6f}\n"
        f"mean_abs_theta_over_time: {np.mean(mean_abs_thetas):.6f}\n"
        f"success_rate(|theta|<0.2, |omega|<0.5): {success_rate:.6f}\n"
    )

    print("\n===== Simulation Summary =====")
    print(summary.strip())

    with open(os.path.join(args.output_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)

    save_plots(episode_stats[0], returns, args.output_dir)
    print(f"Saved results to: {args.output_dir}")


if __name__ == "__main__":
    main()

