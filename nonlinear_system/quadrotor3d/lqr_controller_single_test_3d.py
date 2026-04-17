#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small test runner for single-gain LQR on Quadrotor3D."""

import argparse
import os
import numpy as np

from quadrotor3d_env import Quadrotor3DEnv
from quadrotor3d import Quadrotor


def parse_args():
    p = argparse.ArgumentParser(description="Quick test for single-gain LQR controller")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--k_path",
        type=str,
        default="saved_models/lqr_single/K_discrete_dt_0.01.npy",
        help="Path to single discrete LQR gain K with shape [4, 12]",
    )
    return p.parse_args()


def resolve_path(path: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    candidates = [path, os.path.join(here, path), os.path.join(repo, path)]
    for c in candidates:
        c_abs = os.path.abspath(c)
        if os.path.exists(c_abs):
            return c_abs
    return os.path.abspath(path)


def run_episode(env, u_eq, Kd, cfg, seed):
    obs, _ = env.reset(seed=seed)

    rewards = []
    reached = False
    reach_step = None
    t = 0
    while True:
        u = u_eq - Kd @ obs
        u = np.clip(u, env.action_space.low, env.action_space.high)
        next_obs, reward, terminated, truncated, info = env.step(u)
        rewards.append(float(reward))

        if (not reached) and bool(info.get("hover_region", False)):
            reached = True
            reach_step = t + 1

        obs = next_obs
        if terminated or truncated:
            break
        t += 1

    return {
        "return": float(np.sum(rewards)),
        "steps": int(len(rewards)),
        "reached": reached,
        "reach_step": reach_step,
    }


def main():
    cfg = parse_args()

    k_path = resolve_path(cfg.k_path)
    if not os.path.exists(k_path):
        raise FileNotFoundError(f"K not found: {k_path}")
    Kd = np.load(k_path)
    if Kd.shape != (4, 12):
        raise ValueError(f"K must have shape [4,12], got {Kd.shape}")

    quad = Quadrotor(dtype=None)
    u_eq = np.ones(4, dtype=np.float32) * float(quad.hover_thrust)

    env = Quadrotor3DEnv(seed=cfg.seed)

    stats = []
    for ep in range(cfg.episodes):
        out = run_episode(env, u_eq, Kd, cfg, seed=cfg.seed + ep)
        stats.append(out)
        status = "REACHED" if out["reached"] else "NOT_REACHED"
        print(
            f"[ep {ep + 1:03d}] return={out['return']:.6f} steps={out['steps']} "
            f"{status} reach_step={out['reach_step']}"
        )

    returns = np.array([s["return"] for s in stats], dtype=np.float64)
    success = np.array([s["reached"] for s in stats], dtype=np.float64)
    steps_to_reach = [s["reach_step"] for s in stats if s["reach_step"] is not None]
    mean_reach = float(np.mean(steps_to_reach)) if len(steps_to_reach) else np.nan

    print("\n=== Single-Gain LQR Test Summary ===")
    print(f"k_path: {k_path}")
    print(f"episodes: {cfg.episodes}")
    print(f"mean_return: {returns.mean():.6f}")
    print(f"std_return: {returns.std():.6f}")
    print(f"success_rate: {success.mean():.6f}")
    print(f"mean_reach_step: {mean_reach:.6f}")


if __name__ == "__main__":
    main()
