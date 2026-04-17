#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate imitation_policy_lqr_single_best.pth in Quadrotor3DEnv rollouts."""

import argparse
import os
import numpy as np
import torch

from quadrotor3d_env import Quadrotor3DEnv
from network.PolicyNet import PolicyNet


def parse_args():
    p = argparse.ArgumentParser(description="Test imitation LQR-single policy in real simulation")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="saved_models/imitation_lqr_single/imitation_policy_lqr_single_best.pth",
        help="Path to imitation policy checkpoint",
    )
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
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


def build_policy_from_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if "state_dict" not in ckpt:
        raise ValueError(f"Invalid checkpoint, missing state_dict: {ckpt_path}")
    if "x_mean" not in ckpt or "x_std" not in ckpt:
        raise ValueError(f"Invalid checkpoint, missing x_mean/x_std: {ckpt_path}")

    state = ckpt["state_dict"]
    hidden_dim = int(state["layers.0.weight"].shape[0])
    input_dim = int(state["layers.0.weight"].shape[1])
    output_dim = int(state["output_layer.weight"].shape[0])
    layer_ids = {
        int(k.split(".")[1])
        for k in state.keys()
        if k.startswith("layers.") and k.endswith(".weight")
    }
    n_layers = max(layer_ids) + 1 if layer_ids else 1

    model = PolicyNet(input_dim, hidden_dim, output_dim, n_layers=n_layers).to(device)
    model.load_state_dict(state)
    model.eval()

    x_mean = torch.tensor(np.asarray(ckpt["x_mean"], dtype=np.float32), device=device)
    x_std = torch.tensor(np.asarray(ckpt["x_std"], dtype=np.float32), device=device)
    return model, x_mean, x_std


def run_episode(env, model, x_mean, x_std, cfg, seed, device):
    obs, _ = env.reset(seed=seed)

    rewards = []
    reached = False
    reach_step = None
    t = 0
    while True:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        obs_n = (obs_t - x_mean) / (x_std + 1e-6)
        with torch.no_grad():
            action = model(obs_n).detach().cpu().numpy().astype(np.float32)

        next_obs, reward, terminated, truncated, info = env.step(action)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = resolve_path(cfg.checkpoint)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model, x_mean, x_std = build_policy_from_checkpoint(ckpt_path, device)
    env = Quadrotor3DEnv(seed=cfg.seed)

    stats = []
    for ep in range(cfg.episodes):
        out = run_episode(env, model, x_mean, x_std, cfg, seed=cfg.seed + ep, device=device)
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

    print("\n=== Imitation Policy (LQR Single) Test Summary ===")
    print(f"checkpoint: {ckpt_path}")
    print(f"episodes: {cfg.episodes}")
    print(f"mean_return: {returns.mean():.6f}")
    print(f"std_return: {returns.std():.6f}")
    print(f"success_rate: {success.mean():.6f}")
    print(f"mean_reach_step: {mean_reach:.6f}")


if __name__ == "__main__":
    main()
