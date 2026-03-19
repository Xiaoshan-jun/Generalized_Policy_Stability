#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_test_quadrotor3d.py

Full evaluation script for a trained SB3 PPO/SAC model on Quadrotor3DEnv.

What it does:
  1) Loads a saved SB3 model (PPO or SAC)
  2) Runs deterministic evaluation for N episodes:
       - ep_return, ep_length
       - success rate (ending near equilibrium)
       - optional safety-termination rate
  3) Runs a local perturbation (hover recovery) test
  4) Generates rollout plots for a few episodes:
       - 12D state curves
       - 4 thrust curves
       - reward curve
       - state norm curve
  5) (Optional) ROA-like random initial state sweep and summarizes outcomes

Python: 3.8 compatible.
"""

import os
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, SAC

# ---- CHANGE THIS IMPORT to match your env file name ----
from quadrotor3d_env import Quadrotor3DEnv  # noqa: F401


# ---------------------------
# Utilities
# ---------------------------
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def state_labels_12d() -> List[str]:
    return [
        "x", "y", "z",
        "roll", "pitch", "yaw",
        "vx", "vy", "vz",
        "wx", "wy", "wz",
    ]


def action_labels_4d() -> List[str]:
    return ["u1", "u2", "u3", "u4"]


def safe_norm(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        return float("inf")
    return float(np.linalg.norm(x))


def load_model(algo: str, model_path: str, device: str = "auto"):
    algo = algo.lower().strip()
    if algo == "ppo":
        return PPO.load(model_path, device=device)
    if algo == "sac":
        return SAC.load(model_path, device=device)
    raise ValueError("algo must be 'ppo' or 'sac'")


# ---------------------------
# Rollout
# ---------------------------
def rollout_episode(
    env: Quadrotor3DEnv,
    model,
    seed: Optional[int] = None,
    max_steps: Optional[int] = None,
    deterministic: bool = True,
    set_initial_state: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Runs one episode rollout. If set_initial_state is provided and env exposes
    env.unwrapped.x_current, it force-sets the state before stepping.

    Returns dict with arrays:
      obs: [T+1, 12]
      act: [T, 4]
      rew: [T]
      done: scalar
    """
    obs, _ = env.reset(seed=seed)

    if set_initial_state is not None:
        # best-effort set internal state
        try:
            u = env.unwrapped if hasattr(env, "unwrapped") else env
            if hasattr(u, "x_current"):
                u.x_current = torch.tensor(set_initial_state, dtype=torch.float32)
                obs = np.asarray(set_initial_state, dtype=np.float32).copy()
        except Exception:
            pass

    if max_steps is None:
        # For Gymnasium envs with truncation horizon, this is often enough
        max_steps = getattr(env, "max_steps", 500)

    obs_traj = [np.asarray(obs, dtype=np.float32).copy()]
    act_traj = []
    rew_traj = []

    done = False
    for _t in range(int(max_steps)):
        act, _ = model.predict(obs, deterministic=deterministic)
        next_obs, rew, terminated, truncated, _ = env.step(act)

        act_traj.append(np.asarray(act, dtype=np.float32).copy())
        rew_traj.append(float(rew))
        obs_traj.append(np.asarray(next_obs, dtype=np.float32).copy())

        obs = next_obs
        done = bool(terminated) or bool(truncated)
        if done:
            break

    return {
        "obs": np.asarray(obs_traj, dtype=np.float32),   # [T+1,12]
        "act": np.asarray(act_traj, dtype=np.float32),   # [T,4]
        "rew": np.asarray(rew_traj, dtype=np.float32),   # [T]
        "done": np.asarray(done, dtype=np.bool_),
    }


# ---------------------------
# Plotting
# ---------------------------
def plot_rollout(
    out_path: str,
    rollout: Dict[str, np.ndarray],
    dt: float,
    title_prefix: str = "Quadrotor3D",
):
    obs = rollout["obs"]  # [T+1,12]
    act = rollout["act"]  # [T,4]
    rew = rollout["rew"]  # [T]

    T = act.shape[0]
    t_axis = np.arange(T, dtype=np.float32) * float(dt)

    # Compute norms for debugging
    obs_norm = np.linalg.norm(obs[:T], axis=1)

    fig = plt.figure(figsize=(14, 14))

    # ---- states ----
    ax1 = plt.subplot(4, 1, 1)
    labels = state_labels_12d()
    for i in range(obs.shape[1]):
        lbl = labels[i] if i < len(labels) else "x[{}]".format(i)
        ax1.plot(t_axis, obs[:T, i], label=lbl)
    ax1.set_title("{}: State Trajectory (12D)".format(title_prefix))
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("state")
    ax1.legend(ncol=4, fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ---- actions ----
    ax2 = plt.subplot(4, 1, 2)
    a_labels = action_labels_4d()
    for i in range(act.shape[1]):
        lbl = a_labels[i] if i < len(a_labels) else "u[{}]".format(i)
        ax2.plot(t_axis, act[:, i], label=lbl)
    ax2.set_title("{}: Control Inputs (4 thrusts)".format(title_prefix))
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("thrust")
    ax2.legend(ncol=4, fontsize=10)
    ax2.grid(True, alpha=0.3)

    # ---- reward ----
    ax3 = plt.subplot(4, 1, 3)
    ax3.plot(t_axis, rew, label="reward")
    ax3.set_title("{}: Reward".format(title_prefix))
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("reward")
    ax3.grid(True, alpha=0.3)

    # ---- state norm ----
    ax4 = plt.subplot(4, 1, 4)
    ax4.plot(t_axis, obs_norm, label="||x||")
    ax4.set_title("{}: State Norm".format(title_prefix))
    ax4.set_xlabel("time (s)")
    ax4.set_ylabel("norm")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------------
# Metrics / Tests
# ---------------------------
def evaluate_episodes(
    env: Quadrotor3DEnv,
    model,
    n_episodes: int,
    seed: int,
    dt: float,
    max_steps: Optional[int],
    success_radius: float,
    deterministic: bool = True,
) -> Dict[str, float]:
    returns = []
    lengths = []
    final_norms = []
    finite_episodes = 0
    success = 0

    for ep in range(int(n_episodes)):
        r = rollout_episode(
            env=env,
            model=model,
            seed=seed + ep,
            max_steps=max_steps,
            deterministic=deterministic,
            set_initial_state=None,
        )
        obs = r["obs"]
        rew = r["rew"]
        T = rew.shape[0]

        ep_ret = float(np.sum(rew))
        fin = obs[-1]
        fin_norm = safe_norm(fin)

        returns.append(ep_ret)
        lengths.append(float(T))
        final_norms.append(fin_norm)

        if np.isfinite(fin_norm):
            finite_episodes += 1
            if fin_norm <= success_radius:
                success += 1

    ret_mean = float(np.mean(returns)) if len(returns) else float("nan")
    ret_std = float(np.std(returns)) if len(returns) else float("nan")
    len_mean = float(np.mean(lengths)) if len(lengths) else float("nan")
    fin_mean = float(np.mean(final_norms)) if len(final_norms) else float("nan")

    succ_rate = float(success) / float(finite_episodes) if finite_episodes > 0 else 0.0

    return {
        "ep_rew_mean": ret_mean,
        "ep_rew_std": ret_std,
        "ep_len_mean": len_mean,
        "final_norm_mean": fin_mean,
        "success_rate": succ_rate,
        "n_episodes": float(n_episodes),
    }


def hover_recovery_test(
    env: Quadrotor3DEnv,
    model,
    n_trials: int,
    seed: int,
    perturb_radius: float,
    max_steps: Optional[int],
    success_radius: float,
    deterministic: bool = True,
) -> Dict[str, float]:
    """
    Sample near equilibrium and see if it recovers to within success_radius.
    """
    # Equilibrium from env if available
    if hasattr(env, "obs_equ"):
        eq = env.obs_equ.detach().cpu().numpy().astype(np.float32) if torch.is_tensor(env.obs_equ) else np.asarray(env.obs_equ, dtype=np.float32)
    else:
        eq = np.zeros((env.observation_space.shape[0],), dtype=np.float32)

    successes = 0
    finite = 0

    rng = np.random.default_rng(seed)
    for k in range(int(n_trials)):
        delta = rng.uniform(-perturb_radius, perturb_radius, size=eq.shape).astype(np.float32)
        x0 = (eq + delta).astype(np.float32)

        r = rollout_episode(
            env=env,
            model=model,
            seed=seed + 10000 + k,
            max_steps=max_steps,
            deterministic=deterministic,
            set_initial_state=x0,
        )
        fin = r["obs"][-1]
        fin_norm = safe_norm(fin)
        if np.isfinite(fin_norm):
            finite += 1
            if fin_norm <= success_radius:
                successes += 1

    rate = float(successes) / float(finite) if finite > 0 else 0.0
    return {
        "hover_trials": float(n_trials),
        "hover_success_rate": rate,
        "hover_successes": float(successes),
        "hover_finite": float(finite),
    }


def roa_sweep_test(
    env: Quadrotor3DEnv,
    model,
    n_samples: int,
    seed: int,
    max_steps: Optional[int],
    success_radius: float,
    deterministic: bool = True,
) -> Dict[str, float]:
    """
    Sample random initial states uniformly from observation_space and evaluate success.
    This is a crude proxy for region-of-attraction (ROA) coverage.
    """
    low = np.asarray(env.observation_space.low, dtype=np.float32)
    high = np.asarray(env.observation_space.high, dtype=np.float32)
    low_f = np.where(np.isfinite(low), low, -1.0)
    high_f = np.where(np.isfinite(high), high, 1.0)

    rng = np.random.default_rng(seed)
    successes = 0
    finite = 0
    avg_return = 0.0

    for i in range(int(n_samples)):
        x0 = rng.uniform(low_f, high_f).astype(np.float32)

        r = rollout_episode(
            env=env,
            model=model,
            seed=seed + 20000 + i,
            max_steps=max_steps,
            deterministic=deterministic,
            set_initial_state=x0,
        )

        ep_ret = float(np.sum(r["rew"]))
        fin = r["obs"][-1]
        fin_norm = safe_norm(fin)

        avg_return += ep_ret

        if np.isfinite(fin_norm):
            finite += 1
            if fin_norm <= success_radius:
                successes += 1

    avg_return = avg_return / float(max(1, n_samples))
    rate = float(successes) / float(finite) if finite > 0 else 0.0
    return {
        "roa_samples": float(n_samples),
        "roa_success_rate": rate,
        "roa_avg_return": float(avg_return),
        "roa_successes": float(successes),
        "roa_finite": float(finite),
    }


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="sac", choices=["sac", "ppo"])
    parser.add_argument("--model_path", type=str, default="", help="Path to SB3 .zip (without or with .zip)")
    parser.add_argument("--out_dir", type=str, default="./eval_out_quadrotor3d")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_eval_episodes", type=int, default=50)
    parser.add_argument("--n_plot_episodes", type=int, default=3)

    # Env params (should match training!)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--max_time", type=float, default=2.0)

    # Evaluation params
    parser.add_argument("--max_steps", type=int, default=0, help="0 = use env max_steps")
    parser.add_argument("--success_radius", type=float, default=0.5, help="Success if ||x_T|| <= radius")
    parser.add_argument("--hover_trials", type=int, default=50)
    parser.add_argument("--hover_perturb_radius", type=float, default=0.1)
    parser.add_argument("--roa_samples", type=int, default=200)

    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)

    # normalize model path
    if args.model_path == "":
        model_path = os.path.join(
            os.path.dirname(__file__),
            "saved_models",
            args.algo,
            "quadrotor3d_model",
        )
    else:
        model_path = args.model_path

    # remove .zip if provided
    if model_path.endswith(".zip"):
        model_path = model_path[:-4]

    # Create env
    env = Quadrotor3DEnv(dt=float(args.dt), max_time=float(args.max_time))
    dt = float(args.dt)
    max_steps = None if int(args.max_steps) <= 0 else int(args.max_steps)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.algo, model_path, device=device)
    print("Loaded {} model from: {}.zip (device={})".format(args.algo.upper(), model_path, device))

    # ---- Eval episodes ----
    metrics = evaluate_episodes(
        env=env,
        model=model,
        n_episodes=int(args.n_eval_episodes),
        seed=int(args.seed),
        dt=dt,
        max_steps=max_steps,
        success_radius=float(args.success_radius),
        deterministic=True,
    )

    # ---- Hover recovery ----
    hover_metrics = hover_recovery_test(
        env=env,
        model=model,
        n_trials=int(args.hover_trials),
        seed=int(args.seed),
        perturb_radius=float(args.hover_perturb_radius),
        max_steps=max_steps,
        success_radius=float(args.success_radius),
        deterministic=True,
    )

    # ---- ROA sweep (optional) ----
    roa_metrics = roa_sweep_test(
        env=env,
        model=model,
        n_samples=int(args.roa_samples),
        seed=int(args.seed),
        max_steps=max_steps,
        success_radius=float(args.success_radius),
        deterministic=True,
    )

    # ---- Print summary ----
    print("\n=== Evaluation Summary ===")
    for k, v in metrics.items():
        print("{:>20s}: {}".format(k, v))
    print("\n=== Hover Recovery Test ===")
    for k, v in hover_metrics.items():
        print("{:>20s}: {}".format(k, v))
    print("\n=== ROA-like Sweep ===")
    for k, v in roa_metrics.items():
        print("{:>20s}: {}".format(k, v))

    # ---- Save summary to txt ----
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== Evaluation Summary ===\n")
        for k, v in metrics.items():
            f.write("{:>20s}: {}\n".format(k, v))
        f.write("\n=== Hover Recovery Test ===\n")
        for k, v in hover_metrics.items():
            f.write("{:>20s}: {}\n".format(k, v))
        f.write("\n=== ROA-like Sweep ===\n")
        for k, v in roa_metrics.items():
            f.write("{:>20s}: {}\n".format(k, v))
    print("\nWrote:", summary_path)

    # ---- Generate rollout plots ----
    for i in range(int(args.n_plot_episodes)):
        r = rollout_episode(
            env=env,
            model=model,
            seed=int(args.seed) + 500 + i,
            max_steps=max_steps,
            deterministic=True,
            set_initial_state=None,
        )
        out_path = os.path.join(out_dir, "rollout_episode_{:02d}.png".format(i))
        plot_rollout(
            out_path=out_path,
            rollout=r,
            dt=dt,
            title_prefix="{} Eval Ep {}".format(args.algo.upper(), i),
        )
        print("Saved plot:", out_path)

    print("\nDone. Outputs in:", out_dir)


if __name__ == "__main__":
    main()