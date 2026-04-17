#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train 20 SAC controllers on Pendulum-v1 with different reward functions.

Controllers are grouped into:
  - Fast convergers : aggressive angle penalties, minimal action cost
  - Conservative    : heavy action/smoothness penalties, prefer cautious control
  - Balanced        : intermediate trade-offs and alternative reward shapes
  - Shaped          : non-standard reward formulations (L1, cos, potential, etc.)
"""

import os
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor


# ---------------------------------------------------------------------------
# Custom reward wrapper
# ---------------------------------------------------------------------------

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class CustomRewardPendulum(gym.Wrapper):
    """
    Wraps Pendulum-v1 and replaces its reward with a configurable function.

    reward_fn(theta, thdot, action, prev_action, step) -> float
      theta      : angle in [-pi, pi], 0 = upright
      thdot      : angular velocity
      action     : scalar torque applied this step
      prev_action: torque applied the previous step (for smoothness terms)
      step       : current episode step count
    """

    def __init__(self, reward_fn, name="custom"):
        env = gym.make("Pendulum-v1")
        super().__init__(env)
        self.reward_fn = reward_fn
        self.controller_name = name
        self._prev_action = 0.0
        self._step = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_action = 0.0
        self._step = 0
        return obs, info

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)
        theta = angle_normalize(np.arctan2(obs[1], obs[0]))
        thdot = obs[2]
        u = float(action[0]) if hasattr(action, "__len__") else float(action)
        reward = self.reward_fn(theta, thdot, u, self._prev_action, self._step)
        self._prev_action = u
        self._step += 1
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# 20 reward configurations
# ---------------------------------------------------------------------------

def make_reward_configs():
    """Return list of (name, description, reward_fn) tuples."""

    configs = []

    # ---- GROUP 1: FAST CONVERGERS (5) ------------------------------------
    # Idea: large angle penalty, tiny or zero action penalty → aggressive control

    def fast1(theta, thdot, u, pu, step):
        """Very high angle weight, negligible action cost."""
        return -(5.0 * theta**2 + 0.1 * thdot**2 + 0.0001 * u**2)

    configs.append(("fast_1_high_angle", "5x angle weight, minimal action penalty", fast1))

    def fast2(theta, thdot, u, pu, step):
        """Extreme angle weight, zero action cost."""
        return -(10.0 * theta**2 + 0.05 * thdot**2)

    configs.append(("fast_2_extreme_angle", "10x angle weight, no action cost", fast2))

    def fast3(theta, thdot, u, pu, step):
        """Exponential bonus near upright + base penalty."""
        base = -(theta**2 + 0.1 * thdot**2 + 0.001 * u**2)
        bonus = 2.0 * np.exp(-5.0 * theta**2)   # large bonus when near vertical
        return base + bonus

    configs.append(("fast_3_exp_bonus", "Base penalty + Gaussian bonus near upright", fast3))

    def fast4(theta, thdot, u, pu, step):
        """Sparse big reward inside goal cone + continuous small push."""
        continuous = -(theta**2 + 0.05 * thdot**2)
        sparse = 5.0 if abs(theta) < 0.15 and abs(thdot) < 0.5 else 0.0
        return continuous + sparse

    configs.append(("fast_4_sparse_bonus", "Sparse +5 reward inside goal cone", fast4))

    def fast5(theta, thdot, u, pu, step):
        """High angle penalty, slight velocity bonus for moving toward upright."""
        return -(4.0 * theta**2 + 0.2 * thdot**2)

    configs.append(("fast_5_velocity_damp", "4x angle, 0.2x velocity, no action cost", fast5))

    # ---- GROUP 2: CONSERVATIVE (5) ---------------------------------------
    # Idea: heavy action penalty and/or smoothness reward → gentle, stable control

    def cons1(theta, thdot, u, pu, step):
        """Standard weights but 50x action penalty."""
        return -(theta**2 + 0.1 * thdot**2 + 0.05 * u**2)

    configs.append(("conservative_1_action50x", "50x action penalty vs standard", cons1))

    def cons2(theta, thdot, u, pu, step):
        """Very high action penalty — extremely cautious torque."""
        return -(theta**2 + 0.1 * thdot**2 + 0.5 * u**2)

    configs.append(("conservative_2_action500x", "500x action penalty vs standard", cons2))

    def cons3(theta, thdot, u, pu, step):
        """Penalize action *changes* (smoothness) on top of standard cost."""
        delta_u = (u - pu) ** 2
        return -(theta**2 + 0.1 * thdot**2 + 0.001 * u**2 + 0.5 * delta_u)

    configs.append(("conservative_3_smooth", "Penalise action changes (jerk penalty)", cons3))

    def cons4(theta, thdot, u, pu, step):
        """Energy-aware: penalise kinetic energy (thdot^2) heavily."""
        kinetic = 0.5 * thdot**2     # proportional to physical KE
        return -(theta**2 + kinetic + 0.05 * u**2)

    configs.append(("conservative_4_energy", "Heavy kinetic-energy penalty", cons4))

    def cons5(theta, thdot, u, pu, step):
        """Penalise both action magnitude and action changes, moderate angle."""
        return -(0.5 * theta**2 + 0.2 * thdot**2 + 0.1 * u**2 + 0.3 * (u - pu)**2)

    configs.append(("conservative_5_combo", "Moderate angle + action + jerk penalty", cons5))

    # ---- GROUP 3: BALANCED WITH DIFFERENT SHAPES (5) ---------------------

    def bal1(theta, thdot, u, pu, step):
        """Standard Pendulum-v1 reward (baseline)."""
        return -(theta**2 + 0.1 * thdot**2 + 0.001 * u**2)

    configs.append(("balanced_1_standard", "Standard Pendulum-v1 reward (baseline)", bal1))

    def bal2(theta, thdot, u, pu, step):
        """Cosine-based reward — smooth, bounded, naturally shaped."""
        return np.cos(theta) - 0.1 * thdot**2 - 0.001 * u**2

    configs.append(("balanced_2_cosine", "cos(theta) reward — smooth Lyapunov shape", bal2))

    def bal3(theta, thdot, u, pu, step):
        """L1 norm instead of L2 — less aggressive near goal, harder far away."""
        return -(abs(theta) + 0.1 * abs(thdot) + 0.001 * abs(u))

    configs.append(("balanced_3_L1", "L1 norm penalty (abs instead of squared)", bal3))

    def bal4(theta, thdot, u, pu, step):
        """Quartic angle penalty — steeper cost far from upright."""
        return -(theta**4 + 0.1 * thdot**2 + 0.001 * u**2)

    configs.append(("balanced_4_quartic", "Quartic angle penalty — steeper far from goal", bal4))

    def bal5(theta, thdot, u, pu, step):
        """Focus on velocity: once near upright, damp velocity strongly."""
        near = float(abs(theta) < 0.3)
        vel_weight = 1.0 + 4.0 * near    # 5x velocity penalty near upright
        return -(theta**2 + vel_weight * 0.1 * thdot**2 + 0.001 * u**2)

    configs.append(("balanced_5_adaptive_vel", "Adaptive velocity weight near upright", bal5))

    # ---- GROUP 4: SHAPED / EXOTIC (5) ------------------------------------

    def shaped1(theta, thdot, u, pu, step):
        """Log-barrier around goal — very flat far away, sharp near upright."""
        barrier = -np.log(1.0 + theta**2) - 0.1 * thdot**2 - 0.001 * u**2
        return barrier

    configs.append(("shaped_1_log_barrier", "Log penalty — flat far, sharp near upright", shaped1))

    def shaped2(theta, thdot, u, pu, step):
        """Asymmetric: larger penalty when pendulum falls past horizontal."""
        far_penalty = 2.0 if abs(theta) > np.pi / 2 else 1.0
        return -(far_penalty * theta**2 + 0.1 * thdot**2 + 0.001 * u**2)

    configs.append(("shaped_2_asymmetric", "2x angle penalty past horizontal", shaped2))

    def shaped3(theta, thdot, u, pu, step):
        """Time-annealed action penalty: strict early → lenient later."""
        decay = max(0.001, 0.1 * np.exp(-step / 150.0))
        return -(theta**2 + 0.1 * thdot**2 + decay * u**2)

    configs.append(("shaped_3_annealed_action", "Action penalty decays over episode", shaped3))

    def shaped4(theta, thdot, u, pu, step):
        """Potential-based shaping with hand-crafted Lyapunov potential."""
        # phi(s) = -(theta^2 + 0.1*thdot^2)  (LQR-like)
        phi_curr = -(theta**2 + 0.1 * thdot**2)
        # we approximate phi_next ≈ phi_curr + d/dt * dt (can't compute exactly here)
        # Instead use a shaped reward that includes the potential value directly
        return phi_curr - 0.001 * u**2

    configs.append(("shaped_4_lyapunov_potential", "Lyapunov potential shaping reward", shaped4))

    def shaped5(theta, thdot, u, pu, step):
        """Mixed: high angle weight early, adds velocity focus once near goal."""
        angle_w = 5.0 if abs(theta) > 0.5 else 1.0
        vel_w = 0.1 if abs(theta) > 0.5 else 0.5
        return -(angle_w * theta**2 + vel_w * thdot**2 + 0.001 * u**2)

    configs.append(("shaped_5_region_adaptive", "Region-adaptive weights: aggressive far, stable near", shaped5))

    assert len(configs) == 20, f"Expected 20 configs, got {len(configs)}"
    return configs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_controller(name, description, reward_fn, save_dir, timesteps=200_000):
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"Desc    : {description}")
    print(f"Steps   : {timesteps}")
    print(f"{'='*60}")

    env = Monitor(CustomRewardPendulum(reward_fn, name=name))
    model = SAC("MlpPolicy", env, verbose=0, seed=42)
    model.learn(total_timesteps=timesteps)

    model_path = os.path.join(save_dir, name)
    model.save(model_path)
    env.close()
    print(f"Saved -> {model_path}.zip")
    return model


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_controller(model, n_episodes=10, max_steps=200):
    """
    Returns:
        mean_return       : average episode return (standard Pendulum reward)
        steps_to_upright  : avg steps to first reach |theta|<0.1 (or max_steps if never)
        mean_action_norm  : average |u| per episode (conservatism metric)
    """
    env = gym.make("Pendulum-v1")
    returns, steps_up, action_norms = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        ep_ret = 0.0
        first_up = max_steps
        ep_actions = []

        for t in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += reward
            u = float(action[0])
            ep_actions.append(abs(u))

            theta = angle_normalize(np.arctan2(obs[1], obs[0]))
            if abs(theta) < 0.1 and first_up == max_steps:
                first_up = t

            if terminated or truncated:
                break

        returns.append(ep_ret)
        steps_up.append(first_up)
        action_norms.append(np.mean(ep_actions))

    env.close()
    return np.mean(returns), np.mean(steps_up), np.mean(action_norms)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_summary(names, mean_returns, steps_upright, action_norms, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    colors = plt.cm.tab20(np.linspace(0, 1, len(names)))
    short_names = [n.replace("_", "\n") for n in names]

    # 1) Mean return (higher = better)
    axes[0].barh(short_names, mean_returns, color=colors)
    axes[0].set_xlabel("Mean Episode Return", fontsize=13)
    axes[0].set_title("Performance\n(higher = better)", fontsize=14)
    axes[0].axvline(0, color="k", linewidth=0.8)

    # 2) Steps to upright (lower = faster convergence)
    axes[1].barh(short_names, steps_upright, color=colors)
    axes[1].set_xlabel("Steps to |θ| < 0.1  (lower = faster)", fontsize=13)
    axes[1].set_title("Convergence Speed\n(lower = faster)", fontsize=14)

    # 3) Mean |action| (lower = more conservative)
    axes[2].barh(short_names, action_norms, color=colors)
    axes[2].set_xlabel("Mean |u| (lower = conservative)", fontsize=13)
    axes[2].set_title("Control Effort\n(lower = conservative)", fontsize=14)

    plt.tight_layout()
    out = os.path.join(save_dir, "controller_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSummary plot saved -> {out}")


def plot_trajectories(models_dict, save_dir, n_steps=200):
    """Plot a phase-plane trajectory for each controller from the same start."""
    n = len(models_dict)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = axes.flatten()

    env = gym.make("Pendulum-v1")
    init_theta = np.pi    # start hanging down
    init_thdot = 0.0

    for ax, (name, model) in zip(axes, models_dict.items()):
        env.reset()
        env.unwrapped.state = np.array([init_theta, init_thdot])
        obs = np.array([np.cos(init_theta), np.sin(init_theta), init_thdot])

        thetas, omegas = [init_theta], [init_thdot]
        for _ in range(n_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            thetas.append(angle_normalize(np.arctan2(obs[1], obs[0])))
            omegas.append(obs[2])
            if terminated or truncated:
                break

        ax.plot(thetas, omegas, linewidth=1.5)
        ax.plot(thetas[0], omegas[0], "go", markersize=6, label="start")
        ax.plot(thetas[-1], omegas[-1], "ro", markersize=6, label="end")
        ax.axhline(0, color="k", linewidth=0.4, linestyle="--")
        ax.axvline(0, color="k", linewidth=0.4, linestyle="--")
        ax.set_title(name.replace("_", " "), fontsize=8)
        ax.set_xlabel("θ (rad)", fontsize=7)
        ax.set_ylabel("ω (rad/s)", fontsize=7)
        ax.tick_params(labelsize=6)

    env.close()

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle("Phase-plane trajectories from θ=π (hanging down)", fontsize=13)
    plt.tight_layout()
    out = os.path.join(save_dir, "trajectories.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Trajectory plot saved -> {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, "saved_models", "20controllers")
    results_dir = os.path.join(script_dir, "results", "20controllers")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    configs = make_reward_configs()

    # --- Training -----------------------------------------------------------
    trained_models = {}
    for name, description, reward_fn in configs:
        model_path = os.path.join(save_dir, name + ".zip")
        if os.path.exists(model_path):
            print(f"[skip] {name} already trained, loading...")
            trained_models[name] = SAC.load(model_path.replace(".zip", ""))
        else:
            model = train_controller(name, description, reward_fn, save_dir, timesteps=200_000)
            trained_models[name] = model

    # --- Evaluation ---------------------------------------------------------
    print("\n\nEvaluating all controllers...")
    results = {}
    for name, model in trained_models.items():
        mean_ret, steps_up, act_norm = evaluate_controller(model)
        results[name] = {
            "mean_return": float(mean_ret),
            "steps_to_upright": float(steps_up),
            "mean_action_norm": float(act_norm),
        }
        print(f"  {name:45s}  return={mean_ret:8.1f}  steps_up={steps_up:6.1f}  |u|={act_norm:.3f}")

    # Save results as JSON
    json_path = os.path.join(results_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {json_path}")

    # --- Plots --------------------------------------------------------------
    names = list(results.keys())
    mean_returns = [results[n]["mean_return"] for n in names]
    steps_upright = [results[n]["steps_to_upright"] for n in names]
    action_norms = [results[n]["mean_action_norm"] for n in names]

    plot_summary(names, mean_returns, steps_upright, action_norms, results_dir)
    plot_trajectories(trained_models, results_dir)

    # --- Ranking summary ----------------------------------------------------
    print("\n--- Fastest convergers (lowest steps_to_upright) ---")
    ranked_fast = sorted(names, key=lambda n: results[n]["steps_to_upright"])
    for i, n in enumerate(ranked_fast[:5], 1):
        print(f"  {i}. {n:45s}  steps={results[n]['steps_to_upright']:.1f}")

    print("\n--- Most conservative (lowest mean |u|) ---")
    ranked_cons = sorted(names, key=lambda n: results[n]["mean_action_norm"])
    for i, n in enumerate(ranked_cons[:5], 1):
        print(f"  {i}. {n:45s}  |u|={results[n]['mean_action_norm']:.3f}")

    print("\n--- Best overall return ---")
    ranked_ret = sorted(names, key=lambda n: results[n]["mean_return"], reverse=True)
    for i, n in enumerate(ranked_ret[:5], 1):
        print(f"  {i}. {n:45s}  return={results[n]['mean_return']:.1f}")


if __name__ == "__main__":
    main()
