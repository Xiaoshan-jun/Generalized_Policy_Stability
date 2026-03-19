#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quadrotor 3D Simulator — matched to StepNetTrainer3D training setup.

The RL policy (PPO or SAC) drives the environment.
StepNet + LyapunovNet are evaluated to verify the Lyapunov stability condition:

    sum_k  sigma_k * V(x_k)  <=  (1 - alpha) * V(x_0)

where
    V(x) = |V_RL(x) - V_RL(x*)| + ||phi(x) - phi(x*)||^2 + beta*||x - x*||^2
    sigma = StepNet(x_0)   (normalised to sum = 1)

Usage
-----
    # Single checkpoint
    python simulate_quadrotor.py \
        --rl_model_path  saved_models/ppo/quadrotor3d_model \
        --stepnet_path   saved_models/ppo/15steps_quadrotor3d/stepnet_best.pth \
        --residual_path  saved_models/ppo/15steps_quadrotor3d/residual_net_best.pth \
        --algo ppo --n_steps 15 --n_episodes 5 --out_dir results/

    # Compare multiple n_steps checkpoints  (mirrors the training loop in main())
    python simulate_quadrotor.py \
        --rl_model_path saved_models/ppo/quadrotor3d_model \
        --compare_nsteps 5 10 15 20 \
        --compare_dir    saved_models/ppo \
        --algo ppo --n_episodes 3 --out_dir results/compare/
"""

import argparse
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    from stable_baselines3 import PPO, SAC
    _HAS_SB3 = True
except ImportError:
    _HAS_SB3 = False

from quadrotor3d_env import Quadrotor3DEnv


# ══════════════════════════════════════════════════════════════════════════════
# Network definitions  (must match training)
# ══════════════════════════════════════════════════════════════════════════════

class StepNet(nn.Module):
    def __init__(self, n_input=12, n_hidden=128, n_steps=15, n_layers=3):
        super().__init__()
        self.n_steps = n_steps
        self.layers = nn.ModuleList([nn.Linear(n_input, n_hidden)])
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        self.output_layer = nn.Linear(n_hidden, n_steps)
        self.activation = nn.LeakyReLU(0.01)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        for layer in self.layers:
            x = self.activation(layer(x))
        logits = self.output_layer(x)
        return self.n_steps * torch.softmax(logits, dim=1)


class LyapunovNet(nn.Module):
    """phi(x) — always outputs a scalar [batch, 1]."""
    def __init__(self, n_input=12, n_hidden=128, n_layers=3,
                 leaky_relu_slope=0.01):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(n_input, n_hidden)])
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        self.output_layer = nn.Linear(n_hidden, 1)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, a=leaky_relu_slope,
                                    nonlinearity="leaky_relu")
            nn.init.zeros_(layer.bias)
        nn.init.kaiming_normal_(self.output_layer.weight, a=leaky_relu_slope,
                                nonlinearity="leaky_relu")
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.output_layer(x)   # [batch, 1]


# ══════════════════════════════════════════════════════════════════════════════
# Lyapunov evaluator  (mirrors StepNetTrainer3D.get_residual_value)
# ══════════════════════════════════════════════════════════════════════════════

class LyapunovEvaluator:
    def __init__(self, rl_model, algo_type: str,
                 stepnet: StepNet, residual_net: LyapunovNet,
                 equilibrium: np.ndarray, beta: float, alpha: float,
                 device: torch.device):
        self.rl_model   = rl_model
        self.algo_type  = algo_type.lower()
        self.stepnet    = stepnet.eval()
        self.residual   = residual_net.eval()
        self.beta       = beta
        self.alpha      = alpha
        self.device     = device
        self.eq         = torch.from_numpy(equilibrium.astype(np.float32)).to(device)
        # RL model may be on a different device (e.g. cuda) than stepnet/residual
        self.rl_device  = next(iter(rl_model.policy.parameters())).device

        with torch.no_grad():
            self.v_eq = float(self._vrl(self.eq.unsqueeze(0)).item())

    @torch.no_grad()
    def _vrl(self, x_batch: torch.Tensor) -> torch.Tensor:
        """Run RL value/Q function. Moves input to RL device, output back to self.device."""
        x = x_batch.to(self.rl_device)
        if self.algo_type == "ppo":
            return self.rl_model.policy.predict_values(x).view(-1).to(self.device)
        a = self.rl_model.actor(x)
        q1, q2 = self.rl_model.critic(x, a)
        return torch.min(q1, q2).view(-1).to(self.device)

    @torch.no_grad()
    def V(self, x_batch: torch.Tensor) -> torch.Tensor:
        """V(x) = |V_RL - V_RL*| + ||phi - phi*||^2 + beta*||x - x*||^2"""
        v_rl   = self._vrl(x_batch)
        phi_x  = self.residual(x_batch).squeeze(-1)   # [B]
        phi_eq = self.residual(self.eq.unsqueeze(0)).squeeze(-1)  # [1]
        diff   = x_batch - self.eq.unsqueeze(0)
        return (torch.abs(v_rl - self.v_eq)
                + (phi_x - phi_eq) ** 2
                + self.beta * torch.sum(diff ** 2, dim=-1))

    @torch.no_grad()
    def sigma(self, x0: torch.Tensor) -> torch.Tensor:
        """Normalised StepNet weights summing to 1, shape (n_steps,)."""
        raw = self.stepnet(x0.unsqueeze(0))
        s   = torch.relu(raw).view(-1)
        return s / (s.sum() + 1e-8)

    @torch.no_grad()
    def stability_margin(self, traj: torch.Tensor) -> float:
        """
        margin = sum_k sigma_k V(x_k) - (1-alpha) V(x_0)
        Negative -> stable.  Positive -> violation.
        """
        x0     = traj[0:1]
        sig    = self.sigma(x0)
        n      = min(len(sig), len(traj) - 1)
        future = self.V(traj[1:n + 1])
        V0     = self.V(x0).item()
        return float((sig[:n] * future).sum()) - (1.0 - self.alpha) * V0


# ══════════════════════════════════════════════════════════════════════════════
# Episode runner
# ══════════════════════════════════════════════════════════════════════════════

def run_episode(env, rl_model, lyap: Optional[LyapunovEvaluator],
                x0: Optional[np.ndarray], n_steps: int,
                device: torch.device) -> dict:
    obs, _ = env.reset()
    if x0 is not None:
        env.x_current = torch.from_numpy(x0.astype(np.float32))
        obs = x0.astype(np.float32)

    states, actions, rewards, times = [], [], [], []
    t, done = 0.0, False
    while not done:
        action, _ = rl_model.predict(obs, deterministic=True)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        states.append(obs.copy())
        actions.append(np.asarray(action, dtype=np.float32).copy())
        rewards.append(reward)
        times.append(t)
        obs   = next_obs
        t    += env.dt
        done  = terminated or truncated

    states  = np.array(states,  dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)
    times   = np.array(times,   dtype=np.float32)

    lyap_V = lyap_sig = margins = None
    if lyap is not None and len(states) >= 2:
        traj_t = torch.from_numpy(states).float().to(device)
        lyap_V = lyap.V(traj_t).cpu().numpy()

        # rolling stability margins over each n_steps window
        margins = np.array([
            lyap.stability_margin(traj_t[i: i + n_steps + 1])
            for i in range(len(states) - 1)
            if traj_t[i: i + n_steps + 1].shape[0] >= 2
        ], dtype=np.float32)

        lyap_sig = lyap.sigma(traj_t[0]).cpu().numpy()

    return dict(states=states, actions=actions, rewards=rewards, times=times,
                terminated=terminated, truncated=truncated,
                lyap_V=lyap_V, lyap_sig=lyap_sig, margins=margins)


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

_C = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800",
      "#9C27B0", "#00BCD4", "#FF5722", "#8BC34A"]
STATE_LABELS = [
    "pos_x [m]", "pos_y [m]", "pos_z [m]",
    "roll [rad]", "pitch [rad]", "yaw [rad]",
    "vel_x [m/s]", "vel_y [m/s]", "vel_z [m/s]",
    "ω_x [rad/s]", "ω_y [rad/s]", "ω_z [rad/s]",
]


def plot_episode(ep: dict, n_steps: int, title="Episode", save_path=None):
    states, actions, rewards, times = (ep[k] for k in
                                       ("states", "actions", "rewards", "times"))
    lyap_V, margins, sig = ep["lyap_V"], ep["margins"], ep["lyap_sig"]
    has_lyap = lyap_V is not None

    nrows = 4 if has_lyap else 3
    fig = plt.figure(figsize=(22, 5 * nrows), constrained_layout=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    outer = gridspec.GridSpec(nrows, 2, figure=fig)

    # 3-D trajectory
    ax3 = fig.add_subplot(outer[0, 0], projection="3d")
    ax3.plot(states[:, 0], states[:, 1], states[:, 2], lw=1.5, color=_C[0])
    ax3.scatter(*states[0, :3],  s=60, color="green", zorder=5, label="start")
    ax3.scatter(*states[-1, :3], s=60, color="red",   zorder=5, label="end")
    ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")
    ax3.set_title("3-D Trajectory"); ax3.legend(fontsize=8)

    # Cumulative reward
    ax_r = fig.add_subplot(outer[0, 1])
    ax_r.plot(times, rewards,            color="grey",  alpha=0.5, lw=1,   label="step")
    ax_r.plot(times, np.cumsum(rewards), color=_C[2],   lw=1.8, label="cumulative")
    ax_r.axhline(0, color="k", lw=0.4, ls="--")
    ax_r.set_xlabel("t [s]"); ax_r.set_ylabel("reward")
    ax_r.set_title("Reward"); ax_r.legend(fontsize=8)

    # States
    inner_s = gridspec.GridSpecFromSubplotSpec(
        4, 3, subplot_spec=outer[1, :], hspace=0.65, wspace=0.35)
    for i in range(12):
        ax = fig.add_subplot(inner_s[i // 3, i % 3])
        ax.plot(times, states[:, i], lw=1.1, color=_C[i % len(_C)], alpha=0.85)
        ax.axhline(0, color="k", lw=0.4, ls="--")
        ax.set_title(STATE_LABELS[i], fontsize=7, pad=2)
        ax.tick_params(labelsize=6); ax.set_xlabel("t [s]", fontsize=6)
        if not np.isfinite(states[:, i]).all():
            ax.set_facecolor("#fff0f0")

    # Motor thrusts
    inner_a = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=outer[2, :], hspace=0.5, wspace=0.35)
    for i in range(4):
        ax = fig.add_subplot(inner_a[i // 2, i % 2])
        ax.plot(times, actions[:, i], lw=1.1, color=_C[i], alpha=0.85)
        ax.set_title(f"Motor {i+1} thrust [N]", fontsize=8, pad=2)
        ax.tick_params(labelsize=6); ax.set_xlabel("t [s]", fontsize=6)

    # Lyapunov panels
    if has_lyap:
        inner_l = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=outer[3, :], wspace=0.4)

        ax_v = fig.add_subplot(inner_l[0])
        ax_v.plot(times, lyap_V, lw=1.4, color="#E91E63")
        ax_v.axhline(0, color="k", lw=0.4, ls="--")
        ax_v.set_xlabel("t [s]"); ax_v.set_ylabel("V(x)")
        ax_v.set_title("Lyapunov value V(x_t)")

        ax_m = fig.add_subplot(inner_l[1])
        dt = float(times[1] - times[0]) * 0.8 if len(times) > 1 else 0.01
        mt = times[:len(margins)]
        ax_m.bar(mt, margins, width=dt,
                 color=np.where(margins > 0, "#E53935", "#43A047"), alpha=0.75)
        ax_m.axhline(0, color="k", lw=1)
        ax_m.set_xlabel("t [s]"); ax_m.set_ylabel("margin")
        n_viol = int((margins > 0).sum())
        ax_m.set_title(f"Stability margin  ({n_viol}/{len(margins)} violations)")

        ax_sig = fig.add_subplot(inner_l[2])
        ax_sig.bar(np.arange(1, len(sig) + 1), sig, color="#1565C0", alpha=0.8)
        ax_sig.set_xlabel("future step k"); ax_sig.set_ylabel("σ_k")
        ax_sig.set_title(f"StepNet σ at x_0  (n_steps={n_steps})")

    status = "TRUNCATED" if ep["truncated"] else "TERMINATED"
    footer = (f"Exit: {status}  |  Steps: {len(times)}  |  ΣR = {rewards.sum():.4f}"
              + (f"  |  violations: {int((margins > 0).sum())}/{len(margins)}"
                 if margins is not None else ""))
    fig.text(0.01, 0.005, footer, fontsize=8,
             color="steelblue" if ep["truncated"] else "crimson")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  saved  {save_path}")
    return fig


def plot_multi_summary(episodes, n_steps, save_path=None):
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(episodes)))
    has_lyap = episodes[0]["lyap_V"] is not None
    ncols = 4 if has_lyap else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    fig.suptitle(f"Multi-episode summary  (n_steps={n_steps})", fontsize=11,
                 fontweight="bold")

    for k, ep in enumerate(episodes):
        s, t, c = ep["states"], ep["times"], cmap[k]
        axes[0].plot(s[:, 0], s[:, 1], lw=1.2, color=c, alpha=0.8)
        axes[0].scatter(*s[0, :2], s=20, color=c)
        axes[1].plot(t, s[:, 2], lw=1.2, color=c, alpha=0.8, label=f"ep{k+1}")
        axes[2].plot(t, np.cumsum(ep["rewards"]), lw=1.2, color=c, alpha=0.8)
        if has_lyap and ep["lyap_V"] is not None:
            axes[3].plot(t, ep["lyap_V"], lw=1.2, color=c, alpha=0.8)

    labels = [("X [m]", "Y [m]", "XY footprint"),
              ("t [s]",  "Z [m]",   "Altitude"),
              ("t [s]",  "Σ reward","Cumulative reward"),
              ("t [s]",  "V(x)",    "Lyapunov V(x)")]
    for ax, (xl, yl, ttl) in zip(axes, labels[:ncols]):
        ax.axhline(0, color="k", lw=0.4, ls="--")
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ttl)
    axes[1].legend(fontsize=7)

    if has_lyap:
        viol = [float((ep["margins"] > 0).mean())
                for ep in episodes if ep["margins"] is not None]
        fig.text(0.5, -0.02,
                 f"Mean violation rate: {np.mean(viol)*100:.1f}%  "
                 f"Max: {np.max(viol)*100:.1f}%",
                 ha="center", fontsize=9, color="crimson")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  saved  {save_path}")
    return fig


def plot_nsteps_comparison(results: dict, save_path=None):
    """Bar charts comparing violation rate, cumulative reward, episode length."""
    keys = sorted(results.keys())
    viol  = [np.mean([(ep["margins"] > 0).mean()
                      for ep in results[n] if ep["margins"] is not None])
             for n in keys]
    cum_r = [np.mean([ep["rewards"].sum() for ep in results[n]]) for n in keys]
    T_mean= [np.mean([len(ep["times"])    for ep in results[n]]) for n in keys]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Checkpoint comparison across n_steps", fontsize=11,
                 fontweight="bold")

    axes[0].bar(keys, [v * 100 for v in viol], color="#E53935", alpha=0.8)
    axes[0].set(xlabel="n_steps", ylabel="violation rate [%]",
                title="Lyapunov violation rate", xticks=keys)

    axes[1].bar(keys, cum_r, color="#1565C0", alpha=0.8)
    axes[1].set(xlabel="n_steps", ylabel="Σ reward",
                title="Mean cumulative reward", xticks=keys)

    axes[2].bar(keys, T_mean, color="#2E7D32", alpha=0.8)
    axes[2].set(xlabel="n_steps", ylabel="steps",
                title="Mean episode length", xticks=keys)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  saved  {save_path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_rl(model_path, algo):
    assert _HAS_SB3, "stable-baselines3 required."
    cls = PPO if algo.lower() == "ppo" else SAC
    m = cls.load(model_path)
    print(f"Loaded {algo.upper()} from  {model_path}")
    return m


def load_lyapunov(sp, rp, n_steps, state_dim, hidden_dim, n_layers,
                  equilibrium, beta, alpha, rl_model, algo, device):
    snet = StepNet(n_input=state_dim, n_hidden=hidden_dim,
                   n_steps=n_steps, n_layers=n_layers).to(device)
    snet.load_state_dict(torch.load(sp, map_location=device))

    rnet = LyapunovNet(n_input=state_dim, n_hidden=hidden_dim,
                       n_layers=n_layers).to(device)  # output is always scalar
    rnet.load_state_dict(torch.load(rp, map_location=device))

    print(f"  StepNet  <- {sp}")
    print(f"  Residual <- {rp}")
    return LyapunovEvaluator(rl_model=rl_model, algo_type=algo,
                             stepnet=snet, residual_net=rnet,
                             equilibrium=equilibrium,
                             beta=beta, alpha=alpha, device=device)


def run_batch(env, rl_model, lyap, cfg, n_steps, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(cfg.device)
    np.random.seed(cfg.seed)
    episodes = []
    for k in range(cfg.n_episodes):
        x0 = np.array(cfg.x0, np.float32) if (k == 0 and cfg.x0) else None
        ep = run_episode(env, rl_model, lyap, x0=x0,
                         n_steps=n_steps, device=device)
        episodes.append(ep)
        m_str = (f"  viol={100*(ep['margins']>0).mean():.1f}%"
                 if ep["margins"] is not None else "")
        print(f"  ep{k+1:3d} | steps={len(ep['times']):5d} | "
              f"ΣR={ep['rewards'].sum():+.4f} | "
              f"{'TRUNC' if ep['truncated'] else 'TERM'}{m_str}")
        plot_episode(ep, n_steps=n_steps,
                     title=f"Episode {k+1}  (n_steps={n_steps})",
                     save_path=os.path.join(out_dir, f"ep_{k+1:02d}.png"))
    if cfg.n_episodes > 1:
        plot_multi_summary(episodes, n_steps,
                           save_path=os.path.join(out_dir, "summary.png"))
    return episodes


_HERE       = os.path.dirname(os.path.abspath(__file__))
_SAVED_ROOT = os.path.join(_HERE, "saved_models")


def _default_paths(algo: str, n_steps: int):
    """Mirror StepNetTrainer3D.train() save-directory structure."""
    base = os.path.join(_SAVED_ROOT, algo)
    return (
        os.path.join(base, "quadrotor3d_model"),
        os.path.join(base, f"{n_steps}steps_quadrotor3d", "stepnet_best.pth"),
        os.path.join(base, f"{n_steps}steps_quadrotor3d", "residual_net_best.pth"),
    )


def build_parser():
    p = argparse.ArgumentParser(
        description="Quadrotor 3D Lyapunov Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--algo",    default="ppo", choices=["ppo", "sac"])
    p.add_argument("--n_steps", type=int, default=15)
    # paths -- default to StepNetTrainer3D save structure; override if needed
    p.add_argument("--rl_model_path", default=None,
                   help="Default: saved_models/<algo>/quadrotor3d_model")
    p.add_argument("--stepnet_path",  default=None,
                   help="Default: saved_models/<algo>/<n_steps>steps_quadrotor3d/stepnet_best.pth")
    p.add_argument("--residual_path", default=None,
                   help="Default: saved_models/<algo>/<n_steps>steps_quadrotor3d/residual_net_best.pth")
    # compare mode
    p.add_argument("--compare_nsteps", type=int, nargs="+", default=None)
    p.add_argument("--compare_dir",    default=None,
                   help="Base dir for compare mode. Default: saved_models/")
    # network hyper-params (must match training)
    p.add_argument("--hidden_dim", type=int,   default=128)
    p.add_argument("--n_layers",   type=int,   default=3)
    p.add_argument("--alpha",      type=float, default=0.05)
    p.add_argument("--beta",       type=float, default=0.01)
    # simulation
    p.add_argument("--n_episodes", type=int,   default=5)
    p.add_argument("--x0",         type=float, nargs=12, default=None)
    p.add_argument("--dt",         type=float, default=0.01)
    p.add_argument("--max_time",   type=float, default=2.0)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--device",     default="cpu")
    p.add_argument("--out_dir",    default="sim_results")
    return p


def main(args=None):
    cfg = build_parser().parse_args(args)

    # Fill in default paths from training save structure
    default_rl, default_sn, default_rn = _default_paths(cfg.algo, cfg.n_steps)
    if cfg.rl_model_path is None:
        cfg.rl_model_path = default_rl
    if cfg.stepnet_path  is None:
        cfg.stepnet_path  = default_sn
    if cfg.residual_path is None:
        cfg.residual_path = default_rn
    if cfg.compare_dir   is None:
        cfg.compare_dir   = _SAVED_ROOT

    print(f"RL model  : {cfg.rl_model_path}")
    print(f"StepNet   : {cfg.stepnet_path}")
    print(f"Residual  : {cfg.residual_path}")

    os.makedirs(cfg.out_dir, exist_ok=True)

    env         = Quadrotor3DEnv(dt=cfg.dt, max_time=cfg.max_time, seed=cfg.seed)
    state_dim   = int(env.observation_space.shape[0])
    equilibrium = env.obs_equ.numpy().astype(np.float32)
    device      = torch.device(cfg.device)
    rl_model    = load_rl(cfg.rl_model_path, cfg.algo)

    # compare mode
    if cfg.compare_nsteps:
        compare_results = {}
        for n in cfg.compare_nsteps:
            base = os.path.join(cfg.compare_dir, cfg.algo,
                                f"{n}steps_quadrotor3d")
            sp = os.path.join(base, "stepnet_best.pth")
            rp = os.path.join(base, "residual_net_best.pth")
            if not (os.path.exists(sp) and os.path.exists(rp)):
                print(f"  SKIP n_steps={n}: checkpoints not found in {base}")
                continue
            lyap = load_lyapunov(sp, rp, n, state_dim, cfg.hidden_dim,
                                 cfg.n_layers, equilibrium, cfg.beta, cfg.alpha,
                                 rl_model, cfg.algo, device)
            print(f"\n-- n_steps = {n} --")
            out  = os.path.join(cfg.out_dir, f"nsteps_{n}")
            compare_results[n] = run_batch(env, rl_model, lyap, cfg,
                                           n_steps=n, out_dir=out)
        if len(compare_results) > 1:
            plot_nsteps_comparison(
                compare_results,
                save_path=os.path.join(cfg.out_dir, "nsteps_comparison.png"))
        return

    # single checkpoint mode
    lyap = None
    if os.path.exists(cfg.stepnet_path) and os.path.exists(cfg.residual_path):
        lyap = load_lyapunov(cfg.stepnet_path, cfg.residual_path,
                             cfg.n_steps, state_dim, cfg.hidden_dim,
                             cfg.n_layers, equilibrium, cfg.beta, cfg.alpha,
                             rl_model, cfg.algo, device)
    else:
        print("Lyapunov checkpoints not found -- running policy only (no V(x) diagnostics).")
        print(f"  stepnet  missing: {not os.path.exists(cfg.stepnet_path)}")
        print(f"  residual missing: {not os.path.exists(cfg.residual_path)}")

    run_batch(env, rl_model, lyap, cfg, n_steps=cfg.n_steps,
              out_dir=cfg.out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()