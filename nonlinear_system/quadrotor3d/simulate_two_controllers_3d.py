#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sequential two-controller simulation tester.

Strategy
--------
Phase 1 — Controller A (Attitude Stabiliser):
    Fly until (roll, pitch, yaw, wx, wy, wz) all stay within
    `switch_angle_tol` / `switch_omega_tol` for `switch_hold_steps`
    consecutive steps.

Phase 2 — Controller B (Position Stabiliser):
    Take over and drive (pos, vel) → 0 while keeping attitude level.

The raw Quadrotor dynamics are stepped directly (no Gym overhead),
so we can set arbitrary initial conditions.

Usage
-----
# Single trial from random initial state
python simulate_two_controllers_3d.py

# Custom initial state (hanging-down scenario)
python simulate_two_controllers_3d.py \
    --roll 0.5 --pitch -0.4 --yaw 1.2 \
    --wx 0.1 --wy -0.05 --wz 0.0 \
    --px 3.0 --py -2.0 --pz 5.0 \
    --vx 1.0 --vy 0.5 --vz -0.3

# Multiple random trials
python simulate_two_controllers_3d.py --n_trials 10 --random_init
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from stable_baselines3 import SAC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quadrotor3d import Quadrotor
from quadrotor3d_env import wrap_to_pi


# ============================================================================
# Dynamics wrapper (no Gym, pure numpy)
# ============================================================================

class QuadrotorSim:
    """Thin wrapper around Quadrotor for direct simulation."""

    def __init__(self, dt: float = 0.01):
        self.system = Quadrotor(None.__class__)          # dtype unused for numpy path
        import torch
        self.system = Quadrotor(torch.float32)
        self.hover_thrust = float(self.system.hover_thrust)
        self.dt = float(dt)
        self.u_lo = np.zeros(4, dtype=np.float32)
        self.u_hi = np.full(4, 2.5 * self.hover_thrust, dtype=np.float32)

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        action = np.clip(action, self.u_lo, self.u_hi).astype(np.float32)
        xdot   = self.system.dynamics(state, action).astype(np.float32)
        x_next = state + xdot * self.dt
        x_next[5] = wrap_to_pi(float(x_next[5]))
        return x_next.astype(np.float32)


# ============================================================================
# Switch condition
# ============================================================================

class AttitudeSwitchMonitor:
    """
    Signals True when attitude + angular rate have stayed within tolerance
    for `hold_steps` consecutive steps.
    """

    def __init__(self, angle_tol: float, omega_tol: float, hold_steps: int):
        self.angle_tol  = float(angle_tol)
        self.omega_tol  = float(omega_tol)
        self.hold_steps = int(hold_steps)
        self._count     = 0

    def update(self, state: np.ndarray) -> bool:
        roll, pitch, yaw = state[3], state[4], state[5]
        wx,   wy,   wz   = state[9], state[10], state[11]

        att_ok   = (abs(roll)  < self.angle_tol and
                    abs(pitch) < self.angle_tol and
                    abs(yaw)   < self.angle_tol)
        omega_ok = (abs(wx) < self.omega_tol and
                    abs(wy) < self.omega_tol and
                    abs(wz) < self.omega_tol)

        if att_ok and omega_ok:
            self._count += 1
        else:
            self._count = 0

        return self._count >= self.hold_steps

    @property
    def consecutive_ok(self) -> int:
        return self._count

    def reset(self):
        self._count = 0


# ============================================================================
# Core simulation
# ============================================================================

def run_simulation(
    model_A,
    model_B,
    init_state: np.ndarray,
    dt: float = 0.01,
    max_steps: int = 1500,
    switch_angle_tol: float = 0.10,
    switch_omega_tol: float = 0.15,
    switch_hold_steps: int  = 30,
    final_pos_tol: float    = 0.10,
    final_vel_tol: float    = 0.15,
) -> dict:
    """
    Run the two-phase simulation from `init_state`.

    Returns a dict with:
        states          [T, 12]
        actions         [T,  4]
        phases          [T]      0 = controller A, 1 = controller B
        switch_step     int or None
        reached_eq      bool
        eq_step         int or None
        t               [T]  time axis in seconds
    """
    sim     = QuadrotorSim(dt=dt)
    monitor = AttitudeSwitchMonitor(switch_angle_tol, switch_omega_tol, switch_hold_steps)

    state   = init_state.copy().astype(np.float32)
    phase   = 0          # 0 = A, 1 = B
    switch_step = None
    reached_eq  = False
    eq_step     = None

    states  = []
    actions = []
    phases  = []

    for step in range(max_steps):
        obs = state.copy()

        # Select controller
        if phase == 0:
            action, _ = model_A.predict(obs, deterministic=True)
            if monitor.update(state):
                phase = 1
                switch_step = step
        else:
            action, _ = model_B.predict(obs, deterministic=True)

        action = np.asarray(action, dtype=np.float32).reshape(4)
        state  = sim.step(state, action)

        states.append(state.copy())
        actions.append(action.copy())
        phases.append(phase)

        # Check full equilibrium (only meaningful once in phase B)
        if phase == 1 and not reached_eq:
            px, py, pz = state[0], state[1], state[2]
            vx, vy, vz = state[6], state[7], state[8]
            pos_ok = abs(px)<final_pos_tol and abs(py)<final_pos_tol and abs(pz)<final_pos_tol
            vel_ok = abs(vx)<final_vel_tol and abs(vy)<final_vel_tol and abs(vz)<final_vel_tol
            if pos_ok and vel_ok:
                reached_eq = True
                eq_step    = step

        # Hard crash detection
        if (not np.isfinite(state).all() or
                state[2] < -0.5 or
                abs(state[3]) > 2.0 or
                abs(state[4]) > 2.0):
            print(f"  [crash] at step {step}")
            break

    states  = np.array(states,  dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    phases  = np.array(phases,  dtype=np.int32)
    T       = states.shape[0]
    t       = np.arange(T, dtype=np.float32) * dt

    return {
        "states":      states,
        "actions":     actions,
        "phases":      phases,
        "switch_step": switch_step,
        "reached_eq":  reached_eq,
        "eq_step":     eq_step,
        "t":           t,
    }


# ============================================================================
# Plotting
# ============================================================================

_STATE_LABELS  = ["x","y","z","roll","pitch","yaw","vx","vy","vz","wx","wy","wz"]
_ACTION_LABELS = ["u1","u2","u3","u4"]

# colours for each phase background
_PHASE_COLORS = ["#d0e8ff", "#ffe8d0"]   # blue = A, orange = B


def _shade_phases(ax, t, phases, alpha=0.25):
    """Draw a coloured background band for each controller phase."""
    if len(t) == 0:
        return
    starts = [0]
    for i in range(1, len(phases)):
        if phases[i] != phases[i-1]:
            starts.append(i)
    starts.append(len(phases))

    for k in range(len(starts) - 1):
        i0, i1 = starts[k], starts[k+1]
        ph = phases[i0]
        ax.axvspan(t[i0], t[min(i1, len(t)-1)], alpha=alpha,
                   color=_PHASE_COLORS[ph], linewidth=0)


def plot_trial(result: dict, save_path: str, title: str = ""):
    states  = result["states"]
    actions = result["actions"]
    phases  = result["phases"]
    t       = result["t"]
    T       = len(t)

    fig = plt.figure(figsize=(18, 16))
    fig.suptitle(title or "Two-Controller Simulation", fontsize=14, y=1.002)
    gs  = GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)

    # ── helper: add switch / eq markers ──────────────────────────────────
    def _vlines(ax):
        if result["switch_step"] is not None:
            ts = t[result["switch_step"]]
            ax.axvline(ts, color="navy", linewidth=1.5, linestyle="--",
                       label="switch A→B")
        if result["eq_step"] is not None:
            te = t[result["eq_step"]]
            ax.axvline(te, color="darkgreen", linewidth=1.5, linestyle=":",
                       label="equilibrium reached")

    # ── (0,0) Position ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    _shade_phases(ax, t, phases)
    for i, lbl in enumerate(["x","y","z"]):
        ax.plot(t, states[:, i], label=lbl)
    _vlines(ax)
    ax.set_title("Position [m]"); ax.set_xlabel("t (s)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── (0,1) Velocity ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    _shade_phases(ax, t, phases)
    for i, lbl in zip([6,7,8], ["vx","vy","vz"]):
        ax.plot(t, states[:, i], label=lbl)
    _vlines(ax)
    ax.set_title("Linear Velocity [m/s]"); ax.set_xlabel("t (s)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── (1,0) Attitude ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    _shade_phases(ax, t, phases)
    for i, lbl in zip([3,4,5], ["roll","pitch","yaw"]):
        ax.plot(t, states[:, i], label=lbl)
    _vlines(ax)
    ax.set_title("Attitude [rad]"); ax.set_xlabel("t (s)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── (1,1) Angular velocity ────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    _shade_phases(ax, t, phases)
    for i, lbl in zip([9,10,11], ["wx","wy","wz"]):
        ax.plot(t, states[:, i], label=lbl)
    _vlines(ax)
    ax.set_title("Angular Velocity [rad/s]"); ax.set_xlabel("t (s)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── (2,0) Motor thrusts ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    _shade_phases(ax, t, phases)
    for i in range(4):
        ax.plot(t, actions[:, i], label=f"u{i+1}")
    _vlines(ax)
    ax.set_title("Motor Thrusts [N]"); ax.set_xlabel("t (s)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── (2,1) Controller phase timeline ───────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    ax.fill_between(t, phases, step="post", alpha=0.7, color="steelblue")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["A (Attitude)", "B (Position)"])
    ax.set_title("Active Controller"); ax.set_xlabel("t (s)")
    ax.grid(True, alpha=0.3)
    if result["switch_step"] is not None:
        ax.axvline(t[result["switch_step"]], color="navy", linewidth=1.5,
                   linestyle="--", label="switch")
        ax.legend(fontsize=8)

    # ── (3,0-1) 3-D position trajectory ──────────────────────────────────
    ax3d = fig.add_subplot(gs[3, :], projection="3d")
    n_A  = result["switch_step"] if result["switch_step"] is not None else T
    ax3d.plot(states[:n_A, 0], states[:n_A, 1], states[:n_A, 2],
              color="steelblue", linewidth=1.5, label="Phase A")
    if result["switch_step"] is not None:
        ax3d.plot(states[n_A:, 0], states[n_A:, 1], states[n_A:, 2],
                  color="darkorange", linewidth=1.5, label="Phase B")
    ax3d.scatter(*states[0, :3],  color="green",  s=60, zorder=5, label="start")
    ax3d.scatter(*states[-1, :3], color="red",    s=60, zorder=5, label="end")
    ax3d.scatter(0, 0, 0,         color="black",  s=80, marker="*", label="equilibrium")
    ax3d.set_xlabel("x [m]"); ax3d.set_ylabel("y [m]"); ax3d.set_zlabel("z [m]")
    ax3d.set_title("3-D Position Trajectory")
    ax3d.legend(fontsize=8)

    # legend for phase shading
    patch_A = mpatches.Patch(color=_PHASE_COLORS[0], alpha=0.5, label="Controller A")
    patch_B = mpatches.Patch(color=_PHASE_COLORS[1], alpha=0.5, label="Controller B")
    fig.legend(handles=[patch_A, patch_B], loc="upper right", fontsize=9)

    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot -> {save_path}")


def plot_multi_trial_summary(all_results: list, save_path: str):
    """
    Summary figure across N trials:
      - attitude norms over time (phase A)
      - position norms over time (phase B)
      - switch times bar chart
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) attitude convergence during phase A
    ax = axes[0]
    ax.set_title("Attitude norm during Phase A")
    ax.set_xlabel("Step")
    ax.set_ylabel("||roll,pitch,yaw|| [rad]")

    for res in all_results:
        sw = res["switch_step"] if res["switch_step"] is not None else len(res["states"])
        att_norm = np.linalg.norm(res["states"][:sw, 3:6], axis=1)
        ax.plot(att_norm, alpha=0.6, linewidth=1)
    ax.grid(True, alpha=0.3)

    # (b) position convergence during phase B
    ax = axes[1]
    ax.set_title("Position norm during Phase B")
    ax.set_xlabel("Step (relative to switch)")
    ax.set_ylabel("||x,y,z|| [m]")

    for res in all_results:
        sw = res["switch_step"]
        if sw is None:
            continue
        pos_norm = np.linalg.norm(res["states"][sw:, 0:3], axis=1)
        ax.plot(pos_norm, alpha=0.6, linewidth=1)
    ax.grid(True, alpha=0.3)

    # (c) switch steps
    ax = axes[2]
    switch_steps = [r["switch_step"] if r["switch_step"] is not None else -1
                    for r in all_results]
    ax.bar(range(len(switch_steps)), switch_steps, color="steelblue", alpha=0.8)
    ax.set_title("Switch step per trial\n(-1 = never switched)")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Step")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Summary plot -> {save_path}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Sequential two-controller simulation tester"
    )

    # model paths
    p.add_argument("--model_A", type=str,
                   default="saved_models/two_controllers/attitude_stabiliser",
                   help="Path to Controller A checkpoint (without .zip)")
    p.add_argument("--model_B", type=str,
                   default="saved_models/two_controllers/position_stabiliser",
                   help="Path to Controller B checkpoint (without .zip)")

    # simulation
    p.add_argument("--dt",        type=float, default=0.01)
    p.add_argument("--max_steps", type=int,   default=2000,
                   help="Maximum simulation steps per trial")
    p.add_argument("--n_trials",  type=int,   default=5,
                   help="Number of simulation trials")
    p.add_argument("--random_init", action="store_true",
                   help="Use random initial states (same distribution as training)")

    # manual initial state (used when --random_init is not set)
    p.add_argument("--px",    type=float, default=3.0)
    p.add_argument("--py",    type=float, default=-2.0)
    p.add_argument("--pz",    type=float, default=5.0)
    p.add_argument("--roll",  type=float, default=0.5)
    p.add_argument("--pitch", type=float, default=-0.4)
    p.add_argument("--yaw",   type=float, default=1.0)
    p.add_argument("--vx",    type=float, default=1.0)
    p.add_argument("--vy",    type=float, default=0.5)
    p.add_argument("--vz",    type=float, default=-0.3)
    p.add_argument("--wx",    type=float, default=0.1)
    p.add_argument("--wy",    type=float, default=-0.05)
    p.add_argument("--wz",    type=float, default=0.0)

    # switch condition
    p.add_argument("--switch_angle_tol",  type=float, default=0.10,
                   help="Max |roll|/|pitch|/|yaw| to count as stable [rad]")
    p.add_argument("--switch_omega_tol",  type=float, default=0.15,
                   help="Max |wx|/|wy|/|wz| to count as stable [rad/s]")
    p.add_argument("--switch_hold_steps", type=int,   default=30,
                   help="Consecutive stable steps required before switch")

    # final equilibrium tolerances
    p.add_argument("--final_pos_tol", type=float, default=0.10)
    p.add_argument("--final_vel_tol", type=float, default=0.15)

    # training reset bounds (for --random_init)
    p.add_argument("--pos_bound",   type=float, default=10.0)
    p.add_argument("--angle_bound", type=float, default=0.7)
    p.add_argument("--vel_bound",   type=float, default=3.0)
    p.add_argument("--omega_bound", type=float, default=0.2)

    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results", "two_controllers_sim")
    os.makedirs(results_dir, exist_ok=True)

    # ── resolve model paths ───────────────────────────────────────────────
    def _resolve(path):
        if os.path.exists(path + ".zip"):
            return path
        full = os.path.join(script_dir, path)
        if os.path.exists(full + ".zip"):
            return full
        raise FileNotFoundError(
            f"Cannot find model at '{path}.zip' or '{full}.zip'\n"
            f"Train first with:  python train_two_controllers_3d.py"
        )

    path_A = _resolve(args.model_A)
    path_B = _resolve(args.model_B)

    print(f"Loading Controller A from: {path_A}.zip")
    model_A = SAC.load(path_A)
    print(f"Loading Controller B from: {path_B}.zip")
    model_B = SAC.load(path_B)

    # ── build initial states ──────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)

    if args.random_init:
        # same distribution as training (quadrotor3d_env.py)
        x_lo = np.array([-args.pos_bound,   -args.pos_bound,  0.0,
                          -args.angle_bound, -args.angle_bound, -np.pi,
                          -args.vel_bound,   -args.vel_bound,  -args.vel_bound,
                          -args.omega_bound, -args.omega_bound, -args.omega_bound],
                         dtype=np.float32)
        x_hi = np.array([args.pos_bound,    args.pos_bound,   args.pos_bound * 2.0,
                          args.angle_bound,  args.angle_bound, np.pi,
                          args.vel_bound,    args.vel_bound,   args.vel_bound,
                          args.omega_bound,  args.omega_bound, args.omega_bound],
                         dtype=np.float32)
        init_states = [
            (rng.random(12) * (x_hi - x_lo) + x_lo).astype(np.float32)
            for _ in range(args.n_trials)
        ]
        # wrap yaw
        for s in init_states:
            s[5] = wrap_to_pi(float(s[5]))
    else:
        # single manual state, repeated for n_trials
        manual = np.array([
            args.px,    args.py,    args.pz,
            args.roll,  args.pitch, args.yaw,
            args.vx,    args.vy,    args.vz,
            args.wx,    args.wy,    args.wz,
        ], dtype=np.float32)
        init_states = [manual.copy() for _ in range(args.n_trials)]

    # ── simulate ──────────────────────────────────────────────────────────
    all_results = []
    print(f"\nRunning {args.n_trials} trial(s) ...")
    print(f"Switch condition: |att| < {args.switch_angle_tol} rad  "
          f"|omega| < {args.switch_omega_tol} rad/s  "
          f"for {args.switch_hold_steps} consecutive steps\n")

    for trial_idx, init_state in enumerate(init_states):
        print(f"Trial {trial_idx+1}/{args.n_trials}")
        print(f"  Init: px={init_state[0]:.2f} py={init_state[1]:.2f} "
              f"pz={init_state[2]:.2f} | "
              f"roll={init_state[3]:.2f} pitch={init_state[4]:.2f} "
              f"yaw={init_state[5]:.2f}")

        result = run_simulation(
            model_A, model_B,
            init_state      = init_state,
            dt              = args.dt,
            max_steps       = args.max_steps,
            switch_angle_tol  = args.switch_angle_tol,
            switch_omega_tol  = args.switch_omega_tol,
            switch_hold_steps = args.switch_hold_steps,
            final_pos_tol   = args.final_pos_tol,
            final_vel_tol   = args.final_vel_tol,
        )

        sw = result["switch_step"]
        eq = result["eq_step"]
        T  = len(result["states"])

        print(f"  Switch to B : step {sw} ({sw*args.dt:.2f}s)"
              if sw is not None else "  Switch to B : NEVER (attitude not stabilised)")
        print(f"  Equilibrium : step {eq} ({eq*args.dt:.2f}s)"
              if eq is not None else "  Equilibrium : NOT REACHED")
        print(f"  Total steps : {T}  ({T*args.dt:.2f}s)")
        print(f"  Final state : pos=({result['states'][-1,0]:.3f}, "
              f"{result['states'][-1,1]:.3f}, {result['states'][-1,2]:.3f})  "
              f"att=({np.degrees(result['states'][-1,3]):.1f}°, "
              f"{np.degrees(result['states'][-1,4]):.1f}°, "
              f"{np.degrees(result['states'][-1,5]):.1f}°)")

        all_results.append(result)

        title = (f"Trial {trial_idx+1} | "
                 f"switch@{sw}steps | "
                 f"{'eq reached@'+str(eq)+'steps' if eq else 'eq NOT reached'}")
        plot_path = os.path.join(results_dir, f"trial_{trial_idx+1:02d}.png")
        plot_trial(result, save_path=plot_path, title=title)

    # ── summary ───────────────────────────────────────────────────────────
    n_switched = sum(1 for r in all_results if r["switch_step"] is not None)
    n_reached  = sum(1 for r in all_results if r["reached_eq"])

    print(f"\n{'='*55}")
    print(f"Summary  ({args.n_trials} trials)")
    print(f"  Controllers switched : {n_switched}/{args.n_trials}")
    print(f"  Equilibrium reached  : {n_reached}/{args.n_trials}")
    if n_switched > 0:
        sw_times = [r["switch_step"] * args.dt
                    for r in all_results if r["switch_step"] is not None]
        print(f"  Mean switch time     : {np.mean(sw_times):.2f}s "
              f"± {np.std(sw_times):.2f}s")
    if n_reached > 0:
        eq_times = [r["eq_step"] * args.dt
                    for r in all_results if r["eq_step"] is not None]
        print(f"  Mean eq. reach time  : {np.mean(eq_times):.2f}s "
              f"± {np.std(eq_times):.2f}s")
    print(f"{'='*55}")

    if args.n_trials > 1:
        plot_multi_trial_summary(
            all_results,
            save_path=os.path.join(results_dir, "summary.png"),
        )


if __name__ == "__main__":
    main()
