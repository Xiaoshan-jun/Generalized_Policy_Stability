"""
Visualization for the 2D quadrotor controller.

Layout
------
Left  : animated 2D view — drone body, orientation, thrust arrows, trajectory trail
Right : scrolling time-series of theta, theta_dot, u_left, u_right, V(s) with a
        moving cursor that stays in sync with the animation.

Usage
-----
Run directly to visualize the saved two-stage pitch controller::

    python visualize_quadrotor2d.py

Or import and call ``animate()`` with your own states/actions/values arrays.
"""

import os
import sys

import matplotlib
matplotlib.use("TkAgg")          # change to "Qt5Agg" if TkAgg is unavailable
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch

# ── make the local modules importable ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from quadrotor2d_env import Quadrotor2DEnv  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Value-function query
# ─────────────────────────────────────────────────────────────────────────────

def get_value(model, obs: np.ndarray) -> float:
    """
    Query the model's value estimate V(s) for a single observation.

    - PPO  : reads the value head directly.
    - SAC  : approximates V(s) ≈ min Q(s, π(s)) using the deterministic actor.
    """
    from stable_baselines3 import SAC

    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(model.device)
    with torch.no_grad():
        if isinstance(model, SAC):
            action_t = model.actor._predict(obs_t, deterministic=True)
            q1, q2 = model.critic(obs_t, action_t)
            return torch.min(q1, q2).item()
        else:
            return model.policy.predict_values(obs_t).item()


# ─────────────────────────────────────────────────────────────────────────────
# Rollout helper
# ─────────────────────────────────────────────────────────────────────────────

def rollout(model, env: Quadrotor2DEnv, seed: int = 0):
    """
    Run one episode and return arrays of states, actions, and value estimates.

    Returns
    -------
    states  : (T, 6)  [x, z, theta, vx, vz, theta_dot]
    actions : (T, 2)  [u_left, u_right]
    values  : (T,)    V(s) at each state
    dt      : float
    """
    obs, _ = env.reset(seed=seed)
    states, actions, values = [obs.copy()], [], [get_value(model, obs)]

    while True:
        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32)
        obs, _, terminated, truncated, _ = env.step(action)
        states.append(obs.copy())
        actions.append(action.copy())
        values.append(get_value(model, obs))
        if terminated or truncated:
            break

    # Align: drop the last state so index i = state after action i.
    states  = np.asarray(states[:-1],  dtype=np.float32)  # (T, 6)
    actions = np.asarray(actions,       dtype=np.float32)  # (T, 2)
    values  = np.asarray(values[:-1],   dtype=np.float32)  # (T,)
    return states, actions, values, env.dt


# ─────────────────────────────────────────────────────────────────────────────
# Drone drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

ARM   = 0.25   # rotor arm half-length (matches Quadrotor2D.length)
ROTOR = 0.06   # rotor disc radius for drawing


def _rotor_endpoints(cx, cz, theta):
    """Return (left_tip, right_tip) in world frame."""
    c, s = np.cos(theta), np.sin(theta)
    return (cx - ARM * c, cz - ARM * s), (cx + ARM * c, cz + ARM * s)


def _make_drone_artists(ax, color="steelblue", thrust_color="tomato"):
    body,   = ax.plot([], [], lw=3, color=color, solid_capstyle="round", zorder=3)
    rotor_l = plt.Circle((0, 0), ROTOR, color=color, alpha=0.6, zorder=4)
    rotor_r = plt.Circle((0, 0), ROTOR, color=color, alpha=0.6, zorder=4)
    ax.add_patch(rotor_l)
    ax.add_patch(rotor_r)
    arrow_l = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                          arrowprops=dict(arrowstyle="-|>", color=thrust_color, lw=2),
                          zorder=5)
    arrow_r = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                          arrowprops=dict(arrowstyle="-|>", color=thrust_color, lw=2),
                          zorder=5)
    cm_dot, = ax.plot([], [], "o", color=color, ms=6, zorder=6)
    trail,  = ax.plot([], [], "--", color=color, alpha=0.35, lw=1, zorder=2)
    return body, rotor_l, rotor_r, arrow_l, arrow_r, cm_dot, trail


def _update_drone(artists, cx, cz, theta, u_left, u_right,
                  trail_x, trail_z, u_scale=0.04):
    body, rotor_l, rotor_r, arrow_l, arrow_r, cm_dot, trail = artists
    left, right = _rotor_endpoints(cx, cz, theta)

    body.set_data([left[0], right[0]], [left[1], right[1]])
    rotor_l.set_center(left)
    rotor_r.set_center(right)
    cm_dot.set_data([cx], [cz])
    trail.set_data(trail_x, trail_z)

    # Thrust arrows in body-frame "up" direction: (-sin θ, cos θ) in world.
    bx, bz = -np.sin(theta), np.cos(theta)
    for arrow, (rx, rz), thrust in [(arrow_l, left, u_left),
                                    (arrow_r, right, u_right)]:
        scale = thrust * u_scale
        arrow.set_position((rx, rz))
        arrow.xy    = (rx + bx * scale, rz + bz * scale)
        arrow.xyann = (rx, rz)


# ─────────────────────────────────────────────────────────────────────────────
# Main animation function
# ─────────────────────────────────────────────────────────────────────────────

def animate(states: np.ndarray, actions: np.ndarray, dt: float,
            values: np.ndarray = None,
            title: str = "Quadrotor 2-D Controller",
            trail_len: int = 40,
            interval_ms: int = 30,
            save_path: str = None):
    """
    Build and display (or save) the animation.

    Parameters
    ----------
    states     : (T, 6)  [x, z, theta, vx, vz, theta_dot]
    actions    : (T, 2)  [u_left, u_right]
    dt         : float
    values     : (T,)    optional value-function estimates V(s)
    title      : str
    trail_len  : int     number of past positions shown in the trail
    interval_ms: int     milliseconds between frames
    save_path  : str     save to .mp4/.gif if provided, else show interactively
    """
    T = len(states)
    time    = np.arange(T) * dt
    x_pos   = states[:, 0]
    z_pos   = states[:, 1]
    theta   = states[:, 2]
    th_dot  = states[:, 5]
    u_left  = actions[:, 0]
    u_right = actions[:, 1]

    n_ts_rows = 5 if values is not None else 4

    # ── figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    gs = fig.add_gridspec(n_ts_rows, 2,
                          left=0.07, right=0.97,
                          top=0.92, bottom=0.07,
                          wspace=0.38, hspace=0.6)

    ax_drone = fig.add_subplot(gs[:, 0])          # full left column
    ax_th    = fig.add_subplot(gs[0, 1])
    ax_thd   = fig.add_subplot(gs[1, 1])
    ax_ul    = fig.add_subplot(gs[2, 1])
    ax_ur    = fig.add_subplot(gs[3, 1])

    # ── drone axes ───────────────────────────────────────────────────────────
    pad_x = max(np.ptp(x_pos) * 0.3, 0.5)
    pad_z = max(np.ptp(z_pos) * 0.3, 0.5)
    ax_drone.set_xlim(x_pos.min() - pad_x - ARM, x_pos.max() + pad_x + ARM)
    ax_drone.set_ylim(z_pos.min() - pad_z - ARM, z_pos.max() + pad_z + ARM)
    ax_drone.set_aspect("equal")
    ax_drone.set_xlabel("x  (m)")
    ax_drone.set_ylabel("z  (m)")
    ax_drone.set_title("Position & Pitch")
    ax_drone.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax_drone.grid(True, alpha=0.25)
    ax_drone.plot(0, 0, "+", ms=10, color="green", zorder=1, label="equilibrium")
    ax_drone.legend(fontsize=8, loc="upper right")

    drone_artists = _make_drone_artists(ax_drone)

    info_text = ax_drone.text(
        0.02, 0.97, "", transform=ax_drone.transAxes,
        va="top", ha="left", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )

    # ── time-series axes ─────────────────────────────────────────────────────
    ts_axes   = [ax_th,       ax_thd,          ax_ul,           ax_ur]
    ts_data   = [theta,       th_dot,           u_left,          u_right]
    ts_labels = ["θ  (rad)", "θ̇  (rad/s)", "u_left  (N)", "u_right  (N)"]
    ts_colors = ["steelblue", "darkorange",     "tomato",        "mediumpurple"]

    if values is not None:
        ax_val = fig.add_subplot(gs[4, 1])
        ts_axes.append(ax_val)
        ts_data.append(values)
        ts_labels.append("V(s)")
        ts_colors.append("seagreen")

    for ax, data, label, color in zip(ts_axes, ts_data, ts_labels, ts_colors):
        ax.plot(time, data, color=color, lw=1.2)
        ax.set_ylabel(label, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(time[0], time[-1])

    ax_th.axhline(0,  color="gray", lw=0.8, ls="--")
    ax_thd.axhline(0, color="gray", lw=0.8, ls="--")
    if values is not None:
        ax_val.axhline(0, color="gray", lw=0.8, ls="--")
        ax_val.set_xlabel("time  (s)", fontsize=8)
    else:
        ax_ur.set_xlabel("time  (s)", fontsize=8)

    # Highlight the value-function panel to make it visually distinct.
    if values is not None:
        ax_val.set_facecolor("#f0fff4")
        ax_val.spines["left"].set_color("seagreen")
        ax_val.spines["left"].set_linewidth(1.5)

    # vertical cursor on every time-series plot
    cursors = [ax.axvline(time[0], color="black", lw=1, ls=":")
               for ax in ts_axes]

    # moving dot on the value curve so the current value is easy to read
    val_dot = None
    if values is not None:
        val_dot, = ax_val.plot([], [], "o", color="seagreen", ms=5, zorder=5)

    # ── animation update ─────────────────────────────────────────────────────
    def update(i):
        t_start = max(0, i - trail_len)
        _update_drone(
            drone_artists,
            x_pos[i], z_pos[i], theta[i],
            u_left[i], u_right[i],
            x_pos[t_start:i + 1], z_pos[t_start:i + 1],
        )

        val_str = f"\nV(s) = {values[i]:+.1f}" if values is not None else ""
        info_text.set_text(
            f"t = {time[i]:.2f} s\n"
            f"x = {x_pos[i]:+.3f} m\n"
            f"z = {z_pos[i]:+.3f} m\n"
            f"θ = {np.degrees(theta[i]):+.1f}°\n"
            f"θ̇ = {th_dot[i]:+.3f} rad/s"
            + val_str
        )

        for cursor in cursors:
            cursor.set_xdata([time[i], time[i]])

        artists_out = (*drone_artists, info_text, *cursors)
        if val_dot is not None:
            val_dot.set_data([time[i]], [values[i]])
            artists_out = (*artists_out, val_dot)

        return artists_out

    anim = FuncAnimation(fig, update, frames=T,
                         interval=interval_ms, blit=True)

    if save_path:
        anim.save(save_path, fps=int(1000 / interval_ms))
        print(f"Saved to {save_path}")
    else:
        plt.show()

    return anim


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    from stable_baselines3 import SAC, PPO

    here = os.path.dirname(__file__)

    for algo_cls, algo_name in [(SAC, "sac"), (PPO, "ppo")]:
        model_path = os.path.join(
            here, "saved_models", algo_name, "two_stage",
            "quadrotor_pitch_controller.zip",
        )
        if os.path.exists(model_path):
            print(f"Loading {algo_name.upper()} model from: {model_path}")
            model = algo_cls.load(model_path)
            break
    else:
        raise FileNotFoundError(
            "No saved model found. Train first with rl_two_stage_baseline.py."
        )

    env = Quadrotor2DEnv(dt=0.01, max_time=4.0)

    print("Running rollout …")
    states, actions, values, dt = rollout(model, env, seed=42)
    print(f"Episode length: {len(states)} steps  ({len(states) * dt:.2f} s)")

    animate(states, actions, dt, values=values,
            title=f"Quadrotor 2-D — Pitch Controller ({algo_name.upper()})")


if __name__ == "__main__":
    main()
