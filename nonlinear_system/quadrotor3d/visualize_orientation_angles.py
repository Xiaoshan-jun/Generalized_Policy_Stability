#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize how roll/pitch/yaw angles change quadrotor orientation."""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from neural_network_lyapunov.geometry_transform import rpy2rotmat


def setup_ax(ax, title):
    lim = 1.25
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.view_init(elev=22, azim=42)


def draw_world_axes(ax):
    ax.quiver(0, 0, 0, 1, 0, 0, color="gray", linewidth=1.5, alpha=0.7)
    ax.quiver(0, 0, 0, 0, 1, 0, color="gray", linewidth=1.5, alpha=0.7)
    ax.quiver(0, 0, 0, 0, 0, 1, color="gray", linewidth=1.5, alpha=0.7)


def draw_drone_orientation(ax, roll, pitch, yaw, color, label):
    R = rpy2rotmat(np.array([roll, pitch, yaw], dtype=np.float64))
    arm_len = 0.8

    # Drone body axes in world frame
    ex = R @ np.array([1.0, 0.0, 0.0])
    ey = R @ np.array([0.0, 1.0, 0.0])
    ez = R @ np.array([0.0, 0.0, 1.0])

    # Main arms
    p1, p2 = arm_len * ex, -arm_len * ex
    p3, p4 = arm_len * ey, -arm_len * ey
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, linewidth=2.0)
    ax.plot([p3[0], p4[0]], [p3[1], p4[1]], [p3[2], p4[2]], color=color, linewidth=2.0)

    # Body z-axis direction
    ax.quiver(0, 0, 0, ez[0], ez[1], ez[2], color=color, linewidth=2.0, alpha=0.9)
    ax.scatter([0], [0], [0], color=color, s=15, label=label)


def main():
    out_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "results", "quadrotor3d", "orientation"
    )
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(18, 6))
    degrees = [-60, -30, 0, 30, 60]
    radians = [np.deg2rad(d) for d in degrees]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]

    # Roll effect
    ax_roll = fig.add_subplot(1, 3, 1, projection="3d")
    setup_ax(ax_roll, "Roll Changes Orientation (pitch=yaw=0)")
    draw_world_axes(ax_roll)
    for d, r, c in zip(degrees, radians, colors):
        draw_drone_orientation(ax_roll, roll=r, pitch=0.0, yaw=0.0, color=c, label=f"{d} deg")
    ax_roll.legend(loc="upper left", fontsize=8)

    # Pitch effect
    ax_pitch = fig.add_subplot(1, 3, 2, projection="3d")
    setup_ax(ax_pitch, "Pitch Changes Orientation (roll=yaw=0)")
    draw_world_axes(ax_pitch)
    for d, r, c in zip(degrees, radians, colors):
        draw_drone_orientation(ax_pitch, roll=0.0, pitch=r, yaw=0.0, color=c, label=f"{d} deg")
    ax_pitch.legend(loc="upper left", fontsize=8)

    # Yaw effect
    ax_yaw = fig.add_subplot(1, 3, 3, projection="3d")
    setup_ax(ax_yaw, "Yaw Changes Orientation (roll=pitch=0)")
    draw_world_axes(ax_yaw)
    for d, r, c in zip(degrees, radians, colors):
        draw_drone_orientation(ax_yaw, roll=0.0, pitch=0.0, yaw=r, color=c, label=f"{d} deg")
    ax_yaw.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "angle_orientation_relationship.png")
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

