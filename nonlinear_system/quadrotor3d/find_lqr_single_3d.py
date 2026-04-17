#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute a single LQR gain by linearizing only at hover equilibrium."""

import argparse
import json
import os
import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are

from quadrotor3d import Quadrotor


def parse_args():
    p = argparse.ArgumentParser(
        description="Find one LQR gain for quadrotor3d at hover equilibrium only"
    )
    p.add_argument("--dt", type=float, default=0.01, help="Discrete-time step for DARE")
    p.add_argument("--q_pos", type=float, default=25.0)
    p.add_argument("--q_angle", type=float, default=45.0)
    p.add_argument("--q_vel", type=float, default=1.0)
    p.add_argument("--q_omega", type=float, default=0.25)
    p.add_argument("--r_thrust", type=float, default=0.2)
    p.add_argument(
        "--save_dir",
        type=str,
        default="saved_models/lqr_single",
        help="Directory to save A/B/Q/R and single LQR gains",
    )
    return p.parse_args()


def build_qr(args):
    Q = np.diag(
        [
            args.q_pos,
            args.q_pos,
            args.q_pos,
            args.q_angle,
            args.q_angle,
            args.q_angle,
            args.q_vel,
            args.q_vel,
            args.q_vel,
            args.q_omega,
            args.q_omega,
            args.q_omega,
        ]
    )
    R = args.r_thrust * np.eye(4)
    return Q, R


def solve_lqr(A, B, Q, R, dt):
    P_c = solve_continuous_are(A, B, Q, R)
    K_c = np.linalg.solve(R, B.T @ P_c)

    A_d = np.eye(A.shape[0]) + dt * A
    B_d = dt * B
    P_d = solve_discrete_are(A_d, B_d, Q, R)
    K_d = np.linalg.solve(B_d.T @ P_d @ B_d + R, B_d.T @ P_d @ A_d)
    return K_c, K_d, A_d, B_d


def main():
    args = parse_args()
    quad = Quadrotor(dtype=None)

    # Linearize only at hover equilibrium.
    x_eq = np.zeros(12)
    u_eq = np.ones(4) * quad.hover_thrust
    A, B = quad.dynamics_gradient(x_eq, u_eq)
    Q, R = build_qr(args)
    K_c, K_d, A_d, B_d = solve_lqr(A, B, Q, R, args.dt)

    np.set_printoptions(precision=6, suppress=True)
    print(f"hover_thrust = {quad.hover_thrust:.9f}")
    print("x_eq = zeros(12)")
    print(f"u_eq = hover_thrust * ones(4) = {u_eq}")
    print("\nK_continuous (u = u_eq - Kc (x - x_eq)):\n", K_c)
    print(f"\nK_discrete_dt_{args.dt:g} (u = u_eq - Kd (x_k - x_eq)):\n", K_d)

    os.makedirs(args.save_dir, exist_ok=True)
    np.save(os.path.join(args.save_dir, "A.npy"), A)
    np.save(os.path.join(args.save_dir, "B.npy"), B)
    np.save(os.path.join(args.save_dir, "A_d.npy"), A_d)
    np.save(os.path.join(args.save_dir, "B_d.npy"), B_d)
    np.save(os.path.join(args.save_dir, "Q.npy"), Q)
    np.save(os.path.join(args.save_dir, "R.npy"), R)
    np.save(os.path.join(args.save_dir, "K_continuous.npy"), K_c)
    np.save(os.path.join(args.save_dir, f"K_discrete_dt_{args.dt:g}.npy"), K_d)

    metadata = {
        "mode": "single_gain_hover_linearization_only",
        "dt": float(args.dt),
        "hover_thrust": float(quad.hover_thrust),
        "x_eq": x_eq.tolist(),
        "u_eq": u_eq.tolist(),
        "save_dir": os.path.abspath(args.save_dir),
    }
    with open(os.path.join(args.save_dir, "lqr_single_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved single-gain LQR artifacts to: {os.path.abspath(args.save_dir)}")


if __name__ == "__main__":
    main()
