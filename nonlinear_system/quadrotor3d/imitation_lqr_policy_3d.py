#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a policy network to imitate single-gain LQR for Quadrotor3D."""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from quadrotor3d import Quadrotor
from network.PolicyNet import PolicyNet


def parse_args():
    p = argparse.ArgumentParser(description="Imitate single-gain LQR policy for Quadrotor3D")
    p.add_argument(
        "--k_path",
        type=str,
        default="saved_models/lqr_single/K_discrete_dt_0.01.npy",
        help="Path to single discrete LQR gain K with shape [4,12]",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_samples", type=int, default=200000)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--pos_bound", type=float, default=5.0)
    p.add_argument("--angle_bound", type=float, default=1.0)
    p.add_argument("--vel_bound", type=float, default=3.0)
    p.add_argument("--omega_bound", type=float, default=1.5)
    p.add_argument("--u_min_factor", type=float, default=0.0)
    p.add_argument("--u_max_factor", type=float, default=2.5)
    p.add_argument(
        "--save_dir",
        type=str,
        default="saved_models/imitation_lqr_single",
        help="Directory to save checkpoints and dataset stats",
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


def sample_states(rng: np.random.Generator, n: int, cfg) -> np.ndarray:
    x = np.zeros((n, 12), dtype=np.float32)
    x[:, 0] = rng.uniform(-cfg.pos_bound, cfg.pos_bound, size=n)
    x[:, 1] = rng.uniform(-cfg.pos_bound, cfg.pos_bound, size=n)
    x[:, 2] = rng.uniform(0.0, 2.0 * cfg.pos_bound, size=n)
    x[:, 3] = rng.uniform(-cfg.angle_bound, cfg.angle_bound, size=n)
    x[:, 4] = rng.uniform(-cfg.angle_bound, cfg.angle_bound, size=n)
    x[:, 5] = rng.uniform(-np.pi, np.pi, size=n)
    x[:, 6:9] = rng.uniform(-cfg.vel_bound, cfg.vel_bound, size=(n, 3))
    x[:, 9:12] = rng.uniform(-cfg.omega_bound, cfg.omega_bound, size=(n, 3))
    return x


def build_expert_actions(states: np.ndarray, Kd: np.ndarray, u_eq: np.ndarray, u_lo: np.ndarray, u_up: np.ndarray):
    # Expert policy: u = clip(u_eq - Kx)
    u = u_eq[None, :] - states @ Kd.T
    u = np.clip(u, u_lo[None, :], u_up[None, :])
    return u.astype(np.float32)


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for x_b, u_b in loader:
            x_b = x_b.to(device)
            u_b = u_b.to(device)
            pred = model(x_b)
            loss = torch.mean((pred - u_b) ** 2)
            total += float(loss.item()) * x_b.shape[0]
            count += int(x_b.shape[0])
    return total / max(1, count)


def main():
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.save_dir, exist_ok=True)

    k_path = resolve_path(cfg.k_path)
    if not os.path.exists(k_path):
        raise FileNotFoundError(f"K not found: {k_path}")
    Kd = np.load(k_path).astype(np.float32)
    if Kd.shape != (4, 12):
        raise ValueError(f"K must have shape [4,12], got {Kd.shape}")

    quad = Quadrotor(dtype=None)
    u_eq = np.ones(4, dtype=np.float32) * float(quad.hover_thrust)
    u_lo = np.ones(4, dtype=np.float32) * float(cfg.u_min_factor * quad.hover_thrust)
    u_up = np.ones(4, dtype=np.float32) * float(cfg.u_max_factor * quad.hover_thrust)

    states = sample_states(rng, cfg.num_samples, cfg)
    actions = build_expert_actions(states, Kd, u_eq, u_lo, u_up)

    n_val = int(cfg.val_split * cfg.num_samples)
    idx = rng.permutation(cfg.num_samples)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    x_train = states[tr_idx]
    u_train = actions[tr_idx]
    x_val = states[val_idx]
    u_val = actions[val_idx]

    # Normalize state input; keep action in thrust units.
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train_n = (x_train - x_mean) / x_std
    x_val_n = (x_val - x_mean) / x_std

    train_ds = TensorDataset(
        torch.from_numpy(x_train_n).float(),
        torch.from_numpy(u_train).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_val_n).float(),
        torch.from_numpy(u_val).float(),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = PolicyNet(12, cfg.hidden_dim, 4, n_layers=cfg.n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_path = os.path.join(cfg.save_dir, "imitation_policy_lqr_single_best.pth")
    final_path = os.path.join(cfg.save_dir, "imitation_policy_lqr_single_final.pth")
    stats_path = os.path.join(cfg.save_dir, "imitation_summary.txt")
    data_path = os.path.join(cfg.save_dir, "lqr_single_imitation_dataset.npz")

    for ep in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for x_b, u_b in train_loader:
            x_b = x_b.to(device)
            u_b = u_b.to(device)
            pred = model(x_b)
            loss = torch.mean((pred - u_b) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * x_b.shape[0]
            count += int(x_b.shape[0])
        tr_mse = total / max(1, count)
        val_mse = evaluate(model, val_loader, device)
        if val_mse < best_val:
            best_val = val_mse
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "x_mean": x_mean.astype(np.float32),
                    "x_std": x_std.astype(np.float32),
                    "k_path": k_path,
                },
                best_path,
            )
        print(f"[epoch {ep:03d}] train_mse={tr_mse:.8f} val_mse={val_mse:.8f} best_val={best_val:.8f}")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "x_mean": x_mean.astype(np.float32),
            "x_std": x_std.astype(np.float32),
            "k_path": k_path,
        },
        final_path,
    )

    np.savez_compressed(
        data_path,
        x_train=x_train,
        u_train=u_train,
        x_val=x_val,
        u_val=u_val,
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
        Kd=Kd,
        u_eq=u_eq,
        u_lo=u_lo,
        u_up=u_up,
    )

    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"k_path: {k_path}\n")
        f.write(f"num_samples: {cfg.num_samples}\n")
        f.write(f"val_split: {cfg.val_split}\n")
        f.write(f"epochs: {cfg.epochs}\n")
        f.write(f"best_val_mse: {best_val:.10f}\n")
        f.write(f"best_model: {best_path}\n")
        f.write(f"final_model: {final_path}\n")
        f.write(f"dataset: {data_path}\n")

    print("\n=== Imitation LQR(single) Done ===")
    print(f"Best model: {best_path}")
    print(f"Final model: {final_path}")
    print(f"Summary:    {stats_path}")
    print(f"Dataset:    {data_path}")


if __name__ == "__main__":
    main()
