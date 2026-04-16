#!/usr/bin/env python3
"""
Two-stage RL training for the 2D quadrotor.

Stage 1 trains an attitude controller that primarily stabilizes pitch angle.
Stage 2 trains a position controller that stabilizes x/z once the pitch angle
is already small.
"""

import os
from dataclasses import dataclass

import matplotlib
import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

from quadrotor2d_env import Quadrotor2DEnv

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class StageConfig:
    name: str
    algo: str
    time_steps: int
    n_envs: int = 1
    seed: int = 0
    dt: float = 0.01
    max_time: float = 2.0
    device: str = "auto"
    verbose: int = 0
    log_interval: int = 10
    normalize_reward: bool = True


class PitchStabilizationEnv(Quadrotor2DEnv):
    def __init__(self, dt=0.01, max_time=2.0):
        super().__init__(dt=dt, max_time=max_time)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if hasattr(self, "np_random") and self.np_random is not None:
            rand = torch.tensor(self.np_random.random(6), dtype=self.dtype)
        else:
            rand = torch.rand(6, dtype=self.dtype)

        self.x_current = rand * (self.x_up - self.x_lo) + self.x_lo
        self.step_count = 0
        return self.x_current.detach().cpu().numpy().astype(np.float32), {}

    def reached_equilibrium(self, obs) -> bool:
        obs = np.asarray(obs, dtype=np.float32)
        return bool(
            abs(float(obs[2])) < 0.03
            and abs(float(obs[5])) < 0.15
            and abs(float(obs[0])) < 0.5
            and abs(float(obs[1])) < 0.5
            and abs(float(obs[3])) < 2.0
            and abs(float(obs[4])) < 2.0
        )

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)

        action_t = torch.as_tensor(np.asarray(action, dtype=np.float32), dtype=self.dtype)
        act_delta = action_t - self.act_equ
        state = torch.as_tensor(obs, dtype=self.dtype)

        theta = state[2]
        theta_dot = state[5]
        x = state[0]
        z = state[1]
        vx = state[3]
        vz = state[4]
        success = self.reached_equilibrium(obs)

        reward = -(
            80.0 * theta.pow(2).item()
            + 12.0 * theta_dot.pow(2).item()
            + 2.0 * x.pow(2).item()
            + 2.0 * z.pow(2).item()
            + 0.3 * vx.pow(2).item()
            + 0.3 * vz.pow(2).item()
            + 0.15 * act_delta.pow(2).sum().item()
            + 0.25 * (action_t[0] - action_t[1]).pow(2).item()
        )
        if success:
            reward += 50.0

        info["reached_equilibrium"] = success
        info["stage_name"] = "pitch"
        terminated = bool(success or terminated or torch.abs(theta).item() > float(self.x_up[2]))
        return obs, float(reward), terminated, truncated, info


class PositionStabilizationEnv(Quadrotor2DEnv):
    def __init__(self, dt=0.01, max_time=2.0, small_angle_limit=0.12):
        super().__init__(dt=dt, max_time=max_time)
        self.small_angle_limit = float(small_angle_limit)
        self.reset_x_lo = torch.tensor(
            [
                float(self.x_lo[0]),
                float(self.x_lo[1]),
                -self.small_angle_limit,
                float(self.x_lo[3]),
                float(self.x_lo[4]),
                -0.35,
            ],
            dtype=self.dtype,
        )
        self.reset_x_up = torch.tensor(
            [
                float(self.x_up[0]),
                float(self.x_up[1]),
                self.small_angle_limit,
                float(self.x_up[3]),
                float(self.x_up[4]),
                0.35,
            ],
            dtype=self.dtype,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if hasattr(self, "np_random") and self.np_random is not None:
            rand = torch.tensor(self.np_random.random(6), dtype=self.dtype)
        else:
            rand = torch.rand(6, dtype=self.dtype)

        self.x_current = rand * (self.reset_x_up - self.reset_x_lo) + self.reset_x_lo
        self.step_count = 0
        return self.x_current.detach().cpu().numpy().astype(np.float32), {}

    def reached_equilibrium(self, obs) -> bool:
        obs = np.asarray(obs, dtype=np.float32)
        return bool(
            abs(float(obs[0])) < 0.05
            and abs(float(obs[1])) < 0.05
            and abs(float(obs[2])) < 0.03
            and abs(float(obs[3])) < 0.12
            and abs(float(obs[4])) < 0.12
            and abs(float(obs[5])) < 0.10
        )

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)

        action_t = torch.as_tensor(np.asarray(action, dtype=np.float32), dtype=self.dtype)
        act_delta = action_t - self.act_equ
        state = torch.as_tensor(obs, dtype=self.dtype)

        x = state[0]
        z = state[1]
        theta = state[2]
        vx = state[3]
        vz = state[4]
        theta_dot = state[5]
        success = self.reached_equilibrium(obs)

        reward = -(
            20.0 * x.pow(2).item()
            + 30.0 * z.pow(2).item()
            + 4.0 * vx.pow(2).item()
            + 5.0 * vz.pow(2).item()
            + 8.0 * theta.pow(2).item()
            + 2.0 * theta_dot.pow(2).item()
            + 0.1 * act_delta.pow(2).sum().item()
        )
        if success:
            reward += 50.0

        info["reached_equilibrium"] = success
        info["stage_name"] = "position"
        terminated = bool(success or terminated or torch.abs(theta).item() > self.small_angle_limit * 2.0)
        return obs, float(reward), terminated, truncated, info


class TwoStageQuadrotorPolicy:
    def __init__(self, pitch_model, position_model, theta_threshold=0.08):
        self.pitch_model = pitch_model
        self.position_model = position_model
        self.theta_threshold = float(theta_threshold)

    def predict(self, obs, deterministic=True):
        obs = np.asarray(obs, dtype=np.float32)
        if abs(float(obs[2])) > self.theta_threshold:
            return self.pitch_model.predict(obs, deterministic=deterministic)
        return self.position_model.predict(obs, deterministic=deterministic)


class EquilibriumPrintCallback(BaseCallback):
    def __init__(self, stage_name: str, print_every_episodes: int = 20, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.stage_name = stage_name
        self.print_every_episodes = int(print_every_episodes)
        self.success_count = 0
        self.episode_count = 0
        self.episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        if dones is None:
            dones = []

        for env_idx, info in enumerate(infos):
            done = bool(dones[env_idx]) if env_idx < len(dones) else False
            if not done:
                continue

            self.episode_count += 1
            reward = None
            if isinstance(info.get("episode"), dict):
                reward = info["episode"].get("r")
                if reward is not None:
                    self.episode_rewards.append(float(reward))

            reached_equilibrium = bool(info.get("reached_equilibrium", False))
            if reached_equilibrium:
                self.success_count += 1

            if (
                self.print_every_episodes > 0
                and self.episode_count % self.print_every_episodes == 0
                and len(self.episode_rewards) > 0
            ):
                window = self.episode_rewards[-self.print_every_episodes :]
                avg_reward = sum(window) / len(window)
                success_rate = self.success_count / float(self.episode_count)
                print(
                    f"[{self.stage_name}] episodes {self.episode_count - len(window) + 1}-{self.episode_count}: "
                    f"avg_reward={avg_reward:.3f}, success_rate={success_rate:.3f}"
                )

        return True


def resolve_device(device: str) -> str:
    device = device.lower().strip()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    return device


def build_model(algo, env, seed, device, verbose):
    algo = algo.lower().strip()
    resolved_device = resolve_device(device)
    print(f"Using device: {resolved_device}")

    if algo == "ppo":
        return PPO(
            "MlpPolicy",
            env,
            verbose=verbose,
            device=resolved_device,
            seed=seed,
            batch_size=256,
            gamma=0.99,
            learning_rate=3e-4,
        )

    if algo == "sac":
        return SAC(
            "MlpPolicy",
            env,
            verbose=verbose,
            device=resolved_device,
            seed=seed,
            buffer_size=200_000,
            batch_size=256,
            gamma=0.99,
            learning_rate=3e-4,
            train_freq=1,
            gradient_steps=1,
        )

    raise ValueError("algo must be 'ppo' or 'sac'")


def train_stage(stage_cfg: StageConfig, env_cls, save_name):
    def _make_env():
        env = env_cls(dt=stage_cfg.dt, max_time=stage_cfg.max_time)
        return Monitor(env)

    env = make_vec_env(_make_env, n_envs=stage_cfg.n_envs, seed=stage_cfg.seed)
    if stage_cfg.normalize_reward:
        env = VecNormalize(
            env,
            training=True,
            norm_obs=False,
            norm_reward=True,
            clip_reward=10.0,
        )
    model = build_model(
        stage_cfg.algo,
        env,
        stage_cfg.seed,
        stage_cfg.device,
        stage_cfg.verbose,
    )
    equilibrium_callback = EquilibriumPrintCallback(
        stage_name=stage_cfg.name,
        print_every_episodes=20,
    )
    model.learn(
        total_timesteps=int(stage_cfg.time_steps),
        log_interval=stage_cfg.log_interval,
        callback=equilibrium_callback,
    )

    save_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "saved_models", stage_cfg.algo, "two_stage")
    )
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    model.save(save_path)
    if isinstance(env, VecNormalize):
        env.save(os.path.join(save_dir, f"{save_name}_vecnormalize.pkl"))
    print(f"Saved {stage_cfg.name} controller to: {save_path}.zip")
    return model, save_path


def rollout_and_plot(
    policy,
    num_steps=400,
    seed=0,
    dt=0.01,
    max_time=4.0,
    filename="two_stage_quadrotor_rollout.png",
):
    env = Quadrotor2DEnv(dt=dt, max_time=max_time)
    obs, _ = env.reset(seed=seed)

    states = []
    actions = []
    rewards = []
    modes = []

    for _ in range(num_steps):
        mode = "pitch" if abs(float(obs[2])) > policy.theta_threshold else "position"
        action, _ = policy.predict(obs, deterministic=True)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        states.append(obs.copy())
        actions.append(np.asarray(action, dtype=np.float32).copy())
        rewards.append(float(reward))
        modes.append(mode)

        obs = next_obs
        if terminated or truncated:
            break

    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32)
    times = np.arange(states.shape[0]) * dt
    mode_indicator = np.asarray([0 if m == "pitch" else 1 for m in modes], dtype=np.float32)

    fig = plt.figure(figsize=(14, 12))

    ax1 = plt.subplot(4, 1, 1)
    labels = ["x", "z", "theta", "vx", "vz", "theta_dot"]
    for i, label in enumerate(labels):
        ax1.plot(times, states[:, i], label=label)
    ax1.set_title("Two-stage Quadrotor State Trajectory")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("state")
    ax1.legend(ncol=3, fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(4, 1, 2)
    ax2.plot(times, actions[:, 0], label="u_left")
    ax2.plot(times, actions[:, 1], label="u_right")
    ax2.set_title("Control Inputs")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("thrust")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(4, 1, 3)
    ax3.plot(times, rewards, label="reward")
    ax3.set_title("Reward")
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("reward")
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(4, 1, 4)
    ax4.step(times, mode_indicator, where="post")
    ax4.set_yticks([0, 1], labels=["pitch", "position"])
    ax4.set_title("Active Controller")
    ax4.set_xlabel("time (s)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved rollout plot to: {filename}")


def main():
    algo = "sac"
    dt = 0.01
    max_time = 2.0
    seed = 0
    device = "cuda"
    verbose = 0
    log_interval = 20

    pitch_cfg = StageConfig(
        name="pitch",
        algo=algo,
        time_steps=1_000_0000 if algo == "ppo" else 500_000_00,
        n_envs=1,
        seed=seed,
        dt=dt,
        max_time=max_time,
        device=device,
        verbose=verbose,
        log_interval=log_interval,
    )
    position_cfg = StageConfig(
        name="position",
        algo=algo,
        time_steps=1_000_0000 if algo == "ppo" else 500_000_00,
        n_envs=1,
        seed=seed + 1,
        dt=dt,
        max_time=max_time,
        device=device,
        verbose=verbose,
        log_interval=log_interval,
    )

    pitch_model, _ = train_stage(
        stage_cfg=pitch_cfg,
        env_cls=PitchStabilizationEnv,
        save_name="quadrotor_pitch_controller",
    )
    position_model, _ = train_stage(
        stage_cfg=position_cfg,
        env_cls=PositionStabilizationEnv,
        save_name="quadrotor_position_controller",
    )

    two_stage_policy = TwoStageQuadrotorPolicy(
        pitch_model=pitch_model,
        position_model=position_model,
        theta_threshold=0.08,
    )
    rollout_and_plot(
        policy=two_stage_policy,
        seed=seed,
        dt=dt,
        max_time=4.0,
        filename=f"{algo.upper()}_quadrotor_two_stage_rollout.png",
    )


if __name__ == "__main__":
    main()
