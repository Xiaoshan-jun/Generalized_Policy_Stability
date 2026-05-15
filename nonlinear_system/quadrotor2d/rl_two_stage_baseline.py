#!/usr/bin/env python3
"""
Two-stage RL training for the 2D quadrotor.

Stage 1 trains an attitude controller that primarily stabilizes pitch angle.
Stage 2 trains a position controller that stabilizes x/z once the pitch angle
is already small.
"""

import os
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
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
class PitchEnvConfig:
    """All knobs controlling PitchStabilizationEnv reward and termination."""
    # success / crash thresholds
    success_angle_threshold: float = 0.1 * np.pi   # |theta| below this counts as success
    success_theta_dot_threshold: float = 0.9        # |theta_dot| below this counts as success
    crash_z_threshold: float = -10.0                # z below this is a crash
    # per-step reward coefficients
    angle_cost_coef: float = 100.0                  # weight on (1 - cos theta)
    theta_dot_cost_coef: float = 12.0               # weight on theta_dot**2
    # terminal rewards
    success_bonus: float = 50000.0
    crash_penalty: float = 200.0
    truncated_penalty: float = 100.0


@dataclass
class PositionEnvConfig:
    """All knobs controlling PositionStabilizationEnv reward and termination."""
    # reset bounds (theta uses small_angle_limit, theta_dot uses theta_dot_reset)
    reset_x: float = 1.5
    reset_z: float = 1.5
    reset_vx: float = 4.0
    reset_vz: float = 4.0
    theta_dot_reset: float = 0.9
    small_angle_limit: float = 0.1 * np.pi
    # crash thresholds
    crash_z_threshold: float = -10.0
    pitch_crash_multiplier: float = 2.5             # crash when |theta| > small_angle_limit * this
    # success thresholds
    success_x: float = 0.05
    success_z: float = 0.05
    success_theta: float = 0.03
    success_vx: float = 0.12
    success_vz: float = 0.12
    success_theta_dot: float = 0.10
    # branch boundary
    close_threshold: float = 0.5                    # |x|,|z| both below this → close branch
    # close-branch per-step weights
    close_w_x: float = 1.0
    close_w_z: float = 1.0
    close_w_vx: float = 1.0
    close_w_vz: float = 1.0
    close_w_theta: float = 2.0
    close_w_theta_dot: float = 1.0
    close_w_act: float = 1.0
    # far-branch per-step weights
    far_w_x: float = 5.0
    far_w_z: float = 5.0
    far_w_theta: float = 0.5
    far_w_theta_dot: float = 0.2
    # terminal rewards
    success_bonus: float = 30000.0
    crash_penalty: float = 10000.0
    truncated_penalty: float = 5000.0


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
    resume_path: str = ""   # path to a saved model zip (without .zip) to resume from
    env_cfg: Any = None     # PitchEnvConfig or PositionEnvConfig (None → defaults)


class PitchStabilizationEnv(Quadrotor2DEnv):
    """
    Observation is reduced to [theta, theta_dot] (2D).
    The full 6D state is still used internally for dynamics and reward, but
    the policy only sees the two pitch states it can directly control.
    """

    def __init__(self, dt=0.01, max_time=2.0, env_cfg: PitchEnvConfig = None):
        super().__init__(dt=dt, max_time=max_time)
        self.cfg = env_cfg if env_cfg is not None else PitchEnvConfig()
        self.reset_x_lo = torch.tensor(
            [
                float(self.x_lo[0]),
                float(self.x_lo[1]),
                -np.pi,
                float(self.x_lo[3]),
                float(self.x_lo[4]),
                float(self.x_lo[5]),
            ],
            dtype=self.dtype,
        )
        self.reset_x_up = torch.tensor(
            [
                float(self.x_up[0]),
                float(self.x_up[1]),
                np.pi,
                float(self.x_up[3]),
                float(self.x_up[4]),
                float(self.x_up[5]),
            ],
            dtype=self.dtype,
        )
        # Override to expose only [theta, theta_dot] to the policy.
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.pi,       float(self.x_lo[5])], dtype=np.float32),
            high=np.array([np.pi,       float(self.x_up[5])], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

    def _pitch_obs(self, full_obs: np.ndarray) -> np.ndarray:
        return np.array([full_obs[2], full_obs[5]], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        # Skip Quadrotor2DEnv.reset() to avoid consuming 6 random numbers for
        # an x_current that we immediately overwrite; call gymnasium.Env.reset()
        # directly just to seed np_random.
        super(Quadrotor2DEnv, self).reset(seed=seed)

        if self.np_random is not None:
            rand = torch.tensor(self.np_random.random(6), dtype=self.dtype)
        else:
            rand = torch.rand(6, dtype=self.dtype)

        self.x_current = rand * (self.reset_x_up - self.reset_x_lo) + self.reset_x_lo
        self.step_count = 0
        full_obs = self.x_current.detach().cpu().numpy().astype(np.float32)
        return self._pitch_obs(full_obs), {}

    def reached_equilibrium(self, pitch_obs) -> bool:
        # pitch_obs is [theta, theta_dot]
        pitch_obs = np.asarray(pitch_obs, dtype=np.float32)
        if not (abs(float(pitch_obs[0])) < self.cfg.success_angle_threshold
                and abs(float(pitch_obs[1])) < self.cfg.success_theta_dot_threshold):
            return False
        return True

    def step(self, action):
        full_obs, _, terminated, truncated, info = super().step(action)
        cfg = self.cfg

        state = torch.as_tensor(full_obs, dtype=self.dtype)
        theta     = state[2]
        theta_dot = state[5]
        z         = state[1]

        crashed   = float(z) < cfg.crash_z_threshold
        pitch_obs = self._pitch_obs(full_obs)
        success   = self.reached_equilibrium(pitch_obs)

        # (1 - cos(theta)) is bounded in [0, 2] across the full rotation range
        # and matches theta^2/2 near zero — a smooth, wrap-around angle cost.
        angle_cost = (1.0 - torch.cos(theta)).item()

        reward = -(
            cfg.angle_cost_coef * angle_cost
            + cfg.theta_dot_cost_coef * theta_dot.pow(2).item()
        )
        terminated = bool(success or crashed)

        if crashed:
            reward -= cfg.crash_penalty
        if success:
            reward += cfg.success_bonus
        elif truncated:
            reward -= cfg.truncated_penalty

        info["reached_equilibrium"] = success
        info["stage_name"] = "pitch"
        info["crashed"] = crashed
        info["theta"] = float(theta.item())

        return pitch_obs, float(reward), terminated, truncated, info


class PositionStabilizationEnv(Quadrotor2DEnv):
    def __init__(self, dt=0.01, max_time=2.0, env_cfg: PositionEnvConfig = None):
        super().__init__(dt=dt, max_time=max_time)
        self.cfg = env_cfg if env_cfg is not None else PositionEnvConfig()
        self.small_angle_limit = float(self.cfg.small_angle_limit)
        self.reset_x_lo = torch.tensor(
            [-self.cfg.reset_x, -self.cfg.reset_z, -self.small_angle_limit,
             -self.cfg.reset_vx, -self.cfg.reset_vz, -self.cfg.theta_dot_reset],
            dtype=self.dtype,
        )
        self.reset_x_up = torch.tensor(
            [ self.cfg.reset_x,  self.cfg.reset_z,  self.small_angle_limit,
              self.cfg.reset_vx,  self.cfg.reset_vz,  self.cfg.theta_dot_reset],
            dtype=self.dtype,
        )

    def reset(self, *, seed=None, options=None):
        # Skip Quadrotor2DEnv.reset() to avoid consuming 6 random numbers for
        # an x_current that we immediately overwrite; call gymnasium.Env.reset()
        # directly just to seed np_random.
        super(Quadrotor2DEnv, self).reset(seed=seed)

        if self.np_random is not None:
            rand = torch.tensor(self.np_random.random(6), dtype=self.dtype)
        else:
            rand = torch.rand(6, dtype=self.dtype)

        self.x_current = rand * (self.reset_x_up - self.reset_x_lo) + self.reset_x_lo
        self.step_count = 0
        return self.x_current.detach().cpu().numpy().astype(np.float32), {}

    def reached_equilibrium(self, obs) -> bool:
        obs = np.asarray(obs, dtype=np.float32)
        cfg = self.cfg
        return bool(
            abs(float(obs[0])) < cfg.success_x
            and abs(float(obs[1])) < cfg.success_z
            and abs(float(obs[2])) < cfg.success_theta
            and abs(float(obs[3])) < cfg.success_vx
            and abs(float(obs[4])) < cfg.success_vz
            and abs(float(obs[5])) < cfg.success_theta_dot
        )

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        cfg = self.cfg

        action_t = torch.as_tensor(np.asarray(action, dtype=np.float32), dtype=self.dtype)
        act_delta = action_t - self.act_equ
        state = torch.as_tensor(obs, dtype=self.dtype)

        x = state[0]
        z = state[1]
        theta = state[2]
        vx = state[3]
        vz = state[4]
        theta_dot = state[5]

        crashed = (
            float(z) < cfg.crash_z_threshold
            or torch.abs(theta).item() > self.small_angle_limit * cfg.pitch_crash_multiplier
        )
        success = self.reached_equilibrium(obs)

        close_to_equilibrium = (
            abs(float(x)) < cfg.close_threshold and abs(float(z)) < cfg.close_threshold
        )
        if close_to_equilibrium:
            state_cost = (
                cfg.close_w_x * x.pow(2).item()
                + cfg.close_w_z * z.pow(2).item()
                + cfg.close_w_vx * vx.pow(2).item()
                + cfg.close_w_vz * vz.pow(2).item()
                + cfg.close_w_theta * theta.pow(2).item()
                + cfg.close_w_theta_dot * theta_dot.pow(2).item()
                + cfg.close_w_act * act_delta.pow(2).sum().item()
            )
        else:
            state_cost = (
                cfg.far_w_x * x.pow(2).item()
                + cfg.far_w_z * z.pow(2).item()
                + cfg.far_w_theta * theta.pow(2).item()
                + cfg.far_w_theta_dot * theta_dot.pow(2).item()
            )
        reward = -state_cost
        terminated = bool(success or crashed)

        if success:
            reward += cfg.success_bonus
        elif crashed:
            reward -= cfg.crash_penalty
        elif truncated:
            reward -= cfg.truncated_penalty

        info["reached_equilibrium"] = success
        info["stage_name"] = "position"
        info["crashed"] = crashed
        return obs, float(reward), terminated, truncated, info


class TwoStageQuadrotorPolicy:
    def __init__(self, pitch_model, position_model, theta_threshold=0.08):
        self.pitch_model = pitch_model
        self.position_model = position_model
        self.theta_threshold = float(theta_threshold)

    def predict(self, obs, deterministic=True):
        obs = np.asarray(obs, dtype=np.float32)
        if abs(float(obs[2])) > self.theta_threshold:
            pitch_obs = np.array([obs[2], obs[5]], dtype=np.float32)
            return self.pitch_model.predict(pitch_obs, deterministic=deterministic)
        return self.position_model.predict(obs, deterministic=deterministic)


class EquilibriumPrintCallback(BaseCallback):
    def __init__(
        self,
        stage_name: str,
        eval_env_fn=None,
        eval_every_episodes: int = 2_000,
        eval_num_episodes: int = 100,
        print_every_episodes: int = 100,
        save_path: str = "",
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.stage_name = stage_name
        self.eval_env_fn = eval_env_fn
        self.eval_every_episodes = int(eval_every_episodes)
        self.eval_num_episodes = int(eval_num_episodes)
        self.print_every_episodes = int(print_every_episodes)
        self.save_path = str(save_path)
        self.success_count = 0
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_successes = []

        self.stop_after_perfect_eval = True
        self.should_stop = False

    def _save_checkpoint(self):
        if not self.save_path or self.model is None:
            return
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.model.save(self.save_path)
        vec_env = self.model.get_vec_normalize_env()
        if vec_env is not None:
            vec_env.save(self.save_path + "_vecnormalize.pkl")
        print(f"[{self.stage_name}] checkpoint saved -> {self.save_path}.zip")

    def _run_deterministic_eval(self):
        if self.eval_env_fn is None or self.model is None:
            return

        eval_env = self.eval_env_fn()
        total_reward = 0.0
        total_success = 0

        for episode_idx in range(self.eval_num_episodes):
            init_obs, _ = eval_env.reset(seed=100_000 + episode_idx)
            obs = init_obs.copy()
            # Snapshot the full state (x, z always available via x_current).
            init_full = eval_env.x_current.detach().cpu().numpy().copy()
            episode_reward = 0.0

            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += float(reward)
                if terminated or truncated:
                    success = bool(info.get("reached_equilibrium", False))
                    total_reward += episode_reward
                    total_success += int(success)
                    if not success:
                        final_full = eval_env.x_current.detach().cpu().numpy()
                        print(
                            f"  [fail] episode {episode_idx:3d} | "
                            f"init:  x={init_full[0]:+.3f} z={init_full[1]:+.3f} "
                            f"θ={np.degrees(init_full[2]):+.1f}° "
                            f"θ̇={init_full[5]:+.3f} | "
                            f"final: x={final_full[0]:+.3f} z={final_full[1]:+.3f} "
                            f"θ={np.degrees(final_full[2]):+.1f}° "
                            f"θ̇={final_full[5]:+.3f}"
                        )
                    break

        avg_reward = total_reward / float(self.eval_num_episodes)
        success_rate = total_success / float(self.eval_num_episodes)
        print(
            f"[{self.stage_name}] deterministic eval over {self.eval_num_episodes} episodes: "
            f"avg_reward={avg_reward:.3f}, success_rate={success_rate:.3f}"
        )
        if self.stop_after_perfect_eval and success_rate >= 0.95:
            print(
                f"[{self.stage_name}] deterministic eval success_rate={success_rate:.3f} >= 0.95, "
                "stopping training early."
            )
            self.should_stop = True

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
            self.episode_successes.append(1 if reached_equilibrium else 0)
            if reached_equilibrium:
                self.success_count += 1

            if (
                self.eval_every_episodes > 0
                and self.episode_count % self.eval_every_episodes == 0
            ):
                self._run_deterministic_eval()
                self._save_checkpoint()
                if self.should_stop:
                    return False

            if (
                self.print_every_episodes > 0
                and self.episode_count % self.print_every_episodes == 0
                and len(self.episode_rewards) > 0
            ):
                window = self.episode_rewards[-self.print_every_episodes :]
                avg_reward = sum(window) / len(window)
                success_window = self.episode_successes[-100:]
                success_rate = sum(success_window) / float(len(success_window))
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


def train_stage(stage_cfg: StageConfig, env_cls, save_name, callback=None):
    env_kwargs = {"dt": stage_cfg.dt, "max_time": stage_cfg.max_time}
    if stage_cfg.env_cfg is not None:
        env_kwargs["env_cfg"] = stage_cfg.env_cfg

    def _make_env():
        env = env_cls(**env_kwargs)
        return Monitor(env)

    def _make_eval_env():
        return env_cls(**env_kwargs)

    env = make_vec_env(_make_env, n_envs=stage_cfg.n_envs, seed=stage_cfg.seed)
    if stage_cfg.normalize_reward:
        env = VecNormalize(
            env,
            training=True,
            norm_obs=False,
            norm_reward=True,
            clip_reward=10.0,
        )
    if stage_cfg.resume_path and os.path.exists(stage_cfg.resume_path + ".zip"):
        algo_cls = SAC if stage_cfg.algo == "sac" else PPO
        print(f"[{stage_cfg.name}] Resuming from: {stage_cfg.resume_path}.zip")
        model = algo_cls.load(
            stage_cfg.resume_path,
            env=env,
            device=resolve_device(stage_cfg.device),
        )
    else:
        model = build_model(
            stage_cfg.algo,
            env,
            stage_cfg.seed,
            stage_cfg.device,
            stage_cfg.verbose,
        )

    save_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "saved_models", stage_cfg.algo, "two_stage")
    )
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)

    if callback is None:
        callback = EquilibriumPrintCallback(
            stage_name=stage_cfg.name,
            eval_env_fn=_make_eval_env,
            eval_every_episodes=2_000,
            eval_num_episodes=100,
            print_every_episodes=100,
            save_path=save_path,
        )
    model.learn(
        total_timesteps=int(stage_cfg.time_steps),
        log_interval=stage_cfg.log_interval,
        callback=callback,
        reset_num_timesteps=not bool(stage_cfg.resume_path),
    )

    model.save(save_path)
    if isinstance(env, VecNormalize):
        env.save(save_path + "_vecnormalize.pkl")
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
    # ─────────────────────────────────────────────────────────────────────────
    # All changeable training args. Pitch and position stages are fully
    # independent — edit each StageConfig separately.
    # ─────────────────────────────────────────────────────────────────────────
    train_mode = "pitch"   # "pitch", "position", or "both"

    pitch_cfg = StageConfig(
        name="pitch",
        algo="sac",
        time_steps=500_000_00,   # for PPO use 1_000_000
        n_envs=1,
        seed=0,
        dt=0.01,
        max_time=4.0,
        device="cuda",
        verbose=0,
        log_interval=20,
        normalize_reward=True,
        env_cfg=PitchEnvConfig(),
    )
    pitch_resume = False         # True → resume from saved pitch checkpoint

    position_cfg = StageConfig(
        name="position",
        algo="sac",
        time_steps=500_000_00,   # for PPO use 1_000_000
        n_envs=1,
        seed=1,
        dt=0.01,
        max_time=10.0,
        device="cuda",
        verbose=0,
        log_interval=20,
        normalize_reward=False,
        env_cfg=PositionEnvConfig(),
    )
    position_resume = True       # True → resume from saved position checkpoint
    # ─────────────────────────────────────────────────────────────────────────

    if train_mode not in ("pitch", "position", "both"):
        raise ValueError("train_mode must be 'pitch', 'position', or 'both'")

    pitch_save_name    = "quadrotor_pitch_controller"
    position_save_name = "quadrotor_position_controller"

    pitch_model = None
    position_model = None

    if train_mode in ("pitch", "both"):
        if pitch_resume:
            pitch_save_dir = os.path.join(
                os.path.dirname(__file__), "saved_models", pitch_cfg.algo, "two_stage"
            )
            pitch_cfg.resume_path = os.path.join(pitch_save_dir, pitch_save_name)
        pitch_model, _ = train_stage(
            stage_cfg=pitch_cfg,
            env_cls=PitchStabilizationEnv,
            save_name=pitch_save_name,
        )

    if train_mode in ("position", "both"):
        if position_resume:
            position_save_dir = os.path.join(
                os.path.dirname(__file__), "saved_models", position_cfg.algo, "two_stage"
            )
            position_cfg.resume_path = os.path.join(position_save_dir, position_save_name)
        position_model, _ = train_stage(
            stage_cfg=position_cfg,
            env_cls=PositionStabilizationEnv,
            save_name=position_save_name,
        )

    if train_mode == "both":
        two_stage_policy = TwoStageQuadrotorPolicy(
            pitch_model=pitch_model,
            position_model=position_model,
            theta_threshold=0.1*np.pi,
        )
        rollout_and_plot(
            policy=two_stage_policy,
            seed=pitch_cfg.seed,
            dt=pitch_cfg.dt,
            max_time=4.0,
            filename=f"{pitch_cfg.algo.upper()}_quadrotor_two_stage_rollout.png",
        )


if __name__ == "__main__":
    main()
