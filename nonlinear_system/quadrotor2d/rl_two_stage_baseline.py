#!/usr/bin/env python3
"""
Two-stage RL training for the 2D quadrotor.

Stage 1 trains an attitude controller that primarily stabilizes pitch angle.
Stage 2 trains a position controller that stabilizes x/z once the pitch angle
is already small.
"""

import os
from dataclasses import dataclass

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


class PitchStabilizationEnv(Quadrotor2DEnv):
    """
    Observation is reduced to [theta, theta_dot] (2D).
    The full 6D state is still used internally for dynamics and reward, but
    the policy only sees the two pitch states it can directly control.
    """

    def __init__(self, dt=0.01, max_time=2.0):
        super().__init__(dt=dt, max_time=max_time)
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
        if not (abs(float(pitch_obs[0])) < np.pi * 0.1
                and abs(float(pitch_obs[1])) < 0.9):
            return False
        # x, z, vx, vz must also lie within the easy position stage reset range
        # so the handoff starts from a state the position controller can handle.
        x_lim, z_lim, vx_lim, vz_lim, _ = _POSITION_CURRICULUM[-1]
        full = self.x_current.detach().cpu().numpy()
        return bool(
            abs(float(full[0])) <= x_lim
            and abs(float(full[1])) <= z_lim
            and abs(float(full[3])) <= vx_lim
            and abs(float(full[4])) <= vz_lim
        )

    def step(self, action):
        full_obs, _, terminated, truncated, info = super().step(action)

        state = torch.as_tensor(full_obs, dtype=self.dtype)
        theta     = state[2]
        theta_dot = state[5]
        z         = state[1]

        crashed   = float(z) < -10.0
        pitch_obs = self._pitch_obs(full_obs)
        success   = self.reached_equilibrium(pitch_obs)

        # Reward focused solely on pitch stabilization.
        # (1 - cos(theta)) is bounded in [0, 2] across the full rotation range;
        # coefficient 160 matches the curvature of the old 80*theta^2 near zero
        # (since 1-cos(theta) ≈ theta^2/2 for small theta).
        angle_cost = (1.0 - torch.cos(theta)).item()  # in [0, 2]

        reward = -(
            160.0 * angle_cost
            + 12.0 * theta_dot.pow(2).item()
        )
        terminated = bool(success or crashed)

        if crashed:
            reward -= 200.0
        if success:
            reward += 5000.0
        elif terminated or truncated:
            reward -= 100.0

        info["reached_equilibrium"] = success
        info["stage_name"] = "pitch"
        info["crashed"] = crashed
        info["theta"] = float(theta.item())

        return pitch_obs, float(reward), terminated, truncated, info


# Curriculum stages for position training: (x_lim, z_lim, vx_lim, vz_lim, label)
# theta/theta_dot reset range is always ±small_angle_limit / ±0.9 (pitch already small at handoff).
_POSITION_CURRICULUM = [
    (0.5,  0.5,  1.0,  1.0,  "easy"),
    (1.5,  1.5,  2.5,  2.5,  "medium"),
    (3.0,  3.0,  4.0,  4.0,  "full"),
]


class PositionStabilizationEnv(Quadrotor2DEnv):
    def __init__(self, dt=0.01, max_time=2.0, small_angle_limit=0.1*np.pi,
                 curriculum_level=0):
        super().__init__(dt=dt, max_time=max_time)
        self.small_angle_limit = float(small_angle_limit)
        self.curriculum_level = int(curriculum_level)
        self._update_reset_bounds()

    def _update_reset_bounds(self):
        x, z, vx, vz, _ = _POSITION_CURRICULUM[self.curriculum_level]
        self.reset_x_lo = torch.tensor(
            [-x, -z, -self.small_angle_limit, -vx, -vz, -0.9],
            dtype=self.dtype,
        )
        self.reset_x_up = torch.tensor(
            [ x,  z,  self.small_angle_limit,  vx,  vz,  0.9],
            dtype=self.dtype,
        )

    def set_curriculum_level(self, level: int):
        self.curriculum_level = min(int(level), len(_POSITION_CURRICULUM) - 1)
        self._update_reset_bounds()

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

        crashed = float(z) < -10.0
        success = self.reached_equilibrium(obs)

        reward = -(
            5.0 * x.pow(2).item()
            + 5.0 * z.pow(2).item()
            + 5.0 * vx.pow(2).item()
            + 5.0 * vz.pow(2).item()
            + 10.0 * theta.pow(2).item()
            + 5.0 * theta_dot.pow(2).item()
            + 5.0 * act_delta.pow(2).sum().item()
        )
        terminated = bool(
            success or crashed or terminated or torch.abs(theta).item() > self.small_angle_limit * 2.0
        )

        if crashed:
            reward -= 2000.0
        if success:
            reward += 5000.0
        elif terminated or truncated:
            reward -= 2000.0

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
        eval_every_episodes: int = 10_000,
        eval_num_episodes: int = 100,
        print_every_episodes: int = 100,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.stage_name = stage_name
        self.eval_env_fn = eval_env_fn
        self.eval_every_episodes = int(eval_every_episodes)
        self.eval_num_episodes = int(eval_num_episodes)
        self.print_every_episodes = int(print_every_episodes)
        self.success_count = 0
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_successes = []

        self.stop_after_perfect_eval = True
        self.should_stop = False

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


class CurriculumCallback(BaseCallback):
    """
    Curriculum learning for position training.

    Periodically evaluates the deterministic policy at the current curriculum
    level.  When success rate exceeds ``success_threshold`` the reset range
    expands to the next level.  Training stops automatically once the policy
    clears the threshold at the final (full-range) level.
    """

    def __init__(
        self,
        stage_name: str,
        make_eval_env,                  # callable(level: int) -> env
        success_threshold: float = 0.85,
        eval_every_episodes: int = 5_000,
        eval_num_episodes: int = 100,
        print_every_episodes: int = 500,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.stage_name = stage_name
        self.make_eval_env = make_eval_env
        self.success_threshold = success_threshold
        self.eval_every_episodes = eval_every_episodes
        self.eval_num_episodes = eval_num_episodes
        self.print_every_episodes = print_every_episodes

        self.episode_count = 0
        self.episode_rewards: list = []
        self.episode_successes: list = []
        self.current_level = 0
        self.max_level = len(_POSITION_CURRICULUM) - 1
        self.should_stop = False

    # ── curriculum helpers ────────────────────────────────────────────────────

    def _advance(self):
        self.current_level += 1
        # propagates through VecNormalize → DummyVecEnv → Monitor → env
        self.training_env.env_method("set_curriculum_level", self.current_level)
        x, z, vx, vz, label = _POSITION_CURRICULUM[self.current_level]
        print(
            f"[{self.stage_name}] *** curriculum → level {self.current_level} "
            f"({label}): x,z ±{x:.1f}  vx,vz ±{vx:.1f} ***"
        )

    def _run_eval(self):
        eval_env = self.make_eval_env(self.current_level)
        total_success = 0
        total_reward = 0.0

        for ep in range(self.eval_num_episodes):
            obs, _ = eval_env.reset(seed=100_000 + ep)
            ep_reward = 0.0
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                ep_reward += float(reward)
                if terminated or truncated:
                    total_success += int(info.get("reached_equilibrium", False))
                    total_reward += ep_reward
                    break

        sr = total_success / self.eval_num_episodes
        avg_r = total_reward / self.eval_num_episodes
        label = _POSITION_CURRICULUM[self.current_level][4]
        print(
            f"[{self.stage_name}] eval @ level {self.current_level} ({label}): "
            f"success={sr:.2%}  avg_reward={avg_r:.1f}"
        )

        if sr >= self.success_threshold:
            if self.current_level < self.max_level:
                self._advance()
            else:
                print(
                    f"[{self.stage_name}] full-range success {sr:.2%} "
                    f">= {self.success_threshold:.0%} — stopping."
                )
                self.should_stop = True

    # ── SB3 callback interface ────────────────────────────────────────────────

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        if dones is None:
            dones = []

        for idx, info in enumerate(infos):
            if not (bool(dones[idx]) if idx < len(dones) else False):
                continue

            self.episode_count += 1
            if isinstance(info.get("episode"), dict):
                r = info["episode"].get("r")
                if r is not None:
                    self.episode_rewards.append(float(r))

            self.episode_successes.append(
                1 if info.get("reached_equilibrium", False) else 0
            )

            if (self.print_every_episodes > 0
                    and self.episode_count % self.print_every_episodes == 0
                    and self.episode_rewards):
                w = self.episode_rewards[-self.print_every_episodes:]
                sr_w = self.episode_successes[-100:]
                label = _POSITION_CURRICULUM[self.current_level][4]
                print(
                    f"[{self.stage_name}] ep {self.episode_count} "
                    f"(level={self.current_level} {label}): "
                    f"avg_r={sum(w)/len(w):.1f}  "
                    f"sr={sum(sr_w)/len(sr_w):.2%}"
                )

            if (self.eval_every_episodes > 0
                    and self.episode_count % self.eval_every_episodes == 0):
                self._run_eval()
                if self.should_stop:
                    return False

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
    def _make_env():
        env = env_cls(dt=stage_cfg.dt, max_time=stage_cfg.max_time)
        return Monitor(env)

    def _make_eval_env():
        return env_cls(dt=stage_cfg.dt, max_time=stage_cfg.max_time)

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
    if callback is None:
        callback = EquilibriumPrintCallback(
            stage_name=stage_cfg.name,
            eval_env_fn=_make_eval_env,
            eval_every_episodes=10_000,
            eval_num_episodes=100,
            print_every_episodes=100,
        )
    model.learn(
        total_timesteps=int(stage_cfg.time_steps),
        log_interval=stage_cfg.log_interval,
        callback=callback,
        reset_num_timesteps=not bool(stage_cfg.resume_path),
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
        time_steps=1_000_000 if algo == "ppo" else 500_000_00,
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
        time_steps=1_000_000 if algo == "ppo" else 500_000_00,
        n_envs=1,
        seed=seed + 1,
        dt=dt,
        max_time=max_time,
        device=device,
        verbose=verbose,
        log_interval=log_interval,
    )

    _pitch_ckpt = os.path.join(
        os.path.dirname(__file__), "saved_models", algo, "two_stage",
        "quadrotor_pitch_controller1",
    )
    pitch_cfg.resume_path = _pitch_ckpt   # ignored by train_stage if .zip absent
    pitch_model, _ = train_stage(
        stage_cfg=pitch_cfg,
        env_cls=PitchStabilizationEnv,
        save_name="quadrotor_pitch_controller",
    )

    _pos_ckpt = os.path.join(
        os.path.dirname(__file__), "saved_models", algo, "two_stage",
        "quadrotor_position_controller",
    )
    position_cfg.resume_path = _pos_ckpt   # ignored by train_stage if .zip absent

    def _make_pos_eval_env(level=0):
        return PositionStabilizationEnv(dt=dt, max_time=max_time,
                                        curriculum_level=level)

    curriculum_cb = CurriculumCallback(
        stage_name="position",
        make_eval_env=_make_pos_eval_env,
        success_threshold=0.85,
        eval_every_episodes=5_000,
        eval_num_episodes=100,
        print_every_episodes=500,
    )
    position_model, _ = train_stage(
        stage_cfg=position_cfg,
        env_cls=PositionStabilizationEnv,
        save_name="quadrotor_position_controller",
        callback=curriculum_cb,
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
