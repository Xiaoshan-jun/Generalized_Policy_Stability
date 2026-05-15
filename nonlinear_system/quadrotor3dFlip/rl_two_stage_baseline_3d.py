#!/usr/bin/env python3
"""
Two-stage RL training for the 3D quadrotor (gym-pybullet-drones).

Stage 1: Attitude stabilizer. Drives roll/pitch (and yaw rate) from large
         offsets — including upside-down — toward upright. Observation is
         reduced to [roll, pitch, yaw, omega_x, omega_y, omega_z] (6D).

Stage 2: Position stabilizer. Assumes the drone is already roughly upright
         and drives the full 12-D state toward hover.
"""

import os
import sys
from dataclasses import dataclass
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

sys.path.insert(0, os.path.dirname(__file__))
from hover_aviary_env import HoverAviaryEnv  # noqa: E402

matplotlib.use("Agg")


# ─────────────────────────────────────────────────────────────────────────────
# Env configs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AttitudeEnvConfig:
    """Knobs for AttitudeStabilizationEnv."""
    # reset bounds
    hover_z: float          = 2.0       # spawn altitude
    pos_bound: float        = 0.2       # small position jitter
    angle_bound: float      = float(np.pi)  # roll/pitch up to ±π (flip)
    vel_bound: float        = 0.3
    omega_bound: float      = 2.0
    max_episode_secs: float = 3.0
    ctrl_freq: int          = 30
    pyb_freq: int           = 240
    # success thresholds
    success_angle: float    = 0.1 * np.pi   # |roll|, |pitch| below this
    success_omega: float    = 1.0           # ||omega||_inf below this
    # crash
    crash_z_threshold: float = 0.05         # ground impact
    # per-step reward
    roll_cost_coef: float   = 60.0
    pitch_cost_coef: float  = 60.0
    yaw_rate_cost_coef: float = 5.0
    omega_cost_coef: float  = 4.0
    # terminal rewards
    success_bonus: float    = 20000.0
    crash_penalty: float    = 5000.0
    truncated_penalty: float = 200.0


@dataclass
class PositionEnvConfig:
    """Knobs for PositionStabilizationEnv (3D)."""
    # reset bounds
    hover_z: float          = 2.0
    pos_bound: float        = 1.0
    angle_bound: float      = 0.1 * np.pi   # already upright (within handoff)
    vel_bound: float        = 0.5
    omega_bound: float      = 0.5
    max_episode_secs: float = 5.0
    ctrl_freq: int          = 30
    pyb_freq: int           = 240
    # success thresholds
    success_pos: float      = 0.08
    success_angle: float    = 0.08
    success_vel: float      = 0.20
    success_omega: float    = 0.20
    # crash
    crash_z_threshold: float = 0.05
    pitch_crash_multiplier: float = 2.5     # crash if |roll| or |pitch| > angle_bound * this
    # close-branch threshold (on position error magnitude)
    close_threshold: float  = 0.3
    # close-branch per-step weights
    close_w_pos: float      = 5.0
    close_w_angle: float    = 5.0
    close_w_vel: float      = 5.0
    close_w_omega: float    = 1.0
    close_w_act: float      = 0.5
    # far-branch per-step weights
    far_w_pos: float        = 8.0
    far_w_angle: float      = 0.5
    far_w_omega: float      = 0.2
    # terminal rewards
    success_bonus: float    = 30000.0
    crash_penalty: float    = 10000.0
    truncated_penalty: float = 5000.0


@dataclass
class StageConfig:
    name: str
    algo: str
    time_steps: int
    n_envs: int = 1
    seed: int = 0
    device: str = "auto"
    verbose: int = 0
    log_interval: int = 10
    normalize_reward: bool = False
    resume_path: str = ""
    env_cfg: Any = None


# ─────────────────────────────────────────────────────────────────────────────
# Stage envs
# ─────────────────────────────────────────────────────────────────────────────

class AttitudeStabilizationEnv(HoverAviaryEnv):
    """
    Stage-1 env. The drone spawns with arbitrarily large attitude (potentially
    upside-down) and the policy must restore it. The observation is reduced to
    [roll, pitch, yaw, omega_x, omega_y, omega_z] so the policy attends only to
    attitude state.
    """

    def __init__(self, env_cfg: AttitudeEnvConfig = None):
        cfg = env_cfg if env_cfg is not None else AttitudeEnvConfig()
        self.cfg = cfg
        super().__init__(
            hover_z          = cfg.hover_z,
            pos_bound        = cfg.pos_bound,
            angle_bound      = cfg.angle_bound,
            vel_bound        = cfg.vel_bound,
            omega_bound      = cfg.omega_bound,
            max_episode_secs = cfg.max_episode_secs,
            ctrl_freq        = cfg.ctrl_freq,
            pyb_freq         = cfg.pyb_freq,
            # Disable HoverAviaryEnv's own terminal shaping — we apply our own.
            hover_bonus      = 0.0,
            failure_penalty  = 0.0,
            terminate_penalty = 0.0,
            dist_shaping_weight = 0.0,
            reward_scale     = 1.0,
        )
        # 6-D observation override.
        high = np.array([np.pi, np.pi/2, np.pi, 20.0, 20.0, 20.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)

    # ── observation / reward / termination overrides ─────────────────────────

    def _observationSpace(self):
        high = np.array([np.pi, np.pi/2, np.pi, 20.0, 20.0, 20.0], dtype=np.float32)
        return gym.spaces.Box(low=-high, high=high, dtype=np.float32)

    def _computeObs(self):
        rpy   = self.rpy[0]
        ang_v = self.ang_v[0]
        obs = np.concatenate([rpy, ang_v]).astype(np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _computeReward(self):
        cfg   = self.cfg
        rpy   = self.rpy[0]
        ang_v = self.ang_v[0]
        # wrap-around attitude cost: 1 - cos(angle) ∈ [0, 2]
        roll_cost  = 1.0 - np.cos(rpy[0])
        pitch_cost = 1.0 - np.cos(rpy[1])
        cost = (
            cfg.roll_cost_coef  * roll_cost
            + cfg.pitch_cost_coef * pitch_cost
            + cfg.yaw_rate_cost_coef * ang_v[2] ** 2
            + cfg.omega_cost_coef * (ang_v[0] ** 2 + ang_v[1] ** 2)
        )
        return float(-cost)

    def _computeTerminated(self):
        # Allow large attitudes (no termination on roll/pitch). Crash only on
        # ground impact / leaving the world.
        pos = self.pos[0]
        return bool(pos[2] < self.cfg.crash_z_threshold or pos[2] > 10.0)

    # ── success and step wrapping ────────────────────────────────────────────

    def reached_equilibrium(self) -> bool:
        cfg   = self.cfg
        rpy   = self.rpy[0]
        ang_v = self.ang_v[0]
        return bool(
            abs(rpy[0]) < cfg.success_angle
            and abs(rpy[1]) < cfg.success_angle
            and np.all(np.abs(ang_v) < cfg.success_omega)
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        cfg     = self.cfg
        success = self.reached_equilibrium()
        crashed = bool(terminated)   # parent terminates on crash / out-of-bounds

        if success:
            reward += cfg.success_bonus
            terminated = True
        elif crashed:
            reward -= cfg.crash_penalty
        elif truncated:
            reward -= cfg.truncated_penalty

        info["reached_equilibrium"] = success
        info["stage_name"] = "attitude"
        info["crashed"] = crashed and not success
        return obs, float(reward), bool(terminated), bool(truncated), info


class PositionStabilization3DEnv(HoverAviaryEnv):
    """
    Stage-2 env. Spawns the drone already roughly upright (small roll/pitch)
    and randomizes position offset / velocities. Observation is the full 12-D
    state from HoverAviaryEnv.
    """

    def __init__(self, env_cfg: PositionEnvConfig = None):
        cfg = env_cfg if env_cfg is not None else PositionEnvConfig()
        self.cfg = cfg
        super().__init__(
            hover_z          = cfg.hover_z,
            pos_bound        = cfg.pos_bound,
            angle_bound      = cfg.angle_bound,
            vel_bound        = cfg.vel_bound,
            omega_bound      = cfg.omega_bound,
            max_episode_secs = cfg.max_episode_secs,
            ctrl_freq        = cfg.ctrl_freq,
            pyb_freq         = cfg.pyb_freq,
            hover_bonus      = 0.0,
            failure_penalty  = 0.0,
            terminate_penalty = 0.0,
            dist_shaping_weight = 0.0,
            reward_scale     = 1.0,
        )

    def _computeReward(self):
        cfg     = self.cfg
        pos_err = self.pos[0] - self.target_pos
        rpy     = self.rpy[0]
        vel     = self.vel[0]
        ang_v   = self.ang_v[0]
        act     = self._cur_action

        close = bool(np.linalg.norm(pos_err) < cfg.close_threshold)
        if close:
            cost = (
                cfg.close_w_pos   * float(np.dot(pos_err, pos_err))
                + cfg.close_w_angle * float(np.dot(rpy[:2], rpy[:2]))
                + cfg.close_w_vel   * float(np.dot(vel, vel))
                + cfg.close_w_omega * float(np.dot(ang_v, ang_v))
                + cfg.close_w_act   * float(np.dot(act, act))
            )
        else:
            cost = (
                cfg.far_w_pos   * float(np.dot(pos_err, pos_err))
                + cfg.far_w_angle * float(np.dot(rpy[:2], rpy[:2]))
                + cfg.far_w_omega * float(np.dot(ang_v, ang_v))
            )
        return float(-cost)

    def _computeTerminated(self):
        cfg = self.cfg
        pos = self.pos[0]
        rpy = self.rpy[0]
        if pos[2] < cfg.crash_z_threshold or pos[2] > 10.0:
            return True
        flip_limit = cfg.angle_bound * cfg.pitch_crash_multiplier
        if abs(rpy[0]) > flip_limit or abs(rpy[1]) > flip_limit:
            return True
        return False

    def reached_equilibrium(self) -> bool:
        cfg     = self.cfg
        pos_err = self.pos[0] - self.target_pos
        rpy     = self.rpy[0]
        vel     = self.vel[0]
        ang_v   = self.ang_v[0]
        return bool(
            np.all(np.abs(pos_err) < cfg.success_pos)
            and abs(rpy[0]) < cfg.success_angle
            and abs(rpy[1]) < cfg.success_angle
            and np.all(np.abs(vel)   < cfg.success_vel)
            and np.all(np.abs(ang_v) < cfg.success_omega)
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        cfg     = self.cfg
        success = self.reached_equilibrium()
        crashed = bool(terminated)

        if success:
            reward += cfg.success_bonus
            terminated = True
        elif crashed:
            reward -= cfg.crash_penalty
        elif truncated:
            reward -= cfg.truncated_penalty

        info["reached_equilibrium"] = success
        info["stage_name"] = "position"
        info["crashed"] = crashed and not success
        return obs, float(reward), bool(terminated), bool(truncated), info


# ─────────────────────────────────────────────────────────────────────────────
# Two-stage policy (used after training to combine the two controllers)
# ─────────────────────────────────────────────────────────────────────────────

class TwoStageQuadrotor3DPolicy:
    """
    Dispatches between attitude- and position-stage controllers based on the
    current roll/pitch magnitude. Once the policy switches into position mode
    it latches and never returns to attitude mode (mirrors the 2D version).

    Position policy sees the full 12-D obs. Attitude policy sees
    [roll, pitch, yaw, omega_x, omega_y, omega_z] (6-D).
    """

    _ATTITUDE_IDX = [3, 4, 5, 9, 10, 11]   # rpy + ang_v in the 12-D obs

    def __init__(self, attitude_model, position_model,
                 angle_threshold: float = 0.1 * np.pi):
        self.attitude_model = attitude_model
        self.position_model = position_model
        self.angle_threshold = float(angle_threshold)
        self.position_latched = False

    def reset_episode_state(self):
        self.position_latched = False

    def _max_tilt(self, obs):
        return max(abs(float(obs[3])), abs(float(obs[4])))

    def _active(self, obs):
        if self.position_latched:
            return self.position_model
        return (self.attitude_model
                if self._max_tilt(obs) > self.angle_threshold
                else self.position_model)

    def predict(self, obs, deterministic=True):
        obs = np.asarray(obs, dtype=np.float32)
        if not self.position_latched and self._max_tilt(obs) > self.angle_threshold:
            attitude_obs = obs[self._ATTITUDE_IDX]
            return self.attitude_model.predict(attitude_obs, deterministic=deterministic)
        self.position_latched = True
        return self.position_model.predict(obs, deterministic=deterministic)


# ─────────────────────────────────────────────────────────────────────────────
# Training infrastructure (callback, helpers, train_stage)
# ─────────────────────────────────────────────────────────────────────────────

class EquilibriumPrintCallback(BaseCallback):
    """Logs success-rate and periodically saves a checkpoint."""

    def __init__(
        self,
        stage_name: str,
        eval_env_fn=None,
        eval_every_episodes: int = 1_000,
        eval_num_episodes: int = 20,
        print_every_episodes: int = 100,
        save_path: str = "",
        stop_success_rate: float = 0.95,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.stage_name = stage_name
        self.eval_env_fn = eval_env_fn
        self.eval_every_episodes = int(eval_every_episodes)
        self.eval_num_episodes = int(eval_num_episodes)
        self.print_every_episodes = int(print_every_episodes)
        self.save_path = str(save_path)
        self.stop_success_rate = float(stop_success_rate)
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_successes = []
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
        for ep_idx in range(self.eval_num_episodes):
            obs, _ = eval_env.reset(seed=100_000 + ep_idx)
            ep_reward = 0.0
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                ep_reward += float(reward)
                if terminated or truncated:
                    total_reward += ep_reward
                    total_success += int(bool(info.get("reached_equilibrium", False)))
                    break
        eval_env.close()
        avg_reward = total_reward / float(self.eval_num_episodes)
        success_rate = total_success / float(self.eval_num_episodes)
        print(
            f"[{self.stage_name}] eval over {self.eval_num_episodes} eps: "
            f"avg_reward={avg_reward:.2f}, success_rate={success_rate:.3f}"
        )
        if success_rate >= self.stop_success_rate:
            print(
                f"[{self.stage_name}] success_rate={success_rate:.3f} >= "
                f"{self.stop_success_rate:.2f}, stopping training early."
            )
            self.should_stop = True

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        if dones is None:
            dones = []
        for idx, info in enumerate(infos):
            done = bool(dones[idx]) if idx < len(dones) else False
            if not done:
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
                print(
                    f"[{self.stage_name}] ep {self.episode_count}: "
                    f"avg_r={sum(w)/len(w):.2f}  "
                    f"sr={sum(sr_w)/len(sr_w):.2%}"
                )

            if (self.eval_every_episodes > 0
                    and self.episode_count % self.eval_every_episodes == 0):
                self._run_deterministic_eval()
                self._save_checkpoint()
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
            buffer_size=300_000,
            batch_size=256,
            gamma=0.99,
            learning_rate=3e-4,
            train_freq=1,
            gradient_steps=1,
        )
    raise ValueError("algo must be 'ppo' or 'sac'")


def train_stage(stage_cfg: StageConfig, env_cls, save_name, callback=None):
    env_kwargs = {}
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
            stage_cfg.algo, env, stage_cfg.seed,
            stage_cfg.device, stage_cfg.verbose,
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
            eval_every_episodes=1_000,
            eval_num_episodes=20,
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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ─────────────────────────────────────────────────────────────────────────
    # All changeable training args. Attitude and position stages are fully
    # independent — edit each StageConfig separately.
    # ─────────────────────────────────────────────────────────────────────────
    train_mode = "attitude"   # "attitude", "position", or "both"

    attitude_cfg = StageConfig(
        name="attitude",
        algo="sac",
        time_steps=2_000_000,
        n_envs=1,
        seed=0,
        device="cuda",
        verbose=0,
        log_interval=20,
        normalize_reward=False,
        env_cfg=AttitudeEnvConfig(),
    )
    attitude_resume = False

    position_cfg = StageConfig(
        name="position",
        algo="sac",
        time_steps=2_000_000,
        n_envs=1,
        seed=1,
        device="cuda",
        verbose=0,
        log_interval=20,
        normalize_reward=False,
        env_cfg=PositionEnvConfig(),
    )
    position_resume = False
    # ─────────────────────────────────────────────────────────────────────────

    if train_mode not in ("attitude", "position", "both"):
        raise ValueError("train_mode must be 'attitude', 'position', or 'both'")

    attitude_save_name = "quadrotor3d_attitude_controller"
    position_save_name = "quadrotor3d_position_controller"

    attitude_model = None
    position_model = None

    if train_mode in ("attitude", "both"):
        if attitude_resume:
            attitude_save_dir = os.path.join(
                os.path.dirname(__file__), "saved_models", attitude_cfg.algo, "two_stage"
            )
            attitude_cfg.resume_path = os.path.join(attitude_save_dir, attitude_save_name)
        attitude_model, _ = train_stage(
            stage_cfg=attitude_cfg,
            env_cls=AttitudeStabilizationEnv,
            save_name=attitude_save_name,
        )

    if train_mode in ("position", "both"):
        if position_resume:
            position_save_dir = os.path.join(
                os.path.dirname(__file__), "saved_models", position_cfg.algo, "two_stage"
            )
            position_cfg.resume_path = os.path.join(position_save_dir, position_save_name)
        position_model, _ = train_stage(
            stage_cfg=position_cfg,
            env_cls=PositionStabilization3DEnv,
            save_name=position_save_name,
        )

    if train_mode == "both":
        two_stage_policy = TwoStageQuadrotor3DPolicy(
            attitude_model=attitude_model,
            position_model=position_model,
            angle_threshold=0.1 * np.pi,
        )
        print("Trained both stages. Combined policy is available as "
              "TwoStageQuadrotor3DPolicy(attitude_model, position_model).")
        _ = two_stage_policy   # keep reference; rollout/visualize lives elsewhere


if __name__ == "__main__":
    main()
