import os
import sys
import time
from dataclasses import dataclass

import matplotlib
import numpy as np
import torch
from stable_baselines3 import PPO, SAC

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from network.PolicyNet import StepNet
from network.lyapunov_net import LyapunovNet
from quadrotor2d_env import Quadrotor2DEnv
from rl_two_stage_baseline import (
    PitchStabilizationEnv,
    PositionStabilizationEnv,
    TwoStageQuadrotorPolicy,
)


@dataclass
class StageModels:
    rl_model: object
    stepnet: StepNet
    residual_net: LyapunovNet
    optimizer: torch.optim.Optimizer
    residual_optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau
    residual_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau
    value_at_equilibrium: float


class TwoStageStepNetTrainer:
    def __init__(
        self,
        pitch_model_path: str,
        position_model_path: str,
        env,
        algo_type: str = "ppo",
        n_steps: int = 20,
        hidden_dim: int = 128,
        n_layers: int = 3,
        alpha: float = 0.05,
        beta: float = 0.01,
        initial_lr: float = 2e-4,
        max_rollout_steps: int = 200,
        theta_threshold: float = 0.08,
        device=None,
    ):
        self.algo_type = algo_type.lower().strip()
        self.env = env
        self.n_steps = int(n_steps)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_rollout_steps = int(max_rollout_steps)
        self.theta_threshold = float(theta_threshold)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.initial_lr = float(initial_lr)
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

        if len(self.env.observation_space.shape) != 1:
            raise ValueError(f"Expected 1D observation vector, got shape {self.env.observation_space.shape}")
        self.state_dim = int(self.env.observation_space.shape[0])

        self.equilibrium_state = self._init_equilibrium_state()
        self.stage_order = ("pitch", "position")

        pitch_rl_model = self._load_rl_model(pitch_model_path)
        position_rl_model = self._load_rl_model(position_model_path)
        self.policy = TwoStageQuadrotorPolicy(
            pitch_model=pitch_rl_model,
            position_model=position_rl_model,
            theta_threshold=self.theta_threshold,
        )

        self.stage_models = {
            "pitch": self._build_stage_models(pitch_rl_model),
            "position": self._build_stage_models(position_rl_model),
        }
        self._print_setup()

    def _load_rl_model(self, model_path: str):
        if self.algo_type == "ppo":
            model = PPO.load(model_path, device=self.device)
        elif self.algo_type == "sac":
            model = SAC.load(model_path, device=self.device)
        else:
            raise ValueError(f"Unsupported algorithm type: {self.algo_type} (use 'ppo' or 'sac')")
        return model

    def _build_stage_models(self, rl_model):
        stepnet = StepNet(
            n_input=self.state_dim,
            n_hidden=self.hidden_dim,
            n_steps=self.n_steps,
            n_layers=self.n_layers,
        ).to(self.device)
        residual_net = LyapunovNet(
            n_input=self.state_dim,
            n_hidden=self.hidden_dim,
            n_layers=self.n_layers,
        ).to(self.device)

        optimizer = torch.optim.Adam(stepnet.parameters(), lr=self.initial_lr)
        residual_optimizer = torch.optim.Adam(residual_net.parameters(), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=500, verbose=True
        )
        residual_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            residual_optimizer, mode="min", factor=0.5, patience=500, verbose=True
        )
        value_at_equilibrium = abs(self.get_value(self.equilibrium_state, rl_model))

        return StageModels(
            rl_model=rl_model,
            stepnet=stepnet,
            residual_net=residual_net,
            optimizer=optimizer,
            residual_optimizer=residual_optimizer,
            scheduler=scheduler,
            residual_scheduler=residual_scheduler,
            value_at_equilibrium=float(value_at_equilibrium),
        )

    def _init_equilibrium_state(self):
        if hasattr(self.env, "obs_equ"):
            eq = getattr(self.env, "obs_equ")
            if torch.is_tensor(eq):
                return eq.detach().cpu().numpy().astype(np.float32)
            return np.asarray(eq, dtype=np.float32)
        return np.zeros(self.state_dim, dtype=np.float32)

    def _print_setup(self):
        print("\nTwo-stage Lyapunov trainer setup:")
        print(f"State dim: {self.state_dim}")
        print(f"Equilibrium state: {self.equilibrium_state}")
        print(f"Theta threshold: {self.theta_threshold}")
        print(f"Device: {self.device}")
        for stage_name in self.stage_order:
            print(
                f"{stage_name} value at equilibrium: "
                f"{self.stage_models[stage_name].value_at_equilibrium:.6f}"
            )

    def _env_reset(self):
        out = self.env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, _ = out
            return obs
        return out

    def _env_step(self, action):
        out = self.env.step(action)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated) or bool(truncated)
            return obs, float(reward), done, info
        if isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            return obs, float(reward), bool(done), info
        raise ValueError(f"env.step returned unexpected tuple of length {len(out)}")

    def get_stage_name(self, obs: np.ndarray) -> str:
        return "pitch" if abs(float(obs[2])) > self.theta_threshold else "position"

    def get_value(self, obs: np.ndarray, rl_model) -> float:
        obs = np.asarray(obs, dtype=np.float32)
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.algo_type == "ppo":
                value = rl_model.policy.predict_values(obs_tensor)
                return float(value.squeeze().cpu().numpy())
            action = rl_model.actor(obs_tensor)
            q1, q2 = rl_model.critic(obs_tensor, action)
            return float(torch.min(q1, q2).squeeze().cpu().numpy())

    def _get_vrl_tensor(self, state_tensor_batched: torch.Tensor, stage_name: str) -> torch.Tensor:
        rl_model = self.stage_models[stage_name].rl_model
        with torch.no_grad():
            if self.algo_type == "ppo":
                return rl_model.policy.predict_values(state_tensor_batched).view(-1)
            action = rl_model.actor(state_tensor_batched)
            q1, q2 = rl_model.critic(state_tensor_batched, action)
            return torch.min(q1, q2).view(-1)

    def sample_initial_observations(self, batch_size: int) -> np.ndarray:
        half = batch_size // 2
        remainder = batch_size - half

        max_time = self.env.max_steps * self.env.dt
        pitch_env = PitchStabilizationEnv(dt=self.env.dt, max_time=max_time)
        position_env = PositionStabilizationEnv(dt=self.env.dt, max_time=max_time)

        samples = []
        for i in range(half):
            obs, _ = pitch_env.reset(seed=i)
            samples.append(obs)
        for i in range(remainder):
            obs, _ = position_env.reset(seed=10_000 + i)
            samples.append(obs)

        samples = np.asarray(samples, dtype=np.float32)
        np.random.shuffle(samples)
        return samples

    def reset_env_to_observation(self, obs0: np.ndarray) -> np.ndarray:
        _ = self._env_reset()
        try:
            unwrapped = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
            if hasattr(unwrapped, "x_current"):
                unwrapped.x_current = torch.tensor(obs0, dtype=torch.float32)
                if hasattr(unwrapped, "step_count"):
                    unwrapped.step_count = 0
                return obs0.astype(np.float32)
        except Exception:
            pass
        obs = self._env_reset()
        return np.asarray(obs, dtype=np.float32)

    def collect_trajectory(self, initial_obs: np.ndarray, max_steps=None):
        if max_steps is None:
            max_steps = self.max_rollout_steps

        obs = self.reset_env_to_observation(np.asarray(initial_obs, dtype=np.float32))

        states = []
        stages = []
        for _ in range(max_steps):
            states.append(obs.copy())
            stages.append(self.get_stage_name(obs))
            action, _ = self.policy.predict(obs, deterministic=True)
            obs, _, done, _ = self._env_step(action)
            obs = np.asarray(obs, dtype=np.float32)
            if done:
                break

        if len(states) == 0:
            return np.empty((0, self.state_dim), dtype=np.float32), []

        return np.asarray(states, dtype=np.float32), stages

    def get_residual_value(self, state_batched: torch.Tensor, stage_name: str) -> torch.Tensor:
        if state_batched.ndim != 2:
            raise ValueError(f"Expected batched state [B,obs_dim], got shape {tuple(state_batched.shape)}")

        stage = self.stage_models[stage_name]
        v_rl = self._get_vrl_tensor(state_batched, stage_name)
        term1 = torch.abs(v_rl - float(stage.value_at_equilibrium))

        phi_x = stage.residual_net(state_batched)
        eq = torch.from_numpy(self.equilibrium_state).float().to(self.device).unsqueeze(0)
        phi_eq = stage.residual_net(eq)
        term2 = torch.sum((phi_x - phi_eq) ** 2, dim=-1)

        diff = state_batched - eq
        term3 = self.beta * torch.sum(diff**2, dim=-1)
        return term1 + term2 + term3

    def _segment_indices(self, stages):
        segments = {stage_name: [] for stage_name in self.stage_order}
        start = 0
        n = len(stages)

        while start < n:
            stage_name = stages[start]
            end = start + 1
            while end < n and stages[end] == stage_name:
                end += 1
            if end - start >= self.n_steps + 1:
                segments[stage_name].append((start, end))
            start = end
        return segments

    def compute_loss(self, initial_obs_batch: np.ndarray):
        stage_losses = {
            stage_name: torch.zeros((), device=self.device) for stage_name in self.stage_order
        }
        stage_counts = {stage_name: 0 for stage_name in self.stage_order}

        for obs0 in initial_obs_batch:
            states, stages = self.collect_trajectory(obs0, max_steps=self.max_rollout_steps)
            if states.shape[0] < self.n_steps + 1:
                continue

            segments = self._segment_indices(stages)
            for stage_name in self.stage_order:
                for start, end in segments[stage_name]:
                    max_offset = end - start - (self.n_steps + 1)
                    if max_offset < 0:
                        continue
                    offset = 0 if max_offset == 0 else np.random.randint(0, max_offset + 1)
                    window = states[start + offset : start + offset + self.n_steps + 1]

                    x0 = torch.from_numpy(window[0]).float().to(self.device).unsqueeze(0)
                    traj = torch.from_numpy(window).float().to(self.device)

                    stage = self.stage_models[stage_name]
                    sigma = stage.stepnet(x0)
                    V = self.get_residual_value(traj, stage_name)

                    V0 = V[0]
                    future = V[1:]
                    weighted_sum = torch.sum(sigma.view(-1)[: self.n_steps] * future) / self.n_steps
                    violation = torch.relu(weighted_sum - (1.0 - self.alpha) * V0)

                    stage_losses[stage_name] = stage_losses[stage_name] + violation
                    stage_counts[stage_name] += 1

        total_loss = torch.zeros((), device=self.device)
        averaged_losses = {}
        for stage_name in self.stage_order:
            count = max(stage_counts[stage_name], 1)
            averaged = stage_losses[stage_name] / float(count)
            averaged_losses[stage_name] = averaged
            total_loss = total_loss + averaged

        return total_loss, averaged_losses, stage_counts

    def plot_training_debug(self, out_dir: str, epoch: int, n_examples: int = 6):
        os.makedirs(out_dir, exist_ok=True)
        initial_obs = self.sample_initial_observations(n_examples)

        for example_idx, obs0 in enumerate(initial_obs):
            states, stages = self.collect_trajectory(obs0, max_steps=self.max_rollout_steps)
            if states.shape[0] == 0:
                continue

            times = np.arange(states.shape[0]) * self.env.dt
            stage_indicator = np.asarray([0 if s == "pitch" else 1 for s in stages], dtype=np.float32)

            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            plt.plot(times, states[:, 2], label="theta")
            plt.plot(times, states[:, 0], label="x")
            plt.plot(times, states[:, 1], label="z")
            plt.legend()
            plt.grid(True)
            plt.title(f"Epoch {epoch} example {example_idx} trajectory")

            plt.subplot(2, 1, 2)
            plt.step(times, stage_indicator, where="post")
            plt.yticks([0, 1], labels=["pitch", "position"])
            plt.grid(True)
            plt.xlabel("time (s)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"epoch_{epoch:05d}_ex{example_idx}_traj.png"))
            plt.close()

    def train(self, n_epochs: int = 1000, batch_size: int = 256, save_root=None, debug_every: int = 50):
        print("\nStarting two-stage Lyapunov training...")
        if save_root is None:
            base_dir = os.path.join(os.path.dirname(__file__), "saved_models")
            save_dir = os.path.join(base_dir, self.algo_type, f"{self.n_steps}steps_two_stage_quadrotor")
        else:
            save_dir = os.path.join(save_root, self.algo_type, f"{self.n_steps}steps_two_stage_quadrotor")

        save_dir = os.path.abspath(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        print(f"Models will be saved to: {save_dir}")

        losses = []
        pitch_losses = []
        position_losses = []
        best_loss = float("inf")
        max_grad_norm = 5.0
        total_start_time = time.time()

        for epoch in range(n_epochs):
            t0 = time.time()
            initial_obs = self.sample_initial_observations(batch_size)

            for stage_name in self.stage_order:
                stage = self.stage_models[stage_name]
                stage.optimizer.zero_grad(set_to_none=True)
                stage.residual_optimizer.zero_grad(set_to_none=True)

            total_loss, stage_loss_map, stage_counts = self.compute_loss(initial_obs)
            total_loss.backward()

            for stage_name in self.stage_order:
                stage = self.stage_models[stage_name]
                torch.nn.utils.clip_grad_norm_(stage.stepnet.parameters(), max_grad_norm)
                torch.nn.utils.clip_grad_norm_(stage.residual_net.parameters(), max_grad_norm)
                stage.optimizer.step()
                stage.residual_optimizer.step()
                stage.scheduler.step(stage_loss_map[stage_name].detach())
                stage.residual_scheduler.step(stage_loss_map[stage_name].detach())

            cur_total = float(total_loss.detach().cpu().item())
            cur_pitch = float(stage_loss_map["pitch"].detach().cpu().item())
            cur_position = float(stage_loss_map["position"].detach().cpu().item())

            losses.append(cur_total)
            pitch_losses.append(cur_pitch)
            position_losses.append(cur_position)

            if cur_total < best_loss:
                best_loss = cur_total
                for stage_name in self.stage_order:
                    stage = self.stage_models[stage_name]
                    torch.save(stage.stepnet.state_dict(), os.path.join(save_dir, f"{stage_name}_stepnet_best.pth"))
                    torch.save(stage.residual_net.state_dict(), os.path.join(save_dir, f"{stage_name}_residual_net_best.pth"))
                print(f"New best @ epoch {epoch + 1}: loss={best_loss:.6f}")

            print(
                f"Epoch {epoch + 1}/{n_epochs} | "
                f"Total: {cur_total:.6f} | "
                f"Pitch: {cur_pitch:.6f} ({stage_counts['pitch']} windows) | "
                f"Position: {cur_position:.6f} ({stage_counts['position']} windows) | "
                f"Time: {time.time() - t0:.2f}s"
            )

            if (epoch + 1) % debug_every == 0 or epoch == 0:
                self.plot_training_debug(
                    out_dir=os.path.join(save_dir, "debug_plots"),
                    epoch=epoch + 1,
                )

        print(f"\nTraining completed in {(time.time() - total_start_time) / 60:.2f} minutes")
        print(f"Best loss: {best_loss:.6f}")

        for stage_name in self.stage_order:
            stage = self.stage_models[stage_name]
            torch.save(stage.stepnet.state_dict(), os.path.join(save_dir, f"{stage_name}_stepnet_final.pth"))
            torch.save(stage.residual_net.state_dict(), os.path.join(save_dir, f"{stage_name}_residual_net_final.pth"))

        plt.figure(figsize=(10, 5))
        plt.plot(losses, label="Total Loss")
        plt.plot(pitch_losses, label="Pitch Loss")
        plt.plot(position_losses, label="Position Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Two-stage Lyapunov training ({self.algo_type}, {self.n_steps} steps)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "training_losses.png"))
        plt.close()

        return {
            "total_losses": losses,
            "pitch_losses": pitch_losses,
            "position_losses": position_losses,
        }


def main():
    algo = "ppo"
    n_epochs = 1000
    batch_size = 256
    n_steps_list = [5, 10, 15, 20]
    theta_threshold = 0.08

    env = Quadrotor2DEnv()

    model_dir = os.path.join(os.path.dirname(__file__), "saved_models", algo, "two_stage")
    pitch_model_path = os.path.join(model_dir, "quadrotor_pitch_controller")
    position_model_path = os.path.join(model_dir, "quadrotor_position_controller")

    results = {}
    for n_steps in n_steps_list:
        print(f"\n--- Two-stage training with n_steps = {n_steps} ---")

        trainer = TwoStageStepNetTrainer(
            pitch_model_path=pitch_model_path,
            position_model_path=position_model_path,
            env=env,
            algo_type=algo,
            n_steps=n_steps,
            hidden_dim=128,
            n_layers=3,
            alpha=0.05,
            beta=0.01,
            initial_lr=2e-4,
            max_rollout_steps=200,
            theta_threshold=theta_threshold,
        )

        results[n_steps] = trainer.train(n_epochs=n_epochs, batch_size=batch_size)

    plt.figure(figsize=(10, 6))
    for n_steps, loss_dict in results.items():
        plt.plot(loss_dict["total_losses"], label=f"n_steps={n_steps}")
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("Two-stage Lyapunov training comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("two_stage_stability_loss_comparison.png")
    plt.close()

    print("Two-stage Lyapunov training completed.")


if __name__ == "__main__":
    main()
