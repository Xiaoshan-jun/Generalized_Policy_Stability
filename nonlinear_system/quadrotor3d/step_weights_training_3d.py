# step_weight_training_3d.py
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import gymnasium as gym
except Exception:
    import gym  # type: ignore

from stable_baselines3 import PPO, SAC

# Add parent directory so network/* imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from network.PolicyNet import StepNet
from network.lyapunov_net import LyapunovNet


class StepNetTrainer3D:
    """
    Learn a Lyapunov-like function V(x) along trajectories induced by a fixed RL policy,
    for a 12D 3D quadrotor environment (Gymnasium).

    Key details:
      - PPO: use V(s)
      - SAC: use Q(s, pi(s)) (min over critics)
      - Lyapunov candidate:
          V(x) = |V_RL(x) - V_RL(x*)| + ||phi(x) - phi(x*)||^2 + beta||x-x*||^2
      - Stability loss:
          ReLU( sum_k sigma_k V(x_k) - (1-alpha) V(x0) )
        where sigma comes from StepNet(x0) and is normalized to a distribution.
    """

    def __init__(
        self,
        model_path: str,
        env,
        algo_type: str = "ppo",
        n_steps: int = 20,
        hidden_dim: int = 128,
        n_layers: int = 3,
        alpha: float = 0.05,
        beta: float = 0.01,
        initial_lr: float = 2e-4,
        max_rollout_steps: int = 200,
        equilibrium_from_env: bool = True,
        device: Optional[str] = None,
    ):
        self.algo_type = algo_type.lower().strip()
        self.env = env
        self.n_steps = int(n_steps)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_rollout_steps = int(max_rollout_steps)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load RL model
        if self.algo_type == "ppo":
            self.rl_model = PPO.load(model_path)
            print("Successfully loaded PPO model")
        elif self.algo_type == "sac":
            self.rl_model = SAC.load(model_path)
            print("Successfully loaded SAC model")
        else:
            raise ValueError("algo_type must be 'ppo' or 'sac'")

        # Obs dim
        if len(self.env.observation_space.shape) != 1:
            raise ValueError(f"Expected 1D obs vector, got {self.env.observation_space.shape}")
        self.state_dim = int(self.env.observation_space.shape[0])

        # Networks
        self.stepnet = StepNet(
            n_input=self.state_dim,
            n_hidden=hidden_dim,
            n_steps=self.n_steps,
            n_layers=n_layers,
        ).to(self.device)

        self.residual_net = LyapunovNet(
            n_input=self.state_dim,
            n_hidden=hidden_dim,
            n_layers=n_layers,
        ).to(self.device)

        # Opt
        self.optimizer = torch.optim.Adam(self.stepnet.parameters(), lr=initial_lr)
        self.residual_optimizer = torch.optim.Adam(self.residual_net.parameters(), lr=initial_lr)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=500, verbose=True
        )
        self.residual_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.residual_optimizer, mode="min", factor=0.5, patience=500, verbose=True
        )

        # Equilibrium
        self.equilibrium_state, self.value_at_equilibrium = self._init_equilibrium(equilibrium_from_env)
        self._print_equilibrium_info()

    # ---------------------------
    # Gymnasium helpers
    # ---------------------------
    def _env_reset(self) -> np.ndarray:
        out = self.env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, _info = out
            return np.asarray(obs, dtype=np.float32)
        return np.asarray(out, dtype=np.float32)

    def _env_step(self, action: np.ndarray):
        out = self.env.step(action)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated) or bool(truncated)
            return np.asarray(obs, dtype=np.float32), float(reward), done, info
        if isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            return np.asarray(obs, dtype=np.float32), float(reward), bool(done), info
        raise ValueError("Unexpected env.step output format")

    # ---------------------------
    # RL value-like signals
    # ---------------------------
    def _get_vrl_tensor(self, state_batched: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.algo_type == "ppo":
                v = self.rl_model.policy.predict_values(state_batched)
                return v.view(-1)
            else:
                a = self.rl_model.actor(state_batched)
                q1, q2 = self.rl_model.critic(state_batched, a)
                return torch.min(q1, q2).view(-1)

    def get_value(self, obs: np.ndarray) -> float:
        obs = np.asarray(obs, dtype=np.float32)
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        v = self._get_vrl_tensor(obs_t)
        return float(v.squeeze().cpu().item())

    # ---------------------------
    # Equilibrium
    # ---------------------------
    def _init_equilibrium(self, equilibrium_from_env: bool) -> Tuple[np.ndarray, float]:
        if equilibrium_from_env and hasattr(self.env, "obs_equ"):
            eq = getattr(self.env, "obs_equ")
            if torch.is_tensor(eq):
                eq_obs = eq.detach().cpu().numpy().astype(np.float32)
            else:
                eq_obs = np.asarray(eq, dtype=np.float32)
            v_eq = self.get_value(eq_obs)
            return eq_obs, float(v_eq)

        return self.find_practical_equilibrium(simulation_steps=300)

    def _print_equilibrium_info(self):
        print("\nEquilibrium used:")
        print(f"State dim: {self.state_dim}")
        print(f"Equilibrium state (obs): {self.equilibrium_state}")
        print(f"Value at equilibrium: {self.value_at_equilibrium:.6f}")

    def find_practical_equilibrium(self, simulation_steps: int = 300) -> Tuple[np.ndarray, float]:
        obs = self._env_reset()
        states = []
        values = []
        for _ in range(simulation_steps):
            states.append(obs.copy())
            values.append(self.get_value(obs))
            action, _ = self.rl_model.predict(obs, deterministic=True)
            obs, _, done, _ = self._env_step(action)
            if done:
                break
        last_n = min(10, len(states))
        eq_state = np.mean(states[-last_n:], axis=0).astype(np.float32)
        eq_value = float(np.mean(values[-last_n:]))
        return eq_state, eq_value

    # ---------------------------
    # Initial sampling + reset-to-state
    # ---------------------------
    def sample_initial_observations(self, batch_size: int) -> np.ndarray:
        low = np.asarray(self.env.observation_space.low, dtype=np.float32)
        high = np.asarray(self.env.observation_space.high, dtype=np.float32)
        low_f = np.where(np.isfinite(low), low, -1.0)
        high_f = np.where(np.isfinite(high), high, 1.0)
        return np.random.uniform(low_f, high_f, size=(batch_size, self.state_dim)).astype(np.float32)

    def reset_env_to_observation(self, obs0: np.ndarray) -> np.ndarray:
        _ = self._env_reset()
        try:
            unwrapped = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
            if hasattr(unwrapped, "x_current"):
                unwrapped.x_current = torch.tensor(obs0, dtype=torch.float32)
                return obs0.astype(np.float32)
        except Exception:
            pass
        return self._env_reset()

    # ---------------------------
    # Trajectories
    # ---------------------------
    def collect_trajectory(self, initial_obs: np.ndarray, max_steps: Optional[int] = None) -> np.ndarray:
        if max_steps is None:
            max_steps = self.max_rollout_steps

        obs = np.asarray(initial_obs, dtype=np.float32)
        obs = self.reset_env_to_observation(obs)

        states = []
        for _ in range(int(max_steps)):
            states.append(obs.copy())
            action, _ = self.rl_model.predict(obs, deterministic=True)
            obs, _, done, _ = self._env_step(action)
            if done:
                break
        if len(states) == 0:
            return np.zeros((0, self.state_dim), dtype=np.float32)
        return np.asarray(states, dtype=np.float32)

    # ---------------------------
    # Lyapunov candidate
    # ---------------------------
    def get_residual_value(self, state_batched: torch.Tensor) -> torch.Tensor:
        """
        V(x) = |V_RL(x) - V_RL(x*)| + ||phi(x) - phi(x*)||^2 + beta||x-x*||^2
        """
        if state_batched.ndim != 2:
            raise ValueError("state_batched must be [B,obs_dim]")

        v_rl = self._get_vrl_tensor(state_batched)  # [B]

        eq_np = np.asarray(self.equilibrium_state, dtype=np.float32)
        eq = torch.from_numpy(eq_np).to(self.device).unsqueeze(0)  # [1,obs_dim]
        v_eq = self._get_vrl_tensor(eq).view(1)  # [1]

        term1 = torch.abs(v_rl - v_eq.item())

        phi_x = self.residual_net(state_batched)
        phi_eq = self.residual_net(eq)
        term2 = torch.sum((phi_x - phi_eq) ** 2, dim=-1)

        diff = state_batched - eq
        term3 = self.beta * torch.sum(diff ** 2, dim=-1)

        return term1 + term2 + term3

    # ---------------------------
    # Sigma normalization (robust)
    # ---------------------------
    def _normalize_sigma(self, sigma_raw: torch.Tensor) -> torch.Tensor:
        """
        Make sure sigma is nonnegative and sums to 1 across steps.
        Works even if StepNet outputs arbitrary real numbers.
        """
        sigma = torch.relu(sigma_raw)
        sigma = sigma.view(-1)[: self.n_steps]
        s = sigma.sum()
        return sigma / (s + 1e-8)

    # ---------------------------
    # Loss
    # ---------------------------
    def compute_loss(self, initial_obs_batch: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        stability_loss = mean_i ReLU( sum_k sigma_k V(x_k) - (1-alpha) V(x0) )
        where sigma is normalized to sum to 1.
        """
        stability_loss = torch.zeros((), device=self.device)
        n = int(len(initial_obs_batch))
        if n == 0:
            return stability_loss, stability_loss

        valid = 0
        for obs0 in initial_obs_batch:
            states = self.collect_trajectory(obs0, max_steps=self.n_steps + 1)
            if states.shape[0] < self.n_steps + 1:
                continue
            valid += 1

            traj = torch.from_numpy(states[: self.n_steps + 1]).float().to(self.device)  # [n_steps+1,obs_dim]
            x0 = traj[0:1]  # [1,obs_dim]

            sigma_raw = self.stepnet(x0)  # expected [1,n_steps] or similar
            sigma = self._normalize_sigma(sigma_raw)  # [n_steps]

            V = self.get_residual_value(traj).view(-1)  # [n_steps+1]
            V0 = V[0]
            future = V[1:]  # [n_steps]

            weighted_future = torch.sum(sigma * future)  # sigma sums to 1

            violation = torch.relu(weighted_future - (1.0 - self.alpha) * V0)
            stability_loss = stability_loss + violation

        if valid == 0:
            return stability_loss, stability_loss

        stability_loss = stability_loss / float(valid)
        return stability_loss, stability_loss

    # ---------------------------
    # Debug plots (optional)
    # ---------------------------
    def plot_training_debug(self, out_dir: str, epoch: int, n_examples: int = 6, seed: int = 0):
        os.makedirs(out_dir, exist_ok=True)
        rng = np.random.default_rng(seed + epoch)

        initial_obs = self.sample_initial_observations(n_examples)

        margins = []
        lengths = []
        skipped = 0

        for i, obs0 in enumerate(initial_obs):
            states = self.collect_trajectory(obs0, max_steps=self.n_steps + 1)
            T = states.shape[0]
            lengths.append(T)
            if T < self.n_steps + 1:
                skipped += 1
                continue

            traj = torch.from_numpy(states[: self.n_steps + 1]).float().to(self.device)
            x0 = traj[0:1]

            with torch.no_grad():
                sigma = self._normalize_sigma(self.stepnet(x0))
                V = self.get_residual_value(traj).view(-1)
                margin = (torch.sum(sigma * V[1:]) - (1.0 - self.alpha) * V[0]).item()
                margins.append(margin)

            # V(t)
            plt.figure()
            plt.plot(np.arange(self.n_steps + 1), V.detach().cpu().numpy())
            plt.xlabel("t")
            plt.ylabel("V(x_t)")
            plt.title(f"Epoch {epoch} ex{i}: V(t) (T={T}, margin={margin:.2e})")
            plt.grid(True)
            plt.savefig(os.path.join(out_dir, f"epoch_{epoch:05d}_ex{i}_V.png"))
            plt.close()

            # sigma
            plt.figure()
            plt.bar(np.arange(1, self.n_steps + 1), sigma.detach().cpu().numpy())
            plt.xlabel("k (future step)")
            plt.ylabel("sigma_k")
            plt.title(f"Epoch {epoch} ex{i}: sigma")
            plt.grid(True)
            plt.savefig(os.path.join(out_dir, f"epoch_{epoch:05d}_ex{i}_sigma.png"))
            plt.close()

        # margins hist
        plt.figure()
        if len(margins) > 0:
            plt.hist(margins, bins=20)
        plt.xlabel("margin = sum(sigma_k V_k) - (1-alpha)*V0")
        plt.ylabel("count")
        plt.title(f"Epoch {epoch}: margin histogram (skipped={skipped}/{n_examples})")
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, f"epoch_{epoch:05d}_margins_hist.png"))
        plt.close()

        # lengths hist
        plt.figure()
        plt.hist(lengths, bins=20)
        plt.xlabel("trajectory length T")
        plt.ylabel("count")
        plt.title(f"Epoch {epoch}: trajectory lengths")
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, f"epoch_{epoch:05d}_trajlen_hist.png"))
        plt.close()

        if len(margins) > 0:
            print(
                "[debug epoch {}] skipped={}/{} | margins min/mean/max = {:.2e} / {:.2e} / {:.2e}".format(
                    epoch, skipped, n_examples, np.min(margins), np.mean(margins), np.max(margins)
                )
            )
        else:
            print("[debug epoch {}] skipped={}/{} | no valid rollouts".format(epoch, skipped, n_examples))

    # ---------------------------
    # Train
    # ---------------------------
    def train(self, n_epochs: int = 1000, batch_size: int = 256, save_root: Optional[str] = None):
        print("\nStarting training...")
        if save_root is None:
            base_dir = os.path.join(os.path.dirname(__file__), "..", "saved_models")
            save_dir = os.path.join(base_dir, self.algo_type, "{}steps_quadrotor3d".format(self.n_steps))
        else:
            save_dir = os.path.join(save_root, self.algo_type, "{}steps_quadrotor3d".format(self.n_steps))

        save_dir = os.path.abspath(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        print("Models will be saved to:", save_dir)

        debug_dir = os.path.join(save_dir, "debug_plots")
        debug_every = 50
        debug_examples = 6

        losses = []
        stability_losses = []
        best_loss = float("inf")
        max_grad_norm = 5.0
        total_start_time = time.time()

        for epoch in range(n_epochs):
            t0 = time.time()
            initial_obs = self.sample_initial_observations(batch_size)

            self.optimizer.zero_grad(set_to_none=True)
            self.residual_optimizer.zero_grad(set_to_none=True)

            total_loss, stability_loss = self.compute_loss(initial_obs)
            if total_loss.requires_grad and total_loss.grad_fn is not None:
                total_loss.backward()
            else:
                print(f"  Epoch {epoch + 1}: skipping backward (no valid rollouts)")

            torch.nn.utils.clip_grad_norm_(self.stepnet.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.residual_net.parameters(), max_grad_norm)

            self.optimizer.step()
            self.residual_optimizer.step()

            self.scheduler.step(total_loss.detach())
            self.residual_scheduler.step(total_loss.detach())

            cur = float(total_loss.detach().cpu().item())
            losses.append(cur)
            stability_losses.append(float(stability_loss.detach().cpu().item()))

            if cur < best_loss:
                best_loss = cur
                torch.save(self.stepnet.state_dict(), os.path.join(save_dir, "stepnet_best.pth"))
                torch.save(self.residual_net.state_dict(), os.path.join(save_dir, "residual_net_best.pth"))
                print("New best @ epoch {}: loss={:.6f}".format(epoch + 1, best_loss))

            print(
                "Epoch {}/{} | Stability Loss: {:.6f} | Time: {:.2f}s".format(
                    epoch + 1, n_epochs, float(stability_loss.detach().cpu().item()), time.time() - t0
                )
            )

            if (epoch + 1) % debug_every == 0 or epoch == 0:
                self.plot_training_debug(debug_dir, epoch + 1, n_examples=debug_examples, seed=123)

        print("\nTraining completed in {:.2f} minutes".format((time.time() - total_start_time) / 60.0))
        print("Best loss:", best_loss)

        torch.save(self.stepnet.state_dict(), os.path.join(save_dir, "stepnet_final.pth"))
        torch.save(self.residual_net.state_dict(), os.path.join(save_dir, "residual_net_final.pth"))

        # Loss plot
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label="Total Loss")
        plt.plot(stability_losses, label="Stability Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss ({}, {} steps) - Quadrotor3D".format(self.algo_type, self.n_steps))
        plt.legend()
        plt.savefig(os.path.join(save_dir, "training_losses.png"))
        plt.close()

        return losses, stability_losses


def main():
    # Change to your env import
    from quadrotor3d_env import Quadrotor3DEnv  # <-- ensure this matches your filename

    env = Quadrotor3DEnv(dt=0.01, max_time=2.0)

    algo = "ppo"  # "ppo" or "sac"
    n_steps = 15
    n_epochs = 1000
    batch_size = 256
    n_steps_list = [1, 5, 10, 15, 20, 25, 30, 35, 40]
    model_path = os.path.join(os.path.dirname(__file__), "saved_models", algo, "quadrotor3d_model")
    results = {}

    for n_steps in n_steps_list:
        print(f"\n--- Training with n_steps = {n_steps} ---")
        trainer = StepNetTrainer3D(
            model_path=model_path,
            env=env,
            algo_type=algo,
            n_steps=n_steps,
            hidden_dim=128,
            n_layers=3,
            alpha=0.05,
            beta=0.01,
            initial_lr=2e-4,
            max_rollout_steps=200,
            equilibrium_from_env=True,
        )

        losses, stability_losses = trainer.train(n_epochs=n_epochs, batch_size=batch_size)

        results[n_steps] = {
            "losses": losses,
            "stability_losses": stability_losses,
        }

        torch.save(trainer.stepnet.state_dict(), f"stepnet_nsteps_{n_steps}.pth")
        torch.save(trainer.residual_net.state_dict(), f"residual_net_nsteps_{n_steps}.pth")

    # After training, you can compare results.
    # For example, plot stability loss curves for all n_steps.
    plt.figure(figsize=(10, 6))
    for n_steps, res in results.items():
        plt.plot(res["stability_losses"], label=f"n_steps={n_steps}")
    plt.xlabel("Epoch")
    plt.ylabel("Stability Loss")
    plt.legend()
    plt.title("Stability Loss vs. Epoch for Different n_steps")
    plt.grid(True)
    plt.savefig("stability_loss_comparison.png")
    plt.show()

    print("Training with all n_steps completed!")


if __name__ == "__main__":
    main()