import os
import sys
import time
import numpy as np
import torch
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# Prefer Gymnasium, but fall back to Gym if needed
try:
    import gymnasium as gym
    GYMNASIUM = True
except Exception:
    import gym  # type: ignore
    GYMNASIUM = False

from stable_baselines3 import PPO, SAC
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add the parent directory to path (so network/* imports work)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from network.PolicyNet import StepNet
from network.lyapunov_net import LyapunovNet


class StepNetTrainer:
    """
    Trainer for learning a Lyapunov-like function V(x) along trajectories induced by a fixed RL policy.
    Adapted to generic continuous-control Gym/Gymnasium envs (e.g., a 2D quadrotor).

    Key changes vs your Pendulum version:
      - No pendulum-specific state/obs construction
      - Samples initial observations directly from observation_space bounds
      - Reset/step compatible with Gymnasium (terminated/truncated) and classic Gym (done)
      - Equilibrium can be taken from env.obs_equ if available (nice for quadrotor hover)
      - SAC "value-like" signal is consistently Q(s, pi(s))
    """

    def __init__(
        self,
        model_path: str,
        env,  # pass an env instance (recommended for custom envs)
        algo_type: str = "ppo",
        n_steps: int = 20,
        hidden_dim: int = 128,
        n_layers: int = 3,
        alpha: float = 0.05,
        beta: float = 0.01,
        initial_lr: float = 2e-4,
        max_rollout_steps: int = 200,
        equilibrium_from_env: bool = True,
        device=None,
    ):
        self.algo_type = algo_type.lower().strip()
        self.env = env
        self.n_steps = int(n_steps)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_rollout_steps = int(max_rollout_steps)

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load trained RL model
        if self.algo_type == "ppo":
            self.rl_model = PPO.load(model_path)
            print("Successfully loaded PPO model")
        elif self.algo_type == "sac":
            self.rl_model = SAC.load(model_path)
            print("Successfully loaded SAC model")
        else:
            raise ValueError(f"Unsupported algorithm type: {algo_type} (use 'ppo' or 'sac')")

        # Observation/action dimensions
        if not hasattr(self.env, "observation_space") or not hasattr(self.env, "action_space"):
            raise ValueError("Env must have observation_space and action_space.")

        if len(self.env.observation_space.shape) != 1:
            raise ValueError(f"Expected 1D observation vector, got shape {self.env.observation_space.shape}")

        self.state_dim = int(self.env.observation_space.shape[0])

        # Networks
        self.stepnet = StepNet(
            n_input=self.state_dim, n_hidden=hidden_dim, n_steps=self.n_steps, n_layers=n_layers
        ).to(self.device)

        self.residual_net = LyapunovNet(
            n_input=self.state_dim, n_hidden=hidden_dim, n_layers=n_layers
        ).to(self.device)

        # Optimizers + schedulers
        self.optimizer = torch.optim.Adam(self.stepnet.parameters(), lr=initial_lr)
        self.residual_optimizer = torch.optim.Adam(self.residual_net.parameters(), lr=initial_lr)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=500, verbose=True
        )
        self.residual_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.residual_optimizer, mode="min", factor=0.5, patience=500, verbose=True
        )

        # Equilibrium (use env-provided hover state if available)
        self.equilibrium_state, self.value_at_equilibrium = self._init_equilibrium(equilibrium_from_env)
        self._print_equilibrium_info()

    # ---------------------------
    # Gym/Gymnasium helpers
    # ---------------------------
    def _env_reset(self):
        out = self.env.reset()
        # Gymnasium: (obs, info), Gym: obs
        if isinstance(out, tuple) and len(out) == 2:
            obs, _info = out
            return obs
        return out

    def _env_step(self, action):
        out = self.env.step(action)
        # Gymnasium: (obs, reward, terminated, truncated, info)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated) or bool(truncated)
            return obs, float(reward), done, info
        # Gym: (obs, reward, done, info)
        if isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            return obs, float(reward), bool(done), info
        raise ValueError(f"env.step returned unexpected tuple of length {len(out)}")

    # ---------------------------
    # Value functions
    # ---------------------------
    def get_value(self, obs: np.ndarray) -> float:
        """
        Get a scalar "value-like" signal for an observation.
        PPO: true V(s) from value head
        SAC: Q(s, pi(s)) (min of critics) as a consistent proxy
        """
        obs = np.asarray(obs, dtype=np.float32)
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.algo_type == "ppo":
                v = self.rl_model.policy.predict_values(obs_tensor)
                return float(v.squeeze().cpu().numpy())
            else:
                # SAC: Q(s, pi(s))
                a = self.rl_model.actor(obs_tensor)
                q1, q2 = self.rl_model.critic(obs_tensor, a)
                q = torch.min(q1, q2)
                return float(q.squeeze().cpu().numpy())

    def _get_vrl_tensor(self, state_tensor_batched: torch.Tensor) -> torch.Tensor:
        """
        state_tensor_batched: [B, obs_dim]
        returns: [B] value-like tensor on self.device
        """
        with torch.no_grad():
            if self.algo_type == "ppo":
                v = self.rl_model.policy.predict_values(state_tensor_batched)  # [B,1] or [B]
                return v.view(-1)
            else:
                a = self.rl_model.actor(state_tensor_batched)  # [B, act_dim]
                q1, q2 = self.rl_model.critic(state_tensor_batched, a)
                q = torch.min(q1, q2)
                return q.view(-1)

    # ---------------------------
    # Equilibrium
    # ---------------------------
    def _init_equilibrium(self, equilibrium_from_env: bool):
        """
        Prefer env.obs_equ if it exists (your quadrotor env has this).
        Otherwise, do a "practical equilibrium" rollout.
        """
        if equilibrium_from_env and hasattr(self.env, "obs_equ"):
            eq = getattr(self.env, "obs_equ")
            if torch.is_tensor(eq):
                eq_obs = eq.detach().cpu().numpy().astype(np.float32)
            else:
                eq_obs = np.asarray(eq, dtype=np.float32)
            v_eq = abs(self.get_value(eq_obs))
            return eq_obs, float(v_eq)

        # Fallback: rollout from reset and average last few states
        return self.find_practical_equilibrium(simulation_steps=300)

    def _print_equilibrium_info(self):
        print("\nEquilibrium used:")
        print(f"State dim: {self.state_dim}")
        print(f"Equilibrium state (obs): {self.equilibrium_state}")
        print(f"Value at equilibrium (abs): {self.value_at_equilibrium:.6f}")

    def find_practical_equilibrium(self, simulation_steps: int = 300) -> tuple[np.ndarray, float]:
        obs = self._env_reset()
        obs = np.asarray(obs, dtype=np.float32)

        states = []
        values = []

        for _ in range(simulation_steps):
            states.append(obs.copy())
            values.append(self.get_value(obs))

            action, _ = self.rl_model.predict(obs, deterministic=True)
            obs, _, done, _ = self._env_step(action)
            obs = np.asarray(obs, dtype=np.float32)
            if done:
                break

        last_n = min(10, len(states))
        eq_state = np.mean(states[-last_n:], axis=0).astype(np.float32)
        eq_value = float(abs(np.mean(values[-last_n:])))
        return eq_state, eq_value

    # ---------------------------
    # Initial condition sampling + reset-to-state
    # ---------------------------
    def sample_initial_observations(self, batch_size: int) -> np.ndarray:
        """
        Samples initial observations uniformly from the observation_space bounds.
        For quadrotor hover-centric training, you may later want a narrower distribution.
        """
        low = np.asarray(self.env.observation_space.low, dtype=np.float32)
        high = np.asarray(self.env.observation_space.high, dtype=np.float32)

        # Handle infinite bounds (if any)
        low_f = np.where(np.isfinite(low), low, -1.0)
        high_f = np.where(np.isfinite(high), high, 1.0)

        obs0 = np.random.uniform(low_f, high_f, size=(batch_size, self.state_dim)).astype(np.float32)
        return obs0

    def reset_env_to_observation(self, obs0: np.ndarray) -> np.ndarray:
        """
        Try to reset environment, then force its internal state to match obs0 if possible.
        Your Quadrotor2DEnv stores state in env.unwrapped.x_current (torch tensor).
        If env doesn't allow setting state, we just use env.reset() and ignore obs0.
        """
        _ = self._env_reset()

        # Best effort: set internal state
        try:
            unwrapped = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
            if hasattr(unwrapped, "x_current"):
                # assume observation equals state vector
                unwrapped.x_current = torch.tensor(obs0, dtype=torch.float32)
                return obs0.astype(np.float32)
        except Exception:
            pass

        # Fallback: no direct state set possible
        obs = self._env_reset()
        return np.asarray(obs, dtype=np.float32)

    # ---------------------------
    # Trajectories
    # ---------------------------
    def collect_trajectory(self, initial_obs: np.ndarray, max_steps:  None):
        """
        Roll out policy starting from initial_obs (best-effort).
        Returns:
          states: [T, obs_dim]
        """
        if max_steps is None:
            max_steps = self.max_rollout_steps

        obs = np.asarray(initial_obs, dtype=np.float32)
        obs = self.reset_env_to_observation(obs)

        states = []
        for _ in range(max_steps):
            states.append(obs.copy())
            action, _ = self.rl_model.predict(obs, deterministic=True)
            obs, _, done, _ = self._env_step(action)
            obs = np.asarray(obs, dtype=np.float32)
            if done:
                break

        if len(states) == 0:
            return np.array([], dtype=np.float32)

        return np.asarray(states, dtype=np.float32)

    # ---------------------------
    # Lyapunov construction
    # ---------------------------
    def get_residual_value(self, state_batched: torch.Tensor) -> torch.Tensor:
        """
        Lyapunov candidate:
          V(x) = |V_RL(x) - V_RL(x*)| + ||phi(x) - phi(x*)||^2 + beta ||x - x*||^2

        state_batched: [B, obs_dim]
        returns: [B] (or scalar if B=1) tensor
        """
        if state_batched.ndim != 2:
            raise ValueError(f"Expected batched state [B,obs_dim], got shape {tuple(state_batched.shape)}")

        # Term 1: |V_RL(x) - V_RL(x*)|
        v_rl = self._get_vrl_tensor(state_batched)  # [B]
        term1 = torch.abs(v_rl - float(self.value_at_equilibrium))

        # Term 2: ||phi(x) - phi(x*)||^2
        phi_x = self.residual_net(state_batched)  # [B, dphi] or [B]
        eq = torch.from_numpy(np.asarray(self.equilibrium_state, dtype=np.float32)).to(self.device).unsqueeze(0)
        phi_eq = self.residual_net(eq)  # [1, dphi]
        term2 = torch.sum((phi_x - phi_eq) ** 2, dim=-1)

        # Term 3: beta ||x - x*||^2
        diff = state_batched - eq
        term3 = self.beta * torch.sum(diff**2, dim=-1)

        return term1 + term2 + term3

    # ---------------------------
    # Loss
    # ---------------------------
    def compute_loss(self, initial_obs_batch: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Stability-only loss:
          ReLU( mean_k sigma_k V(x_k) - (1-alpha) V(x0) )
        """
        stability_loss = torch.zeros((), device=self.device)

        n = len(initial_obs_batch)
        if n == 0:
            return stability_loss, stability_loss

        for obs0 in initial_obs_batch:
            states = self.collect_trajectory(obs0, max_steps=self.n_steps + 1)
            if states.shape[0] < self.n_steps + 1:
                continue

            x0 = torch.from_numpy(states[0]).float().to(self.device).unsqueeze(0)  # [1,obs_dim]

            # StepNet weights sigma: expected [1, n_steps] (depends on your StepNet impl)
            sigma = self.stepnet(x0)  # ideally nonnegative normalized weights

            # Compute V along the trajectory (batched)
            traj = torch.from_numpy(states[: self.n_steps + 1]).float().to(self.device)  # [n_steps+1, obs_dim]
            V = self.get_residual_value(traj)  # [n_steps+1]

            V0 = V[0]
            future = V[1:]  # [n_steps]

            # weighted average
            # NOTE: if your StepNet returns shape [1,n_steps], this works
            weighted_sum = torch.sum(sigma.view(-1)[: self.n_steps] * future) / self.n_steps

            violation = torch.relu(weighted_sum - (1.0 - self.alpha) * V0)
            stability_loss = stability_loss + violation

        stability_loss = stability_loss / float(n)
        total_loss = stability_loss
        return total_loss, stability_loss
    # ---------------------------
    # visualization
    # ---------------------------
    def plot_training_debug(self, out_dir: str, epoch: int, n_examples: int = 6, seed: int = 0):
        """
        Lightweight debug plots during training.
        Saves:
          - V(t) for a few rollouts
          - sigma for those rollouts
          - a margin histogram
          - trajectory length histogram
        """
        import os
        import numpy as np
        import torch
        import matplotlib.pyplot as plt

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
                sigma = self.stepnet(x0).view(-1)[: self.n_steps]
                V = self.get_residual_value(traj).view(-1)

                V0 = V[0]
                future = V[1:]
                weighted_future = torch.sum(sigma * future) / float(self.n_steps)
                margin = (weighted_future - (1.0 - self.alpha) * V0).item()
                margins.append(margin)

            # V(t)
            plt.figure()
            plt.plot(np.arange(self.n_steps + 1), V.detach().cpu().numpy())
            plt.xlabel("t")
            plt.ylabel("V(x_t)")
            plt.title(f"Epoch {epoch} ex{i} : V(t)  (T={T}, margin={margin:.2e})")
            plt.grid(True)
            plt.savefig(os.path.join(out_dir, f"epoch_{epoch:05d}_ex{i}_V.png"))
            plt.close()

            # sigma
            plt.figure()
            plt.bar(np.arange(1, self.n_steps + 1), sigma.detach().cpu().numpy())
            plt.xlabel("k (future step)")
            plt.ylabel("sigma_k")
            plt.title(f"Epoch {epoch} ex{i} : sigma")
            plt.grid(True)
            plt.savefig(os.path.join(out_dir, f"epoch_{epoch:05d}_ex{i}_sigma.png"))
            plt.close()

        # margins hist
        plt.figure()
        if len(margins) > 0:
            plt.hist(margins, bins=20)
        plt.xlabel("margin = weighted_future - (1-alpha)*V0")
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

        # quick console stats
        if len(margins) > 0:
            print(
                f"[debug epoch {epoch}] skipped={skipped}/{n_examples} | "
                f"margins min/mean/max = {np.min(margins):.2e} / {np.mean(margins):.2e} / {np.max(margins):.2e}"
            )
        else:
            print(f"[debug epoch {epoch}] skipped={skipped}/{n_examples} | no valid (T>=n_steps+1) rollouts")

    # ---------------------------
    # Train
    # ---------------------------
    def train(self, n_epochs: int = 1000, batch_size: int = 256, save_root: None = None):
        print("\nStarting training...")
        if save_root is None:
            base_dir = os.path.join(os.path.dirname(__file__), "..", "saved_models")
            save_dir = os.path.join(base_dir, self.algo_type, f"{self.n_steps}steps_quadrotor")
        else:
            save_dir = os.path.join(save_root, self.algo_type, f"{self.n_steps}steps_quadrotor")

        save_dir = os.path.abspath(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        print(f"Models will be saved to: {save_dir}")
        debug_dir = os.path.join(save_dir, "debug_plots")
        debug_every = 50  # plot every 50 epochs (tune this)
        debug_examples = 6
        losses = []
        stability_losses = []

        best_loss = float("inf")
        max_grad_norm = 5.0
        total_start_time = time.time()

        for epoch in range(n_epochs):
            t0 = time.time()

            # sample initial obs
            initial_obs = self.sample_initial_observations(batch_size)

            self.optimizer.zero_grad(set_to_none=True)
            self.residual_optimizer.zero_grad(set_to_none=True)

            total_loss, stability_loss = self.compute_loss(initial_obs)
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.stepnet.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.residual_net.parameters(), max_grad_norm)

            self.optimizer.step()
            self.residual_optimizer.step()

            self.scheduler.step(total_loss.detach())
            self.residual_scheduler.step(total_loss.detach())

            losses.append(float(total_loss.detach().cpu().item()))
            stability_losses.append(float(stability_loss.detach().cpu().item()))

            # save best
            cur = losses[-1]
            if cur < best_loss:
                best_loss = cur
                torch.save(self.stepnet.state_dict(), os.path.join(save_dir, "stepnet_best.pth"))
                torch.save(self.residual_net.state_dict(), os.path.join(save_dir, "residual_net_best.pth"))
                print(f"New best @ epoch {epoch+1}: loss={best_loss:.6f}")

            # print progress
            print(
                f"Epoch {epoch+1}/{n_epochs} | "
                f"Stability Loss: {stability_losses[-1]:.6f} | "
                f"Time: {time.time() - t0:.2f}s"
            )
            if (epoch + 1) % debug_every == 0 or epoch == 0:
                self.plot_training_debug(
                    out_dir=debug_dir,
                    epoch=epoch + 1,
                    n_examples=debug_examples,
                    seed=123,
                )

        print(f"\nTraining completed in {(time.time() - total_start_time) / 60:.2f} minutes")
        print(f"Best loss: {best_loss:.6f}")

        # save final
        torch.save(self.stepnet.state_dict(), os.path.join(save_dir, "stepnet_final.pth"))
        torch.save(self.residual_net.state_dict(), os.path.join(save_dir, "residual_net_final.pth"))

        # plot loss
        plt.figure(figsize=(10, 5))
        plt.plot(stability_losses, label="Stability Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training Loss ({self.algo_type}, {self.n_steps} steps) - Quadrotor")
        plt.legend()
        plt.savefig(os.path.join(save_dir, "training_losses.png"))
        plt.close()

        return losses, stability_losses


def main():
    from quadrotor2d_env import Quadrotor2DEnv  # Or your 3D version if used

    env = Quadrotor2DEnv()  # Adjust to your environment

    algo = "ppo"  # or "sac"
    n_epochs = 1000
    batch_size = 256
    n_steps_list = [5, 10, 15, 20, 25, 30, 35, 40]

    model_path_base = os.path.join(os.path.dirname(__file__), "..", "saved_models", algo, "quadrotor_model")

    results = {}

    for n_steps in n_steps_list:
        print(f"\n--- Training with n_steps = {n_steps} ---")

        trainer = StepNetTrainer(
            model_path=model_path_base,
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
