import gymnasium as gym
import numpy as np
import torch
import quadrotor_2d


class Quadrotor2DEnv(gym.Env):
    """
    Gymnasium API:
      reset(...) -> (obs, info)
      step(a) -> (obs, reward, terminated, truncated, info)

    State/obs: 6D
    Action: 2D thrusts in [0, 8]
    Reward: negative quadratic cost around equilibrium (LQR-style)
    """

    metadata = {"render_modes": []}

    def __init__(self, dt=0.01, max_time=2.0):
        super().__init__()

        self.dtype = torch.float32
        self.dt = float(dt)
        self.max_steps = int(float(max_time) / self.dt)

        # bounds (torch for internal math, np for spaces)
        self.x_lo = torch.tensor(
            [-0.3, -0.3, -np.pi * 0.3, -1.5, -1.5, -0.9], dtype=self.dtype
        )
        self.x_up = -self.x_lo

        self.u_lo = torch.tensor([0.0, 0.0], dtype=self.dtype)
        self.u_up = torch.tensor([8.0, 8.0], dtype=self.dtype)

        self.action_space = gym.spaces.Box(
            low=self.u_lo.cpu().numpy().astype(np.float32),
            high=self.u_up.cpu().numpy().astype(np.float32),
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=self.x_lo.cpu().numpy().astype(np.float32),
            high=self.x_up.cpu().numpy().astype(np.float32),
            shape=(6,),
            dtype=np.float32,
        )

        # dynamics
        self.system = quadrotor_2d.Quadrotor2D(self.dtype)

        # equilibrium
        self.obs_equ = torch.zeros((6,), dtype=self.dtype)
        self.act_equ = self.system.u_equilibrium

        # cost matrices
        self.lqr_Q = torch.diag(
            torch.tensor(
                [10, 10, 10, 1, 1, self.system.length / 2.0 / np.pi], dtype=self.dtype
            )
        )
        self.lqr_R = torch.tensor([[0.1, 0.05], [0.05, 0.1]], dtype=self.dtype)

        # state
        self.x_current = torch.zeros((6,), dtype=self.dtype)
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Use Gymnasium's RNG if available for reproducibility
        # self.np_random is set by super().reset(seed=seed)
        if hasattr(self, "np_random") and self.np_random is not None:
            rand = torch.tensor(self.np_random.random(6), dtype=self.dtype)
        else:
            rand = torch.rand(6, dtype=self.dtype)

        self.x_current = rand * (self.x_up - self.x_lo) + self.x_lo
        self.step_count = 0

        obs = self.x_current.detach().cpu().numpy().astype(np.float32)
        info = {}
        return obs, info

    def step(self, action):
        # Ensure correct shape/dtype
        action = np.asarray(action, dtype=np.float32).reshape(2,)
        # Clip to action bounds (important: SB3 policies can output slightly out-of-bounds)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action_t = torch.from_numpy(action).to(dtype=self.dtype)

        # next state via dynamics
        x_next_np = self.system.next_pose(self.x_current, action_t, self.dt)
        x_next_np[2] = (x_next_np[2] + np.pi) % (2 * np.pi) - np.pi  # wrap theta to [-π, π]
        x_next = torch.as_tensor(x_next_np, dtype=self.dtype)

        # reward (negative quadratic cost about equilibrium)
        act_delta = action_t - self.act_equ
        obs_delta = x_next - self.obs_equ
        reward = -(
            act_delta.dot(self.lqr_R @ act_delta).item()
            + obs_delta.dot(self.lqr_Q @ obs_delta).item()
        )

        self.x_current = x_next
        self.step_count += 1

        # termination/truncation
        terminated = False
        # If you want a safety termination condition, add it here, e.g.:
        # terminated = bool((torch.abs(x_next[0]) > 2.0) or (torch.abs(x_next[1]) > 2.0))

        truncated = self.step_count >= self.max_steps

        obs = x_next.detach().cpu().numpy().astype(np.float32)
        info = {}
        return obs, float(reward), bool(terminated), bool(truncated), info
