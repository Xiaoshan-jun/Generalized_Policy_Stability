# quadrotor3d_env.py
import gymnasium as gym
import numpy as np
import torch

# Import your dynamics class (the "Quadrotor" class you pasted)
# If it's in a file like quadrotor_3d.py, change the import accordingly.
from quadrotor3d import Quadrotor # <-- CHANGE ME to your module name


def wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class Quadrotor3DEnv(gym.Env):
    """
    Gymnasium API:
      reset(...) -> (obs, info)
      step(a) -> (obs, reward, terminated, truncated, info)

    State/obs (12D):
      [pos_x, pos_y, pos_z, roll, pitch, yaw,
       vel_x, vel_y, vel_z, omega_x, omega_y, omega_z]

    Action (4D): per-motor thrusts, clipped to [u_min, u_max].

    Reward: negative quadratic cost around equilibrium (hover).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dt: float = 0.01,
        max_time: float = 2.0,
        # State bounds (reasonable defaults; tune as you like)
        pos_bound: float = 1.0,
        angle_bound: float = 0.7,     # rad
        vel_bound: float = 3.0,       # m/s
        omega_bound: float = 5.0,     # rad/s
        # Action bounds as multiples of hover_thrust
        u_min_factor: float = 0.0,
        u_max_factor: float = 2.5,
        # LQR-style costs (diagonal weights + simple R)
        q_pos: float = 10.0,
        q_angle: float = 10.0,
        q_vel: float = 1.0,
        q_omega: float = 0.5,
        r_thrust: float = 0.1,
        seed: int = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.dtype = torch.float32
        self.dt = float(dt)
        self.max_steps = int(float(max_time) / self.dt)
        self.device = torch.device(device)

        # Dynamics (physics)
        self.system = Quadrotor(self.dtype)

        # Equilibrium
        self.obs_equ = torch.zeros((12,), dtype=self.dtype)
        self.act_equ = torch.ones((4,), dtype=self.dtype) * float(self.system.hover_thrust)

        # Bounds
        x_lo = np.array(
            [-pos_bound, -pos_bound, 0.0,          # z lower bound at 0 by default (ground)
             -angle_bound, -angle_bound, -np.pi,  # yaw in [-pi, pi]
             -vel_bound, -vel_bound, -vel_bound,
             -omega_bound, -omega_bound, -omega_bound],
            dtype=np.float32,
        )
        x_up = np.array(
            [pos_bound, pos_bound, pos_bound * 2.0,  # allow higher z
             angle_bound, angle_bound, np.pi,
             vel_bound, vel_bound, vel_bound,
             omega_bound, omega_bound, omega_bound],
            dtype=np.float32,
        )
        self.x_lo = torch.from_numpy(x_lo).to(dtype=self.dtype)
        self.x_up = torch.from_numpy(x_up).to(dtype=self.dtype)

        u_lo = np.ones((4,), dtype=np.float32) * float(u_min_factor * self.system.hover_thrust)
        u_up = np.ones((4,), dtype=np.float32) * float(u_max_factor * self.system.hover_thrust)
        self.u_lo = torch.from_numpy(u_lo).to(dtype=self.dtype)
        self.u_up = torch.from_numpy(u_up).to(dtype=self.dtype)

        self.observation_space = gym.spaces.Box(
            low=x_lo, high=x_up, shape=(12,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=u_lo, high=u_up, shape=(4,), dtype=np.float32
        )

        # Quadratic costs
        Q_diag = np.array(
            [q_pos, q_pos, q_pos,
             q_angle, q_angle, q_angle,
             q_vel, q_vel, q_vel,
             q_omega, q_omega, q_omega],
            dtype=np.float32,
        )
        self.lqr_Q = torch.diag(torch.from_numpy(Q_diag).to(dtype=self.dtype))
        self.lqr_R = (r_thrust * torch.eye(4, dtype=self.dtype))

        # State
        self.x_current = torch.zeros((12,), dtype=self.dtype)
        self.step_count = 0

        if seed is not None:
            self.reset(seed=seed)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # sample uniform within bounds using Gymnasium RNG
        if hasattr(self, "np_random") and self.np_random is not None:
            rand = torch.tensor(self.np_random.random(12), dtype=self.dtype)
        else:
            rand = torch.rand(12, dtype=self.dtype)

        self.x_current = rand * (self.x_up - self.x_lo) + self.x_lo

        # Wrap yaw to [-pi, pi] explicitly
        yaw = float(self.x_current[5].item())
        self.x_current[5] = torch.tensor(wrap_to_pi(yaw), dtype=self.dtype)

        self.step_count = 0
        obs = self.x_current.detach().cpu().numpy().astype(np.float32)
        return obs, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(4,)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Euler integration: x_{t+1} = x_t + f(x_t, u_t) * dt
        x_np = self.x_current.detach().cpu().numpy().astype(np.float32)
        xdot = self.system.dynamics(x_np, action).astype(np.float32)
        x_next = x_np + xdot * float(self.dt)

        # Wrap yaw (index 5) to [-pi, pi]
        x_next[5] = wrap_to_pi(float(x_next[5]))

        # Optional: keep roll/pitch in a reasonable range (not strictly necessary)
        # x_next[3] = np.clip(x_next[3], self.observation_space.low[3], self.observation_space.high[3])
        # x_next[4] = np.clip(x_next[4], self.observation_space.low[4], self.observation_space.high[4])

        # Reward = -quadratic cost about equilibrium, evaluated at x_next
        u_t = torch.from_numpy(action).to(dtype=self.dtype)
        x_next_t = torch.from_numpy(x_next).to(dtype=self.dtype)

        du = u_t - self.act_equ
        dx = x_next_t - self.obs_equ

        reward = -(
            (du @ (self.lqr_R @ du)).item()
            + (dx @ (self.lqr_Q @ dx)).item()
        )
        reward *= 1e-4  # start here (try 1e-3 or 1e-5 if needed)

        # Update state
        self.x_current = x_next_t
        self.step_count += 1

        # Termination/truncation
        terminated = False
        terminated = (
                (x_next[2] < -0.1) or
                (abs(x_next[3]) > 1.2) or
                (abs(x_next[4]) > 1.2) or
                (abs(x_next[0]) > 5.0) or (abs(x_next[1]) > 5.0) or
                (not np.isfinite(x_next).all())
        )

        # A simple "safety terminate" example (disabled by default):
        # terminated = bool((abs(x_next[0]) > 5.0) or (abs(x_next[1]) > 5.0) or (x_next[2] < -0.2))

        truncated = self.step_count >= self.max_steps

        obs = x_next.astype(np.float32)
        return obs, float(reward), bool(terminated), bool(truncated), {}