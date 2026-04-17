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

    Action (4D): per-motor thrusts in [u_min, u_max].
        The SAC actor uses a tanh output layer; SB3 automatically rescales
        tanh ∈ (-1, 1) to the action space bounds [u_min, u_max].

    Reward: negative quadratic cost around equilibrium (hover).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dt: float = 0.01,
        max_time: float = 10.0,
        # State bounds (reasonable defaults; tune as you like)
        pos_bound: float = 5.0,
        hover_z: float = 10.0,        # equilibrium altitude; reset z in [hover_z - pos_bound, hover_z + pos_bound]
        angle_bound: float = 0.15,    # rad (~8.6°) — small-tilt regime
        init_scale: float = 1.0,      # curriculum: start at this fraction of full bounds (0,1]
        vel_bound: float = 0.1,       # m/s  — near-zero initial velocity
        omega_bound: float = 0.05,    # rad/s — near-zero initial angular rate
        # Action bounds as multiples of hover_thrust
        u_min_factor: float = 0.0,
        u_max_factor: float = 2.5,  # tanh=+1 → u_max_factor × hover_thrust
        # LQR-style costs (diagonal weights + simple R)
        q_pos: float = 50.0,    # heavy: drive x,y,z to origin
        q_angle: float = 5.0,   # light: only need rough levelling
        q_vel: float = 20.0,    # medium-high: stop before reaching target
        q_omega: float = 2.0,   # light: angular rates are secondary
        r_thrust: float = 0.1,
        reward_scale: float = 5e-3,
        smoothness_weight: float = 0.05,
        hover_bonus: float = 1.0,
        success_reward: float = 50.0,       # one-time bonus when drone first reaches equilibrium
        dist_shaping_weight: float = 2.0,   # dense bonus: reward shrinking pos distance
        failure_penalty: float = 20.0,      # applied at episode end if equilibrium never reached
        terminate_penalty: float = 10.0,
        terminate_z_min: float = 0.5,    # crash if z drops below this (above ground)
        terminate_xy_bound: float = 15,
        terminate_angle_bound: float = 1.5,
        pos_tol: float = 0.20,
        angle_tol: float = 0.15,
        vel_tol: float = 0.40,
        omega_tol: float = 0.30,
        seed: int = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.dtype = torch.float32
        self.dt = float(dt)
        self.max_steps = max(500, int(float(max_time) / self.dt))
        self.device = torch.device(device)

        # Dynamics (physics)
        self.system = Quadrotor(self.dtype)

        # Equilibrium: hover at (0, 0, hover_z) with zero attitude/rates
        self.obs_equ = torch.zeros((12,), dtype=self.dtype)
        self.obs_equ[2] = float(hover_z)
        self.act_equ = torch.ones((4,), dtype=self.dtype) * float(self.system.hover_thrust)

        # Full-scale reset bounds (absolute state space)
        self._hover_z   = float(hover_z)
        self._x_lo_full = torch.tensor(
            [-pos_bound, -pos_bound, max(0.0, hover_z - pos_bound),
             -angle_bound, -angle_bound, -np.pi,
             -vel_bound, -vel_bound, -vel_bound,
             -omega_bound, -omega_bound, -omega_bound],
            dtype=self.dtype,
        )
        self._x_up_full = torch.tensor(
            [pos_bound, pos_bound, hover_z + pos_bound,
             angle_bound, angle_bound, np.pi,
             vel_bound, vel_bound, vel_bound,
             omega_bound, omega_bound, omega_bound],
            dtype=self.dtype,
        )
        # Apply initial curriculum scale
        self.x_lo = self._x_lo_full.clone()
        self.x_up = self._x_up_full.clone()
        self.set_scale(init_scale)

        # Observation space bounds in ERROR space (all centered at 0)
        obs_lo = np.array(
            [-pos_bound, -pos_bound, -pos_bound,
             -angle_bound, -angle_bound, -np.pi,
             -vel_bound, -vel_bound, -vel_bound,
             -omega_bound, -omega_bound, -omega_bound],
            dtype=np.float32,
        )
        obs_up = np.array(
            [pos_bound, pos_bound, pos_bound,
             angle_bound, angle_bound, np.pi,
             vel_bound, vel_bound, vel_bound,
             omega_bound, omega_bound, omega_bound],
            dtype=np.float32,
        )

        # Physical thrust bounds (used in piecewise denormalisation)
        self.u_min   = float(u_min_factor * self.system.hover_thrust)   # 0 N
        self.u_hover = float(self.system.hover_thrust)                  # hover thrust
        self.u_max   = float(u_max_factor * self.system.hover_thrust)

        self.observation_space = gym.spaces.Box(
            low=obs_lo, high=obs_up, shape=(12,), dtype=np.float32
        )
        # Normalised action space: policy outputs tanh ∈ (-1, 1)
        #   -1  →  u_min   (zero thrust)
        #    0  →  u_hover (equilibrium, natural "do-nothing" output)
        #   +1  →  u_max   (2.5 × hover)
        self.action_space = gym.spaces.Box(
            low  = -np.ones(4, dtype=np.float32),
            high =  np.ones(4, dtype=np.float32),
            dtype = np.float32,
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
        self.reward_scale = float(reward_scale)
        self.smoothness_weight = float(smoothness_weight)
        self.hover_bonus = float(hover_bonus)
        self.success_reward = float(success_reward)
        self.dist_shaping_weight = float(dist_shaping_weight)
        self.failure_penalty = float(failure_penalty)
        self.terminate_penalty = float(terminate_penalty)
        self.terminate_z_min = float(terminate_z_min)
        self.terminate_xy_bound = (
            float(terminate_xy_bound) if terminate_xy_bound is not None else float(max(15.0, pos_bound * 1.5))
        )
        self.terminate_angle_bound = (
            float(terminate_angle_bound)
            if terminate_angle_bound is not None
            else float(max(1.2, angle_bound * 2.0))
        )
        self.pos_tol = float(pos_tol)
        self.angle_tol = float(angle_tol)
        self.vel_tol = float(vel_tol)
        self.omega_tol = float(omega_tol)

        # State
        self.x_current = torch.zeros((12,), dtype=self.dtype)
        self.step_count = 0
        self.prev_action = self.act_equ.clone()
        self.prev_pos_dist = 0.0   # used for distance-shaping reward
        self._reached_equilibrium = False

        if seed is not None:
            self.reset(seed=seed)

    def set_scale(self, scale: float):
        """Expand or shrink the reset sampling range (curriculum learning).

        scale=0.1 → very small initial perturbations around equilibrium
        scale=1.0 → full bounds

        Invariants kept regardless of scale:
          - z lower bound >= 0 (no underground resets)
          - yaw always samples full [-pi, pi]
        """
        s = float(np.clip(scale, 0.0, 1.0))
        # Scale deviation from equilibrium for each dimension
        equ_z = self._hover_z
        self.x_lo = self._x_lo_full.clone()
        self.x_up = self._x_up_full.clone()
        # Scale all dims except yaw (index 5) by s
        for i in [0, 1, 3, 4, 6, 7, 8, 9, 10, 11]:
            self.x_lo[i] = self._x_lo_full[i] * s
            self.x_up[i] = self._x_up_full[i] * s
        # z: scale deviation around hover_z
        self.x_lo[2] = torch.tensor(max(0.0, equ_z + (self._x_lo_full[2].item() - equ_z) * s), dtype=self.dtype)
        self.x_up[2] = torch.tensor(equ_z + (self._x_up_full[2].item() - equ_z) * s, dtype=self.dtype)
        # yaw always full circle
        self.x_lo[5] = torch.tensor(-np.pi, dtype=self.dtype)
        self.x_up[5] = torch.tensor( np.pi, dtype=self.dtype)

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
        self.prev_action = self.act_equ.clone()
        self._reached_equilibrium = False
        state = self.x_current.detach().cpu().numpy().astype(np.float32)
        equ_pos = self.obs_equ[0:3].numpy()
        self.prev_pos_dist = float(np.linalg.norm(state[0:3] - equ_pos))
        # Return position error instead of absolute position
        obs = state.copy()
        obs[0:3] = state[0:3] - equ_pos
        return obs, {}

    def step(self, action):
        # Piecewise denormalisation: tanh ∈ [-1,1] → physical thrust
        #   [-1, 0]  →  [u_min,   u_hover]
        #   [ 0,+1]  →  [u_hover, u_max  ]
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(4,), -1.0, 1.0)
        thrust = np.where(
            action <= 0,
            self.u_min   + (self.u_hover - self.u_min) * (action + 1.0),
            self.u_hover + (self.u_max - self.u_hover) * action,
        ).astype(np.float32)

        # Euler integration: x_{t+1} = x_t + f(x_t, u_t) * dt
        x_np = self.x_current.detach().cpu().numpy().astype(np.float32)
        xdot = self.system.dynamics(x_np, thrust).astype(np.float32)
        x_next = x_np + xdot * float(self.dt)

        # Wrap yaw (index 5) to [-pi, pi]
        x_next[5] = wrap_to_pi(float(x_next[5]))

        # Reward shaping around hover equilibrium
        u_t = torch.from_numpy(thrust).to(dtype=self.dtype)
        x_next_t = torch.from_numpy(x_next).to(dtype=self.dtype)

        du = u_t - self.act_equ
        dx = x_next_t - self.obs_equ

        # State + effort cost
        state_cost = (dx @ (self.lqr_Q @ dx)).item()
        effort_cost = (du @ (self.lqr_R @ du)).item()
        # Smooth controls to reduce oscillatory thrust commands
        delta_u = u_t - self.prev_action
        smooth_cost = (delta_u @ delta_u).item()

        reward = -self.reward_scale * (
            state_cost + effort_cost + self.smoothness_weight * smooth_cost
        )

        # Dense distance-shaping: reward proportional to how much closer to equilibrium position
        equ_pos = self.obs_equ[0:3].numpy()
        cur_pos_dist = float(np.linalg.norm(x_next[0:3] - equ_pos))
        reward += self.dist_shaping_weight * (self.prev_pos_dist - cur_pos_dist)
        self.prev_pos_dist = cur_pos_dist

        # Update state
        self.x_current = x_next_t
        self.step_count += 1

        # Termination/truncation
        terminated = False
        terminated = (
                (x_next[2] < self.terminate_z_min) or
                (abs(x_next[3]) > self.terminate_angle_bound) or
                (abs(x_next[4]) > self.terminate_angle_bound) or
                (abs(x_next[0]) > self.terminate_xy_bound) or
                (abs(x_next[1]) > self.terminate_xy_bound) or
                (not np.isfinite(x_next).all())
        )

        # Bonus when entering a small neighborhood around equilibrium
        equ_np = self.obs_equ.numpy()
        pos_ok = np.all(np.abs(x_next[0:3] - equ_np[0:3]) <= self.pos_tol)
        angle_ok = np.all(np.abs(x_next[3:6] - equ_np[3:6]) <= self.angle_tol)
        vel_ok = np.all(np.abs(x_next[6:9]) <= self.vel_tol)
        omega_ok = np.all(np.abs(x_next[9:12]) <= self.omega_tol)
        in_hover = bool(pos_ok and angle_ok and vel_ok and omega_ok)
        if in_hover:
            if not self._reached_equilibrium:
                reward += self.success_reward   # one-time bonus on first reach
            self._reached_equilibrium = True
            reward += self.hover_bonus
        if terminated:
            reward -= self.terminate_penalty

        truncated = self.step_count >= self.max_steps

        # Failure penalty: episode ended without ever reaching equilibrium
        if (terminated or truncated) and not self._reached_equilibrium:
            reward -= self.failure_penalty
        self.prev_action = u_t.clone()

        # Return position error (centered at 0) instead of absolute position
        obs = x_next.astype(np.float32).copy()
        obs[0:3] = x_next[0:3] - self.obs_equ.numpy()[0:3]
        info = {
            "state_cost": state_cost,
            "effort_cost": effort_cost,
            "smooth_cost": smooth_cost,
            "hover_region": in_hover,
        }
        return obs, float(reward), bool(terminated), bool(truncated), info
