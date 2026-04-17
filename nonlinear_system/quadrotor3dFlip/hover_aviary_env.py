"""
hover_aviary_env.py
====================
RL-ready Gymnasium environment for single-drone hover stabilisation,
built on top of BaseAviary.

Observation (12D):
    [pos_x, pos_y, pos_z,           # relative to target hover position
     roll,  pitch, yaw,             # Euler angles (rad)
     vel_x, vel_y, vel_z,           # world-frame linear velocity (m/s)
     ang_vx, ang_vy, ang_vz]        # world-frame angular velocity (rad/s)

Action (4D): normalised motor commands in [-1, 1].
    Mapped to RPM via BaseAviary._normalizedActionToRPM():
        -1  -> 0 RPM
         0  -> HOVER_RPM
        +1  -> MAX_RPM

Reward: dense quadratic cost around equilibrium + distance shaping +
        hover bonus + failure / crash penalties.
"""

import numpy as np
import pybullet as p
import gymnasium as gym
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from BaseAviary import BaseAviary


class HoverAviaryEnv(BaseAviary):
    """
    Single-drone hover environment.

    The agent must bring the drone to the target hover position
    (default: [0, 0, hover_z]) and keep it there.

    Parameters
    ----------
    hover_z : float
        Target hover altitude (m).
    pos_bound : float
        Half-width of the reset sampling range for x, y (m).
        z resets in [hover_z - pos_bound, hover_z + pos_bound].
    angle_bound : float
        Reset range for roll/pitch (rad). Yaw sampled from [-pi, pi].
    vel_bound : float
        Reset range for linear velocity (m/s).
    omega_bound : float
        Reset range for angular velocity (rad/s).
    max_episode_secs : float
        Episode length in seconds.
    q_pos, q_angle, q_vel, q_omega : float
        Quadratic cost weights.
    r_action : float
        Control effort penalty weight.
    dist_shaping_weight : float
        Potential-based distance-shaping coefficient.
    hover_bonus : float
        One-time reward given each step the drone is in the hover region.
    failure_penalty : float
        Penalty applied at episode end if hover was never reached.
    terminate_penalty : float
        Penalty for crash / out-of-bounds termination.
    pos_tol, angle_tol, vel_tol, omega_tol : float
        Tolerances that define the "hover region".
    """

    def __init__(
        self,
        hover_z: float               = 1.0,
        pos_bound: float             = 0.5,
        angle_bound: float           = 0.15,
        vel_bound: float             = 0.1,
        omega_bound: float           = 0.05,
        max_episode_secs: float      = 5.0,
        # reward weights
        q_pos: float                 = 50.0,
        q_angle: float               = 5.0,
        q_vel: float                 = 20.0,
        q_omega: float               = 2.0,
        r_action: float              = 0.1,
        dist_shaping_weight: float   = 0.5,
        hover_bonus: float           = 0.2,
        failure_penalty: float       = 20.0,
        terminate_penalty: float     = 10.0,
        reward_scale: float          = 1e-3,
        smoothness_weight: float     = 0.05,
        # hover region tolerances
        pos_tol: float               = 0.08,
        angle_tol: float             = 0.08,
        vel_tol: float               = 0.20,
        omega_tol: float             = 0.20,
        # BaseAviary passthrough
        drone_model: DroneModel      = DroneModel.CF2X,
        pyb_freq: int                = 240,
        ctrl_freq: int               = 30,
        gui: bool                    = False,
        record: bool                 = False,
    ):
        self.hover_z             = float(hover_z)
        self.pos_bound           = float(pos_bound)
        self.angle_bound         = float(angle_bound)
        self.vel_bound           = float(vel_bound)
        self.omega_bound         = float(omega_bound)
        self.q_pos               = float(q_pos)
        self.q_angle             = float(q_angle)
        self.q_vel               = float(q_vel)
        self.q_omega             = float(q_omega)
        self.r_action            = float(r_action)
        self.dist_shaping_weight = float(dist_shaping_weight)
        self.hover_bonus         = float(hover_bonus)
        self.failure_penalty     = float(failure_penalty)
        self.terminate_penalty   = float(terminate_penalty)
        self.reward_scale        = float(reward_scale)
        self.smoothness_weight   = float(smoothness_weight)
        self.pos_tol             = float(pos_tol)
        self.angle_tol           = float(angle_tol)
        self.vel_tol             = float(vel_tol)
        self.omega_tol           = float(omega_tol)

        # Target hover position in world frame
        self.target_pos = np.array([0.0, 0.0, self.hover_z], dtype=np.float32)

        # Episode step limit (filled after super().__init__ sets PYB_STEPS_PER_CTRL)
        self._max_episode_secs = float(max_episode_secs)

        # Initialise BaseAviary (starts PyBullet, loads URDF, etc.)
        super().__init__(
            drone_model      = drone_model,
            num_drones       = 1,
            initial_xyzs     = np.array([[0.0, 0.0, self.hover_z]]),
            initial_rpys     = np.zeros((1, 3)),
            physics          = Physics.PYB,
            pyb_freq         = pyb_freq,
            ctrl_freq        = ctrl_freq,
            gui              = gui,
            record           = record,
            obstacles        = False,
            user_debug_gui   = False,
        )

        self._max_steps = int(self._max_episode_secs * self.CTRL_FREQ)

        # Runtime state (reset each episode)
        self._prev_action        = np.zeros(4, dtype=np.float32)
        self._prev_dist          = 0.0
        self._reached_hover      = False
        self._episode_steps      = 0

    # =========================================================================
    # BaseAviary abstract methods
    # =========================================================================

    def _actionSpace(self):
        """Normalised [-1, 1] per motor."""
        return gym.spaces.Box(
            low  = -np.ones(4, dtype=np.float32),
            high =  np.ones(4, dtype=np.float32),
            dtype = np.float32,
        )

    def _observationSpace(self):
        """12D observation: pos_error(3) + rpy(3) + vel(3) + ang_v(3)."""
        high = np.array([
            self.pos_bound * 2,  self.pos_bound * 2,  self.pos_bound * 2,  # pos error
            np.pi, np.pi/2, np.pi,                                          # rpy
            5.0,   5.0,    5.0,                                             # vel
            10.0,  10.0,   10.0,                                            # ang_v
        ], dtype=np.float32)
        return gym.spaces.Box(low=-high, high=high, dtype=np.float32)

    def _computeObs(self):
        """Return 12D observation for drone 0."""
        pos   = self.pos[0]
        rpy   = self.rpy[0]
        vel   = self.vel[0]
        ang_v = self.ang_v[0]
        pos_err = pos - self.target_pos
        obs = np.concatenate([pos_err, rpy, vel, ang_v]).astype(np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _preprocessAction(self, action):
        """Map normalised action → RPM for all drones (shape: NUM_DRONES × 4)."""
        action = np.clip(np.array(action, dtype=np.float32).reshape(4,), -1.0, 1.0)
        rpm    = self._normalizedActionToRPM(action)
        return np.tile(rpm, (self.NUM_DRONES, 1))

    def _computeReward(self):
        """Quadratic + shaping reward. Failure penalty applied in step()."""
        pos   = self.pos[0]
        rpy   = self.rpy[0]
        vel   = self.vel[0]
        ang_v = self.ang_v[0]

        pos_err = pos - self.target_pos
        state_cost = (
            self.q_pos   * np.dot(pos_err, pos_err) +
            self.q_angle * np.dot(rpy,     rpy)     +
            self.q_vel   * np.dot(vel,     vel)      +
            self.q_omega * np.dot(ang_v,   ang_v)
        )
        action_cost  = self.r_action * np.dot(self._prev_action, self._prev_action)
        delta_a      = self._cur_action - self._prev_action
        smooth_cost  = np.dot(delta_a, delta_a)

        reward = -self.reward_scale * (
            state_cost + action_cost + self.smoothness_weight * smooth_cost
        )

        # Dense distance shaping
        cur_dist = float(np.linalg.norm(pos_err))
        reward  += self.dist_shaping_weight * (self._prev_dist - cur_dist)
        self._prev_dist = cur_dist

        return float(reward)

    def _computeTerminated(self):
        """Crash or out-of-bounds."""
        pos = self.pos[0]
        rpy = self.rpy[0]
        return bool(
            pos[2] < 0.05                        or   # hit the ground
            abs(pos[0]) > self.pos_bound * 4     or
            abs(pos[1]) > self.pos_bound * 4     or
            pos[2] > self.hover_z + self.pos_bound * 4 or
            abs(rpy[0]) > 1.5                    or   # roll > ~85°
            abs(rpy[1]) > 1.5                         # pitch > ~85°
        )

    def _computeTruncated(self):
        return self._episode_steps >= self._max_steps

    def _computeInfo(self):
        pos_err = self.pos[0] - self.target_pos
        return {
            "pos_error":    float(np.linalg.norm(pos_err)),
            "hover_region": self._in_hover_region(),
            "episode_steps": self._episode_steps,
        }

    # =========================================================================
    # Override reset / step to handle per-episode state
    # =========================================================================

    def reset(self, seed=None, options=None):
        """Randomise initial state around the hover point, then reset BaseAviary."""
        if seed is not None:
            np.random.seed(seed)

        # Sample random initial pose within bounds
        dx  = np.random.uniform(-self.pos_bound,   self.pos_bound)
        dy  = np.random.uniform(-self.pos_bound,   self.pos_bound)
        dz  = np.random.uniform(-self.pos_bound,   self.pos_bound)
        roll  = np.random.uniform(-self.angle_bound, self.angle_bound)
        pitch = np.random.uniform(-self.angle_bound, self.angle_bound)
        yaw   = np.random.uniform(-np.pi, np.pi)

        init_xyz = np.array([[
            self.target_pos[0] + dx,
            self.target_pos[1] + dy,
            max(0.1, self.target_pos[2] + dz),
        ]])
        init_rpy = np.array([[roll, pitch, yaw]])

        # Update BaseAviary's init pose before its reset
        self.INIT_XYZS = init_xyz
        self.INIT_RPYS = init_rpy

        obs, info = super().reset(seed=seed, options=options)

        # Set initial velocity and angular velocity via PyBullet
        vx  = np.random.uniform(-self.vel_bound,   self.vel_bound)
        vy  = np.random.uniform(-self.vel_bound,   self.vel_bound)
        vz  = np.random.uniform(-self.vel_bound,   self.vel_bound)
        wx  = np.random.uniform(-self.omega_bound, self.omega_bound)
        wy  = np.random.uniform(-self.omega_bound, self.omega_bound)
        wz  = np.random.uniform(-self.omega_bound, self.omega_bound)
        p.resetBaseVelocity(
            self.DRONE_IDS[0],
            linearVelocity  = [vx, vy, vz],
            angularVelocity = [wx, wy, wz],
            physicsClientId = self.CLIENT,
        )
        self._updateAndStoreKinematicInformation()

        # Reset per-episode bookkeeping
        self._prev_action   = np.zeros(4, dtype=np.float32)
        self._cur_action    = np.zeros(4, dtype=np.float32)
        self._prev_dist     = float(np.linalg.norm(self.pos[0] - self.target_pos))
        self._reached_hover = False
        self._episode_steps = 0

        return self._computeObs(), self._computeInfo()

    def step(self, action):
        action = np.clip(np.array(action, dtype=np.float32).reshape(4,), -1.0, 1.0)
        self._cur_action = action

        obs, reward, terminated, truncated, info = super().step(action)

        self._episode_steps += 1

        # Check hover region
        in_hover = self._in_hover_region()
        if in_hover:
            self._reached_hover = True
            reward += self.hover_bonus

        if terminated:
            reward -= self.terminate_penalty

        # Failure penalty: episode ends without ever reaching hover
        if (terminated or truncated) and not self._reached_hover:
            reward -= self.failure_penalty

        info["hover_region"] = in_hover
        self._prev_action = action.copy()
        return obs, reward, terminated, truncated, info

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _in_hover_region(self) -> bool:
        pos_err = self.pos[0] - self.target_pos
        return bool(
            np.all(np.abs(pos_err)      <= self.pos_tol)   and
            np.all(np.abs(self.rpy[0])  <= self.angle_tol) and
            np.all(np.abs(self.vel[0])  <= self.vel_tol)   and
            np.all(np.abs(self.ang_v[0])<= self.omega_tol)
        )


# =============================================================================
# Quick sanity check
# =============================================================================

if __name__ == "__main__":
    env = HoverAviaryEnv(hover_z=1.0, gui=False)
    obs, info = env.reset(seed=0)
    print(f"obs shape : {obs.shape}")
    print(f"obs       : {obs}")
    print(f"action space: {env.action_space}")
    print(f"obs space   : {env.observation_space}")

    total_reward = 0.0
    for i in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"Episode ended at step {i+1}  total_reward={total_reward:.3f}  hover={info['hover_region']}")
            break
    env.close()
