import numpy as np
import torch


def _is_numpy(x):
    return isinstance(x, np.ndarray)


def _cross(a, b):
    if _is_numpy(a):
        return np.cross(a, b)
    return torch.cross(a, b, dim=0)


def _quat_normalize(q):
    if _is_numpy(q):
        norm = np.linalg.norm(q)
        if norm <= 0:
            raise ValueError("Quaternion norm must be positive.")
        return q / norm
    norm = torch.linalg.norm(q)
    if torch.any(norm <= 0):
        raise ValueError("Quaternion norm must be positive.")
    return q / norm


def _quat_to_rotmat(q):
    """
    Convert a unit quaternion [qw, qx, qy, qz] to a world-from-body rotation.
    """
    q = _quat_normalize(q)
    qw, qx, qy, qz = q
    if _is_numpy(q):
        return np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz),
             2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz),
             2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx),
             1 - 2 * (qx * qx + qy * qy)],
        ],
                        dtype=q.dtype)

    R = torch.empty((3, 3), dtype=q.dtype, device=q.device)
    R[0, 0] = 1 - 2 * (qy * qy + qz * qz)
    R[0, 1] = 2 * (qx * qy - qw * qz)
    R[0, 2] = 2 * (qx * qz + qw * qy)
    R[1, 0] = 2 * (qx * qy + qw * qz)
    R[1, 1] = 1 - 2 * (qx * qx + qz * qz)
    R[1, 2] = 2 * (qy * qz - qw * qx)
    R[2, 0] = 2 * (qx * qz - qw * qy)
    R[2, 1] = 2 * (qy * qz + qw * qx)
    R[2, 2] = 1 - 2 * (qx * qx + qy * qy)
    return R


def _quat_derivative(q, omega_body):
    """
    Quaternion kinematics for body-frame angular velocity omega_body.
    q_dot = 0.5 * q ⊗ [0, omega]
    """
    q = _quat_normalize(q)
    qw, qx, qy, qz = q
    wx, wy, wz = omega_body
    if _is_numpy(q):
        omega_matrix = np.array([
            [0.0, -wx, -wy, -wz],
            [wx, 0.0, wz, -wy],
            [wy, -wz, 0.0, wx],
            [wz, wy, -wx, 0.0],
        ],
                                dtype=q.dtype)
        return 0.5 * omega_matrix @ q

    omega_matrix = torch.empty((4, 4), dtype=q.dtype, device=q.device)
    omega_matrix[0, 0] = 0
    omega_matrix[0, 1] = -wx
    omega_matrix[0, 2] = -wy
    omega_matrix[0, 3] = -wz
    omega_matrix[1, 0] = wx
    omega_matrix[1, 1] = 0
    omega_matrix[1, 2] = wz
    omega_matrix[1, 3] = -wy
    omega_matrix[2, 0] = wy
    omega_matrix[2, 1] = -wz
    omega_matrix[2, 2] = 0
    omega_matrix[2, 3] = wx
    omega_matrix[3, 0] = wz
    omega_matrix[3, 1] = wy
    omega_matrix[3, 2] = -wx
    omega_matrix[3, 3] = 0
    return 0.5 * (omega_matrix @ q)


class QuadrotorFull:
    """
    Full-attitude quadrotor dynamics with quaternion orientation.

    State:
      [pos_x, pos_y, pos_z,
       quat_w, quat_x, quat_y, quat_z,
       vel_x, vel_y, vel_z,
       omega_x, omega_y, omega_z]

    Control:
      [u1, u2, u3, u4] motor thrusts in Newtons.

    The linear velocity is expressed in the world frame and the angular
    velocity is expressed in the body frame.
    """

    def __init__(self, dtype=torch.float32):
        self.mass = 0.468
        self.gravity = 9.81
        self.arm_length = 0.225
        self.inertia = np.array([4.9e-3, 4.9e-3, 8.8e-3], dtype=np.float64)
        self.z_torque_to_force_factor = 1.1 / 29
        self.dtype = dtype
        self.hover_thrust = self.mass * self.gravity / 4

    @property
    def x_equilibrium(self):
        if isinstance(self.dtype, torch.dtype):
            x_eq = torch.zeros((13,), dtype=self.dtype)
            x_eq[3] = 1.0
            return x_eq
        x_eq = np.zeros((13,), dtype=self.dtype)
        x_eq[3] = 1.0
        return x_eq

    def thrust_to_force_torque(self, u):
        if _is_numpy(u):
            allocation = np.array([
                [1, 1, 1, 1],
                [0, self.arm_length, 0, -self.arm_length],
                [-self.arm_length, 0, self.arm_length, 0],
                [
                    self.z_torque_to_force_factor,
                    -self.z_torque_to_force_factor,
                    self.z_torque_to_force_factor,
                    -self.z_torque_to_force_factor,
                ],
            ],
                                  dtype=u.dtype)
            return allocation @ u

        allocation = torch.tensor([
            [1, 1, 1, 1],
            [0, self.arm_length, 0, -self.arm_length],
            [-self.arm_length, 0, self.arm_length, 0],
            [
                self.z_torque_to_force_factor,
                -self.z_torque_to_force_factor,
                self.z_torque_to_force_factor,
                -self.z_torque_to_force_factor,
            ],
        ],
                                  dtype=u.dtype,
                                  device=u.device)
        return allocation @ u

    def dynamics(self, x, u):
        """
        Continuous-time state derivative.
        """
        q = x[3:7]
        vel = x[7:10]
        omega = x[10:13]
        force_torque = self.thrust_to_force_torque(u)
        thrust_total = force_torque[0]
        body_torque = force_torque[1:]

        R = _quat_to_rotmat(q)
        if _is_numpy(x):
            gravity = np.array([0.0, 0.0, -self.gravity], dtype=x.dtype)
            thrust_world = R @ np.array([0.0, 0.0, thrust_total], dtype=x.dtype)
            inertia = self.inertia.astype(x.dtype, copy=False)
            omega_dot = (_cross(-omega, inertia * omega) + body_torque) / inertia
            quat_dot = _quat_derivative(q, omega)
            vel_dot = gravity + thrust_world / self.mass
            return np.hstack((vel, quat_dot, vel_dot, omega_dot))

        gravity = torch.tensor([0.0, 0.0, -self.gravity],
                               dtype=x.dtype,
                               device=x.device)
        thrust_world = R @ torch.tensor([0.0, 0.0, 1.0],
                                        dtype=x.dtype,
                                        device=x.device) * thrust_total
        inertia = torch.tensor(self.inertia, dtype=x.dtype, device=x.device)
        omega_dot = (_cross(-omega, inertia * omega) + body_torque) / inertia
        quat_dot = _quat_derivative(q, omega)
        vel_dot = gravity + thrust_world / self.mass
        return torch.cat((vel, quat_dot, vel_dot, omega_dot), dim=0)

    def step_forward(self, x, u, dt):
        """
        Forward-Euler integration with quaternion renormalization.
        """
        x_next = x + self.dynamics(x, u) * dt
        if _is_numpy(x_next):
            x_next = x_next.copy()
            x_next[3:7] = _quat_normalize(x_next[3:7])
            return x_next
        x_next = x_next.clone()
        x_next[3:7] = _quat_normalize(x_next[3:7])
        return x_next
