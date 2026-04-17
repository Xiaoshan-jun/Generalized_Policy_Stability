"""
dynamics_gradient_aviary.py
============================
Standalone dynamics and Jacobian computation for the BaseAviary quadrotor.

The BaseAviary._dynamics() is coupled to PyBullet state management, so we
re-implement the pure math here as a stateless numpy function, then use
finite-difference to obtain the Jacobians A = df/dx and B = df/du.

State layout (13D):
    x = [pos_x, pos_y, pos_z,          # 0:3
         qx, qy, qz, qw,               # 3:7  (quaternion, scalar-last)
         vel_x, vel_y, vel_z,           # 7:10
         p, q, r]                       # 10:13  body angular rates

Input (4D):
    u = [rpm0, rpm1, rpm2, rpm3]

Usage
-----
    from dynamics_gradient_aviary import AviaryParams, aviary_dynamics, aviary_gradient

    params = AviaryParams()          # defaults match CF2X from gym-pybullet-drones
    x0    = hover_state(params)      # equilibrium state
    u0    = hover_rpm(params)        # hover RPMs

    f  = aviary_dynamics(x0, u0, params)       # 13D derivative
    A, B = aviary_gradient(x0, u0, params)     # (13,13), (13,4) Jacobians
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation


# ============================================================================
# Physical parameters
# ============================================================================

@dataclass
class AviaryParams:
    """Physical parameters matching BaseAviary / CF2X defaults."""
    M:  float = 0.027          # kg  — Crazyflie 2.x mass
    G:  float = 9.8            # m/s²
    L:  float = 0.0397         # m   — arm length
    KF: float = 3.16e-10       # N/(RPM²) — thrust coefficient
    KM: float = 7.94e-12       # N·m/(RPM²) — torque coefficient
    Ixx: float = 1.4e-5        # kg·m²
    Iyy: float = 1.4e-5
    Izz: float = 2.17e-5
    model: str = "CF2X"        # "CF2X" | "CF2P" | "RACE"
    dt: float = 1.0 / 240.0   # integration timestep (PyBullet default)

    @property
    def J(self) -> np.ndarray:
        return np.diag([self.Ixx, self.Iyy, self.Izz])

    @property
    def J_INV(self) -> np.ndarray:
        return np.diag([1.0/self.Ixx, 1.0/self.Iyy, 1.0/self.Izz])

    @property
    def GRAVITY(self) -> float:
        return self.G * self.M

    @property
    def hover_rpm(self) -> float:
        """Single-motor RPM for steady hover (total thrust = weight)."""
        return float(np.sqrt(self.GRAVITY / (4.0 * self.KF)))


# ============================================================================
# Quaternion utilities
# ============================================================================

def quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [qx, qy, qz, qw] (scalar-last) to 3×3 rotation matrix.
    Matches scipy / PyBullet convention.
    """
    return Rotation.from_quat(quat).as_matrix()


def integrate_quat(quat: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """
    Quaternion integration via matrix exponential (matches BaseAviary._integrateQ).

    quat  : [qx, qy, qz, qw]  scalar-last
    omega : body angular rates [p, q, r]
    """
    omega_norm = np.linalg.norm(omega)
    if np.isclose(omega_norm, 0.0):
        return quat.copy()
    p_, q_, r_ = omega
    # BaseAviary uses scalar-first ordering internally in _integrateQ,
    # but the stored quaternion is PyBullet scalar-last [qx,qy,qz,qw].
    # We replicate the exact matrix used in _integrateQ.
    lambda_ = 0.5 * np.array([
        [ 0,   r_, -q_,  p_],
        [-r_,  0,   p_,  q_],
        [ q_, -p_,  0,   r_],
        [-p_, -q_, -r_,  0 ]
    ])
    theta = omega_norm * dt / 2.0
    quat_new = (np.eye(4) * np.cos(theta) +
                2.0 / omega_norm * lambda_ * np.sin(theta)) @ quat
    return quat_new / np.linalg.norm(quat_new)   # keep unit quaternion


# ============================================================================
# Pure continuous-time dynamics  (ẋ = f(x, u))
# ============================================================================

def aviary_dynamics(x: np.ndarray, rpm: np.ndarray, p: AviaryParams) -> np.ndarray:
    """
    Continuous-time state derivative  ẋ = f(x, u).

    Parameters
    ----------
    x   : (13,) state  [pos(3), quat(4), vel(3), rates(3)]
    rpm : (4,)  motor RPMs
    p   : AviaryParams

    Returns
    -------
    xdot : (13,) state derivative
    """
    pos      = x[0:3]
    quat     = x[3:7]
    vel      = x[7:10]
    rates    = x[10:13]   # body angular rates [p, q, r]

    R = quat_to_rotmat(quat)   # 3×3 rotation matrix (body → world)

    # ── Thrust & torques ────────────────────────────────────────────────
    forces   = rpm**2 * p.KF                          # (4,) per-motor thrust [N]
    thrust_b = np.array([0.0, 0.0, np.sum(forces)])   # total thrust in body frame
    thrust_w = R @ thrust_b                            # thrust in world frame
    grav_w   = np.array([0.0, 0.0, p.GRAVITY])
    acc      = (thrust_w - grav_w) / p.M              # world-frame acceleration

    z_torques = rpm**2 * p.KM
    if p.model == "RACE":
        z_torques = -z_torques
    z_torque = -z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3]

    if p.model == "RACE":
        x_torque = ( forces[0] + forces[1] - forces[2] - forces[3]) * (p.L / np.sqrt(2))
        y_torque = (-forces[0] + forces[1] + forces[2] - forces[3]) * (p.L / np.sqrt(2))
    elif p.model == "CF2X":
        x_torque = -(forces[0] + forces[1] - forces[2] - forces[3]) * (p.L / np.sqrt(2))
        y_torque = (-forces[0] + forces[1] + forces[2] - forces[3]) * (p.L / np.sqrt(2))
    elif p.model == "CF2P":
        x_torque = (forces[1] - forces[3]) * p.L
        y_torque = (-forces[0] + forces[2]) * p.L
    else:
        raise ValueError(f"Unknown model: {p.model}")

    torques  = np.array([x_torque, y_torque, z_torque])
    torques -= np.cross(rates, p.J @ rates)           # gyroscopic coupling
    rates_dot = p.J_INV @ torques                     # angular acceleration

    # ── Quaternion kinematics ────────────────────────────────────────────
    # q̇ = 0.5 * Ξ(q) * ω   (standard quaternion derivative)
    # Here we use the same matrix form as _integrateQ but as a rate:
    pr, qr, rr = rates
    Xi = 0.5 * np.array([
        [ 0,   rr, -qr,  pr],
        [-rr,  0,   pr,  qr],
        [ qr, -pr,  0,   rr],
        [-pr, -qr, -rr,  0 ]
    ])
    quat_dot = Xi @ quat

    xdot = np.concatenate([
        vel,          # pos_dot  = vel
        quat_dot,     # quat_dot = 0.5 * Ξ(ω) * q
        acc,          # vel_dot  = acceleration
        rates_dot,    # rates_dot = J⁻¹(τ - ω × Jω)
    ])
    return xdot


# ============================================================================
# Discrete-time dynamics  x_{t+1} = f_d(x_t, u_t)
# ============================================================================

def aviary_dynamics_discrete(x: np.ndarray, rpm: np.ndarray,
                              params: AviaryParams) -> np.ndarray:
    """
    Euler-integrated discrete step matching BaseAviary._dynamics().

    Returns x_{t+1}.
    """
    dt   = params.dt
    xdot = aviary_dynamics(x, rpm, params)

    pos_new   = x[0:3]  + dt * xdot[0:3]
    quat_new  = integrate_quat(x[3:7], x[10:13], dt)   # matrix-exp integration
    vel_new   = x[7:10] + dt * xdot[7:10]
    rates_new = x[10:13]+ dt * xdot[10:13]

    x_new = np.concatenate([pos_new, quat_new, vel_new, rates_new])
    return x_new


# ============================================================================
# Jacobians via finite difference
# ============================================================================

def aviary_gradient(
    x: np.ndarray,
    rpm: np.ndarray,
    params: AviaryParams,
    eps: float = 1e-5,
    discrete: bool = True,
) -> tuple:
    """
    Compute Jacobians of the dynamics at (x, rpm).

    Parameters
    ----------
    x        : (13,) state
    rpm      : (4,)  motor RPMs
    params   : AviaryParams
    eps      : finite-difference step size
    discrete : if True, differentiate the discrete-time map x_{t+1} = f_d(x,u)
               if False, differentiate the continuous-time derivative ẋ = f(x,u)

    Returns
    -------
    A : (13, 13)  df/dx
    B : (13,  4)  df/drpm
    """
    f = aviary_dynamics_discrete if discrete else aviary_dynamics
    nx = x.shape[0]
    nu = rpm.shape[0]
    nf = f(x, rpm, params).shape[0]

    A = np.zeros((nf, nx))
    B = np.zeros((nf, nu))

    f0 = f(x, rpm, params)

    for i in range(nx):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        # Re-normalise quaternion after perturbation
        xp[3:7] /= np.linalg.norm(xp[3:7])
        xm[3:7] /= np.linalg.norm(xm[3:7])
        A[:, i] = (f(xp, rpm, params) - f(xm, rpm, params)) / (2 * eps)

    for j in range(nu):
        up = rpm.copy(); up[j] += eps
        um = rpm.copy(); um[j] -= eps
        B[:, j] = (f(x, up, params) - f(x, um, params)) / (2 * eps)

    return A, B


# ============================================================================
# Convenience: equilibrium state and RPMs
# ============================================================================

def hover_state(params: AviaryParams, z: float = 0.0) -> np.ndarray:
    """Return the hover equilibrium state at altitude z."""
    x = np.zeros(13)
    x[2] = z                 # pos_z = z
    x[6] = 1.0               # quat = [0,0,0,1] (identity, scalar-last)
    return x


def hover_rpm(params: AviaryParams) -> np.ndarray:
    """Return the 4-motor RPM vector for steady hover."""
    return np.full(4, params.hover_rpm)


# ============================================================================
# Quick test / demo
# ============================================================================

if __name__ == "__main__":
    params = AviaryParams()
    x0  = hover_state(params, z=1.0)
    u0  = hover_rpm(params)

    print("=== AviaryParams ===")
    print(f"  mass       : {params.M} kg")
    print(f"  hover RPM  : {params.hover_rpm:.1f}")
    print(f"  dt         : {params.dt:.6f} s")

    xdot = aviary_dynamics(x0, u0, params)
    print("\n=== Continuous-time ẋ at hover (should be ≈ 0) ===")
    labels = ["ṗx","ṗy","ṗz","q̇x","q̇y","q̇z","q̇w","v̇x","v̇y","v̇z","ṗ","q̇","ṙ"]
    for lbl, v in zip(labels, xdot):
        print(f"  {lbl:6s} = {v:+.6e}")

    A, B = aviary_gradient(x0, u0, params, discrete=False)
    print(f"\n=== Jacobians (continuous) at hover ===")
    print(f"  A shape: {A.shape}   B shape: {B.shape}")
    print(f"\n  A =\n{np.array2string(A, precision=3, suppress_small=True)}")
    print(f"\n  B =\n{np.array2string(B, precision=3, suppress_small=True)}")
