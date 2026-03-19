#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pendulum environment with bounded uncertain disturbance torque."""

from __future__ import annotations

import numpy as np
from gymnasium.envs.classic_control.pendulum import PendulumEnv, angle_normalize


class UncertainDisturbancePendulumEnv(PendulumEnv):
    """
    Pendulum dynamics with additive uncertain disturbance torque.

    Dynamics:
        theta_ddot = f(theta, theta_dot, u + d)
    where d is sampled each step from U[-disturbance_max, disturbance_max].
    """

    def __init__(
        self,
        disturbance_max: float = 0.5,
        b: float = 0.13,
        g: float = 9.81,
        m: float = 1.0,
        l: float = 1.0,
        render_mode=None,
        **kwargs,
    ):
        super().__init__(render_mode=render_mode, g=g, **kwargs)
        self.disturbance_max = float(disturbance_max)
        self.b = float(b)
        self.m = float(m)
        self.l = float(l)

    def step(self, u):
        th, thdot = self.state
        u = float(np.clip(u, -self.max_torque, self.max_torque)[0])
        disturbance = float(
            self.np_random.uniform(-self.disturbance_max, self.disturbance_max)
        )

        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)
        newthdot = thdot + (
            3 * self.g / (2 * self.l) * np.sin(th)
            + 3.0 / (self.m * self.l**2) * (u + disturbance)
            - self.b * thdot
        ) * self.dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * self.dt

        self.state = np.array([newth, newthdot], dtype=np.float64)
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), -costs, False, False, {"disturbance": disturbance}
