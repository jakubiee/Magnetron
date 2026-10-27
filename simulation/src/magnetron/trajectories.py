from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

ELEMENTARY_CHARGE = 1.602176634e-19
ELECTRON_MASS = 9.1093837139e-31  

@dataclass
class TrajectoryResult:
    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray

class ElectronTrajectory:
    def __init__(self, electric_field: Callable, magnetic_field: Callable):
        self.electric_field = electric_field
        self.magnetic_field = magnetic_field

    def _rhs(self, time, state):
        x, y, vx, vy, vz = state

        E = np.asarray(self.electric_field(x, y), dtype=float)
        B = np.asarray(self.magnetic_field(x, y), dtype=float)
        velocity = np.array([vx, vy, vz])

        acceleration = (-ELEMENTARY_CHARGE / ELECTRON_MASS * (E + np.cross(velocity, B)))

        return np.array([vx, vy, acceleration[0], acceleration[1], acceleration[2]])

    def solve(self, initial_position, initial_velocity, duration, *, max_step, rtol=1e-8, atol=1e-10):
        x0, y0 = initial_position
        vx0, vy0, vz0 = initial_velocity

        initial_state = np.array([x0, y0, vx0, vy0, vz0], dtype=float)

        solution = solve_ivp(self._rhs, (0.0, duration), initial_state, method="DOP853", max_step=max_step, rtol=rtol, atol=atol)

        if not solution.success:
            raise RuntimeError(solution.message)

        return TrajectoryResult(time=solution.t, position=solution.y[:2].T, velocity=solution.y[2:].T)