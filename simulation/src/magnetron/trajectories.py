from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp


ELEMENTARY_CHARGE = 1.602176634e-19
ELECTRON_MASS = 9.1093837139e-31
BOLTZMANN_CONSTANT = 1.380649e-23
ARGON_ATOMIC_MASS = 39.948 * 1.66053906660e-27


@dataclass
class ArgonGas:
    pressure_pa: float = 1.0
    temperature_k: float = 300.0

    def number_density(self) -> float:
        if self.pressure_pa <= 0 or self.temperature_k <= 0:
            raise ValueError("Pressure and temperature must be positive.")

        return self.pressure_pa / (BOLTZMANN_CONSTANT * self.temperature_k)


@dataclass
class CrossSectionTable:
    energy_ev: np.ndarray
    sigma_m2: np.ndarray

    def __post_init__(self):
        self.energy_ev = np.asarray(self.energy_ev, dtype=float)
        self.sigma_m2 = np.asarray(self.sigma_m2, dtype=float)
        self.validate()

    def validate(self) -> None:
        if self.energy_ev.ndim != 1 or self.sigma_m2.ndim != 1:
            raise ValueError("Cross-section data must be one-dimensional.")

        if len(self.energy_ev) != len(self.sigma_m2) or len(self.energy_ev) < 2:
            raise ValueError("Invalid cross-section table.")

        if not np.all(np.isfinite(self.energy_ev)) or not np.all(
            np.isfinite(self.sigma_m2)
        ):
            raise ValueError("Cross-section data must contain only finite values.")

        if np.any(np.diff(self.energy_ev) <= 0):
            raise ValueError("Energy values must be strictly increasing.")

        if np.any(self.energy_ev < 0) or np.any(self.sigma_m2 < 0):
            raise ValueError("Energy and cross sections must be nonnegative.")

    def get(self, energy_ev: float) -> float:

        return float(
            np.interp(
                energy_ev,
                self.energy_ev,
                self.sigma_m2,
                left=0.0,
                right=0.0,
            )
        )


@dataclass
class ArgonCollisionData:
    elastic: CrossSectionTable
    excitation: CrossSectionTable
    ionization: CrossSectionTable
    excitation_threshold_ev: float = 11.55
    ionization_threshold_ev: float = 15.76

    def cross_sections(self, energy_ev: float) -> np.ndarray:
        elastic = self.elastic.get(energy_ev)

        excitation = 0.0
        if energy_ev >= self.excitation_threshold_ev:
            excitation = self.excitation.get(energy_ev)

        ionization = 0.0
        if energy_ev >= self.ionization_threshold_ev:
            ionization = self.ionization.get(energy_ev)

        return np.array([elastic, excitation, ionization], dtype=float)

    def total_cross_section(self, energy_ev: float) -> float:
        total = self.elastic.get(energy_ev)
        if energy_ev >= self.excitation_threshold_ev:
            total += self.excitation.get(energy_ev)
        if energy_ev >= self.ionization_threshold_ev:
            total += self.ionization.get(energy_ev)
        return total


def make_argon_collision_data() -> ArgonCollisionData:
    scale = 1e-20

    elastic = CrossSectionTable(
        energy_ev=np.array(
            [
                0.0,
                0.01,
                0.1,
                0.3,
                0.5,
                1.0,
                2.0,
                5.0,
                10.0,
                15.0,
                20.0,
                30.0,
                50.0,
                100.0,
                200.0,
                500.0,
                1000.0,
            ]
        ),
        sigma_m2=scale
        * np.array(
            [
                6.0,
                6.0,
                3.0,
                0.15,
                0.8,
                2.5,
                4.0,
                5.0,
                4.5,
                4.1,
                3.7,
                3.2,
                2.7,
                2.0,
                1.5,
                1.0,
                0.7,
            ]
        ),
    )

    excitation = CrossSectionTable(
        energy_ev=np.array(
            [
                0.0,
                11.55,
                11.62,
                12.0,
                14.0,
                16.0,
                18.0,
                20.0,
                25.0,
                30.0,
                35.0,
                40.0,
                45.0,
                50.0,
                60.0,
                70.0,
                80.0,
                100.0,
                200.0,
                300.0,
                500.0,
                1000.0,
            ]
        ),
        sigma_m2=scale
        * np.array(
            [
                0.0,
                0.0,
                0.001,
                0.003,
                0.035,
                0.097,
                0.175,
                0.256,
                0.428,
                0.542,
                0.615,
                0.658,
                0.683,
                0.696,
                0.700,
                0.689,
                0.670,
                0.627,
                0.454,
                0.357,
                0.254,
                0.154,
            ]
        ),
    )

    ionization = CrossSectionTable(
        energy_ev=np.array(
            [
                0.0,
                15.76,
                16.0,
                18.0,
                20.0,
                25.0,
                30.0,
                40.0,
                50.0,
                75.0,
                100.0,
                150.0,
                200.0,
                300.0,
                500.0,
                1000.0,
            ]
        ),
        sigma_m2=scale
        * np.array(
            [
                0.0,
                0.0,
                0.03,
                0.15,
                0.40,
                0.90,
                1.50,
                2.20,
                2.50,
                2.90,
                3.00,
                2.60,
                2.30,
                1.80,
                1.30,
                0.80,
            ]
        ),
    )

    return ArgonCollisionData(
        elastic=elastic,
        excitation=excitation,
        ionization=ionization,
        excitation_threshold_ev=11.55,
        ionization_threshold_ev=15.76,
    )


@dataclass
class ParticleBirth:
    time: float
    position: np.ndarray
    velocity: np.ndarray
    species: str


@dataclass
class CollisionEvent:
    time: float
    position: np.ndarray
    energy_before_ev: float
    energy_after_ev: float
    kind: str


@dataclass
class TrajectoryResult:
    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    status: str
    message: str
    collisions: list[CollisionEvent] = field(default_factory=list)
    products: list[ParticleBirth] = field(default_factory=list)


@dataclass
class ElectronCascadeResult:
    trajectories: list[TrajectoryResult] = field(default_factory=list)
    truncated: bool = False

    @property
    def collisions(self) -> list[CollisionEvent]:
        return [
            collision
            for trajectory in self.trajectories
            for collision in trajectory.collisions
        ]

    @property
    def ionizations(self) -> list[CollisionEvent]:
        return [
            collision for collision in self.collisions if collision.kind == "ionization"
        ]


class ElectronTrajectory:
    def __init__(
        self,
        electric_field: Callable,
        magnetic_field: Callable,
        inside_domain: Callable | None = None,
        gas: ArgonGas | None = None,
        collision_data: ArgonCollisionData | None = None,
        seed: int | None = None,
    ):
        self.electric_field = electric_field
        self.magnetic_field = magnetic_field
        self.inside_domain = inside_domain
        self.gas = gas
        self.collision_data = collision_data
        self.rng = np.random.default_rng(seed)

        if (gas is None) != (collision_data is None):
            raise ValueError("Provide both gas and collision_data.")

        self.gas_density = gas.number_density() if gas is not None else None

        if collision_data is not None:
            collision_data.elastic.validate()
            collision_data.excitation.validate()
            collision_data.ionization.validate()

    def _inside(self, x: float, y: float) -> bool:
        if self.inside_domain is None:
            return True
        return bool(self.inside_domain(x, y))

    def _rhs(self, time: float, state: np.ndarray) -> np.ndarray:
        x, y, vx, vy, vz = state[:5]

        if not self._inside(x, y):
            return np.array([vx, vy, 0.0, 0.0, 0.0])

        E = self.electric_field(x, y)
        B = self.magnetic_field(x, y)

        if len(E) != 3 or len(B) != 3:
            raise ValueError(
                "Electric and magnetic field callbacks must return 3 components."
            )

        if not all(np.isfinite(component) for component in (*E, *B)):
            raise ValueError(f"Non-finite field encountered at x={x}, y={y}.")

        ex, ey, ez = E
        bx, by, bz = B
        charge_to_mass = -ELEMENTARY_CHARGE / ELECTRON_MASS

        derivative = np.empty(5, dtype=float)
        derivative[0] = vx
        derivative[1] = vy
        derivative[2] = charge_to_mass * (ex + vy * bz - vz * by)
        derivative[3] = charge_to_mass * (ey + vz * bx - vx * bz)
        derivative[4] = charge_to_mass * (ez + vx * by - vy * bx)
        return derivative

    @staticmethod
    def _energy_ev(velocity: np.ndarray) -> float:
        speed = np.linalg.norm(velocity)
        return float(0.5 * ELECTRON_MASS * speed**2 / ELEMENTARY_CHARGE)

    def _rhs_with_collisions(self, time: float, state: np.ndarray) -> np.ndarray:
        derivative = self._rhs(time, state[:5])
        velocity = state[2:5]
        speed = np.sqrt(np.dot(velocity, velocity))
        energy_ev = 0.5 * ELECTRON_MASS * speed**2 / ELEMENTARY_CHARGE
        total_cross_section = self.collision_data.total_cross_section(energy_ev)
        collision_rate = self.gas_density * total_cross_section * speed

        extended_derivative = np.empty(6, dtype=float)
        extended_derivative[:5] = derivative
        extended_derivative[5] = collision_rate
        return extended_derivative

    def _random_direction(self) -> np.ndarray:
        z = self.rng.uniform(-1.0, 1.0)
        phi = self.rng.uniform(0.0, 2.0 * np.pi)
        radius = np.sqrt(1.0 - z**2)

        return np.array(
            [
                radius * np.cos(phi),
                radius * np.sin(phi),
                z,
            ]
        )

    def _scatter_direction(self, velocity: np.ndarray) -> np.ndarray:
        speed = np.linalg.norm(velocity)
        if speed == 0.0:
            return self._random_direction()

        direction = velocity / speed
        reference = (
            np.array([0.0, 0.0, 1.0])
            if abs(direction[2]) < 0.9
            else np.array([1.0, 0.0, 0.0])
        )

        e1 = np.cross(direction, reference)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(direction, e1)

        theta = np.deg2rad(20.0) * self.rng.random()
        phi = self.rng.uniform(0.0, 2.0 * np.pi)

        new_direction = np.cos(theta) * direction + np.sin(theta) * (
            np.cos(phi) * e1 + np.sin(phi) * e2
        )
        return new_direction / np.linalg.norm(new_direction)

    @staticmethod
    def _velocity_from_energy(energy_ev: float, direction: np.ndarray) -> np.ndarray:
        energy_ev = max(0.0, energy_ev)
        speed = np.sqrt(2.0 * energy_ev * ELEMENTARY_CHARGE / ELECTRON_MASS)
        return speed * direction

    def _handle_collision(
        self,
        time: float,
        state: np.ndarray,
        collisions: list[CollisionEvent],
        products: list[ParticleBirth],
    ) -> np.ndarray:
        state = state.copy()
        position = state[:2].copy()
        velocity = state[2:5].copy()
        energy_before = self._energy_ev(velocity)

        cross_sections = self.collision_data.cross_sections(energy_before)
        total = float(np.sum(cross_sections))
        if total <= 0.0:
            return state

        probabilities = cross_sections / total
        kind = self.rng.choice(
            ["elastic", "excitation", "ionization"],
            p=probabilities,
        )

        if kind == "elastic":
            energy_after = energy_before

        elif kind == "excitation":
            energy_after = max(
                0.0,
                energy_before - self.collision_data.excitation_threshold_ev,
            )
            products.append(
                ParticleBirth(
                    time=time,
                    position=position.copy(),
                    velocity=np.zeros(3),
                    species="Ar*",
                )
            )

        else:
            available_energy = max(
                0.0,
                energy_before - self.collision_data.ionization_threshold_ev,
            )
            energy_after = 0.5 * available_energy
            secondary_energy = 0.5 * available_energy

            secondary_velocity = self._velocity_from_energy(
                secondary_energy,
                self._random_direction(),
            )
            products.append(
                ParticleBirth(
                    time=time,
                    position=position.copy(),
                    velocity=secondary_velocity,
                    species="electron",
                )
            )
            products.append(
                ParticleBirth(
                    time=time,
                    position=position.copy(),
                    velocity=np.zeros(3),
                    species="Ar+",
                )
            )

        new_direction = self._scatter_direction(velocity)
        state[2:5] = self._velocity_from_energy(energy_after, new_direction)

        collisions.append(
            CollisionEvent(
                time=time,
                position=position,
                energy_before_ev=energy_before,
                energy_after_ev=energy_after,
                kind=str(kind),
            )
        )

        return state

    def solve(
        self,
        initial_position,
        initial_velocity,
        duration,
        *,
        max_step,
        rtol=1e-8,
        atol=1e-10,
    ) -> TrajectoryResult:
        x0, y0 = map(float, initial_position)
        vx0, vy0, vz0 = map(float, initial_velocity)

        if duration <= 0:
            raise ValueError("Duration must be positive.")
        if max_step <= 0:
            raise ValueError("max_step must be positive.")

        state = np.array([x0, y0, vx0, vy0, vz0], dtype=float)

        if not self._inside(x0, y0):
            raise ValueError("Initial position is outside the simulation domain.")

        def left_domain(time, event_state):
            return 1.0 if self._inside(event_state[0], event_state[1]) else -1.0

        left_domain.terminal = True
        left_domain.direction = -1

        if self.gas is None:
            events = [left_domain] if self.inside_domain is not None else None
            solution = solve_ivp(
                self._rhs,
                (0.0, duration),
                state,
                method="DOP853",
                max_step=max_step,
                rtol=rtol,
                atol=atol,
                events=events,
            )

            if not solution.success:
                raise RuntimeError(solution.message)

            status = "left_domain" if solution.status == 1 else "completed"
            message = (
                "Trajectory left the simulation domain."
                if status == "left_domain"
                else "Trajectory reached the requested duration."
            )

            return TrajectoryResult(
                time=solution.t,
                position=solution.y[:2].T,
                velocity=solution.y[2:5].T,
                status=status,
                message=message,
            )

        times: list[float] = []
        states: list[np.ndarray] = []
        collisions: list[CollisionEvent] = []
        products: list[ParticleBirth] = []

        current_time = 0.0
        status = "completed"

        while current_time < duration:
            if np.linalg.norm(state[2:5]) < 1e3:
                status = "thermalized"
                break

            random_number = max(self.rng.random(), np.finfo(float).tiny)
            collision_threshold = -np.log(random_number)
            extended_state = np.append(state, 0.0)

            events = []
            domain_event_index = None

            if self.inside_domain is not None:
                domain_event_index = len(events)
                events.append(left_domain)

            def collision_reached(time, event_state):
                return event_state[5] - collision_threshold

            collision_reached.terminal = True
            collision_reached.direction = 1
            collision_event_index = len(events)
            events.append(collision_reached)

            solution = solve_ivp(
                self._rhs_with_collisions,
                (current_time, duration),
                extended_state,
                method="DOP853",
                max_step=max_step,
                rtol=rtol,
                atol=atol,
                events=events,
            )

            if not solution.success:
                raise RuntimeError(solution.message)

            times.extend(solution.t.tolist())
            states.extend(row.copy() for row in solution.y[:5].T)

            current_time = float(solution.t[-1])
            state = solution.y[:5, -1].copy()

            if (
                domain_event_index is not None
                and len(solution.t_events[domain_event_index]) > 0
            ):
                status = "left_domain"
                break

            collision_occurred = len(solution.t_events[collision_event_index]) > 0
            if not collision_occurred:
                status = "completed"
                break

            state = self._handle_collision(
                current_time,
                state,
                collisions,
                products,
            )

        messages = {
            "completed": "Trajectory reached the requested duration.",
            "left_domain": "Trajectory left the simulation domain.",
            "thermalized": "Electron energy became very low in this simplified model.",
        }

        if states:
            states_array = np.asarray(states, dtype=float)
            times_array = np.asarray(times, dtype=float)
        else:
            states_array = state.reshape(1, -1)
            times_array = np.array([current_time], dtype=float)

        return TrajectoryResult(
            time=times_array,
            position=states_array[:, :2],
            velocity=states_array[:, 2:5],
            status=status,
            message=messages.get(status, status),
            collisions=collisions,
            products=products,
        )

    def solve_cascade(
        self,
        initial_position,
        initial_velocity,
        duration,
        *,
        max_step,
        max_electrons=1000,
        rtol=1e-8,
        atol=1e-10,
    ) -> ElectronCascadeResult:
        if self.gas is None:
            raise ValueError("Electron cascades require gas and collision data.")
        if duration <= 0:
            raise ValueError("Duration must be positive.")
        if max_step <= 0:
            raise ValueError("max_step must be positive.")
        if max_electrons < 1:
            raise ValueError("max_electrons must be at least 1.")

        pending = deque(
            [
                (
                    0.0,
                    np.asarray(initial_position, dtype=float),
                    np.asarray(initial_velocity, dtype=float),
                )
            ]
        )
        cascade = ElectronCascadeResult()

        while pending and len(cascade.trajectories) < max_electrons:
            birth_time, position, velocity = pending.popleft()
            remaining_time = duration - birth_time
            if remaining_time <= 0:
                continue

            trajectory = self.solve(
                position,
                velocity,
                remaining_time,
                max_step=max_step,
                rtol=rtol,
                atol=atol,
            )
            trajectory.time += birth_time

            for collision in trajectory.collisions:
                collision.time += birth_time

            for product in trajectory.products:
                product.time += birth_time
                if product.species == "electron" and product.time < duration:
                    pending.append((product.time, product.position, product.velocity))

            cascade.trajectories.append(trajectory)

        cascade.truncated = bool(pending)
        return cascade
