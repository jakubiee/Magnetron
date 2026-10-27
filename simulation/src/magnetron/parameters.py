import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class GeometryParameters:
    target_diameter: float = 0.0508
    target_thickness: float = 0.003
    magnet_height: float = 0.010
    inner_magnet_diameter: float = 0.020
    outer_magnet_diameter: float = 0.010
    outer_magnet_ring_radius: float = 0.025
    magnet_to_target_distance: float = 0.005
    anode_cathode_gap: float = 0.003
    anode_width: float = 0.004


@dataclass(frozen=True)
class ElectrostaticParameters:
    target_voltage: float = -400.0
    ground_voltage: float = 0.0
    eps_r_air: float = 1.0
    eps_0: float = 8.8541878128e-12


@dataclass(frozen=True)
class MeshSettings:
    min_size: float = 0.01
    max_size: float = 1.0


@dataclass(frozen=True)
class MagneticParameters:
    remanence: float = 1.3
    mu_r_air: float = 1.0
    mu_r_magnet: float = 1.05
    mu_r_backing: float = 1000.0
    mu_0: float = 4e-7 * np.pi
