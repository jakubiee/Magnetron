from dataclasses import dataclass


@dataclass
class Geometry:
    """Geometry parameters for the magnetron simulation."""

    # Target dimensions
    target_diameter: float = 50.8
    target_thickness: float = 3

    # Magnets
    magnet_height: float = 10
    inner_magnet_diameter: float = 18
    outer_magnet_diameter: float = 10
    outer_magnet_ring_radius: float = 22
    magnet_to_target_distance: float = 5


geometry = Geometry()
