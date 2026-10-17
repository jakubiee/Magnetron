
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.interpolate import griddata

if TYPE_CHECKING:
    from src.magnetron.magnetostatics import SimulationResult


def plot_magnetic_field(
    result: SimulationResult,
    output_path,
    *,
    grid_shape: tuple[int, int] = (300, 250),
    target_diameter: float = 50.8,
    target_thickness: float = 3.0,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    points = result.domain.geometry.x
    x, y = points[:, 0], points[:, 1]
    nx, ny = grid_shape
    x_grid = np.linspace(x.min(), x.max(), nx)
    y_grid = np.linspace(y.min(), y.max(), ny)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

    def interpolate(values: np.ndarray) -> np.ndarray:
        return griddata((x, y), values, (x_mesh, y_mesh), method="linear")

    bx = interpolate(result.bx.x.array)
    by = interpolate(result.by.x.array)
    b_magnitude = interpolate(result.b_magnitude.x.array)

    figure, axis = plt.subplots(figsize=(12, 7))
    contour = axis.contourf(
        x_mesh, y_mesh, b_magnitude, levels=100, cmap="turbo"
    )
    axis.streamplot(
        x_mesh,
        y_mesh,
        bx,
        by,
        density=3,
        color="black",
        linewidth=0.8,
        arrowsize=0.8,
        arrowstyle="-|>",
    )
    target = Rectangle(
        (-target_diameter / 2, 0),
        target_diameter,
        target_thickness,
        fill=False,
        edgecolor="black",
        linewidth=1.5,
        zorder=10,
    )
    axis.add_patch(target)
    colorbar = figure.colorbar(contour, ax=axis)
    colorbar.set_label("|B| [T]")
    axis.set_aspect("equal")
    axis.set_xlabel("x [mm]")
    axis.set_ylabel("y [mm  ]")
    axis.set_title("Magnetic field magnitude, field lines and target")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def target_surface_field_profile(
    result: SimulationResult,
    *,
    target_diameter: float = 50.8,
    target_thickness: float = 3.0,
    samples: int = 300,
):
    if samples < 2:
        raise ValueError("samples must be at least 2")

    points = result.domain.geometry.x
    radius = np.linspace(0.0, target_diameter / 2, samples)
    surface_y = np.full_like(radius, target_thickness)

    def interpolate(values: np.ndarray) -> np.ndarray:
        return griddata(
            (points[:, 0], points[:, 1]),
            values,
            (radius, surface_y),
            method="linear",
        )

    b_parallel = interpolate(result.bx.x.array)
    b_perpendicular = interpolate(result.by.x.array)
    return radius, b_parallel, b_perpendicular


def plot_target_surface_profile(
    result: SimulationResult,
    output_path: str | Path,
    *,
    target_diameter: float = 50.8,
    target_thickness: float = 3.0,
    samples: int = 300,
):
    """Save ``B_parallel`` and ``B_perpendicular`` on target surface."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    radius, b_parallel, b_perpendicular = target_surface_field_profile(
        result,
        target_diameter=target_diameter,
        target_thickness=target_thickness,
        samples=samples,
    )

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(radius, b_parallel, label=r"$B_{\parallel}$", linewidth=2)
    axis.plot(radius, b_perpendicular, label=r"$B_{\perp}$", linewidth=2)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("r [mm]")
    axis.set_ylabel("B [T]")
    axis.set_title("Magnetic-field profile on target surface")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return radius, b_parallel, b_perpendicular
