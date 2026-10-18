from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.interpolate import griddata
from .magnetostatic import SimulationResult
from .electrostatic import ElectrostaticResult
from dolfinx import plot as dolfinx_plot


def _interpolate_field(
    result: SimulationResult, values: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray
):
    points = result.domain.geometry.x[:, :2]
    values = np.asarray(values).reshape(-1)

    if len(points) != len(values):
        raise ValueError(
            "Field values must correspond to mesh vertices. "
            f"Got {len(values)} values for {len(points)} vertices."
        )

    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

    return griddata(
        points,
        values,
        (x_mesh, y_mesh),
        method="linear",
    )


def plot_magnetic_field(
    result: SimulationResult,
    output_path: str | Path,
    *,
    grid_shape: tuple[int, int] = (300, 250),
    target_diameter: float = 50.8,
    target_thickness: float = 3.0,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if min(grid_shape) < 2:
        raise ValueError("Both grid dimensions must be at least 2.")

    points = result.domain.geometry.x[:, :2]
    x, y = points[:, 0], points[:, 1]

    nx, ny = grid_shape
    x_grid = np.linspace(x.min(), x.max(), nx)
    y_grid = np.linspace(y.min(), y.max(), ny)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

    bx = _interpolate_field(result, result.bx.x.array, x_grid, y_grid)
    by = _interpolate_field(result, result.by.x.array, x_grid, y_grid)
    b_magnitude = _interpolate_field(result, result.b_magnitude.x.array, x_grid, y_grid)

    figure, axis = plt.subplots(figsize=(12, 7))

    contour = axis.contourf(
        x_mesh,
        y_mesh,
        b_magnitude,
        levels=100,
        cmap="turbo",
    )

    bx_stream = np.nan_to_num(bx, nan=0.0)
    by_stream = np.nan_to_num(by, nan=0.0)

    axis.streamplot(
        x_grid,
        y_grid,
        bx_stream,
        by_stream,
        density=2.0,
        color="black",
        linewidth=0.7,
        arrowsize=0.8,
        arrowstyle="-|>",
    )

    # target = Rectangle(
    #     (-target_diameter / 2, 0),
    #     target_diameter,
    #     target_thickness,
    #     fill=False,
    #     edgecolor="black",
    #     linewidth=1.5,
    #     zorder=10,
    # )
    # axis.add_patch(target)

    colorbar = figure.colorbar(contour, ax=axis)
    colorbar.set_label("|B| [T]")

    axis.set_aspect("equal")
    axis.set_xlabel("x [mm]")
    axis.set_ylabel("y [mm]")
    axis.set_title("Magnetic field magnitude and field lines")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)

    return output_path


def plot_electric_field(
    result: ElectrostaticResult,
    output_path: str | Path,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    topology, _, _ = dolfinx_plot.vtk_mesh(
        result.domain,
        result.domain.topology.dim,
    )
    triangles = topology.reshape(-1, 4)[:, 1:]
    points = result.domain.geometry.x[:, :2]

    triangulation = mtri.Triangulation(
        points[:, 0],
        points[:, 1],
        triangles,
    )

    potential = result.potential.x.array
    e_magnitude = result.e_magnitude.x.array

    figure, axes = plt.subplots(1, 2, figsize=(15, 6))

    potential_plot = axes[0].tricontourf(
        triangulation,
        potential,
        levels=40,
        cmap="coolwarm",
    )
    figure.colorbar(
        potential_plot,
        ax=axes[0],
        label="Potential [V]",
    )
    axes[0].set_title("Electric potential")
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")
    axes[0].set_aspect("equal")

    field_plot = axes[1].tricontourf(
        triangulation,
        e_magnitude,
        levels=40,
        cmap="inferno",
    )
    figure.colorbar(
        field_plot,
        ax=axes[1],
        label="|E| [V/m]",
    )
    axes[1].set_title("Electric field magnitude")
    axes[1].set_xlabel("x [mm]")
    axes[1].set_ylabel("y [mm]")
    axes[1].set_aspect("equal")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)

    return output_path
