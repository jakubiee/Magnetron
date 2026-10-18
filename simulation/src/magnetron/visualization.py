
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.interpolate import griddata
from .magnetostatic import SimulationResult


def _interpolate_field(result: SimulationResult, values: np.ndarray, x_grid: np.ndarray,y_grid: np.ndarray):
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


def plot_magnetic_field(result: SimulationResult, output_path: str | Path, *, grid_shape: tuple[int, int] = (300, 250), target_diameter: float = 50.8,
target_thickness: float = 3.0,):
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
    b_magnitude = _interpolate_field(
        result, result.b_magnitude.x.array, x_grid, y_grid
    )

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

