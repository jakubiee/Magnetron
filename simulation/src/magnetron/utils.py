from __future__ import annotations

import numpy as np
import matplotlib.tri as mtri


class FieldInterpolator2D:
    def __init__(self, domain, fields):
        self.domain = domain
        self._last_point = None
        self._last_components = None
        self.names = list(fields)
        dof_coordinates = np.asarray(
            next(iter(fields.values())).function_space.tabulate_dof_coordinates()[
                :, :2
            ],
            dtype=float,
        )
        self.points = np.unique(
            np.round(dof_coordinates, decimals=12),
            axis=0,
        )
        point_indices = {
            (float(point[0]), float(point[1])): index
            for index, point in enumerate(self.points)
        }

        self.values = np.empty((len(self.points), len(self.names)), dtype=float)
        for column, (name, function) in enumerate(fields.items()):
            coordinates = np.asarray(
                np.round(
                    function.function_space.tabulate_dof_coordinates()[:, :2],
                    decimals=12,
                ),
                dtype=float,
            )
            values = np.asarray(function.x.array.real, dtype=float)
            if len(coordinates) != len(values):
                raise ValueError(
                    f"Mismatch between DOF coordinates ({len(coordinates)}) "
                    f"and field values ({len(values)})."
                )

            try:
                indices = np.fromiter(
                    (
                        point_indices[(float(point[0]), float(point[1]))]
                        for point in coordinates
                    ),
                    dtype=np.int64,
                    count=len(coordinates),
                )
            except KeyError as error:
                raise ValueError(
                    f"Field '{name}' uses a different mesh from the interpolator."
                ) from error

            sums = np.zeros(len(self.points), dtype=float)
            counts = np.zeros(len(self.points), dtype=np.int32)
            np.add.at(sums, indices, values)
            np.add.at(counts, indices, 1)
            present = counts > 0
            if not np.all(present):
                raise ValueError(
                    f"Field '{name}' uses a different mesh from the interpolator."
                )
            self.values[present, column] = sums[present] / counts[present]

        tdim = domain.topology.dim
        domain.topology.create_connectivity(tdim, 0)
        connectivity = domain.topology.connectivity(tdim, 0)
        num_cells = (
            domain.topology.index_map(tdim).size_local
            + domain.topology.index_map(tdim).num_ghosts
        )
        geometry_points = domain.geometry.x[:, :2]
        triangles = np.empty((num_cells, 3), dtype=np.int32)

        for cell in range(num_cells):
            vertices = connectivity.links(cell)
            if len(vertices) != 3:
                raise ValueError("Expected triangular cells in the FEM mesh.")
            triangles[cell] = [
                point_indices[
                    (
                        float(np.round(point[0], decimals=12)),
                        float(np.round(point[1], decimals=12)),
                    )
                ]
                for point in geometry_points[vertices]
            ]

        triangulation = mtri.Triangulation(
            self.points[:, 0],
            self.points[:, 1],
            triangles,
        )
        self.trifinder = triangulation.get_trifinder()
        self.triangles = triangles

    def evaluate_components(self, x: float, y: float):
        if self._last_point == (x, y):
            return self._last_components

        triangle_index = int(self.trifinder(float(x), float(y)))
        if triangle_index < 0:
            raise ValueError(f"Point ({x:.6g}, {y:.6g}) lies outside the FEM mesh")

        vertex_indices = self.triangles[triangle_index]
        triangle = self.points[vertex_indices]
        x0, y0 = triangle[0]
        x1, y1 = triangle[1]
        x2, y2 = triangle[2]
        denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        weight0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denominator
        weight1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denominator
        weights = np.array([weight0, weight1, 1.0 - weight0 - weight1])
        interpolated = weights @ self.values[vertex_indices]

        self._last_point = (x, y)
        self._last_components = interpolated
        return interpolated

    def evaluate(self, x: float, y: float):
        return dict(zip(self.names, self.evaluate_components(x, y), strict=True))


def make_air_domain_checker(domain, cell_tags, air_tag):
    tdim = domain.topology.dim
    if tdim != 2:
        raise ValueError("Air domain checker requires a 2D mesh.")

    domain.topology.create_connectivity(tdim, 0)

    points = domain.geometry.x[:, :2]
    connectivity = domain.topology.connectivity(tdim, 0)
    num_cells = domain.topology.index_map(tdim).size_local

    air_cells = cell_tags.find(air_tag)
    air_cells = air_cells[air_cells < num_cells]

    triangles = []
    for cell in air_cells:
        vertices = connectivity.links(int(cell))
        if len(vertices) != 3:
            raise ValueError("Expected triangular cells. Check the mesh cell type.")
        triangles.append(vertices)

    if not triangles:
        raise ValueError("No local Air cells found. Check the Air physical group tag.")

    triangulation = mtri.Triangulation(
        points[:, 0],
        points[:, 1],
        np.asarray(triangles, dtype=np.int32),
    )
    finder = triangulation.get_trifinder()

    def inside_air(x: float, y: float) -> bool:
        return finder(float(x), float(y)) >= 0

    return inside_air
