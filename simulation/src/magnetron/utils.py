from __future__ import annotations

import numpy as np
from dolfinx import geometry
class FieldInterpolator2D:
    def __init__(self, domain, fields):
        self.domain = domain
        self.fields = fields
        self.tree = geometry.bb_tree(domain, domain.topology.dim)

    def evaluate(self, x: float, y: float):
        point = np.array([[x, y, 0.0]], dtype=np.float64)

        candidates = geometry.compute_collisions_points(self.tree, point)

        colliding_cells = geometry.compute_colliding_cells(self.domain, candidates, point)

        cells = colliding_cells.links(0)

        if len(cells) == 0:
            raise ValueError(f"Point ({x:.6g}, {y:.6g}) lies outside the FEM mesh")

        cell = np.array([cells[0]], dtype=np.int32)
        values = {}

        for name, function in self.fields.items():
            evaluated = function.eval(point, cell)
            values[name] = float(np.asarray(evaluated).reshape(-1)[0])

        return values
    

