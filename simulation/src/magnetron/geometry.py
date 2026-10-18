from __future__ import annotations

import gmsh

from typing import Protocol
from .parameters import GeometryParameters


class MagnetronGeometry:
    def __init__(self, parameters: GeometryParameters):
        self.parameters = parameters

    def create(self):
        geo = gmsh.model.occ
        entities = self._create_geometry(geo)

        names = list(entities)
        objects = [(2, entities[name]) for name in names]

        _, fragments = geo.fragment(objects, [])
        geo.synchronize()

        groups: dict[str, list[tuple[int, int]]] = {
            name: list(fragment_map)
            for name, fragment_map in zip(names, fragments, strict=True)
        }

        solid_names = [name for name in names if name != "air"]
        solid_tags = {
            tag for name in solid_names for dim, tag in groups[name] if dim == 2
        }

        groups["air"] = [
            (dim, tag) for dim, tag in geo.getEntities(2) if tag not in solid_tags
        ]

        return groups

    def _create_geometry(self, geo: gmsh.model.occ):
        g = self.parameters

        magnet_bottom_y = -g.magnet_to_target_distance - g.magnet_height

        return {
            "air": geo.addRectangle(-40, -30, 0, 80, 60),
            "target": geo.addRectangle(
                -g.target_diameter / 2,
                0,
                0,
                g.target_diameter,
                g.target_thickness,
            ),
            "backing_plate": geo.addRectangle(
                -g.target_diameter / 2,
                -g.magnet_to_target_distance,
                0,
                g.target_diameter,
                g.magnet_to_target_distance,
            ),
            "inner_magnet": geo.addRectangle(
                -g.inner_magnet_diameter / 2,
                magnet_bottom_y,
                0,
                g.inner_magnet_diameter,
                g.magnet_height,
            ),
            "outer_magnet_left": geo.addRectangle(
                -g.outer_magnet_diameter / 2 - g.outer_magnet_ring_radius,
                magnet_bottom_y,
                0,
                g.outer_magnet_diameter,
                g.magnet_height,
            ),
            "outer_magnet_right": geo.addRectangle(
                g.outer_magnet_ring_radius - g.outer_magnet_diameter / 2,
                magnet_bottom_y,
                0,
                g.outer_magnet_diameter,
                g.magnet_height,
            ),
            "anode_left": self._add_polygon(
                geo,
                [
                    (
                        -g.target_diameter / 2 - g.anode_cathode_gap,
                        magnet_bottom_y - g.anode_cathode_gap,
                    ),
                    (
                        -g.target_diameter / 2 - g.anode_cathode_gap - g.anode_width,
                        magnet_bottom_y - g.anode_cathode_gap,
                    ),
                    (
                        -g.target_diameter / 2 - g.anode_cathode_gap - g.anode_width,
                        g.target_thickness + g.anode_width + g.anode_cathode_gap,
                    ),
                    (
                        -g.target_diameter / 2 + g.anode_width,
                        g.target_thickness + g.anode_width + g.anode_cathode_gap,
                    ),
                    (
                        -g.target_diameter / 2 + g.anode_width,
                        g.target_thickness + g.anode_cathode_gap,
                    ),
                    (
                        -g.target_diameter / 2 - g.anode_cathode_gap,
                        g.target_thickness + g.anode_cathode_gap,
                    ),
                ],
            ),
            "anode_right": self._add_polygon(
                geo,
                [
                    (
                        g.target_diameter / 2 + g.anode_cathode_gap,
                        magnet_bottom_y - g.anode_cathode_gap,
                    ),
                    (
                        g.target_diameter / 2 + g.anode_cathode_gap + g.anode_width,
                        magnet_bottom_y - g.anode_cathode_gap,
                    ),
                    (
                        g.target_diameter / 2 + g.anode_cathode_gap + g.anode_width,
                        g.target_thickness + g.anode_width + g.anode_cathode_gap,
                    ),
                    (
                        g.target_diameter / 2 - g.anode_width,
                        g.target_thickness + g.anode_width + g.anode_cathode_gap,
                    ),
                    (
                        g.target_diameter / 2 - g.anode_width,
                        g.target_thickness + g.anode_cathode_gap,
                    ),
                    (
                        g.target_diameter / 2 + g.anode_cathode_gap,
                        g.target_thickness + g.anode_cathode_gap,
                    ),
                ],
            ),
        }

    def show(self):
        gmsh.initialize()
        try:
            gmsh.model.add("magnetron")
            self.create()
            gmsh.fltk.run()
        finally:
            gmsh.finalize()

    def _add_polygon(self, geo, points):
        point_tags = [geo.addPoint(x, y, 0) for x, y in points]

        line_tags = [
            geo.addLine(
                point_tags[i],
                point_tags[(i + 1) % len(point_tags)],
            )
            for i in range(len(point_tags))
        ]

        loop = geo.addCurveLoop(line_tags)
        return geo.addPlaneSurface([loop])
