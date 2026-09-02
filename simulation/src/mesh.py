from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import gmsh


class GeometryParameters(Protocol):
    target_diameter: float
    target_thickness: float
    magnet_height: float
    inner_magnet_diameter: float
    outer_magnet_diameter: float
    outer_magnet_ring_radius: float
    magnet_to_target_distance: float


@dataclass(frozen=True)
class MeshSettings:
    min_size: float = 0.01
    max_size: float = 1.0


class MagnetronMesh:
    AIR = 1
    TARGET = 2
    INNER_MAGNET = 10
    OUTER_MAGNET_LEFT = 11
    OUTER_MAGNET_RIGHT = 12
    BACKING_PLATE = 20

    def __init__(
        self,
        geometry: GeometryParameters,
        settings: MeshSettings | None,
    ):
        self.geometry = geometry
        self.settings = settings or MeshSettings()

    def generate(
        self,
        output_path: str | Path,
        show_gui: bool = False,
    ):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if gmsh.isInitialized():
            gmsh.finalize()

        gmsh.initialize()
        try:
            gmsh.option.setNumber(
                "Mesh.CharacteristicLengthMin", self.settings.min_size
            )
            gmsh.option.setNumber(
                "Mesh.CharacteristicLengthMax", self.settings.max_size
            )
            gmsh.model.add("magnetron")

            geo = gmsh.model.occ
            entities = self._create_geometry(geo)
            groups = self._fragment_and_group(geo, entities)
            self._add_physical_groups(groups)

            gmsh.model.mesh.generate(2)
            gmsh.write(str(output_path))

            if show_gui:
                gmsh.fltk.run()
        finally:
            gmsh.finalize()

        return output_path

    def _create_geometry(self, geo: object) -> dict[str, int]:
        g = self.geometry
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
                -g.magnet_to_target_distance - g.magnet_height,
                0,
                g.inner_magnet_diameter,
                g.magnet_height,
            ),
            "outer_magnet_left": geo.addRectangle(
                -g.outer_magnet_diameter / 2 - g.outer_magnet_ring_radius,
                -g.magnet_to_target_distance - g.magnet_height,
                0,
                g.outer_magnet_diameter,
                g.magnet_height,
            ),
            "outer_magnet_right": geo.addRectangle(
                g.outer_magnet_ring_radius - g.outer_magnet_diameter / 2,
                -g.magnet_to_target_distance - g.magnet_height,
                0,
                g.outer_magnet_diameter,
                g.magnet_height,
            ),
        }

    def _fragment_and_group(
        self, geo: object, entities: dict[str, int]
    ):
        names = list(entities)
        objects = [(2, entities[name]) for name in names]
        _, fragments = geo.fragment(objects, [])
        geo.synchronize()

        groups = dict(zip(names, fragments, strict=True))
        solid_names = [name for name in names if name != "air"]
        solid_tags = {tag for name in solid_names for _, tag in groups[name]}
        groups["air"] = [
            entity for entity in geo.getEntities(2) if entity[1] not in solid_tags
        ]
        return groups

    def _add_physical_groups(self, groups):
        definitions = (
            ("air", self.AIR, "Air"),
            ("target", self.TARGET, "Target"),
            ("inner_magnet", self.INNER_MAGNET, "InnerMagnet"),
            (
                "outer_magnet_left",
                self.OUTER_MAGNET_LEFT,
                "OuterMagnetLeft",
            ),
            (
                "outer_magnet_right",
                self.OUTER_MAGNET_RIGHT,
                "OuterMagnetRight",
            ),
            ("backing_plate", self.BACKING_PLATE, "BackingPlate"),
        )

        for key, tag, name in definitions:
            gmsh.model.addPhysicalGroup(
                2, [entity_tag for _, entity_tag in groups[key]], tag
            )
            gmsh.model.setPhysicalName(2, tag, name)
