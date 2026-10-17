from pathlib import Path

import gmsh

from .geometry import MagnetronGeometry
from .parameters import MeshSettings


class MagnetronMesh:
    PHYSICAL_GROUPS = {
        "air": (1, "Air"),
        "target": (2, "Target"),
        "inner_magnet": (10, "InnerMagnet"),
        "outer_magnet_left": (11, "OuterMagnetLeft"),
        "outer_magnet_right": (12, "OuterMagnetRight"),
        "backing_plate": (20, "BackingPlate"),
    }

    def __init__(
        self,
        geometry: MagnetronGeometry,
        settings: MeshSettings | None = None,
    ):
        self.geometry = geometry
        self.settings = settings or MeshSettings()

    def generate(self, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if gmsh.isInitialized():
            raise RuntimeError(
                "Gmsh is already initialized. Finalize it before "
                "generating a new mesh."
            )

        gmsh.initialize()

        try:
            gmsh.option.setNumber(
                "Mesh.CharacteristicLengthMin",
                self.settings.min_size,
            )
            gmsh.option.setNumber(
                "Mesh.CharacteristicLengthMax",
                self.settings.max_size,
            )

            gmsh.model.add("magnetron")

            groups = self.geometry.create()

            self._add_physical_groups(groups)

            gmsh.model.mesh.generate(2)
            gmsh.write(str(output_path))

        finally:
            gmsh.finalize()

        return output_path

    def _add_physical_groups(self, groups) -> None:
        for key, (tag, name) in self.PHYSICAL_GROUPS.items():
            surface_tags = [
                tag
                for dimension, tag in groups[key]
                if dimension == 2
            ]

            if not surface_tags:
                raise ValueError(
                    f"No surfaces found for physical group '{name}'."
                )

            gmsh.model.addPhysicalGroup(2, surface_tags, tag)
            gmsh.model.setPhysicalName(2, tag, name)

def mesh_view(mesh_path):
    mesh_path = Path(mesh_path)

    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    if gmsh.isInitialized():
        raise RuntimeError(
            "Gmsh is already initialized. Finalize it before viewing a mesh."
        )

    gmsh.initialize()

    try:
        gmsh.open(str(mesh_path))
        gmsh.fltk.run()
    finally:
        gmsh.finalize()