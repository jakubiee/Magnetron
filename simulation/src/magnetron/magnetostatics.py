from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, io
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmsh
from dolfinx.mesh import locate_entities_boundary


@dataclass(frozen=True)
class MagnetostaticSettings:
    remanence: float = 1.32
    relative_permeability_air: float = 1.0
    relative_permeability_target: float = 1.0
    relative_permeability_magnet: float = 1.05
    relative_permeability_backing_plate: float = 1.0


@dataclass(frozen=True)
class SimulationResult:
    directory: Path
    fields_file: Path
    metadata_file: Path
    domain: Any
    potential: Any
    bx: Any
    by: Any
    b_magnitude: Any


class MagnetostaticSimulation:
    _MU0 = 4e-7 * np.pi
    _GROUPS = (
        "Air",
        "Target",
        "InnerMagnet",
        "OuterMagnetLeft",
        "OuterMagnetRight",
        "BackingPlate",
    )

    def __init__(
        self,
        mesh_path,
        settings = None,
        *,
        comm: MPI.Intracomm = MPI.COMM_WORLD,
    ) -> None:
        self.mesh_path = Path(mesh_path)
        self.settings = settings or MagnetostaticSettings()
        self.comm = comm

    def solve(self, results_dir):
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        mesh_data = gmsh.read_from_msh(
            str(self.mesh_path), self.comm, gdim=2
        )
        domain = mesh_data.mesh
        cell_tags = mesh_data.cell_tags
        group_tags = self._physical_group_tags(mesh_data.physical_groups)
        potential, bx, by, b_magnitude = self._solve_fields(
            domain, cell_tags, group_tags
        )

        fields_file = results_dir / "fields.xdmf"
        with io.XDMFFile(domain.comm, str(fields_file), "w") as xdmf:
            xdmf.write_mesh(domain)
            xdmf.write_function(potential)
            xdmf.write_function(bx)
            xdmf.write_function(by)
            xdmf.write_function(b_magnitude)

        metadata_file = results_dir / "simulation.json"
        if domain.comm.rank == 0:
            metadata_file.write_text(
                json.dumps(
                    {
                        "mesh": str(self.mesh_path),
                        "settings": asdict(self.settings),
                        "fields": {
                            "A_z": "magnetic vector potential [T m]",
                            "B_x": "magnetic flux density x-component [T]",
                            "B_y": "magnetic flux density y-component [T]",
                            "B_magnitude": "magnetic flux density magnitude [T]",
                        },
                    },
                    indent=2,
                )
                + "\n"
            )
        domain.comm.barrier()

        return SimulationResult(
            directory=results_dir,
            fields_file=fields_file,
            metadata_file=metadata_file,
            domain=domain,
            potential=potential,
            bx=bx,
            by=by,
            b_magnitude=b_magnitude,
        )

    def _physical_group_tags(self, physical_groups: dict) -> dict[str, int]:
        missing = [name for name in self._GROUPS if name not in physical_groups]
        if missing:
            names = ", ".join(missing)
            raise ValueError(f"Mesh is missing physical groups: {names}")
        return {name: physical_groups[name].tag for name in self._GROUPS}

    def _solve_fields(self, domain, cell_tags, tags: dict[str, int]):
        potential_space = fem.functionspace(domain, ("Lagrange", 1))
        boundary_condition = self._zero_boundary_condition(
            domain, potential_space
        )
        permeability = self._permeability(domain, cell_tags, tags)

        trial = ufl.TrialFunction(potential_space)
        test = ufl.TestFunction(potential_space)
        dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)

        magnetization = self.settings.remanence / self._MU0
        a = ufl.inner(ufl.grad(trial), ufl.grad(test)) / permeability * ufl.dx
        source = (
            -magnetization * test.dx(0) * dx(tags["InnerMagnet"])
            + magnetization * test.dx(0) * dx(tags["OuterMagnetLeft"])
            + magnetization * test.dx(0) * dx(tags["OuterMagnetRight"])
        )
        potential = LinearProblem(
            a,
            source,
            bcs=[boundary_condition],
            petsc_options_prefix="magnetostatic_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        ).solve()
        potential.name = "A_z"

        result_space = fem.functionspace(domain, ("Lagrange", 1))
        bx = self._project(potential.dx(1), result_space)
        bx.name = "B_x"
        by = self._project(-potential.dx(0), result_space)
        by.name = "B_y"
        b_magnitude = self._project(
            ufl.sqrt(potential.dx(1) ** 2 + potential.dx(0) ** 2),
            result_space,
        )
        b_magnitude.name = "B_magnitude"
        return potential, bx, by, b_magnitude

    def _zero_boundary_condition(self, domain, function_space):
        facet_dimension = domain.topology.dim - 1
        boundary_facets = locate_entities_boundary(
            domain,
            facet_dimension,
            lambda coordinates: np.full(coordinates.shape[1], True),
        )
        boundary_dofs = fem.locate_dofs_topological(
            function_space, facet_dimension, boundary_facets
        )
        zero = fem.Function(function_space)
        zero.x.array[:] = 0.0
        return fem.dirichletbc(zero, boundary_dofs)

    def _permeability(self, domain, cell_tags, tags: dict[str, int]):
        space = fem.functionspace(domain, ("DG", 0))
        permeability = fem.Function(space)
        permeability.x.array[:] = self._MU0 * self.settings.relative_permeability_air

        materials = {
            "Target": self.settings.relative_permeability_target,
            "InnerMagnet": self.settings.relative_permeability_magnet,
            "OuterMagnetLeft": self.settings.relative_permeability_magnet,
            "OuterMagnetRight": self.settings.relative_permeability_magnet,
            "BackingPlate": self.settings.relative_permeability_backing_plate,
        }
        for group, relative_permeability in materials.items():
            cells = cell_tags.find(tags[group])
            permeability.x.array[cells] = self._MU0 * relative_permeability
        return permeability

    @staticmethod
    def _project(expression, function_space):
        trial = ufl.TrialFunction(function_space)
        test = ufl.TestFunction(function_space)
        return LinearProblem(
            trial * test * ufl.dx,
            expression * test * ufl.dx,
            petsc_options_prefix="projection_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        ).solve()
