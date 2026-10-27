from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import ufl
import numpy as np
from mpi4py import MPI

from dolfinx import fem, io
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmsh
from dolfinx.mesh import locate_entities_boundary


from .parameters import MagneticParameters


@dataclass
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
    GROUPS = (
        "Air",
        "Target",
        "InnerMagnet",
        "OuterMagnetLeft",
        "OuterMagnetRight",
        "BackingPlate",
        "AnodeLeft",
        "AnodeRight",
    )

    def __init__(
        self,
        mesh_path,
        parameters: MagneticParameters,
        *,
        comm: MPI.Intracomm = MPI.COMM_WORLD,
    ):
        self.mesh_path = mesh_path
        self.parameters = parameters
        self.comm = comm

    def solve(self, results_dir):
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        if not self.mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {self.mesh_path}")

        mesh_data = gmsh.read_from_msh(
            str(self.mesh_path),
            self.comm,
            gdim=2,
        )

        domain = mesh_data.mesh
        cell_tags = mesh_data.cell_tags

        coordinates = domain.geometry.x

        if domain.comm.rank == 0:
            print("Mesh coordinate bounds [m]:")
            print("  x:", coordinates[:, 0].min(), coordinates[:, 0].max())
            print("  y:", coordinates[:, 1].min(), coordinates[:, 1].max())

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
            metadata = {
                "mesh": str(self.mesh_path),
                "parameters": asdict(self.parameters),
                "fields": {
                    "A_z": "Magnetic vector potential [T m]",
                    "B_x": "Magnetic flux density x-component [T]",
                    "B_y": "Magnetic flux density y-component [T]",
                    "B_magnitude": "Magnetic flux density magnitude [T]",
                },
            }

            metadata_file.write_text(json.dumps(metadata, indent=2) + "\n")

        domain.comm.barrier()

        if domain.comm.rank == 0:
            print("Magnetostatic problem solved.")
            print(f"Fields: {fields_file}")
            print(f"Metadata: {metadata_file}")
            print(
                "|B| min/max [T]:",
                np.min(b_magnitude.x.array),
                np.max(b_magnitude.x.array),
            )

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

    def _physical_group_tags(self, physical_groups):
        missing = [name for name in self.GROUPS if name not in physical_groups]

        if missing:
            raise ValueError(
                "Mesh is missing physical groups: " + ", ".join(missing)
            )

        return {name: physical_groups[name].tag for name in self.GROUPS}

    def _solve_fields(self, domain, cell_tags, tags):
        V = fem.functionspace(domain, ("Lagrange", 1))

        permeability = self._permeability(domain, cell_tags, tags)

        facet_dimension = domain.topology.dim - 1

        boundary_facets = locate_entities_boundary(
            domain, facet_dimension, lambda x: np.full(x.shape[1], True)
        )

        boundary_dofs = fem.locate_dofs_topological(V, facet_dimension, boundary_facets)

        zero = fem.Function(V)
        zero.name = "Az_boundary"
        zero.x.array[:] = 0.0

        bc = fem.dirichletbc(zero, boundary_dofs)

        trial = ufl.TrialFunction(V)
        test = ufl.TestFunction(V)

        dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)

        a = (ufl.inner(ufl.grad(trial), ufl.grad(test)) / permeability) * ufl.dx

        M = self.parameters.remanence / self.parameters.mu_0

        source = (
            -M * test.dx(0) * dx(tags["InnerMagnet"])
            + M * test.dx(0) * dx(tags["OuterMagnetLeft"])
            + M * test.dx(0) * dx(tags["OuterMagnetRight"])
        )

        potential = LinearProblem(
            a,
            source,
            bcs=[bc],
            petsc_options_prefix="magnetostatic_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        ).solve()

        potential.name = "A_z"

        Q = fem.functionspace(domain, ("Lagrange", 1))

        bx = self._project(potential.dx(1), Q, prefix="bx_")
        bx.name = "B_x"

        by = self._project(-potential.dx(0), Q, prefix="by_")
        by.name = "B_y"

        b_magnitude = self._project(
            ufl.sqrt(potential.dx(1) ** 2 + potential.dx(0) ** 2), Q, "project_bmag_"
        )
        b_magnitude.name = "B_magnitude"

        potential.x.scatter_forward()
        bx.x.scatter_forward()
        by.x.scatter_forward()
        b_magnitude.x.scatter_forward()

        return potential, bx, by, b_magnitude

    def _permeability(self, domain, cell_tags, tags):
        V0 = fem.functionspace(domain, ("DG", 0))
        permeability = fem.Function(V0)
        permeability.name = "Permeability"

        permeability.x.array[:] = self.parameters.mu_0 * self.parameters.mu_r_air

        material_permeabilities = {
            "Target": self.parameters.mu_r_air,
            "InnerMagnet": self.parameters.mu_r_magnet,
            "OuterMagnetLeft": self.parameters.mu_r_magnet,
            "OuterMagnetRight": self.parameters.mu_r_magnet,
            "BackingPlate": self.parameters.mu_r_backing,
            "AnodeLeft": self.parameters.mu_r_air,
            "AnodeRight": self.parameters.mu_r_air,
        }

        for name, mu_r in material_permeabilities.items():
            cells = cell_tags.find(tags[name])
            permeability.x.array[cells] = self.parameters.mu_0 * mu_r

        permeability.x.scatter_forward()
        return permeability

    @staticmethod
    def _project(expression, function_space, prefix: str):
        trial = ufl.TrialFunction(function_space)
        test = ufl.TestFunction(function_space)

        problem = LinearProblem(
            ufl.inner(trial, test) * ufl.dx,
            expression * test * ufl.dx,
            petsc_options_prefix=prefix,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )

        return problem.solve()
