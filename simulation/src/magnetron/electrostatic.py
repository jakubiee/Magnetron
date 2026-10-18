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
from dolfinx.io import gmsh as dolfinx_gmsh
from dolfinx.mesh import locate_entities_boundary


@dataclass
class ElectrostaticResult:
    directory: Path
    fields_file: Path
    metadata_file: Path
    domain: Any
    potential: Any
    ex: Any
    ey: Any
    e_magnitude: Any

class ElectrostaticSimulation:
    TARGET_GROUP = "Target"

    def __init__(self, mesh_path, parameters, *, comm=MPI.COMM_WORLD):
        self.mesh_path = Path(mesh_path)
        self.parameters = parameters
        self.comm = comm

    def solve(self, results_dir):
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        if not self.mesh_path.exists():
            raise FileNotFoundError(f"Mesh not found: {self.mesh_path}")

        mesh_data = dolfinx_gmsh.read_from_msh(str(self.mesh_path), self.comm, gdim=2)
        domain = mesh_data.mesh
        cell_tags = mesh_data.cell_tags

        target_tag = self._find_physical_group_tag(mesh_data.physical_groups, self.TARGET_GROUP)

        potential = self._solve_potential(domain, cell_tags, target_tag)
        ex, ey, e_magnitude = self._calculate_electric_field(domain, potential)

        fields_file = results_dir / "electrostatic_fields.xdmf"

        with io.XDMFFile(domain.comm, str(fields_file), "w") as xdmf:
            xdmf.write_mesh(domain)
            xdmf.write_function(potential)
            xdmf.write_function(ex)
            xdmf.write_function(ey)
            xdmf.write_function(e_magnitude)

        metadata_file = results_dir / "electrostatic_metadata.json"
        if domain.comm.rank == 0:
            metadata = {
                "mesh": str(self.mesh_path),
                "parameters": asdict(self.parameters),
                "fields": {
                    "potential": "V [V]",
                    "ex": "E_x [V/m]",
                    "ey": "E_y [V/m]",
                    "e_magnitude": "|E| [V/m]",
                },
                "boundary_conditions": {
                    "target": self.parameters.target_voltage,
                    "outer_boundary": self.parameters.ground_voltage,
                },
                "model": "Charge-free electrostatics with constant permittivity",
            }
            metadata_file.write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )

        domain.comm.barrier()

        return ElectrostaticResult(
            directory=results_dir,
            fields_file=fields_file,
            metadata_file=metadata_file,
            domain=domain,
            potential=potential,
            ex=ex,
            ey=ey,
            e_magnitude=e_magnitude,
        )

    @staticmethod
    def _find_physical_group_tag(physical_groups, group_name):
        if group_name not in physical_groups:
            available = ", ".join(sorted(physical_groups.keys()))
            raise ValueError(
                f"Physical group '{group_name}' not found. "
                f"Available groups: {available}"
            )

        group = physical_groups[group_name]
        return group.tag if hasattr(group, "tag") else int(group)

    def _solve_potential(self, domain, cell_tags, target_tag):
        tdim = domain.topology.dim
        fdim = tdim-1

        V = fem.functionspace(domain, ("Lagrange", 1))

        target_cells = cell_tags.find(target_tag)
        if len(target_cells) == 0:
            raise ValueError("The Target physical group contains no mesh cells.")

        target_dofs = fem.locate_dofs_topological(V, tdim, target_cells)

        boundary_facets = locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))

        boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

        target_bc = fem.dirichletbc(self.parameters.target_voltage, target_dofs, V)
        ground_bc = fem.dirichletbc(self.parameters.ground_voltage, boundary_dofs, V)

        v = ufl.TrialFunction(V)
        w = ufl.TestFunction(V)

        epsilon = (self.parameters.eps_0 * self.parameters.eps_r_air)

        a = epsilon * ufl.inner(ufl.grad(v), ufl.grad(w)) * ufl.dx
        L = fem.Constant(domain, self.parameters.ground_voltage) * w * ufl.dx

        problem = LinearProblem(a, L, bcs=[target_bc, ground_bc], petsc_options_prefix="electrostatic_", petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

        potential = problem.solve()
        potential.name = "Potential"
        potential.x.scatter_forward()

        return potential

    @staticmethod
    def _calculate_electric_field(domain, potential):
        V = fem.functionspace(domain, ("Lagrange", 1))
        dx = ufl.Measure("dx", domain=domain)

        ex = ElectrostaticSimulation._project(-potential.dx(0), V, dx, "electric_ex_", "E_x")
        ey = ElectrostaticSimulation._project(-potential.dx(1), V, dx, "electric_ey_", "E_y")

        e_magnitude = ElectrostaticSimulation._project(ufl.sqrt(potential.dx(0) ** 2 + potential.dx(1)** 2), V, dx, "electric_magnitude_", "E_magnitude")

        return ex, ey, e_magnitude

    @staticmethod
    def _project(expression, space, dx, prefix, name):
        trial = ufl.TrialFunction(space)
        test = ufl.TestFunction(space)

        problem = LinearProblem(
            ufl.inner(trial, test) * dx,
            ufl.inner(expression, test) * dx,
            petsc_options_prefix=prefix,
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
            }
        )

        function = problem.solve()
        function.name = name
        function.x.scatter_forward()
        return function
