import numpy as np
import ufl
import pyvista as pv
from petsc4py import PETSc
from mpi4py import MPI
from pathlib import Path
from dataclasses import dataclass
from dolfinx.io import gmsh
from dolfinx import fem
from dolfinx.mesh import locate_entities_boundary
from dolfinx.fem import locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem
from dolfinx import fem, plot



class ElectrostaticSimulation:
    _GROUPS = (
        "Air",
        "Target",
        "InnerMagnet",
        "OuterMagnetLeft",
        "OuterMagnetRight",
        "BackingPlate",
    )
        
    def __init__(self, mesh_path, settings=None, *, comm: MPI.Intracomm = MPI.COMM_WORLD):
        self.mesh_path = Path(mesh_path)
        self.settings = settings
        self.comm = comm

    def _physical_group_tags(self, physical_groups: dict) -> dict[str, int]:
        missing = [name for name in self._GROUPS if name not in physical_groups]
        if missing:
            names = ", ".join(missing)
            raise ValueError(f"Mesh is missing physical groups: {names}")
        return {name: physical_groups[name].tag for name in self._GROUPS}

    def solve(self, results_dir):
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        mesh_data = gmsh.read_from_msh(str(self.mesh_path), self.comm, gdim=2)

        domain = mesh_data.mesh
        cell_tags = mesh_data.cell_tags
        group_tags = self._physical_group_tags(mesh_data.physical_groups)


        print("physical groups:", group_tags)

        self._solve_fields(domain, cell_tags, group_tags)

    def _solve_fields(self, domain, cell_tags, group_tags):
        V = fem.functionspace(domain, ("Lagrange", 1))

        V_taget = -400.0
        V_ground = 0.0

        fdim = domain.topology.dim - 1
        domain.topology.create_connectivity(fdim, fdim - 1)

        facets = locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True))

        dofs = locate_dofs_topological(V, fdim, facets)

        ground = fem.Constant(domain, PETSc.ScalarType(V_ground))
        bc = fem.dirichletbc(ground, dofs, V)

        eps0 = 8.854187817e-12
        eps_r_air = 1.0

        eps = fem.Constant(domain, PETSc.ScalarType(eps0 * eps_r_air))

        potential = ufl.TrialFunction(V)
        test = ufl.TestFunction(V)

        a = eps * ufl.inner(ufl.grad(potential), ufl.grad(test)) * ufl.dx

        L = fem.Constant(domain, PETSc.ScalarType(0.0)) * test * ufl.dx

        problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix="electrostatic_", petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

        phi = problem.solve()
        phi.name = "Potential"

        print("Electrostatic problem solved.")
        print("Potential min:", np.min(phi.x.array))
        print("Potential max:", np.max(phi.x.array))

        Q = fem.functionspace(domain, ("Lagrange", 1))
        Ex_expr = -phi.dx(0)
        Ey_expr = -phi.dx(1)

        def project(expr, space, prefix):
            u = ufl.TrialFunction(space)
            v = ufl.TestFunction(space)

            problem = LinearProblem(u*v*ufl.dx, expr*v*ufl.dx, petsc_options_prefix=prefix, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
            return problem.solve()

        Ex = project(Ex_expr, Q, "project_Ex_")
        Ey = project(Ey_expr, Q, "project_Ey_")

        E_mag = project(ufl.sqrt(Ex**2 + Ey**2), Q, "project_E_mag_")

        print("|E| min:", np.min(E_mag.x.array), "V/m")
        print("|E| max:", np.max(E_mag.x.array), "V/m")

        topology, cell_types, geometry = plot.vtk_mesh(Q)
        grid = pv.UnstructuredGrid(topology, cell_types, geometry)

        grid.point_data["Potential"] = phi.x.array
        grid.point_data["E_mag"] = E_mag.x.array

        plotter = pv.Plotter()
        plotter.add_mesh(
            grid,
            scalars="Potential",
            cmap="coolwarm",
            show_edges=False,
            scalar_bar_args={"title": "Potential [V]"},
        )
        plotter.view_xy()
        plotter.enable_parallel_projection()
        plotter.show()