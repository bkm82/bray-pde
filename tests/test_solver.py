import pytest

import numpy as np
from solver.solver import solver_1d

from solver.mesher import create_1Dmesh


# Create a mesh for some integration testing with the meshing
@pytest.fixture
def four_cell_mesh():
    mesh = create_1Dmesh(x=[0, 1], n_cells=4)
    mesh.set_cell_temperature(0)
    mesh.set_dirichlet_boundary("left", 50)
    mesh.set_neumann_boundary("right")
    mesh.set_thermal_diffusivity(0.0001)
    return mesh


@pytest.fixture
def explicit_solver(four_cell_mesh):
    return solver_1d(
        mesh=four_cell_mesh,
        initial_time=0,
        n_time_steps=10,
        time_step_size=1,
        method="explicit",
    )


def test_initiate_solver(explicit_solver):
    assert explicit_solver.initial_time == 0
    assert explicit_solver.n_time_steps == 10
    assert explicit_solver.time_step_size == 1
    assert explicit_solver.method == "explicit"
    assert explicit_solver.mesh.n_cells == 4
    assert explicit_solver.mesh.thermal_diffusivity == 0.0001
