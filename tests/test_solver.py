import numpy as np
import pytest
from solver.solver import solver_1d
from solver.mesher import create_1Dmesh


# TODO move this integration test to its own section
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
def integration_test_explicit_solver(four_cell_mesh):
    return solver_1d(
        mesh=four_cell_mesh,
        initial_time=0,
        n_time_steps=10,
        time_step_size=1,
        method="explicit",
    )


def test_initiate_solver(integration_test_explicit_solver):
    assert integration_test_explicit_solver.initial_time == 0
    assert integration_test_explicit_solver.n_time_steps == 10
    assert integration_test_explicit_solver.time_step_size == 1
    assert integration_test_explicit_solver.method == "explicit"
    assert integration_test_explicit_solver.mesh.n_cells == 4
    assert integration_test_explicit_solver.mesh.thermal_diffusivity == 0.0001


## End integration test


@pytest.fixture
def mock_mesh(mocker):
    """
    Create to create a mock mesh object for use in testing the solver.
    Mesh configuration
    N_elements = 4
    Left boundary = Dirchilet at 50c
    Right boundary = neuiman with q=0
    :return: mesh
    """
    mesh = mocker.MagicMock()
    mesh.xcell_center = np.array([0.125, 0.375, 0.625, 0.875])
    mesh.delta_x = 0.25
    mesh.temperature = np.array([0, 0, 0, 0])
    mesh.n_cells = 4
    mesh.thermal_diffusivity = 0.0001
    mesh.differentiation_matrix = np.array(
        [[-3, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 1, -1]]
    )
    mesh.boundary_condition_array = np.array(np.array([100, 0, 0, 0]))

    return mesh


@pytest.fixture
def explicit_solver(mock_mesh):
    return solver_1d(
        mesh=mock_mesh,
        initial_time=0,
        n_time_steps=10,
        time_step_size=1,
        method="explicit",
    )


@pytest.mark.parametrize(
    "solver_fixture", ["explicit_solver", "integration_test_explicit_solver"]
)
def test_solver_initiation(solver_fixture, request):
    solver_instance = request.getfixturevalue(solver_fixture)
    assert solver_instance.initial_time == 0
    assert solver_instance.n_time_steps == 10
    assert solver_instance.time_step_size == 1
    assert solver_instance.method == "explicit"
    assert solver_instance.mesh.n_cells == 4
    assert solver_instance.mesh.thermal_diffusivity == 0.0001


def test_solver_take_step(explicit_solver):
    solver_instance = explicit_solver
    solver_instance.take_step(delta_t=1)
    expected_temperature = np.array([0.16, 0, 0, 0])
    np.testing.assert_almost_equal(
        solver_instance.mesh.temperature, expected_temperature, decimal=5
    )


if __name__ == "__main__":
    pytest.main()
