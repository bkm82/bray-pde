import numpy as np
import pandas as pd
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
        time_step_size=1,
        method="explicit",
    )


def test_initiate_solver(integration_test_explicit_solver):
    assert integration_test_explicit_solver.initial_time == 0
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
        time_step_size=1,
        method="explicit",
    )


# # Create test cases to test the implicit solver
@pytest.fixture
def implicit_solver(mock_mesh):
    return solver_1d(
        mesh=mock_mesh,
        initial_time=0,
        time_step_size=1,
        method="implicit",
    )


@pytest.mark.parametrize(
    "solver_fixture, expected_method",
    [
        ("explicit_solver", "explicit"),
        ("integration_test_explicit_solver", "explicit"),
        ("implicit_solver", "implicit"),
    ],
)
def test_solver_initiation(solver_fixture, expected_method, request):
    solver_instance = request.getfixturevalue(solver_fixture)
    assert solver_instance.initial_time == 0
    assert solver_instance.time_step_size == 1
    assert solver_instance.method == expected_method
    assert solver_instance.mesh.n_cells == 4
    assert solver_instance.mesh.thermal_diffusivity == 0.0001


# Expected temperature after steping forward 1 and 3 steps
step_forward_expected_results = [
    ("explicit_solver", np.array([0.16, 0, 0, 0]), np.array([0.4777, 0.000766, 0, 0])),
    (
        "integration_test_explicit_solver",
        np.array([0.16, 0, 0, 0]),
        np.array([0.4777, 0.000766, 0, 0]),
    ),
    (
        "implicit_solver",
        np.array([1.59236073e-01, 2.53965675e-04, 4.05049955e-07, 6.47044657e-10]),
        np.array([4.75432619e-01, 1.51572460e-03, 4.02797229e-06, 9.65628626e-09]),
    ),
]


@pytest.mark.parametrize(
    "solver_fixture, expected, three_step_expected", step_forward_expected_results
)
def test_solver_take_step_new(solver_fixture, expected, three_step_expected, request):
    solver_instance = request.getfixturevalue(solver_fixture)
    solver_instance.take_step()
    expected_temperature = expected
    np.testing.assert_almost_equal(
        solver_instance.mesh.temperature, expected_temperature, decimal=5
    )


# Test that the solver can be called
@pytest.mark.parametrize(
    "solver_fixture, expected, three_step_expected", step_forward_expected_results
)
def test_solver_solve(solver_fixture, expected, three_step_expected, request):
    solver_instance = request.getfixturevalue(solver_fixture)
    solver_instance.solve(t_initial=0, t_final=1)
    expected_temperature = expected
    np.testing.assert_almost_equal(
        solver_instance.mesh.temperature, expected_temperature, decimal=5
    )


@pytest.mark.parametrize(
    "solver_fixture, expected, three_step_expected", step_forward_expected_results
)
def test_solver_solve_multiple_timesteps(
    solver_fixture, expected, three_step_expected, request
):
    solver_instance = request.getfixturevalue(solver_fixture)
    solver_instance.solve(t_initial=0, t_final=3)
    expected_temperature = three_step_expected
    np.testing.assert_almost_equal(
        solver_instance.mesh.temperature, expected_temperature, decimal=5
    )


def test_solver_save_creates_saved_state(explicit_solver):
    solver_instance = explicit_solver
    solver_instance.save_state()
    assert hasattr(solver_instance, "saved_state_list")


@pytest.mark.xfail(reason="pandas dataframe unable to determine index")
def test_solver_save_state_accepts_atribute_names(explicit_solver):
    solver_instance = explicit_solver
    solver_instance.save_state("time_step_size")
    expected_list = pd.concat([pd.DataFrame({"time_step_size": 1}, index=[0])])
    pd.testing.assert_frame_equal(
        pd.concat(solver_instance.saved_state_list), expected_list
    )


def test_solver_save_state_accepts_keywords(explicit_solver):
    solver_instance = explicit_solver
    solver_instance.save_state(
        "method", x_position=np.array([0.125, 0.375, 0.625, 0.875])
    )
    expected_list = pd.concat(
        [
            pd.DataFrame(
                {
                    "method": "explicit",
                    "x_position": np.array([0.125, 0.375, 0.625, 0.875]),
                }
            )
        ]
    )
    pd.testing.assert_frame_equal(
        pd.concat(solver_instance.saved_state_list), expected_list
    )


def test_integration_solve_save_state(explicit_solver):
    solver_instance = explicit_solver
    solver_instance.solve(t_initial=0, t_final=1)
    time_zero_dict = dict(
        method="explicit",
        time=0,
        x_cordinates=np.array([0.125, 0.375, 0.625, 0.875]),
        temperature=np.array([0, 0, 0, 0]),
    )
    time_one_dict = dict(
        method="explicit",
        time=1,
        x_cordinates=np.array([0.125, 0.375, 0.625, 0.875]),
        temperature=np.array([0.16, 0, 0, 0]),
    )
    expected_list = []
    expected_list.append(pd.DataFrame(time_zero_dict))
    expected_list.append(pd.DataFrame(time_one_dict))
    expected_data_frame = pd.concat(expected_list)

    pd.testing.assert_frame_equal(
        pd.concat(solver_instance.saved_state_list), expected_data_frame
    )

    pd.testing.assert_frame_equal(solver_instance.saved_data, expected_data_frame)


if __name__ == "__main__":
    pytest.main()
