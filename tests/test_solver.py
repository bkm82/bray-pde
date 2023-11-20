import numpy as np
import pandas as pd
import pytest
from solver.solver import solver_1d, SteadySolver, Solver
from solver.mesher import heat_diffusion_mesh
from solver import solver
from solver import cartesian_mesh

from unittest.mock import patch, MagicMock


# TODO move this integration test to its own section
# Create a mesh for some integration testing with the meshing
@pytest.fixture
def four_cell_mesh():
    mesh = heat_diffusion_mesh(x=[0, 1], n_cells=4)
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
    mesh.laplacian = np.array(
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
        method="explicit", x_position=np.array([0.125, 0.375, 0.625, 0.875])
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


# @pytest.mark.xfail(reason="data type of time_step_size int64 vs float64")
def test_integration_solve_save_state(explicit_solver):
    solver_instance = explicit_solver
    solver_instance.solve(t_initial=0, t_final=1)
    time_zero_dict = dict(
        method="explicit",
        time_step_size=1,
        time=0,
        x_cordinates=np.array([0.125, 0.375, 0.625, 0.875]),
        temperature=np.array([0, 0, 0, 0]),
    )
    time_one_dict = dict(
        method="explicit",
        time_step_size=1,
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


def test_init():
    from solver import solver

    with patch.object(solver, "main", MagicMock()) as mock_main:
        with patch.object(solver, "__name__", "__main__"):
            solver.init()
    mock_main.assert_called_once()


# from unittest import mock
# def test_init():
#     from solver import solver


#     with mock.patch.object(solver, "main", return_value = 42):
#         with mock.patch.object(solver, "__name__", "__main__"):
#             with mock.patch.object(solver.sys,'exit') as mock_exit:
#                 solver.init()
#                 assert mock_exit.call_args[0][0] == 42
@pytest.fixture
def mock_linear_convective_mesh_upwind(mocker):
    """
    Create to create a mock mesh object for use in testing the solver.
    Mesh configuration
    N_elements = 4
    Left boundary = Dirchilet 1
    :return: mesh
    """
    mesh = mocker.MagicMock()
    mesh.xcell_center = np.array([0.125, 0.375, 0.625, 0.875])
    mesh.delta_x = 0.25
    mesh.phi = np.array([1, 1, 0, 0])
    mesh.n_cells = 4
    mesh.laplacian = np.array(
        [[-1, 0, 0, 0], [1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1]]
    )
    mesh.boundary_condition_array = np.array(np.array([1, 0, 0, 0]))
    mesh.convection_coefficent = 1

    mesh.discretization_type = "upwind"
    return mesh


@pytest.fixture
def upwind_solver(mock_linear_convective_mesh_upwind):
    return solver.linear_convection_solver(
        mesh=mock_linear_convective_mesh_upwind,
        initial_time=0,
        time_step_size=0.25,
        method="explicit",
    )


def test_upwind_take_step_(upwind_solver):
    upwind_solver.take_step()
    expected_phi = [1, 1, 1, 0]

    np.testing.assert_array_equal(upwind_solver.mesh.phi, expected_phi)


def test_upwind_solver(upwind_solver):
    upwind_solver.solve(2)
    expected_phi = [1, 1, 1, 1]

    np.testing.assert_array_equal(upwind_solver.mesh.phi, expected_phi)


@pytest.fixture
def mock_linear_convective_mesh_mcormack(mocker):
    """
    Create to create a mock mesh object for use in testing the solver.
    Mesh configuration
    N_elements = 4
    Left boundary = Dirchilet 1
    :return: mesh
    """
    mesh = mocker.MagicMock()
    mesh.xcell_center = np.array([0, 0.25, 0.5, 0.75, 1])
    mesh.delta_x = 0.25
    mesh.phi = np.array([1, 1, 0, 0, 0])
    mesh.n_cells = 5
    mesh.laplacian = np.array(
        [
            [0, 0, 0, 0, 0],
            [-1, 1, 0, 0, 0],
            [0, -1, 1, 0, 0],
            [0, 0, -1, 1, 0],
            [0, 0, 0, -1, 1],
        ]
    )
    mesh.predictor_differentiation_matrix = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, -1, 1, 0, 0],
            [0, 0, -1, 1, 0],
            [0, 0, 0, -1, 1],
            [0, 0, 0, 0, -1],
        ]
    )
    mesh.boundary_condition_array = np.array(np.array([0, 0, 0, 0, 0]))
    mesh.convection_coefficent = 1

    mesh.discretization_type = "maccormack"
    return mesh


@pytest.fixture
def maccormack_solver(mock_linear_convective_mesh_mcormack):
    return solver.linear_convection_solver(
        mesh=mock_linear_convective_mesh_mcormack,
        initial_time=0,
        time_step_size=0.25,
        method="explicit",
    )


def test_maccormack_take_step_predictor(maccormack_solver):
    maccormack_solver.take_step()
    expected_predictor = [1, 2, 0, 0, 0]

    np.testing.assert_array_equal(maccormack_solver.predictor, expected_predictor)


def test_maccormack_take_step(maccormack_solver):
    maccormack_solver.take_step()
    expected_phi = [1, 1, 1, 0, 0]

    np.testing.assert_array_equal(maccormack_solver.mesh.phi, expected_phi)


@pytest.fixture
def solved_temp_1d():
    return np.array([25, 15, 5])


class TestSteadySolver:
    @pytest.fixture
    def lp(self):
        return np.array([[-3, 1, 0], [1, -2, 1], [0, 1, -3]])

    @pytest.fixture
    def bc(self):
        return np.array([60, 0, 0])

    # @pytest.fixture
    # def solved_temp(self):
    #     return np.array([25, 15, 5])

    def test_solve_steady(self, lp, bc, solved_temp_1d):
        expectd = solved_temp_1d
        actual = SteadySolver().solve(laplacian=lp, boundary_condition_array=bc)
        np.testing.assert_array_equal(x=actual, y=expectd)


class TestCartesianMesh_Integration:
    @pytest.fixture
    def CartesianMesh_1d(self):
        mesh = cartesian_mesh.CartesianMesh(
            dimensions=1, n_cells=[3], cordinates=[(0, 1)]
        )
        mesh.set_dirichlet_boundary("left", 30)
        mesh.set_dirichlet_boundary("right", 0)
        return mesh

    def test_solve_steady(
        self, CartesianMesh_1d, solved_temp_1d
    ):  # CartesianMesh_1d, solved_temp_1d):
        actual = solver.Solver(mesh=CartesianMesh_1d)
        actual.solve_steady()
        actual_phi = actual.mesh.phi.get_phi()
        expected = solved_temp_1d
        np.testing.assert_array_equal(x=actual_phi, y=expected)

    @pytest.fixture
    def CartesianMesh_2d(self):
        mesh = cartesian_mesh.CartesianMesh(
            dimensions=2, n_cells=[3, 4], cordinates=[(0, 1), (0, 2)]
        )
        mesh.set_dirichlet_boundary("left", 30)
        mesh.set_dirichlet_boundary("right", 30)
        mesh.set_dirichlet_boundary("bottom", 30)
        mesh.set_neumann_boundary("top", -10)
        return mesh

    @pytest.fixture
    def steady_2d_solved(self):
        return np.array(
            [
                [28.72076498, 27.94615599, 28.72076498],
                [29.70707761, 29.46041556, 29.70707761],
                [29.93022914, 29.86469587, 29.93022914],
                [29.98686165, 29.97407644, 29.98686165],
            ]
        )

    def test_solve_steady_2d(self, CartesianMesh_2d, steady_2d_solved):
        actual = solver.Solver(mesh=CartesianMesh_2d)
        actual.solve_steady()
        actual_phi = actual.mesh.phi.get_phi()
        expected = steady_2d_solved
        np.testing.assert_array_almost_equal(x=actual_phi, y=expected)

    def test_solve_unsteady_2d(self, CartesianMesh_2d, steady_2d_solved):
        actual = solver.Solver(
            mesh=CartesianMesh_2d,
            method="implicit",
        )
        actual.solve(t_final=20)
        actual_phi = actual.mesh.phi.get_phi()
        expected = steady_2d_solved
        # verify it matches the steady case
        np.testing.assert_array_almost_equal(x=actual_phi, y=expected)

    def test_unsteady_error_report_2d(self, CartesianMesh_2d, steady_2d_solved):
        """Test that the solver can report the difference between itself and steady state"""
        actual = solver.Solver(
            mesh=CartesianMesh_2d,
            method="implicit",
        )
        actual.solve(t_final=20, compute_error_flag=True)
        actual_phi = actual.mesh.phi.get_phi()
        steady_phi = steady_2d_solved

        expected = np.sqrt(np.sum((actual_phi - steady_phi) ** 2 / steady_phi))
        np.testing.assert_almost_equal(actual.error, expected)

    def test_record_time_2d(self, CartesianMesh_2d, steady_2d_solved):
        """Test that the solver can take an input to record every x timesteps"""
        actual = solver.Solver(
            mesh=CartesianMesh_2d,
            method="implicit",
        )
        # record_list = list()
        actual.solve(t_final=20, compute_error_flag=True, record_step=10)

        # verify it matches the steady case
        assert len(actual.saved_state_list) == 3

        actual_phi = actual.saved_state_list[2]["phi"]
        expected = steady_2d_solved
        # verify it matches the steady case
        np.testing.assert_array_almost_equal(x=actual_phi, y=expected)


if __name__ == "__main__":
    pytest.main()
