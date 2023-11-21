from solver.cartesian_solver import CartesianSolver, SteadySolver
import pytest
import numpy as np


@pytest.fixture
def mock_mesh(mocker):
    """
    Create to create a mock mesh object for use in testing the solver.
    """
    mesh = mocker.MagicMock()
    mesh.laplacian = np.array([[-3, 1, 0], [1, -2, 1], [0, 1, -3]])
    mesh.boundary_condition_array = np.array([60, 0, 0])
    return mesh


class TestCartesianSolver:
    @pytest.fixture
    def lp(self):
        return np.array([[-3, 1, 0], [1, -2, 1], [0, 1, -3]])

    @pytest.fixture
    def bc(self):
        return np.array([60, 0, 0])

    @pytest.fixture
    def solved_temp(self):
        return np.array([25, 15, 5])

    def test_solve_steady(self, lp, bc, solved_temp):
        expectd = solved_temp
        actual = SteadySolver().solve(laplacian=lp, boundary_condition_array=bc)

        np.testing.assert_array_equal(x=actual, y=expectd)

    def test_carteisan_steady_solve(self, lp, bc, solved_temp):
        expectd = solved_temp
        # init = CartesianSolver(method = "steady")
        init = CartesianSolver()
        actual = init.solve(laplacian=lp, boundary_condition_array=bc)
        np.testing.assert_array_equal(x=actual, y=expectd)

    # def test_steady_solver_init(self, lp, bc):
    #     solver = SteadySolver(
    #         laplacian=lp,
    #         boundary_condition_array=bc
    #     )
    #     np.testing.assert_array_equal(x = solver.laplacian, y =  lp)


if __name__ == "__main__":
    pytest.main()
