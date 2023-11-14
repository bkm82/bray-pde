from solver import cartesian_solver
import pytest
import numpy as np


class TestCartesianSolver:
    def test_carteisan_steady_solve(self):
        ## a simple 1d case with 30 dirichlet on left, 0 dirichlet on right)
        laplacian = np.array([[-3, 1, 0], [1, -2, 1], [0, 1, -3]])
        boundary_condition = np.array([60, 0, 0])
        expectd = np.array([25, 15, 5])
        actual = cartesian_solver.CartesianSolver(
            laplacian=laplacian, boundary_condition_array=boundary_condition
        ).solve_steady()
        np.testing.assert_array_equal(x=actual, y=expectd)


if __name__ == "__main__":
    pytest.main()
