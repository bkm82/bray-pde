import numpy as np


class CartesianSolver:
    """A solver to solve a cartesian mesh"""

    def __init__(self, laplacian, boundary_condition_array):
        self.laplacian = laplacian
        self.boundary_condition_array = boundary_condition_array

    def solve_steady(self):
        return np.linalg.solve(self.laplacian, -self.boundary_condition_array)


class SteadySolver(CartesianSolver):
    def __init__(self):
        pass

    def solve(self, laplacian, boundary_condition_array):
        return np.linalg.solve(laplacian, -boundary_condition_array)
