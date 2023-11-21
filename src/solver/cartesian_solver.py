import numpy as np


class SteadySolver(object):
    def solve(self, laplacian, boundary_condition_array):
        return np.linalg.solve(laplacian, -boundary_condition_array)


class CartesianSolver:
    def __init__(self, solver=SteadySolver()):
        self.solver = solver

    #     """A solver to solve a cartesian mesh"""
    def solve(self, **kwags):
        return self.solver.solve(**kwags)
