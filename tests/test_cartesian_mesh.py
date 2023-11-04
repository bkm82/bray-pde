import pytest
import numpy as np
from solver.cartesian_mesh import cartesian_mesh


class Test_1d_cartesian_mesh:
    @pytest.fixture
    def one_d_mesh(self):
        return cartesian_mesh(dimensions=1, cordinates=[(0, 1)], n_cells=[4])

    def test_1d_cell_width(self, one_d_mesh):
        assert one_d_mesh.x_grid.cell_width == 0.25

    def test_1d_cell_cordinates(self, one_d_mesh):
        np.testing.assert_array_equal(
            x=one_d_mesh.x_grid.cell_cordinates,
            y=np.array([0.125, 0.375, 0.625, 0.875]),
        )


class Test_2d_cartesian_mesh:
    @pytest.fixture
    def two_d_mesh(self):
        return cartesian_mesh(dimensions=2, cordinates=[(0, 1), (0, 2)], n_cells=[4, 4])

    cell_width_inputs = [("x_grid", 0.25), ("y_grid", 0.5)]

    @pytest.mark.parametrize("dimension,expected", cell_width_inputs)
    def test_2d_cell_width(self, two_d_mesh, dimension, expected):
        grid = getattr(two_d_mesh, dimension)
        assert grid.cell_width == expected

    coordinates_inputs = [
        ("x_grid", np.array([0.125, 0.375, 0.625, 0.875])),
        ("y_grid", np.array([0.25, 0.75, 1.25, 1.75])),
    ]

    @pytest.mark.parametrize("dimension,expected", coordinates_inputs)
    def test_2d_cell_cordinates(self, two_d_mesh, dimension, expected):
        grid = getattr(two_d_mesh, dimension)
        np.testing.assert_array_equal(x=grid.cell_cordinates, y=expected)


class Test_cartesian_mesh_exceptions:
    """Test features expected to raise an exception"""

    exception_inputs = [
        ({"dimensions": 3}),
        ({"dimensions": 2, "cordinates": [(0, 1)]}),
        ({"dimensions": 2, "n_cells": [5]}),
        ({"dimensions": 1}),
    ]

    @pytest.mark.parametrize("inputs", exception_inputs)
    def test_three_dimensions_raises(self, inputs):
        with pytest.raises(ValueError):
            cartesian_mesh(**inputs)
