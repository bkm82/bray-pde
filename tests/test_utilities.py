from solver import utilities
from solver import cartesian_mesh
import numpy as np
import pytest
import math
import logging
from unittest.mock import PropertyMock


class TestMeshReshaper:
    @pytest.fixture
    def long_array(self):
        return np.arange(10)

    @pytest.fixture
    def wide_array(self, long_array):
        return np.reshape(long_array, (2, 5))

    def test_reshape_long(self, long_array, wide_array):
        actual = utilities.MeshReshaper(x_cells=5, y_cells=2).to_long(wide_array)
        expected = long_array

        np.testing.assert_array_equal(x=actual, y=expected)

    def test_reshape_wide(self, long_array, wide_array):
        actual = utilities.MeshReshaper(x_cells=5, y_cells=2).to_wide(long_array)
        expected = wide_array

        np.testing.assert_array_equal(x=actual, y=expected)


class TestEnergyBalance:
    @pytest.fixture
    def mock_mesh(self, mocker):
        """
        Create to create a mock mesh object for use in testing the solver.
        Mesh configuration
        N_elements =
        Left boundary = Dirchilet at 50c
        Right boundary = neuiman with q=0
        :return: mesh
        """

        mesh = mocker.MagicMock()
        mesh.phi.phi = np.array(
            [
                [28.72076498, 27.94615599, 28.72076498],
                [29.70707761, 29.46041556, 29.70707761],
                [29.93022914, 29.86469587, 29.93022914],
                [29.98686165, 29.97407644, 29.98686165],
            ]
        )
        mesh.x_bc_reshape = np.array(
            [[60.0, 0.0, 60.0], [60.0, 0.0, 60.0], [60.0, 0.0, 60.0], [60.0, 0.0, 60.0]]
        )

        mesh.y_bc_reshape = np.array(
            [[-5.0, -5.0, -5.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [60.0, 60.0, 60.0]]
        )
        mesh.phi.get_phi = mocker.MagicMock(return_value=mesh.phi.phi)

        mesh.boundary_condition_dict = {
            "left": "dirichlet",
            "right": "dirichlet",
            "bottom": "dirichlet",
            "top": "neuimann",
        }

        mesh.grid["x_grid"].cell_width = 5
        # mocker.MagicMock(return_value = [4,5])
        # mesh.grid["y_grid"].cell_width = 3

        mesh.n_cells = [3, 4]
        mesh.conductivity = 1
        return mesh

    def test_my_mock_mesh(self, mock_mesh):
        expected = np.array(
            [
                [28.72076498, 27.94615599, 28.72076498],
                [29.70707761, 29.46041556, 29.70707761],
                [29.93022914, 29.86469587, 29.93022914],
                [29.98686165, 29.97407644, 29.98686165],
            ]
        )
        actual = mock_mesh.phi.get_phi()
        np.testing.assert_array_equal(x=actual, y=expected)

    @pytest.fixture
    def cartesian_mesh(self):
        mesh = cartesian_mesh.CartesianMesh(
            dimensions=2, n_cells=[3, 4], cordinates=[(0, 1), (0, 2)], conductivity=1
        )

        mesh.set_dirichlet_boundary(side="left", phi=30)
        mesh.set_dirichlet_boundary(side="right", phi=30)
        mesh.set_dirichlet_boundary(side="bottom", phi=30)
        mesh.set_neumann_boundary(side="top", flux=-10)
        mesh.phi.set_phi(
            [
                [28.72076498, 27.94615599, 28.72076498],
                [29.70707761, 29.46041556, 29.70707761],
                [29.93022914, 29.86469587, 29.93022914],
                [29.98686165, 29.97407644, 29.98686165],
            ]
        )
        return mesh

    @pytest.mark.parametrize(
        "side,expected_fixture",
        [
            ("left", 4.965199833748016),
            ("right", 4.965199833748016),
            ("bottom", 0.06960033250395742),
            ("top", -10),
            ("all", 0.00000001),
            ("cells", 0.00000001),
        ],
    )
    def test_left_energy_balance(self, cartesian_mesh, side, expected_fixture):
        actual = utilities.EnergyBalance(mesh=cartesian_mesh).flux(side)
        expected = expected_fixture
        logging.debug(f"actual:{actual}, expected:{expected}")
        assert math.isclose(actual, expected, abs_tol=0.000001)


class TestAxisParse:
    @pytest.mark.parametrize("input_str,expected", [("x_grid", "x"), ("y_grid", "y")])
    def test_axis_parse(self, input_str, expected):
        actual = utilities.Parser().parse(input_str)
        assert actual == expected
