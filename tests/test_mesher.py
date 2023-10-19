import pytest
import numpy as np

from solver.mesher import create_1Dmesh


@pytest.fixture
def three_point_mesh():
    return create_1Dmesh(x=[0, 1], n_points=3)


@pytest.fixture
def four_point_mesh():
    return create_1Dmesh(x=[0, 1], n_points=4)


def test_1dmesh_discritize(three_point_mesh):
    assert np.array_equal(three_point_mesh.node, np.array([0, 0.5, 1]))


def test_set_deltax(three_point_mesh):
    assert three_point_mesh.delta_x == 0.5


def test_set_n_points(three_point_mesh):
    assert three_point_mesh.n_points == 3


def test_set_differentiation_matrix(three_point_mesh):
    expected_differentiation_matrix = np.array([[-2, 1, 0], [1, -2, 1], [0, 1, -2]])

    assert np.array_equal(
        three_point_mesh.differentiation_matrix, expected_differentiation_matrix
    )


def test_set_temperature_vector(three_point_mesh):
    expected_temperature = np.array([0, 0, 0])
    assert np.array_equal(three_point_mesh.temperature, expected_temperature)


# def test_internal_initial_temperature(four_point_mesh):
#     set_internal_temperature(four_point_mesh, 20)

#     assert np.array_equal(four_point_mesh.temperature, np.array([0, 20, 20, 0]))
