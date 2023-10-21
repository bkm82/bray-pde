import pytest
import numpy as np

from solver.mesher import create_1Dmesh


# @pytest.fixture
# def three_point_mesh():
#     return create_1Dmesh(x=[0, 1], n_cells=3)


@pytest.fixture
def four_cell_mesh():
    return create_1Dmesh(x=[0, 1], n_cells=4)


def test_1dmesh_discritize(four_cell_mesh):
    assert np.array_equal(
        four_cell_mesh.xcell_center, np.array([0.125, 0.375, 0.625, 0.875])
    )


def test_set_deltax(four_cell_mesh):
    assert four_cell_mesh.delta_x == 0.25


def test_set_n_cells(four_cell_mesh):
    assert four_cell_mesh.n_cells == 4


def test_set_diffusion_const(four_cell_mesh):
    four_cell_mesh.set_thermal_diffusivity(4)
    assert four_cell_mesh.thermal_diffusivity == 4


def test_set_differentiation_matrix(four_cell_mesh):
    expected_differentiation_matrix = np.array(
        [[-2, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 1, -2]]
    )

    assert np.array_equal(
        four_cell_mesh.differentiation_matrix, expected_differentiation_matrix
    )


def test_internal_initial_temperature(four_cell_mesh):
    four_cell_mesh.set_cell_temperature(20)

    assert np.array_equal(four_cell_mesh.temperature, np.array([20, 20, 20, 20]))


def init_boundary_condtion(four_cell_mesh):
    expected_boundary_condition_array = np.array([0, 0, 0, 0])
    assert np.array_equal(
        four_cell_mesh.boundary_condition_array, expected_boundary_condition_array
    )


def test_set_left_dirclet_BC_array(four_cell_mesh):
    four_cell_mesh.set_dirichlet_boundary("left", 50)

    assert np.array_equal(
        four_cell_mesh.boundary_condition_array, np.array([100, 0, 0, 0])
    )


def test_set_left_dirclet_D2matrix(four_cell_mesh):
    expected_differentiation_matrix = np.array(
        [[-3, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 1, -2]]
    )

    four_cell_mesh.set_dirichlet_boundary("left", 50)

    assert np.array_equal(
        four_cell_mesh.differentiation_matrix, expected_differentiation_matrix
    )


def test_set_right_dirclet_BC_array(four_cell_mesh):
    four_cell_mesh.set_dirichlet_boundary("right", 50)

    assert np.array_equal(
        four_cell_mesh.boundary_condition_array, np.array([0, 0, 0, 100])
    )


def test_set_right_dirclet_D2matrix(four_cell_mesh):
    expected_differentiation_matrix = np.array(
        [[-2, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 1, -3]]
    )

    four_cell_mesh.set_dirichlet_boundary("right", 50)

    assert np.array_equal(
        four_cell_mesh.differentiation_matrix, expected_differentiation_matrix
    )


def test_unsuported_boundary_conndtion_raises(four_cell_mesh):
    with pytest.raises(ValueError):
        four_cell_mesh.set_dirichlet_boundary("top", 50)


def test_set_left_neumann_BC(four_cell_mesh):
    four_cell_mesh.set_neumann_boundary("left", 50)

    assert np.array_equal(
        four_cell_mesh.boundary_condition_array, np.array([200, 0, 0, 0])
    )


def test_set_left_neumann_D2matrix(four_cell_mesh):
    expected_differentiation_matrix = np.array(
        [[-1, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 1, -2]]
    )

    four_cell_mesh.set_neumann_boundary("left")

    assert np.array_equal(
        four_cell_mesh.differentiation_matrix, expected_differentiation_matrix
    )


def test_set_right_neumann_BC(four_cell_mesh):
    four_cell_mesh.set_neumann_boundary("right", 50)

    assert np.array_equal(
        four_cell_mesh.boundary_condition_array, np.array([0, 0, 0, 200])
    )


def test_set_right_neumann_D2matrix(four_cell_mesh):
    expected_differentiation_matrix = np.array(
        [[-2, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 1, -1]]
    )

    four_cell_mesh.set_neumann_boundary("right")

    assert np.array_equal(
        four_cell_mesh.differentiation_matrix, expected_differentiation_matrix
    )
