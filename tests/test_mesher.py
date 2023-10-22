import pytest
import numpy as np

from solver.mesher import create_1Dmesh


class Test_mesh:
    expected_n_cells: int = 4
    expected_xcell_center = np.array([0.125, 0.375, 0.625, 0.875])
    mesh = create_1Dmesh(x=[0, 1], n_cells=4)
    expected_delta_x = 0.25
    expected_differentiation_matrix = np.array(
        [[-2, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 1, -2]]
    )

    expected_boundary_condition_left_dirichlet = np.array([100, 0, 0, 0])
    # expected_boundary_condition_right_dirichlet
    # expected_boundary_condition_left_neumann
    # expected_boundary_condition_right_neuman =

    def test_set_n_cells(self):
        assert self.mesh.n_cells == self.expected_n_cells

    def test_discritization(self):
        np.testing.assert_allclose(
            actual=self.mesh.xcell_center,
            desired=self.expected_xcell_center,
            atol=0.00001,
        )

    def test_delta_x(self):
        assert self.mesh.delta_x == self.expected_delta_x

    def test_set_diffusion_const(self):
        self.mesh.set_thermal_diffusivity(4)
        assert self.mesh.thermal_diffusivity == 4

    def test_initiate_differentiation_matrix(self):
        np.testing.assert_allclose(
            actual=self.mesh.differentiation_matrix,
            desired=self.expected_differentiation_matrix,
            atol=0.000001,
        )

    def test_set_internal_initial_temperature(self):
        self.mesh.set_cell_temperature(20)
        np.testing.assert_allclose(
            actual=self.mesh.temperature, desired=np.full(self.expected_n_cells, 20)
        )

    def test_boundary_condition_initialized(self):
        np.testing.assert_allclose(
            actual=self.mesh.boundary_condition_array,
            desired=np.zeros(self.expected_n_cells),
        )

    def test_left_dirclet_BC_array(self):
        self.mesh.set_dirichlet_boundary("left", 50)

        np.testing.assert_allclose(
            actual=self.mesh.boundary_condition_array,
            desired=self.expected_boundary_condition_left_dirichlet,
        )


# Test that a different x range can work
class Test_x_range(Test_mesh):
    mesh = create_1Dmesh(x=[0, 4], n_cells=4)
    expected_xcell_center = np.array([0.5, 1.5, 2.5, 3.5])
    expected_delta_x = 1


# Test that adding a mesh_type can be accepted
class Test_finite_volume(Test_mesh):
    mesh = create_1Dmesh(x=[0, 1], n_cells=4, mesh_type="finite_volume")


# Test that adding a 5 point finite_difference mesh type modifies the behavior
@pytest.mark.xfail(reason=" finite_diff boundary conditions not implemented")
class Test_finite_difference(Test_mesh):
    mesh = create_1Dmesh(x=[0, 1], n_cells=5, mesh_type="finite_difference")
    expected_xcell_center = np.array([0, 0.25, 0.5, 0.75, 1])
    expected_n_cells = 5
    expected_differentiation_matrix = np.array(
        [
            [-2, 1, 0, 0, 0],
            [1, -2, 1, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 1, -2, 1],
            [0, 0, 0, 1, -2],
        ]
    )

    ###### VERIFY THIS IS CORRECT
    expected_boundary_condition_left_dirichlet = np.array([0, 0, 0, 0, 0])
    ###### VERIFY THIS IS CORRECT


@pytest.fixture
def four_cell_mesh():
    return create_1Dmesh(x=[0, 1], n_cells=4)


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
