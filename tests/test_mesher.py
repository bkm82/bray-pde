import pytest
import numpy as np

from solver.mesher import create_1Dmesh


class Test_mesh:
    n_cells: int = 4
    x_range = [0, 1]
    mesh_type = "finite_volume"

    expected_xcell_center = np.array([0.125, 0.375, 0.625, 0.875])
    expected_delta_x = 0.25
    expected_differentiation_matrix = np.array(
        [[-2, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 1, -2]]
    )

    @pytest.fixture
    def mesh_fixture(self):
        return create_1Dmesh(
            self.x_range, n_cells=self.n_cells, mesh_type=self.mesh_type
        )

    expected_boundary_condition_left_dirichlet = np.array([100, 0, 0, 0])
    expected_boundary_condition_left_dirichlet_d2matrix = np.array(
        [[-3, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 1, -2]]
    )

    expected_boundary_condition_right_dirichlet = np.array([0, 0, 0, 100])
    expected_boundary_condition_right_dirichlet_d2matrix = np.array(
        [[-2, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 1, -3]]
    )

    # expected neumann with q = 50
    expected_boundary_condition_left_neumann = np.array([200, 0, 0, 0])
    expected_boundary_condition_left_neumann_d2matrix = np.array(
        [[-1, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 1, -2]]
    )

    expected_boundary_condition_right_neumann = np.array([0, 0, 0, 200])
    expected_boundary_condition_right_neumann_d2matrix = np.array(
        [[-2, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 1, -1]]
    )

    def test_set_n_cells(self, mesh_fixture):
        assert mesh_fixture.n_cells == self.n_cells

    def test_discritization(self, mesh_fixture):
        np.testing.assert_allclose(
            actual=mesh_fixture.xcell_center,
            desired=self.expected_xcell_center,
            atol=0.00001,
        )

    def test_delta_x(self, mesh_fixture):
        assert mesh_fixture.delta_x == self.expected_delta_x

    def test_set_diffusion_const(self, mesh_fixture):
        mesh_fixture.set_thermal_diffusivity(4)
        assert mesh_fixture.thermal_diffusivity == 4

    def test_initiate_differentiation_matrix(self, mesh_fixture):
        np.testing.assert_allclose(
            actual=mesh_fixture.differentiation_matrix,
            desired=self.expected_differentiation_matrix,
            atol=0.000001,
        )

    def test_set_internal_initial_temperature(self, mesh_fixture):
        mesh_fixture.set_cell_temperature(20)
        np.testing.assert_allclose(
            actual=mesh_fixture.temperature, desired=np.full(self.n_cells, 20)
        )

    def test_boundary_condition_initialized(self, mesh_fixture):
        np.testing.assert_allclose(
            actual=mesh_fixture.boundary_condition_array,
            desired=np.zeros(self.n_cells),
        )

    def test_left_dirclet_BC_array(self, mesh_fixture):
        mesh_fixture.set_dirichlet_boundary("left", 50)

        np.testing.assert_allclose(
            actual=mesh_fixture.boundary_condition_array,
            desired=self.expected_boundary_condition_left_dirichlet,
        )

    def test_left_dirclet_D2_Matrix(self, mesh_fixture):
        mesh_fixture.set_dirichlet_boundary("left", 50)

        np.testing.assert_allclose(
            actual=mesh_fixture.differentiation_matrix,
            desired=self.expected_boundary_condition_left_dirichlet_d2matrix,
        )

    def test_right_dirclet_BC_array(self, mesh_fixture):
        mesh_fixture.set_dirichlet_boundary("right", 50)
        np.testing.assert_allclose(
            actual=mesh_fixture.boundary_condition_array,
            desired=self.expected_boundary_condition_right_dirichlet,
        )

    def test_right_dirclet_D2_Matrix(self, mesh_fixture):
        mesh_fixture.set_dirichlet_boundary("right", 50)

        np.testing.assert_allclose(
            actual=mesh_fixture.differentiation_matrix,
            desired=self.expected_boundary_condition_right_dirichlet_d2matrix,
        )

    def test_left_neumann_BC_array(self, mesh_fixture):
        mesh_fixture.set_neumann_boundary("left", 50)

        np.testing.assert_allclose(
            actual=mesh_fixture.boundary_condition_array,
            desired=self.expected_boundary_condition_left_neumann,
        )

    def test_left_neumann_D2_Matrix(self, mesh_fixture):
        mesh_fixture.set_neumann_boundary("left", 50)

        np.testing.assert_allclose(
            actual=mesh_fixture.differentiation_matrix,
            desired=self.expected_boundary_condition_left_neumann_d2matrix,
        )

    def test_right_neumann_BC_array(self, mesh_fixture):
        mesh_fixture.set_neumann_boundary("right", 50)
        np.testing.assert_allclose(
            actual=mesh_fixture.boundary_condition_array,
            desired=self.expected_boundary_condition_right_neumann,
        )

    def test_dirchlet_BC_array_can_be_corrected(self, mesh_fixture):
        mesh_fixture.set_neumann_boundary("right", 50)
        mesh_fixture.set_dirichlet_boundary("right", 50)
        np.testing.assert_allclose(
            actual=mesh_fixture.boundary_condition_array,
            desired=self.expected_boundary_condition_right_dirichlet,
        )

    def test_right_neumann_D2_Matrix(self, mesh_fixture):
        mesh_fixture.set_neumann_boundary("right", 50)

        np.testing.assert_allclose(
            actual=mesh_fixture.differentiation_matrix,
            desired=self.expected_boundary_condition_right_neumann_d2matrix,
        )

    def test_unsuported_boundary_conndtion_raises(self, mesh_fixture):
        with pytest.raises(ValueError):
            mesh_fixture.set_neumann_boundary("lleft", 50)


# Test that a different x range can work
class Test_x_range(Test_mesh):
    x_range = [0, 4]
    expected_xcell_center = np.array([0.5, 1.5, 2.5, 3.5])
    expected_delta_x = 1
    expected_n_cells = 4
    expected_boundary_condition_left_neumann = np.array([50, 0, 0, 0])
    expected_boundary_condition_right_neumann = np.array([0, 0, 0, 50])


# Test that adding a 5 point finite_difference mesh type modifies the behavior
# @pytest.mark.xfail(reason=" finite_diff boundary conditions not implemented")
class Test_finite_difference(Test_mesh):
    n_cells = 5
    mesh_type = "finite_difference"

    # Expected Paramaters
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

    expected_boundary_condition_left_dirichlet = np.array([0, 0, 0, 0, 0])
    expected_boundary_condition_left_dirichlet_d2matrix = np.array(
        [
            [0, 0, 0, 0, 0],
            [1, -2, 1, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 1, -2, 1],
            [0, 0, 0, 1, -2],
        ]
    )

    expected_boundary_condition_right_dirichlet = np.array([0, 0, 0, 0, 0])
    expected_boundary_condition_right_dirichlet_d2matrix = np.array(
        [
            [-2, 1, 0, 0, 0],
            [1, -2, 1, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 1, -2, 1],
            [0, 0, 0, 0, 0],
        ]
    )

    # neuiman should be 2*q*delta_x when a point is on the boudnary
    expected_boundary_condition_left_neumann = np.array([25, 0, 0, 0, 0])
    expected_boundary_condition_left_neumann_d2matrix = np.array(
        [
            [-2, 2, 0, 0, 0],
            [1, -2, 1, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 1, -2, 1],
            [0, 0, 0, 1, -2],
        ]
    )

    expected_boundary_condition_right_neumann = np.array([0, 0, 0, 0, 25])

    expected_boundary_condition_right_neumann_d2matrix = np.array(
        [
            [-2, 1, 0, 0, 0],
            [1, -2, 1, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 1, -2, 1],
            [0, 0, 0, 2, -2],
        ]
    )


@pytest.fixture
def four_cell_mesh():
    return create_1Dmesh(x=[0, 1], n_cells=4)


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
