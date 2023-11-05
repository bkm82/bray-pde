import pytest
import numpy as np
from solver.mesher import heat_diffusion_mesh
from solver.mesher import create_1Dmesh
from unittest.mock import patch, MagicMock
from solver import mesher


class Test_mesh:
    n_cells: int = 4
    x_range = [0, 1]
    mesh_type: str = "finite_volume"
    expected_xcell_center = np.array([0.125, 0.375, 0.625, 0.875])
    expected_delta_x: float = 0.25

    @pytest.fixture
    def mesh_fixture(self):
        return create_1Dmesh(
            self.x_range, n_cells=self.n_cells, mesh_type=self.mesh_type
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


############################################################
class Test_maccormac_linear_convection_mesh(Test_mesh):
    n_cells = 5
    mesh_type = "finite_difference"

    # Expected Paramaters
    expected_xcell_center = np.array([0, 0.25, 0.5, 0.75, 1])
    expected_n_cells = 5
    expected_laplacian_matrix = np.array(
        [
            [1, 0, 0, 0, 0],
            [-1, 1, 0, 0, 0],
            [0, -1, 1, 0, 0],
            [0, 0, -1, 1, 0],
            [0, 0, 0, -1, 1],
        ]
    )

    expected_predictor_differentiation_matrix = np.array(
        [
            [-1, 1, 0, 0, 0],
            [0, -1, 1, 0, 0],
            [0, 0, -1, 1, 0],
            [0, 0, 0, -1, 1],
            [0, 0, 0, 0, -1],
        ]
    )

    expected_left_dirichlet_laplacian_matrix = np.array(
        [
            [0, 0, 0, 0, 0],
            [-1, 1, 0, 0, 0],
            [0, -1, 1, 0, 0],
            [0, 0, -1, 1, 0],
            [0, 0, 0, -1, 1],
        ]
    )
    expected_left_dirichlet_predctor_laplacian_matrix = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, -1, 1, 0, 0],
            [0, 0, -1, 1, 0],
            [0, 0, 0, -1, 1],
            [0, 0, 0, 0, -1],
        ]
    )

    expected_left_dirichlet_boundary_condition_array = np.array([0, 0, 0, 0, 0])

    expected_left_dirichlet_phi = np.array([5, 0, 0, 0, 0])

    @pytest.fixture
    def mesh_fixture(self):
        return mesher.linear_convection_mesh(
            self.x_range,
            n_cells=self.n_cells,
            mesh_type=self.mesh_type,
            convection_coefficient=1,
            discretization_type="maccormack",
        )

    def test_maccormack_laplacian_matrix(self, mesh_fixture):
        np.testing.assert_array_equal(
            mesh_fixture.laplacian_matrix, self.expected_laplacian_matrix
        )

    def test_maccormack_predictor_differentiation_matrix(self, mesh_fixture):
        np.testing.assert_array_equal(
            mesh_fixture.predictor_differentiation_matrix,
            self.expected_predictor_differentiation_matrix,
        )

    def test_mcormak_left_dirichlet_laplacian_matrix(self, mesh_fixture):
        mesh_fixture.set_dirichlet_boundary("left", 5)
        np.testing.assert_array_equal(
            mesh_fixture.laplacian_matrix,
            self.expected_left_dirichlet_laplacian_matrix,
        )

    def test_mcormak_left_dirichlet_predictor_differentiation_matrix(
        self, mesh_fixture
    ):
        mesh_fixture.set_dirichlet_boundary("left", 5)
        np.testing.assert_array_equal(
            mesh_fixture.predictor_differentiation_matrix,
            self.expected_left_dirichlet_predctor_laplacian_matrix,
        )


############################################################
class Test_central_linear_convection_mesh(Test_mesh):
    n_cells = 5
    mesh_type = "finite_difference"

    # Expected Paramaters
    expected_xcell_center = np.array([0, 0.25, 0.5, 0.75, 1])
    expected_n_cells = 5
    expected_laplacian_matrix = np.array(
        [
            [0, -0.5, 0, 0, 0],
            [0.5, 0, -0.5, 0, 0],
            [0, 0.5, 0, -0.5, 0],
            [0, 0, 0.5, 0, -0.5],
            [0, 0, 0, 0.5, 0],
        ]
    )
    expected_left_dirichlet_laplacian_matrix = np.array(
        [
            [0, 0, 0, 0, 0],
            [0.5, 0, -0.5, 0, 0],
            [0, 0.5, 0, -0.5, 0],
            [0, 0, 0.5, 0, -0.5],
            [0, 0, 0, 0.5, 0],
        ]
    )

    expected_right_laplacian_matrix = np.array(
        [
            [0, -0.5, 0, 0, 0],
            [0.5, 0, -0.5, 0, 0],
            [0, 0.5, 0, -0.5, 0],
            [0, 0, 0.5, 0, -0.5],
            [0, 0, 1.5, -1, 0.5],
        ]
    )
    expected_left_dirichlet_boundary_condition_array = np.array([0, 0, 0, 0, 0])

    expected_left_dirichlet_phi = np.array([5, 0, 0, 0, 0])

    @pytest.fixture
    def mesh_fixture(self):
        return mesher.linear_convection_mesh(
            self.x_range,
            n_cells=self.n_cells,
            mesh_type=self.mesh_type,
            convection_coefficient=1,
            discretization_type="central",
        )

    def test_central_laplacian_matrix(self, mesh_fixture):
        np.testing.assert_array_equal(
            mesh_fixture.laplacian_matrix, self.expected_laplacian_matrix
        )

    def test_central_left_dirichlet_laplacian_matrix(self, mesh_fixture):
        mesh_fixture.set_dirichlet_boundary("left", 5)
        np.testing.assert_array_equal(
            mesh_fixture.laplacian_matrix,
            self.expected_left_dirichlet_laplacian_matrix,
        )

    def test_central_left_dirichlet_boundary_condtion(self, mesh_fixture):
        mesh_fixture.phi.set_phi(0)
        mesh_fixture.set_dirichlet_boundary("left", 5)
        np.testing.assert_array_equal(
            mesh_fixture.boundary_condition_array,
            self.expected_left_dirichlet_boundary_condition_array,
        )

    def test_central_left_dirichlet_phi(self, mesh_fixture):
        mesh_fixture.phi.set_phi(0)
        mesh_fixture.set_dirichlet_boundary("left", 5)
        np.testing.assert_array_equal(
            mesh_fixture.phi.get_phi(),
            self.expected_left_dirichlet_phi,
        )

    def test_right_laplacian_matrix(self, mesh_fixture):
        mesh_fixture.set_right_boundary()
        np.testing.assert_array_equal(
            mesh_fixture.laplacian_matrix,
            self.expected_right_laplacian_matrix,
        )

    def test_discritization_error(self):
        with pytest.raises(ValueError):
            mesher.linear_convection_mesh(
                self.x_range,
                n_cells=self.n_cells,
                mesh_type=self.mesh_type,
                convection_coefficient=1,
                discretization_type="downwind",
            )


class Test_linear_convection_mesh(Test_mesh):
    mesh_type = "finite_difference"
    expected_xcell_center = np.array([0, 1 / 3, 2 / 3, 1])
    expected_delta_x: float = 1 / 3

    @pytest.fixture
    def mesh_fixture(self):
        return mesher.linear_convection_mesh(
            self.x_range,
            n_cells=self.n_cells,
            mesh_type=self.mesh_type,
            convection_coefficient=1,
        )

    expected_laplacian_matrix = np.array(
        [[-1, 0, 0, 0], [1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1]]
    )

    expected_left_dirichlet_laplacian_matrix = np.array(
        [[0, 0, 0, 0], [1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1]]
    )
    expected_left_boundary_condition_array = np.array([0, 0, 0, 0])

    def test_initiate_laplacian_matrix(self, mesh_fixture):
        # mesh_fixture.set_centeral_
        np.testing.assert_allclose(
            actual=mesh_fixture.laplacian_matrix,
            desired=self.expected_laplacian_matrix,
            atol=0.000001,
        )

    def test_initiate_differentation_matrix_neg(self):
        """Test that a negative convcection coefficent shows unsupported error"""
        with pytest.raises(ValueError):
            mesher.linear_convection_mesh(
                x=[0, 1], n_cells=4, convection_coefficient=-1
            )

    def test_set_phi(self, mesh_fixture):
        mesh_fixture.phi.set_phi(phi=[1, 2, 4, 5])
        np.testing.assert_equal(mesh_fixture.phi.get_phi(), [1, 2, 4, 5])

    def test_set_phi_float(self, mesh_fixture):
        mesh_fixture.phi.set_phi(phi=5.0)
        np.testing.assert_equal(mesh_fixture.phi.get_phi(), [5, 5, 5, 5])

    def test_set_phi_int(self, mesh_fixture):
        mesh_fixture.phi.set_phi(phi=5)
        np.testing.assert_equal(mesh_fixture.phi.get_phi(), [5, 5, 5, 5])

    def test_set_phi_float_lenght(self, mesh_fixture):
        mesh_fixture.phi.set_phi(phi=5.0)
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(mesh_fixture.phi.get_phi(), [5, 5, 5, 5, 5])

    def test_set_phi_wrong_shape_error(self, mesh_fixture):
        """Ensure the shape of phi matches the shape of x"""
        with pytest.raises(ValueError):
            mesh_fixture.phi.set_phi([1, 2, 3, 4, 5])

    def test_set_phi_wrong_type(self, mesh_fixture):
        with pytest.raises(TypeError):
            mesh_fixture.phi.set_phi("string")

    def test_convection_coeff_saved(self, mesh_fixture):
        assert mesh_fixture.convection_coefficent == 1

    def test_left_dirichlet_boundary(self, mesh_fixture):
        mesh_fixture.set_dirichlet_boundary("left", 5)
        np.testing.assert_array_equal(
            mesh_fixture.boundary_condition_array,
            self.expected_left_boundary_condition_array,
        )

    def test_left_dirichlet_differentiaton_matrix(self, mesh_fixture):
        mesh_fixture.set_dirichlet_boundary("left", 5)
        np.testing.assert_array_equal(
            mesh_fixture.laplacian_matrix,
            self.expected_left_dirichlet_laplacian_matrix,
        )

    def test_right_dirichlet_boundary(self, mesh_fixture):
        """because right bc is not implemented yet ensure a value error si raised"""
        with pytest.raises(ValueError):
            mesh_fixture.set_dirichlet_boundary("right", 5)


class Test_heat_diffusion_mesh(Test_mesh):
    @pytest.fixture
    def mesh_fixture(self):
        return heat_diffusion_mesh(
            self.x_range, n_cells=self.n_cells, mesh_type=self.mesh_type
        )

    expected_laplacian_matrix = np.array(
        [[-2, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 1, -2]]
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

    def test_set_diffusion_const(self, mesh_fixture):
        mesh_fixture.set_thermal_diffusivity(4)
        assert mesh_fixture.thermal_diffusivity == 4

    def test_initiate_laplacian_matrix(self, mesh_fixture):
        np.testing.assert_allclose(
            actual=mesh_fixture.laplacian_matrix,
            desired=self.expected_laplacian_matrix,
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
            actual=mesh_fixture.laplacian_matrix,
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
            actual=mesh_fixture.laplacian_matrix,
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
            actual=mesh_fixture.laplacian_matrix,
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
            actual=mesh_fixture.laplacian_matrix,
            desired=self.expected_boundary_condition_right_neumann_d2matrix,
        )

    def test_unsuported_boundary_conndtion_raises(self, mesh_fixture):
        with pytest.raises(ValueError):
            mesh_fixture.set_neumann_boundary("lleft", 50)


# Test that a different x range can work
class Test_x_range(Test_heat_diffusion_mesh):
    x_range = [0, 4]
    expected_xcell_center = np.array([0.5, 1.5, 2.5, 3.5])
    expected_delta_x = 1
    expected_n_cells = 4
    expected_boundary_condition_left_neumann = np.array([50, 0, 0, 0])
    expected_boundary_condition_right_neumann = np.array([0, 0, 0, 50])


class Test_finite_difference(Test_heat_diffusion_mesh):
    n_cells = 5
    mesh_type = "finite_difference"

    # Expected Paramaters
    expected_xcell_center = np.array([0, 0.25, 0.5, 0.75, 1])
    expected_n_cells = 5
    expected_laplacian_matrix = np.array(
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
def five_cell_mesh():
    return heat_diffusion_mesh(x=[0, 1], n_cells=5, mesh_type="finite_difference")


def test_finite_difference_dirichlet_set_temperature(five_cell_mesh):
    """
    Regression test for issue 10
    Description
    When using the finite difference, setting the dirichlet boundary
    does not enforce the temeprature
    """

    five_cell_mesh.set_cell_temperature(20)
    five_cell_mesh.set_dirichlet_boundary("left", 50)
    five_cell_mesh.set_dirichlet_boundary("right", 45)
    expected_temperature = np.array([50, 20, 20, 20, 45])
    np.testing.assert_allclose(
        actual=five_cell_mesh.temperature, desired=expected_temperature
    )


@pytest.mark.xfail(reason="feature improvment recomendation")
def test_finite_difference_dirichlet_overwride_temperature(five_cell_mesh):
    """
    Regression test for issue 10
    Description
    Initial resolution for issue 10 allowed for setting the boundary temperatures after the initial temperatures
    However if the internal temperatures are set after the boundarys will be overwritten
    Below is a testcase of optimal behavior
    """

    five_cell_mesh.set_dirichlet_boundary("left", 50)
    five_cell_mesh.set_cell_temperature(20)
    five_cell_mesh.set_dirichlet_boundary("right", 45)

    expected_temperature = np.array([50, 20, 20, 20, 45])
    np.testing.assert_allclose(
        actual=five_cell_mesh.temperature, desired=expected_temperature
    )


def test_init():
    from solver import mesher

    with patch.object(mesher, "main", MagicMock()) as mock_main:
        with patch.object(mesher, "__name__", "__main__"):
            mesher.init()
            mock_main.assert_called_once()


class Test_upwind_linear_convection_mesh_finite_volume(Test_linear_convection_mesh):
    n_cells = 4
    mesh_type = "finite_volume"
    discritization_type = "upwind"
    expected_delta_x = 0.25
    # Expected Paramaters
    expected_xcell_center = np.array([0.125, 0.375, 0.625, 0.875])
    expected_n_cells = 4
    expected_laplacian_matrix = np.array(
        [
            [-1, 0, 0, 0],
            [1, -1, 0, 0],
            [0, 1, -1, 0],
            [0, 0, 1, -1],
        ]
    )
    expected_left_dirichlet_laplacian_matrix = np.array(
        [
            [-1, 0, 0, 0],
            [1, -1, 0, 0],
            [0, 1, -1, 0],
            [0, 0, 1, -1],
        ]
    )

    expected_right_laplacian_matrix = np.array(
        [
            [-1, 0, 0, 0],
            [1, -1, 0, 0],
            [0, 1, -1, 0],
            [0, 0, 1, -1],
        ]
    )

    expected_left_boundary_condition_array = np.array([5, 0, 0, 0])

    expected_left_dirichlet_phi = np.array([0, 0, 0, 0])
