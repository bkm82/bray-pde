import numpy as np


class create_1Dmesh:
    """
    A class representing a 1D Mesh.

    Atributes:
    xcell_center (np.array): An array of node x positions
    n_points (int): The number of mesh points
    delta_x (float) : The distance between mesh points

    """

    def __init__(self, x, n_cells, mesh_type="finite_volume"):
        """
        Initialize the Mesh object.

        Keyword Arguments:
        x -- the spatial domain of the mesh in the form [x_min, x_max]
        n_cells -- the number of points for discritization.
        use n_cells as the number of points for the finite volume case

        Example
        mesh = create_1Dmesh(x=[0, 1], n_cells=3)
        mesh.xcell_center = np.array([0.125,0.375, 0.625, 0.875])
        mesh.deltax = 0.25
        """
        self.n_cells = n_cells
        self.mesh_type = mesh_type
        if mesh_type == "finite_volume":
            self.delta_x = (x[1] - x[0]) / (n_cells)
            self.xcell_center = np.linspace(
                x[0] + (self.delta_x / 2), x[1] - self.delta_x / 2, n_cells
            )
        elif mesh_type == "finite_difference":
            self.delta_x = (x[1] - x[0]) / (n_cells - 1)
            self.xcell_center = np.linspace(x[0], x[1], n_cells)
        else:
            raise ValueError("Mesh type not supported")

        self.create_differentiation_matrix(self.xcell_center)
        self.boundary_condition_array = np.zeros(n_cells)
        # self.temperature = np.zeros(n_cells)

    def create_differentiation_matrix(self, nodes):
        """Initialaze an empty differential matrix atribute."""
        self.create_differentiation_matrix = None


class heat_diffusion_mesh(create_1Dmesh):
    """Create a heat diffusion mesh."""

    def __init__(self, x, n_cells: int, mesh_type: str = "finite_volume"):
        """
        Initialize a heat diffusion mesh object.

        Parameters:
           x (type) : the spatial discritization of the domain
           n_cells (int): The number of cells to discritize the domain into
           mesh_type (string) : finite_voluem (default) or finite_difference
        """
        super().__init__(x, n_cells, mesh_type)
        self.temperature = np.zeros(n_cells)

    def set_cell_temperature(self, temperature):
        """
        Set the temperature for internal nodes.

        Example:running mesh.set_internal_temperature(20) would result
        in np.array([20, 20, 20, 20]
        """
        self.temperature = temperature * np.ones(self.n_cells)

    def set_thermal_diffusivity(self, thermal_diffusivity):
        """Set a diffusion constant in square meters per second."""
        self.thermal_diffusivity = thermal_diffusivity

    def create_differentiation_matrix(self, nodes):
        """Create a differentiation matrix."""
        shape = np.shape(nodes)[0]
        upper = np.diagflat(np.repeat(1, shape - 1), 1)
        middle = -2 * np.identity(shape)
        differentiation_matrix = upper + np.transpose(upper) + middle
        self.differentiation_matrix = differentiation_matrix

    def set_dirichlet_boundary(self, side, temperature):
        """Update boundary array and D2 for a dirichlet boundary."""
        if side == "left":
            array_index = 0
        elif side == "right":
            array_index = -1
        else:
            raise ValueError("Side must input must be left or right")

        if self.mesh_type == "finite_volume":
            self.boundary_condition_array[array_index] = 2 * temperature
            self.differentiation_matrix[array_index, array_index] = -3

        elif self.mesh_type == "finite_difference":
            self.boundary_condition_array[array_index] = 0
            self.differentiation_matrix[array_index, :] = 0
            self.temperature[array_index] = temperature
        else:
            raise ValueError("mesh must be finite_volume or finite_difference")

    def set_neumann_boundary(self, side, flux=0):
        """Update boundary array and D2 for a neumann boundary."""
        if side == "left":
            array_index = 0
            next_col_index = 1
        elif side == "right":
            array_index = -1
            next_col_index = -2
        else:
            raise ValueError("Side must input must be left or right")

        if self.mesh_type == "finite_volume":
            self.boundary_condition_array[array_index] = flux / self.delta_x
            self.differentiation_matrix[array_index, array_index] = -1
        elif self.mesh_type == "finite_difference":
            self.boundary_condition_array[array_index] = 2 * flux * self.delta_x

            self.differentiation_matrix[array_index, next_col_index] = 2
        else:
            raise ValueError(
                "mesh_type unsupported, please input a finite_volume or finite_difference as mesh type"
            )


class linear_convection__mesh(create_1Dmesh):
    """
    Create a linear convection diffusion mesh.

    Attributes:
    n_cells (int): The number of cells used to discritze the domain
    mesh_type (str): finite_volume or finite_difference
    """

    def __init__(
        self,
        x,
        n_cells: int,
        mesh_type: str = "finite_volume",
        convection_coefficient=1,
        discretization_type: str = "upwind",
    ):
        """
        Initialize a lienar convection mesh object.

        Parameters:
           x : the spatial discritization of the domain
           n_cells (int): The number of cells to discritize the domain into
           mesh_type (string) : finite_voluem (default) or finite_difference
        Attributes:
           phi: the quantity of interest being transported
           convection_coefficent: a constant convection coefficent
        """
        super().__init__(x, n_cells, mesh_type)

        self.discretization_type = discretization_type
        if self.discretization_type == "upwind":
            self.create_upwind_differentiation_matrix(self.xcell_center)
        elif self.discretization_type == "central":
            self.create_central_differentiation_matrix(self.xcell_center)
        else:
            raise ValueError("discritization type not supported")

        if convection_coefficient <= 0:
            raise ValueError("only positive convection coefficents are supported")
        self.convection_coefficent = convection_coefficient
        self.phi = np.zeros(n_cells)

    def set_phi(self, phi):
        """
        Set the value of phi for internal nodes.

        Parameters:
        phi (int, float, list):list of phi at every x value
        """
        if isinstance(phi, (float, int)):
            self.phi = phi * np.ones(self.n_cells)
        elif isinstance(phi, list):
            if np.array(phi).shape != self.xcell_center.shape:
                raise ValueError("the shape of phi must match the xcell_center shape")
            self.phi = np.array(phi)
        else:
            raise TypeError("The phi type inputed not supported")

    def create_upwind_differentiation_matrix(self, nodes):
        """Create a differentiation matrix."""
        shape = np.shape(nodes)[0]
        lower = np.diagflat(np.repeat(1, shape - 1), -1)
        middle = -1 * np.identity(shape)
        differentiation_matrix = lower + middle
        self.differentiation_matrix = differentiation_matrix

    def create_central_differentiation_matrix(self, nodes):
        shape = np.shape(nodes)[0]
        upper = np.diagflat(np.repeat(-0.5, shape - 1), 1)
        differentiation_matrix = upper + np.transpose(-upper)
        self.differentiation_matrix = differentiation_matrix

    def set_dirichlet_boundary(self, side: str, phi: float):
        """Update boundary array and D2 for a dirichlet boundary."""
        if side == "left":
            array_index = 0

        else:
            raise ValueError("Only left side implemented")

        if self.mesh_type == "finite_volume":
            self.boundary_condition_array[array_index] = phi

        elif self.mesh_type == "finite_difference":
            self.boundary_condition_array[array_index] = 0
            self.differentiation_matrix[array_index, :] = 0
            self.phi[array_index] = phi
        else:
            raise ValueError("mesh must be finite_volume or finite_difference")

    def set_right_boundary(self):
        """
        Set the differentiation matrix using a 2nd order left discritization.

        For finite volume only
        """
        self.differentiation_matrix[-1, -3:] = [1.5, -1, 0.5]


def main():
    pass


def init():
    if __name__ == "__main__":
        main()


init()
