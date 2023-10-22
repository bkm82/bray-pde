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

        self.differentiation_matrix = create_differentiation_matrix(self.xcell_center)
        self.boundary_condition_array = np.zeros(n_cells)

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

    def set_dirichlet_boundary(self, side, temperature):
        """Update boundary array and D2 for a dirichlet boundary."""
        if side == "left":
            array_index = 0
        elif side == "right":
            array_index = -1
        else:
            raise ValueError("Side must input must be left or right")

        self.boundary_condition_array[array_index] = 2 * temperature
        self.differentiation_matrix[array_index, array_index] = -3

    def set_neumann_boundary(self, side, flux=0):
        """Update boundary array and D2 for a neumann boundary."""
        if side == "left":
            array_index = 0
        elif side == "right":
            array_index = -1
        # else:
        #     raise ValueError("Side must input must be left or right")

        self.boundary_condition_array[array_index] = flux / self.delta_x
        self.differentiation_matrix[array_index, array_index] = -1


def create_differentiation_matrix(nodes):
    """Create a differentiation matrix."""
    shape = np.shape(nodes)[0]
    upper = np.diagflat(np.repeat(1, shape - 1), 1)
    middle = -2 * np.identity(shape)
    differentiation_matrix = upper + np.transpose(upper) + middle
    return differentiation_matrix


# def main():


# if __name__ == "__main__":
#     main()
