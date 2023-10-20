import numpy as np


class create_1Dmesh:
    """
    A class representing a 1D Mesh.

    Atributes:
    node (np.array): An array of node x positions
    n_points (int): The number of mesh points
    delta_x (float) : The distance between mesh points

    """

    def __init__(self, x, n_cells):
        """
        Initialize the Mesh object.

        Keyword Arguments:
        x -- the spatial domain of the mesh in the form [x_min, x_max]
        n_points -- the number of points for discritization

        Example
        mesh = create_1Dmesh(x=[0, 1], n_points=3)
        mesh.node = np.array([0, 0.5, 1])
        mesh.deltax = 0.5
        """
        self.n_cells = n_cells

        self.delta_x = (x[1] - x[0]) / (n_cells)
        self.xcell_center = np.linspace(
            x[0] + (self.delta_x / 2), x[1] - self.delta_x / 2, n_cells
        )
        self.differentiation_matrix = create_differentiation_matrix(self.xcell_center)

    def set_cell_temperature(self, temperature):
        """
        Set the temperature for internal nodes.

        Example:running mesh.set_internal_temperature(20) would result
        in np.array([20, 20, 20, 20]
        """
        self.temperature = temperature * np.ones(self.n_cells)


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
