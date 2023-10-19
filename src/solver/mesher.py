import numpy as np


class create_1Dmesh:
    """
    A class representing a 1D Mesh.

    Atributes:
    node (np.array): An array of node x positions
    n_points (int): The number of mesh points
    delta_x (float) : The distance between mesh points

    """

    def __init__(self, x, n_points):
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
        self.n_points = n_points
        self.node = np.linspace(x[0], x[1], n_points)
        self.delta_x = (x[1] - x[0]) / (n_points - 1)


def main():
    pass


if __name__ == "__main__":
    main()
