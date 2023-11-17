import numpy as np
import logging
from solver.mesher import side_selector

logger = logging.getLogger("__utilities__")
logger.setLevel(logging.INFO)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Set the formatter for the console handler
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S%p",
)
console_handler.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)


class MeshReshaper:
    """Object to store a mesh shape and return a long or wide mesh"""

    def __init__(self, x_cells, y_cells):
        """Store the mesh dimensions"""
        self.x_cells = x_cells
        self.y_cells = y_cells

    def to_long(self, array):
        """Return a long array"""
        return np.reshape(array, self.x_cells * self.y_cells)

    def to_wide(self, array):
        """Return a wide array"""
        return np.reshape(array, (self.y_cells, self.x_cells))


class EnergyBalance:
    """Object to calculate the energy balance for a mesh"""

    def __init__(self, mesh, side_selector=side_selector()):
        self.mesh = mesh
        self.side_selector = side_selector
        self.x_cells = self.mesh.n_cells[0]
        self.y_cells = self.mesh.n_cells[1]
        self.x_width = self.mesh.grid["x_grid"].cell_width
        self.y_width = self.mesh.grid["y_grid"].cell_width
        self.phi = self.mesh.phi.get_phi()
        self.x_flux()

    def x_flux(self):
        x_flux_diff_matrix = np.zeros((self.x_cells, self.x_cells))
        y_identity = np.identity(self.y_cells)

        for side in ["left", "right"]:
            boundary_index = self.side_selector.boundary_index(side)
            differentiation_value = self.differntiation_value(side)

        d2_x = np.kron(y_identity, x_flux_diff_matrix)
        self.x_flux = (
            (d2_x @ self.phi.flatten()) + self.mesh.x_bc_reshape.flatten()
        ) * (self.mesh.conductivity * self.y_width / self.x_width)

    def flux(self, side: str):
        if side == "left":
            return np.sum(self.x_flux.reshape(self.phi.shape)[:, 0])
        if side == "right":
            return np.sum(self.x_flux.reshape(self.phi.shape)[:, -1])

    def differntiation_value(self, side: str):
        if self.mesh.boundary_condition_dict[side] == "dirichlet":
            return -2
        elif self.mesh.boundary_condition_dict[side] == "neuimann":
            return 0
        else:
            recieved = self.mesh.boundary_condition_dict[side]
            raise ValueError(f"Boundary conditions needed, got {recieved}")
