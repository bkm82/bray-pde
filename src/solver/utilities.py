import numpy as np
import logging
from solver.mesher import side_selector

# create logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Set the formatter for the console handler
formatter = logging.Formatter(
    "%(name)s:%(levelname)s:%(funcName)s:%(message)s",
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
        self.x_flux = self.set_x_flux()
        self.y_flux = self.set_y_flux()

    def set_x_flux(self):
        x_flux_diff_matrix = np.zeros((self.x_cells, self.x_cells))
        y_identity = np.identity(self.y_cells)

        for side in ["left", "right"]:
            boundary_index = self.side_selector.boundary_index(side)
            differentiation_value = self.differntiation_value(side)
            x_flux_diff_matrix[boundary_index, boundary_index] = differentiation_value

        d2_x = np.kron(y_identity, x_flux_diff_matrix)

        self.bray_x_flux = (
            ((d2_x @ self.phi.flatten()) + self.mesh.x_bc_reshape.flatten())
            * (self.mesh.conductivity * self.y_width / self.x_width)
        ).reshape(self.phi.shape)
        return (
            ((d2_x @ self.phi.flatten()) + self.mesh.x_bc_reshape.flatten())
            * (self.mesh.conductivity * self.y_width / self.x_width)
        ).reshape(self.phi.shape)

    def set_y_flux(self):
        y_flux_diff_matrix = np.zeros((self.y_cells, self.y_cells))
        x_identity = np.identity(self.x_cells)

        for side in ["top", "bottom"]:
            boundary_index = self.side_selector.boundary_index(side)
            differentiation_value = self.differntiation_value(side)
            y_flux_diff_matrix[boundary_index, boundary_index] = differentiation_value

        d2_y = np.kron(y_flux_diff_matrix, x_identity)

        return (
            ((d2_y @ self.phi.flatten()) + self.mesh.y_bc_reshape.flatten())
            * (self.mesh.conductivity * self.x_width / self.y_width)
        ).reshape(self.phi.shape)

    def flux(self, side: str):
        if side == "left":
            return np.sum(self.x_flux[:, 0])
        if side == "right":
            return np.sum(self.x_flux[:, -1])
        if side == "top":
            return np.sum(self.y_flux[0, :])
        if side == "bottom":
            return np.sum(self.y_flux[-1, :])
        if side == "all":
            total_flux = np.sum(self.y_flux + self.x_flux)
            logger.info(
                f"\n Left Flux: {np.sum(self.x_flux[:,0])} W  \
                \n Right Flux: {np.sum(self.x_flux[:,-1])} W  \
                \n Bottom Flux: {np.sum(self.x_flux[-1,:])} W  \
                \n Top Flux: {np.sum(self.y_flux[0,:])} W  \
                \n Total Flux: {total_flux}"
            )

            return total_flux

    def differntiation_value(self, side: str):
        if self.mesh.boundary_condition_dict[side] == "dirichlet":
            return -2
        elif self.mesh.boundary_condition_dict[side] == "neumann":
            return 0
        else:
            recieved = self.mesh.boundary_condition_dict[side]
            raise ValueError(f"Boundary conditions needed, got {recieved}")
