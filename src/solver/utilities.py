import numpy as np
import logging
from solver.mesher import side_selector
from matplotlib import pyplot as plt

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
        self.generation = self.set_generation()
        self.x_flux = self.set_x_flux()
        self.y_flux = self.set_y_flux()
        self.cell_flux = self.set_cell_flux()

    def set_generation(self):
        generation = self.mesh.generation.reshape(self.phi.shape) * (
            self.y_width * self.x_width
        )
        return generation

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

    def set_cell_flux(self):
        """Determine the heat flux for each cell."""
        x_identity = np.identity(self.x_cells)
        y_identity = np.identity(self.y_cells)

        d2y = np.kron(self.mesh.d2y_unscaled, x_identity)

        cell_y_flux = (
            d2y @ self.phi.flatten() + self.mesh.y_bc_reshape.flatten()
        ).reshape(self.phi.shape) * (
            self.mesh.conductivity * self.x_width / self.y_width
        )

        d2x = np.kron(y_identity, self.mesh.d2x_unscaled)
        cell_x_flux = (
            d2x @ self.phi.flatten() + self.mesh.x_bc_reshape.flatten()
        ).reshape(self.phi.shape) * (
            self.mesh.conductivity * self.y_width / self.x_width
        )

        return cell_x_flux + cell_y_flux + self.generation

    # cell_flux_se = np.sum((cell_flux**2))

    def flux(self, side: str):
        if side == "left":
            return np.sum(self.x_flux[:, 0])
        if side == "right":
            return np.sum(self.x_flux[:, -1])
        if side == "top":
            return np.sum(self.y_flux[0, :])
        if side == "bottom":
            return np.sum(self.y_flux[-1, :])
        if side == "cells":
            return np.sum((self.cell_flux**2))
        if side == "generation":
            return np.sum(self.generation)
        if side == "all":
            total_flux = np.sum(self.y_flux + self.x_flux + self.generation)
            logger.info(
                f"\n Left Flux: {np.sum(self.x_flux[:,0])} W  \
                \n Right Flux: {np.sum(self.x_flux[:,-1])} W  \
                \n Bottom Flux: {np.sum(self.y_flux[-1,:])} W  \
                \n Generation: {np.sum(self.generation)} W \
                \n Top Flux: {np.sum(self.y_flux[0,:])} W  \
                \n Cell Sum Squared Error: {np.sum((self.cell_flux**2))} W \
                \n Flux leaving boundarys: {np.sum(self.y_flux+ self.x_flux)}W \
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


class Parser:
    def parse(self, string: str):
        x = string.split("_")
        return x[0]


class Plotter:
    """A class to plot a 2d Cartesian mesh at various timepoints"""

    def __init__(self, mesh):
        self.mesh = mesh

    def transient_plotter(self, data_list, name, phi_min=10, phi_max=30):
        """plot 4 seperate transient temperature profiles

        args
        data_list: A list of dictionarys that have the time point as the key and phi values as the values
        mesh: the mesh
        """
        fig, axes = plt.subplots(4, figsize=(10, 12))
        fig2, axes2 = plt.subplots(4, figsize=(10, 12))
        list_length = len(data_list)
        for row, ax, ax2 in zip(
            [0, round(list_length / 4), round(3 * list_length / 4), list_length - 1],
            axes.flat,
            axes2.flat,
        ):
            # for time, data in data_list[row].items():
            time = data_list[row]["time"]
            data = data_list[row]["phi"]
            x_cords = self.mesh.grid["x_grid"].cell_cordinates
            y_cords = self.mesh.grid["y_grid"].cell_cordinates
            logger.debug(row)
            xv, yv = np.meshgrid(x_cords, y_cords)

            tmin = phi_min
            tmax = phi_max
            y_cells = data.shape[0]

            # Plot the temperature profiles for each time step
            im = ax.pcolormesh(xv, yv, data, vmax=tmax, vmin=tmin)
            ax.set_title(f"Temperature Profile at Time {time} s")

            # Plot the the temperature distributions for various horizontal lines
            group = [1, 2, 3]
            midline = round(y_cells / 2)
            top_quarter_line = round(y_cells / 4)
            bottom_quarter_line = round(3 * y_cells / 4)

            line_list = [midline, top_quarter_line, bottom_quarter_line]
            color_group = {1: "red", 2: "blue", 3: "green"}
            color_group_name = {1: "midline", 2: "top", 3: "bottom"}

            for g in group:
                line_row = line_list[g - 1]
                ax.axhline(y=(1 - (line_row / y_cells)), color=color_group[g])
                ax2.plot(
                    x_cords,
                    data[line_row, :],
                    color=color_group[g],
                    label=color_group_name[g],
                )

                ax2.axis((0, 5, tmin, tmax + 1))
                ax2.set_ylabel("temperature (celcius)")
                ax2.set_title(f"midpoint temperature at time {time} s")
                ax2.legend()

                fig.subplots_adjust(bottom=0.1, top=0.9, hspace=0.4)
                fig.colorbar(
                    im, ax=axes.ravel().tolist(), label="temperature (celcius)"
                )

                fig2.subplots_adjust(bottom=0.1, top=0.9, hspace=0.5)
                fig.savefig(f"{name}_temperature_profile.png")
                fig2.savefig(f"{name}_line_distributions.png")
