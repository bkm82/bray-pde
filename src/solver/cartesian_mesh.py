from solver.mesher import (
    grid,
    differentiation_matrix,
    boundary_condition,
    side_selector,
    cell_phi,
)
from typing import Sequence, Tuple, List
import numpy as np


class CartesianMesh:
    """
    A cartesian mesh up to 2D.

    Atributes:
    dimensions: int =  Number of dimensions (1d or 2d)
    n_cells:List[int]: Number of cells to discritize each dimension
    cordinates:Sequence[Tuple[float, float]] = The bounds each dimension
    mesh_type: str = The mesh type. currently only finite_volume.

    Example:
    To create a 2d mesh with
        3 cells in the x axis going from 0 to 1
        4 cells in the y axis going from 0 to 2
        cartesian_mesh(
            dimensions = 2
            n_cells = [3, 4]
            cordinates = [(0,1), (0,2)]
    """

    def __init__(
        self,
        dimensions: int = 2,
        n_cells: List[int] = [4, 4],
        cordinates: Sequence[Tuple[float, float]] = [(0, 1), (0, 1)],
        mesh_type: str = "finite_volume",
    ) -> None:
        """
        Init the cartesian mesh object.

        Args:
            dimensions:int = Mesh dimensionality (default 2d)
            cordinates (list of tupples):list of each dimensions cordinates
            mesh_type: str = finite_volume (default) or finite_difference

        """
        self.validate_inputs(n_cells, cordinates, dimensions, mesh_type)
        self.n_cells = n_cells
        self.dimensions = dimensions
        self.cordinates = cordinates
        self.mesh_type = mesh_type
        self.grid = self.initalize_grid()
        self.differentiation_matrix = self.initalize_differentiation_matrix()
        self.initalize_phi()
        self.boundary_condition = self.initalize_boundary_condition()
        self.set_laplacian()

    def validate_inputs(
        self,
        n_cells: List[int] = [4, 4],
        cordinates: Sequence[Tuple[float, float]] = [(0, 1), (0, 1)],
        dimensions: int = 2,
        mesh_type: str = "finite_volume",
    ) -> None:
        """
        Validate the cartesian mesh inputs.

        Args:
            dimensions:int = Mesh dimensionality (default 2d)
            cordinates (list of tupples):list of each dimensions cordinates
            mesh_type: str = finite_volume (default) or finite_difference

        """
        # Implemented dimensions and mesh types
        self.implemented_dimensions: Tuple[str, str] = ("x", "y")
        self.implemented_mesh_types: str = "finite_volume"

        if dimensions > len(self.implemented_dimensions):
            raise ValueError("mesh dimesnionality not implemented")

        # Validate cordinates were given for each dimension
        if len(cordinates) != dimensions:
            raise ValueError("number of cordinates needs to match dimesnions")
        if len(n_cells) != dimensions:
            raise ValueError("legth of n_cells list needs to match dimension")

        if mesh_type not in self.implemented_mesh_types:
            raise ValueError(f"{mesh_type}: is not an implemented mesh")

    def initalize_grid(self):
        grid_dict = {}
        for index in range(0, self.dimensions):
            grid_dict[f"{self.implemented_dimensions[index]}_grid"] = grid(
                n_cells=self.n_cells[index],
                cordinates=self.cordinates[index],
                mesh_type=self.mesh_type,
            )
            # flip the y cordinates so the origin is in the top left corner
            if index == 1:
                grid_dict["y_grid"].cell_cordinates = np.flip(
                    grid_dict["y_grid"].cell_cordinates
                )
        return grid_dict

    def initalize_differentiation_matrix(self):
        """
        Create a differentiation matrix for each dimension.

        returns: a dictionary of differentiation matrix
        """
        differentiation_matrix_dict = {}
        for index in range(0, self.dimensions):
            differentiation_matrix_dict[
                f"{self.implemented_dimensions[index]}_differentiation_matrix"
            ] = differentiation_matrix(
                n_cells=self.n_cells[index],
            )

        return differentiation_matrix_dict

    def initalize_boundary_condition(self):
        """
        Create a boundary condition array for each dimension.

        returns: a dictionary of boundary conditions
        """
        boundary_condition_dict = {}
        for index in range(0, self.dimensions):
            boundary_condition_dict[
                f"{self.implemented_dimensions[index]}_boundary_condition_array"
            ] = boundary_condition(
                n_cells=self.n_cells[index], mesh_type=self.mesh_type
            )

        return boundary_condition_dict

    def initalize_phi(self):
        self.phi = cell_phi(self.n_cells, self.dimensions, self.mesh_type)

    def set_dirichlet_boundary(self, side: str, phi: float):
        """
        Set the dirichlet boundary.

        Updates differentiation matrix, and boundary condition array
        args:
        side: the side to set (left, right (1d) top, bottom (2d))
        phi: the value to set the boundary
        """
        axis = side_selector().axis(side)

        self.differentiation_matrix[
            f"{axis}_differentiation_matrix"
        ].set_dirichlet_boundary(side=side, mesh_type=self.mesh_type)

        self.boundary_condition[
            f"{axis}_boundary_condition_array"
        ].set_dirichlet_boundary(side, phi)

        self.set_laplacian()
        self.set_boundary_condition_array()
        self.phi.set_dirichlet_boundary(side, phi)

    def set_neumann_boundary(self, side: str, flux: float):
        """
        Set a neuman boundary.

        args:
        side: str = left, right (1d), top, bottom (2d)
        flux: float = flux into the boundary (negative if out)
        """
        axis = side_selector().axis(side)

        self.differentiation_matrix[
            f"{axis}_differentiation_matrix"
        ].set_neumann_boundary(side, self.mesh_type)

        self.boundary_condition[
            f"{axis}_boundary_condition_array"
        ].set_neumann_boundary(
            side=side, flux=flux, cell_width=self.grid[f"{axis}_grid"].cell_width
        )

        self.set_laplacian()
        self.set_boundary_condition_array()

    def set_laplacian(self):
        """Combine the differentiation matricies into a single matrix."""
        if self.dimensions == 1:
            # d2x = self.x_differentiation_matrix.get_matrix()
            self.laplacian = self.differentiation_matrix[
                "x_differentiation_matrix"
            ].get_matrix() * (1 / self.grid["x_grid"].cell_width ** 2)
        elif self.dimensions == 2:
            d2x = self.differentiation_matrix[
                "x_differentiation_matrix"
            ].get_matrix() * (1 / self.grid["x_grid"].cell_width ** 2)

            d2y = self.differentiation_matrix[
                "y_differentiation_matrix"
            ].get_matrix() * (1 / self.grid["y_grid"].cell_width ** 2)
            Ix = np.identity(self.grid["x_grid"].n_cells)
            Iy = np.identity(self.grid["y_grid"].n_cells)
            self.laplacian = np.kron(Iy, d2x) + np.kron(d2y, Ix)

    def set_boundary_condition_array(self):
        """Combine boundary conditions into a single array."""
        if self.dimensions == 1:
            self.boundary_condition_array = self.boundary_condition[
                "x_boundary_condition_array"
            ].get_array() * (1 / self.grid["x_grid"].cell_width ** 2)

        elif self.dimensions == 2:
            x_bc_array = self.boundary_condition[
                "x_boundary_condition_array"
            ].get_array()

            y_bc_array = self.boundary_condition[
                "y_boundary_condition_array"
            ].get_array()

            x_cells = self.grid["x_grid"].n_cells
            y_cells = self.grid["y_grid"].n_cells
            dx = self.grid["x_grid"].cell_width
            dy = self.grid["y_grid"].cell_width

            x_bc_reshape = x_bc_array.reshape(1, x_cells).repeat(y_cells, axis=0)
            y_bc_reshape = y_bc_array.reshape(y_cells, 1).repeat(x_cells, axis=1)

            self.boundary_condition_array = (
                (x_bc_reshape * (1 / dx**2)) + (y_bc_reshape * (1 / dy**2))
            ).reshape(x_cells * y_cells)
