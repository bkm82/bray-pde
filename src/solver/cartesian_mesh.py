from solver.mesher import (
    grid,
    differentiation_matrix,
    boundary_condition,
    side_selector,
    cell_phi,
)

from solver.utilities import Parser
from typing import Sequence, Tuple, List
import numpy as np
import logging

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

# from cartesian_solver import CartesianSolver


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
        conductivity: float = 1,
        diffusivity: float = 1,
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
        self.diffusivity = diffusivity
        self.grid = self.initalize_grid()
        self.differentiation_matrix = self.initalize_differentiation_matrix()
        self.initalize_phi()
        self.boundary_condition = self.initalize_boundary_condition()
        self.set_laplacian()
        self.boundary_condition_dict: dict[str, str] = {}
        self.conductivity = conductivity

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

        self.boundary_condition_dict[side] = "dirichlet"
        self.set_laplacian()
        self.set_boundary_condition_array()
        self.phi.set_dirichlet_boundary(side, phi)
        self.generation = np.zeros(self.n_cells).flatten()

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
        self.boundary_condition_dict[side] = "neumann"

        self.set_laplacian()
        self.set_boundary_condition_array()

    def set_laplacian(self):
        """Combine the differentiation matricies into a single matrix."""
        if self.dimensions == 1:
            # d2x = self.x_differentiation_matrix.get_matrix()
            self.laplacian = self.differentiation_matrix[
                "x_differentiation_matrix"
            ].get_matrix() * (self.diffusivity / self.grid["x_grid"].cell_width ** 2)
        elif self.dimensions == 2:
            self.d2x_unscaled = self.differentiation_matrix[
                "x_differentiation_matrix"
            ].get_matrix()
            self.d2y_unscaled = self.differentiation_matrix[
                "y_differentiation_matrix"
            ].get_matrix()
            d2x = self.d2x_unscaled * (
                self.diffusivity / self.grid["x_grid"].cell_width ** 2
            )

            d2y = self.d2y_unscaled * (
                self.diffusivity / self.grid["y_grid"].cell_width ** 2
            )
            Ix = np.identity(self.grid["x_grid"].n_cells)
            Iy = np.identity(self.grid["y_grid"].n_cells)
            self.laplacian = np.kron(Iy, d2x) + np.kron(d2y, Ix)

    def set_boundary_condition_array(self):
        """Combine boundary conditions into a single array."""
        if not hasattr(self, "generation"):
            self.generation = np.zeros(self.n_cells).flatten()
        if self.dimensions == 1:
            self.boundary_condition_array = self.boundary_condition[
                "x_boundary_condition_array"
            ].get_array() * (self.diffusivity / self.grid["x_grid"].cell_width ** 2)

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

            self.x_bc_reshape = x_bc_array.reshape(1, x_cells).repeat(y_cells, axis=0)
            self.y_bc_reshape = y_bc_array.reshape(y_cells, 1).repeat(x_cells, axis=1)
            logger.debug(f"generation in set bc {self.generation.shape}")
            square_boundary_condition = (
                (self.x_bc_reshape * (1 / dx**2))
                + (self.y_bc_reshape * (1 / dy**2))
            ) * self.diffusivity
            self.boundary_condition_array = square_boundary_condition.reshape(
                x_cells * y_cells
            ) + self.generation * (self.diffusivity / self.conductivity)

            logger.debug(
                f"combined boundary condition array{self.boundary_condition_array.shape}"
            )

    def set_generation(self, function):
        """
        set a generation function
        args:
        function: the generation function
            set_generation(lambda x : x**2) for 1d,
            set_generation(lambda x,y: 2x + y**2) for 2d

            note: the function must contain a variable for all active axies
            variables even if it is a zero i.e. (lambda x,y: 1x + 0*y)
            it can also be a named function:
            def my_function (x, y):
                return(2*x+3*y +5)
            set_generation(my_function)
        Behavior
            This function generates a dictionary for each grid axis in grid.items()
            using the slice function to evalueate the function in the grid space required
            (analagous to linspace). The function is then evaluated on the grid and flattened
            The set_boundary_condition_array is then called to ensure it is added to the boundary condition array
        Reference (https://stackoverflow.com/questions/22774726/numpy-evaluate-function-on-a-grid-of-points)
        """
        grid_dict = {}
        logger.debug(f"grid:{self.grid}")
        # Generate a dictionary for each axis as either a column or a row.
        # Analagous to meshgrid
        for i, (names, cordinate_value) in enumerate(self.grid.items()):
            grid_dict[(Parser().parse(names))] = cordinate_value.cell_cordinates[
                (None,) * i
                + (slice(None),)
                + (None,) * (len(self.grid.items()) - i - 1)
            ]

        logger.debug(f"grid_dict:{grid_dict}")
        self.generation = (function(**grid_dict)).flatten(order="F")
        logger.debug(f"generation:{function(**grid_dict)}")
        logger.debug(f"generation reshape:{self.generation}")
        logger.debug(f"boundary_condition_array:{self.boundary_condition_array}")
        self.set_boundary_condition_array()
