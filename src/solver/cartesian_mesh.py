from solver.mesher import grid, differentiation_matrix
from typing import Sequence, Tuple, List


class cartesian_mesh:
    """
    A cartesian mesh up to 2D.

    Atributes:
    {dim}_grid (dim = x , y)
    """

    def __init__(
        self,
        n_cells: List[int] = [4, 4],
        cordinates: Sequence[Tuple[float, float]] = [(0, 1), (0, 1)],
        dimensions: int = 2,
        mesh_type: str = "finite_volume",
    ) -> None:
        """
        Init the cartesian mesh object.

        Args:
            dimensions:int = Mesh dimensionality (default 2d)
            cordinates (list of tupples):list of each dimensions cordinates
            mesh_type: str = finite_volume (default) or finite_difference

        """
        # Validate input
        implemented_dimensions = [1, 2]
        if dimensions not in implemented_dimensions:
            raise ValueError("mesh dimesnionality not implemented")

        # Validate cordinates were given for each dimension
        if len(cordinates) != dimensions:
            raise ValueError("number of cordinates needs to match dimesnions")
        if len(n_cells) != dimensions:
            raise ValueError("each dimension needs number of cells")

        # # Discritize each dimmension
        grid_list = ["x_grid", "y_grid"]
        for index, grid_name in enumerate(grid_list[:dimensions]):
            setattr(
                self,
                grid_name,
                grid(
                    n_cells=n_cells[index],
                    cordinates=cordinates[index],
                    mesh_type=mesh_type,
                ),
            )

        matrix_list = ["x_differentiation_matrix", "y_differentiation_matrix"]

        for index, matrix_name in enumerate(matrix_list[:dimensions]):
            setattr(
                self,
                matrix_name,
                differentiation_matrix(
                    n_cells=n_cells[index],
                ),
            )
        # self.x_differentiation_matrix = differentiation_matrix(n_cells[0])
        # x_differentiation_matrix
