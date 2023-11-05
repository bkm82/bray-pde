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
        self.validate_inputs(n_cells, cordinates, dimensions, mesh_type)
        self.n_cells = n_cells
        self.dimensions = dimensions
        self.cordinates = cordinates
        self.mesh_type = mesh_type
        self.discritize()

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

    def discritize(self):
        """Discritize the mesh for each axis dimension."""
        grid_list = self.create_atribute_list("grid")
        matrix_list = self.create_atribute_list("differentiation_matrix")

        for index, (grid_name, matrix_name) in enumerate(
            zip(grid_list[: self.dimensions], matrix_list[: self.dimensions])
        ):
            setattr(
                self,
                grid_name,
                grid(
                    n_cells=self.n_cells[index],
                    cordinates=self.cordinates[index],
                    mesh_type=self.mesh_type,
                ),
            )

            setattr(
                self,
                matrix_name,
                differentiation_matrix(
                    n_cells=self.n_cells[index],
                ),
            )

    def create_atribute_list(self, atribute_name):
        return [f"{dim}_{atribute_name}" for dim in self.implemented_dimensions]
