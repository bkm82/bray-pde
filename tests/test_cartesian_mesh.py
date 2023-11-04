import pytest
from solver.cartesian_mesh import cartesian_mesh


class Test_cartesian_mesh:
    @pytest.fixture
    def cartesian_mesh(self):
        return cartesian_mesh()


class Test_cartesian_mesh_exceptions:
    """Test features expected to raise an exception"""

    exception_inputs = [
        ({"dimensions": 3}),
        ({"dimensions": 2, "cordinates": [(0, 1)]}),
        ({"dimensions": 2, "n_cells": [5]}),
    ]

    @pytest.mark.parametrize("inputs", exception_inputs)
    def test_three_dimensions_raises(self, inputs):
        with pytest.raises(ValueError):
            cartesian_mesh(**inputs)
