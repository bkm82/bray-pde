import numpy as np
import pandas as pd
from typing import List
from solver.cartesian_mesh import CartesianMesh

# from solver.utilities import MeshReshaper


class ImplicitStep(object):
    """An object to take an implicit step"""

    def step(self, k, laplacian, boundary_condition_array, phi):
        """Solve the form ay = x + b.

        y = returned
        a = [k*laplacian + I]
        x = phi
        b = k + boundary_condition_array
        """
        # solve the form ay = x+b where x= current temp, y= new temp
        identity_matrix = np.identity(laplacian.shape[0])
        a = -k * laplacian + identity_matrix
        b = k * boundary_condition_array
        return np.linalg.solve(a, (phi + b))


class ExplicitStep(object):
    """An object to take an explicit step."""

    def step(self, k, laplacian, boundary_condition_array, phi):
        """Solve the form y = ax + b.

        y = returned
        a = [k*laplacian + I]
        x = phi
        b = k + boundary_condition_array
        """
        identity_matrix = np.identity(laplacian.shape[0])
        a = k * laplacian + identity_matrix
        b = k * boundary_condition_array
        return a @ phi + b


class SteadySolver(object):
    def solve(self, laplacian, boundary_condition_array):
        return np.linalg.solve(laplacian, -boundary_condition_array)


class Stepper(object):
    """A stepper object to determine the step type to take."""

    def __init__(self, ExplicitStep=ExplicitStep(), ImplicitStep=ImplicitStep()):
        self.explicit_step = ExplicitStep
        self.implicit_step = ImplicitStep

    def take_step(self, method, **kwags):
        """Take the appropriate step depending on the method."""
        if method == "explicit":
            return self.explicit_step.step(**kwags)
        if method == "implicit":
            return self.implicit_step.step(**kwags)


class Saver(object):
    """An object that saves the state of the solution to a data frame."""

    def __init__(self):
        self.saved_state_list = []

    def save_state(self, *args, **kwargs):
        """
        Save the object atributes specified by appending it to the saved_state_list.

        Inputs
        *args must be atributes of the object

        Outputs:
        Appends to a self.saved_state_list
        """
        saved_state_dictionary = {}
        for arg in args:
            saved_state_dictionary[arg] = getattr(self, arg)
        for key, value in kwargs.items():
            saved_state_dictionary[key] = value
        self.saved_state_list.append(pd.DataFrame(saved_state_dictionary))

    def update_save_dictionary(self, **kwargs):
        self.save_dictionary = {
            "method": self.method,
            "time_step_size": self.time_step_size,
            "time": self.current_time,
        }
        for key, value in kwargs.items():
            self.save_dictionary[key] = value


class Solver(Saver):
    """main solver class."""

    def __init__(
        self,
        mesh,
        initial_time=0,
        time_step_size=1,
        method="explicit",
        stepper=Stepper(),
        steady_solver=SteadySolver(),
    ):
        """Initiate the main solver:

        args:
        mesh: mesh object (reqires laplacian and boundary_condition_array atributes)
        initial_time:
        """
        self.initial_time = initial_time
        self.step_size = time_step_size
        self.method = method

        self.saved_state_list = []
        self.current_time = initial_time
        self.mesh = mesh
        self.laplacian = self.mesh.laplacian
        self.boundary_condition_array = self.mesh.boundary_condition_array
        self.stepper = stepper
        self.steady_solver = steady_solver

    def take_step(self, k, atribute):
        return self.stepper.take_step(
            method=self.method,
            k=k,
            laplacian=self.laplacian,
            boundary_condition_array=self.boundary_condition_array,
            phi=atribute,
        )

    def solve_steady(self):
        """Call the steady solver"""
        phi_shape = self.mesh.phi.get_phi().shape

        solved_phi = self.steady_solver.solve(
            laplacian=self.laplacian,
            boundary_condition_array=self.boundary_condition_array,
        )
        phi_reshape = np.reshape(solved_phi, phi_shape)
        self.mesh.phi.set_phi(phi_reshape.tolist())

    def solve(self, t_final, t_initial=0):
        """
        Run the solver for unitil the final time is reached.

        Inputs:
        t_initial = the initial time (default 0)
        t_final = the final time
        """
        self.current_time = t_initial
        while self.current_time < t_final:
            self.take_step()
            self.current_time = self.current_time + self.time_step_size


class solver_1d(Solver):
    def __init__(self, mesh, initial_time=0, time_step_size=1, method="explicit"):
        super().__init__(mesh, initial_time, time_step_size, method)
        self.time_step_size = self.step_size

    def take_step(self):
        """
        Take a single step forward in temprature.

        Inputs:
        delta_t: the time step size to take
        """
        k = (
            self.mesh.thermal_diffusivity
            * self.time_step_size
            / (self.mesh.delta_x**2)
        )
        self.mesh.temperature = super().take_step(k, self.mesh.temperature)

    def solve(self, t_final, t_initial=0):
        """
        Run the solver for unitil the final time is reached.

        Inputs:
        t_initial = the initial time (default 0)
        t_final = the final time
        """

        self.current_time = t_initial
        super().update_save_dictionary(
            x_cordinates=self.mesh.xcell_center, temperature=self.mesh.temperature
        )
        super().save_state(**self.save_dictionary)
        while self.current_time < t_final:
            self.take_step()
            self.current_time = self.current_time + self.time_step_size
            self.update_save_dictionary(
                temperature=self.mesh.temperature, x_cordinates=self.mesh.xcell_center
            )
            super().save_state(**self.save_dictionary)

        # # Save the data into a single data frame for ploting
        self.saved_data = pd.concat(self.saved_state_list)


class linear_convection_solver(Solver):
    def __init__(self, mesh, initial_time=0, time_step_size=1, method="explicit"):
        super().__init__(mesh, initial_time, time_step_size, method)
        self.predictor_laplacian = mesh.preictor_laplacian
        self.time_step_size = self.step_size

    def take_step(self):
        """
        Take a single step forward in time.

        Inputs:
        delta_t: the time step size to take
        """
        k = self.mesh.convection_coefficent * self.time_step_size / (self.mesh.delta_x)
        self.courant_coefficent = k
        if self.mesh.discretization_type == "maccormack":
            self.mesh.phi = self.maccormack_take_step(k, self.mesh.phi)
        else:
            self.mesh.phi = super().take_step(k, self.mesh.phi)

    def maccormack_take_step(self, k, atribute):
        laplacian = self.mesh.laplacian
        identity_matrix = np.identity(self.mesh.n_cells)
        predictor_matrix = self.mesh.predictor_differentiation_matrix
        # self.predictor = (identity_matrix - predictor_matrix)@atribute
        if self.method == "explicit":
            self.predictor = (identity_matrix - k * predictor_matrix) @ atribute

            return 0.5 * (
                atribute + ((identity_matrix - k * laplacian) @ self.predictor)
            )

        elif self.method == "implicit":
            raise Exception("implicit not implemented for maccormack")
        else:
            raise Exception("implicit or explicit method needed")

    def solve(self, t_final, t_initial=0):
        """
        Run the solver for unitil the final time is reached.

        Inputs:
        t_initial = the initial time (default 0)
        t_final = the final time
        """

        self.current_time = t_initial
        self.courant_coefficent = (
            self.mesh.convection_coefficent * self.time_step_size / (self.mesh.delta_x)
        )

        self.update_save_dictionary(
            phi=self.mesh.phi,
            courant=self.courant_coefficent,
            discritization=self.mesh.discretization_type,
        )

        self.save_state(**self.save_dictionary)
        while self.current_time < t_final:
            self.take_step()
            self.current_time = self.current_time + self.time_step_size

            self.update_save_dictionary(
                phi=self.mesh.phi,
                courant=self.courant_coefficent,
                discritization=self.mesh.discretization_type,
            )
            self.save_state(**self.save_dictionary)

        # # Save the data into a single data frame for ploting
        self.saved_data = pd.concat(self.saved_state_list)


def main():
    pass


def init():
    if __name__ == "__main__":
        main()


# init()
