import numpy as np
import pandas as pd


class main_solver:
    def __init__(self, mesh, initial_time=0, time_step_size=1, method="explicit"):
        self.initial_time = initial_time
        self.time_step_size = time_step_size
        self.method = method
        self.mesh = mesh
        self.saved_state_list = []
        self.current_time = initial_time

    def solver_take_step(self, k, atribute):
        differentiation_matrix = self.mesh.differentiation_matrix
        identity_matrix = np.identity(self.mesh.n_cells)
        # identity_matrix =
        if self.method == "explicit":
            # solve the form y = ax + b
            a = k * differentiation_matrix + identity_matrix
            b = k * self.mesh.boundary_condition_array

            return a @ atribute + b

        if self.method == "implicit":
            # solve the form ay = x+b where x= current temp, y= new temp
            a = -k * differentiation_matrix + identity_matrix
            b = k * self.mesh.boundary_condition_array
            return np.linalg.solve(a, (atribute + b))

    def save_state(self, *args, **kwargs):
        """
        Save the object atributes specified.

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

        self.saved_state_list.append(
            pd.DataFrame(
                saved_state_dictionary,
            )
        )

    def update_save_dictionary(self, **kwargs):
        self.save_dictionary = {
            "method": self.method,
            "time_step_size": self.time_step_size,
            "time": self.current_time,
            "x_cordinates": self.mesh.xcell_center,
        }

        for key, value in kwargs.items():
            self.save_dictionary[key] = value


class solver_1d(main_solver):
    def __init__(self, mesh, initial_time=0, time_step_size=1, method="explicit"):
        super().__init__(mesh, initial_time, time_step_size, method)

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

        self.mesh.temperature = self.solver_take_step(k, self.mesh.temperature)

    def solve(self, t_final, t_initial=0):
        """
        Run the solver for unitil the final time is reached.

        Inputs:
        t_initial = the initial time (default 0)
        t_final = the final time
        """

        self.current_time = t_initial
        self.update_save_dictionary(temperature=self.mesh.temperature)

        self.save_state(**self.save_dictionary)
        while self.current_time < t_final:
            self.take_step()
            self.current_time = self.current_time + self.time_step_size
            self.update_save_dictionary(temperature=self.mesh.temperature)
            self.save_state(**self.save_dictionary)

        # # Save the data into a single data frame for ploting
        self.saved_data = pd.concat(self.saved_state_list)


class linear_convection_solver(main_solver):
    def __init__(self, mesh, initial_time=0, time_step_size=1, method="explicit"):
        super().__init__(mesh, initial_time, time_step_size, method)

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
            self.mesh.phi = self.solver_take_step(k, self.mesh.phi)

    def maccormack_take_step(self, k, atribute):
        differentiation_matrix = self.mesh.differentiation_matrix
        identity_matrix = np.identity(self.mesh.n_cells)
        predictor_matrix = self.mesh.predictor_differentiation_matrix
        # self.predictor = (identity_matrix - predictor_matrix)@atribute
        if self.method == "explicit":
            self.predictor = (identity_matrix - k * predictor_matrix) @ atribute

            return 0.5 * (
                atribute
                + ((identity_matrix - k * differentiation_matrix) @ self.predictor)
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
        discritization_type = self.mesh.discretization_type

        # if discritization_type == "maccormack":
        #     self.update_save_dictionary(
        #         phi=self.mesh.phi,
        #         courant=self.courant_coefficent,
        #         discritization_type = "maccormack")

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
