import numpy as np
import pandas as pd


class main_solver:
    def __init__(self, mesh, initial_time=0, time_step_size=1, method="explicit"):
        self.initial_time = initial_time
        self.time_step_size = time_step_size
        self.method = method
        self.mesh = mesh
        self.saved_state_list = []

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

    def update_save_dictionary(self):
        self.save_dictionary = {
            "method": self.method,
            "time_step_size": self.time_step_size,
            "time": self.current_time,
            "x_cordinates": self.mesh.xcell_center,
        }


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
        delta_t = the desired time step
        """

        self.current_time = t_initial

        self.update_save_dictionary()
        self.save_dictionary.update({"temperature": self.mesh.temperature})
        self.save_state(**self.save_dictionary)
        while self.current_time < t_final:
            self.take_step()
            self.current_time = self.current_time + self.time_step_size
            self.update_save_dictionary()
            self.save_dictionary.update({"temperature": self.mesh.temperature})
            self.save_state(**self.save_dictionary)

        # # Save the data into a single data frame for ploting
        self.saved_data = pd.concat(self.saved_state_list)


#     def update_save_dictionary(self):
#         self.save_dictionary = {
#             "method": self.method,
#             "time_step_size": self.time_step_size,
#             "time": self.current_time,
#             "x_cordinates": self.mesh.xcell_center,
# }
#         self.save_dictionary.update({"temperature": self.mesh.temperature})


def main():
    pass


def init():
    if __name__ == "__main__":
        main()


# init()
