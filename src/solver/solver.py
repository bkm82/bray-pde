import numpy as np
import pandas as pd


class solver_1d:
    def __init__(
        self, mesh, n_time_steps, initial_time=0, time_step_size=1, method="explicit"
    ):
        self.initial_time = initial_time
        self.time_step_size = time_step_size
        self.n_time_steps = n_time_steps
        self.method = method
        self.mesh = mesh
        self.saved_state_list = []

    def take_step(self, delta_t):
        """Set the temperature to the next timestep."""
        k = self.mesh.thermal_diffusivity * delta_t / (self.mesh.delta_x**2)
        identity_matrix = np.identity(self.mesh.n_cells)
        current_temperature = self.mesh.temperature
        differentiation_matrix = k * self.mesh.differentiation_matrix

        if self.method == "explicit":
            # solve the form y = ax + b
            a = differentiation_matrix + identity_matrix
            b = k * self.mesh.boundary_condition_array
            self.mesh.temperature = a @ current_temperature + b

        if self.method == "implicit":
            # solve the form ay = x+b where x= current temp, y= new temp
            a = -differentiation_matrix + identity_matrix
            b = k * self.mesh.boundary_condition_array
            self.mesh.temperature = np.linalg.solve(a, (current_temperature + b))

    def solve(self, t_final, delta_t=1, t_initial=0):
        """
        Run the solver for unitil the final time is reached.

        Inputs:
        t_initial = the initial time (default 0)
        t_final = the final time
        delta_t = the desired time step
        """
        current_time = t_initial
        # save the inital state
        self.save_state(
            "method",
            time=current_time,
            x_cordinates=self.mesh.xcell_center,
            temperature=self.mesh.temperature,
        )
        # Loop through time steps saving
        while current_time < t_final:
            self.take_step(delta_t)
            current_time = current_time + delta_t
            self.save_state(
                "method",
                time=current_time,
                x_cordinates=self.mesh.xcell_center,
                temperature=self.mesh.temperature,
            )

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


# def main():


# if __name__ == "__main__":
#     main()
