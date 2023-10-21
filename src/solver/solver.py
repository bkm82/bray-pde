import numpy as np


class solver_1d:
    def __init__(
        self, mesh, n_time_steps, initial_time=0, time_step_size=1, method="explicit"
    ):
        self.initial_time = initial_time
        self.time_step_size = time_step_size
        self.n_time_steps = n_time_steps
        self.method = method
        self.mesh = mesh

    def take_step(self, delta_t):
        """Set the temperature to the next timestep."""
        k = self.mesh.thermal_diffusivity * delta_t / (self.mesh.delta_x**2)
        identity_matrix = np.identity(self.mesh.n_cells)
        current_temperature = self.mesh.temperature
        if self.method == "explicit":
            # solve the form y = ax + b
            a = (k * self.mesh.differentiation_matrix) + identity_matrix
            b = k * self.mesh.boundary_condition_array
            self.mesh.temperature = a @ current_temperature + b

        if self.method == "implicit":
            # solve the form ay = bx where x = current temp, y= new temp

            a = identity_matrix - (k * self.mesh.differentiation_matrix)
            b = self.mesh.temperature + (k * self.mesh.boundary_condition_array)

            self.mesh.temperature = np.linalg.solve(a, b)


# def main():


# if __name__ == "__main__":
#     main()
