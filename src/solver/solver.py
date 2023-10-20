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


# def main():


# if __name__ == "__main__":
#     main()
