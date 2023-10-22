import solver
import mesher
import logging
import matplotlib.pyplot as plt
import plotnine as pn
import pandas as pd

logging.basicConfig(level=logging.WARNING)


def create_mesh(n_cells):
    mesh = mesher.create_1Dmesh(x=[0, 1], n_cells=n_cells)
    mesh.set_cell_temperature(0)  # set initial conditions to 0 celcicus
    mesh.set_dirichlet_boundary("left", 50)  # Left to 50 c
    mesh.set_neumann_boundary("right")
    mesh.set_thermal_diffusivity(0.0001)  # m^2/s
    return mesh


def main():
    n_cells = 20
    time_step_size = 15  # seconds
    # time_max = 30000  # seconds
    time_max = 600
    print(create_mesh(4).delta_x)

    explicit_solution_unstable = solver.solver_1d(
        mesh=create_mesh(n_cells),
        initial_time=0,
        time_step_size=15,
        method="explicit",
    )
    explicit_solution_unstable.solve(600)

    explicit_solution_stable = solver.solver_1d(
        mesh=create_mesh(n_cells),
        initial_time=0,
        time_step_size=1,
        method="explicit",
    )
    explicit_solution_stable.solve(time_max)

    logging.warning(
        f"explicit_solution saved data \n {explicit_solution_stable.saved_data.head(10)}"
    )

    # add a model name
    # explicit_unstable_data = explicit_solution_unstable.saved_data
    # explicit_stable_data = explicit_solution_stable.saved_data

    plot_data = pd.concat(
        [explicit_solution_unstable.saved_data, explicit_solution_stable.saved_data]
    )
    time_points = [0, 60, 3600, time_max]  # time points that you want to plot
    plot_data_filterd_bool = plot_data["time"].isin(time_points)
    plot_data_filtered = plot_data[plot_data_filterd_bool]

    plot = (
        pn.ggplot(
            plot_data_filtered,
            pn.aes("x_cordinates", "temperature", color="factor(time)"),
        )
        + pn.geom_line()
        + pn.geom_point()
        + pn.facet_wrap("time_step")
    )

    print(plot)


if __name__ == "__main__":
    main()
