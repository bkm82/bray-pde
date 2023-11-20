from solver.cartesian_mesh import CartesianMesh
from solver.solver import Solver
from solver.utilities import EnergyBalance, Plotter
from solver.mesher import linear_convection_mesh
from matplotlib import pyplot as plt
import numpy as np
import logging
import math

# create logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(message)s",
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Define a various mesh sizes with non equal cell widths and heights
mesh_size_dict = {
    "course": (5, 2),
    "medium": (20, 8),
    "fine": (100, 10),
    "ultra_fine": (200, 50),
}
# Define Problem constants
x_range = (0, 5)
y_range = (0, 1)
mesh_type = "finite_volume"
conductivity = 0.456  ## W/mK
diffusivity = 0.146 * 10 ** (-6)  # m^2/s
temp_left = 30  ## decrees celcius
temp_right = 30  ## degrees celcius
temp_bottom = 30  # degrees celsius
top_flux = -10  # W/m


# Create a function that creates the mesh
def create_mesh(
    x_cells=80,
    y_cells=12,
    temp_left=temp_left,
    temp_right=temp_right,
    temp_bottom=temp_bottom,
    top_flux=top_flux,
    conductivity=conductivity,
    diffusivity=diffusivity,
):
    mesh = CartesianMesh(
        dimensions=2,
        n_cells=[x_cells, y_cells],
        cordinates=[x_range, y_range],
        mesh_type=mesh_type,
        conductivity=conductivity,
        diffusivity=diffusivity,
    )

    mesh.set_dirichlet_boundary(side="left", phi=temp_left)
    mesh.set_dirichlet_boundary(side="right", phi=temp_right)
    mesh.set_dirichlet_boundary(side="bottom", phi=temp_bottom)
    mesh.set_neumann_boundary(side="top", flux=top_flux / conductivity)
    return mesh


mesh_dict = {}
energy_balance_dict = {}
for mesh_size, cells in mesh_size_dict.items():
    x_cells, y_cells = cells

    mesh_dict[mesh_size] = create_mesh(x_cells=x_cells, y_cells=y_cells)
    # Solve the steady Case
    Solver(mesh=mesh_dict[mesh_size]).solve_steady()
    energy_balance_dict[mesh_size] = EnergyBalance(mesh=mesh_dict[mesh_size])
    logger.info(f"\n Mesh Size:{mesh_size}")
    total_flux = energy_balance_dict[mesh_size].flux("all")

fig, ax = plt.subplots(len(mesh_dict), figsize=(10, 12))
i = 0
for mesh_size, mesh in mesh_dict.items():
    x_cords = mesh.grid["x_grid"].cell_cordinates
    y_cords = mesh.grid["y_grid"].cell_cordinates
    logger.debug(x_cords)
    xv, yv = np.meshgrid(x_cords, y_cords)
    axis = plt.subplot(len(mesh_dict), 1, i + 1)
    axis.set_title(mesh_size)
    steady = axis.pcolormesh(xv, yv, mesh.phi.get_phi())

    i = i + 1

    fig.colorbar(steady, label="temperature (celcius)")
plt.savefig("steady.png")

generation_mesh_dict = {}
generation_energy_balance_dict = {}


def gen_function(x, y):
    return 0 * y + 50 * np.exp(-((x - 2.5) ** 2))


for mesh_size, cells in mesh_size_dict.items():
    x_cells, y_cells = cells

    # Create the mesh
    generation_mesh_dict[mesh_size] = create_mesh(x_cells=x_cells, y_cells=y_cells)

    # Add the generation function
    generation_mesh_dict[mesh_size].set_generation(function=gen_function)

    # Solve the steady Case with generation
    Solver(mesh=generation_mesh_dict[mesh_size]).solve_steady()
    generation_energy_balance_dict[mesh_size] = EnergyBalance(
        mesh=generation_mesh_dict[mesh_size]
    )
    logger.info(f"\n Mesh Size:{mesh_size}")
    total_flux = generation_energy_balance_dict[mesh_size].flux("all")

fig, ax = plt.subplots(len(generation_mesh_dict), figsize=(10, 12))
i = 0
for mesh_size, mesh in generation_mesh_dict.items():
    x_cords = mesh.grid["x_grid"].cell_cordinates
    y_cords = mesh.grid["y_grid"].cell_cordinates
    logger.debug(x_cords)
    xv, yv = np.meshgrid(x_cords, y_cords)
    axis = plt.subplot(len(generation_mesh_dict), 1, i + 1)
    axis.set_title(mesh_size)
    steady = axis.pcolormesh(xv, yv, mesh.phi.get_phi())

    i = i + 1

    fig.colorbar(steady, label="temperature (celcius)")
plt.savefig("generation.png")

transient_mesh = create_mesh()
transient_mesh.phi.set_phi(30)

# Create the implicit solver
transient_solver = Solver(mesh=transient_mesh, method="implicit", time_step_size=10000)

transient_solver.solve(
    t_final=14000000, record_step=100, compute_error_flag=True, tolerance=0.1
)

part3_plotter = Plotter(mesh=transient_mesh)
part3_plotter.transient_plotter(
    data_list=transient_solver.saved_state_list, name="transient"
)


def create_advection_diffusion_mesh(velocity, x_cells=60, y_cells=20):
    # Create our standard 2d cartesian mesh
    mesh = create_mesh(x_cells=x_cells, y_cells=y_cells, temp_left=50)
    mesh.phi.set_phi(30)

    # Create a 1d finite volume linear convective mesh (from HW5)
    convection_mesh = linear_convection_mesh(
        x=x_range,
        n_cells=x_cells,
        mesh_type="finite_volume",
        discretization_type="upwind",
    )

    convection_mesh.set_dirichlet_boundary("left", phi=50)

    dx = mesh.grid["x_grid"].cell_width
    Iy = np.identity(y_cells)
    # make the 2d laplacian using the kronecker delta and Iy
    # This puts the 1d laplacian into every 1 in an identity of the shape y_cells x y_cells
    twod_x_lap = (velocity / dx) * np.kron(Iy, convection_mesh.laplacian)
    # logger.debug(f"{twod_x_lap}")
    twod_x_bc = (velocity / (dx)) * convection_mesh.boundary_condition_array.reshape(
        1, x_cells
    ).repeat(y_cells, axis=0).flatten()

    # add the convective terms to the 2d diffusion mesh
    mesh.laplacian = mesh.laplacian + twod_x_lap
    mesh.boundary_condition_array = mesh.boundary_condition_array + twod_x_bc
    return mesh


ultra_slow_dict = {
    "name": "ultra_slow",
    "velocity": 0.000001,
    "time_step_size": 10000,
    "t_final": 10000000,
    "record_step_time": 10,
    "tolerance": 0.01,
}

mixed_dict = {
    "name": "mixed",
    "velocity": 0.00001,
    "time_step_size": 1000,
    "t_final": 1000000,
    "record_step_time": 10,
    "tolerance": 0.01,
}

slow_dict = {
    "name": "slow",
    "velocity": 0.01,
    "time_step_size": 1,
    "t_final": 1000,
    "record_step_time": 10,
    "tolerance": 0.01,
}

med_dict = {
    "name": "med",
    "velocity": 0.1,
    "time_step_size": 0.1,
    "t_final": 100,
    "record_step_time": 10,
    "tolerance": 0.01,
}

fast_dict = {
    "name": "fast",
    "velocity": 1,
    "time_step_size": 0.01,
    "t_final": 10,
    "record_step_time": 10,
    "tolerance": 0.01,
}

settings_list = [fast_dict, med_dict, slow_dict, mixed_dict, ultra_slow_dict]
###settings_list= [mixed_dict]

# # make a dictionary of solvers for the various time velocities
mesh_dict = {}
solutions_dict = {}
# Create the mesh
for settings in settings_list:
    mesh_dict[settings["name"]] = create_advection_diffusion_mesh(
        velocity=settings["velocity"]
    )

    solutions_dict[settings["name"]] = Solver(
        mesh=mesh_dict[settings["name"]],
        method="implicit",
        time_step_size=settings["time_step_size"],
    )

    solutions_dict[settings["name"]].solve(
        t_final=settings["t_final"],
        record_step=settings["record_step_time"],
        compute_error_flag=True,
        tolerance=settings["tolerance"],
    )

for name, solution in solutions_dict.items():
    part4_plotter = Plotter(mesh=mesh_dict[name]).transient_plotter(
        data_list=solution.saved_state_list,
        name=f"advection_diffusion_{name}",
        phi_min=30,
        phi_max=50,
    )
