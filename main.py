from enum import StrEnum
from typing import Mapping, Protocol, Sequence, Callable
from enum import Enum
import pathlib as pl
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import argparse
from dataclasses import dataclass
import numpy.typing as npt
import h5py
import cProfile

# =============================================================================================================
# File handlers (including the protocol) for saving and loading simulation state
# =============================================================================================================


class FileHandler(Protocol):
    def save_state(self, filename: str | pl.Path, data: Mapping[str, Sequence[float] | npt.NDArray]) -> None: ...

    def load_state(self, filename: str | pl.Path) -> tuple[npt.NDArray, ...]: ...


class FileHandlerType(StrEnum):
    CSV = "CSV"
    HDF5 = "HDF5"


class CSVFileHandler:
    header = ["x_pos", "y_pos", "x_vel", "y_vel", "radius", "mass"]

    def save_state(self, filename: str | pl.Path, data: Mapping[str, Sequence[float] | npt.NDArray]) -> None:
        file = pl.Path(filename).with_suffix(".csv")
        with open(file, "w") as f:
            f.write(",".join(self.header) + "\n")
            num_particles = len(data["x_pos"])
            for i in range(num_particles):
                row = [
                    str(data["x_pos"][i]),
                    str(data["y_pos"][i]),
                    str(data["x_vel"][i]),
                    str(data["y_vel"][i]),
                    str(data["radius"][i]),
                    str(data["mass"][i]),
                ]
                f.write(",".join(row) + "\n")

    def load_state(self, filename: str | pl.Path) -> tuple[npt.NDArray, ...]:
        data = {key: [] for key in self.header}
        file = pl.Path(filename).with_suffix(".csv")
        with open(file, "r") as f:
            next(f)  # Skip header
            for line in f:
                values = line.strip().split(",")
                for key, value in zip(self.header, values):
                    data[key].append(float(value))
        return tuple(np.array(value) for value in data.values())


class HDF5FileHandler:
    def save_state(self, filename: str | pl.Path, data: Mapping[str, Sequence[float] | npt.NDArray]) -> None:
        file = pl.Path(filename).with_suffix(".h5")
        with h5py.File(file, "w") as f:
            for key, value in data.items():
                f.create_dataset(key, data=value)

    def load_state(self, filename: str | pl.Path) -> tuple[npt.NDArray, ...]:
        data = {}
        file = pl.Path(filename).with_suffix(".h5")
        with h5py.File(file, "r") as f:
            for key in f.keys():
                data[key] = f[key][:]  # type: ignore
        return tuple(data.values())


_file_handler_register: dict[FileHandlerType, FileHandler] = {}

# =============================================================================================================
# Checking for collisions algorithms
# =============================================================================================================


class CollisionCheckerAlgorithm(Enum):
    NUMPY_NORM = 1
    DISTANCE_SQUARED = 2


def get_collision_pairs_numpy_norm(positions: npt.NDArray, radii: npt.NDArray) -> list[tuple[int, int]]:
    """Orgininal implementation using np.linalg.norm to compute distances. This method is very inefficient for small radii, but very powerful for large arrays due to numpy optimizations."""
    collisions = []
    num_particles = positions.shape[0]
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < (radii[i] + radii[j]):
                collisions.append((i, j))
    return collisions


def get_collision_pairs_distance_squared(positions: npt.NDArray, radii: npt.NDArray) -> list[tuple[int, int]]:
    """Implementation using distance squared to avoid computing square roots. This method is more efficient for small radii."""
    collisions = []
    num_particles = positions.shape[0]
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            x_diff = positions[i, 0] - positions[j, 0]
            y_diff = positions[i, 1] - positions[j, 1]
            dist_sq = x_diff**2 + y_diff**2
            radius_sum = radii[i] + radii[j]
            if dist_sq < radius_sum**2:
                collisions.append((i, j))
    return collisions


_collision_pair_finder_register: dict[CollisionCheckerAlgorithm, Callable[[npt.NDArray, npt.NDArray], list[tuple[int, int]]]] = {}

# =============================================================================================================
# Solving collisions algorithms
# =============================================================================================================


def resolve_collision(positions: npt.NDArray, velocities: npt.NDArray, masses: npt.NDArray, i: int, j: int) -> None:
    r_rel_ij = positions[i] - positions[j]
    r_rel_ji = positions[j] - positions[i]
    v_rel_ij = velocities[i] - velocities[j]
    v_rel_ji = velocities[j] - velocities[i]
    dist = np.linalg.norm(r_rel_ij)
    if np.dot(v_rel_ij, r_rel_ij) < 0:
        norm_r_ij = r_rel_ij / dist
        norm_r_ji = r_rel_ji / dist
        impulse_i = 2 * masses[i] / (masses[i] + masses[j]) * np.dot(v_rel_ij, norm_r_ij) * norm_r_ij
        impulse_j = 2 * masses[j] / (masses[i] + masses[j]) * np.dot(norm_r_ji, v_rel_ji) * norm_r_ji
        velocities[i] -= impulse_i
        velocities[j] -= impulse_j


_resolve_collision_register: dict[int, Callable[[npt.NDArray, npt.NDArray, npt.NDArray, int, int], None]] = {
    1: resolve_collision,
}

# =============================================================================================================
# Settings class for storing simulation parameters
# =============================================================================================================


@dataclass
class SimulationConfig:
    NUM_PARTICLES: int = 1000
    BOX_SIZE: float = 1.0
    DT: float = 0.01
    MIN_RADIUS: float = 0.005
    MAX_RADIUS: float = 0.02
    MIN_VELOCITY: float = -0.1
    MAX_VELOCITY: float = 0.1
    SCATTER_SCALE: float = 1000
    REPEAT_ANIMATION: bool = True
    WRITE_STATE: bool = False
    # A very rough estimation for the maximum number of files that can be created
    FILES_LIMIT: int = int(200 * 1024 * 1024 / (150 * NUM_PARTICLES))
    FILE_HANDLER_TYPE: FileHandlerType = FileHandlerType.CSV
    PAIR_SOLVER: CollisionCheckerAlgorithm = CollisionCheckerAlgorithm.DISTANCE_SQUARED
    COLLISION_RESOLVER: int = 1


# =============================================================================================================
# Main simulation class
# =============================================================================================================


class ParticleSimulation:
    """
    Simulates the motion and interaction of particles within a bounded environment.

    This class handles the initialization, movement, and collision resolution of
    particles inside a simulation box. It supports running simulations with or
    without animations, where particles are depicted as moving circles with varying
    radii and velocities. The interactions between particles include collision detection
    and resolution based on physical dynamics principles.

    :ivar config: Configuration settings for the simulation, defining parameters
        such as the number of particles, box size, velocity bounds, and more.
    :type config: SimulationConfig
    :ivar total_elapsed_time: The accumulated time spent processing collisions in seconds.
    :type total_elapsed_time: float
    :ivar radii: The radii of individual particles in the simulation.
    :type radii: numpy.ndarray
    :ivar masses: The masses of individual particles, derived from their radii.
    :type masses: numpy.ndarray
    :ivar positions: The positions of the particles in the 2D simulation space.
    :type positions: numpy.ndarray
    :ivar velocities: The velocities of particles in the 2D simulation space.
    :type velocities: numpy.ndarray
    :ivar particles: Handles the scatter plot representing particles for visualization,
        used only in animated simulations. None if animation is not enabled.
    :type particles: matplotlib.collections.PathCollection or None
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config: SimulationConfig = config
        self.total_elapsed_time: float = 0.0
        self.radii: npt.NDArray = np.zeros(config.NUM_PARTICLES)
        self.masses: npt.NDArray = np.zeros(config.NUM_PARTICLES)
        self.positions: npt.NDArray = np.zeros((config.NUM_PARTICLES, 2))
        self.velocities: npt.NDArray = np.zeros((config.NUM_PARTICLES, 2))
        self.particles = None
        self.file_counter: int = 0

        # Dependency injections for file handling and collision solving
        self.file_handler: FileHandler = _file_handler_register[config.FILE_HANDLER_TYPE]
        self.pair_solver: Callable[[npt.NDArray, npt.NDArray], list[tuple[int, int]]] = _collision_pair_finder_register[config.PAIR_SOLVER]
        self.collision_solver: Callable[[npt.NDArray, npt.NDArray, npt.NDArray, int, int], None] = _resolve_collision_register[config.COLLISION_RESOLVER]

        self.output_dir = pl.Path("simulation_outputs")
        self.output_dir.mkdir(exist_ok=True)
        pl.Path(self.output_dir / "settings.txt").write_text("\n".join([f"{key}: {value}" for key, value in vars(config).items()]))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # If no calculation files are created, remove the output directory
        if self.file_counter == 0:
            for file in self.output_dir.iterdir():
                file.unlink()
            self.output_dir.rmdir()

        # Log the performance metrics of the simulation
        with open(self.output_dir / "performance_log.txt", "w") as f:
            f.write(f"Total elapsed time: {self.total_elapsed_time} seconds\n")
            f.write(f"Average time per step: {self.total_elapsed_time / (self.file_counter if self.file_counter > 0 else 1)} seconds\n")

    def save_state(self, filename: str | pl.Path) -> None:
        """
        Saves the current state of all particles to a text file in CSV format.

        Each row represents one particle with the following columns:
        x_position, y_position, x_velocity, y_velocity, radius, mass

        :param filename: Name of the file to save the state to
        :type filename: str
        """
        data = {
            "x_pos": self.positions[:, 0],
            "y_pos": self.positions[:, 1],
            "x_vel": self.velocities[:, 0],
            "y_vel": self.velocities[:, 1],
            "radius": self.radii,
            "mass": self.masses,
        }
        self.file_handler.save_state(filename=filename, data=data)

    def load_state(self, filename: str) -> None:
        """
        Loads particle state from a CSV file and initializes the simulation with it.

        Expects a file created by save_state() containing comma-separated values
        with columns: x_position, y_position, x_velocity, y_velocity, radius, mass

        :param filename: Name of the file to load the state from
        :type filename: str
        :raises ValueError: If the loaded data doesn't match the configured number of particles
        """
        data = self.file_handler.load_state(filename=filename)
        x_pos, y_pos, x_vel, y_vel, radius, mass = data

        # Convert to numpy arrays (and group positions and velocities)
        self.positions = np.array(list(zip(x_pos, y_pos)))
        self.velocities = np.array(list(zip(x_vel, y_vel)))
        self.radii = np.array(radius)
        self.masses = np.array(mass)

        # Verify particle count
        if len(self.positions) != self.config.NUM_PARTICLES:
            raise ValueError(f"Loaded data contains {len(self.positions)} particles, but {self.config.NUM_PARTICLES} configured")

    def initialize_particles(self, seed: int = 42) -> None:
        """
        Initializes particles with random positions, velocities, radii, and masses based on configuration
        parameters. The method uses the given seed value to ensure reproducibility in the random number
        generation process. Particles' radii, positions and velocities are assigned within specified bounds,
        and masses are calculated based on the radii.

        :param seed: Seed value for the random number generator.
        :type seed: int
        :return: This method does not return any value.
        :rtype: None
        """
        np.random.seed(seed)
        self.radii = np.random.uniform(self.config.MIN_RADIUS, self.config.MAX_RADIUS, self.config.NUM_PARTICLES)
        self.masses = self.radii**2
        self.positions = np.random.uniform(
            self.radii[:, None],
            (self.config.BOX_SIZE - self.radii)[:, None],
            (self.config.NUM_PARTICLES, 2),
        )
        self.velocities = np.random.uniform(
            self.config.MIN_VELOCITY,
            self.config.MAX_VELOCITY,
            (self.config.NUM_PARTICLES, 2),
        )

    def get_collision_pairs(self) -> list[tuple[int, int]]:
        """
        Checks for collisions among particles in the simulation.

        The method iterates through all pairs of particles and computes the distance
        between their positions. If the distance between two particles is less than
        the sum of their radii, it identifies a collision and stores the pair of indices
        representing the colliding particles.
        """
        return self.pair_solver(self.positions, self.radii)

    def resolve_collision(self, i: int, j: int) -> None:
        """
        Resolves a collision between two objects identified by their indices, i and j.
        This function adjusts the velocities of the objects based on their relative
        positions and velocities. It ensures that the objects bounce off each other
        according to the rules of elastic collisions.

        The collision resolution considers the masses, positions, and velocities
        of the objects. The approach assumes that the objects would not overlap
        post-collision and calculates impulses to adjust their velocities.

        :param i: Index of the first object involved in the collision.
        :param j: Index of the second object involved in the collision.
        :type i: int
        :type j: int
        :return: This function does not return any value.
        :rtype: None
        """
        self.collision_solver(self.positions, self.velocities, self.masses, i, j)

    def move(self, step: int) -> None:
        """
        Updates the positions of particles and handles wall collisions as part of a
        step in the simulation. Additionally, checks for particle collisions
        and updates the elapsed time of the simulation.

        :param step: The current simulation step number.
        :type step: int
        :return: None
        """
        self.positions += self.velocities * self.config.DT

        for i in range(self.config.NUM_PARTICLES):
            for d in range(2):
                if self.positions[i, d] - self.radii[i] < 0 or self.positions[i, d] + self.radii[i] > self.config.BOX_SIZE:
                    self.velocities[i, d] *= -1

        start_time = time.time()
        pairs = self.get_collision_pairs()
        [self.resolve_collision(i, j) for i, j in pairs]

        elapsed_time = time.time() - start_time
        self.total_elapsed_time += elapsed_time
        print(f"\nStep: {step}")
        print(f"Elapsed time: {elapsed_time}")

        if self.config.WRITE_STATE:
            if step % 1 == 0 and self.file_counter < self.config.FILES_LIMIT:
                self.file_counter += 1
                self.save_state(self.output_dir / f"simulation_state_step_{step}")
            else:
                print(f"Too many files, skipping call for 'save_state' at step {step}")

    def update_animation(self, step: int):
        """
        Updates the animation state by moving the particles and setting their offsets.
        This function is typically used as an update function in animation loops.
        The positions of the particles are updated based on the given step, and
        the updated positions are applied to the visualization. It ensures that
        particle rendering reflects the most recent positions.

        :param step: The step value determining how much each particle moves.
        :type step: int
        :return: The updated particle artists for rendering.
        :rtype: tuple
        """
        self.move(step)

        if self.particles is not None:
            self.particles.set_offsets(self.positions)
        return (self.particles,)

    def run_simulation(self, num_steps: int, animate: bool = False) -> None:
        """
        Run the simulation for a specified number of steps, with an optional animation.

        This method is used to either animate the simulation or run it step-by-step
        without visualization. If animation is enabled, it sets up the plotting
        environment and animates the movement of particles. Otherwise, it proceeds with
        a step-by-step update of the simulation state.

        :param num_steps: The number of steps the simulation should run.
        :type num_steps: int
        :param animate: Whether to animate the simulation. Defaults to False.
        :type animate: bool, optional
        :return: This method does not return anything.
        :rtype: None
        """
        if animate:
            matplotlib.use("TkAgg")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_xlim(0, self.config.BOX_SIZE)
            ax.set_ylim(0, self.config.BOX_SIZE)
            self.particles = ax.scatter(
                self.positions[:, 0],
                self.positions[:, 1],
                s=(self.radii * self.config.SCATTER_SCALE) ** 2,
                alpha=0.6,
            )
            ani = animation.FuncAnimation(
                fig,
                self.update_animation,  # type: ignore
                frames=num_steps,
                interval=20,
                blit=True,
                repeat=self.config.REPEAT_ANIMATION,
            )
            plt.show()
        else:
            for step in range(num_steps):
                self.move(step)

        print(f"\nTotal elapsed time: {self.total_elapsed_time} seconds")
        print(f"Average time per step: {self.total_elapsed_time / num_steps} seconds")


# =============================================================================================================
# Main function and argument parsing
# =============================================================================================================


def log_simulation_config(config: SimulationConfig) -> None:
    print("Simulation Configuration:")
    print(f"Number of particles: {config.NUM_PARTICLES}")
    print(f"Box size: {config.BOX_SIZE}")
    print("Number of dimensions: 2")
    print(f"Write state: {config.WRITE_STATE}")
    print(f"Files limit: {config.FILES_LIMIT}")
    print(f"Repeat animation: {config.REPEAT_ANIMATION}")
    print(f"File handler type: {config.FILE_HANDLER_TYPE.value}")
    print(f"Collision pair solver algorithm: {config.PAIR_SOLVER.name}")


def main():
    """
    Main entry point for the particle collision simulation. This function parses
    command-line arguments, initializes the simulation configuration, and starts the
    simulation process. The simulation includes options for enabling animations and
    choosing the version of the collision function implementation.

    :raises SystemExit: If the required arguments are missing or invalid when parsing
        command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Particle collision simulation")
    # Add arguments
    parser.add_argument("--animate", type=str, required=False, help="1 if simulations should be animated, 0 otherwise")
    parser.add_argument("--file_handler", type=str, required=False, default="HDF5", help="Type of file handler to use: CSV or HDF5")
    parser.add_argument("--pair_solver", type=str, required=False, default="DISTANCE_SQUARED", help="Algorithm to use for collision pair finding: NUMPY_NORM or DISTANCE_SQUARED")
    args = parser.parse_args()

    config = SimulationConfig()

    # -----------------------------------------------------------------------------------------
    # Animation
    # -----------------------------------------------------------------------------------------

    ANIMATE = True

    # -----------------------------------------------------------------------------------------
    # Select file handler based on argument
    # -----------------------------------------------------------------------------------------

    _file_handler_register[FileHandlerType.CSV] = CSVFileHandler()
    _file_handler_register[FileHandlerType.HDF5] = HDF5FileHandler()

    if args.file_handler.upper() not in FileHandlerType.__members__:
        print(f"Error! Unknown file handler type: {args.file_handler}")
        exit(1)

    config.FILE_HANDLER_TYPE = FileHandlerType[args.file_handler.upper()]
    config.WRITE_STATE = True

    # -----------------------------------------------------------------------------------------
    # Select collision solver iteration based on argument
    # -----------------------------------------------------------------------------------------

    _collision_pair_finder_register[CollisionCheckerAlgorithm.NUMPY_NORM] = get_collision_pairs_numpy_norm
    _collision_pair_finder_register[CollisionCheckerAlgorithm.DISTANCE_SQUARED] = get_collision_pairs_distance_squared

    if args.pair_solver.upper() not in CollisionCheckerAlgorithm.__members__:
        print(f"Error! Unknown collision pair solver algorithm: {args.pair_solver}")
        exit(1)

    config.PAIR_SOLVER = CollisionCheckerAlgorithm[args.pair_solver.upper()]

    # -----------------------------------------------------------------------------------------
    # Run the simulation
    # -----------------------------------------------------------------------------------------

    # Log the configuration to the console
    log_simulation_config(config)
    with ParticleSimulation(config=config) as simulation:
        simulation.initialize_particles()
        simulation.run_simulation(num_steps=100, animate=ANIMATE)


if __name__ == "__main__":
    main()
    # cProfile.run("main()", sort="cumtime")
