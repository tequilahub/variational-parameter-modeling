import numpy as np
import time
import math

def generate_coordinates(count: int, max_distance: int, iteration: int):
    np_random = np.random.default_rng(iteration*int(time.time()))

    coordinates = np.array([[0, 0, 0]], dtype=np.float64, ndmin=2)
    # randomize the prev_coord out of the coordinates
    for i in range(1, count):
        prev_coord = np_random.choice(coordinates)
        min_distance = 0
        # make sure we have a minimum of 0.5 angstrom between the atom cores
        while min_distance <= 0.5:
            new_coord = prev_coord + (np_random.random(3) * max_distance)

            min_distance = np.min(np.linalg.norm(new_coord - coordinates, axis=1))
        coordinates = np.append(coordinates, new_coord.reshape(1, 3), axis=0)
    return coordinates


def generate_linear_coordinates(atom_count: int, iteration: int, total_iterations: int):
    max_distance = 2.5
    min_distance = 0.5

    step_distance = min_distance + (max_distance - min_distance) * (iteration / (total_iterations - 1))
    z_positions = np.arange(atom_count, dtype=np.float64) * step_distance

    coordinates = np.zeros((atom_count, 3), dtype=np.float64)
    coordinates[:, 2] = z_positions

    return coordinates

def generate_ring_coordinates(atom_count: int, iteration: int, total_iterations: int,
                              min_distance: float = 0.5, max_distance: float = 3.0):
    # interpolate target side length
    step_distance = min_distance + (max_distance - min_distance) * (iteration / (total_iterations - 1))

    # circumcircle radius formula for n-gon with side length d
    radius = step_distance / (2 * math.sin(math.pi / atom_count))

    # generate polygon points
    coordinates = np.zeros((atom_count, 3), dtype=np.float64)
    for i in range(atom_count):
        angle = 2 * math.pi * i / atom_count
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        coordinates[i, 0] = x
        coordinates[i, 1] = y
        # z stays 0 (ring in xy-plane)

    return coordinates


# ------------- Perfect Matching -------------

def unique_edges(n):
    def gen(atoms):
        if not atoms:
            yield []
            return
        a = atoms[0]
        for i in range(1, len(atoms)):
            for rest in gen(atoms[1:i] + atoms[i+1:]):
                yield [(a, atoms[i])] + rest
    return list(gen(list(range(n))))

def best_edges(geometry: np.ndarray, alpha=np.pi/2):
    num_atoms, _ = geometry.shape

    distances = []
    ue = unique_edges(num_atoms)
    for possible in ue:
        dist = 0
        for tuple in possible:
            d = np.linalg.norm(geometry[tuple[1]] - geometry[tuple[0]])
            dist += np.tanh(d*alpha)
        distances.append(dist)

    return [x for _, x in sorted(zip(distances, ue))]