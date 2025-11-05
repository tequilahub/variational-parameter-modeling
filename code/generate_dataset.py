import warnings
warnings.filterwarnings('ignore')

import os
import tequila as tq
import numpy as np
import csv
import time

from .edge_heuristic import generate_coordinates, best_edges, generate_linear_coordinates, generate_ring_coordinates
from .load_dataset import load_dataset


MAX_ATOMS = 6
AMOUNT = 100

OUTPUT_DIR = f"/data/h{MAX_ATOMS}"
CSV_FILE = os.path.join(OUTPUT_DIR, "CHANGE.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)
last_dict = None


def geometry_to_string(array):
    geometry_lines = []
    for arr in array:
        x, y, z = float(arr[0]), float(arr[1]), float(arr[2])
        line = f"H  {x:.5f}  {y:.5f}  {z:.5f}"
        geometry_lines.append(line)
    return "\n".join(geometry_lines)


def save_data(data, file):
    file_exists = os.path.isfile(file)

    with open(file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            'geometry', 'target', 'edges', 'coeff', 'fci', 'energy'
        ])

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'geometry': data['geometry'],
            'target': data['target'],
            'edges': data['edges'],
            'coeff': data['coeff'],
            'fci': data['fci'],
            'energy': data['energy']
        })
    return file



def generate_step(args):
    idx, max_atoms, amount = args
    global last_dict
    geometry = generate_linear_coordinates(max_atoms, idx, amount) #generate_coordinates(4, 4, idx)] # Linear
    #geometry = generate_coordinates(max_atoms, max_distance=4, iteration=idx)   # Random
    #geometry = generate_ring_coordinates(max_atoms, iteration=idx, total_iterations=amount) # Ring
    geometry_string = geometry_to_string(geometry)

    edges = best_edges(geometry)

    guess = np.eye(max_atoms)
    for edge in edges[0]:
        guess[edge[0], edge[1]] = 1.0
        guess[edge[1], edge[0]] = -1.0

    mol = tq.Molecule(geometry=geometry_string, basis_set="sto-3g").use_native_orbitals()

    U = mol.make_ansatz("HCB-SPA", edges=edges[0])

    min = tq.chemistry.optimize_orbitals(mol, circuit=U, initial_guess=guess.T, silent=True, use_hcb=True)
    fci = mol.compute_energy("fci")

    res = tq.minimize(tq.ExpectationValue(U=U, H=min.molecule.make_hardcore_boson_hamiltonian()), silent=True)
    last_dict = res.variables

    error = abs(res.energy - fci) * 1000
    print(f"Error: {round(error, 3)}    i: {idx}") #spa er: {abs(spa.energy - fci) * 1000}")

    new = {}
    i = 0
    for k, v in last_dict:
        new[k] = (v + np.pi) % (2 * np.pi) - np.pi
        i += 1
    angles = [v for v in new.values()]

    result = {
        'geometry': geometry,
        'target': np.array(angles),
        'edges': edges[:4], # saving all graphs for bigger molecules is memory expensive
        'coeff': np.array(min.mo_coeff),
        'fci': fci,
        'energy': res.energy
    }

    return save_data(result, CSV_FILE)


def generate_data(amount, max_atoms):
    filenames = []
    for i in range(amount):
        file = generate_step((i, max_atoms, amount))
        if file is not None:
            filenames.append(file)
    print(f"Finished creating {len(filenames)}/{amount} molecules")
    return filenames


if __name__ == "__main__":
    print("Generating molecules...")
    s = time.time()
    saved_files = generate_data(AMOUNT, MAX_ATOMS)
    print(time.time() - s)

    print("Loading saved data...")
    edges, geometries, targets, operators, fci, error = load_dataset(CSV_FILE)
    print(f"Successfully loaded {len(edges)} molecules")