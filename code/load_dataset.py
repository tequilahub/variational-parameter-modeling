# -*- coding: utf-8 -*-
import glob
import os
import csv

import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
import numpy as np
import ast

def load_dataset(csv_file: str, amount=20000):
    """Load all molecule data from a CSV file"""
    if not os.path.isfile(csv_file):
        print(f"No CSV file found at {csv_file}")
        return [], [], [], [], [], []

    edges, geometries, targets, coeffs, fcis, energies = [], [], [], [], [], []

    with open(csv_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            geometry_data = row['geometry'].replace('[', '').replace(']', '').split()
            geometry_array = np.array([float(x) for x in geometry_data], dtype=np.float64)
            g = geometry_array.reshape(-1, 3)
            geometries.append(g)
            atoms, _ = g.shape

            coeff_data = row['coeff'].replace('[', '').replace(']', '').split()
            coeffs_array = np.array([float(x) for x in coeff_data], dtype=np.float64)
            coeffs.append(coeffs_array.reshape(-1, atoms))

            target_data = row['target'].strip('[]').split()
            target_array = np.array([float(x) for x in target_data], dtype=np.float64)
            edge_data = ast.literal_eval(row['edges'])

            targets.append(target_array)
            edges.append(edge_data)

            fcis.append(float(row['fci']))
            energies.append(float(row['energy']))

    print(f"Loaded {len(geometries)} molecules from {csv_file}")
    return edges[:amount], geometries[:amount], targets[:amount], coeffs[:amount], fcis[:amount], energies[:amount]

class MolDataset(torch.utils.data.Dataset):
    def __init__(self, edges, geometries, targets, coeffs, fci, energies):
        self.edges = edges
        self.geometries = geometries
        self.targets = targets
        self.coeffs = coeffs
        self.fcis = fci
        self.energies = energies

    def __len__(self):
        return len(self.geometries)

    def __getitem__(self, idx):
       return self.edges[idx], self.geometries[idx], self.targets[idx], self.coeffs[idx], self.fcis[idx], self.energies[idx]


def get_dataset(fp) -> MolDataset:
    data_dir = "thesis-jonas/data/"
    edges, geometries, targets, coeffs, fcis, energies = load_dataset(os.path.join(data_dir,fp))
    full_dataset = MolDataset(edges, np.array(geometries), np.array(targets), np.array(coeffs), np.array(fcis), np.array(energies))
    return full_dataset

def get_dataloader(ds, batch_size, shuffle=True) -> DataLoader:
    full_dataset = ConcatDataset(ds)
    return DataLoader(full_dataset, batch_size=batch_size, num_workers=8, shuffle=shuffle)


if __name__ == "__main__":
    edges, geometries, targets, coeffs, fcis, energies = load_dataset("../data/h4-linear-tups/moleculesA.csv")
    print(coeffs)