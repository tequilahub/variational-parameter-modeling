import tequila as tq
import numpy as np
import torch
import os
import warnings
warnings.filterwarnings("ignore")

from .generate_dataset import geometry_to_string
from .models.mixed_schnet import MixedSchNet
from .load_dataset import get_dataset, get_dataloader

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def load_model():
    model = MixedSchNet()
    model.load_state_dict(torch.load(os.path.join(f"{os.getcwd()}/variational-parameter-modeling", f"MixedSchNet.pth"), map_location=device))
    model.to(device)
    model.eval()

    return model


def evaluate(model, eval_loader, device, n):
    model.eval()
    vals = []
    energies = []
    fcis = []
    geometry_index = []
    better_count = 0
    errors = []
    with (torch.no_grad()):
        for edges, geometry, target, coeff, fci, energy in eval_loader:

            z = torch.ones(n, dtype=torch.long, device=device).to(device)
            batch = torch.zeros(n, dtype=torch.long, device=device).to(device)
            geometry = geometry.float().to(device)
            angles = model(z=z, pos=geometry[0], batch=batch, edges=edges)

            geometry = geometry.cpu().numpy()
            point = geometry[0][1][2]

            mol = tq.Molecule(geometry=geometry_to_string(geometry[0]), basis_set="sto-3g").use_native_orbitals()
            edges = [
                [tuple((pair[0].item(), pair[1].item())) for pair in edge]
                for edge in edges
            ]

            U = mol.make_ansatz("HCB-SPA", edges=edges[0])


            min = tq.chemistry.optimize_orbitals(mol, circuit=U, initial_guess=coeff[0].cpu().numpy(), silent=True, use_hcb=True)

            H = min.molecule.make_hardcore_boson_hamiltonian()
            E = tq.ExpectationValue(U=U, H=H)

            pred_dict = {}
            count = 0
            for param_key in U.make_parameter_map():
                pred_dict[param_key] = angles[count].cpu().item()
                count += 1

            #print(target, pred_dict)
            pred = tq.simulate(E, pred_dict)
            if pred < energy.item():
                better_count += 1
            e = (pred-energy)*1000
            print(e)
            errors.append(e)

            vals.append(pred)
            energies.append(energy)
            fcis.append(fci)
            geometry_index.append(point)

    print("better_count:", better_count)
    print("mean error: ", np.array(errors).mean())
    print(errors)
    return np.array(geometry_index), np.array(vals), np.array(energies), np.array(fcis)

def evaluate_model(n, str):
    loader = get_dataloader([get_dataset(fp=f"h{n}/{str}")], batch_size=1)
    model = load_model()

    goemetry_index, pred, target, fci = evaluate(model, loader, device, n)

    import matplotlib.pyplot as plt
    goemetry_index = np.array(goemetry_index)
    pred = np.array(pred)
    target = np.array(target)
    fci = np.array(fci)

    sorted_indices = np.argsort(goemetry_index)
    x_sorted = goemetry_index[sorted_indices]
    target_sorted = target[sorted_indices]
    fci_sorted = fci[sorted_indices]

    plt.figure(figsize=(10, 10))
    plt.plot(x_sorted, target_sorted, label='Baseline (minimization) Energy', color='orange', linewidth=2)
    plt.plot(x_sorted, fci_sorted, label='FCI Energy', color='green', linewidth=2)
    plt.scatter(goemetry_index, pred, label='Predicted Energy', color='blue', s=16)

    for x, y_pred, y_actual in zip(goemetry_index, pred, target):
        x = float(x)
        y_pred = float(y_pred)
        y_actual = float(y_actual)
        plt.plot([x, x], [y_pred, y_actual], color='gray', linestyle='--', linewidth=0.8)

    plt.xlabel(f'Atomic Distance of linear H$_{n}$')
    plt.ylabel('Energy')
    plt.title('Predicted vs Minimized vs FCI -- Energy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    n = 10
    evaluate_model(n=n, str="test_linear_100.csv")

