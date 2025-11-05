import tequila as tq
import numpy as np
import torch

from .generate_dataset import geometry_to_string

def evaluate(model, eval_loader, device):
    model.eval()
    vals = []
    better_count = 0
    with (torch.no_grad()):
        for edges, geometry, target, coeff, fci, energy in eval_loader:
            n, _ = geometry[0].shape

            z = torch.ones(n, dtype=torch.long, device=device).to(device)
            batch = torch.zeros(n, dtype=torch.long, device=device).to(device)
            geometry = geometry.float().to(device)
            angles = model(z=z, pos=geometry[0], batch=batch, edges=edges)

            geometry = geometry.cpu().numpy()

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

            for count, param_key in enumerate(U.make_parameter_map()):
                pred_dict[param_key] = angles[count].cpu().item()


            pred = tq.simulate(E, pred_dict)
            error = (pred-energy)*1000
            if error < 0:
                better_count += 1

            vals.append(error)
    return np.array(vals).mean(), better_count

def evaluate_structure(model, loaders, device, epoch):
    avg = []
    for name, loader in loaders.items():
        r, b = evaluate(model, loader, device)
        avg.append(r)
        print(f"{name},  Error: {round(r, 3)},  Better Instances: {b},   @epoch: {epoch}")
    return np.array(avg).mean()