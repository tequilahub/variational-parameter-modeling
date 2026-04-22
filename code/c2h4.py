import tequila as tq
import numpy as np
import sunrise as sun
import warnings
warnings.filterwarnings("ignore")
import torch
import os

from .generate_dataset import geometry_to_string
from .models.mixed_schnet import MixedSchNet
from .load_dataset import get_dataset, get_dataloader

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = MixedSchNet()
model.load_state_dict(torch.load(os.path.join(f"{os.getcwd()}/variational-parameter-modeling", f"MixedSchNet.pth"), map_location=device))
model.to(device)
model.eval()

print("HEREEEE")
print(model)
variabls = []
variabls_predicted = []
energies = []
energies_predicted = []
fcis = []
distances = np.arange(-0.5, 1, (1-(-0.5))/100)
guess = np.eye(2)
guess[0] = [1.0, 1.0]
guess[1] = [1.0, -1.]
guess = guess.T
for idx, r in enumerate(distances):
    geometry = f"""H {r+1.1610} 0.0000 1.0238
    C {r+0.6579} 0.0000 0.0639
    H {r+1.3352} 0.0000 -0.7815
    C -0.6579 0.0000 -0.0639
    H -1.3355 0.0000 0.7812
    H -1.1608 0.0000 -1.0239"""
    # selection = [3,8]
    # if idx>3:
    #     selection=[8,9]
    mol = tq.Molecule(geometry=geometry, basis_set='sto-3g', frozen_core=True, units='a')#.use_native_orbitals()
    mol, edges = sun.CLPO.generate_CLPO_molecule_edges(mol)
    # print(f"edges: {edges}")
    # sun.plot_MO(mol)

    U = mol.make_ansatz("HCB-SPA", edges=edges)
    

    opt = tq.chemistry.optimize_orbitals(mol, U, use_hcb=True, silent=True)
    mol = opt.molecule
    last_dict = opt.mo_coeff
    # sun.plot_MO(mol,'opt')
    # break

    H = mol.make_hardcore_boson_hamiltonian()

    E = tq.ExpectationValue(U,H)
    result = tq.minimize(E, silent=True)

    new = {}
    i = 0
    for k, v in result.variables.items():
        new[k] = (v + np.pi) % (2 * np.pi) - np.pi
        i += 1
    angles = [v for v in new.values()]


    energies.append(result.energy)
    variabls.append(angles)


    # H2
    mol2 = tq.Molecule(f"H 0 0 0\nH 0 0 {(1+r)*1.25}", "sto-3g").use_native_orbitals()


    z = torch.ones(2, dtype=torch.long, device=device).to(device)
    batch = torch.zeros(2, dtype=torch.long, device=device).to(device)
    geo = np.array([[[0.0,0.0,0.0],[0.0,0.0, (1+r)*1.25]]])
    geo = torch.tensor(geo)
    geo = geo.float().to(device)

    edges=[[[torch.tensor([0]), torch.tensor([1])]]]#, [tensor([2]), tensor([3])], [tensor([4]), tensor([5])], [tensor([6]), tensor([7])], [tensor([8]), tensor([9])]]]

    angles = model(z=z, pos=geo[0], batch=batch, edges=edges).detach().numpy()

    new = {}
    for count, param_key in enumerate(U.make_parameter_map()):
        if count == 3:
            new[param_key] = angles[0]
        else:
            new[param_key] = list(result.variables.values())[count]


    energy = tq.simulate(E, variables=new)

    energies_predicted.append(energy)
    variabls_predicted.append(angles[0])

    print(f"error: {(energy-result.energy)*1000} mHa")

import matplotlib.pyplot as plt
import numpy as np

# Convert lists properly
energies = np.array(energies)
fcis = np.array(fcis)
energies_predicted = np.array(energies_predicted)

# Convert variables (list of dict_values) → 2D array
variables_array = np.array(variabls)[:,3]
print(variables_array.shape)

variabls_predicted = np.array(variabls_predicted)

x = np.arange(len(energies))

# ---- Plot 1: Energy vs FCI ----
plt.figure()
plt.plot(distances, energies, label="VQE Energy")
plt.plot(distances, energies_predicted, label="Predicted Energy", linestyle="dashed")
#plt.plot(distances, fcis, label="FCI Energy")
plt.xlabel("C-C distance [Å]")
plt.ylabel("Energy [Eh]")
plt.title("Optimized vs Predicted - Energies")
plt.legend()
plt.savefig("energies_c2.pdf")
plt.show()

# ---- Plot 2: Variables ----
plt.figure()

# Plot each variable separately

plt.plot(distances, variables_array, label=f"VQE Parameter")
plt.plot(distances, variabls_predicted, label="Predicted Parameter", linestyle="dashed")
plt.xlabel("C-C distance [Å]")
plt.ylabel("Variable Values")
plt.title("Optimized vs Predicted - Variables")
plt.legend()
plt.savefig("variables_c2.pdf")
plt.show()