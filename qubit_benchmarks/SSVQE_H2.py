import numpy as np
from qiskit import QuantumCircuit
import qiskit
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_algorithms import NumPyEigensolver
from qiskit.circuit.library import TwoLocal

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakeSherbrooke

backend = qiskit.Aer.get_backend('qasm_simulator')
N = 2
energy_levels = 3

global shots
shots = 2**20

step_size = 0.6 # Step Size for theta_dot
max_iter = 60

# This is the SSVQE algorithm applied to an H2 molecule
#g = {Radius: [I, Z1, X1X2, Z2, Z1Z2]}
g = {0.30: [1.01018E+00, -8.08649E-01, 1.60818E-01, -8.08649E-01, 1.32880E-02],
    0.35: [7.01273E-01, -7.47416E-01, 1.62573E-01, -7.47416E-01, 1.31036E-02],
    0.40: [4.60364E-01, -6.88819E-01, 1.64515E-01, -6.88819E-01, 1.29140E-02],
    0.45: [2.67547E-01, -6.33890E-01, 1.66621E-01, -6.33890E-01, 1.27192E-02],
    0.50: [1.10647E-01, -5.83080E-01, 1.68870E-01, -5.83080E-01, 1.25165E-02],
    0.55: [-1.83734E-02, -5.36489E-01, 1.71244E-01, -5.36489E-01, 1.23003E-02],
    0.65: [-2.13932E-01, -4.55433E-01, 1.76318E-01, -4.55433E-01, 1.18019E-02],
    0.75: [-3.49833E-01, -3.88748E-01, 1.81771E-01, -3.88748E-01, 1.11772E-02],
    0.85: [-4.45424E-01, -3.33747E-01, 1.87562E-01, -3.33747E-01, 1.04061E-02],
    0.95: [-5.13548E-01, -2.87796E-01, 1.93650E-01, -2.87796E-01, 9.50345E-03],
    1.05: [-5.62600E-01, -2.48783E-01, 1.99984E-01, -2.48783E-01, 8.50998E-03],
    1.15: [-5.97973E-01, -2.15234E-01, 2.06495E-01, -2.15234E-01, 7.47722E-03],
    1.25: [-6.23223E-01, -1.86173E-01, 2.13102E-01, -1.86173E-01, 6.45563E-03],
    1.35: [-6.40837E-01, -1.60926E-01, 2.19727E-01, -1.60926E-01, 5.48623E-03],
    1.45: [-6.52661E-01, -1.38977E-01, 2.26294E-01, -1.38977E-01, 4.59760E-03],
    1.55: [-6.60117E-01, -1.19894E-01, 2.32740E-01, -1.19894E-01, 3.80558E-03],
    1.65: [-6.64309E-01, -1.03305E-01, 2.39014E-01, -1.03305E-01, 3.11545E-03],
    1.75: [-6.66092E-01, -8.88906E-02, 2.45075E-01, -8.88906E-02, 2.52480E-03],
    1.85: [-6.66126E-01, -7.63712E-02, 2.50896E-01, -7.63712E-02, 2.02647E-03],
    1.95: [-6.64916E-01, -6.55065E-02, 2.56458E-01, -6.55065E-02, 1.61100E-03],
    2.05: [-6.62844E-01, -5.60866E-02, 2.61750E-01, -5.60866E-02, 1.26812E-03]
    }


def SSVQE(params, get_energy=False):
    global ansatz
    global H
    global iterations
    hamiltonian = H
    '''#Noisy Backend
    device_backend = FakeSherbrooke() 
    coupling_map = device_backend.coupling_map
    noise_model = NoiseModel.from_backend(device_backend)
    basis_gates = noise_model.basis_gates
    estimator = Estimator(options={"shots": shots, "noise_model":noise_model, "coupling_map":coupling_map, "basis_gates":basis_gates})
    '''
    estimator = Estimator()#options={"shots": 2**20})
    #'''
    loss = 0.0
    weights = [1.0, 0.8, 0.6] # Weights to create joint loss function
    energies = [] #List to store the energies found
    k=len(weights) # Number of desired eigenstates
    for i in range(k):
        qc = QuantumCircuit(hamiltonian.num_qubits)
        for j in range(i):
            qc.x(j) # Need orthogonal initial states for SSVQE
        qc = qc.compose(ansatz.assign_parameters(params)) # Apply the unitary ansatz
        result = estimator.run(qc, hamiltonian).result()
        loss += weights[i]*result.values[0]
        energies.append(result.values[0])
    if(get_energy):
        print(f"Energy: {energies}]")
        return energies
    else:
        print(f"Current Loss: {loss}")
        return loss

global H
global ansatz
all_energies = []
all_stdevs = []
all_exact_energies = []
ansatz = TwoLocal(2, ["rx", "ry"], "cz", reps=2)
x0 = 2*np.pi*np.random.random(ansatz.num_parameters) - np.pi # Random Initial Parameters
hhdis_ref = np.array([0.35, 0.40, 0.45, 0.50, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05,
                     1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95, 2.05])
for R in hhdis_ref:
    #R=0.95
    H = SparsePauliOp.from_list([("II", g[R][0]), ("ZI", g[R][1]), ("XX", g[R][2]), ("IZ", g[R][3]), ("ZZ", g[R][4])])

    #'''# Exact Diagonalization
    print("Exactly Solving With Numpy...")
    exact_solver = NumPyEigensolver(k=energy_levels)
    exact_result = exact_solver.compute_eigenvalues(H)
    exact_solution = exact_result.eigenvalues
    #exact_solution = np.linalg.eig(H.to_matrix()).eigenvalues
    print(exact_solution)
    print()
    all_exact_energies.append(exact_solution)
    #'''

    #''' Run SSVQE
    from scipy.optimize import minimize
    res = minimize(SSVQE, x0, method="BFGS")#, options={"maxiter":max_iter})
    print("SSVQE Result!")
    print(SSVQE(res.x, get_energy=True))
    all_energies.append(SSVQE(res.x, get_energy=True))
    x0 = res.x[:]
    #'''

print(all_energies)

colors = ["royalblue", "cornflowerblue", "lightsteelblue"]

all_energies = np.array(all_energies)
all_exact_energies = np.array(all_exact_energies)
print("SSVQE Energies: "+str(all_energies))

labels = ["Ground State", "First Excited", "Second Excited", "Exact Solution"]
label_flag = True
for energy_level in range(energy_levels):
    plt.scatter(hhdis_ref, all_energies[:, energy_level], c=colors[energy_level], label=labels[energy_level])
for energy_level in range(energy_levels):
    if(label_flag):
        plt.plot(hhdis_ref, all_exact_energies[:, energy_level], c='g', linestyle="dashed", label="Exact Solution")
        label_flag = False
    else:
        plt.plot(hhdis_ref, all_exact_energies[:, energy_level], c='g', linestyle="dashed")
plt.legend()
plt.title("SSVQE Hydrogen Molecule")
plt.xlabel("Interatomic Distance (Angstroms)")
plt.ylabel("Energy (Ha)")
plt.show()

deviation = all_energies - all_exact_energies
print("SSVQE Error: "+str(deviation))
for energy_level in range(energy_levels):
    plt.scatter(hhdis_ref, deviation[:, energy_level], c=colors[energy_level], label=labels[energy_level])
plt.legend()
plt.plot(hhdis_ref, np.zeros(len(hhdis_ref)), c='g', linestyle="dashed")
plt.title("SSVQE Hydrogen Error")
plt.xlabel("Interatomic Distance (Angstroms)")
plt.ylabel("Energy (Ha)")
plt.show()

print("Max Error (Ha): "+str(np.amax(np.abs(deviation))))
print("Mean Error (Ha): "+str(np.mean(np.abs(deviation))))
print("Results Saved!")