import numpy as np
from qiskit import QuantumCircuit
import qiskit
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
shots = 2**24

step_size = 0.6 # Step Size for theta_dot
max_iter = 4000

global Ham

#Load Hamiltonian from dict
ham_dict = {'IIII': -1.866761025, 'IIIX': 0.0081555637, 'IIIY': 0.0, 'IIIZ': -1.5062380875, 'IIXI': 0.00871086576375, 'IIXX': 0.0008161726125, 'IIXY': 0.0, 'IIXZ': 0.00871086576375, 'IIYI': 0.0, 'IIYX': 0.0, 'IIYY': 0.0008161726125, 'IIYZ': 0.0, 'IIZI': -0.25037708374999995, 'IIZX': 0.0081555637, 'IIZY': 0.0, 'IIZZ': 0.11014585375000002, 'IXII': -0.0013020425375000005, 'IXIX': 0.00768880275, 'IXIY': 0.0, 'IXIZ': -0.0013020425375000005, 'IXXI': 0.007460543738749999, 'IXXX': 0.00768880275, 'IXXY': 0.0, 'IXXZ': 0.007460543738749999, 'IXYI': 0.0, 'IXYX': 0.0, 'IXYY': 0.00768880275, 'IXYZ': 0.0, 'IXZI': -0.0031941684400000002, 'IXZX': 0.00768880275, 'IXZY': 0.0, 'IXZZ': -0.0031941684400000002, 'IYII': 0.0, 'IYIX': 0.0, 'IYIY': 0.00768880275, 'IYIZ': 0.0, 'IYXI': 0.0, 'IYXX': 0.0, 'IYXY': 0.00768880275, 'IYXZ': 0.0, 'IYYI': 0.00267351721375, 'IYYX': -0.00768880275, 'IYYY': 0.0, 'IYYZ': 0.00267351721375, 'IYZI': 0.0, 'IYZX': 0.0, 'IYZY': 0.00768880275, 'IYZZ': 0.0, 'IZII': -0.19311867874999994, 'IZIX': 0.0081555637, 'IZIY': 0.0, 'IZIZ': 0.16740425875000003, 'IZXI': -0.007860042313750001, 'IZXX': 0.0008161726125, 'IZXY': 0.0, 'IZXZ': -0.007860042313750001, 'IZYI': 0.0, 'IZYX': 0.0, 'IZYY': 0.0008161726125, 'IZYZ': 0.0, 'IZZI': -0.20329534, 'IZZX': 0.0081555637, 'IZZY': 0.0, 'IZZZ': 0.15722759749999998, 'XIII': 0.0016256045454999999, 'XIIX': -0.0003783041275, 'XIIY': 0.0, 'XIIZ': 0.0016256045454999999, 'XIXI': 0.0025132852675, 'XIXX': 0.00132143895, 'XIXY': 0.0, 'XIXZ': 0.0025132852675, 'XIYI': 0.0, 'XIYX': 0.0, 'XIYY': 0.00132143895, 'XIYZ': 0.0, 'XIZI': -0.0016441700355, 'XIZX': -0.0003783041275, 'XIZY': 0.0, 'XIZZ': -0.0016441700355, 'XXII': 0.00290167052, 'XXIX': -0.0003783041275, 'XXIY': 0.0, 'XXIZ': 0.00290167052, 'XXXI': 0.001237219293, 'XXXX': 0.00132143895, 'XXXY': 0.0, 'XXXZ': 0.001237219293, 'XXYI': 0.0, 'XXYX': 0.0, 'XXYY': 0.00132143895, 'XXYZ': 0.0, 'XXZI': 0.00044455825000000015, 'XXZX': -0.0003783041275, 'XXZY': 0.0, 'XXZZ': 0.00044455825000000015, 'XYII': 0.0, 'XYIX': 0.0, 'XYIY': -0.0003783041275, 'XYIZ': 0.0, 'XYXI': 0.0, 'XYXX': 0.0, 'XYXY': 0.00132143895, 'XYXZ': 0.0, 'XYYI': 0.0005418288130000001, 'XYYX': -0.00132143895, 'XYYY': 0.0, 'XYYZ': 0.0005418288130000001, 'XYZI': 0.0, 'XYZX': 0.0, 'XYZY': -0.0003783041275, 'XYZZ': 0.0, 'XZII': 0.0012218552494999999, 'XZIX': -0.0003783041275, 'XZIY': 0.0, 'XZIZ': 0.0012218552494999999, 'XZXI': -0.00044259597749999983, 'XZXX': 0.00132143895, 'XZXY': 0.0, 'XZXZ': -0.00044259597749999983, 'XZYI': 0.0, 'XZYX': 0.0, 'XZYY': 0.00132143895, 'XZYZ': 0.0, 'XZZI': -0.0019358112195, 'XZZX': -0.0003783041275, 'XZZY': 0.0, 'XZZZ': -0.0019358112195, 'YIII': 0.0, 'YIIX': 0.0, 'YIIY': -0.0003783041275, 'YIIZ': 0.0, 'YIXI': 0.0, 'YIXX': 0.0, 'YIXY': 0.00132143895, 'YIXZ': 0.0, 'YIYI': 0.0029221982825, 'YIYX': -0.00132143895, 'YIYY': 0.0, 'YIYZ': 0.0029221982825, 'YIZI': 0.0, 'YIZX': 0.0, 'YIZY': -0.0003783041275, 'YIZZ': 0.0, 'YXII': 0.0, 'YXIX': 0.0, 'YXIY': -0.0003783041275, 'YXIZ': 0.0, 'YXXI': 0.0, 'YXXX': 0.0, 'YXXY': 0.00132143895, 'YXXZ': 0.0, 'YXYI': 0.000833469997, 'YXYX': -0.00132143895, 'YXYY': 0.0, 'YXYZ': 0.000833469997, 'YXZI': 0.0, 'YXZX': 0.0, 'YXZY': -0.0003783041275, 'YXZZ': 0.0, 'YYII': 5.421072499999999e-05, 'YYIX': 0.0003783041275, 'YYIY': 0.0, 'YYIZ': 5.421072499999999e-05, 'YYXI': -0.0008334699970000001, 'YYXX': -0.00132143895, 'YYXY': 0.0, 'YYXZ': -0.0008334699970000001, 'YYYI': 0.0, 'YYYX': 0.0, 'YYYY': -0.00132143895, 'YYYZ': 0.0, 'YYZI': 0.004024539505, 'YYZX': 0.0003783041275, 'YYZY': 0.0, 'YYZZ': 0.004024539505, 'YZII': 0.0, 'YZIX': 0.0, 'YZIY': -0.0003783041275, 'YZIZ': 0.0, 'YZXI': 0.0, 'YZXX': 0.0, 'YZXY': 0.00132143895, 'YZXZ': 0.0, 'YZYI': -0.0015468994725, 'YZYX': -0.00132143895, 'YZYY': 0.0, 'YZYZ': -0.0015468994725, 'YZZI': 0.0, 'YZZX': 0.0, 'YZZY': -0.0003783041275, 'YZZZ': 0.0, 'ZIII': -0.16674459249999996, 'ZIIX': 0.0081555637, 'ZIIY': 0.0, 'ZIIZ': 0.19377834500000002, 'ZIXI': 0.008191304408749999, 'ZIXX': 0.0008161726125, 'ZIXY': 0.0, 'ZIXZ': 0.008191304408749999, 'ZIYI': 0.0, 'ZIYX': 0.0, 'ZIYY': 0.0008161726125, 'ZIYZ': 0.0, 'ZIZI': -0.15621359625000003, 'ZIZX': 0.0081555637, 'ZIZY': 0.0, 'ZIZZ': 0.20430934124999994, 'ZXII': -0.003525430535, 'ZXIX': 0.00768880275, 'ZXIY': 0.0, 'ZXIZ': -0.003525430535, 'ZXXI': -0.01228801681125, 'ZXXX': 0.00768880275, 'ZXXY': 0.0, 'ZXXZ': -0.01228801681125, 'ZXYI': 0.0, 'ZXYX': 0.0, 'ZXYY': 0.00768880275, 'ZXYZ': 0.0, 'ZXZI': -0.0021528659875, 'ZXZX': 0.00768880275, 'ZXZY': 0.0, 'ZXZZ': -0.0021528659875, 'ZYII': 0.0, 'ZYIX': 0.0, 'ZYIY': 0.00768880275, 'ZYIZ': 0.0, 'ZYXI': 0.0, 'ZYXX': 0.0, 'ZYXY': 0.00768880275, 'ZYXZ': 0.0, 'ZYYI': 0.00267351721375, 'ZYYX': -0.00768880275, 'ZYYY': 0.0, 'ZYYZ': 0.00267351721375, 'ZYZI': 0.0, 'ZYZX': 0.0, 'ZYZY': 0.00768880275, 'ZYZZ': 0.0, 'ZZII': -0.19311867875, 'ZZIX': 0.0081555637, 'ZZIY': 0.0, 'ZZIZ': 0.16740425874999998, 'ZZXI': -0.00786004231375, 'ZZXX': 0.0008161726125, 'ZZXY': 0.0, 'ZZXZ': -0.00786004231375, 'ZZYI': 0.0, 'ZZYX': 0.0, 'ZZYY': 0.0008161726125, 'ZZYZ': 0.0, 'ZZZI': -0.20329534, 'ZZZX': 0.0081555637, 'ZZZY': 0.0, 'ZZZZ': 0.15722759749999998}
print("Constructing Hamiltonian from dict...")
ham_list = []
for key in ham_dict.keys():
    ham_list.append((key, ham_dict[key]))
H = SparsePauliOp.from_list(ham_list)

def SSVQE(params):
    global ansatz
    global H
    global iterations
    hamiltonian = H
    estimator = Estimator()#options={"shots":2**20})
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
    print(f"Current Loss: {loss}")
    return loss


def SSVQE_Energy(params):
    global ansatz
    global H
    global iterations
    hamiltonian = H
    estimator = Estimator()#options={"shots":2**20})
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
    print(f"Energy: {energies}]")
    return energies

global ansatz

#'''# Exact Diagonalization
print("Exactly Solving With Numpy...")
exact_solver = NumPyEigensolver(k=energy_levels)
exact_result = exact_solver.compute_eigenvalues(H)
exact_solution = exact_result.eigenvalues
#exact_solution = np.linalg.eig(H.to_matrix()).eigenvalues
print(exact_solution)
print()
#'''

#''' Run SSVQE
from scipy.optimize import minimize
ansatz = TwoLocal(4, ["rx", "ry"], "cz", reps=4)
x0 = 2*np.pi*np.random.random(ansatz.num_parameters) - np.pi # Random Initial Parameters
res = minimize(SSVQE, x0, method="BFGS", options={"maxiter": 300})
#res = minimize(SSVQE, x0, method="cobyla")#, options={"maxiter":300})
print("SSVQE Result!")
print(SSVQE_Energy(res.x))
print("Exact Result!")
print(exact_solution)
#'''