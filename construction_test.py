from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from log_3_mcx_x_cx_ccx import log3_cnx, classical_gates_circuit
from qiskit import QuantumCircuit, Aer, execute
import numpy as np

def test_with_unitary_simulator():
    backend = Aer.get_backend('unitary_simulator')
    nmax = 10
    count_correct_solutions = 0
    for ncontrol in range(1, nmax+1):
        nqubits = ncontrol + 2
        # log3 circuit
        log3_qc = classical_gates_circuit(ncontrol)
        job = execute(log3_qc, backend)
        res = job.result().get_unitary(log3_qc, 2)
        # verify
        qc = QuantumCircuit(nqubits)
        qc.mcx(control_qubits=list(range(ncontrol)), target_qubit=nqubits - 1)
        job = execute(qc, backend)
        res_verify = job.result().get_unitary(qc, 2)
        if res == res_verify:
            count_correct_solutions += 1
    assert count_correct_solutions == nmax

def test_random_computational_basis_state_with_statevector_simulator():
    backend = Aer.get_backend('aer_simulator')
    nc = 100
    nqubits = nc+2
    circ = QuantumCircuit(nqubits)
    init_state = ''
    for k in range(nc):
        u = np.random.randint(2)
        init_state += str(u)
    init_state += '0'
    if init_state[:nc] == '1'*nc:
        init_state += '1'
    else:
        init_state += '0'
    circ.x([i for i in range(nc) if init_state[i] == '1'])
    gate = log3_cnx(ncontrol=nc)
    circ.append(gate, list(range(nqubits)))
    circ = transpile(circ, basis_gates=['x', 'cx', 'ccx'])
    circ.measure_all()
    result = backend.run(circ).result()
    counts = result.get_counts(circ)
    assert list(counts.keys())[0] == init_state[::-1]

def test_all_ones_control_with_statevector_simulator():
    backend = Aer.get_backend('aer_simulator')
    nc = 100
    nqubits = nc+2
    circ = QuantumCircuit(nqubits)
    init_state = '1'*nc+'01' # the last bit is set to 1 since it is supposed to be flipped by the gate
    circ.x([i for i in range(nc) if init_state[i] == '1'])
    gate = log3_cnx(ncontrol=nc)
    circ.append(gate, list(range(nqubits)))
    circ = transpile(circ, basis_gates=['x', 'cx', 'ccx'])
    circ.measure_all()
    result = backend.run(circ).result()
    counts = result.get_counts(circ)
    assert list(counts.keys())[0] == init_state[::-1]
