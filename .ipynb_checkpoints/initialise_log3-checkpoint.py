from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from polylog_mcx import one_ancilla_mcx

for nc in range(2, 31): #for testing, take n=5. Otherwise, as high as you want
    print('Optimizing', nc)
    qc = QuantumCircuit(nc+2)
    gate = one_ancilla_mcx(nc)
    qc.append(gate, list(range(nc))+[nc+1, nc])
    qc = transpile(qc, basis_gates=['u', 'cx'], optimization_level=3)
    qc.qasm(filename='initialise_log3/log3_'+str(nc)+'.qasm')