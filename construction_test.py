'''
We thank the anonymous referee who provided this test file.
'''
import collections
import itertools
import random
import numpy as np
import pytest
import qiskit
from qiskit.compiler import transpile
from log_3_mcx_x_cx_ccx import log3_cnx

def bulk_simulate_result_of_applying_classical_circuit_to(circuit: qiskit.QuantumCircuit, input_states: np.ndarray) -> np.ndarray:
    states = np.copy(input_states)
    buffer = np.zeros_like(states[0])
    for instruction in circuit:
        if instruction.operation.name == 'x':
            assert len(instruction.qubits) == 1
            q = instruction.qubits[0]._index
            np.bitwise_not(states[q], out=states[q])
        elif instruction.operation.name == 'cx':
            assert len(instruction.qubits) == 2
            c = instruction.qubits[0]._index
            t = instruction.qubits[1]._index
            states[t] ^= states[c]
        elif instruction.operation.name == 'ccx':
            assert len(instruction.qubits) == 3
            a = instruction.qubits[0]._index
            b = instruction.qubits[1]._index
            t = instruction.qubits[2]._index
            np.bitwise_and(states[a], states[b], out=buffer)
            states[t] ^= buffer
        elif instruction.operation.name == 'barrier':
            pass
        elif instruction.operation.name == 'measure':
            pass
        else:
            raise NotImplementedError(f'{instruction=}')
    return states

def compute_depth(circuit: qiskit.QuantumCircuit) -> int:
    qubit_depth = collections.defaultdict(int)
    known = ['x', 'cx', 'ccx']
    for instruction in circuit:
        if instruction.operation.name in known:
            qs = [q._index for q in instruction.qubits]
            layer = max(qubit_depth[q] for q in qs) + 1
            for q in qs:
                qubit_depth[q] = layer
        elif instruction.operation.name == 'barrier':
            pass
        elif instruction.operation.name == 'measure':
            pass
        else:
            raise NotImplementedError(f'{instruction=}')
    return max(qubit_depth.values())

@pytest.mark.parametrize('num_controls', [10, 100, 200, 500, 1000])
def test_depth(num_controls: int):
    num_qubits = num_controls + 2
    gate = log3_cnx(ncontrol=num_controls)
    circuit = qiskit.QuantumCircuit(num_qubits)
    circuit.append(gate, list(range(num_qubits)))
    circuit = transpile(circuit, basis_gates=['x', 'cx', 'ccx'])
    depth = compute_depth(circuit)
    lg_n = num_controls.bit_length()
    expected = 27 * lg_n**3 - 808
    assert depth <= expected

@pytest.mark.parametrize('num_controls', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 32, 64, 100])
def test_fuzz_random_cases(num_controls: int):
    num_samples = 1024
    num_qubits = num_controls + 2
    target_index = num_controls + 1
    gate = log3_cnx(ncontrol=num_controls)
    circuit = qiskit.QuantumCircuit(num_qubits)
    circuit.append(gate, list(range(num_qubits)))
    circuit = transpile(circuit, basis_gates=['x', 'cx', 'ccx'])
    input_states = np.random.randint(low=0, high=(1 << 64) - 1, size=(num_qubits, num_samples // 64), dtype=np.uint64,)
    controls_satisfied = ~np.zeros_like(input_states[0])
    for k in range(num_controls):
        controls_satisfied &= input_states[k]
    expected = np.copy(input_states)
    expected[target_index] ^= controls_satisfied
    actual = bulk_simulate_result_of_applying_classical_circuit_to(circuit, input_states)
    assert np.array_equal(actual, expected)

@pytest.mark.parametrize('num_controls', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 32, 64, 100])
@pytest.mark.parametrize('num_off_controls', [0, 1, 2])
def test_low_weight_cases(num_controls: int, num_off_controls):
    num_qubits = num_controls + 2
    target_index = num_controls + 1
    ancilla_index = num_controls
    gate = log3_cnx(ncontrol=num_controls)
    circuit = qiskit.QuantumCircuit(num_qubits)
    circuit.append(gate, list(range(num_qubits)))
    circuit = transpile(circuit, basis_gates=['x', 'cx', 'ccx'])
    cases = list(itertools.combinations(range(num_controls), num_off_controls))
    input_states = np.ones(
    shape=(num_qubits, len(cases)), dtype=np.bool_,)
    for shot_index, on_bits in enumerate(cases):
        for q in on_bits:
            input_states[q, shot_index] = False
        input_states[ancilla_index, shot_index] = random.random() < 0.5
        input_states[target_index, shot_index] = random.random() < 0.5
    input_states = np.packbits(input_states, axis=1)
    controls_satisfied = ~np.zeros_like(input_states[0])
    for q in range(num_controls):
        controls_satisfied &= input_states[q]
    expected = np.copy(input_states)
    expected[target_index] ^= controls_satisfied
    actual = bulk_simulate_result_of_applying_classical_circuit_to(circuit, input_states)
    assert np.array_equal(actual, expected)