import numpy as np
from qiskit import QuantumCircuit
from qiskit.compiler import transpile

def one_ancilla_mcx(ncontrol):
    even = [k for k in range(ncontrol) if k % 2 == 0]
    odd = [k for k in range(ncontrol) if k % 2 == 1]
    qc = QuantumCircuit(ncontrol+2)
    qc.mcx(even, ncontrol+1, odd+[ncontrol], mode='v-chain-dirty')
    qc.mcx(odd+[ncontrol+1], ncontrol, even, mode='v-chain-dirty')
    qc.mcx(even, ncontrol + 1, odd + [ncontrol], mode='v-chain-dirty')
    qc.mcx(odd+[ncontrol+1], ncontrol, even + [ncontrol + 1], mode='v-chain-dirty')
    return qc.to_gate()

polylog_memory = {} # Dynamic Programming
polylog_depth = {}
polylog_size = {}
qc = QuantumCircuit(3)
qc.cx(0, 2)
polylog_depth[1] = 1
polylog_size[1] = 1
polylog_memory[1] = qc.to_gate()
for nc in range(2, 25): #for testing, take n=5. Otherwise, as high as you want
    qc = QuantumCircuit(nc+2)
    gate = one_ancilla_mcx(nc)
    qc.append(gate, list(range(nc))+[nc+1, nc])
    qc = transpile(qc, basis_gates=['u', 'cx'], optimization_level=3)
    polylog_memory[nc] = qc.to_gate()
    polylog_depth[nc] = qc.depth()
    polylog_size[nc] = qc.size()

def construct_registers(ncontrol):
    p = int(np.floor(np.sqrt(ncontrol)))
    if ncontrol <= p*(p+1):
        regsize = p
    else:
        regsize = p+1
    placed = 0
    registers = []
    while placed < ncontrol:
        new_reg = []
        for k in range(regsize):
            if placed < ncontrol:
                new_reg.append(placed)
                placed += 1
            else:
                break
        registers.append(new_reg)
    return registers

def construct_gates(registers, trace_depth_and_size = False):
    gates = {}
    gates[len(registers[0])+1] = polylog_CnX(len(registers[0])+1, trace_depth_and_size)
    for k in range(len(registers)):
        if not(len(registers[k]) in gates.keys()):
            gates[len(registers[k])] = polylog_CnX(len(registers[k]), trace_depth_and_size)
    return gates

def associate_ancillae(registers):
    even_regs = [registers[i] for i in range(2, len(registers)) if i % 2 == 0]
    odd_regs = [registers[i] for i in range(1, len(registers)) if i % 2 == 1]
    even_ancillae = []
    c = 0
    while len(even_ancillae) < len(even_regs):
        local = odd_regs[c]
        for l in local:
            even_ancillae.append(l)
        c += 1
    odd_ancillae = []
    c = 0
    while len(odd_ancillae) < len(odd_regs):
        local = even_regs[c]
        for l in local:
            odd_ancillae.append(l)
        c += 1
    ancillae = [0]
    for k in range(1, len(registers)):
        if k % 2 == 0:
            ancillae.append(even_ancillae[0])
            even_ancillae.pop(0)
        if k % 2 == 1:
            ancillae.append(odd_ancillae[0])
            odd_ancillae.pop(0)
    return ancillae

def apply_control(circuit, black_controls, white_controls, ancilla, target, gates):
    all_controls = white_controls+black_controls
    if len(white_controls)>0:
        circuit.x(white_controls)
    circuit.append(gates[len(all_controls)], all_controls+[ancilla]+[target])
    if len(white_controls)>0:
        circuit.x(white_controls)
    return circuit

def polylog_CnX(ncontrol, trace_depth_and_size = False):
    if ncontrol in polylog_memory.keys():
        return polylog_memory[ncontrol]
    else:
        registers = construct_registers(ncontrol = ncontrol)
        intermediate_ancillae = associate_ancillae(registers)
        gates = construct_gates(registers, trace_depth_and_size)
        nqubits = ncontrol+2
        target = ncontrol+1
        ancilla = ncontrol
        qc = QuantumCircuit(nqubits)
        # C_{R_0}^a
        qc = apply_control(circuit=qc, black_controls=registers[0], white_controls=[], ancilla=target, target=ancilla,
                           gates=gates)
        # prod i even
        for i in range(2, len(registers), 2):
            qc = apply_control(circuit=qc, black_controls=registers[i], white_controls=[],
                               ancilla=intermediate_ancillae[i], target=i-1, gates=gates)
        # prod i odd
        for i in range(1, len(registers), 2):
            qc = apply_control(circuit=qc, black_controls=registers[i], white_controls=[],
                               ancilla=intermediate_ancillae[i], target=i-1, gates=gates)
        # prod C_{\overline{R_0}\cup a}^t
        qc = apply_control(circuit=qc, black_controls=[ancilla], white_controls=registers[0][:len(registers)-1],
                           ancilla=intermediate_ancillae[1], target=target, gates=gates)
        # prod i even
        for i in range(2, len(registers), 2):
            qc = apply_control(circuit=qc, black_controls=registers[i], white_controls=[],
                               ancilla=intermediate_ancillae[i], target=i-1, gates=gates)
        # prod i odd
        for i in range(1, len(registers), 2):
            qc = apply_control(circuit=qc, black_controls=registers[i], white_controls=[],
                               ancilla=intermediate_ancillae[i], target=i-1, gates=gates)
        # C_{R_0}^a
        qc = apply_control(circuit=qc, black_controls=registers[0], white_controls=[], ancilla=target, target=ancilla,
                           gates=gates)
        # prod i even
        for i in range(2, len(registers), 2):
            qc = apply_control(circuit=qc, black_controls=registers[i], white_controls=[],
                               ancilla=intermediate_ancillae[i], target=i-1, gates=gates)
        # prod i odd
        for i in range(1, len(registers), 2):
            qc = apply_control(circuit=qc, black_controls=registers[i], white_controls=[],
                               ancilla=intermediate_ancillae[i], target=i-1, gates=gates)
        # prod C_{\overline{R_0}\cup a}^t
        qc = apply_control(circuit=qc, black_controls=[ancilla], white_controls=registers[0][:len(registers)-1],
                           ancilla=intermediate_ancillae[1], target=target, gates=gates)
        # prod i even
        for i in range(2, len(registers), 2):
            qc = apply_control(circuit=qc, black_controls=registers[i], white_controls=[],
                               ancilla=intermediate_ancillae[i], target=i-1, gates=gates)
        # prod i odd
        for i in range(1, len(registers), 2):
            qc = apply_control(circuit=qc, black_controls=registers[i], white_controls=[],
                               ancilla=intermediate_ancillae[i], target=i-1, gates=gates)
    gate = qc.to_gate()
    polylog_memory[ncontrol] = gate
    if trace_depth_and_size:
        d0 = len(registers[0])
        deven = max([len(registers[i]) for i in range(2, len(registers), 2)])
        dodd = max([len(registers[i]) for i in range(1, len(registers), 2)])
        d2 = d0+1
        total_depth = 2 * polylog_depth[d0] + 4 * polylog_depth[deven] + 4 * polylog_depth[dodd] + 2 * polylog_depth[
            d2] + 4
        total_size = 2 * polylog_size[d0] + 4 * sum(
            [polylog_size[len(registers[i])] for i in range(1, len(registers))]) + 2 * polylog_size[d2] + 4 * (
                                 len(registers) - 1)

        polylog_size[ncontrol] = total_size
        polylog_depth[ncontrol] = total_depth
    return gate

def access_polylog_depth(nc):
    if nc in polylog_depth.keys():
        return polylog_depth[nc]
    else:
        print('Such gate has not been compiled yet.')

def access_polylog_size(nc):
    if nc in polylog_size.keys():
        return polylog_size[nc]
    else:
        print('Such gate has not been compiled yet.')

