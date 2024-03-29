from qiskit import QuantumCircuit
from qiskit.compiler import transpile
import numpy as np

print('The circuits are compiled in the [u, cx] basis.')
log_3_memory = {}
log_3_size = {}
log_3_depth = {}
log_3_depth[1] = 1
log_3_size[1] = 1
qc = QuantumCircuit(3)
qc.cx(0, 2)
log_3_memory[1] = qc.to_gate()
for nc in range(2, 31): #for testing, take n=5. Otherwise, as high as you want
    qc = QuantumCircuit.from_qasm_file('initialise_log3/log3_'+str(nc)+'.qasm')
    log_3_memory[nc] = qc.to_gate()
    log_3_depth[nc] = qc.depth()
    log_3_size[nc] = qc.size()

def one_ancilla_mcx(ncontrol):
    even = [k for k in range(ncontrol) if k % 2 == 0]
    odd = [k for k in range(ncontrol) if k % 2 == 1]
    qc = QuantumCircuit(ncontrol+2)
    qc.mcx(even, ncontrol+1, odd+[ncontrol], mode='v-chain-dirty')
    qc.mcx(odd+[ncontrol+1], ncontrol, even, mode='v-chain-dirty')
    qc.mcx(even, ncontrol + 1, odd + [ncontrol], mode='v-chain-dirty')
    qc.mcx(odd+[ncontrol+1], ncontrol, even + [ncontrol + 1], mode='v-chain-dirty')
    return qc.to_gate()

def apply_control(circuit, black_controls, white_controls, ancilla, target, gates):
    all_controls = white_controls+black_controls
    if len(white_controls)>0:
        circuit.x(white_controls)
    circuit.append(gates[len(all_controls)], all_controls+[ancilla]+[target])
    if len(white_controls)>0:
        circuit.x(white_controls)
    return circuit

def log3_construct_gates(registers, R0star, trace_depth_and_size = False):
    gates = {}
    gates[len(R0star)+1] = log3_cnx(len(R0star)+1, trace_depth_and_size)
    for k in range(len(registers)):
        if not(len(registers[k]) in gates.keys()):
            gates[len(registers[k])] = log3_cnx(len(registers[k]), trace_depth_and_size)
    return gates

def log3_cnx(ncontrol, trace_depth_and_size = False):
    if ncontrol in log_3_memory.keys():
        return log_3_memory[ncontrol]
    else:
        registers, R0star, R0b = construct_log3_registers(ncontrol)
        b = len(R0star)
        gates = log3_construct_gates(registers, R0star, trace_depth_and_size)
        nqubits = ncontrol+2
        target = ncontrol+1
        ancilla = ncontrol
        qc = QuantumCircuit(nqubits)
        for _ in range(2):
            # C_{R_0}^a
            qc = apply_control(circuit=qc, black_controls=registers[0], white_controls=[], ancilla=target, target=ancilla,
                               gates=gates)
            # prod i
            for i in range(1, b+1):
                qc = apply_control(circuit=qc, black_controls=registers[i], white_controls=[],
                                   ancilla=R0b[i-1], target=R0star[i-1], gates=gates)
            # prod C_{\overline{R_0^*}\cup a}^t
            qc = apply_control(circuit=qc, black_controls=[ancilla], white_controls=R0star,
                               ancilla=R0b[0], target=target, gates=gates)
            # prod i
            for i in range(1, b+1):
                qc = apply_control(circuit=qc, black_controls=registers[i], white_controls=[],
                                   ancilla=R0b[i-1], target=R0star[i-1], gates=gates)
    gate = qc.to_gate()
    log_3_memory[ncontrol] = gate
    if trace_depth_and_size:
        depth = 2*(log_3_depth[len(registers[0])]+2*log_3_depth[len(registers[1])]+log_3_depth[len(R0star)+1])
        log_3_depth[ncontrol] = depth
    return gate

def construct_log3_registers(ncontrol):
    p = int(np.sqrt(ncontrol))
    R0 = list(range(2*p))
    registers = [R0]
    index = 2*p
    for i in range(1, p+1):
        locreg = []
        for _ in range(p):
            if index < ncontrol:
                locreg.append(index)
                index += 1
        if len(locreg) == 0:
            break
        else:
            registers.append(locreg)
    b = len(registers)-1
    R0star = list(range(b))
    R0b = list(range(b, 2*p))
    return registers, R0star, R0b

def access_polylog_depth(nc):
    if nc in log_3_depth.keys():
        return log_3_depth[nc]
    else:
        print('Such gate has not been compiled yet.')

def access_polylog_size(nc):
    if nc in log_3_size.keys():
        return log_3_size[nc]
    else:
        print('Such gate has not been compiled yet.')