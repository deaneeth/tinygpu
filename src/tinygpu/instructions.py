def op_ld(gpu, tid, rd, addr):
    gpu.registers[tid, rd] = gpu.memory[addr]

def op_st(gpu, tid, addr, rs):
    gpu.memory[addr] = gpu.registers[tid, rs]

def op_add(gpu, tid, rd, rs1, rs2):
    gpu.registers[tid, rd] = gpu.registers[tid, rs1] + gpu.registers[tid, rs2]

def op_mul(gpu, tid, rd, rs1, rs2):
    gpu.registers[tid, rd] = gpu.registers[tid, rs1] * gpu.registers[tid, rs2]

def op_set(gpu, tid, rd, imm):
    gpu.registers[tid, rd] = imm

# Instruction dispatch table
INSTRUCTIONS = {
    "LD": op_ld,
    "ST": op_st,
    "ADD": op_add,
    "MUL": op_mul,
    "SET": op_set,
}
