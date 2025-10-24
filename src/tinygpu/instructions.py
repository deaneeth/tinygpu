def _resolve(gpu, tid, operand):
    if isinstance(operand, tuple) and operand[0] == "R":
        return int(gpu.registers[tid, operand[1]])
    elif isinstance(operand, int):
        return int(operand)
    else:
        return operand

def op_set(gpu, tid, rd_operand, imm_operand):
    # rd_operand must be a register operand ("R", idx)
    if not (isinstance(rd_operand, tuple) and rd_operand[0] == "R"):
        raise TypeError("SET target must be a register")
    rd = rd_operand[1]
    gpu.registers[tid, rd] = _resolve(gpu, tid, imm_operand)

def op_add(gpu, tid, rd_operand, op1, op2):
    if not (isinstance(rd_operand, tuple) and rd_operand[0] == "R"):
        raise TypeError("ADD target must be a register")
    rd = rd_operand[1]
    v1 = _resolve(gpu, tid, op1)
    v2 = _resolve(gpu, tid, op2)
    gpu.registers[tid, rd] = v1 + v2

def op_mul(gpu, tid, rd_operand, op1, op2):
    if not (isinstance(rd_operand, tuple) and rd_operand[0] == "R"):
        raise TypeError("MUL target must be a register")
    rd = rd_operand[1]
    v1 = _resolve(gpu, tid, op1)
    v2 = _resolve(gpu, tid, op2)
    gpu.registers[tid, rd] = v1 * v2

def op_ld(gpu, tid, rd_operand, addr_operand):
    if not (isinstance(rd_operand, tuple) and rd_operand[0] == "R"):
        raise TypeError("LD destination must be a register")
    rd = rd_operand[1]
    a = _resolve(gpu, tid, addr_operand)
    a = int(a)
    gpu.registers[tid, rd] = int(gpu.memory[a])

def op_st(gpu, tid, addr_operand, rs_operand):
    a = _resolve(gpu, tid, addr_operand)
    a = int(a)
    val = _resolve(gpu, tid, rs_operand)
    gpu.memory[a] = int(val)

def op_jmp(gpu, tid, target):
    gpu.pc[tid] = _resolve(gpu, tid, target)

def op_beq(gpu, tid, r1, r2, target):
    if _resolve(gpu, tid, r1) == _resolve(gpu, tid, r2):
        gpu.pc[tid] = _resolve(gpu, tid, target)
    else:
        gpu.pc[tid] += 1

def op_bne(gpu, tid, r1, r2, target):
    if _resolve(gpu, tid, r1) != _resolve(gpu, tid, r2):
        gpu.pc[tid] = _resolve(gpu, tid, target)
    else:
        gpu.pc[tid] += 1

def op_sync(gpu, tid):
    # crude barrier: wait until all threads reach this PC
    target_pc = gpu.pc[tid]
    if all(p == target_pc for p in gpu.pc):
        gpu.pc[:] = target_pc + 1  # release barrier

INSTRUCTIONS = {
    "SET": op_set,
    "ADD": op_add,
    "MUL": op_mul,
    "LD":  op_ld,
    "ST":  op_st,
    "JMP": op_jmp,
    "BEQ": op_beq,
    "BNE": op_bne,
    "SYNC": op_sync,
}