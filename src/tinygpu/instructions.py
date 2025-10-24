def _resolve(gpu, tid, operand):
    if isinstance(operand, tuple) and operand[0] == "R":
        return int(gpu.registers[tid, operand[1]])
    elif isinstance(operand, int):
        return int(operand)
    else:
        return operand

def op_set(gpu, tid, rd_operand, imm_operand):
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
    gpu.registers[tid, rd] = int(v1 + v2)

def op_mul(gpu, tid, rd_operand, op1, op2):
    if not (isinstance(rd_operand, tuple) and rd_operand[0] == "R"):
        raise TypeError("MUL target must be a register")
    rd = rd_operand[1]
    v1 = _resolve(gpu, tid, op1)
    v2 = _resolve(gpu, tid, op2)
    gpu.registers[tid, rd] = int(v1 * v2)

def op_ld(gpu, tid, rd_operand, addr_operand):
    if not (isinstance(rd_operand, tuple) and rd_operand[0] == "R"):
        raise TypeError("LD destination must be a register")
    rd = rd_operand[1]
    a = _resolve(gpu, tid, addr_operand)
    a = int(a)
    gpu.registers[tid, rd] = int(gpu.memory[a])

def op_st(gpu, tid, addr_operand, rs_operand):
    a = int(_resolve(gpu, tid, addr_operand))
    val = int(_resolve(gpu, tid, rs_operand))
    gpu.memory[a] = val
    
 # control flow ops

def op_jmp(gpu, tid, target):
    # set PC to target (target expected to be an immediate int)
    gpu.pc[tid] = int(_resolve(gpu, tid, target))

def op_beq(gpu, tid, op1, op2, target):
    if _resolve(gpu, tid, op1) == _resolve(gpu, tid, op2):
        gpu.pc[tid] = int(_resolve(gpu, tid, target))
    else:
        # increment will be handled by core step (if unchanged)
        gpu.pc[tid] = int(gpu.pc[tid])

def op_bne(gpu, tid, op1, op2, target):
    if _resolve(gpu, tid, op1) != _resolve(gpu, tid, op2):
        gpu.pc[tid] = int(_resolve(gpu, tid, target))
    else:
        gpu.pc[tid] = int(gpu.pc[tid])

def op_sync(gpu, tid):
    """
    Barrier: mark this thread as waiting. Core will release all waiting
    threads when every active thread reaches a sync (basic implementation).
    """
    gpu.sync_waiting[tid] = True
    # do not advance PC here; core will advance the waiting threads when all reached

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