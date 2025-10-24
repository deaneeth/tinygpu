# src/tinygpu/instructions.py

def _resolve(gpu, tid, operand):
    """
    operand is either:
      - ("R", idx)  -> return value from registers[tid, idx]
      - int         -> immediate (return int)
      - str         -> label (not resolved here)
    Always returns a plain Python int when possible.
    """
    if isinstance(operand, tuple) and operand[0] == "R":
        idx = int(operand[1])
        return int(gpu.registers[tid, idx])
    elif isinstance(operand, int):
        return int(operand)
    else:
        # For labels etc - return as-is (caller must handle)
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


# dispatch
INSTRUCTIONS = {
    "SET": op_set,
    "ADD": op_add,
    "MUL": op_mul,
    "LD": op_ld,
    "ST": op_st,
}
