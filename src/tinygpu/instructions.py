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


def op_cswap(gpu, tid, addr_a_operand, addr_b_operand):
    """
    Compare-and-swap on global memory:
      if memory[a] > memory[b]: swap memory[a], memory[b]
    addr operands can be immediates or register operands.
    """
    a = int(_resolve(gpu, tid, addr_a_operand))
    b = int(_resolve(gpu, tid, addr_b_operand))

    # bounds check (defensive)
    if a < 0 or a >= gpu.mem_size or b < 0 or b >= gpu.mem_size:
        return

    va = int(gpu.memory[a])
    vb = int(gpu.memory[b])
    if va > vb:
        gpu.memory[a], gpu.memory[b] = vb, va

# Flags helper: set bitmask in gpu.flags[tid]
def _set_flags_from_result(gpu, tid, diff):
    """
    diff = a - b
    sets flags bitmask:
      bit0 (1): Z (zero) if diff == 0
      bit1 (2): N (negative) if diff <  0
      bit2 (4): G (greater)  if diff >  0
    """
    z = 1 if diff == 0 else 0
    n = 1 if diff <  0 else 0
    g = 1 if diff >  0 else 0
    gpu.flags[tid] = (z << 0) | (n << 1) | (g << 2)

# CMP Ra, Rb
def op_cmp(gpu, tid, op1, op2):
    v1 = _resolve(gpu, tid, op1)
    v2 = _resolve(gpu, tid, op2)
    diff = int(v1) - int(v2)
    _set_flags_from_result(gpu, tid, diff)

# BRGT target    -> branch if greater (G bit set)
def op_brgt(gpu, tid, target):
    if (gpu.flags[tid] & 0b100) != 0:
        gpu.pc[tid] = int(_resolve(gpu, tid, target))
    else:
        gpu.pc[tid] = int(gpu.pc[tid])  # leave unchanged (core increments)

# BRLT target    -> branch if less (N bit set)
def op_brlt(gpu, tid, target):
    if (gpu.flags[tid] & 0b010) != 0:
        gpu.pc[tid] = int(_resolve(gpu, tid, target))
    else:
        gpu.pc[tid] = int(gpu.pc[tid])

# BRZ target     -> branch if equal (Z bit set)
def op_brz(gpu, tid, target):
    if (gpu.flags[tid] & 0b001) != 0:
        gpu.pc[tid] = int(_resolve(gpu, tid, target))
    else:
        gpu.pc[tid] = int(gpu.pc[tid])


def op_shld(gpu, tid, rd_operand, saddr_operand):
    """
    SHLD Rd, saddr  -> Rd = shared[block_id][saddr]
    saddr can be immediate or register operand.
    """
    if not (isinstance(rd_operand, tuple) and rd_operand[0] == "R"):
        raise TypeError("SHLD target must be a register")
    rd = rd_operand[1]
    # resolve shared address index
    sidx = int(_resolve(gpu, tid, saddr_operand))
    # bounds check
    if sidx < 0 or sidx >= gpu.shared_size:
        gpu.registers[tid, rd] = 0
        return
    # determine block id from tid and current threads_per_block (more robust
    # than relying on register R5 which user code can overwrite)
    if getattr(gpu, "threads_per_block", 0) > 0:
        block_id = int(tid // gpu.threads_per_block)
    else:
        block_id = int(gpu.registers[tid, 5]) if gpu.num_registers > 5 else 0

    # bounds-check block_id against allocated shared memory
    if block_id < 0 or block_id >= getattr(gpu, "num_blocks", 1):
        gpu.registers[tid, rd] = 0
        return

    gpu.registers[tid, rd] = int(gpu.shared[block_id, sidx])


def op_shst(gpu, tid, saddr_operand, rs_operand):
    """
    SHST saddr, Rs  -> shared[block_id][saddr] = Rs
    """
    sidx = int(_resolve(gpu, tid, saddr_operand))
    if sidx < 0 or sidx >= gpu.shared_size:
        return
    # determine block id from tid and current threads_per_block to avoid
    # relying on register R5 which user code may overwrite
    if getattr(gpu, "threads_per_block", 0) > 0:
        block_id = int(tid // gpu.threads_per_block)
    else:
        block_id = int(gpu.registers[tid, 5]) if gpu.num_registers > 5 else 0
    val = int(_resolve(gpu, tid, rs_operand))
    # bounds-check block_id before writing
    if block_id < 0 or block_id >= getattr(gpu, "num_blocks", 1):
        return
    gpu.shared[block_id, sidx] = val


def op_syncb(gpu, tid):
    """
    Block barrier: mark this thread as waiting at block-level barrier.
    The core's step() will release the whole block when every active thread in that block waits.
    """
    gpu.sync_waiting_block[tid] = True


# Instruction set mapping
INSTRUCTIONS = {
    "SET": op_set,
    "ADD": op_add,
    "MUL": op_mul,
    "LD": op_ld,
    "ST": op_st,
    "JMP": op_jmp,
    "BEQ": op_beq,
    "BNE": op_bne,
    "SYNC": op_sync,
    "CSWAP": op_cswap,
    "CMP": op_cmp,
    "BRGT": op_brgt,
    "BRLT": op_brlt,
    "BRZ": op_brz,
    "SHLD": op_shld,
    "SHST": op_shst,
    "SYNCB": op_syncb
}
