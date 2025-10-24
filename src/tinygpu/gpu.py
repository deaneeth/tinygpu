# src/tinygpu/gpu.py
import numpy as np
from .instructions import INSTRUCTIONS

class TinyGPU:
    def __init__(self, num_threads=8, num_registers=8, mem_size=256):
        self.num_threads = num_threads
        self.num_registers = num_registers
        self.mem_size = mem_size

        # registers[tid, reg]
        self.registers = np.zeros((num_threads, num_registers), dtype=np.int32)

        # shared memory
        self.memory = np.zeros(mem_size, dtype=np.int32)

        # per-thread program counter
        self.pc = np.zeros(num_threads, dtype=np.int32)

        # per-thread "active" flag (unused lanes when needed)
        self.active = np.ones(num_threads, dtype=bool)

        # barrier / sync state: per-thread whether it's waiting at a sync point
        self.sync_waiting = np.zeros(num_threads, dtype=bool)

        # histories for visualization
        self.history_registers = []  # list of arrays shape=(num_threads, num_registers)
        self.history_memory = []     # list of arrays shape=(mem_size,)
        self.history_pc = []         # list of pc array shape=(num_threads,)

        self.program = []
        self.labels = {}

        # initialize thread ids in a fixed register (R7 by convention)
        for tid in range(self.num_threads):
            # make sure register index exists
            if self.num_registers > 7:
                self.registers[tid, 7] = tid
            else:
                # if too few registers, set R0 as thread id (unlikely), but warn
                self.registers[tid, 0] = tid

    def load_program(self, program, labels=None):
        self.program = program
        self.labels = labels or {}
        self.pc[:] = 0
        self.sync_waiting[:] = False
        self.active[:] = True
        self.history_registers = []
        self.history_memory = []
        self.history_pc = []

    def step(self):
        """
        Execute one cycle: each active thread executes the instruction at its PC.
        Instructions that modify PC are expected to set self.pc[tid] inside the instruction.
        If an instruction doesn't change PC, we increment it by 1 automatically.
        SYNC instructions should set sync_waiting[tid] = True and then the core will release them
        when all threads have reached the same sync point (or simple condition).
        """
        # Loop threads and execute their instruction if active and in-range
        for tid in range(self.num_threads):
            if not self.active[tid]:
                continue
            if self.pc[tid] < 0 or self.pc[tid] >= len(self.program):
                # thread finished
                self.active[tid] = False
                continue

            instr, args = self.program[self.pc[tid]]
            func = INSTRUCTIONS.get(instr)
            before_pc = int(self.pc[tid])

            if func:
                # execute instruction
                func(self, tid, *args)

                # If instruction didn't change PC (still same before_pc), increment
                if int(self.pc[tid]) == before_pc and not self.sync_waiting[tid]:
                    # increment to next instruction
                    self.pc[tid] = before_pc + 1

        # handle global barrier: if any thread is waiting at a sync point, check if we can release
        if self.sync_waiting.any():
            # crude policy: release when all active threads have sync_waiting True
            # only consider threads that are still active
            active_waiting = self.sync_waiting[self.active]
            if active_waiting.size > 0 and active_waiting.all():
                # move all waiting threads forward by 1 and clear waiting flags
                for tid in range(self.num_threads):
                    if self.active[tid] and self.sync_waiting[tid]:
                        self.pc[tid] = int(self.pc[tid]) + 1
                        self.sync_waiting[tid] = False

        # record state
        self.history_registers.append(self.registers.copy())
        self.history_memory.append(self.memory.copy())
        self.history_pc.append(self.pc.copy())

    def run(self, max_cycles=1000):
        for cycle in range(max_cycles):
            if not self.active.any():
                break
            self.step()
