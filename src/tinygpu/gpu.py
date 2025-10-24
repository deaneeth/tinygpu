import numpy as np
from .instructions import INSTRUCTIONS

class TinyGPU:
    def __init__(self, num_threads=8, num_registers=8, mem_size=256):
        
        self.num_threads = num_threads
        self.num_registers = num_registers
        self.mem_size = mem_size

        # Initialize registers
        self.registers = np.zeros((num_threads, num_registers), dtype=np.int32)

        # Each thread has its own register file
        self.registers = np.zeros((num_threads, num_registers), dtype=np.int32)
        for tid in range(self.num_threads):
            self.registers[tid, 7] = tid  # R7 = thread_id

        # Shared global memory
        self.memory = np.zeros(mem_size, dtype=np.int32)

        # Program counters per thread (for divergence simulation later)
        self.pc = np.zeros(num_threads, dtype=np.int32)

        # History for visualization
        self.history_registers = []
        self.history_memory = []

        self.program = []
        self.labels = {}

    def load_program(self, program, labels):
        self.program = program
        self.labels = labels
        self.pc[:] = 0

    def step(self):
        # execute one instruction per active thread
        for tid in range(self.num_threads):
            if self.pc[tid] < 0 or self.pc[tid] >= len(self.program):
                continue  # thread finished
            instr, args = self.program[self.pc[tid]]
            func = INSTRUCTIONS.get(instr)
            if func:
                before_pc = self.pc[tid]
                func(self, tid, *args)
                # if instruction didn’t modify PC (e.g., ALU op), increment
                if self.pc[tid] == before_pc:
                    self.pc[tid] += 1

        self.history_registers.append(self.registers.copy())
        self.history_memory.append(self.memory.copy())

    def run(self, max_cycles=100):
        for _ in range(max_cycles):
            if all(pc >= len(self.program) for pc in self.pc):
                break
            self.step()
