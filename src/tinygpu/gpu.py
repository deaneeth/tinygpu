# src/tinygpu/gpu.py
import numpy as np
from .instructions import INSTRUCTIONS

class TinyGPU:
    def __init__(self, num_threads=8, num_registers=8, mem_size=256):
        # core sizes
        self.num_threads = num_threads
        self.num_registers = num_registers
        self.mem_size = mem_size

        # registers and memory
        self.registers = np.zeros((num_threads, num_registers), dtype=np.int32)
        self.memory = np.zeros(mem_size, dtype=np.int32)

        # per-thread PC, active mask
        self.pc = np.zeros(num_threads, dtype=np.int32)
        self.active = np.ones(num_threads, dtype=bool)

        # flags from earlier enhancement (int8 bitmask)
        self.flags = np.zeros(num_threads, dtype=np.int8)

        # global sync waiting (already used for SYNC)
        self.sync_waiting = np.zeros(num_threads, dtype=bool)

        # block-level sync waiting (SYNCB)
        self.sync_waiting_block = np.zeros(num_threads, dtype=bool)

        # grid / shared memory defaults (1 block covering all threads)
        self.num_blocks = 1
        self.threads_per_block = num_threads
        self.shared_size = 0
        self.shared = np.zeros((1, 0), dtype=np.int32)  # shape (num_blocks, shared_size)

        # history for visualization
        self.history_registers = []
        self.history_memory = []
        self.history_pc = []
        self.history_flags = []
        self.history_shared = []  # for debugging/visualization of shared memory

        self.program = []
        self.labels = {}

        # initialize thread id in R7 and block/thread info in R5/R6 if possible
        for tid in range(self.num_threads):
            if self.num_registers > 7:
                self.registers[tid, 7] = tid  # global thread id
            else:
                self.registers[tid, 0] = tid

    def set_grid(self, num_blocks: int, threads_per_block: int, shared_size: int = 0):
        """
        Configure grid parameters and allocate shared memory.
        Must call before running (or call before load_program/run).
        """
        self.num_blocks = int(num_blocks)
        self.threads_per_block = int(threads_per_block)
        self.shared_size = int(shared_size)

        total_threads = self.num_blocks * self.threads_per_block
        if total_threads != self.num_threads:
            # resize register and pc arrays to match requested total threads
            old_regs = self.registers.copy()
            old_num_threads = self.num_threads
            self.num_threads = total_threads
            self.registers = np.zeros((self.num_threads, self.num_registers), dtype=np.int32)
            # copy what fits
            min_threads = min(old_num_threads, self.num_threads)
            self.registers[:min_threads, :old_regs.shape[1]] = old_regs[:min_threads, :]

            self.pc = np.zeros(self.num_threads, dtype=np.int32)
            self.active = np.ones(self.num_threads, dtype=bool)
            self.flags = np.zeros(self.num_threads, dtype=np.int8)
            self.sync_waiting = np.zeros(self.num_threads, dtype=bool)
            self.sync_waiting_block = np.zeros(self.num_threads, dtype=bool)

        # allocate shared memory
        self.shared = np.zeros((self.num_blocks, self.shared_size), dtype=np.int32)

        # initialize block_id (R5) and thread_in_block (R6) registers for each thread if available
        for tid in range(self.num_threads):
            block_id = tid // self.threads_per_block
            thread_in_block = tid % self.threads_per_block
            if self.num_registers > 5:
                self.registers[tid, 5] = block_id
            if self.num_registers > 6:
                self.registers[tid, 6] = thread_in_block
            # keep R7 as global tid (already set in __init__)

    def load_program(self, program, labels=None):
        self.program = program
        self.labels = labels or {}
        self.pc[:] = 0
        self.sync_waiting[:] = False
        self.sync_waiting_block[:] = False
        self.active[:] = True
        self.history_registers = []
        self.history_memory = []
        self.history_pc = []
        self.history_flags = []
        self.history_shared = []

    def step(self):
        """
        Execute one cycle: each active thread executes one instruction at its PC.
        Interactions:
          - SYNC (global) uses sync_waiting
          - SYNCB (block) uses sync_waiting_block (released per-block)
        """
        for tid in range(self.num_threads):
            if not self.active[tid]:
                continue
            if self.pc[tid] < 0 or self.pc[tid] >= len(self.program):
                self.active[tid] = False
                continue

            instr, args = self.program[self.pc[tid]]
            func = INSTRUCTIONS.get(instr)
            before_pc = int(self.pc[tid])

            if func:
                func(self, tid, *args)

                # if instruction didn't change PC (and not waiting), increment
                if int(self.pc[tid]) == before_pc and not self.sync_waiting[tid] and not self.sync_waiting_block[tid]:
                    self.pc[tid] = before_pc + 1

        # handle global barrier release (existing behavior)
        if self.sync_waiting.any():
            active_waiting = self.sync_waiting[self.active]
            if active_waiting.size > 0 and active_waiting.all():
                for tid in range(self.num_threads):
                    if self.active[tid] and self.sync_waiting[tid]:
                        self.pc[tid] = int(self.pc[tid]) + 1
                        self.sync_waiting[tid] = False

        # handle block-level barriers
        if self.sync_waiting_block.any():
            # check each block independently
            for b in range(self.num_blocks):
                # find tids in this block
                start = b * self.threads_per_block
                end = start + self.threads_per_block
                block_active_mask = self.active[start:end]
                if not block_active_mask.any():
                    continue
                block_waiting = self.sync_waiting_block[start:end][block_active_mask]
                # if all active threads in block are waiting, release them
                if block_waiting.size > 0 and block_waiting.all():
                    # advance pc for waiting threads in this block
                    for tid in range(start, end):
                        if self.active[tid] and self.sync_waiting_block[tid]:
                            self.pc[tid] = int(self.pc[tid]) + 1
                            self.sync_waiting_block[tid] = False

        # record history
        self.history_registers.append(self.registers.copy())
        self.history_memory.append(self.memory.copy())
        self.history_pc.append(self.pc.copy())
        self.history_flags.append(self.flags.copy())
        self.history_shared.append(self.shared.copy())

    def run(self, max_cycles=1000):
        for cycle in range(max_cycles):
            if not self.active.any():
                break
            self.step()
            
     # --- Step debugger helpers ---

    def step_single(self):
        """
        Execute exactly one cycle and record state (alias to step()).
        Useful for interactive stepping.
        """
        self.step()

    def snapshot(self, mem_slice=None, regs_threads=None):
        """
        Return a human-friendly snapshot of current state.
        - mem_slice: (start, end) to extract part of global memory (tuple) or None for full memory.
        - regs_threads: list of thread indices to show registers for, or None for all.
        Returns a dict.
        """
        if mem_slice:
            start, end = mem_slice
            mem_view = self.memory[start:end].tolist()
        else:
            mem_view = self.memory.tolist()

        if regs_threads is None:
            regs_view = {tid: self.registers[tid, :].tolist() for tid in range(self.num_threads)}
        else:
            regs_view = {tid: self.registers[tid, :].tolist() for tid in regs_threads}

        return {
            "cycle": len(self.history_pc),
            "pc": self.pc.tolist(),
            "active": self.active.tolist(),
            "flags": self.flags.tolist(),
            "registers": regs_view,
            "memory_slice": mem_view,
            "shared": self.shared.copy().tolist() if hasattr(self, "shared") else None
        }

    def rewind(self, cycles=1):
        """
        Rewind simulation by 'cycles' steps using stored history.
        Note: this only restores state from history arrays, and discards newer history.
        """
        if cycles <= 0:
            return

        if cycles > len(self.history_registers):
            raise ValueError("Not enough history to rewind that many cycles.")

        # target index after rewind
        target = len(self.history_registers) - cycles
        # restore last snapshot at index target-1 if target>0 else initial
        if target == 0:
            # reset to initial empty state
            self.registers[:] = 0
            self.memory[:] = 0
            self.pc[:] = 0
            self.flags[:] = 0
            if hasattr(self, "shared"):
                self.shared[:] = 0
            self.history_registers = []
            self.history_memory = []
            self.history_pc = []
            self.history_flags = []
            self.history_shared = []
        else:
            self.registers[:] = self.history_registers[target - 1].copy()
            self.memory[:] = self.history_memory[target - 1].copy()
            self.pc[:] = self.history_pc[target - 1].copy()
            self.flags[:] = self.history_flags[target - 1].copy()
            if hasattr(self, "shared") and len(self.history_shared) >= target:
                self.shared[:] = self.history_shared[target - 1].copy()
            # trim history
            self.history_registers = self.history_registers[:target]
            self.history_memory = self.history_memory[:target]
            self.history_pc = self.history_pc[:target]
            self.history_flags = self.history_flags[:target]
            self.history_shared = self.history_shared[:target]            
            

    def load_kernel(self, program, labels=None, grid=(1, None), args=None, shared_size=0):
        """
        Load a kernel program and configure grid/thread mapping.

        - program, labels: assembled program (list, dict) (same as load_program)
        - grid: (num_blocks, threads_per_block). threads_per_block None -> keep current
        - args: list of scalar kernel arguments. These will be written into registers R0..Rk for ALL threads.
        - shared_size: allocate per-block shared memory size (optional)
        """
        num_blocks, tpb = grid
        if tpb is None:
            tpb = self.threads_per_block if hasattr(self, "threads_per_block") else (self.num_threads // num_blocks)
        # configure grid (this may resize internal thread arrays if total differs)
        self.set_grid(int(num_blocks), int(tpb), shared_size=int(shared_size))

        # set kernel args into registers R0..Rk for every thread (if provided)
        if args:
            for tid in range(self.num_threads):
                for i, val in enumerate(args):
                    # write into register i (R0, R1, ...)
                    if i < self.num_registers:
                        self.registers[tid, i] = int(val)

        # finally load program and reset pcs/history
        self.load_program(program, labels)

    def run_kernel(self, max_cycles=1000):
        """
        Convenience wrapper: run until completion or max_cycles.
        """
        self.run(max_cycles=max_cycles)
