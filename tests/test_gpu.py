import numpy as np
from tinygpu.gpu import TinyGPU


def test_tinygpu_init():
    gpu = TinyGPU(num_threads=4, num_registers=8, mem_size=16)
    assert gpu.num_threads == 4
    assert gpu.num_registers == 8
    assert gpu.mem_size == 16
    assert gpu.registers.shape == (4, 8)
    assert gpu.memory.shape == (16,)
    assert gpu.pc.shape == (4,)
    assert gpu.active.shape == (4,)


def test_tinygpu_step_and_run():
    gpu = TinyGPU(num_threads=2, num_registers=4, mem_size=8)
    # Fake program: increment R0, halt after 2 steps
    gpu.program = [
        ("LOAD", [("R", 0), 1]),
        ("ADD", [("R", 0), ("R", 0), 1]),
    ]
    gpu.pc[:] = 0
    gpu.active[:] = True
    gpu.sync_waiting[:] = False
    gpu.history_registers = []
    gpu.history_memory = []
    gpu.history_pc = []
    # Patch INSTRUCTIONS for test
    from tinygpu import instructions

    def fake_load(self, tid, reg, val):
        self.registers[tid, reg[1]] = val

    def fake_add(self, tid, reg, reg2, val):
        self.registers[tid, reg[1]] = self.registers[tid, reg2[1]] + val

    instructions.INSTRUCTIONS["LOAD"] = fake_load
    instructions.INSTRUCTIONS["ADD"] = fake_add
    # After first step, both instructions are executed for both threads
    gpu.step()
    # LOAD sets R0=1, then ADD sets R0=2 in the same step
    assert np.all(gpu.registers[:, 0] == 2)
