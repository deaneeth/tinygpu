# examples/run_test_cmp.py
import os
from src.tinygpu.gpu import TinyGPU
from src.tinygpu.assembler import assemble_file

example_path = os.path.join(os.path.dirname(__file__), "test_cmp.tgpu")
program, labels = assemble_file(example_path)

gpu = TinyGPU(num_threads=8, num_registers=8, mem_size=64)
gpu.load_program(program, labels)
gpu.run(max_cycles=20)

print("R0 per thread:", gpu.registers[:, 0])
print("Flags per thread:", gpu.flags)
