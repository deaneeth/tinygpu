# examples/run_odd_even_sort.py
import os
import numpy as np
from src.tinygpu.gpu import TinyGPU
from src.tinygpu.assembler import assemble_file
from src.tinygpu.visualizer import visualize

# configuration
ARRAY_LEN = 16            # must be even for odd-even transposition; adjust as needed
NUM_THREADS = ARRAY_LEN // 2
MEM_BASE = 0              # start of array in memory
MEM_SIZE = 256
MAX_CYCLES = 400

# load program text and patch the SET R1 immediate to ARRAY_LEN
prog_path = os.path.join(os.path.dirname(__file__), "odd_even_sort.tgpu")
with open(prog_path, "r") as f:
    prog_text = f.read()

# quick injection: replace "SET R1, 8" with the actual number if present
prog_text = prog_text.replace("SET R1, 8", f"SET R1, {ARRAY_LEN}")

# write temp program file for assembler
tmp_path = os.path.join(os.path.dirname(__file__), "odd_even_sort_tmp.tgpu")
with open(tmp_path, "w") as f:
    f.write(prog_text)

# assemble
program, labels = assemble_file(tmp_path)

# create gpu with enough registers
gpu = TinyGPU(num_threads=NUM_THREADS, num_registers=12, mem_size=MEM_SIZE)

# initialize array in memory
arr = np.random.randint(0, 100, size=ARRAY_LEN)
print("Initial array:", arr.tolist())
for i in range(ARRAY_LEN):
    gpu.memory[MEM_BASE + i] = int(arr[i])
gpu.memory[MEM_BASE + ARRAY_LEN] = 9999   # sentinel guard value

# load program and run
gpu.load_program(program, labels)
gpu.run(max_cycles=MAX_CYCLES)

# print result
sorted_arr = [int(gpu.memory[MEM_BASE + i]) for i in range(ARRAY_LEN)]
print("Sorted array:", sorted_arr)

# visualize
visualize(gpu, show_pc=True)
