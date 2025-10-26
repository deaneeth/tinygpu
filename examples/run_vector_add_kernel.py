import os
import numpy as np
from src.tinygpu.assembler import assemble_file
from src.tinygpu.gpu import TinyGPU
from src.tinygpu.visualizer import visualize, save_animation

# config
ARRAY_LEN = 8
NUM_BLOCKS = 1
TPB = ARRAY_LEN  # one block all threads
MEM_SIZE = 64
MAX_CYCLES = 50

prog_path = os.path.join(os.path.dirname(__file__), "vector_add.tgpu")
program, labels = assemble_file(prog_path)

# create gpu with total threads = NUM_BLOCKS * TPB
gpu = TinyGPU(num_threads=NUM_BLOCKS * TPB, num_registers=12, mem_size=MEM_SIZE)

# init memory: A at 0..7, B at 8..15
for i in range(ARRAY_LEN):
    gpu.memory[i] = i
    gpu.memory[8 + i] = i * 2

# launch kernel: grid = (blocks, threads_per_block)
gpu.load_kernel(program, labels=labels, grid=(NUM_BLOCKS, TPB), args=None, shared_size=0)

# run
gpu.run_kernel(max_cycles=MAX_CYCLES)

# inspect results (C at 16..)
print("Result C:", gpu.memory[16:16 + ARRAY_LEN].tolist())
visualize(gpu, show_pc=True)

# optional gif
# save_animation(gpu, out_path=os.path.join(os.path.dirname(__file__), "vector_add_kernel.gif"), fps=12, max_frames=120)
