# examples/run_block_shared_sum.py
import os
import numpy as np
from src.tinygpu.gpu import TinyGPU
from src.tinygpu.assembler import assemble_file

ARRAY_LEN = 8              # total elements (must equal num_blocks * tpb)
NUM_BLOCKS = 2
TPB = 4                    # threads per block
SHARED_SIZE = TPB
MEM_SIZE = 256

prog_path = os.path.join(os.path.dirname(__file__), "block_shared_sum.tgpu")
program, labels = assemble_file(prog_path)

# create gpu with total threads
gpu = TinyGPU(num_threads=NUM_BLOCKS * TPB, num_registers=12, mem_size=MEM_SIZE)
gpu.set_grid(NUM_BLOCKS, TPB, shared_size=SHARED_SIZE)

# prepare input values per thread in global memory at index tid
arr = np.arange(1, ARRAY_LEN + 1)  # [1,2,3,...]
print("Input values per tid:", arr.tolist())
for tid in range(ARRAY_LEN):
    gpu.memory[tid] = int(arr[tid])

gpu.load_program(program, labels)
gpu.run(max_cycles=200)

# read back block results at mem[100 + block_id] (as used in kernel)
results = [int(gpu.memory[100 + b]) for b in range(NUM_BLOCKS)]
print("Block sums (expected):", results)
print("Expected manual sums:", [int(sum(arr[b*TPB:(b+1)*TPB])) for b in range(NUM_BLOCKS)])
