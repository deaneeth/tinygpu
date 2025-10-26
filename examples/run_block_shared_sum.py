import os
import sys
import numpy as np
from tinygpu.visualizer import save_animation
import time

# ensure src/ is on sys.path so examples can import the package
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, src_path)

from tinygpu.gpu import TinyGPU  # noqa: E402
from tinygpu.assembler import assemble_file  # noqa: E402

ARRAY_LEN = 8  # total elements (must equal num_blocks * tpb)
NUM_BLOCKS = 2
TPB = 4  # threads per block
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

# Save animation GIF to src/outputs/<script_name>/
try:
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "src", "outputs", script_name
    )
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_gif = os.path.join(output_dir, f"{script_name}_{timestamp}.gif")
    save_animation(gpu, out_path=out_gif, fps=10, max_frames=200, dpi=100)
    print("Saved GIF:", os.path.abspath(out_gif))
except Exception as e:
    print("Could not save GIF:", e)

# read back block results at mem[100 + block_id] (as used in kernel)
results = [int(gpu.memory[100 + b]) for b in range(NUM_BLOCKS)]
print("Block sums (expected):", results)
print(
    "Expected manual sums:",
    [int(sum(arr[b * TPB : (b + 1) * TPB])) for b in range(NUM_BLOCKS)],
)
