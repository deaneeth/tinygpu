import os
import time
import numpy as np
from tinygpu.gpu import TinyGPU
from tinygpu.assembler import assemble_file
from tinygpu.visualizer import visualize, save_animation

# ---- config ----
ARRAY_LEN = 8  # must be power of two
NUM_THREADS = ARRAY_LEN // 2
MEM_SIZE = (
    ARRAY_LEN + 1 + 64
)  # array + result + buffer | e.g. 8 + 1 + 64 = 73, any cushion is fine
MAX_CYCLES = 80
# ----------------

# load and assemble program
prog_path = os.path.join(os.path.dirname(__file__), "reduce_sum.tgpu")
program, labels = assemble_file(prog_path)

# create gpu with enough registers (R0-R9 used)
gpu = TinyGPU(num_threads=NUM_THREADS, num_registers=12, mem_size=MEM_SIZE)

# initialize array in memory to sum over
arr = np.random.randint(1, 10, size=ARRAY_LEN)
print("Initial array:", arr.tolist())
for i in range(ARRAY_LEN):
    gpu.memory[i] = int(arr[i])
gpu.memory[ARRAY_LEN : ARRAY_LEN + 10] = 0  # clear some buffer space

# load program and run
gpu.load_program(program, labels)
gpu.run(max_cycles=MAX_CYCLES)

# print result
result = gpu.memory[0]
print("Final sum (mem[0]):", int(result))
print("Expected sum:", int(np.sum(arr)))

visualize(gpu, show_pc=True)

# Save gif
script_name = os.path.splitext(os.path.basename(__file__))[0]  # e.g., run_reduce_sum
output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", script_name)
os.makedirs(output_dir, exist_ok=True)

# create timestamped output path
timestamp = time.strftime("%Y%m%d-%H%M%S")
out_gif = os.path.join(output_dir, f"{script_name}_{timestamp}.gif")

# Save animation
save_animation(
    gpu,
    out_path=out_gif,
    fps=10,
    max_frames=200,
    dpi=100,
)

print("Saved GIF:", os.path.abspath(out_gif))
