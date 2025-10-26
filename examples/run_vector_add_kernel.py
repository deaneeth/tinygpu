import os
import sys

# make local 'src' package available so imports resolve when running this script
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, src_path)

from tinygpu.assembler import assemble_file  # noqa: E402
from tinygpu.gpu import TinyGPU  # noqa: E402
from tinygpu.visualizer import visualize, save_animation  # noqa: E402

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
gpu.load_kernel(
    program, labels=labels, grid=(NUM_BLOCKS, TPB), args=None, shared_size=0
)

# run
gpu.run_kernel(max_cycles=MAX_CYCLES)

# inspect results (C at 16..)
print("Result C:", gpu.memory[16 : 16 + ARRAY_LEN].tolist())
visualize(gpu, show_pc=True)
# Save animation GIF to src/outputs/<script_name>/
try:
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "src", "outputs", script_name
    )
    os.makedirs(output_dir, exist_ok=True)
    import time

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_gif = os.path.join(output_dir, f"{script_name}_{timestamp}.gif")
    save_animation(gpu, out_path=out_gif, fps=12, max_frames=120, dpi=100)
    print("Saved GIF:", os.path.abspath(out_gif))
except Exception as e:
    print("Could not save GIF:", e)
