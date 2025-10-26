import os
import time
from tinygpu.assembler import assemble_file
from tinygpu.gpu import TinyGPU
from tinygpu.visualizer import save_animation

program, labels = assemble_file("examples/test_kernel_args.tgpu")
gpu = TinyGPU(num_threads=8, num_registers=8, mem_size=64)
gpu.load_kernel(program, labels=labels, grid=(1, 8), args=[10, 5])
gpu.run_kernel(max_cycles=10)
print("mem[0..8]:", gpu.memory[:8].tolist())  # expect [10+5+0, 10+5+1, ...]
print("R0 per thread:", gpu.registers[:, 0].tolist())  # expect [10, 10, ...]
print("R1 per thread:", gpu.registers[:, 1].tolist())  # expect [5, 5, ...]
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
