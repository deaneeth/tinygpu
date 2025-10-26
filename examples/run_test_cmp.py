# examples/run_test_cmp.py
import os
import sys
import time
from tinygpu.visualizer import save_animation

# make local 'src' package available so imports resolve when running this script
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, src_path)

from tinygpu.gpu import TinyGPU  # noqa: E402
from tinygpu.assembler import assemble_file  # noqa: E402

example_path = os.path.join(os.path.dirname(__file__), "test_cmp.tgpu")
program, labels = assemble_file(example_path)

gpu = TinyGPU(num_threads=8, num_registers=8, mem_size=64)
gpu.load_program(program, labels)
gpu.run(max_cycles=20)

print("R0 per thread:", gpu.registers[:, 0])
print("Flags per thread:", gpu.flags)

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
