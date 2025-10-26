import os
import time
from tinygpu.gpu import TinyGPU
from tinygpu.assembler import assemble_file
from tinygpu.visualizer import visualize
from tinygpu.visualizer import save_animation

# Path to sync test program
example_path = os.path.join(os.path.dirname(__file__), "sync_test.tgpu")

# Assemble program
program, labels = assemble_file(example_path)

# Create GPU with 4 threads (for demo)
gpu = TinyGPU(num_threads=4, num_registers=8, mem_size=32)

# Load program into GPU
gpu.load_program(program, labels)

# Run simulation
gpu.run(max_cycles=50)

# Print results (sum in R0 per thread)
print("Final R0 register per thread:")
print(gpu.registers[:, 0])

# Visualize execution
visualize(gpu)

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
