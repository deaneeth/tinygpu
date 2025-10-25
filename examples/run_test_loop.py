import os
from tinygpu.gpu import TinyGPU
from tinygpu.assembler import assemble_file
from tinygpu.visualizer import visualize

# Path to loop program
example_path = os.path.join(os.path.dirname(__file__), "test_loop.tgpu")

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
