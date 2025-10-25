import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from tinygpu.gpu import TinyGPU
from tinygpu.assembler import assemble_file
from tinygpu.visualizer import visualize

# Path to program
example_path = os.path.join(os.path.dirname(__file__), "vector_add.tgpu")

# Assemble program
program, labels = assemble_file(example_path)

# Initialize GPU (8 threads, 8 registers, 64 memory slots for demo)
gpu = TinyGPU(num_threads=8, num_registers=8, mem_size=64)

# Initialize memory manually for demo (A[0..7], B[0..7])
for i in range(8):
    gpu.memory[i] = i                           # A[i] = 0..7
    gpu.memory[8 + i] = i * 2               # B[i] = 0,2,4,...14

# Load program
gpu.load_program(program, labels)

print("Assembled program:")
for instr, args in program:
    print(instr, args)

# Run program
gpu.run(max_cycles=20)
print("Memory A:", gpu.memory[:8])
print("Memory B:", gpu.memory[8:16])
print("Memory C:", gpu.memory[16:24])
for t in range(gpu.num_threads):
    print(f"Thread {t}  R7(thread_id)={gpu.registers[t,7]}  R0={gpu.registers[t,0]}  R1={gpu.registers[t,1]}  R2={gpu.registers[t,2]}")
    print(f"Thread {t}  R3={gpu.registers[t,3]}  R4={gpu.registers[t,4]}  R5={gpu.registers[t,5]}  R6={gpu.registers[t,6]}")

# Print result memory (C[0..7] should be A[i]+B[i])
print("Result C:", gpu.memory[16:24])

# Visualize
visualize(gpu)
