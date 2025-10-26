from tinygpu.assembler import assemble_file
from tinygpu.gpu import TinyGPU 


program, labels = assemble_file("examples/test_kernel_args.tgpu")
gpu = TinyGPU(num_threads=8, num_registers=8, mem_size=64)
gpu.load_kernel(program, labels=labels, grid=(1,8), args=[10, 5])
gpu.run_kernel(max_cycles=10)
print("mem[0..8]:", gpu.memory[:8].tolist())  # expect [10+5+0, 10+5+1, ...]
print("R0 per thread:", gpu.registers[:, 0].tolist())  # expect [10, 10, ...]
print("R1 per thread:", gpu.registers[:, 1].tolist())  # expect [5, 5, ...]