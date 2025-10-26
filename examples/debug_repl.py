import os
import sys

# make local 'src' package available so imports resolve when running this script
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, src_path)

from tinygpu.assembler import assemble_file  # noqa: E402
from tinygpu.gpu import TinyGPU  # noqa: E402
from tinygpu.visualizer import visualize  # noqa: E402

# config - change program path
prog_path = os.path.join(os.path.dirname(__file__), "test_loop.tgpu")
program, labels = assemble_file(prog_path)

gpu = TinyGPU(num_threads=4, num_registers=8, mem_size=64)
gpu.load_program(program, labels)

print("TinyGPU debug REPL")
print("Commands: s(step), n <k>(step k), p(print snapshot), v(visualize),")
print("r <k>(rewind k), q(quit)")

while True:
    cmd = input("dbg> ").strip().split()
    if not cmd:
        continue
    c = cmd[0]
    if c in ("q", "quit"):
        break
    if c in ("s", "step"):
        gpu.step_single()
        snap = gpu.snapshot(mem_slice=(0, 16), regs_threads=[0, 1])
        print(f"cycle {snap['cycle']} pc: {snap['pc']}")
        print("R0..R1 for threads 0,1:")
        for tid in [0, 1]:
            print(f" T{tid} regs:", snap["registers"][tid][:4])
    elif c in ("n", "stepk"):
        k = int(cmd[1]) if len(cmd) > 1 else 1
        for _ in range(k):
            gpu.step_single()
        print("advanced", k, "cycles")
    elif c in ("p", "print"):
        snap = gpu.snapshot(mem_slice=(0, 32))
        print("PC:", snap["pc"])
        print("Flags:", snap["flags"])
        print("Mem[0..32]:", snap["memory_slice"])
    elif c in ("v", "viz", "visualize"):
        visualize(gpu, show_pc=True)
    elif c in ("r", "rewind"):
        k = int(cmd[1]) if len(cmd) > 1 else 1
        try:
            gpu.rewind(k)
            print("rewound", k, "cycles")
        except Exception as e:
            print("rewind error:", e)
    else:
        print("unknown command")
