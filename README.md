# TinyGPU 🐉⚡  

[![PyPI version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://pypi.org/project/tinygpu)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/deaneeth/tinygpu/actions/workflows/ci.yml/badge.svg)](https://github.com/deaneeth/tinygpu/actions)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/github/actions/workflow/status/deaneeth/tinygpu/ci.yml?label=tests)](https://github.com/deaneeth/tinygpu/actions)

TinyGPU is a **tiny educational GPU simulator** - inspired by [Tiny8](https://github.com/sql-hkr/tiny8), designed to demonstrate how GPUs execute code in parallel. It models a small **SIMT (Single Instruction, Multiple Threads)** system with per-thread registers, global memory, synchronization barriers, branching, and a minimal GPU-like instruction set.

> 🎓 *Built for learning and visualization - see how threads, registers, and memory interact across cycles!*
 
| Odd-Even Sort | Reduction |
|---------------|------------|
| ![Odd-Even Sort](src/outputs/run_odd_even_sort/run_odd_even_sort_20251026-212558.gif) | ![Reduction](src/outputs/run_reduce_sum/run_reduce_sum_20251026-212712.gif) |

---

## 🚀 What's New in v2.0.0

- **Enhanced Instruction Set**:
  - Added `SHLD` and `SHST` for robust shared memory operations.
  - Improved `SYNC` semantics for better thread coordination.
- **Visualizer Improvements**:
  - Export execution as GIFs with enhanced clarity.
  - Added support for saving visuals directly from the simulator.
- **Refactored Core**:
  - Simplified step semantics for better extensibility.
  - Optimized performance for larger thread counts.
- **CI/CD Updates**:
  - Integrated linting (`ruff`, `black`) and testing workflows.
  - Automated builds and tests on GitHub Actions.
- **Documentation**:
  - Expanded examples and added detailed usage instructions.

---

## 💡 Why TinyGPU?

TinyGPU was built as a **learning-first GPU simulator** - simple enough for beginners, but powerful enough to illustrate real GPU execution concepts like threads, synchronization, and divergence.

- ⚡ **Lightweight:**  
  Zero heavy dependencies - runs entirely in Python with clean, readable code.
- 🧩 **Educational:**  
  Demonstrates GPU fundamentals: thread execution, shared memory, branching, and synchronization barriers.
- 🔁 **Fast feedback loop:**  
  Write a `.tgpu` file, run it, and instantly visualize how your threads interact.
- 🧠 **Extensible:**  
  Add your own instructions, modify architecture parameters, or visualize new metrics.
- 🎨 **Visual clarity:**  
  Each program can be rendered as a heatmap or exported as a GIF - perfect for teaching and demos.

---

## 👥 Who Should Use It?

- 🧑‍🎓 **Students** - learn GPU basics through hands-on examples.  
- 👩‍🏫 **Educators** - build step-by-step visual lessons about parallel computing.  
- 🧑‍🔬 **Researchers & hobbyists** - prototype kernel logic or explore synchronization models.  
- 💡 **Developers** - experiment with SIMD-style concepts without real GPU dependencies.

---

## 🚦 Get Started

1. **Install & run locally** - see the [Quickstart](#-quickstart) section below.  
2. **Explore the examples** - try `vector_add`, `odd_even_sort`, and `reduce_sum` kernels.  
3. **Visualize execution** - watch registers, memory, and threads evolve in real-time.  
4. **Experiment!** - tweak instruction behavior or add your own to understand how GPUs schedule and synchronize work.

> 🧭 TinyGPU aims to make GPU learning *intuitive, visual, and interactive* - from classroom demos to self-guided exploration.

---

## ✨ Highlights

- 🧩 **GPU-like instruction set:**  
  `SET`, `ADD`, `MUL`, `LD`, `ST`, `JMP`, `BNE`, `BEQ`, `SYNC`, `CSWAP`, `SHLD`, `SHST`.
- 🧠 **Per-thread registers & PCs** - each thread executes the same kernel independently.
- 🧱 **Shared global memory** for inter-thread operations.
- 🔄 **Synchronization barriers** (`SYNC`) for parallel coordination.
- 🎨 **Visualizer & GIF exporter** - view execution as heatmaps or export to animated GIFs.
- 🧮 **Example kernels included:**
  - Vector addition  
  - Odd-even sort (GPU-style bubble sort)  
  - Parallel reduction (sum of array)  
  - Loop test & synchronization demo  

---

## 🖼️ Example Visuals

> Located in `src/outputs/` — run the example scripts to generate these GIFs (they're saved under `src/outputs/<script_name>/`).

| Example | Description | GIF Preview |
|---------|-------------|-------------|
| Vector Add | Parallel vector addition (A+B -> C) | ![Vector Add](src/outputs/run_vector_add/run_vector_add_20251026-212734.gif) |
| Block Shared Sum | Per-block shared memory sum example | ![Block Shared Sum](src/outputs/run_block_shared_sum/run_block_shared_sum_20251026-212542.gif) |
| Odd-Even Sort | GPU-style odd-even transposition sort | ![Odd-Even Sort](src/outputs/run_odd_even_sort/run_odd_even_sort_20251026-212558.gif) |
| Parallel Reduction | Sum reduction across an array | ![Reduction](src/outputs/run_reduce_sum/run_reduce_sum_20251026-212712.gif) |
| Sync Test | Synchronization / barrier demonstration | ![Sync Test](src/outputs/run_sync_test/run_sync_test_20251027-000818.gif) |
| Loop Test | Branching and loop behavior demo | ![Test Loop](src/outputs/run_test_loop/run_test_loop_20251026-212814.gif) |
| Compare Test | Comparison and branching example | ![Test CMP](src/outputs/run_test_cmp/run_test_cmp_20251026-212823.gif) |
| Kernel Args Test | Demonstrates passing kernel arguments | ![Kernel Args](src/outputs/run_test_kernel_args/run_test_kernel_args_20251026-212830.gif) |

---

## 🚀 Quickstart

### Clone and install

```bash
git clone https://github.com/deaneeth/tinygpu.git
cd tinygpu
pip install -e .
pip install -r requirements-dev.txt
```

### Run an example

```bash
python -m examples.run_odd_even_sort
```

> Produces: `src/outputs/run_odd_even_sort/run_odd_even_sort_*.gif` — a visual GPU-style sorting process.

### Other examples

```bash
python -m examples.run_vector_add
python -m examples.run_reduce_sum
python -m examples.run_test_loop
python -m examples.run_sync_test
```

---

## 🧩 Project Layout

```text
.
├─ .github/
│  └─ workflows/
│     └─ ci.yml
├─ docs/
│  └─ index.md
├─ examples/
│  ├─ odd_even_sort_tmp.tgpu
│  ├─ odd_even_sort.tgpu
│  ├─ reduce_sum.tgpu
│  ├─ run_odd_even_sort.py
│  ├─ run_reduce_sum.py
│  ├─ run_sync_test.py
│  ├─ run_test_loop.py
│  ├─ run_vector_add.py
│  ├─ sync_test.tgpu
│  ├─ test_loop.tgpu
│  └─ vector_add.tgpu
├─ src/outputs/
│  ├─ run_block_shared_sum/
│  ├─ run_odd_even_sort/
│  ├─ run_reduce_sum/
│  ├─ run_sync_test/
│  ├─ run_test_cmp/
│  ├─ run_test_kernel_args/
│  ├─ run_test_loop/
│  └─ run_vector_add/
├─ src/
│  └─ tinygpu/
│     ├─ __init__.py
│     ├─ assembler.py
│     ├─ gpu.py
│     ├─ instructions.py
│     └─ visualizer.py
├─ tests/
│  ├─ test_assembler.py
│  ├─ test_gpu_core.py
│  ├─ test_gpu.py
│  └─ test_programs.py
├─ LICENSE
├─ pyproject.toml
├─ README.md
└─ requirements-dev.txt
```

---

## 🧠 Assembly Reference (Mini ISA)

TinyGPU uses a **minimal instruction set** designed for clarity and education - simple enough for beginners, but expressive enough to build realistic GPU-style kernels.

| **Instruction**             | **Operands**                            | **Description** |
|-----------------------------|------------------------------------------|-----------------|
| `SET Rd, imm`               | `Rd` = destination register, `imm` = immediate value | Set register `Rd` to an immediate constant. |
| `ADD Rd, Ra, Rb`            | `Rd` = destination, `Ra` + `Rb` | Add two registers and store result in `Rd`. |
| `ADD Rd, Ra, imm`           | `Rd` = destination, `Ra` + immediate | Add register and immediate value. |
| `MUL Rd, Ra, Rb`            | Multiply two registers. | `Rd = Ra * Rb` |
| `MUL Rd, Ra, imm`           | Multiply register by immediate. | `Rd = Ra * imm` |
| `LD Rd, addr`               | Load from memory address into register. | `Rd = mem[addr]` |
| `LD Rd, Rk`                 | Load from address in register `Rk`. | `Rd = mem[Rk]` |
| `ST addr, Rs`               | Store register into memory address. | `mem[addr] = Rs` |
| `ST Rk, Rs`                 | Store value from `Rs` into memory at address in register `Rk`. | `mem[Rk] = Rs` |
| `JMP target`                | Label or immediate. | Unconditional jump — sets PC to `target`. |
| `BEQ Ra, Rb, target`        | Branch if equal. | Jump to `target` if `Ra == Rb`. |
| `BNE Ra, Rb, target`        | Branch if not equal. | Jump to `target` if `Ra != Rb`. |
| `SYNC`                      | *(no operands)* | Synchronization barrier — all threads must reach this point before continuing. |
| `CSWAP addrA, addrB`        | Compare-and-swap memory values. | If `mem[addrA] > mem[addrB]`, swap them. Used for sorting. |
| `SHLD addr, Rs`             | Load shared memory into register. | `Rs = shared_mem[addr]` |
| `SHST addr, Rs`             | Store register into shared memory. | `shared_mem[addr] = Rs` |
| `CMP Rd, Ra, Rb` *(optional)* | Compare and set flag or register. | Used internally for extended examples (e.g., prefix-scan). |
| `NOP` *(optional)*          | *(no operands)* | No operation; placeholder instruction. |

---

### 💡 Notes & Conventions

- **Registers:**  
  - Each thread has its own register file (`R0`–`R7` or more depending on configuration).  
  - `R7` is automatically initialized to the thread ID (`tid`).
- **Memory:**  
  - Global memory is shared between all threads.
  - Addresses can be immediate (`16`) or register-based (`R2`).
- **Labels & Comments:**  
  - Use labels like `loop:` for jumps.
  - Comments start with `;`.
- **Synchronization:**  
  - Use `SYNC` when threads must finish a phase before the next (e.g., sorting or reduction).
- **Execution:**  
  - Each instruction executes per-thread.
  - Threads can branch independently (SIMT divergence is handled via per-thread PCs).

---

## 🧮 Example: Odd-Even Sort Kernel

```asm
SET R0, 0        ; phase_counter
SET R1, 8        ; num_phases == N (set to array length here, adjust in runner)
SET R3, 0        ; parity (0 = even phase, 1 = odd phase)

loop_phase:
    ; compute base index = tid * 2
    MUL R4, R7, 2      ; R4 = tid * 2
    ADD R5, R4, R3     ; R5 = index = tid*2 + parity
    ADD R6, R5, 1      ; R6 = index + 1
    CSWAP R5, R6       ; compare & swap memory[index], memory[index+1]
    SYNC               ; synchronize threads across the phase

    ADD R3, R3, 1      ; parity = parity + 1
    BNE R3, 2, noreset
    SET R3, 0
noreset:
    ADD R0, R0, 1
    BNE R0, R1, loop_phase

done:
    JMP done
```

---

## 🧩 Writing & Running Programs

1. Write your `.tgpu` assembly file in `examples/`.

2. Assemble and run it in Python:

   ```python
   from src.tinygpu.assembler import assemble_file
   from src.tinygpu.gpu import TinyGPU

   prog, labels = assemble_file("examples/vector_add.tgpu")
   gpu = TinyGPU(num_threads=8, num_registers=8, mem_size=64)
   gpu.load_program(prog, labels)
   gpu.run(max_cycles=100)
   ```

3. Visualize the run:

   ```python
   from src.tinygpu.visualizer import visualize
   visualize(gpu, show_pc=True)
   ```

4. Export as GIF:

   ```python
   from src.tinygpu.visualizer import save_animation
   save_animation(gpu, out_path="examples/my_run.gif", fps=10, max_frames=200)
   ```

---

## 🧰 Development & Testing

Run tests:

```bash
pytest
```

Run linters:

```bash
ruff .
black --check src/ tests/
```

CI builds and runs tests automatically on push/pull.

---

## 📘 License

MIT - see [LICENSE](LICENSE)

---

## 🌟 Credits & Inspiration

❤️ Built by [Deaneeth](https://github.com/deaneeth)

> Inspired by the educational design of [Tiny8 CPU Simulator](https://github.com/sql-hkr/tiny8).

TinyGPU extends these ideas into the world of **parallel GPU computing**, emphasizing **clarity, simplicity, and visualization** for all learners.
