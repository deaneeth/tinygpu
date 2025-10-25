import os
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import shutil
import imageio


# Simple static visualization of TinyGPU execution history
def visualize(gpu, show_pc=True):
    """
    Simple static visualization (like earlier working version).
    Plots registers (threads x regs), memory history (active slice), and PC heatmap.
    """
    reg_hist = np.array(gpu.history_registers)  # (cycles, threads, regs)
    mem_hist = np.array(gpu.history_memory)  # (cycles, mem_size)
    pc_hist = np.array(gpu.history_pc)  # (cycles, threads)

    if reg_hist.size == 0:
        print("No history recorded. Run gpu.run(...) first.")
        return

    # Get number of cycles recorded
    cycles = reg_hist.shape[0]

    # Reshape registers to (threads*regs, cycles)
    regs_reshaped = reg_hist.reshape(cycles, -1).T

    # Detect active memory region: columns that ever changed or are non-zero initially
    mem_changed = (
        np.any(mem_hist != mem_hist[0], axis=0)
        if mem_hist.size
        else np.zeros(0, dtype=bool)
    )
    mem_nonzero = (
        np.any(mem_hist != 0, axis=0) if mem_hist.size else np.zeros(0, dtype=bool)
    )
    active_mask = mem_changed | mem_nonzero
    if active_mask.any():
        start = int(np.where(active_mask)[0][0])
        end = int(np.where(active_mask)[0][-1]) + 1
    else:
        start, end = 0, min(32, gpu.mem_size)  # fallback small window

    # Plotting
    fig, axs = plt.subplots(
        3 if show_pc else 2,
        1,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [2, 1, 0.5] if show_pc else [2, 1]},
    )
    ax_regs = axs[0]
    ax_mem = axs[1]
    ax_pc = axs[2] if show_pc else None

    # Registers (threads*regs x cycles)
    im1 = ax_regs.imshow(regs_reshaped, aspect="auto", cmap="inferno", origin="lower")
    ax_regs.set_title("Registers over time (threads × regs)")
    fig.colorbar(im1, ax=ax_regs, label="value")

    # Memory: show active slice only (cycles x active_len)
    mem_to_plot = mem_hist[:, start:end].T if mem_hist.size else np.zeros((1, 1))
    im2 = ax_mem.imshow(mem_to_plot, aspect="auto", cmap="plasma", origin="lower")
    ax_mem.set_title("Memory over time (active slice)")
    fig.colorbar(im2, ax=ax_mem, label="value")

    # Program Counter per thread
    if show_pc:
        pcs = pc_hist.T
        im3 = ax_pc.imshow(pcs, aspect="auto", cmap="viridis", origin="lower")
        ax_pc.set_title("Program Counter (per-thread) over time")
        fig.colorbar(im3, ax=ax_pc, label="PC")

    plt.tight_layout()
    plt.show()


# Full TinyGPU execution animation and GIF saving
def save_animation(gpu, out_path="tinygpu_run.gif", fps=10, max_frames=200, dpi=100):
    """
    Full TinyGPU v3~v4 style animation:
    - Registers (threads × regs) at top
    - Memory evolution in middle
    - Program Counter per thread at bottom
    """

    reg_hist = np.array(gpu.history_registers)  # (cycles, threads, regs)
    mem_hist = np.array(gpu.history_memory)  # (cycles, mem_size)
    pc_hist = np.array(gpu.history_pc)  # (cycles, threads)
    cycles = reg_hist.shape[0]

    if cycles == 0:
        raise RuntimeError("No history recorded. Run gpu.run(...) first.")

    # detect active memory region (0:ARRAY_LEN typically)
    mem_changed = np.any(mem_hist != mem_hist[0], axis=0)
    mem_nonzero = np.any(mem_hist != 0, axis=0)
    active_mask = mem_changed | mem_nonzero
    if active_mask.any():
        mem_start = int(np.where(active_mask)[0][0])
        mem_end = int(np.where(active_mask)[0][-1]) + 1
    else:
        mem_start, mem_end = 0, min(32, gpu.mem_size)

    # color scale clamp (ignore extreme sentinel)
    all_vals = mem_hist[:, mem_start:mem_end]
    vmin, vmax = np.percentile(all_vals, 1), np.percentile(all_vals, 99)

    # cycle sampling
    if max_frames and cycles > max_frames:
        indices = np.linspace(0, cycles - 1, max_frames, dtype=int)
    else:
        indices = np.arange(cycles)

    tmpdir = tempfile.mkdtemp(prefix="tinygpu_frames_")
    frame_files = []

    # generate frames
    for i, frame_idx in enumerate(indices):
        fig, axs = plt.subplots(
            3, 1, figsize=(8, 6), dpi=dpi, gridspec_kw={"height_ratios": [2, 1, 0.5]}
        )
        ax_regs, ax_mem, ax_pc = axs

        # --- REGISTERS cumulative ---
        regs = (
            reg_hist[: frame_idx + 1].reshape(-1, gpu.num_threads * gpu.num_registers).T
        )
        ax_regs.imshow(regs, aspect="auto", cmap="inferno", origin="lower")
        ax_regs.set_title("Registers over time (threads × regs)")
        ax_regs.set_xlabel("Cycle")
        ax_regs.set_ylabel("Thread × Reg")

        # --- MEMORY cumulative ---
        mem_frame = mem_hist[: frame_idx + 1, mem_start:mem_end].T
        ax_mem.imshow(
            mem_frame,
            aspect="auto",
            cmap="plasma",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        ax_mem.set_title(f"Memory over time (up to cycle {frame_idx})")
        ax_mem.set_xlabel("Cycle")
        ax_mem.set_ylabel("Memory index")

        # --- PROGRAM COUNTER cumulative ---
        pc_frame = pc_hist[: frame_idx + 1].T
        ax_pc.imshow(pc_frame, aspect="auto", cmap="viridis", origin="lower")
        ax_pc.set_title("Program Counter (per-thread)")
        ax_pc.set_xlabel("Cycle")
        ax_pc.set_ylabel("Thread")

        fname = os.path.join(tmpdir, f"frame_{i:05d}.png")
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        frame_files.append(fname)

    # combine all frames (resize consistent)
    imgs = [imageio.imread(f) for f in frame_files]
    target_shape = min(img.shape[:2] for img in imgs)
    imgs = [img[: target_shape[0], : target_shape[1]] for img in imgs]

    # save gif
    imageio.mimsave(out_path, imgs, fps=fps, loop=0)
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"✅ GIF saved: {out_path}")
