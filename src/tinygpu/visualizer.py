import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import tempfile
import shutil

def visualize(gpu, show_pc=True):
    reg_hist = np.array(gpu.history_registers)    # shape (cycles, threads, regs)
    mem_hist = np.array(gpu.history_memory)       # shape (cycles, mem_size)
    pc_hist  = np.array(gpu.history_pc)           # shape (cycles, threads)

    cycles = reg_hist.shape[0]

    # reshape registers into (cycles, threads*regs) for heatmap
    regs_reshaped = reg_hist.reshape(cycles, -1).T

    fig, axs = plt.subplots(3 if show_pc else 2, 1, figsize=(12, 9), gridspec_kw={'height_ratios': [2, 1, 0.5] if show_pc else [2,1]})
    ax_regs = axs[0]
    ax_mem = axs[1]
    ax_pc  = axs[2] if show_pc else None

    im1 = ax_regs.imshow(regs_reshaped, aspect='auto', cmap='inferno')
    ax_regs.set_title("Registers over time (threads × regs)")
    fig.colorbar(im1, ax=ax_regs, label='value')

    im2 = ax_mem.imshow(mem_hist.T, aspect='auto', cmap='plasma')
    ax_mem.set_title("Memory over time")
    fig.colorbar(im2, ax=ax_mem, label='value')

    if show_pc:
        # pc_hist: shape (cycles, threads) -> transpose to (threads, cycles)
        pcs = pc_hist.T
        # plot PC heatmap (it shows divergence when rows differ)
        im3 = ax_pc.imshow(pcs, aspect='auto', cmap='viridis')
        ax_pc.set_title("Program Counter (per-thread) over time")
        fig.colorbar(im3, ax=ax_pc, label='PC')

        # optional: mark divergence lines (where not all PCs equal)
        divergence = np.any(pc_hist != pc_hist[0, :].reshape(-1, 1).T, axis=1) if cycles > 0 else np.array([])
        # We can overlay vertical shading where divergence occurs:
        for t_idx, diverged in enumerate(divergence):
            if diverged:
                # small translucent rectangle
                ax_regs.axvspan(t_idx - 0.5, t_idx + 0.5, color='gray', alpha=0.06)
                ax_mem.axvspan(t_idx - 0.5, t_idx + 0.5, color='gray', alpha=0.06)
                ax_pc.axvspan(t_idx - 0.5, t_idx + 0.5, color='gray', alpha=0.06)

    plt.tight_layout()
    plt.show()

def save_animation(gpu, out_path="tinygpu_run.gif", fps=10, max_frames=200, dpi=100):
    """
    Create an animated GIF from the GPU history.
    - gpu: TinyGPU instance (must have history_registers, history_memory, history_pc)
    - out_path: output gif path
    - fps: frames per second
    - max_frames: if set, will subsample frames to at most this many (keeps GIF small)
    - dpi: image DPI for saved frames
    """
    # histories
    reg_hist = np.array(gpu.history_registers)  # (cycles, threads, regs)
    mem_hist = np.array(gpu.history_memory)     # (cycles, mem_size)
    pc_hist  = np.array(gpu.history_pc)         # (cycles, threads)

    cycles = reg_hist.shape[0]
    if cycles == 0:
        raise RuntimeError("No history recorded. Run gpu.run(...) before calling save_animation().")

    # Optionally subsample frames to keep gif size small
    if max_frames is not None and cycles > max_frames:
        # pick frame indices evenly
        indices = np.linspace(0, cycles-1, max_frames, dtype=int)
    else:
        indices = np.arange(cycles)

    tmpdir = tempfile.mkdtemp(prefix="tinygpu_frames_")
    frames = []

    try:
        for i, frame_idx in enumerate(indices):
            # Use history up to this cycle so animation shows evolution (cumulative)
            regs_frame = reg_hist[frame_idx].T
            mem_frame  = mem_hist[frame_idx][np.newaxis, :]  # keep as 2-D
            pc_frame   = pc_hist[frame_idx][np.newaxis, :]

            # Prepare heatmaps: reshape registers to (cycles, threads*regs)
            regs_reshaped = regs_frame.reshape(regs_frame.shape[0], -1).T

            fig = plt.figure(figsize=(10, 7), dpi=dpi)
            gs = fig.add_gridspec(3, 1, height_ratios=(2, 1, 0.5), hspace=0.4)

            ax_regs = fig.add_subplot(gs[0, 0])
            im1 = ax_regs.imshow(regs_reshaped, aspect='auto', cmap='inferno', origin='lower')
            ax_regs.set_title(f"Registers up to cycle {frame_idx} (threads × regs)")
            plt.colorbar(im1, ax=ax_regs, fraction=0.02, pad=0.01)

            ax_mem = fig.add_subplot(gs[1, 0])
            im2 = ax_mem.imshow(mem_frame, aspect='auto', cmap='plasma',
                    origin='lower',
                    vmin=np.min(mem_frame),
                    vmax=np.max(mem_frame))
            ax_mem.set_title("Memory over time")
            plt.colorbar(im2, ax=ax_mem, fraction=0.02, pad=0.01)

            ax_pc = fig.add_subplot(gs[2, 0])
            pcs = pc_frame  # (threads, cycles_so_far)
            im3 = ax_pc.imshow(pcs, aspect='auto', cmap='viridis', origin='lower')
            ax_pc.set_title("Program Counter (per-thread) over time")
            plt.colorbar(im3, ax=ax_pc, fraction=0.02, pad=0.01)

            # save frame
            fname = os.path.join(tmpdir, f"frame_{i:05d}.png")
            plt.tight_layout(pad=0.2)
            fig.savefig(fname, bbox_inches='tight')
            plt.close(fig)
            frames.append(fname)

        # compose gif
        images = []
        for fname in frames:
            images.append(imageio.imread(fname))
        imageio.mimsave(out_path, images, fps=fps, loop=0)  # loop forever

    finally:
        # clean up temporary frames (comment this line to keep frames)
        shutil.rmtree(tmpdir)

    return out_path