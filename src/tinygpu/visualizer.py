# src/tinygpu/visualizer.py
import matplotlib.pyplot as plt
import numpy as np

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
