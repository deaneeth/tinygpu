import matplotlib.pyplot as plt
import numpy as np

def visualize(gpu):
    reg_hist = np.array(gpu.history_registers)
    mem_hist = np.array(gpu.history_memory)

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    # Registers: shape (cycles, threads, regs)
    im1 = axs[0].imshow(
        reg_hist[:, :, :].reshape(reg_hist.shape[0], -1).T,
        aspect="auto", cmap="inferno"
    )
    axs[0].set_title("Registers over time (threads × regs)")
    fig.colorbar(im1, ax=axs[0])

    # Memory: shape (cycles, mem_size)
    im2 = axs[1].imshow(mem_hist.T, aspect="auto", cmap="plasma")
    axs[1].set_title("Memory over time")
    fig.colorbar(im2, ax=axs[1])

    plt.show()
