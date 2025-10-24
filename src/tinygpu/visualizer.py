import matplotlib.pyplot as plt
import numpy as np

def visualize(gpu, show_pc=False):
    import matplotlib.pyplot as plt
    import numpy as np

    reg_hist = np.array(gpu.history_registers)
    mem_hist = np.array(gpu.history_memory)

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    im1 = axs[0].imshow(reg_hist.reshape(reg_hist.shape[0], -1).T,
                        aspect="auto", cmap="inferno")
    axs[0].set_title("Registers over time (threads × regs)")
    fig.colorbar(im1, ax=axs[0])

    im2 = axs[1].imshow(mem_hist.T, aspect="auto", cmap="plasma")
    axs[1].set_title("Memory over time")
    fig.colorbar(im2, ax=axs[1])

    if show_pc:
        pcs = np.array(gpu.pc)
        for t, pc in enumerate(pcs):
            axs[0].plot(pc, t, 'wo', markersize=2)
    plt.show()

