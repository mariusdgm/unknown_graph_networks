import numpy as np
import matplotlib.pyplot as plt

def show_matrix_with_cell_grid(
    A,
    title,
    *,
    figsize=(8, 6),
    vmin=None,
    vmax=None,
    grid_color="#C8C8C8",   # muted grey
    grid_alpha=0.35,        # make it softer
    grid_lw=0.6,            # thin lines
    show_ticks=True,
):
    A = np.asarray(A)
    nrows, ncols = A.shape

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(A, aspect="auto", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

    # put grid lines between cells
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color=grid_color, linestyle="-", linewidth=grid_lw, alpha=grid_alpha)

    # hide minor tick marks (but keep the grid)
    ax.tick_params(which="minor", bottom=False, left=False)

    if show_ticks:
        ax.set_xlabel("j")
        ax.set_ylabel("i")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.tight_layout()
    plt.show()
    
def plot_impulse_node_trajectories(inter_states, inter_times, title, ylim=(0, 1)):
    """
    inter_states: (K, Ksub+1, N)
    inter_times:  (K, Ksub+1)
    """
    inter_states = np.asarray(inter_states)
    inter_times  = np.asarray(inter_times)

    t = inter_times.reshape(-1)
    x = inter_states.reshape(-1, inter_states.shape[-1])

    plt.figure(figsize=(10, 4))
    for i in range(x.shape[1]):
        plt.plot(t, x[:, i], linewidth=1)
    plt.xlabel("time")
    plt.ylabel("opinion $x_i$")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.ylim(*ylim)
    plt.show()