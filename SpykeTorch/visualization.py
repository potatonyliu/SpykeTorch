from functools import wraps
from math import ceil, sqrt

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path


def _to_numpy(data):
    """
    Recursively convert tensors to numpy arrays within a data structure.

    Args:
        data: tensor or tensors in dict, list, tupple.

    Returns:
        Any: return in the same data structure as argument, with tensors converted to numpy arrays.
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, dict):
        return {k: _to_numpy(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_to_numpy(item) for item in data]
    elif not isinstance(data, np.ndarray):
        raise TypeError(
            "Can only convert tensors, dictories, lists and tuples of tensors to numpy array for visualization."
        )
    return data


def ensure_numpy(func):
    """
    Decorator to make all tensor arguments into numpy array for visualization.

    Args:
        func (callable): function to ensure arguments are numpy arrays.

    Return:
        callable: wrapped original function that converts tensors to numpy at run time.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        args = [_to_numpy(a) if isinstance(a, (torch.Tensor, dict, list, tuple, np.ndarray)) else a for a in args]
        return func(*args, **kwargs)
    return wrapper


@ensure_numpy
def spikes_over_time(spk):
    """
    Simple graph of total spike-count over time.

    Args:
        spk(tensor): Spike tensor.
    """
    rate_t = np.sum(spk, axis=(1, 2, 3))

    plt.figure()
    plt.plot(rate_t)
    plt.title("Fireing rate over time")
    plt.xlabel("t")
    plt.ylabel("Total spikes")
    plt.show()


@ensure_numpy
def spikes_map(spk):
    """
    Simple map of total spikes and their 2D distribution.

    Args:
        spk(tensor): Spike tensor.
    """
    spikes_map = np.sum(spk, axis=(0, 1))

    plt.figure()
    plt.imshow(spikes_map)
    plt.title("Spike count summed over time")
    plt.colorbar()
    plt.show()

@ensure_numpy
def plot_weights(weights, save_path: Path | None = None):
    """
    Plot weights of shape [f_out, f_in, kH, kW]

    Args:
        weights(tensor): weight tensor directly from the model parameter, e.g. conv1.weight.
        save_dir(Path): if set, save the weight as an image into this directory and not show the plot.
     """
    kernels = np.average(weights, 1)
    f_out, kH, kW = kernels.shape
    nrows = ceil(sqrt(f_out))
    ncols = ceil(f_out / nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize = (16,12))
    for i in range(f_out):
        title = f"Kernel {i}"
        axes[i // ncols, i % ncols].set_title(title)
        axes[i // ncols, i % ncols].axis('off')
        axes[i // ncols, i % ncols].imshow(kernels[i], cmap='RdBu', vmin = 0, vmax = 1, interpolation = 'nearest')
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


@ensure_numpy
def plot_weights_detailed(weights, save_dir: Path, prefix: str):
    """
    Plot each output kernel's per-input-channel weights as separate images.

    One PNG per output channel, showing all f_in input sub-kernels side by side.
    Saved as {save_dir}/{prefix}_f{i:02d}.png.

    Args:
        weights: weight tensor [f_out, f_in, kH, kW]
        save_dir: directory to save images into
        prefix: filename prefix, e.g. "l1_epoch_0_kernels_detail"
    """
    f_out, f_in, kH, kW = weights.shape
    for i in range(f_out):
        fig, axes = plt.subplots(1, f_in, figsize=(2.5 * f_in, 3))
        if f_in == 1:
            axes = [axes]
        fig.suptitle(f"Output kernel {i}", fontsize=11)
        for c in range(f_in):
            axes[c].imshow(weights[i, c], cmap='RdBu', vmin=0, vmax=1, interpolation='nearest')
            axes[c].set_title(f"in {c}", fontsize=8)
            axes[c].axis('off')
        plt.tight_layout()
        plt.savefig(save_dir / f"{prefix}_f{i:02d}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


def save_curves(history, layer, log_dir):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0,0].plot(history["diversity"])
    axes[0,0].set_title("Diversity")
    axes[0,1].plot(history["spike_rate"])
    axes[0,1].set_title("Spike Rate")
    axes[1,0].plot(history["w_mean"])
    axes[1,0].set_title("Weight Mean")
    axes[1,1].plot(history["w_std"])
    axes[1,1].set_title("Weight Std")
    plt.suptitle(f"Layer {layer} Training")
    plt.tight_layout()
    plt.savefig(log_dir / f"l{layer}_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)




