import matplotlib.pyplot as plt
import numpy as np
import torch
from functools import wraps

def _to_numpy(data):
    '''
    Recursively convert tensors to numpy arrays within a data structure.

    Args:
        data: tensor or tensors in dict, list, tupple.

    Returns:
        Any: return in the same data structure as argument, with tensors converted to numpy arrays.
    '''
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, dict):
        return {k:_to_numpy(v) for k,v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_to_numpy(item) for item in data]
    elif not isinstance(data, np.ndarray):
        raise TypeError("Can only convert tensors, dictories, lists and tuples of tensors to numpy array for visualization.")
    return data

def ensure_numpy(func):
    '''
    Decorator to make all tensor arguments into numpy array for visualization.

    Args:   
        func (callable): function to ensure arguments are numpy arrays.

    Return:
        callable: wrapped original function that converts tensors to numpy at run time.
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        np_args = [_to_numpy(a) for a in args]
        np_kwargs = {k:_to_numpy(v) for k,v in kwargs.items()}
        return func(*np_args, **np_kwargs)
    return wrapper

@ensure_numpy
def spikes_over_time(spk):
    '''
    Simple graph of total spike-count over time.

    Args:
        spk(tensor): Spike tensor.
    '''
    rate_t = np.sum(spk, axis=(1,2,3))

    plt.figure()
    plt.plot(rate_t)
    plt.title("Fireing rate over time")
    plt.xlabel("t")
    plt.ylabel("Total spikes")
    plt.show()
    
@ensure_numpy
def spikes_map(spk):
    '''
    Simple map of total spikes and their 2D distribution.
    
    Args:
        spk(tensor): Spike tensor.
    '''
    spikes_map = np.sum(spk, axis=(0,1))

    plt.figure()
    plt.imshow(spikes_map)
    plt.title("Spike count summed over time")
    plt.colorbar()
    plt.show()
