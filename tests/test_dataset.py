import torch
import numpy as np
from datasets.ncaltech101_dataset import npz_to_spike_tensor, NCaltechDataset
import pytest
from pathlib import Path

def test_npz_to_spike_tensor_logic():
    x = np.array([10, 20, 30], dtype=np.uint16)
    y = np.array([5, 5, 5], dtype=np.uint16)
    p = np.array([0, 1, 0], dtype=np.uint8) # OFF, ON, OFF
    t = np.array([0, 50, 100], dtype=np.uint32)

    T= 5
    H = 32
    W = 32

    spikes = npz_to_spike_tensor(x,y,p,t,T,H,W)
    print("Shape: ", spikes.shape)
    assert spikes.shape == (5,2,32,32), "Shape"
    assert spikes[0, 0, 5, 10] == 1.0, "First event missing!"
    assert spikes[4, 0, 5, 30] == 1.0, "Last event missing!"

@pytest.fixture(scope="module")
def inspect_npz():
    data = Path(__file__).parents[1] / 'data' / 'decoded_npz' / 'ncaltech101' / 'events'
    width = []
    height = []
    spike_count = []
    timesteps = []
    x_min = []
    x_max = []
    y_min = []
    y_max = []

    print(data)
    # print(*(f.stem for f in data.rglob('*.npz')), sep='\n')

    for i, sample in enumerate(data.rglob('*.npz')):
        dump = np.load(sample)
        x = dump['x']
        y = dump['y']
        p = dump['p']
        t = dump['t']
        width.append(x.max()-x.min()+1)
        height.append(y.max()-y.min()+1)
        assert len(p)==len(y) and len(x)==len(t) and len(x)==len(y)
        spike_count.append(len(p))
        timesteps.append(t.max()-t.min())
        x_min.append(x.min())
        x_max.append(x.max())
        y_min.append(y.min())
        y_max.append(y.max())

    return width, height, spike_count, timesteps, x_min, x_max, y_min, y_max



@pytest.fixture(scope="module")
def inspect_raw():
    raw_data = Path(__file__).parents[1] / 'data' / 'raw' / 'ncaltech101' / 'events'
    raw_width = []
    raw_height = []
    raw_x_min = []
    raw_x_max = []
    raw_y_min = []
    raw_y_max = []
    raw_spike_count = []
    raw_timesteps = []

    print("RAW BIN data path:", raw_data)

    for sample in raw_data.rglob('*.bin'):
        data = sample.read_bytes()
        if len(data) % 5 != 0:
            print("bad file:", sample)
            continue

        arr = np.frombuffer(data, dtype=np.uint8).reshape(-1, 5)

        b0 = arr[:, 0]
        b1 = arr[:, 1]
        b2 = arr[:, 2]
        b3 = arr[:, 3]
        b4 = arr[:, 4]

        x = b0.astype(np.uint16)
        y = b1.astype(np.uint16)

        t = ((b2 & 0x7F).astype(np.uint32) << 16) | \
            (b3.astype(np.uint32) << 8) | \
            b4.astype(np.uint32)

        raw_width.append(x.max() - x.min() + 1)
        raw_height.append(y.max() - y.min() + 1)
        raw_x_min.append(x.min())
        raw_x_max.append(x.max())
        raw_y_min.append(y.min())
        raw_y_max.append(y.max())
        raw_spike_count.append(len(t))
        raw_timesteps.append(t.max() - t.min())

    return raw_width, raw_height, raw_spike_count, raw_timesteps, raw_x_min, raw_x_max, raw_y_min, raw_y_max 

def test_dimentions(inspect_npz, inspect_raw):
    
    width, height, spike_count, timesteps, x_min, x_max, y_min, y_max = inspect_npz
    raw_width, raw_height, raw_spike_count, raw_timesteps, raw_x_min, raw_x_max, raw_y_min, raw_y_max = inspect_raw

    # ---- NPZ ranges ----
    npz_width_range = (min(width), max(width))
    npz_height_range = (min(height), max(height))
    npz_spike_count_range = (min(spike_count), max(spike_count))
    npz_timesteps_range = (min(timesteps), max(timesteps))
    npz_x_min_range = (min(x_min), max(x_min))
    npz_x_max_range = (min(x_max), max(x_max))
    npz_y_min_range = (min(y_min), max(y_min))
    npz_y_max_range = (min(y_max), max(y_max))

    print(f'''
    Data Statistics (NPZ):
          width range: {npz_width_range[0]} - {npz_width_range[1]}
          height range: {npz_height_range[0]} - {npz_height_range[1]}
          spike count range: {npz_spike_count_range[0]} - {npz_spike_count_range[1]}
          timesteps range: {npz_timesteps_range[0]} - {npz_timesteps_range[1]}
          x_min range: {npz_x_min_range[0]} - {npz_x_min_range[1]}
          x_max range: {npz_x_max_range[0]} - {npz_x_max_range[1]}
          y_min range: {npz_y_min_range[0]} - {npz_y_min_range[1]}
          y_max range: {npz_y_max_range[0]} - {npz_y_max_range[1]}
    ''')

    # ---- RAW BIN ranges ----
    raw_width_range = (min(raw_width), max(raw_width))
    raw_height_range = (min(raw_height), max(raw_height))
    raw_spike_count_range = (min(raw_spike_count), max(raw_spike_count))
    raw_timesteps_range = (min(raw_timesteps), max(raw_timesteps))
    raw_x_min_range = (min(raw_x_min), max(raw_x_min))
    raw_x_max_range = (min(raw_x_max), max(raw_x_max))
    raw_y_min_range = (min(raw_y_min), max(raw_y_min))
    raw_y_max_range = (min(raw_y_max), max(raw_y_max))

    print(f'''
    RAW BIN Data Statistics:
          width range: {raw_width_range[0]} - {raw_width_range[1]}
          height range: {raw_height_range[0]} - {raw_height_range[1]}
          spike count range: {raw_spike_count_range[0]} - {raw_spike_count_range[1]}
          timesteps range: {raw_timesteps_range[0]} - {raw_timesteps_range[1]}
          x_min range: {raw_x_min_range[0]} - {raw_x_min_range[1]}
          x_max range: {raw_x_max_range[0]} - {raw_x_max_range[1]}
          y_min range: {raw_y_min_range[0]} - {raw_y_min_range[1]}
          y_max range: {raw_y_max_range[0]} - {raw_y_max_range[1]}
    ''')

    assert npz_width_range == raw_width_range, f"width mismatch: {npz_width_range} vs {raw_width_range}"
    assert npz_height_range == raw_height_range, f"height mismatch: {npz_height_range} vs {raw_height_range}"
    assert npz_spike_count_range == raw_spike_count_range, f"spike count mismatch: {npz_spike_count_range} vs {raw_spike_count_range}"
    assert npz_timesteps_range == raw_timesteps_range, f"timesteps mismatch: {npz_timesteps_range} vs {raw_timesteps_range}"

    assert npz_x_min_range == raw_x_min_range, f"x_min mismatch: {npz_x_min_range} vs {raw_x_min_range}"
    assert npz_x_max_range == raw_x_max_range, f"x_max mismatch: {npz_x_max_range} vs {raw_x_max_range}"
    assert npz_y_min_range == raw_y_min_range, f"y_min mismatch: {npz_y_min_range} vs {raw_y_min_range}"
    assert npz_y_max_range == raw_y_max_range, f"y_max mismatch: {npz_y_max_range} vs {raw_y_max_range}"
