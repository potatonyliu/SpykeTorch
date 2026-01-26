'''
This tool inspects the npz files for their dimentions.
'''

import numpy as np
from pathlib import Path

def inspect_npz():
    data = Path(__file__).parents[1] / 'data' / 'decoded_npz' / 'ncaltech101' / 'events'
    raw_data = Path(__file__).parents[1] / 'data' / 'raw' / 'ncaltech101' / 'events'
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
        spike_count.append(len(p) if len(p)==len(y) and len(x)==len(t) and len(x)==len(y) else print("lenths do not match"))
        timesteps.append(t.max()-t.min())
        x_min.append(x.min())
        x_max.append(x.max())
        y_min.append(y.min())
        y_max.append(y.max())

    return width, height, spike_count, timesteps, x_min, x_max, y_min, y_max



# Raw .bin stats
def inspect_raw():
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

def __main__():
    
    width, height, spike_count, timesteps, x_min, x_max, y_min, y_max = inspect_npz()
    raw_width, raw_height, raw_spike_count, raw_timesteps, raw_x_min, raw_x_max, raw_y_min, raw_y_max = inspect_raw() 

        

        print(f'''Data Statistics:
          width range: {min(width)} - {max(width)}
          height range: {min(height)} - {max(height)}
          spike count range: {min(spike_count)} - {max(spike_count)}
          timesteps range: {min(timesteps)} - {max(timesteps)}
          x_min range: {min(x_min)} - {max(x_min)}
          x_max range: {min(x_max)} - {max(x_max)}
          y_min range: {min(y_min)} - {max(y_min)}
          y_max range: {min(y_max)} - {max(y_max)}
          ''')


        print(f'''
    RAW BIN Data Statistics:
          width range: {min(raw_width)} - {max(raw_width)}
          height range: {min(raw_height)} - {max(raw_height)}
          spike count range: {min(raw_spike_count)} - {max(raw_spike_count)}
          timesteps range: {min(raw_timesteps)} - {max(raw_timesteps)}
          x_min range: {min(raw_x_min)} - {max(raw_x_min)}
          x_max range: {min(raw_x_max)} - {max(raw_x_max)}
          y_min range: {min(raw_y_min)} - {max(raw_y_min)}
          y_max range: {min(raw_y_max)} - {max(raw_y_max)}
    ''')

