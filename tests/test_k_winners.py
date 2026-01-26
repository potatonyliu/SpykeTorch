import torch
from SpykeTorch import functional, utils, visualization

cur = utils.make_test_currents(20,10,10,10)
spk, pot = functional.fire(cur, 3, 0.95, True)
# visualization.spikes_over_time(spk)
# visualization.spikes_map(spk)
winners = functional.get_k_winners(pot, spk, 5)
print(winners)
