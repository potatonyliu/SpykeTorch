First 3-layer SNN experiments. Best result 0.39 — does not beat the 2-layer best (0.42).

Architecture: L1=4ch k5, L2=16ch k7, L3=32ch k11. All STDP trained, smean eval. (Best run is this run: run_20260323T22-34-11-747459)

## Training Stats
- L1 Epoch 0: diversity=0.678, w_mean=0.306, time=154s
- L1 Epoch 1: diversity=0.699, w_mean=0.292, time=158s
- L2 Epoch 0: diversity=0.628, w_mean=0.209, time=371s
- L2 Epoch 1: diversity=0.656, w_mean=0.199, time=374s
- L3 Epoch 0: diversity=0.916, w_mean=0.050, time=536s
- L3 Epoch 1: diversity=0.926, w_mean=0.047, time=537s

L3 weights collapsed to near zero (w_mean ~0.05). The follow-up runs tried raising L3 thresholds to fix this, which helped the weights but didn't improve accuracy.

## All 3-layer Runs

| Run | L2 config | L3 thresh (train/pass) | Result | Notes |
|-----|-----------|------------------------|--------|-------|
| run_20260323T21-18-16-529503 | 64ch k11 | 100/50 | killed | First attempt, killed before finishing |
| run_20260323T21-19-18-566115 | 64ch k11 (from checkpoint) | 100/50 | 0.35 | L3 only trained; weights near zero (w_mean 0.024) |
| run_20260323T22-34-11-747459 ⭐ | 16ch k7 | 100/50 | **0.39** | Reduced L2 bottleneck; full fresh training |
| run_20260324T15-48-13-461150 | 16ch k7 (from checkpoint) | 400/200 | 0.38 | Higher L3 thresh → w_mean 0.12, still no improvement |
| run_20260324T20-44-34-306676 | 16ch k7 (from checkpoint) | 800/400 | 0.38 | Even higher thresh → w_mean 0.24, still 0.38 |
| run_20260324T20-44-45-002990 | 16ch k7 (from checkpoint) | 800/400 | 0.38 | Repeat of above, confirmed |

The first two runs loaded L1+L2 from the same checkpoint as the failed 64ch+64ch run. The big jump to 0.39 came from the fresh training with a narrower L2 (16ch instead of 64ch), which likely helped L3 see more abstract features. Raising L3 thresholds after that didn't help — 0.38 seemed to be the floor for that checkpoint. The ceiling appears to be below the 2-layer result.
