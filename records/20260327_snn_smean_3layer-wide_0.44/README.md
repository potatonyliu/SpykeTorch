3-layer SNN with expanding width (8→32→64). New best at 0.44, first to beat the 2-layer ceiling (0.42).

Architecture: L1=8ch k5, L2=32ch k7, L3=64ch k11. All STDP trained, smean eval. (run_20260327T17-06-38-190636)

## Config
- L1: 8ch, k_winners=4, thresh 30/15, 2 epochs
- L2: 32ch, k_winners=16, thresh 250/75, 2 epochs
- L3: 64ch, k_winners=32, thresh 3000/500, 2 epochs
- LinearSVC: class_weight='balanced'

## Training Stats
- L1 Epoch 0: diversity=0.597, w_mean=0.349, time=393s
- L1 Epoch 1: diversity=0.660, w_mean=0.310, time=407s
- L2 Epoch 0: diversity=0.382, w_mean=0.527, time=1070s
- L2 Epoch 1: diversity=0.432, w_mean=0.496, time=1101s
- L3 Epoch 0: diversity=0.516, w_mean=0.380, time=1468s
- L3 Epoch 1: diversity=0.556, w_mean=0.364, time=1583s

## Related runs

| Run | Config | Result | Notes |
|-----|--------|--------|-------|
| run_20260327T17-05-29-942428 | 8→32→64, L1 k=2 | killed | L1 k=2 too restrictive, starved filters |
| run_20260327T17-06-38-190636 ⭐ | 8→32→64, L1 k=4 | **0.44** | Expanding width, best so far |
| run_20260327T17-07-46-402753 | 8→64→32, L2 k=32 | 0.41 | Kirkland-style wide L2, L2 diversity collapsed (0.27→0.37) |
| run_20260327T20-31-32-425039 | 8→32→64, 4 epochs | 0.43 | Same arch, double epochs — no improvement, all layers plateau by ep1-2 |

## Notes
Tried expanding (narrow→wide) vs Kirkland-style (narrow→wide→narrow). Expanding wins. The wide-L2 run (64ch) had terrible L2 diversity — too many filters with not enough distinct input from 8 L1 channels.

4 epochs didn't help either. Diversity curves flatten after epoch 1 for L1 and by epoch 2-3 for L2/L3. The bottleneck isn't training duration.
