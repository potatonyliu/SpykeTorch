# Trained SNN — smean, L1 STDP, pass=15 ⭐ new best

**Type:** snn trained
**Eval:** smean (mean(dim=(-1,-2)), keeps T×C = 60×64 = 3,840 features)
**Result:** 0.42 accuracy (1,649 test samples)

## Config
- L1: STDP trained (L1_perfect_weights=false), passing_threshold=15, training_threshold=30, 2 epochs
- L2: out_channels=64, training_threshold=100, passing_threshold=50, 2 epochs
- LinearSVC: class_weight='balanced'

## Training Stats
- L1 Epoch 0: diversity=0.677, w_mean=0.329, time=201s
- L1 Epoch 1: diversity=0.716, w_mean=0.309, time=207s
- L2 Epoch 0: diversity=0.524, w_mean=0.260, time=1292s
- L2 Epoch 1: diversity=0.684, w_mean=0.196, time=2018s

## Notes
First run with L1 STDP training. New best result, beating perfect weights baseline (0.39).
- L2 diversity (0.684) is *lower* than perfect weights runs (~0.71) yet accuracy is higher (+0.03)
  → L1 STDP learns better features for N-Caltech101 than handcrafted orientation filters
- Compare: perfect weights 64ch/6ep=0.39, L1 STDP 64ch/2ep=0.42
- Paired with run_20260322T17-54-37-121374 (L1 pass=30 → 0.24) — confirms pass=15 is critical
