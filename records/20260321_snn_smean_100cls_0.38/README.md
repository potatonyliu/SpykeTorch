SNN with perfect L1 weights, smean eval, 100 classes. 0.38 accuracy (+0.16 over rand baseline).

L1 had 4 preset orientation filters (not STDP trained). L2 was 32 channels, kernel 11x11, 5 epochs.

L1 diversity stayed low the whole time (0.17 → 0.21), with winners stuck on timesteps 4-6. This was a known problem at the time - L1 was not learning diverse features. L2 compensated decently (diversity 0.85 → 0.86) but was limited by the bottleneck at 4 channels.

This was the baseline before switching to STDP-trained L1 and smean eval. The L1 STDP runs (starting 20260322) improved to 0.42.
