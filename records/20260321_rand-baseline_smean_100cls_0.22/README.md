Random weights baseline for the 0.36 arch (100 classes, L1 perfect weights, 4ch+32ch). Accuracy 0.22 with smean eval. Paired with the SNN run in 20260321_snn_smean_100cls_0.38, which gets +0.16 over this.

L1 had 4 perfect preset orientation filters and was not STDP-trained. L2 had 32 channels, kernel 11, 5 epochs.
