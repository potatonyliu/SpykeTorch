## What I Changed
- Used perfect preset weights for L1, trying to avoid the mainly horizontal features in N-Caltech101 (perhaps because of the computer screen edge in the recording).
- Larger kernels (11) for L2 and higher threshold.
- Evaluated with mean(dim(-1, -2)) on linear layer, no spatial information. This is for better efficiency and also prevent linear layer to take away the credit of classification.

## Results
Accuracy at 0.36. No significant improvement compared to previous linear-only baseline runs with linear layers on the whole spatial features. Yes significant improvement compared to baseline linear layer with mean(dim(-1, -2)). However, this comparison is not fair I realized, since linear layer from raw input also has 2 channels, while this run has 32.
