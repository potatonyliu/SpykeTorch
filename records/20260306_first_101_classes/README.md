This record adds all 101 classes in the N-Caltech101 dataset, with little changes in other parameters. We tuned down learning rates for better generalization. The linearSVC layer did not converge and need more iterations, which is expected. However, it already achieves an accuracy of 0.39 on testing dataset, which is significantly better than random. I need to perform a baseline test to know how well the linear layer itself performs on raw data.

# Major Bottleneck
Layer 1 is still suffering from low diversity ranging from 0.2365 to 0.2582. Winning timesteps are stuck on the same timesteps.

# Future Steps
1. Reduce number of winners to encourage competition
2. Tune up inhibition radius
3. Increase iterations for LinearSVC
