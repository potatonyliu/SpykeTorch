This run is the first run that created recognizable distinct features in visualized kernels. The main adaptation made in this version is to include a training_threshold and a passing_threshold for each layer to allow winners to be selected at later timesteps, while prevent loss of spike rate when training the second layer. Spike rate was managed from 0.1158 to 0.0750 in layer 2. In contrast to previous attempts where spikes eventually go extinct during training.

# Future Steps
1. Add more classes in our dataset. Right now we are using 2 classes for testing, and the model predicts at 100% accuracy, which is not meaningful.
2. Continue tuning parameters of L2, right now the features are not converging perfecly.
3. Continue tuning up inhibition radius or down winners count for L1, right now diversity is low. This may be resolved from adding more classes.
4. Adjust get_k_winners(). This is an experimental step, but we have been seeing early first spikes in the first few timesteps (4-6) even if training threshold is high. Our current method throws away most data and only use the earliest spikes. An ideal model should use ALL output spikes for STDP.
