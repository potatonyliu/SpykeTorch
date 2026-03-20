# Changes I made
A fork of [SpykeTorch](https://github.com/miladmozafari/SpykeTorch). I have changed core functions of the library SpykeTorch to adapt for temporal, real-world event data.

The original library processes data converted from static images with single permanent spikes - once a neuron fires, it stays on. This fork modifies core functions to support temporal spikes from event cameras, where neurons can fire repeatedly.

**Note**: SpykeTorch was built for research purposes to demonstrate the efficiency of training with single spike, synthesized data. This attempt to extend its usage lowers its efficiency. It is mainly for my learning :)

## Core Functions Changed
`snn.py`
- `STDP.forward()`: Replaces simple pre-post ordering with an exponentially decaying weight update rule, so more proximate pre-synaptic spikes get more update. Vectorized operation for each winner. 

`functional.py`
- `get_k_winners()`: Instead of trivially selecting earliest spike time when spikes are permanent, find earliest spike with argmax(), and apply inhibition around a radius and for the feature.
- `fire()`: Support accumulation of potential over timesteps with decay (LIF), which is essential for optic flow.

## New Pipeline
`run_experiment.py`
- Inspired by [KellerJordan](https://github.com/KellerJordan/modded-nanogpt/tree/master/records)'s experimental design. Made logging much more systematic and trackable in `run.txt`, and parameters immediately comparable through configurations. Some of the logs are in `records/`.
- Stratified train/test split

## New Model
- `LiuDeep.py`: 2 layers, train on N-Caltech101 dataset. Currently scaling from 2 classes to more classes.
-  Separated `train_threshold` and `pass_threshold` to preserve spike counts for later layers
-  Preliminary results in `records/`.

## Evaluation
The original implementation of SpykeTorch uses a LinearSVC layer on the last layer's output, summing across all timesteps (C, H, W). When I use this to evaluate, I face some issues:
1. The linear fitting is extremely slow, given the shape of the final layer 32*40*55 features over 101 classes.
2. My model constantly produce lower accuracy (~0.29) than random weight (~0.36), even when there is visually distinctive and clean features on kernels.

In the [Kirkland paper](https://ieeexplore.ieee.org/document/9207075), they used a third layer, where each neuron correspond to a class for classification. It worked for 2 classes since there is competition in WTA, but it would likely fail for 101 classes on unsupervised STDP. I then switched to taking mean across all spatial dimensions, and only perserve temporal data across each channel. This mimics the idea of looking for the strongly activated features for classification, rather than the actual location of each spike. This method makes sense particularly because N-Caltech101 has optic flow. Summing over temporal dimension messes up spatial definition, since all edges and shape sweeps over the space over time. The linearSVC layer that takes input from each channel at each timestep is more loyal to the design of a spiking convolutional neural network, where we care about the presense and composition of high-level features.

So far:
- Baselines:
    - Linear layer that takes input from my random weight model gives 0.19 accuracy on the dataset.
    - Linear layer that takes input directly from raw data gives 0.17 accuracy on the dataset.
- Records:
    - After training, the same linear layer gives 0.36.
 
I think this is the most appropriate evalaution at this stage, but I do want to note that if I evaluate with sum over time and preserving all spatial dimensions, our model performs significantly worse than random weight baseline, potentially due to reasons sketched above.

<img width="588" height="446" alt="l2_epoch_1_kernels" src="https://github.com/user-attachments/assets/d228c19f-f527-4775-9d3e-ca6bfe145ae3" />

L2 kernels after 2 epoches in a run. Visible and distinctive features shown on the kernels.

# Instructions
To download and process the N-Caltech101 dataset:
```
python scripts/download_ncaltech101.py
```
**Note**: you might need to download manually and leave the zip file in `data/raw/ncaltech101/`, then run the script, since the link updates from Mendeley.

**Static Datasets and Temporal Datasets**
The main feature of this fork compared to the original SpykeTorch library is that it added support to temporal datasets. The original library only support static datasets, which is defined as follow:
1. Once a spike occurs, it stays on until the end of the timesteps for the sample.
2. Object and camera stays static, there is no optic flow.

Static datasets are generally generated from static images.

Dynamic datasets are event data with optic flow. This means event videos captured with event cameras with camera movement and / or object movements, a single pixel can spike, then goes back to 0 and repeat.


# Original README for SpykeTorch
High-speed simulator of convolutional spiking neural networks with at most one spike per neuron.

<img src="https://raw.githubusercontent.com/miladmozafari/SpykeTorch/master/logo.png" alt="alt text" width=50%>

SpykeTorch is a PyTorch-based simulator of convolutional spiking neural networks, in which the neurons emit at most one spike per stimulus. SpykeTorch supports STDP and Reward-modulated STDP learning rules. The current code is the early object oriented version of this simulator and you can find the documentation in docs folder in PDF format or in our lab website (http://cnrl.ut.ac.ir/SpykeTorch/doc/) in HTML format. Since SpykeTorch is fully compatible with PyTorch, you can easily use it if you know PyTorch. A tutorial is available in the paper titled "SpykeTorch: Efficient Simulation of Convolutional Spiking Neural Networks with at most one Spike per Neuron" which introduces the SpykeTorch package (https://www.frontiersin.org/articles/10.3389/fnins.2019.00625/full).

**IMPORTANT**: Current version of SpykeTorch does not support negative synaptic weights.

To setup this package, you can install Anaconda or Miniconda
```
# Clone the repository
git clone https://github.com/miladmozafari/SpykeTorch

# Create a new conda environment
conda create -n spyketorchproject python=3
conda activate spyketorchproject

# Install all the dependencies
pip install -r requirements.txt
```
```
# Alternatively, one can just run the following command
pip install git+https://github.com/miladmozafari/SpykeTorch.git
```

**Scripts info:**
 - [`MozafariShallow.py`](MozafariShallow.py): Reimplementation of the paper "First-Spike-Based Visual Categorization Using Reward-Modulated STDP" (https://ieeexplore.ieee.org/document/8356226/).
 - [`MozafariDeep.py`](MozafariDeep.py): Reimplementation of the paper "Bio-Inspired Digit Recognition Using Reward-Modulated Spike-Timing-Dependent Plasticity in Deep Convolutional Networks" (https://www.sciencedirect.com/science/article/abs/pii/S0031320319301906).
 - [`KheradpishehDeep.py`](KheradpishehDeep.py): Reimplementation of the paper "STDP-based spiking deep convolutional neural networks for object recognition" (https://www.sciencedirect.com/science/article/pii/S0893608017302903).
 - [`tutorial.ipynb`](tutorial.ipynb): A brief tutorial on designing, training, and evaluating a SNN with SpykeTorch.


