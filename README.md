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


