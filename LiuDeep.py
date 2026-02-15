###################################################################################
# Reference:                                                                      #
# Kheradpisheh, Saeed Reza, et al.                                                #
# "STDP-based spiking deep convolutional neural networks for object recognition." #
# Neural Networks 99 (2018): 56-67.                                               #
#                                                                                 #
###################################################################################

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.nn.parameter import Parameter
import torchvision
import numpy as np
from SpykeTorch import snn
from SpykeTorch import functional as sf
from SpykeTorch import visualization as vis
from SpykeTorch import utils
from torchvision import transforms

import matplotlib.pyplot as plt

from datasets.ncaltech101_dataset import NCaltechDataset

from tqdm import tqdm
import time

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("[DEVICE]: mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("[DEVICE]: CUDA")
else:
    device = torch.device("cpu")
    print("[DEVICE]: CPU\nCheck CUDA/mps availablity.")

# Testing global variables
MAX_ITERS = 10

class LiuNCaltech101(nn.Module):
    def __init__(self, timesteps):
        super(LiuNCaltech101, self).__init__()

        self.conv1 = snn.Convolution(2, 16, 5, 0.8, 0.1)
        self.conv1_t = 10
        self.k1 = 8
        self.r1 = 20

        self.conv2 = snn.Convolution(16, 150, 2, 0.8, 0.05)
        self.conv2_t = 1
        self.k2 = 8
        self.r2 = 1

        self.stdp1 = snn.STDP(self.conv1, (0.008, -0.035), timesteps = timesteps)
        self.stdp2 = snn.STDP(self.conv2, (0.008, -0.035), timesteps = timesteps)
        self.max_ap = Parameter(torch.Tensor([0.15]))

        self.ctx = {"input_spikes":None, "potentials":None, "output_spikes":None, "winners":None}
        self.spk_cnt1 = 0
        self.spk_cnt2 = 0
    
    def save_data(self, input_spike, potentials, output_spikes, winners):
        self.ctx["input_spikes"] = input_spike
        self.ctx["potentials"] = potentials
        self.ctx["output_spikes"] = output_spikes
        self.ctx["winners"] = winners

    def forward(self, input, max_layer):
        input = sf.pad(input.float(), (2,2,2,2), 0)
        if self.training:
            # torch.mps.synchronize()
            # t0 = time.time()
            cur = self.conv1(input)
            # torch.mps.synchronize()
            # t1 = time.time()
            spk, pot = sf.fire(cur, self.conv1_t, 0.95, True)
            # torch.mps.synchronize()
            # t2 = time.time()

            if max_layer == 1:
                # self.spk_cnt1 += 1
                # if self.spk_cnt1 >= 500:
                #     self.spk_cnt1 = 0
                #     ap = torch.tensor(self.stdp1.learning_rate[0][0].item(), device=self.stdp1.learning_rate[0][0].device) * 2
                #     ap = torch.min(ap, self.max_ap)
                #     an = ap * -0.75
                #     self.stdp1.update_all_learning_rate(ap.item(), an.item())
                # pot = sf.pointwise_inhibition(pot)
                # spk = pot.sign()
                # torch.mps.synchronize()
                # t3 = time.time()
                winners = sf.get_k_winners(pot, spk, self.k1, self.r1)
                # print(f"conv1: {t1-t0:.3f}s | fire: {t2-t1: .3f}s | get_k_winner: {t3-t2:.3f}s.")
                self.save_data(input, pot, spk, winners)
                return spk, pot
            # spk_in = sf.pad(sf.pooling(spk, 2, 2, 1), (1,1,1,1))
            # spk_in = sf.pointwise_inhibition(spk_in)
            # pot = self.conv2(spk_in)
            # spk, pot = sf.fire(pot, self.conv2_t, 0.95, True)
            # if max_layer == 2:
            #     pot = sf.pointwise_inhibition(pot)
            #     spk = pot.sign()
            #     winners = sf.get_k_winners(pot, spk, self.k2, self.r2)
            #     self.save_data(spk_in, pot, spk, winners)
            #     return spk, pot
            # spk_out = sf.pooling(spk, 2, 2, 1)
            # return spk_out
        else:
            pot = self.conv1(input)
            spk, pot = sf.fire(pot, self.conv1_t, 0.95, True)
            pot = self.conv2(sf.pad(sf.pooling(spk, 2, 2, 1), (1,1,1,1)))
            spk, pot = sf.fire(pot, self.conv2_t, 0.95, True)
            spk = sf.pooling(spk, 2, 2, 1)
            return spk

    def stdp(self, layer_idx):
        if layer_idx == 1:
            self.stdp1(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
        if layer_idx == 2:
            self.stdp2(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

def train_unsupervise(network, data, layer_idx):
    network.train()
    for i in tqdm(range(len(data))):
        #TODO: temp
        data_in = data[i][0]
        # data_in = data[i]
        # torch.mps.synchronize()
        # t0 = time.time()
        network(data_in, layer_idx)
        # torch.mps.synchronize()
        # t1 = time.time()
        network.stdp(layer_idx)
        # torch.mps.synchronize()
        # t2 = time.time()
        # print(f"forward: {t1-t0:.3f}s | stdp: {t2-t1:.3f}s.")

def test(network, data, target, layer_idx):
    network.eval()
    ans = [None] * len(data)
    t = [None] * len(data)
    for i in range(len(data)):
        data_in = data[i]
        output,_ = network(data_in, layer_idx).max(dim = 0)
        ans[i] = output.reshape(-1).cpu().numpy()
        t[i] = target[i]
    return np.array(ans), np.array(t)

root = Path(__file__).parent
T = 60
train_dataset = NCaltechDataset(root_dir=root, T=T, H=173, W=233)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

small_dataset = torch.utils.data.Subset(train_dataset, range(10))

model = LiuNCaltech101(timesteps = T)

vis.plot_weights(model.conv1.weight)
train_unsupervise(model, train_dataset, 1)
vis.plot_weights(model.conv1.weight)

## Temp testing
# data = train_dataset[161]
# pot, spk = model(data[0], max_layer=1)
# inp = model.ctx["input_spikes"]
# pot = model.ctx["potentials"]
# spk = model.ctx["output_spikes"]

# vis.spikes_over_time(spk)
# vis.spikes_map(spk)

# print("spike pot mean: ",pot[spk==1].mean())
# print("non-spike pot mean: ",pot[spk==0].mean())


def get_performance(X, y, predictions):
    correct = 0
    silence = 0
    for i in range(len(predictions)):
        if X[i].sum() == 0:
            silence += 1
        else:
            if predictions[i] == y[i]:
                correct += 1
    return (correct/len(X), (len(X)-(correct+silence))/len(X), silence/len(X))

# print(get_performance(train_X, train_y, predict_train))
# print(get_performance(test_X, test_y, predict_test))
