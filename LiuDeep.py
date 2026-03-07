###################################################################################
# Reference:                                                                      #
# Kheradpisheh, Saeed Reza, et al.                                                #
# "STDP-based spiking deep convolutional neural networks for object recognition." #
# Neural Networks 99 (2018): 56-67.                                               #
#                                                                                 #
###################################################################################

from pathlib import Path
from sys import stdout

import numpy as np
import torch
import torch.nn as nn
from datasets.ncaltech101_dataset import NCaltechDataset
from SpykeTorch import functional as sf
from SpykeTorch import snn
from SpykeTorch import visualization as vis
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("[DEVICE]: mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("[DEVICE]: CUDA")
else:
    device = torch.device("cpu")
    print("[DEVICE]: CPU\nCheck CUDA/mps availablity.")


class LiuNCaltech101(nn.Module):
    def __init__(self, config):
        super().__init__()

        T = config["timesteps"]
        l1 = config["layer1"]
        l2 = config["layer2"]

        self.conv1 = snn.Convolution(
            l1["in_channels"],
            l1["out_channels"],
            l1["kernel_size"],
            l1["w_mean"],
            l1["w_std"],
        )
        if "threshold" in l1:
            self.conv1_t_pass = l1["threshold"]
            self.conv1_t_train = l1["threshold"]
        else:
            self.conv1_t_train = l1["training_threshold"]
            self.conv1_t_pass = l1["passing_threshold"]
        self.k1 = l1["k_winners"]
        self.r1 = l1["inhibition_radius"]

        self.conv2 = snn.Convolution(
            l1["out_channels"],
            l2["out_channels"],
            l2["kernel_size"],
            l2["w_mean"],
            l2["w_std"],
        )
        if "threshold" in l2:
            self.conv2_t_pass = l2["threshold"]
            self.conv2_t_train = l2["threshold"]
        else:
            self.conv2_t_train = l2["training_threshold"]
            self.conv2_t_pass = l2["passing_threshold"]
        self.k2 = l2["k_winners"]
        self.r2 = l2["inhibition_radius"]

        self.decay = config["decay"]

        self.stdp1 = snn.STDP(self.conv1, (l1["ltp"], l1["ltd"]), timesteps=T)
        self.stdp2 = snn.STDP(self.conv2, (l2["ltp"], l2["ltd"]), timesteps=T)
        self.max_ap = Parameter(torch.Tensor([config.get("max_ap", 0.15)]))

        self.ctx = {
            "input_spikes": None,
            "potentials": None,
            "output_spikes": None,
            "winners": None,
        }
        self.spk_cnt1 = 0
        self.spk_cnt2 = 0

    def save_data(self, input_spike, potentials, output_spikes, winners):
        self.ctx["input_spikes"] = input_spike
        self.ctx["potentials"] = potentials
        self.ctx["output_spikes"] = output_spikes
        self.ctx["winners"] = winners

    def forward(self, input, max_layer):
        input = sf.pad(input.float(), (2, 2, 2, 2), 0)
        if self.training:
            cur = self.conv1(input)
            if max_layer == 1:
                spk, pot = sf.fire(cur, self.conv1_t_train, 0.95, True)
                winners = sf.get_k_winners(pot, spk, self.k1, self.r1)
                self.save_data(input, pot, spk, winners)
                return spk, pot
            if max_layer == 2:
                spk, pot = sf.fire(cur, self.conv1_t_pass, 0.95, True)
                spk_in = sf.pad(sf.pooling(spk, 2, 2, 1), (1,1,1,1))
                pot = self.conv2(spk_in)
                spk, pot = sf.fire(pot, self.conv2_t_train, 0.95, True)
                winners = sf.get_k_winners(pot, spk, self.k2, self.r2)
                self.save_data(spk_in, pot, spk, winners)
                return spk, pot
        else:
            pot = self.conv1(input)
            spk, pot = sf.fire(pot, self.conv1_t_pass, 0.95, True)
            pot = self.conv2(sf.pad(sf.pooling(spk, 2, 2, 1), (1, 1, 1, 1)))
            spk, pot = sf.fire(pot, self.conv2_t_pass, 0.95, True)
            spk = sf.pooling(spk, 2, 2, 1)
            return spk

    def stdp(self, layer_idx):
        if layer_idx == 1:
            self.stdp1(
                self.ctx["input_spikes"],
                self.ctx["potentials"],
                self.ctx["output_spikes"],
                self.ctx["winners"],
            )
        if layer_idx == 2:
            self.stdp2(
                self.ctx["input_spikes"],
                self.ctx["potentials"],
                self.ctx["output_spikes"],
                self.ctx["winners"],
            )


def train_unsupervise(network, data, layer_idx):
    network.train()
    total_spikes = 0
    total_neurons = 0
    winner_timesteps = []


    # Direct tqdm to stdout to not log as stderr into run.txt
    for data_in, _ in tqdm((data), file=stdout):
        spk, pot = network(data_in.to(device), layer_idx)
        winners = network.ctx["winners"]
        if len(winners) == 0:
            continue
        network.stdp(layer_idx)
        total_spikes += spk.sum().item()
        total_neurons += spk.numel()
        winner_timesteps = torch.as_tensor([w[0] for w in winners], dtype=torch.float).tolist()
    return {"spike_rate": total_spikes / max(total_neurons, 1), "spike_count": total_spikes, "total_neurons": total_neurons, "winner_timesteps": winner_timesteps}


def test(network, data, target, layer_idx):
    network.eval()
    ans = [None] * len(data)
    t = [None] * len(data)
    for i in range(len(data)):
        data_in = data[i]
        output, _ = network(data_in.to(device), layer_idx).max(dim=0)
        ans[i] = output.reshape(-1).cpu().numpy()
        t[i] = target[i]
    return np.array(ans), np.array(t)

def evaluate_linear_probe(model, train_dataset, test_dataset, layer, device):
    model.eval()
    X_train, X_test, y_train, y_test = [], [], [], []
    with torch.no_grad():
        for data_in, label in tqdm(train_dataset, file=stdout):
            spk = model(data_in.to(device), max_layer=layer)
            X_train.append(spk.sum(dim=0).cpu().numpy().flatten())
            y_train.append(label.item())

        for data_in, label in tqdm(test_dataset, file=stdout):
            spk = model(data_in.to(device), max_layer=layer)
            X_test.append(spk.sum(dim=0).cpu().numpy().flatten())
            y_test.append(label.item())
    clf = LinearSVC(max_iter=2000)
    clf.fit(X_train, y_train)
    report = classification_report(y_test, clf.predict(X_test))
    return report
