import os
import sys
import traceback

def log_exception(exc_type, exc_value, exc_tb):
    err = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    log(f"\n=====EXCEPTION=====\n{err}", console=True)
    sys.__excepthook__(exc_type, exc_value, exc_tb)

sys.excepthook = log_exception
# Log source code ASAP. Adapted from KellerJordan.
with open(sys.argv[0]) as f:
    code = f.read()
with open(os.path.join(os.path.dirname(sys.argv[0]), 'LiuDeep.py'), 'r') as f:
    code += f"\n\n{'-'*40}\n# LiuDeep.py\n{'-'*40}\n\n"
    code += f.read()

from pathlib import Path

import torch
from datasets.ncaltech101_dataset import NCaltechDataset
from sklearn.utils.discovery import all_displays
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from LiuDeep import LiuNCaltech101, train_unsupervise, evaluate_linear_probe
from SpykeTorch import visualization as vis
from datetime import datetime

import time
import json
import subprocess

RESUME_RUN = None

config = {
    "device": "mps",
    "model": "LiuNCaltech101",
    "timesteps": 60,
    "decay": 0.95,
    "L1_perfect_weights": True,
    "layer1": {
        "in_channels": 2,
        "out_channels": 4,
        "kernel_size": 5,
        "w_mean": 0.8,
        "w_std": 0.1,
        "inhibition_radius": 10,
        "k_winners": 2,
        "ltp": 0.004,
        "ltd": -0.002,
        "training_threshold": 30,
        "passing_threshold": 15,
        "epochs": 2,
    },
    "layer2": {
        "out_channels": 32,
        "kernel_size": 11,
        "w_mean": 0.8,
        "w_std": 0.1,
        "inhibition_radius": 5,
        "k_winners": 16,
        "ltp": 0.008,
        "ltd": -0.002,
        "training_threshold": 100,
        "passing_threshold": 50,
        "epochs": 2,
    },
}

def log(s, console=False):
    with open(log_dir / "run.txt", "a") as f:
        print(s, file=f)
    if console:
        print(s)

# Claude-generated function
def make_perfect_l1_weights(in_channels, kernel_size):
    """Preset L1 filters: horizontal bar, vertical edge, two diagonals."""
    k = kernel_size
    lo, hi = 0.25, 0.75
    idx = torch.arange(k)

    h  = torch.full((k, k), lo); h[k // 2, :] = hi           # horizontal bar
    v  = torch.full((k, k), lo); v[:, :k // 2] = hi          # vertical left edge
    d1 = torch.full((k, k), lo); d1[idx, idx] = hi            # diagonal \
    d2 = torch.full((k, k), lo); d2[idx, k - 1 - idx] = hi   # diagonal /

    w = torch.stack([h, v, d1, d2]).unsqueeze(1)              # (4, 1, k, k)
    return w.repeat(1, in_channels, 1, 1)                     # (4, in_channels, k, k)

######################################################################

device = torch.device(config["device"])
T = config["timesteps"]
l1 = config["layer1"]
l2 = config["layer2"]

root = Path(__file__).parent



if RESUME_RUN:
    log_dir = root / "logs" / RESUME_RUN
    log("=====RESUMED RUN=====")
else:
    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    diff = subprocess.check_output(["git", "diff"]).decode()
    timestamp = datetime.now().strftime("%Y%m%dT%H-%M-%S-%f")
    log_dir = root / "logs" / f"run_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log(f"git hash: {git_hash}")
    log("=====CONFIG=====")
    log(json.dumps(config, indent=4))
    log("=====GIT DIFF=====")
    log(diff)
    log("=====CODE=====")
    log(code)

class TeeStderr:
    def __init__(self):
        self._stderr = sys.stderr
    def write(self, s):
        self._stderr.write(s)
        if s.strip():
            with open(log_dir / "run.txt", "a") as f:
                f.write(s)
    def flush(self):
        self._stderr.flush()

sys.stderr = TeeStderr()

log("="*100)
log(f"Running Python {sys.version}")
log(f"Running PyTorch {torch.__version__} with mps")

checkpoint_l1 = log_dir / "l1.pt"
checkpoint_l2 = log_dir / "l2.pt"


dataset = NCaltechDataset(root_dir=root, T=T, H=173, W=233)
all_labels = [dataset.samples[i][1] for i in range(len(dataset))]
all_indices = list(range(len(dataset)))
train_idx, test_idx = train_test_split(all_indices, test_size=0.2, random_state=42, stratify=all_labels)

train_dataset = Subset(dataset, train_idx)
test_dataset = Subset(dataset, test_idx)

model = LiuNCaltech101(config)
model.to(device)

log("Random weights baseline", True)
log(evaluate_linear_probe(model, train_dataset, test_dataset, 2, device))

log("Layer 1",True)
if checkpoint_l1.exists():
    checkpoint = torch.load(checkpoint_l1, weights_only=True, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
else:
    history = {"diversity": [], "spike_rate": [], "w_mean": [], "w_std": [], "spike_count": [], "total_neurons": [], "winner_timesteps": []}
    if config["L1_perfect_weights"]:
        w = make_perfect_l1_weights(l1["in_channels"], l1["kernel_size"]).to(device)
        model.conv1.weight.data.copy_(w)
        vis.plot_weights(model.conv1.weight, (log_dir / "l1_fixed_kernels.png"))
        vis.plot_weights_detailed(model.conv1.weight, log_dir, "l1_fixed_kernels_detail")
        torch.save({'state_dict': model.state_dict(), 'config': config}, checkpoint_l1)
    else:
        for epoch in range(config["layer1"]["epochs"]):
            t0 = time.time()
            print("epoch: ", epoch)
            stats = train_unsupervise(model, train_dataset, 1)
            elapsed = time.time() - t0

            w = model.conv1.weight
            w_flat = w.view(w.shape[0], -1).float()
            w_norm = w_flat / (w_flat.norm(dim=1, keepdim=True) + 1e-8)
            sim = (w_norm @ w_norm.T).abs()
            n = w.shape[0]
            diversity = 1.0 - (sim.sum() - n) / (n * (n-1))

            history["diversity"].append(diversity.item())
            history["spike_rate"].append(stats["spike_rate"])
            history["spike_count"].append(stats["spike_count"])
            history["total_neurons"].append(stats["total_neurons"])
            history["winner_timesteps"].append(stats["winner_timesteps"])
            history["w_mean"].append(w.mean().item())
            history["w_std"].append(w.std().item())

            log(f"epoch:{epoch} "
                f"diversity:{diversity.item():.4f} "
                f"w_mean:{w.mean().item():.4f} "
                f"w_std:{w.std().item():.4f} "
                f"time:{elapsed:.1f}s "
                f"spike count:{stats['spike_count']} "
                f"total neurons:{stats['total_neurons']} "
                f"Winner timesteps: {stats['winner_timesteps']} "
                f"spike_rate:{stats['spike_rate']:.4f} ")

            vis.plot_weights(model.conv1.weight, (log_dir / f"l1_epoch_{epoch}_kernels.png"))
            vis.plot_weights_detailed(model.conv1.weight, log_dir, f"l1_epoch_{epoch}_kernels_detail")

    with open(log_dir / "history_l1.json", "w") as f:
        json.dump(history, f, indent=4)
    vis.save_curves(history, 1, log_dir)

    torch.save({
        'state_dict': model.state_dict(),
        'config': config,
        },checkpoint_l1)

log("Layer 2",True)
if checkpoint_l2.exists():
    checkpoint = torch.load(checkpoint_l2, weights_only=True, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
else:
    history = {"diversity": [], "spike_rate": [], "w_mean": [], "w_std": []}
    for epoch in range(config["layer2"]["epochs"]):
        t0 = time.time()
        print("epoch: ", epoch)
        stats = train_unsupervise(model, train_dataset, 2)
        elapsed = time.time() - t0

        w = model.conv2.weight
        w_flat = w.view(w.shape[0], -1).float()
        w_norm = w_flat / (w_flat.norm(dim=1, keepdim=True) + 1e-8)
        sim = (w_norm @ w_norm.T).abs()
        n = w.shape[0]
        diversity = 1.0 - (sim.sum() - n) / (n * (n-1))

        history["diversity"].append(diversity.item())
        history["spike_rate"].append(stats["spike_rate"])
        history["w_mean"].append(w.mean().item())
        history["w_std"].append(w.std().item())

        log(f"epoch:{epoch} "
            f"diversity:{diversity.item():.4f} "
            f"w_mean:{w.mean().item():.4f} "
            f"w_std:{w.std().item():.4f} "
            f"time:{elapsed:.1f}s "
            f"spike_rate:{stats['spike_rate']:.4f} ")

        vis.plot_weights(model.conv2.weight, (log_dir / f"l2_epoch_{epoch}_kernels.png"))
        vis.plot_weights_detailed(model.conv2.weight, log_dir, f"l2_epoch_{epoch}_kernels_detail")

    with open(log_dir / "history_l2.json", "w") as f:
        json.dump(history, f, indent=4)
    vis.save_curves(history, 2, log_dir)

    torch.save({
        'state_dict': model.state_dict(),
        'config': config,
        },checkpoint_l2)
    log(evaluate_linear_probe(model, train_dataset, test_dataset, 2, device))

def get_performance(x, y, predictions):
    correct = 0
    silence = 0
    for i in range(len(predictions)):
        if x[i].sum() == 0:
            silence += 1
        else:
            if predictions[i] == y[i]:
                correct += 1
    return (correct / len(x), (len(x) - (correct + silence)) / len(x), silence / len(x))
