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
    "layer1": {
        "in_channels": 2,
        "out_channels": 16,
        "kernel_size": 5,
        "w_mean": 0.8,
        "w_std": 0.1,
        "inhibition_radius": 5,
        "k_winners": 5,
        "ltp": 0.004,
        "ltd": -0.003,
        "threshold": 15,
        "epochs": 5,
    },
    "layer2": {
        "out_channels": 32,
        "kernel_size": 3,
        "w_mean": 0.8,
        "w_std": 0.05,
        "inhibition_radius": 3,
        "k_winners": 3,
        "ltp": 0.002,
        "ltd": -0.002,
        "threshold": 10,
        "epochs": 5,
    },
}

def log(s, console=False):
    with open(log_dir / "run.txt", "a") as f:
        print(s, file=f)
    if console:
        print(s)

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

train_dataset = NCaltechDataset(root_dir=root, T=T, H=173, W=233)
small_dataset = torch.utils.data.Subset(train_dataset, range(10))

model = LiuNCaltech101(config)
model.to(device)

log("Layer 1",True)
if checkpoint_l1.exists():
    checkpoint = torch.load(checkpoint_l1, weights_only=True, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
else:
    history = {"diversity": [], "spike_rate": [], "w_mean": [], "w_std": []}
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
        history["w_mean"].append(w.mean().item())
        history["w_std"].append(w.std().item())

        log(f"epoch:{epoch} "
            f"diversity:{diversity.item():.4f} "
            f"w_mean:{w.mean().item():.4f} "
            f"w_std:{w.std().item():.4f} "
            f"time:{elapsed:.1f}s "
            f"spike_rate:{stats['spike_rate']:.4f} ")

        vis.plot_weights(model.conv1.weight, (log_dir / f"l1_epoch_{epoch}_kernels.png"))

    with open(log_dir / "history_l1.json", "w") as f:
        json.dump(history, f, indent=4)
    vis.save_curves(history, 1, log_dir)

    torch.save({
        'state_dict': model.state_dict(),
        'config': config,
        },checkpoint_l1)
    log(evaluate_linear_probe(model, train_dataset, 1, device))

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

    with open(log_dir / "history_l2.json", "w") as f:
        json.dump(history, f, indent=4)
    vis.save_curves(history, 2, log_dir)

    torch.save({
        'state_dict': model.state_dict(),
        'config': config,
        },checkpoint_l2)
    log(evaluate_linear_probe(model, train_dataset, 2, device))
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
