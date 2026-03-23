"""
Raw spike baseline classifier.

Bypasses the SNN entirely. Extracts features by summing raw spike tensors
over the time dimension and classifying with LinearSVC.

Use this to establish a floor: if the SNN doesn't beat this, it's not helping.

Usage:
    python3 scripts/raw_baseline.py
"""

from pathlib import Path
from sys import stdout

import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from torch.utils.data import Subset
from tqdm import tqdm

from datasets.ncaltech101_dataset import NCaltechDataset

root = Path(__file__).resolve().parents[1]
T = 60
dataset = NCaltechDataset(root_dir=root, T=T, H=173, W=233)

all_labels = [dataset.samples[i][1] for i in range(len(dataset))]
all_indices = list(range(len(dataset)))
train_idx, test_idx = train_test_split(
    all_indices, test_size=0.2, random_state=42, stratify=all_labels
)
train_set = Subset(dataset, train_idx)
test_set  = Subset(dataset, test_idx)

print(f"Train: {len(train_set)}  Test: {len(test_set)}")

X_train, y_train = [], []
for data_in, label in tqdm(train_set, desc="train features", file=stdout):
    X_train.append(data_in.sum(dim=0).float().numpy().flatten())
    y_train.append(label.item())

X_test, y_test = [], []
for data_in, label in tqdm(test_set, desc="test features", file=stdout):
    X_test.append(data_in.sum(dim=0).float().numpy().flatten())
    y_test.append(label.item())

print("Fitting LinearSVC...")
clf = LinearSVC(max_iter=10000)
clf.fit(X_train, y_train)

print(classification_report(y_test, clf.predict(X_test)))
