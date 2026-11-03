#!/usr/bin/env python3
"""Download the MNIST dataset into ./prep/data using torchvision."""

import os
from torchvision import datasets

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading MNIST to {DATA_DIR} ...")
    train_ds = datasets.MNIST(root=DATA_DIR, train=True, download=True)
    test_ds = datasets.MNIST(root=DATA_DIR, train=False, download=True)
    print(f"Done. Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")


if __name__ == "__main__":
    main()
