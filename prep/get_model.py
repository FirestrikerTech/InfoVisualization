#!/usr/bin/env python3
"""
Train (or load) a clean, compact ReLU MLP (784 -> 32 -> 10) without BatchNorm on MNIST.
Why no BatchNorm: Li et al. (2018) Sec. 5 explains positive homogeneity and scale invariance
of ReLU networks (W1, b1 <- c*W1, c*b1 and W2 <- W2/c). Without BatchNorm, we can apply this
exact invariant rescaling to make unnormalized random directions look like an extremely sharp
funnel while keeping the network output, loss, and accuracy 100% identical.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PREP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PREP_DIR, "data")
MODEL_PATH = os.path.join(PREP_DIR, "mnist_mlp.pt")


class MnistMLP(nn.Module):
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_dim, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 10, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total


def train_model(epochs: int = 12, lr: float = 1e-3, weight_decay: float = 1e-4, seed: int = 42):
    torch.manual_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Simple 0..1 scaling (uint8 / 255.0) so browser JS only needs to divide uint8 by 255.0
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    model = MnistMLP(hidden_dim=32).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        train_loss, train_acc = evaluate(model, train_loader, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
            f"Test Loss: {test_loss:.4f}, Acc: {test_acc*100:.2f}%"
        )

    model.cpu()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_dim": 32,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        },
        MODEL_PATH,
    )
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
