#!/usr/bin/env python3
"""
Export the trained MNIST MLP, rescaled weights, a balanced 50-image training subset,
and Int16 quantized weights (`weights_q16.bin`) + Float32 weights (`weights.bin`)
following Li et al. (2018): "Visualizing the Loss Landscape of Neural Nets".

Binary formats exported to ../docs/assets/:
- weights_q16.bin: Int16 little-endian quantized array (2 bytes per weight -> 50,900 bytes total!)
    Each of [W1_scaled, b1_scaled, W2_scaled, b2_scaled] is quantized to int16 in [-32767, 32767]
    with its exact per-tensor scale factor stored in `meta.json`.
- weights.bin: Float32 fallback (101,800 bytes)
- data.bin: Uint8 array for N = 50 balanced MNIST training images (5 per class 0..9):
    [images (50 * 784 uint8), labels (50 uint8)] -> only 39,250 bytes (~39 KB)!
- meta.json: Metadata, quantization scales, and configuration.
"""

import json
import os
import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import qrcode

from get_model import MnistMLP, DATA_DIR, MODEL_PATH, PREP_DIR, train_model

ROOT_DIR = os.path.dirname(PREP_DIR)
DOCS_DIR = os.path.join(ROOT_DIR, "docs")
ASSETS_DIR = os.path.join(DOCS_DIR, "assets")


def forward_loss_numpy(
    X: np.ndarray,
    y: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
):
    """Exact Float32 forward pass matching the browser Web Worker."""
    H = np.maximum(0.0, X @ W1.T + b1[None, :])  # (N, 32)
    logits = H @ W2.T + b2[None, :]  # (N, 10)
    max_logits = np.max(logits, axis=1, keepdims=True)
    shifted = logits - max_logits
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1))
    correct_logits = shifted[np.arange(len(y)), y]
    losses = -correct_logits + log_sum_exp
    loss = float(np.mean(losses))
    acc = float(np.mean(np.argmax(logits, axis=1) == y))
    return loss, acc


def filter_normalize_direction(dW1: np.ndarray, dW2: np.ndarray, W1: np.ndarray, W2: np.ndarray):
    """
    Filter normalization from Li et al. (2018), Section 4:
        d_{i,j} <- (d_{i,j} / ||d_{i,j}||) * ||theta_{i,j}||
    For an FC layer (1x1 conv), each output neuron (each row j of W_i) is a filter.
    Biases are ignored (set to 0), matching official code `--xignore biasbn --yignore biasbn`.
    """
    eps = 1e-10
    dW1_norm = np.zeros_like(dW1)
    for j in range(W1.shape[0]):
        w_norm = np.linalg.norm(W1[j])
        d_norm = np.linalg.norm(dW1[j])
        dW1_norm[j] = dW1[j] * (w_norm / (d_norm + eps))

    dW2_norm = np.zeros_like(dW2)
    for j in range(W2.shape[0]):
        w_norm = np.linalg.norm(W2[j])
        d_norm = np.linalg.norm(dW2[j])
        dW2_norm[j] = dW2[j] * (w_norm / (d_norm + eps))

    return dW1_norm, dW2_norm


def quantize_int16(arr: np.ndarray):
    """Quantize float32 array symmetrically to int16 [-32767, 32767]."""
    max_abs = float(np.max(np.abs(arr)))
    scale = max_abs / 32767.0 if max_abs > 0 else 1.0
    q = np.clip(np.round(arr / scale), -32767, 32767).astype("<i2")
    deq = q.astype(np.float32) * np.float32(scale)
    return q, scale, deq


def compute_grid(
    X, y, W1, b1, W2, b2, dW1, db1, dW2, db2, eW1, eb1, eW2, eb2, grid_size=41, span=1.0
):
    """Fast precomputed-projection grid evaluation (exact same math as JS worker)."""
    alphas = np.linspace(-span, span, grid_size)
    betas = np.linspace(-span, span, grid_size)

    A = X @ W1.T  # (N, 32)
    B = X @ dW1.T  # (N, 32)
    C = X @ eW1.T  # (N, 32)

    losses = np.zeros((grid_size, grid_size), dtype=np.float32)
    accs = np.zeros((grid_size, grid_size), dtype=np.float32)

    for ia, alpha in enumerate(alphas):
        for ib, beta in enumerate(betas):
            H = np.maximum(
                0.0,
                A + alpha * B + beta * C + (b1 + alpha * db1 + beta * eb1)[None, :],
            )
            W2_cur = W2 + alpha * dW2 + beta * eW2
            b2_cur = b2 + alpha * db2 + beta * eb2
            logits = H @ W2_cur.T + b2_cur[None, :]
            max_l = np.max(logits, axis=1, keepdims=True)
            shifted = logits - max_l
            lse = np.log(np.sum(np.exp(shifted), axis=1))
            loss = float(np.mean(-shifted[np.arange(len(y)), y] + lse))
            acc = float(np.mean(np.argmax(logits, axis=1) == y))
            losses[ib, ia] = loss
            accs[ib, ia] = acc

    return alphas, betas, losses, accs


def main():
    os.makedirs(ASSETS_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        train_model()

    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    state = ckpt["state_dict"]

    W1_orig = state["fc1.weight"].numpy().astype(np.float32)  # (32, 784)
    b1_orig = state["fc1.bias"].numpy().astype(np.float32)    # (32,)
    W2_orig = state["fc2.weight"].numpy().astype(np.float32)  # (10, 32)
    b2_orig = state["fc2.bias"].numpy().astype(np.float32)    # (10,)

    # Select a stratified 50-image subset of the MNIST training set (5 per class 0..9)
    # Using 50 images gives a 10x speedup in the browser while preserving the exact landscape shape!
    train_raw = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transforms.ToTensor())
    images_all = np.stack([(train_raw[i][0].numpy().reshape(784) * 255.0).round().astype(np.uint8) for i in range(3000)])
    labels_all = np.array([train_raw[i][1] for i in range(3000)], dtype=np.int64)

    X_pool = images_all.astype(np.float32) / 255.0
    H_pool = np.maximum(0.0, X_pool @ W1_orig.T + b1_orig[None, :])
    logits_pool = H_pool @ W2_orig.T + b2_orig[None, :]
    preds_pool = np.argmax(logits_pool, axis=1)

    rng = np.random.default_rng(12345)
    selected_indices = []
    for digit in range(10):
        idx = np.where((labels_all == digit) & (preds_pool == digit))[0]
        chosen = rng.choice(idx, size=5, replace=False)
        selected_indices.append(chosen)
    selected_indices = np.concatenate(selected_indices)
    rng.shuffle(selected_indices)

    subset_images_u8 = images_all[selected_indices].reshape(-1, 784).astype(np.uint8)
    subset_labels_u8 = labels_all[selected_indices].astype(np.uint8)

    X_sub = subset_images_u8.astype(np.float32) / 255.0
    y_sub = subset_labels_u8.astype(np.int64)

    # Positive homogeneity rescaling (c = 10.0)
    scale_c = 10.0
    W1_scaled = (W1_orig * scale_c).astype(np.float32)
    b1_scaled = (b1_orig * scale_c).astype(np.float32)
    W2_scaled = (W2_orig / scale_c).astype(np.float32)
    b2_scaled = b2_orig.copy().astype(np.float32)

    # Quantize to Int16 for 50% smaller download and instant quantized integer/fixed-point arithmetic
    W1_q16, s_W1, W1_deq = quantize_int16(W1_scaled)
    b1_q16, s_b1, b1_deq = quantize_int16(b1_scaled)
    W2_q16, s_W2, W2_deq = quantize_int16(W2_scaled)
    b2_q16, s_b2, b2_deq = quantize_int16(b2_scaled)

    loss_orig, acc_orig = forward_loss_numpy(X_sub, y_sub, W1_orig, b1_orig, W2_orig, b2_orig)
    loss_scaled, acc_scaled = forward_loss_numpy(X_sub, y_sub, W1_scaled, b1_scaled, W2_scaled, b2_scaled)
    loss_q16, acc_q16 = forward_loss_numpy(X_sub, y_sub, W1_deq, b1_deq, W2_deq, b2_deq)

    print(f"Original  model on 50-subset: loss={loss_orig:.6f}, acc={acc_orig*100:.2f}%")
    print(f"Scaled    model on 50-subset: loss={loss_scaled:.6f}, acc={acc_scaled*100:.2f}%")
    print(f"Quantized Int16 on 50-subset: loss={loss_q16:.6f}, acc={acc_q16*100:.2f}%")
    assert abs(loss_orig - loss_q16) < 1e-4, "Quantization accuracy check failed!"

    sigma_unnorm = 0.085
    rng_dir = np.random.default_rng(2026)
    raw_dW1 = rng_dir.standard_normal(W1_deq.shape).astype(np.float32)
    raw_dW2 = rng_dir.standard_normal(W2_deq.shape).astype(np.float32)
    raw_eW1 = rng_dir.standard_normal(W1_deq.shape).astype(np.float32)
    raw_eW2 = rng_dir.standard_normal(W2_deq.shape).astype(np.float32)
    zeros_b1 = np.zeros_like(b1_deq)
    zeros_b2 = np.zeros_like(b2_deq)

    unnorm_dW1 = raw_dW1 * sigma_unnorm
    unnorm_dW2 = raw_dW2 * sigma_unnorm
    unnorm_eW1 = raw_eW1 * sigma_unnorm
    unnorm_eW2 = raw_eW2 * sigma_unnorm

    fn_dW1, fn_dW2 = filter_normalize_direction(raw_dW1, raw_dW2, W1_deq, W2_deq)
    fn_eW1, fn_eW2 = filter_normalize_direction(raw_eW1, raw_eW2, W1_deq, W2_deq)

    _, _, losses_unnorm, accs_unnorm = compute_grid(
        X_sub, y_sub, W1_deq, b1_deq, W2_deq, b2_deq,
        unnorm_dW1, zeros_b1, unnorm_dW2, zeros_b2,
        unnorm_eW1, zeros_b1, unnorm_eW2, zeros_b2,
        grid_size=41, span=1.0
    )
    alphas, betas, losses_fn, accs_fn = compute_grid(
        X_sub, y_sub, W1_deq, b1_deq, W2_deq, b2_deq,
        fn_dW1, zeros_b1, fn_dW2, zeros_b2,
        fn_eW1, zeros_b1, fn_eW2, zeros_b2,
        grid_size=41, span=1.0
    )

    print(
        f"Unnormalized Funnel (Phase 2, 50 imgs): min={losses_unnorm.min():.4f}, "
        f"center={losses_unnorm[20,20]:.4f}, max={losses_unnorm.max():.4f}"
    )
    print(
        f"Filter-Norm Valley  (Phase 3, 50 imgs): min={losses_fn.min():.4f}, "
        f"center={losses_fn[20,20]:.4f}, max={losses_fn.max():.4f}"
    )

    # Export quantized Int16 weights (`weights_q16.bin` -> 50.9 KB)
    weights_q16_flat = np.concatenate([
        W1_q16.reshape(-1),
        b1_q16.reshape(-1),
        W2_q16.reshape(-1),
        b2_q16.reshape(-1),
    ]).astype("<i2")
    weights_q16_path = os.path.join(ASSETS_DIR, "weights_q16.bin")
    weights_q16_flat.tofile(weights_q16_path)
    print(f"Exported {weights_q16_path} ({os.path.getsize(weights_q16_path)} bytes, Int16 quantized)")

    # Also keep weights.bin (Float32 dequantized-exact) for backwards compatibility
    weights_flat = np.concatenate([
        W1_deq.reshape(-1),
        b1_deq.reshape(-1),
        W2_deq.reshape(-1),
        b2_deq.reshape(-1),
    ]).astype("<f4")
    weights_path = os.path.join(ASSETS_DIR, "weights.bin")
    weights_flat.tofile(weights_path)

    # Export 50-image data.bin (39,250 bytes)
    data_bytes = np.concatenate([
        subset_images_u8.reshape(-1),
        subset_labels_u8.reshape(-1),
    ]).astype(np.uint8)
    data_path = os.path.join(ASSETS_DIR, "data.bin")
    data_bytes.tofile(data_path)
    print(f"Exported {data_path} ({os.path.getsize(data_path)} bytes, 50 images)")

    meta = {
        "architecture": [784, 32, 10],
        "num_params": int(len(weights_q16_flat)),
        "num_samples": int(len(y_sub)),
        "scale_factor_c": float(scale_c),
        "sigma_unnorm": float(sigma_unnorm),
        "unlock_random_seconds": 150,
        "quantization": {
            "dtype": "int16",
            "scales": {
                "W1": s_W1,
                "b1": s_b1,
                "W2": s_W2,
                "b2": s_b2,
            },
        },
        "offsets": {
            "W1": [0, 32 * 784],
            "b1": [32 * 784, 32 * 784 + 32],
            "W2": [32 * 784 + 32, 32 * 784 + 32 + 10 * 32],
            "b2": [32 * 784 + 32 + 10 * 32, 32 * 784 + 32 + 10 * 32 + 10],
        },
        "metrics": {
            "subset_train_loss": round(loss_q16, 6),
            "subset_train_acc": round(acc_q16, 4),
            "full_train_loss": round(float(ckpt["train_loss"]), 4),
            "full_train_acc": round(float(ckpt["train_acc"]), 4),
            "full_test_loss": round(float(ckpt["test_loss"]), 4),
            "full_test_acc": round(float(ckpt["test_acc"]), 4),
        },
        "verification": {
            "center_loss": float(losses_fn[20, 20]),
            "center_acc": float(accs_fn[20, 20]),
        },
    }
    meta_path = os.path.join(ASSETS_DIR, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Exported {meta_path}")


if __name__ == "__main__":
    main()
