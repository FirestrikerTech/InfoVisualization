#!/usr/bin/env python3
"""
Export the trained MNIST MLP, rescaled weights, a balanced 500-image training subset,
and reference loss landscape grids + verification plots following Li et al. (2018):
"Visualizing the Loss Landscape of Neural Nets".

Binary formats exported to ../docs/assets/:
- weights.bin: Float32 little-endian array containing:
    [W1_scaled (32*784), b1_scaled (32), W2_scaled (10*32), b2_scaled (10)]
  Total floats: 25088 + 32 + 320 + 10 = 25450 floats (101,800 bytes)
- data.bin: Uint8 array containing:
    [images (N * 784 uint8 in 0..255), labels (N uint8 in 0..9)]
  For N = 500: 392000 + 500 = 392,500 bytes
- meta.json: Metadata, reference test points for JS verification, and configuration.
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
    """Exact Float32/Float64 forward pass matching the browser Web Worker."""
    # X: (N, 784), W1: (32, 784), b1: (32,), W2: (10, 32), b2: (10,)
    H = np.maximum(0.0, X @ W1.T + b1[None, :])  # (N, 32)
    logits = H @ W2.T + b2[None, :]  # (N, 10)
    # Numerically stable cross-entropy
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


def compute_grid(
    X, y, W1, b1, W2, b2, dW1, db1, dW2, db2, eW1, eb1, eW2, eb2, grid_size=41, span=1.0
):
    """Fast precomputed-projection grid evaluation (exact same math as JS worker)."""
    alphas = np.linspace(-span, span, grid_size)
    betas = np.linspace(-span, span, grid_size)

    # Precompute projections: X @ W1.T, X @ dW1.T, X @ eW1.T
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

    # Select a stratified 500-image subset of the MNIST training set (50 per class 0..9)
    # Pick samples with low loss so the center minimum is very deep (~0.07) and clean.
    train_raw = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transforms.ToTensor())
    images_all = np.stack([(train_raw[i][0].numpy().reshape(784) * 255.0).round().astype(np.uint8) for i in range(5000)])
    labels_all = np.array([train_raw[i][1] for i in range(5000)], dtype=np.int64)

    X_pool = images_all.astype(np.float32) / 255.0
    H_pool = np.maximum(0.0, X_pool @ W1_orig.T + b1_orig[None, :])
    logits_pool = H_pool @ W2_orig.T + b2_orig[None, :]
    preds_pool = np.argmax(logits_pool, axis=1)

    rng = np.random.default_rng(12345)
    selected_indices = []
    for digit in range(10):
        idx = np.where((labels_all == digit) & (preds_pool == digit))[0]
        chosen = rng.choice(idx, size=48, replace=False)
        idx_mis = np.where((labels_all == digit) & (preds_pool != digit))[0]
        chosen_mis = rng.choice(idx_mis, size=2, replace=False)
        selected_indices.append(np.concatenate([chosen, chosen_mis]))
    selected_indices = np.concatenate(selected_indices)
    rng.shuffle(selected_indices)

    subset_images_u8 = images_all[selected_indices].reshape(-1, 784).astype(np.uint8)
    subset_labels_u8 = labels_all[selected_indices].astype(np.uint8)

    X_sub = subset_images_u8.astype(np.float32) / 255.0
    y_sub = subset_labels_u8.astype(np.int64)

    # Rescaling factor c:
    # By positive homogeneity of ReLU:
    #   W1_scaled = c * W1_orig, b1_scaled = c * b1_orig
    #   W2_scaled = W2_orig / c, b2_scaled = b2_orig
    # For c = 12.0, ||W2_scaled|| becomes 12x smaller (~0.02 per row), so an unnormalized
    # Gaussian perturbation N(0, 1) (whose row norm is sqrt(32) ~ 5.66) is ~250x larger than
    # the row norm of W2_scaled! However, on [-1, 1] we also want a recognizable steep funnel
    # (not just vertical walls at 0.01). Notice in Li et al. (2018), unnormalized directions
    # are scaled by global unit norm or per-coordinate N(0, sigma^2) so alpha in [-1, 1] is meaningful.
    # Let's tune `c` and the unnormalized direction scale so that:
    # 1. With unnormalized Gaussian directions on [-1, 1], it forms a dramatic, sharp funnel
    #    reaching loss ~ 8-12 at the edges of [-1, 1].
    # 2. With filter normalization on [-1, 1], it forms a wide, smooth, flat-bottomed valley
    #    where loss stays low (< 0.5) across a wide central basin and rises gently to ~2.5-4 at edges.
    scale_c = 10.0
    W1_scaled = (W1_orig * scale_c).astype(np.float32)
    b1_scaled = (b1_orig * scale_c).astype(np.float32)
    W2_scaled = (W2_orig / scale_c).astype(np.float32)
    b2_scaled = b2_orig.copy().astype(np.float32)

    loss_orig, acc_orig = forward_loss_numpy(X_sub, y_sub, W1_orig, b1_orig, W2_orig, b2_orig)
    loss_scaled, acc_scaled = forward_loss_numpy(X_sub, y_sub, W1_scaled, b1_scaled, W2_scaled, b2_scaled)

    print(f"Original model on 500-subset: loss={loss_orig:.6f}, acc={acc_orig*100:.2f}%")
    print(f"Scaled   model on 500-subset: loss={loss_scaled:.6f}, acc={acc_scaled*100:.2f}%")
    assert abs(loss_orig - loss_scaled) < 1e-5, "Scale invariance check failed!"
    assert abs(acc_orig - acc_scaled) < 1e-6, "Accuracy invariance check failed!"

    # Check row norms of W1_scaled and W2_scaled:
    w1_row_norms = np.linalg.norm(W1_scaled, axis=1)
    w2_row_norms = np.linalg.norm(W2_scaled, axis=1)
    print(f"Mean filter norm W1_scaled: {w1_row_norms.mean():.4f} (random N(0,1) row norm = {np.sqrt(784):.2f})")
    print(f"Mean filter norm W2_scaled: {w2_row_norms.mean():.4f} (random N(0,1) row norm = {np.sqrt(32):.2f})")

    # Notice: for raw N(0, 1) entries, a row of dW2 has norm sqrt(32) = 5.66, while W2_scaled row
    # has norm ~ 0.15! Even if we normalize each direction vector by a single global scalar so
    # an unnormalized step of size 1 has reasonable magnitude (e.g., sigma = 0.08 per entry),
    # W2_scaled is still perturbed 10x harder relative to its filter norm than in the original network,
    # and 100x harder relative to W1_scaled!
    # Let's test and plot both raw/scaled Gaussian vs Filter-Normalized on [-1, 1] to verify visually.
    rng_dir = np.random.default_rng(2026)
    raw_dW1 = rng_dir.standard_normal(W1_scaled.shape).astype(np.float32)
    raw_dW2 = rng_dir.standard_normal(W2_scaled.shape).astype(np.float32)
    raw_eW1 = rng_dir.standard_normal(W1_scaled.shape).astype(np.float32)
    raw_eW2 = rng_dir.standard_normal(W2_scaled.shape).astype(np.float32)
    zeros_b1 = np.zeros_like(b1_scaled)
    zeros_b2 = np.zeros_like(b2_scaled)

    # For unnormalized random directions, we scale N(0, I) by a fixed global factor sigma_unnorm
    # so that on alpha, beta in [-1, 1] the sharp funnel spans from ~0.12 at (0,0) up to ~12-18 at edges,
    # making a textbook sharp cone/funnel in 3D without excessive flat clipping!
    sigma_unnorm = 0.085
    unnorm_dW1 = raw_dW1 * sigma_unnorm
    unnorm_dW2 = raw_dW2 * sigma_unnorm
    unnorm_eW1 = raw_eW1 * sigma_unnorm
    unnorm_eW2 = raw_eW2 * sigma_unnorm

    # Filter-normalized directions (Li et al., 2018 Eq. 1):
    fn_dW1, fn_dW2 = filter_normalize_direction(raw_dW1, raw_dW2, W1_scaled, W2_scaled)
    fn_eW1, fn_eW2 = filter_normalize_direction(raw_eW1, raw_eW2, W1_scaled, W2_scaled)

    # Also verify that filter-normalized directions on W_orig give the EXACT same loss landscape
    # as filter-normalized directions on W_scaled!
    fn_orig_dW1, fn_orig_dW2 = filter_normalize_direction(raw_dW1, raw_dW2, W1_orig, W2_orig)
    fn_orig_eW1, fn_orig_eW2 = filter_normalize_direction(raw_eW1, raw_eW2, W1_orig, W2_orig)

    _, _, losses_unnorm, accs_unnorm = compute_grid(
        X_sub, y_sub, W1_scaled, b1_scaled, W2_scaled, b2_scaled,
        unnorm_dW1, zeros_b1, unnorm_dW2, zeros_b2,
        unnorm_eW1, zeros_b1, unnorm_eW2, zeros_b2,
        grid_size=41, span=1.0
    )
    alphas, betas, losses_fn, accs_fn = compute_grid(
        X_sub, y_sub, W1_scaled, b1_scaled, W2_scaled, b2_scaled,
        fn_dW1, zeros_b1, fn_dW2, zeros_b2,
        fn_eW1, zeros_b1, fn_eW2, zeros_b2,
        grid_size=41, span=1.0
    )
    _, _, losses_fn_orig, _ = compute_grid(
        X_sub, y_sub, W1_orig, b1_orig, W2_orig, b2_orig,
        fn_orig_dW1, zeros_b1, fn_orig_dW2, zeros_b2,
        fn_orig_eW1, zeros_b1, fn_orig_eW2, zeros_b2,
        grid_size=41, span=1.0
    )

    max_diff_fn = np.max(np.abs(losses_fn - losses_fn_orig))
    print(f"Max diff between Filter-Norm(scaled) and Filter-Norm(orig): {max_diff_fn:.6e}")
    assert max_diff_fn < 1e-4, "Filter normalization scale invariance failed!"

    print(
        f"Unnormalized Funnel (Phase 2): min={losses_unnorm.min():.4f}, "
        f"center={losses_unnorm[20,20]:.4f}, max={losses_unnorm.max():.4f}, "
        f"median={np.median(losses_unnorm):.4f}"
    )
    print(
        f"Filter-Norm Valley  (Phase 3): min={losses_fn.min():.4f}, "
        f"center={losses_fn[20,20]:.4f}, max={losses_fn.max():.4f}, "
        f"median={np.median(losses_fn):.4f}"
    )

    # Save diagnostic comparison figure in prep/
    fig = plt.figure(figsize=(12, 5), dpi=140)
    AA, BB = np.meshgrid(alphas, betas)

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(AA, BB, np.minimum(losses_unnorm, 10.0), cmap="viridis", edgecolor="none")
    ax1.set_title("Phase 2: Gauß OHNE Filternormierung\n(skaliertes Modell → spitzer Trichter)")
    ax1.set_zlim(0, 10)
    ax1.set_xlabel("α")
    ax1.set_ylabel("β")
    ax1.set_zlabel("Loss")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(AA, BB, np.minimum(losses_fn, 10.0), cmap="viridis", edgecolor="none")
    ax2.set_title("Phase 3: Gauß MIT Filternormierung\n(Li et al. 2018 → flaches, breites Tal)")
    ax2.set_zlim(0, 10)
    ax2.set_xlabel("α")
    ax2.set_ylabel("β")
    ax2.set_zlabel("Loss")

    plt.tight_layout()
    plot_path = os.path.join(PREP_DIR, "sanity_check_phases.png")
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Saved sanity check plot to {plot_path}")

    # Export weights.bin
    weights_flat = np.concatenate([
        W1_scaled.reshape(-1),
        b1_scaled.reshape(-1),
        W2_scaled.reshape(-1),
        b2_scaled.reshape(-1),
    ]).astype("<f4")
    weights_path = os.path.join(ASSETS_DIR, "weights.bin")
    weights_flat.tofile(weights_path)
    print(f"Exported {weights_path} ({os.path.getsize(weights_path)} bytes, {len(weights_flat)} floats)")

    # Export data.bin
    data_bytes = np.concatenate([
        subset_images_u8.reshape(-1),
        subset_labels_u8.reshape(-1),
    ]).astype(np.uint8)
    data_path = os.path.join(ASSETS_DIR, "data.bin")
    data_bytes.tofile(data_path)
    print(f"Exported {data_path} ({os.path.getsize(data_path)} bytes)")

    # Export meta.json with model stats, layout offsets, and reference test values
    meta = {
        "architecture": [784, 32, 10],
        "num_params": int(len(weights_flat)),
        "num_samples": int(len(y_sub)),
        "scale_factor_c": float(scale_c),
        "sigma_unnorm": float(sigma_unnorm),
        "unlock_random_seconds": 150,
        "offsets": {
            "W1": [0, 32 * 784],
            "b1": [32 * 784, 32 * 784 + 32],
            "W2": [32 * 784 + 32, 32 * 784 + 32 + 10 * 32],
            "b2": [32 * 784 + 32 + 10 * 32, 32 * 784 + 32 + 10 * 32 + 10],
        },
        "metrics": {
            "subset_train_loss": round(loss_scaled, 6),
            "subset_train_acc": round(acc_scaled, 4),
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

    # Generate QR Code for GitHub Pages URL
    pages_url = "https://firestrikertech.github.io/InfoVisualization/"
    qr = qrcode.QRCode(
        version=2,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=12,
        border=3,
    )
    qr.add_data(pages_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#0f172a", back_color="#ffffff")
    qr_path = os.path.join(DOCS_DIR, "qr.png")
    img.save(qr_path)
    print(f"Generated QR code for {pages_url} at {qr_path}")


if __name__ == "__main__":
    main()
