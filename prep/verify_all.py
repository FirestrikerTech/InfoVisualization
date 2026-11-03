#!/usr/bin/env python3
"""
Verify the exported binary files (`weights_q16.bin`, `data.bin`, `meta.json`) and generate
a 3-panel overview plot of all 3 phases:
- Phase 1 initial startup (Dim #1 & Dim #2): inclined slope/ridge where minimum is NOT visible!
- Phase 2 Unnormalized Gaussian Funnel on 50 images
- Phase 3 Filter-Normalized Valley on 50 images
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from export import (
    ASSETS_DIR,
    PREP_DIR,
    compute_grid,
    filter_normalize_direction,
    forward_loss_numpy,
)


def build_dim_direction_py(dim_index: int, strength: float):
    dW1 = np.zeros((32, 784), dtype=np.float32)
    db1 = np.zeros((32,), dtype=np.float32)
    dW2 = np.zeros((10, 32), dtype=np.float32)
    db2 = np.zeros((10,), dtype=np.float32)
    s = strength
    if dim_index <= 15:
        neuron_a = (dim_index * 3) % 32
        for k in range(784):
            r = k // 28
            c = k % 28
            if r < 4 or r > 24 or c < 4 or c > 24:
                dW1[neuron_a, k] = s * 14.0
        row = (dim_index + 2) % 28
        dW1[neuron_a, row * 28 + 2] = s * 25.0
        cls = dim_index % 10
        dW2[cls, neuron_a] = s * 0.042
        db1[neuron_a] = s * 7.5
    return dW1, db1, dW2, db2


def main():
    meta_path = os.path.join(ASSETS_DIR, "meta.json")
    weights_q16_path = os.path.join(ASSETS_DIR, "weights_q16.bin")
    data_path = os.path.join(ASSETS_DIR, "data.bin")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    q16 = np.fromfile(weights_q16_path, dtype="<i2")
    data_u8 = np.fromfile(data_path, dtype=np.uint8)

    N = meta["num_samples"]
    assert N == 50, f"Expected 50 images, got {N}"
    o = meta["offsets"]
    sc = meta["quantization"]["scales"]

    W1 = (q16[o["W1"][0] : o["W1"][1]].astype(np.float32) * sc["W1"]).reshape(32, 784)
    b1 = (q16[o["b1"][0] : o["b1"][1]].astype(np.float32) * sc["b1"]).reshape(32)
    W2 = (q16[o["W2"][0] : o["W2"][1]].astype(np.float32) * sc["W2"]).reshape(10, 32)
    b2 = (q16[o["b2"][0] : o["b2"][1]].astype(np.float32) * sc["b2"]).reshape(10)

    X = data_u8[: N * 784].reshape(N, 784).astype(np.float32) * np.float32(1.0 / 255.0)
    y = data_u8[N * 784 : N * 784 + N].astype(np.int64)

    loss_c, acc_c = forward_loss_numpy(X, y, W1, b1, W2, b2)
    diff_loss = abs(loss_c - meta["verification"]["center_loss"])
    print(f"Verified Int16 quantized center (0,0) on 50 images: loss={loss_c:.6f} (diff={diff_loss:.2e}), acc={acc_c*100:.2f}%")
    assert diff_loss < 1e-5

    # Phase 1 initial startup (Dim #1 & Dim #2, strength 1.0)
    dW1_p1, db1_p1, dW2_p1, db2_p1 = build_dim_direction_py(1, 1.0)
    eW1_p1, eb1_p1, eW2_p1, eb2_p1 = build_dim_direction_py(2, 1.0)

    alphas, betas, losses_p1, _ = compute_grid(
        X, y, W1, b1, W2, b2,
        dW1_p1, db1_p1, dW2_p1, db2_p1,
        eW1_p1, eb1_p1, eW2_p1, eb2_p1,
        grid_size=41, span=1.0,
    )
    print(
        f"Phase 1 Startup (Dim 1 & 2): min={losses_p1.min():.4f}, "
        f"center={losses_p1[20,20]:.4f}, max={losses_p1.max():.4f}"
    )

    # Phase 2 & 3 directions
    rng = np.random.default_rng(2026)
    raw_dW1 = rng.standard_normal(W1.shape).astype(np.float32)
    raw_dW2 = rng.standard_normal(W2.shape).astype(np.float32)
    raw_eW1 = rng.standard_normal(W1.shape).astype(np.float32)
    raw_eW2 = rng.standard_normal(W2.shape).astype(np.float32)
    zeros_b1 = np.zeros_like(b1)
    zeros_b2 = np.zeros_like(b2)

    sigma = meta["sigma_unnorm"]
    _, _, losses_p2, _ = compute_grid(
        X, y, W1, b1, W2, b2,
        raw_dW1 * sigma, zeros_b1, raw_dW2 * sigma, zeros_b2,
        raw_eW1 * sigma, zeros_b1, raw_eW2 * sigma, zeros_b2,
        grid_size=41, span=1.0,
    )

    fn_dW1, fn_dW2 = filter_normalize_direction(raw_dW1, raw_dW2, W1, W2)
    fn_eW1, fn_eW2 = filter_normalize_direction(raw_eW1, raw_eW2, W1, W2)
    _, _, losses_p3, _ = compute_grid(
        X, y, W1, b1, W2, b2,
        fn_dW1, zeros_b1, fn_dW2, zeros_b2,
        fn_eW1, zeros_b1, fn_eW2, zeros_b2,
        grid_size=41, span=1.0,
    )

    fig = plt.figure(figsize=(15, 4.8), dpi=140)
    AA, BB = np.meshgrid(alphas, betas)

    titles = [
        "Phase 1 Start (Dim #1 & Dim #2)\n(Schräge Flanke / Kamm → Minimum NICHT erkennbar)",
        "Phase 2: Gauß-Zufallsrichtungen (unnormiert)\n(skaliertes Netz → extrem spitzer Trichter)",
        "Phase 3: Gauß + Filternormierung (Li et al. 2018)\n(skalierungsinvariant → breites, flaches Tal)",
    ]
    grids = [losses_p1, losses_p2, losses_p3]

    for i, (title, grid) in enumerate(zip(titles, grids), start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.plot_surface(AA, BB, np.minimum(grid, 10.0), cmap="viridis", edgecolor="none")
        ax.scatter([0.0], [0.0], [grid[20, 20] + 0.15], color="#f43f5e", s=45, zorder=10)
        ax.set_title(title, fontsize=10)
        ax.set_zlim(0, 10)
        ax.set_xlabel("α")
        ax.set_ylabel("β")
        ax.set_zlabel("Loss")

    plt.tight_layout()
    out_img = os.path.join(PREP_DIR, "all_three_phases.png")
    plt.savefig(out_img)
    plt.close(fig)
    print(f"Saved updated 3-phase verification plot to {out_img}")


if __name__ == "__main__":
    main()
