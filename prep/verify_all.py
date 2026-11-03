#!/usr/bin/env python3
"""
Verify the exported binary files (`weights.bin`, `data.bin`, `meta.json`) and generate
a 3-panel overview plot of all 3 phases (Phase 1 Manual Mix, Phase 2 Unnormalized Gaussian
Funnel, Phase 3 Filter-Normalized Valley) to visually and numerically verify everything.
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


def main():
    meta_path = os.path.join(ASSETS_DIR, "meta.json")
    weights_path = os.path.join(ASSETS_DIR, "weights.bin")
    data_path = os.path.join(ASSETS_DIR, "data.bin")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    weights = np.fromfile(weights_path, dtype="<f4")
    data_u8 = np.fromfile(data_path, dtype=np.uint8)

    N = meta["num_samples"]
    o = meta["offsets"]

    W1 = weights[o["W1"][0] : o["W1"][1]].reshape(32, 784)
    b1 = weights[o["b1"][0] : o["b1"][1]].reshape(32)
    W2 = weights[o["W2"][0] : o["W2"][1]].reshape(10, 32)
    b2 = weights[o["b2"][0] : o["b2"][1]].reshape(10)

    X = data_u8[: N * 784].reshape(N, 784).astype(np.float32) * np.float32(1.0 / 255.0)
    y = data_u8[N * 784 : N * 784 + N].astype(np.int64)

    loss_c, acc_c = forward_loss_numpy(X, y, W1, b1, W2, b2)
    diff_loss = abs(loss_c - meta["verification"]["center_loss"])
    diff_acc = abs(acc_c - meta["verification"]["center_acc"])

    print(f"Verified center (0,0) from binary assets: loss={loss_c:.6f} (diff={diff_loss:.2e}), acc={acc_c*100:.2f}%")
    assert diff_loss < 1e-5, f"Center loss mismatch: {diff_loss}"
    assert diff_acc < 1e-6, f"Center acc mismatch: {diff_acc}"

    # Reconstruct Phase 1 default manual basis directions (Top/Bottom vs Left/Right pixel regions)
    dW1_p1 = np.zeros_like(W1)
    eW1_p1 = np.zeros_like(W1)
    for j in range(32):
        s1 = 1.0 if j % 2 == 0 else -1.0
        s2 = 1.0 if (j % 4) < 2 else -1.0
        for r in range(28):
            for c in range(28):
                dW1_p1[j, r * 28 + c] = 0.8 * s1 * (1.0 if r < 14 else -1.0) * 2.2
                eW1_p1[j, r * 28 + c] = 0.8 * s2 * (1.0 if c < 14 else -1.0) * 2.2

    zeros_b1 = np.zeros_like(b1)
    zeros_W2 = np.zeros_like(W2)
    zeros_b2 = np.zeros_like(b2)

    alphas, betas, losses_p1, _ = compute_grid(
        X, y, W1, b1, W2, b2,
        dW1_p1, zeros_b1, zeros_W2, zeros_b2,
        eW1_p1, zeros_b1, zeros_W2, zeros_b2,
        grid_size=41, span=1.0,
    )

    # Phase 2 & 3 directions
    rng = np.random.default_rng(2026)
    raw_dW1 = rng.standard_normal(W1.shape).astype(np.float32)
    raw_dW2 = rng.standard_normal(W2.shape).astype(np.float32)
    raw_eW1 = rng.standard_normal(W1.shape).astype(np.float32)
    raw_eW2 = rng.standard_normal(W2.shape).astype(np.float32)

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
        "Phase 1: Manuelle Basis-Richtungen\n(flache Rinne / kaum Krümmung in W2)",
        "Phase 2: Gauß-Zufallsrichtungen (unnormiert)\n(skaliertes Netz → extrem spitzer Trichter)",
        "Phase 3: Gauß + Filternormierung (Li et al. 2018)\n(skalierungsinvariant → breites, flaches Tal)",
    ]
    grids = [losses_p1, losses_p2, losses_p3]

    for i, (title, grid) in enumerate(zip(titles, grids), start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.plot_surface(AA, BB, np.minimum(grid, 10.0), cmap="viridis", edgecolor="none")
        ax.scatter([0.0], [0.0], [grid[20, 20] + 0.1], color="#f43f5e", s=40, zorder=10)
        ax.set_title(title, fontsize=10)
        ax.set_zlim(0, 10)
        ax.set_xlabel("α")
        ax.set_ylabel("β")
        ax.set_zlabel("Loss")

    plt.tight_layout()
    out_img = os.path.join(PREP_DIR, "all_three_phases.png")
    plt.savefig(out_img)
    plt.close(fig)
    print(f"Saved 3-phase verification plot to {out_img}")


if __name__ == "__main__":
    main()
