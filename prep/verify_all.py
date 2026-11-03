#!/usr/bin/env python3
"""
Verify the analytical/procedural zero-latency landscape synthesizer (`directions.js`)
by plotting its exact 4-panel progression:
1. Phase 1 Startup (Dim #1 & Dim #2): Flat plateau with tilted corner ramp -> NO minimum visible!
2. Phase 1 Sweet Spot (Dim #46 & Dim #47): Coupled layer slice -> recognizable local basin!
3. Phase 2 Gaussian Random Directions (unnormalized): Razor-sharp funnel!
4. Phase 3 Gaussian + Filter Normalization (Li et al., 2018): Wide, flat generalizing valley!
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PREP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_LOSS = 0.058


def hash_signed(n: int) -> float:
    x = (n * 374761393) & 0xFFFFFFFF
    if x & 0x80000000:
        x -= 0x100000000
    x = (((x ^ (x >> 13)) * 1274126177)) & 0xFFFFFFFF
    if x & 0x80000000:
        x -= 0x100000000
    x = x ^ (x >> 16)
    return (x & 0x7FFFFFFF) / 1073741824.0 - 1.0


def soft_hinge(u: float) -> float:
    if u > 12.0:
        return u
    if u < -12.0:
        return 0.0
    return math.log(1.0 + math.exp(u))


def get_dim_profile(dim_idx: int):
    h1 = hash_signed(dim_idx * 101 + 7)
    h2 = hash_signed(dim_idx * 257 + 19)
    h3 = hash_signed(dim_idx * 509 + 31)
    if dim_idx <= 16:
        return {
            "kind": "ramp",
            "slope": (1 if dim_idx % 2 == 0 else -1) * (1.1 + 0.5 * abs(h1)),
            "hingeOffset": 0.25 + 0.25 * h2,
            "curv": 0.04,
            "rippleAmp": 0.12 * h3,
            "rippleFreq": 2.2,
        }
    if dim_idx <= 32:
        return {
            "kind": "crease",
            "slope": 1.4 * h1,
            "hingeOffset": -0.15 + 0.35 * h2,
            "curv": 0.35 + 0.4 * abs(h3),
            "asym": 1.8 + 0.9 * abs(h1),
            "rippleAmp": 0.25 * h2,
            "rippleFreq": 3.1,
        }
    if dim_idx <= 42:
        return {
            "kind": "canyon",
            "centerShift": 0.35 * h1,
            "curv": 3.2 + 1.5 * abs(h2),
            "asym": 0.8 * h3,
        }
    return {
        "kind": "coupled",
        "centerShift": 0.0,
        "curv": 2.8 + 0.9 * abs(h1),
        "quartic": 1.6 + 0.6 * abs(h2),
        "rippleAmp": 0.18 * h3,
    }


def eval_axis(prof, u: float, raw_coord: float) -> float:
    k = prof["kind"]
    if k == "ramp":
        s_sign = 1.0 if prof["slope"] >= 0 else -1.0
        arg = (u - prof["hingeOffset"]) * 2.6 * s_sign
        return (
            0.55 * abs(prof["slope"]) * soft_hinge(arg)
            + prof["curv"] * u * u
            + prof["rippleAmp"] * math.sin(prof["rippleFreq"] * raw_coord)
        )
    if k == "crease":
        shifted = u - prof["hingeOffset"]
        pos_side = (prof["asym"] if shifted > 0 else 0.35) * shifted * shifted
        return (
            pos_side
            + 0.45 * prof["slope"] * u
            + 0.2 * abs(prof["rippleAmp"]) * (1.0 - math.cos(prof["rippleFreq"] * u))
        )
    if k == "canyon":
        d = u - prof["centerShift"]
        return prof["curv"] * d * d + prof["asym"] * (d**3) * 0.25
    return (
        prof["curv"] * u * u
        + prof["quartic"] * (u**4) * 0.35
        + prof["rippleAmp"] * (1.0 - math.cos(2.5 * u))
    )


def synth_grid(mode, alpha_dim=1, alpha_str=1.0, beta_dim=2, beta_str=1.0, seed=2026, filter_norm=False, grid_size=41, span=1.0):
    losses = np.zeros((grid_size, grid_size), dtype=np.float32)
    alphas = np.linspace(-span, span, grid_size)
    betas = np.linspace(-span, span, grid_size)

    if mode == "manual":
        prof_a = get_dim_profile(alpha_dim)
        prof_b = get_dim_profile(beta_dim)
        cross_cpl = 0.35 * hash_signed(alpha_dim * 73 + beta_dim * 139)
        norm_scale = 0.52 if filter_norm else 1.0
        for ib, beta in enumerate(betas):
            ub = beta * beta_str * norm_scale
            vb = eval_axis(prof_b, ub, beta)
            for ia, alpha in enumerate(alphas):
                ua = alpha * alpha_str * norm_scale
                va = eval_axis(prof_a, ua, alpha)
                cross = cross_cpl * ua * ub
                if prof_a["kind"] == "ramp" and prof_b["kind"] == "ramp":
                    cross = 0.65 * soft_hinge(2.4 * (ua + 0.85 * ub - 0.55))
                losses[ib, ia] = BASE_LOSS + max(0.0, va + vb + cross)
        return alphas, betas, losses

    h1 = hash_signed(seed * 17 + 3)
    h2 = hash_signed(seed * 43 + 11)
    h3 = hash_signed(seed * 89 + 23)
    h4 = hash_signed(seed * 131 + 37)
    angle = h1 * math.pi
    cos_t, sin_t = math.cos(angle), math.sin(angle)
    ecc_x = 1.0 + 0.18 * h2
    ecc_y = 1.0 - 0.14 * h2

    for ib, beta in enumerate(betas):
        for ia, alpha in enumerate(alphas):
            rx = (cos_t * alpha - sin_t * beta) * ecc_x
            ry = (sin_t * alpha + cos_t * beta) * ecc_y
            r2 = rx * rx + ry * ry
            theta = math.atan2(ry, rx)
            if not filter_norm:
                steep = 26.5 + 3.0 * h3
                facet = 1.0 + 0.085 * math.cos(3.0 * theta + h3 * 3.0) + 0.055 * math.sin(5.0 * theta + h4 * 4.0)
                cone = steep * (math.sqrt(r2 + 0.018) - math.sqrt(0.018)) * facet + 14.0 * r2 * facet
                losses[ib, ia] = BASE_LOSS + cone
            else:
                v_scale = 2.15 + 0.25 * h3
                aniso = 1.0 + 0.06 * math.cos(2.0 * theta + h3 * 2.0) + 0.035 * math.sin(3.0 * theta + h4 * 3.0)
                basin = v_scale * (0.62 * r2 + 0.42 * r2 * r2) * aniso
                losses[ib, ia] = BASE_LOSS + basin

    return alphas, betas, losses


def main():
    alphas, betas, g_start = synth_grid("manual", 1, 1.0, 2, 1.0)
    _, _, g_sweet = synth_grid("manual", 46, 1.0, 47, 1.0)
    _, _, g_funnel = synth_grid("random", seed=2026, filter_norm=False)
    _, _, g_valley = synth_grid("random", seed=2026, filter_norm=True)

    fig = plt.figure(figsize=(18, 4.5), dpi=140)
    AA, BB = np.meshgrid(alphas, betas)

    items = [
        ("1. Start (Dim #1 & #2)\nFlaches Plateau + Kante (Kein Minimum!)", g_start),
        ("2. Manuell gefunden (Dim #46 & #47)\nLokale Mulde bei gekoppelten Schichten", g_sweet),
        ("3. Gauß OHNE Filternormierung\n(Skaliertes Netz → spitzer Trichter)", g_funnel),
        ("4. Gauß MIT Filternormierung\n(Li et al. 2018 → breites, flaches Tal)", g_valley),
    ]

    for i, (title, grid) in enumerate(items, start=1):
        ax = fig.add_subplot(1, 4, i, projection="3d")
        ax.plot_surface(AA, BB, np.minimum(grid, 10.0), cmap="viridis", edgecolor="none")
        ax.scatter([0.0], [0.0], [grid[20, 20] + 0.12], color="#f43f5e", s=40, zorder=10)
        ax.set_title(title, fontsize=9.5)
        ax.set_zlim(0, 10)
        ax.set_xlabel("α")
        ax.set_ylabel("β")
        ax.set_zlabel("Loss")

    plt.tight_layout()
    out_path = os.path.join(PREP_DIR, "all_three_phases.png")
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved 4-panel verification plot to {out_path}")


if __name__ == "__main__":
    main()
