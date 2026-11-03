/**
 * High-dimensional direction generator & mixer for the MNIST MLP (784 -> 32 -> 10).
 *
 * Implements:
 * 1. Phase 1: Intuitive manual "Direction Mixer" (Basis directions in 25,450-D space).
 *    - Allows the audience to manually adjust sliders to combine interpretable directions
 *      (pixel regions, hidden neuron groups, output class groups, radial weight scaling).
 *    - Typically produces tilted planes, ridges, or shallow slopes because structured/low-rank
 *      or single-layer perturbations fail to capture full-space curvature, UNLESS the user
 *      carefully balances both layers and rotates the plane angle!
 * 2. Phase 2: Isotropic Gaussian Random Directions delta, eta ~ N(0, I) across all 25,450 dims
 *    (unnormalized, scaled by sigma_unnorm so that on the artificially rescaled network it
 *    reveals a dramatic, razor-sharp funnel).
 * 3. Phase 3: Filter-Normalized Random Directions following Li et al. (2018), Section 4:
 *      d_{i,j} <- (d_{i,j} / ||d_{i,j}||) * ||theta_{i,j}||
 *    per neuron (row j of W1 and row j of W2), with biases set to 0 (`--xignore biasbn`).
 *    Reveals the true scale-invariant, wide, flat generalizing valley!
 */

const D_IN = 784;
const D_HID = 32;
const D_OUT = 10;

/**
 * Seeded PRNG (Mulberry32) + Box-Muller Gaussian generator so users/presenter
 * can reproduce or re-roll random directions cleanly.
 */
export function createRng(seed) {
  let a = seed >>> 0;
  function nextUniform() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  function nextGaussian() {
    let u1 = 0;
    let u2 = 0;
    while (u1 <= 1e-12) u1 = nextUniform();
    u2 = nextUniform();
    return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  }
  return { nextUniform, nextGaussian };
}

/**
 * Definition of the intuitive basis directions for Phase 1 ("Richtungs-Mixer").
 */
export const BASIS_COMPONENTS = [
  {
    id: "pixTopBottom",
    label: "Bildregion Oben ↔ Unten",
    shortDesc: "Gewichte für obere vs. untere Bildhälfte",
    defaultAlpha: 0.8,
    defaultBeta: 0.0,
  },
  {
    id: "pixLeftRight",
    label: "Bildregion Links ↔ Rechts",
    shortDesc: "Gewichte für linke vs. rechte Bildhälfte",
    defaultAlpha: 0.0,
    defaultBeta: 0.8,
  },
  {
    id: "pixCenterEdge",
    label: "Bildmitte ↔ Bildrand",
    shortDesc: "Zentrum der Ziffer vs. Außenbereich",
    defaultAlpha: 0.0,
    defaultBeta: 0.0,
  },
  {
    id: "hidGroupAB",
    label: "Neuronen-Gruppe A ↔ B",
    shortDesc: "Versteckte Schicht (Neuronen 1–16 vs. 17–32)",
    defaultAlpha: 0.0,
    defaultBeta: 0.0,
  },
  {
    id: "outDigits",
    label: "Ausgabe: Ziffern 0–4 ↔ 5–9",
    shortDesc: "Klassifikations-Schicht (letzte Schicht)",
    defaultAlpha: 0.0,
    defaultBeta: 0.0,
  },
  {
    id: "radialScale",
    label: "Gewichts-Skalierung (Radial)",
    shortDesc: "Alle Gewichte proportional zu θ* strecken",
    defaultAlpha: 0.0,
    defaultBeta: 0.0,
  },
];

/**
 * Precompute deterministic basis templates so slider adjustments are instantaneous.
 */
export function buildBasisTemplates(W1, b1, W2, b2) {
  const rng = createRng(1337);

  // Pre-generate fixed sign/checker patterns so each basis vector is non-trivial
  const signHid = new Float32Array(D_HID);
  for (let j = 0; j < D_HID; j++) signHid[j] = j % 2 === 0 ? 1.0 : -1.0;

  const templates = {};

  // 1. pixTopBottom: modulates W1 rows on top half vs bottom half of 28x28 image
  {
    const dW1 = new Float32Array(D_HID * D_IN);
    const dW2 = new Float32Array(D_OUT * D_HID);
    for (let j = 0; j < D_HID; j++) {
      const s = signHid[j];
      for (let r = 0; r < 28; r++) {
        const v = r < 14 ? 1.0 : -1.0;
        for (let c = 0; c < 28; c++) {
          dW1[j * D_IN + r * 28 + c] = s * v * 2.2;
        }
      }
    }
    templates.pixTopBottom = { dW1, dW2 };
  }

  // 2. pixLeftRight: modulates W1 rows on left half vs right half of 28x28 image
  {
    const dW1 = new Float32Array(D_HID * D_IN);
    const dW2 = new Float32Array(D_OUT * D_HID);
    for (let j = 0; j < D_HID; j++) {
      const s = j % 4 < 2 ? 1.0 : -1.0;
      for (let r = 0; r < 28; r++) {
        for (let c = 0; c < 28; c++) {
          const v = c < 14 ? 1.0 : -1.0;
          dW1[j * D_IN + r * 28 + c] = s * v * 2.2;
        }
      }
    }
    templates.pixLeftRight = { dW1, dW2 };
  }

  // 3. pixCenterEdge: radial ring in 28x28 pixel space + coupled hidden pattern
  {
    const dW1 = new Float32Array(D_HID * D_IN);
    const dW2 = new Float32Array(D_OUT * D_HID);
    for (let j = 0; j < D_HID; j++) {
      const phase = Math.cos((j + 1) * 1.7);
      for (let r = 0; r < 28; r++) {
        for (let c = 0; c < 28; c++) {
          const dist = Math.hypot(r - 13.5, c - 13.5);
          const v = dist < 7.5 ? 1.0 : -0.55;
          dW1[j * D_IN + r * 28 + c] = phase * v * 2.6;
        }
      }
    }
    templates.pixCenterEdge = { dW1, dW2 };
  }

  // 4. hidGroupAB: perturbs hidden neuron groups (1..16 vs 17..32) with rich internal structure
  {
    const dW1 = new Float32Array(D_HID * D_IN);
    const dW2 = new Float32Array(D_OUT * D_HID);
    for (let j = 0; j < D_HID; j++) {
      const groupSign = j < 16 ? 1.0 : -1.0;
      for (let k = 0; k < D_IN; k++) {
        dW1[j * D_IN + k] = groupSign * rng.nextGaussian() * 1.35;
      }
    }
    for (let c = 0; c < D_OUT; c++) {
      for (let j = 0; j < D_HID; j++) {
        const groupSign = j < 16 ? 1.0 : -1.0;
        dW2[c * D_HID + j] = groupSign * ((c % 2 === 0 ? 1 : -1) * 0.032);
      }
    }
    templates.hidGroupAB = { dW1, dW2 };
  }

  // 5. outDigits: perturbs output layer W2 (classes 0-4 vs 5-9)
  // Because W2_scaled was divided by c=10, even modest perturbations in W2 have strong effect!
  {
    const dW1 = new Float32Array(D_HID * D_IN);
    const dW2 = new Float32Array(D_OUT * D_HID);
    for (let c = 0; c < D_OUT; c++) {
      const classSign = c < 5 ? 1.0 : -1.0;
      for (let j = 0; j < D_HID; j++) {
        const hWave = Math.sin((c + 1) * (j + 1) * 0.9);
        dW2[c * D_HID + j] = classSign * hWave * 0.055;
      }
    }
    templates.outDigits = { dW1, dW2 };
  }

  // 6. radialScale: proportional to the trained weights themselves (opposing signs on W1 vs W2
  // so it breaks scale invariance and curves upward on both sides when combined)
  {
    const dW1 = new Float32Array(D_HID * D_IN);
    const dW2 = new Float32Array(D_OUT * D_HID);
    for (let i = 0; i < dW1.length; i++) dW1[i] = W1[i] * 0.65;
    for (let i = 0; i < dW2.length; i++) dW2[i] = W2[i] * 0.95;
    templates.radialScale = { dW1, dW2 };
  }

  return templates;
}

/**
 * Filter-normalize a direction (dW1, dW2) row-by-row (neuron-by-neuron) to match the filter norms
 * of (W1, W2), exactly following Li et al. (2018) Section 4:
 *   d_{i,j} <- (d_{i,j} / ||d_{i,j}||) * ||theta_{i,j}||
 */
export function filterNormalize(dW1, dW2, W1, W2) {
  const eps = 1e-10;
  const outW1 = new Float32Array(D_HID * D_IN);
  const outW2 = new Float32Array(D_OUT * D_HID);

  for (let j = 0; j < D_HID; j++) {
    const off = j * D_IN;
    let wSq = 0.0;
    let dSq = 0.0;
    for (let k = 0; k < D_IN; k++) {
      const w = W1[off + k];
      const d = dW1[off + k];
      wSq += w * w;
      dSq += d * d;
    }
    const scale = Math.sqrt(wSq) / (Math.sqrt(dSq) + eps);
    for (let k = 0; k < D_IN; k++) {
      outW1[off + k] = dW1[off + k] * scale;
    }
  }

  for (let c = 0; c < D_OUT; c++) {
    const off = c * D_HID;
    let wSq = 0.0;
    let dSq = 0.0;
    for (let j = 0; j < D_HID; j++) {
      const w = W2[off + j];
      const d = dW2[off + j];
      wSq += w * w;
      dSq += d * d;
    }
    const scale = Math.sqrt(wSq) / (Math.sqrt(dSq) + eps);
    for (let j = 0; j < D_HID; j++) {
      outW2[off + j] = dW2[off + j] * scale;
    }
  }

  return { dW1: outW1, dW2: outW2 };
}

/**
 * Build the manual direction pair (delta, eta) from user mixer weights and rotation angle.
 */
export function buildManualDirections(templates, alphaMix, betaMix, angleDeg, W1, W2, useFilterNorm) {
  const rawDW1 = new Float32Array(D_HID * D_IN);
  const rawDW2 = new Float32Array(D_OUT * D_HID);
  const rawEW1 = new Float32Array(D_HID * D_IN);
  const rawEW2 = new Float32Array(D_OUT * D_HID);

  for (const comp of BASIS_COMPONENTS) {
    const id = comp.id;
    const wa = alphaMix[id] || 0.0;
    const wb = betaMix[id] || 0.0;
    const t = templates[id];
    if (Math.abs(wa) > 1e-6) {
      for (let i = 0; i < rawDW1.length; i++) rawDW1[i] += wa * t.dW1[i];
      for (let i = 0; i < rawDW2.length; i++) rawDW2[i] += wa * t.dW2[i];
    }
    if (Math.abs(wb) > 1e-6) {
      for (let i = 0; i < rawEW1.length; i++) rawEW1[i] += wb * t.dW1[i];
      for (let i = 0; i < rawEW2.length; i++) rawEW2[i] += wb * t.dW2[i];
    }
  }

  // Apply 2D plane rotation angle (in degrees) between delta and eta
  const rad = (angleDeg * Math.PI) / 180.0;
  const cosA = Math.cos(rad);
  const sinA = Math.sin(rad);

  let dW1 = new Float32Array(D_HID * D_IN);
  let dW2 = new Float32Array(D_OUT * D_HID);
  let eW1 = new Float32Array(D_HID * D_IN);
  let eW2 = new Float32Array(D_OUT * D_HID);

  for (let i = 0; i < dW1.length; i++) {
    dW1[i] = cosA * rawDW1[i] - sinA * rawEW1[i];
    eW1[i] = sinA * rawDW1[i] + cosA * rawEW1[i];
  }
  for (let i = 0; i < dW2.length; i++) {
    dW2[i] = cosA * rawDW2[i] - sinA * rawEW2[i];
    eW2[i] = sinA * rawDW2[i] + cosA * rawEW2[i];
  }

  if (useFilterNorm) {
    const fnD = filterNormalize(dW1, dW2, W1, W2);
    const fnE = filterNormalize(eW1, eW2, W1, W2);
    dW1 = fnD.dW1;
    dW2 = fnD.dW2;
    eW1 = fnE.dW1;
    eW2 = fnE.dW2;
  }

  return {
    dW1,
    db1: new Float32Array(D_HID),
    dW2,
    db2: new Float32Array(D_OUT),
    eW1,
    eb1: new Float32Array(D_HID),
    eW2,
    eb2: new Float32Array(D_OUT),
  };
}

/**
 * Generate isotropic Gaussian directions delta, eta ~ N(0, I) across all weight parameters.
 * - When useFilterNorm == false (Phase 2): scaled by sigmaUnnorm (default 0.12), producing
 *   the dramatic sharp funnel on the rescaled network.
 * - When useFilterNorm == true  (Phase 3): filter-normalized per neuron according to
 *   Li et al. (2018), producing the wide, flat generalizing valley.
 */
export function buildGaussianDirections(seed, sigmaUnnorm, W1, W2, useFilterNorm) {
  const rng = createRng(seed);

  const rawDW1 = new Float32Array(D_HID * D_IN);
  const rawDW2 = new Float32Array(D_OUT * D_HID);
  const rawEW1 = new Float32Array(D_HID * D_IN);
  const rawEW2 = new Float32Array(D_OUT * D_HID);

  for (let i = 0; i < rawDW1.length; i++) rawDW1[i] = rng.nextGaussian();
  for (let i = 0; i < rawDW2.length; i++) rawDW2[i] = rng.nextGaussian();
  for (let i = 0; i < rawEW1.length; i++) rawEW1[i] = rng.nextGaussian();
  for (let i = 0; i < rawEW2.length; i++) rawEW2[i] = rng.nextGaussian();

  let dW1, dW2, eW1, eW2;

  if (useFilterNorm) {
    const fnD = filterNormalize(rawDW1, rawDW2, W1, W2);
    const fnE = filterNormalize(rawEW1, rawEW2, W1, W2);
    dW1 = fnD.dW1;
    dW2 = fnD.dW2;
    eW1 = fnE.dW1;
    eW2 = fnE.dW2;
  } else {
    dW1 = new Float32Array(D_HID * D_IN);
    dW2 = new Float32Array(D_OUT * D_HID);
    eW1 = new Float32Array(D_HID * D_IN);
    eW2 = new Float32Array(D_OUT * D_HID);
    for (let i = 0; i < dW1.length; i++) {
      dW1[i] = rawDW1[i] * sigmaUnnorm;
      eW1[i] = rawEW1[i] * sigmaUnnorm;
    }
    for (let i = 0; i < dW2.length; i++) {
      dW2[i] = rawDW2[i] * sigmaUnnorm;
      eW2[i] = rawEW2[i] * sigmaUnnorm;
    }
  }

  return {
    dW1,
    db1: new Float32Array(D_HID),
    dW2,
    db2: new Float32Array(D_OUT),
    eW1,
    eb1: new Float32Array(D_HID),
    eW2,
    eb2: new Float32Array(D_OUT),
  };
}
