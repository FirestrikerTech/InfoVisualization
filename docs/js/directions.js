/**
 * High-dimensional direction explorer & generator for the MNIST MLP (784 -> 32 -> 10).
 *
 * Simplified 4-Slider Interface (Phase 1):
 * - Slider 1 (Achse α - Dimension): Scrolls through high-dimensional subspaces (Dim #1 .. #50).
 * - Slider 2 (Achse α - Stärke):    Adjusts the strength/amplitude along the selected dimension.
 * - Slider 3 (Achse β - Dimension): Scrolls through high-dimensional subspaces (Dim #1 .. #50).
 * - Slider 4 (Achse β - Stärke):    Adjusts the strength/amplitude along the selected dimension.
 *
 * Why the audience does NOT immediately see the minimum on startup:
 * - Each dimension index k corresponds to a deterministic localized coordinate/block direction
 *   in the 25,450-dimensional space, PLUS a linear gradient component on early dimensions
 *   (or asymmetric single-layer slice) so that the initial view (Dim #1 & Dim #2) is a
 *   tilted slope / ridge where (0,0) sits on a hillside or flat saddle rather than a nice bowl!
 * - As the user scrolls through dimensions (especially around Dim #24..#32 where hidden and
 *   output layers couple balancedly) and tunes the strengths, they can discover a local valley
 *   by hand—experiencing firsthand how hard it is to find a 2D slice in 25,450 dimensions!
 */

const D_IN = 784;
const D_HID = 32;
const D_OUT = 10;

export const NUM_SCROLL_DIMS = 50;

/**
 * Seeded PRNG (Mulberry32) + Box-Muller Gaussian generator.
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
 * Human-readable description of what each dimension index k (1..50) represents
 * in the 25,450-dimensional parameter space.
 */
export function describeDimension(dimIndex) {
  if (dimIndex <= 16) {
    return `Dim #${dimIndex} · Eingabepixel-Block ${dimIndex}/16 (Schicht 1)`;
  }
  if (dimIndex <= 32) {
    const neuronIdx = dimIndex - 16;
    return `Dim #${dimIndex} · Verstecktes Neuron #${neuronIdx} (Schicht 1+2)`;
  }
  if (dimIndex <= 42) {
    const digit = dimIndex - 33;
    return `Dim #${dimIndex} · Ausgabe-Klasse „Ziffer ${digit}“ (Schicht 2)`;
  }
  return `Dim #${dimIndex} · Schicht-Kopplung #${dimIndex - 42} (Diagonal-Schnitt)`;
}

/**
 * Build a single direction vector (dW1, db1, dW2, db2) for a chosen dimension index (1..50)
 * and user-selected strength (-2.0 .. +2.0), PLUS an optional coordinate shift `(offA, offB)`
 * so that in manual mode the user starts on a steep hillside / ridge AWAY from the minimum,
 * or only perturbs a single coordinate/filter where the minimum is NOT visible!
 *
 * Why did the previous startup still curve up?
 * Because any perturbation centered exactly at a well-trained minimum theta* increases Cross-Entropy
 * in both +alpha and -alpha if it affects active logits!
 * By making Dimensions 1..15 perturb dead/corner pixels (where MNIST pixels are 0) OR a single
 * weight coordinate with a slight offset from the minimum center, the startup landscape is a
 * flat plateau / tilted ramp / ridge where NO minimum is visible!
 * Only if the user scrolls to the coupled dimensions (e.g. Dim 25..35) and tunes the strength
 * do they slice right through the minimum!
 */
export function buildSingleAxisDirection(dimIndex, strength, W1, b1, W2, b2) {
  const dW1 = new Float32Array(D_HID * D_IN);
  const db1 = new Float32Array(D_HID);
  const dW2 = new Float32Array(D_OUT * D_HID);
  const db2 = new Float32Array(D_OUT);

  const s = strength;
  if (Math.abs(s) < 1e-6) {
    return { dW1, db1, dW2, db2 };
  }

  const rng = createRng(dimIndex * 7919 + 101);

  if (dimIndex <= 15) {
    // Border/corner pixel blocks (outer 3-pixel margin of 28x28 MNIST images where digits are zero)
    // plus subtle weight coupling -> creates a flat plateau or a single-sided slope / saddle
    // where NO minimum valley is visible at all!
    const neuronA = (dimIndex * 3) % D_HID;
    for (let k = 0; k < D_IN; k++) {
      const r = Math.floor(k / 28);
      const c = k % 28;
      if (r < 4 || r > 24 || c < 4 || c > 24) {
        dW1[neuronA * D_IN + k] = s * 14.0;
      }
    }
    // Single coordinate perturbation in W1 on edge pixels
    const row = (dimIndex + 2) % 28;
    dW1[neuronA * D_IN + row * 28 + 2] = s * 25.0;
    // Slight asymmetric shift on a single weight in W2 so one corner tilts gently up to Loss ~3.5
    const cls = dimIndex % D_OUT;
    dW2[cls * D_HID + neuronA] = s * 0.042;
    db1[neuronA] = s * 7.5;
  } else if (dimIndex <= 32) {
    // Hidden neuron #n perturbation (creates a 1D canyon / crease, still open on the other axis!)
    const nIdx = dimIndex - 16;
    for (let k = 0; k < D_IN; k++) {
      dW1[nIdx * D_IN + k] = s * rng.nextGaussian() * 4.5;
    }
    for (let c = 0; c < D_OUT; c++) {
      dW2[c * D_HID + nIdx] = s * rng.nextGaussian() * 0.11;
    }
  } else if (dimIndex <= 42) {
    // Output class #c perturbation (1D parabolic trough along one axis, flat on the orthogonal axis)
    const cIdx = dimIndex - 33;
    for (let j = 0; j < D_HID; j++) {
      dW2[cIdx * D_HID + j] = s * rng.nextGaussian() * 0.14;
    }
  } else {
    // Dim 43..50: Coupled multi-layer subspace where manual exploration CAN find a nice minimum!
    for (let i = 0; i < dW1.length; i++) {
      dW1[i] = s * rng.nextGaussian() * 1.8;
    }
    for (let i = 0; i < dW2.length; i++) {
      dW2[i] = s * rng.nextGaussian() * 0.045;
    }
  }

  return { dW1, db1, dW2, db2 };
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
    if (dSq > 1e-16) {
      const scale = Math.sqrt(wSq) / (Math.sqrt(dSq) + eps);
      for (let k = 0; k < D_IN; k++) {
        outW1[off + k] = dW1[off + k] * scale;
      }
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
    if (dSq > 1e-16) {
      const scale = Math.sqrt(wSq) / (Math.sqrt(dSq) + eps);
      for (let j = 0; j < D_HID; j++) {
        outW2[off + j] = dW2[off + j] * scale;
      }
    }
  }

  return { dW1: outW1, dW2: outW2 };
}

/**
 * Build the manual direction pair (delta, eta) from the 4 sliders:
 *   (alphaDim, alphaStrength, betaDim, betaStrength)
 */
export function buildManualDirections4Sliders(
  alphaDim,
  alphaStrength,
  betaDim,
  betaStrength,
  W1,
  b1,
  W2,
  b2,
  useFilterNorm
) {
  const dirA = buildSingleAxisDirection(alphaDim, alphaStrength, W1, b1, W2, b2);
  const dirB = buildSingleAxisDirection(betaDim, betaStrength, W1, b1, W2, b2);

  let dW1 = dirA.dW1;
  let db1 = dirA.db1;
  let dW2 = dirA.dW2;
  let db2 = dirA.db2;

  let eW1 = dirB.dW1;
  let eb1 = dirB.db1;
  let eW2 = dirB.dW2;
  let eb2 = dirB.db2;

  if (useFilterNorm) {
    const fnD = filterNormalize(dW1, dW2, W1, W2);
    const fnE = filterNormalize(eW1, eW2, W1, W2);
    dW1 = fnD.dW1;
    dW2 = fnD.dW2;
    eW1 = fnE.dW1;
    eW2 = fnE.dW2;
    db1 = new Float32Array(D_HID);
    db2 = new Float32Array(D_OUT);
    eb1 = new Float32Array(D_HID);
    eb2 = new Float32Array(D_OUT);
  }

  return { dW1, db1, dW2, db2, eW1, eb1, eW2, eb2 };
}

/**
 * Generate isotropic Gaussian directions delta, eta ~ N(0, I) across all weight parameters.
 * - When useFilterNorm == false (Phase 2): scaled by sigmaUnnorm, producing
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
