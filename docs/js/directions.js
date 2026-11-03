/**
 * Zero-Latency Multi-Dimensional Direction Vector & Landscape Synthesizer.
 *
 * How the 4-Slider Coordinate Vector Model works (Phase 1):
 * - Axis α represents the high-dimensional direction vector δ = (δ_1, δ_2, ..., δ_50).
 * - Axis β represents the high-dimensional direction vector η = (η_1, η_2, ..., η_50).
 * - Slider 1 ("Durch Dimensionen scrollen") ONLY selects which coordinate index i ∈ {1..50}
 *   of vector δ you are currently inspecting! Scrolling Slider 1 does NOT change δ or the
 *   3D plot—it simply updates Slider 2 to show the stored value δ[i].
 * - Slider 2 ("Wert in dieser Dimension") modifies δ[i] and stores it persistently across
 *   all 50 dimensions!
 * - Likewise, Slider 3 selects coordinate j ∈ {1..50} of vector η (without changing the plot),
 *   and Slider 4 modifies and stores η[j]!
 *
 * Startup state:
 * - Only δ[1] = +1.0 and η[2] = +1.0 are non-zero (all other 48 coordinates are 0.00).
 * - Because Dim #1 and Dim #2 are outer border/ramp directions, the initial landscape is a
 *   flat plateau with a tilted edge—NO minimum is visible!
 * - If the user adjusts multiple dimensions (especially coupled dimensions or balanced
 *   coordinates across layers), the combined direction vectors δ and η gradually carve out
 *   a 2D slice where the minimum becomes visible!
 */

export const NUM_SCROLL_DIMS = 50;
const BASE_LOSS = 0.058;

/**
 * Deterministic signed hash in [-1, 1].
 */
function hashSigned(n) {
  let x = (n * 374761393) | 0;
  x = ((x ^ (x >> 13)) * 1274126177) | 0;
  x = x ^ (x >> 16);
  return (x & 0x7fffffff) / 1073741824.0 - 1.0;
}

/**
 * Smooth ReLU / Softplus helper: log(1 + exp(u))
 */
function softHinge(u) {
  if (u > 12.0) return u;
  if (u < -12.0) return 0.0;
  return Math.log(1.0 + Math.exp(u));
}

/**
 * Human-readable description of each dimension index k (1..50) in the parameter space.
 */
export function describeDimension(dimIndex) {
  if (dimIndex <= 16) {
    return `Dim #${dimIndex} · Randpixel-Gewicht ${dimIndex}/16 (Schicht 1)`;
  }
  if (dimIndex <= 32) {
    const neuronIdx = dimIndex - 16;
    return `Dim #${dimIndex} · Verstecktes Neuron #${neuronIdx} (Schicht 1)`;
  }
  if (dimIndex <= 42) {
    const digit = dimIndex - 33;
    return `Dim #${dimIndex} · Ausgabeklasse „Ziffer ${digit}“ (Schicht 2)`;
  }
  return `Dim #${dimIndex} · Schicht-Kopplung #${dimIndex - 42} (Schicht 1+2)`;
}

// Precompute basis properties for all 50 dimensions once at module load (< 0.05 ms)
const DIM_BASIS = new Array(NUM_SCROLL_DIMS + 1);
for (let d = 1; d <= NUM_SCROLL_DIMS; d++) {
  const h1 = hashSigned(d * 101 + 7);
  const h2 = hashSigned(d * 257 + 19);
  const h3 = hashSigned(d * 509 + 31);

  if (d <= 16) {
    // Border/ramp dimensions: contribute linear tilt + one-sided hinge, almost zero curvature!
    // So when only Dim #1 and Dim #2 are active at startup, the plot is a tilted ramp/plateau
    // with NO minimum bowl!
    DIM_BASIS[d] = {
      tilt: (d % 2 === 0 ? 1.0 : -1.0) * (0.85 + 0.35 * Math.abs(h1)),
      hingeDir: d % 3 === 0 ? -1.0 : 1.0,
      hingeWeight: 0.75 + 0.3 * Math.abs(h2),
      curv: 0.02, // negligible curvature -> no minimum visible!
      quartic: 0.0,
      shift: 0.35 * h3,
      wave: 0.08 * h1,
    };
  } else if (d <= 32) {
    // Hidden neuron dimensions: moderate curvature + asymmetric shift/crease
    DIM_BASIS[d] = {
      tilt: 0.55 * h1,
      hingeDir: h2 >= 0 ? 1.0 : -1.0,
      hingeWeight: 0.65 * Math.abs(h2),
      curv: 0.85 + 0.55 * Math.abs(h3),
      quartic: 0.25,
      shift: 0.22 * h1,
      wave: 0.14 * h2,
    };
  } else if (d <= 42) {
    // Output digit logit dimensions: strong curvature along that axis, slight center offset
    DIM_BASIS[d] = {
      tilt: 0.3 * h2,
      hingeDir: 1.0,
      hingeWeight: 0.25,
      curv: 2.1 + 0.9 * Math.abs(h1),
      quartic: 0.7,
      shift: 0.15 * h3,
      wave: 0.1 * h1,
    };
  } else {
    // Dim 43..50: Balanced coupled weights -> strong centered curvature around theta* (0,0)
    DIM_BASIS[d] = {
      tilt: 0.0,
      hingeDir: 1.0,
      hingeWeight: 0.1,
      curv: 2.75 + 0.85 * Math.abs(h1),
      quartic: 1.45 + 0.5 * Math.abs(h2),
      shift: 0.0,
      wave: 0.12 * h3,
    };
  }
}

/**
 * Aggregate the 50-element user vector `vec` (Float32Array of length 51, 1-indexed)
 * into effective 1D directional profile coefficients.
 * Also counts how many dimensions are actively mixed: mixing MORE dimensions naturally
 * averages out random linear tilts and reveals the true convex minimum basin around (0,0)!
 */
function aggregateVectorProfile(vec, useFilterNorm) {
  let sumTilt = 0.0;
  let sumHingePos = 0.0;
  let sumHingeNeg = 0.0;
  let sumCurv = 0.0;
  let sumQuartic = 0.0;
  let sumShiftNum = 0.0;
  let sumWave = 0.0;
  let normSq = 0.0;
  let activeCount = 0;

  for (let d = 1; d <= NUM_SCROLL_DIMS; d++) {
    const v = vec[d];
    if (Math.abs(v) < 1e-4) continue;
    activeCount++;
    const b = DIM_BASIS[d];
    const v2 = v * v;
    normSq += v2;

    sumTilt += v * b.tilt;
    if (v * b.hingeDir >= 0) {
      sumHingePos += Math.abs(v) * b.hingeWeight;
    } else {
      sumHingeNeg += Math.abs(v) * b.hingeWeight;
    }
    sumCurv += v2 * b.curv;
    sumQuartic += v2 * b.quartic;
    sumShiftNum += v * b.shift * b.curv;
    sumWave += v * b.wave;
  }

  // If the user combines multiple dimensions (e.g. >= 3 active dimensions on an axis),
  // high-dimensional concentration of measure kicks in: curvature reinforces while tilts cancel!
  if (activeCount >= 2) {
    const multiBonus = Math.min(2.5, (activeCount - 1) * 0.65);
    sumCurv += normSq * multiBonus;
    sumQuartic += normSq * multiBonus * 0.45;
    sumTilt *= Math.pow(0.65, activeCount - 1);
  }

  if (useFilterNorm && normSq > 1e-6) {
    const invNorm = 1.0 / Math.sqrt(normSq);
    sumTilt *= invNorm * 0.35;
    sumHingePos *= invNorm * 0.35;
    sumHingeNeg *= invNorm * 0.35;
    sumCurv = 1.45;
    sumQuartic = 0.85;
    sumShiftNum = 0.0;
  }

  const effectiveShift = sumCurv > 0.25 ? sumShiftNum / (sumCurv + 0.5) : 0.0;

  return {
    sumTilt,
    sumHingePos,
    sumHingeNeg,
    sumCurv,
    sumQuartic,
    effectiveShift,
    sumWave,
    normSq,
    activeCount,
  };
}

function evalAggregatedAxis(agg, coord) {
  if (agg.normSq < 1e-7) return 0.0;
  const u = coord - agg.effectiveShift;
  const u2 = u * u;
  const hingePart =
    0.48 * agg.sumHingePos * softHinge(2.5 * (coord - 0.22)) +
    0.48 * agg.sumHingeNeg * softHinge(-2.5 * (coord + 0.22));
  const quadPart = agg.sumCurv * u2 + agg.sumQuartic * u2 * u2 * 0.35;
  const tiltPart = 0.35 * agg.sumTilt * coord;
  const wavePart = agg.sumWave * (1.0 - Math.cos(2.6 * coord));
  return Math.max(0.0, hingePart + quadPart + tiltPart + wavePart);
}

/**
 * Synthesize the 41x41 loss grid into `losses` in < 0.1 ms.
 */
export function synthesizeLandscapeGrid(
  losses,
  gridSize,
  span,
  mode,
  alphaVec,
  betaVec,
  randomSeed,
  useFilterNorm
) {
  if (mode === "manual") {
    const aggA = aggregateVectorProfile(alphaVec, useFilterNorm);
    const aggB = aggregateVectorProfile(betaVec, useFilterNorm);

    // Check if we are near the initial startup state (only border dimensions active)
    const isPureRampA = aggA.sumCurv < 0.15 && aggA.normSq > 1e-4;
    const isPureRampB = aggB.sumCurv < 0.15 && aggB.normSq > 1e-4;

    for (let ib = 0; ib < gridSize; ib++) {
      const beta = -span + (2.0 * span * ib) / (gridSize - 1);
      const valB = evalAggregatedAxis(aggB, beta);

      for (let ia = 0; ia < gridSize; ia++) {
        const alpha = -span + (2.0 * span * ia) / (gridSize - 1);
        const valA = evalAggregatedAxis(aggA, alpha);

        let cross = 0.0;
        if (isPureRampA && isPureRampB) {
          // Coupled ReLU sheet on startup (Dim #1 & Dim #2): flat dark plateau with one tilted ramp corner
          const sA = Math.sqrt(aggA.normSq) * (alphaVec[1] >= 0 ? 1 : -1);
          const sB = Math.sqrt(aggB.normSq) * (betaVec[2] >= 0 ? 1 : -1);
          cross = 0.72 * softHinge(2.4 * (0.75 * sA * alpha + 0.85 * sB * beta - 0.45));
        } else {
          // Subtle radial coupling when both axes have curvature
          const bothCurv = Math.min(aggA.sumCurv, aggB.sumCurv);
          cross = 0.28 * bothCurv * alpha * alpha * beta * beta;
        }

        losses[ib * gridSize + ia] = BASE_LOSS + Math.max(0.0, valA + valB + cross);
      }
    }
    return;
  }

  // =========================================================================
  // Phase 2 & Phase 3: Gaussian Random Directions (delta, eta ~ N(0, I))
  // =========================================================================
  const h1 = hashSigned(randomSeed * 17 + 3);
  const h2 = hashSigned(randomSeed * 43 + 11);
  const h3 = hashSigned(randomSeed * 89 + 23);
  const h4 = hashSigned(randomSeed * 131 + 37);

  const angle = h1 * Math.PI;
  const cosT = Math.cos(angle);
  const sinT = Math.sin(angle);

  const eccX = 1.0 + 0.18 * h2;
  const eccY = 1.0 - 0.14 * h2;

  if (!useFilterNorm) {
    // PHASE 2: Unnormalized Gaussian Directions on Rescaled Network (c = 10) -> Sharp Funnel!
    const funnelSteepness = 26.5 + 3.0 * h3;

    for (let ib = 0; ib < gridSize; ib++) {
      const beta = -span + (2.0 * span * ib) / (gridSize - 1);
      for (let ia = 0; ia < gridSize; ia++) {
        const alpha = -span + (2.0 * span * ia) / (gridSize - 1);

        const rx = (cosT * alpha - sinT * beta) * eccX;
        const ry = (sinT * alpha + cosT * beta) * eccY;
        const r2 = rx * rx + ry * ry;

        const theta = Math.atan2(ry, rx);
        const angularFacet =
          1.0 + 0.085 * Math.cos(3.0 * theta + h3 * 3.0) + 0.055 * Math.sin(5.0 * theta + h4 * 4.0);

        const coneLoss =
          funnelSteepness * (Math.sqrt(r2 + 0.018) - Math.sqrt(0.018)) * angularFacet +
          14.0 * r2 * angularFacet;

        losses[ib * gridSize + ia] = BASE_LOSS + coneLoss;
      }
    }
  } else {
    // PHASE 3: Filter-Normalized Gaussian Directions (Li et al., 2018) -> Wide, Flat Valley!
    const valleyScale = 2.15 + 0.25 * h3;

    for (let ib = 0; ib < gridSize; ib++) {
      const beta = -span + (2.0 * span * ib) / (gridSize - 1);
      for (let ia = 0; ia < gridSize; ia++) {
        const alpha = -span + (2.0 * span * ia) / (gridSize - 1);

        const rx = (cosT * alpha - sinT * beta) * eccX;
        const ry = (sinT * alpha + cosT * beta) * eccY;
        const r2 = rx * rx + ry * ry;

        const theta = Math.atan2(ry, rx);
        const gentleAniso =
          1.0 + 0.06 * Math.cos(2.0 * theta + h3 * 2.0) + 0.035 * Math.sin(3.0 * theta + h4 * 3.0);

        const basinLoss = valleyScale * (0.62 * r2 + 0.42 * r2 * r2) * gentleAniso;

        losses[ib * gridSize + ia] = BASE_LOSS + basinLoss;
      }
    }
  }
}
